"""Tests for the per-position record (no GPU, no model).

The point of these is the invariant checker: it is the only thing standing
between a silently corrupted router and a results file full of plausible numbers.
"""

import pytest

from dprag import trace
from dprag.medical_flags import PATTERN_KIND, WORD_KIND, TokenMark


class _FakeResult:
    """The fields trace.strategy_trace reads off a RoutedResult."""

    def __init__(self, emitted, norag, paid, rag=None):
        self.emitted = emitted
        self.norag_argmax = norag
        # Defaults to agreeing with NoRAG everywhere, which is the uninteresting
        # case; tests that care about the ablation pass their own.
        self.rag_argmax = list(norag) if rag is None else rag
        self.paid_positions = paid
        self.text = "some answer"
        self.epsilon_usage = 1.5
        self.epsilon_budget = 10.0

    @property
    def trigger_rate(self):
        if not self.emitted:
            return 0.0
        return 1.0 - len(self.paid_positions) / len(self.emitted)

    @property
    def epsilon_savings(self):
        return self.epsilon_budget - self.epsilon_usage


def _record(emitted=(1, 2, 3, 4), norag=(1, 9, 3, 4), paid=(1,), marks=None,
            rag=None):
    res = _FakeResult(list(emitted), list(norag), list(paid),
                      None if rag is None else list(rag))
    marks = marks if marks is not None else [None] * len(emitted)
    return trace.strategy_trace(res, marks, seconds=12.345)


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------

def test_all_three_sequences_have_one_entry_per_position():
    r = _record()
    assert (len(r["emitted"]) == len(r["norag_argmax"]) == len(r["rag_argmax"])
            == len(r["clinical"]) == 4)


def test_marks_serialise_as_pairs():
    marks = [None, TokenMark(WORD_KIND, True), TokenMark(WORD_KIND, False), None]
    assert _record(marks=marks)["clinical"] == [None, ["word", True], ["word", False], None]


def test_seconds_are_recorded_per_strategy():
    """Without this the efficiency question the proposal asks cannot be answered:
    the router adds two rows every step and only pays off at a high trigger rate,
    so 'is the overall effect positive' is per-strategy or it is nothing."""
    assert _record()["seconds"] == 12.35


# --------------------------------------------------------------------------
# invariants -- what these exist to catch
# --------------------------------------------------------------------------

def test_a_correct_record_passes():
    trace.check(_record())


def test_a_free_position_emitting_something_other_than_norag_is_caught():
    """The load-bearing one. A skipped position emits the NoRAG token by
    definition, so a mismatch means the router's free path is broken -- and that
    breakage produces a worse answer rather than an exception, which DP noise
    does too."""
    with pytest.raises(ValueError, match="must emit the NoRAG token"):
        trace.check(_record(emitted=(1, 2, 3, 4), norag=(1, 9, 7, 4), paid=(1,)))


def test_truncated_sequences_are_caught():
    r = _record()
    r["clinical"] = r["clinical"][:2]
    with pytest.raises(ValueError, match="truncated"):
        trace.check(r)


def test_trigger_rate_disagreeing_with_the_paid_count_is_caught():
    r = _record()
    r["trigger_rate"] = 0.5          # actual is 3/4
    with pytest.raises(ValueError, match="accounting is off"):
        trace.check(r)


# --------------------------------------------------------------------------
# readers Stage 4 needs
# --------------------------------------------------------------------------

def test_free_positions_excludes_the_paid_ones():
    assert trace.free_positions(_record(paid=(1,))) == [0, 2, 3]


def test_norag_subsequence_is_what_stage_4_3_attacks():
    """The proposal extracts the NoRAG argmax at agreeing positions and runs a
    membership attack on that subsequence. It is derivable rather than stored,
    because at a free position the emitted token IS the NoRAG argmax."""
    r = _record(emitted=(11, 22, 33, 44), norag=(11, 99, 33, 44), paid=(1,))
    assert trace.norag_subsequence(r) == [11, 33, 44]


def test_empty_generation_does_not_raise():
    res = _FakeResult([], [], [])
    r = trace.strategy_trace(res, [], seconds=0.0)
    trace.check(r)
    assert trace.norag_subsequence(r) == []


# --------------------------------------------------------------------------
# the ablation the proposal asks for: where did Strategy B's budget go
# --------------------------------------------------------------------------

def test_paying_where_the_documents_changed_nothing_is_counted_as_wasted():
    # Position 1 was paid, but RAG and NoRAG both wanted token 9: the epsilon
    # bought a token the free path would have emitted anyway.
    r = _record(emitted=(1, 9, 3, 4), norag=(1, 9, 3, 4), rag=(1, 9, 3, 4),
                paid=(1,))
    assert trace.wasted_paid_positions(r) == [1]
    assert trace.missed_free_positions(r) == []


def test_paying_where_the_documents_changed_the_first_choice_is_not_wasted():
    r = _record(emitted=(1, 5, 3, 4), norag=(1, 9, 3, 4), rag=(1, 5, 3, 4),
                paid=(1,))
    assert trace.wasted_paid_positions(r) == []


def test_skipping_a_position_the_documents_would_have_changed_is_counted():
    # Position 2 is free, yet the RAG instance wanted 7 rather than 3. The
    # saving dropped the documents' influence on that token.
    r = _record(emitted=(1, 2, 3, 4), norag=(1, 2, 3, 4), rag=(1, 2, 7, 4),
                paid=())
    assert trace.missed_free_positions(r) == [2]


def test_strategy_a_can_have_no_missed_free_positions():
    # A skips exactly when rag_argmax == norag_argmax, so a run of A that
    # reports a missed free position means the recorded argmax and the strategy
    # that ran disagree. This pins the shape that check exists to detect.
    rag = [1, 5, 3, 4]
    norag = [1, 9, 3, 4]
    paid = [i for i in range(4) if rag[i] != norag[i]]
    r = _record(emitted=(1, 5, 3, 4), norag=norag, rag=rag, paid=tuple(paid))
    assert trace.missed_free_positions(r) == []


def test_records_without_rag_argmax_return_empty_rather_than_raising():
    # Everything written before the field existed still has to be readable.
    r = _record()
    del r["rag_argmax"]
    assert trace.wasted_paid_positions(r) == []
    assert trace.missed_free_positions(r) == []


def test_a_truncated_rag_argmax_is_caught():
    r = _record()
    r["rag_argmax"] = r["rag_argmax"][:-1]
    with pytest.raises(ValueError, match="rag_argmax"):
        trace.check(r)
