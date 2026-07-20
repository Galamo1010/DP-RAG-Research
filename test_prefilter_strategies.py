"""Unit tests for the Stage 2 pre-filter strategies (synthetic logits, no GPU).

Run:  uv run pytest test_prefilter_strategies.py -v
"""

import pytest
import torch

from prefilter_strategies import (
    PrefilterDecision,
    jaccard_topk,
    make_strategy_b,
    min_overlap_for_tau,
    strategy_a,
    strategy_b,
)


def _logits(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


# --------------------------------------------------------------------------
# strategy A
# --------------------------------------------------------------------------

def test_a_identical_distributions_are_consistent():
    x = _logits(5.0, 3.0, 1.0, 0.5)
    d = strategy_a(x, x.clone())
    assert d.consistent
    assert d.token_id == 0
    assert d.score == 1.0


def test_a_same_argmax_different_tail_is_still_consistent():
    # Both peak at index 1; the tail ordering differs completely.
    rag = _logits(1.0, 9.0, 8.0, 0.0)
    norag = _logits(2.0, 9.0, 0.5, 7.0)
    assert strategy_a(rag, norag).consistent


def test_a_different_argmax_is_inconsistent():
    rag = _logits(9.0, 1.0, 0.0)
    norag = _logits(1.0, 9.0, 0.0)
    d = strategy_a(rag, norag)
    assert not d.consistent
    assert d.score == 0.0


def test_a_emits_norag_argmax_even_when_inconsistent():
    # The router only emits on `consistent`, but token_id must always describe
    # NoRAG's choice so the caller never accidentally emits RAG's token.
    rag = _logits(9.0, 1.0, 0.0)
    norag = _logits(1.0, 9.0, 0.0)
    assert strategy_a(rag, norag).token_id == 1


def test_a_tie_breaks_to_lowest_index():
    tied = _logits(5.0, 5.0, 1.0)
    other = _logits(5.0, 4.9, 1.0)
    assert strategy_a(tied, other).consistent      # both resolve to index 0
    assert strategy_a(tied, other).token_id == 0


# --------------------------------------------------------------------------
# Jaccard / strategy B
# --------------------------------------------------------------------------

def test_jaccard_identical_topk_is_one():
    x = _logits(5.0, 4.0, 3.0, 2.0, 1.0)
    assert jaccard_topk(x, x.clone(), k=3) == pytest.approx(1.0)


def test_jaccard_disjoint_topk_is_zero():
    rag = _logits(9.0, 8.0, 0.0, 0.0)
    norag = _logits(0.0, 0.0, 9.0, 8.0)
    assert jaccard_topk(rag, norag, k=2) == pytest.approx(0.0)


def test_jaccard_matches_closed_form():
    # top-2 sets {0,1} and {1,2}: intersection 1, union 3 -> 1/3.
    rag = _logits(9.0, 8.0, 1.0)
    norag = _logits(1.0, 9.0, 8.0)
    assert jaccard_topk(rag, norag, k=2) == pytest.approx(1 / 3)


def test_jaccard_k_is_clamped_to_vocabulary():
    rag = _logits(3.0, 2.0, 1.0)
    norag = _logits(1.0, 2.0, 3.0)
    # k beyond the vocab compares the whole vocabulary -> identical sets.
    assert jaccard_topk(rag, norag, k=99) == pytest.approx(1.0)


def test_b_threshold_is_inclusive():
    rag = _logits(9.0, 8.0, 1.0)
    norag = _logits(1.0, 9.0, 8.0)          # Jaccard = 1/3
    assert strategy_b(rag, norag, k=2, tau=1 / 3).consistent
    assert not strategy_b(rag, norag, k=2, tau=0.5).consistent


def test_b_reports_similarity_as_score():
    rag = _logits(9.0, 8.0, 1.0)
    norag = _logits(1.0, 9.0, 8.0)
    assert strategy_b(rag, norag, k=2, tau=0.9).score == pytest.approx(1 / 3)


def test_b_rejects_invalid_k():
    x = _logits(1.0, 2.0)
    with pytest.raises(ValueError):
        jaccard_topk(x, x.clone(), k=0)


# --------------------------------------------------------------------------
# The properties documented in the module docstring
# --------------------------------------------------------------------------

def test_b_can_fire_where_a_cannot():
    """B is NOT a subset of A: same top-k set, different ordering.

    This is the "argmax 不同但 top-k 高度重疊" case the proposal wants B to
    capture -- and it shows A's trigger rate is not an upper bound on B's.
    """
    rag = _logits(9.0, 8.0, 0.0)
    norag = _logits(8.0, 9.0, 0.0)          # same top-2 set {0,1}, swapped order
    assert not strategy_a(rag, norag).consistent
    assert strategy_b(rag, norag, k=2, tau=1.0).consistent


@pytest.mark.parametrize("temperature", [0.1, 0.3, 0.5, 0.7, 1.0])
def test_decisions_are_temperature_invariant(temperature):
    """Scaling logits by 1/T changes neither argmax nor the top-k set."""
    rag = _logits(9.0, 8.0, 3.0, 1.0)
    norag = _logits(8.5, 8.0, 0.5, 2.0)

    base_a = strategy_a(rag, norag)
    warm_a = strategy_a(rag / temperature, norag / temperature)
    assert base_a.consistent == warm_a.consistent
    assert base_a.token_id == warm_a.token_id

    base_b = strategy_b(rag, norag, k=3, tau=0.5)
    warm_b = strategy_b(rag / temperature, norag / temperature, k=3, tau=0.5)
    assert base_b.consistent == warm_b.consistent
    assert base_b.score == pytest.approx(warm_b.score)


def test_min_overlap_for_tau_closed_form():
    # k=20, tau=0.9 -> m >= 2*20*0.9/1.9 = 18.947..., i.e. 19 of 20 must match.
    assert min_overlap_for_tau(20, 0.9) == pytest.approx(18.947368, rel=1e-5)
    # tau=1.0 demands a perfect set match.
    assert min_overlap_for_tau(20, 1.0) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# interchangeable interface
# --------------------------------------------------------------------------

def test_make_strategy_b_matches_strategy_a_signature():
    bound = make_strategy_b(k=3, tau=0.5)
    rag = _logits(9.0, 8.0, 3.0, 1.0)
    norag = _logits(8.5, 8.0, 0.5, 2.0)
    for fn in (strategy_a, bound):
        decision = fn(rag, norag)           # identical call shape
        assert isinstance(decision, PrefilterDecision)
    assert bound(rag, norag).score == pytest.approx(
        strategy_b(rag, norag, k=3, tau=0.5).score
    )


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def test_rejects_batched_input():
    batched = torch.zeros(2, 4)
    with pytest.raises(ValueError):
        strategy_a(batched, batched)


def test_rejects_mismatched_vocabulary():
    with pytest.raises(ValueError):
        strategy_a(_logits(1.0, 2.0), _logits(1.0, 2.0, 3.0))
