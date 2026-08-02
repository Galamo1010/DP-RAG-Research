"""Tests for the Stage 3.1 router, driven by a fake model on CPU.

The router is the most intricate code in the project: two KV caches advancing at
different rates, where a synchronisation error does not raise but silently
conditions the aggregation on the wrong prefix. That class of bug is invisible on
a GPU run -- the answer just comes out slightly wrong -- so correctness is
established here, against scripted logits, before any rented hardware is involved.

Two equivalence properties carry most of the weight:

  * a strategy that never agrees must reproduce plain DPRAG token for token
  * a strategy that always agrees must reproduce the pure NoRAG greedy sequence

Any error in the backlog catch-up shows up as a divergence from one or the other.
"""

import torch

from dprag.router import Router, _composed_epsilon
from dprag.strategies import PrefilterDecision

VOCAB = 12
EOS = 11


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class _FakeOutput:
    def __init__(self, logits, cache):
        self.logits = logits
        self.past_key_values = cache


class _FakeTokenizer:
    eos_token_id = EOS
    pad_token_id = 0

    def apply_chat_template(self, conversations, **kwargs):
        batch = len(conversations)
        return {
            "input_ids": torch.zeros((batch, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch, 3), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class _FakeAggregator:
    """Stands in for DPLogitsAggregator: records what it saw, returns row 1."""

    def __init__(self):
        self.seen: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        self.seen.append(scores.clone())
        # Deterministic: put all mass on the argmax of the first document stream,
        # so sampling is effectively argmax and tests stay reproducible.
        row = scores[1] if scores.shape[0] > 1 else scores[0]
        out = torch.full((1, scores.shape[1]), -1e9)
        out[0, int(row.argmax())] = 1e9
        return out


class _FakeDPModel:
    def __init__(self, scripts_prefilter, scripts_dprag):
        self._prefilter = scripts_prefilter
        self._dprag = scripts_dprag
        self.tokenizer = _FakeTokenizer()
        self.aggregator = _FakeAggregator()
        self.model = _FakeRouterModel(self)

    def dp_logits_aggregator(self, config):
        return self.aggregator


class _FakeRouterModel:
    """Dispatches to a prefilter script or a DPRAG script based on batch width."""

    device = torch.device("cpu")

    def __init__(self, owner):
        self.owner = owner
        self.calls: list[tuple[int, int]] = []

    def __call__(self, input_ids, attention_mask, past_key_values, use_cache):
        batch, n_new = input_ids.shape
        scripts = self.owner._prefilter if batch == 2 else self.owner._dprag
        seen = 0 if past_key_values is None else past_key_values
        advanced = 1 if past_key_values is None else n_new
        next_seen = seen + advanced
        self.calls.append((batch, n_new))
        rows = [
            scripts[row][min(next_seen - 1, len(scripts[row]) - 1)]
            for row in range(batch)
        ]
        logits = torch.tensor(rows, dtype=torch.float32).unsqueeze(1)
        return _FakeOutput(logits, next_seen)


class _FakeConfig:
    def __init__(self, max_new_tokens=6, temperature=1.0, delta=1e-3, token_eps=0.2):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.delta = delta
        self._token_eps = token_eps

    def token_epsilon(self):
        return self._token_eps


def _onehot(index: int, high: float = 10.0) -> list[float]:
    v = [0.0] * VOCAB
    v[index] = high
    return v


def _build(norag_row, doc_row, strategy, n_docs=2, **cfg_kwargs):
    """Build a router over scripted logits.

    norag_row / doc_row are per-step argmax indices. The pre-filter batch is
    always 2 rows (NoRAG, RAG); the DPRAG batch is 1 + n_docs rows. n_docs
    defaults to 2 so the two batches differ in width and the fake can tell them
    apart -- with a single document both would be 2 rows wide.
    """
    scripts_pf = {0: [_onehot(i) for i in norag_row], 1: [_onehot(i) for i in doc_row]}
    scripts_dp = {0: [_onehot(i) for i in norag_row]}
    for row in range(1, n_docs + 1):
        scripts_dp[row] = [_onehot(i) for i in doc_row]
    model = _FakeDPModel(scripts_pf, scripts_dp)
    router = Router(model, strategy, _FakeConfig(**cfg_kwargs))
    return router, model


DOCS = ["doc one", "doc two"]


ALWAYS_AGREE = lambda rag, norag: PrefilterDecision(True, int(norag.argmax()), 1.0)
NEVER_AGREE = lambda rag, norag: PrefilterDecision(False, int(norag.argmax()), 0.0)


# --------------------------------------------------------------------------
# the two equivalence properties
# --------------------------------------------------------------------------

def test_never_agreeing_pays_every_position():
    """The baseline path: every position goes through the DP aggregation."""
    norag = [1, 1, 1, 1]
    doc = [5, 6, 7, 8]           # the aggregator returns this row's argmax
    router, _ = _build(norag, doc, NEVER_AGREE, max_new_tokens=4)
    result = router.generate(DOCS, "q")

    assert result.emitted == doc
    assert result.paid_positions == [0, 1, 2, 3]
    assert result.trigger_rate == 0.0
    assert result.epsilon_usage == result.epsilon_budget


def test_always_agreeing_emits_norag_and_costs_nothing():
    norag = [2, 3, 4, 5]
    doc = [9, 9, 9, 9]
    router, model = _build(norag, doc, ALWAYS_AGREE, max_new_tokens=4)
    result = router.generate(DOCS, "q")

    assert result.emitted == norag
    assert result.paid_positions == []
    assert result.trigger_rate == 1.0
    assert result.epsilon_usage == 0.0
    assert result.epsilon_savings == result.epsilon_budget
    # The whole point: the k+1 batch was never run.
    assert all(batch == 2 for batch, _ in model.model.calls)


def test_epsilon_usage_tracks_the_paid_count():
    """A mixed pattern: usage must equal token_epsilon composed over paid steps."""
    calls = {"n": 0}

    def alternating(rag, norag):
        consistent = calls["n"] % 2 == 0
        calls["n"] += 1
        return PrefilterDecision(consistent, int(norag.argmax()), 1.0 if consistent else 0.0)

    norag = [1, 1, 1, 1, 1, 1]
    doc = [5, 5, 5, 5, 5, 5]
    router, _ = _build(norag, doc, alternating, max_new_tokens=6)
    result = router.generate(DOCS, "q")

    assert result.paid_positions == [1, 3, 5]
    expected = _composed_epsilon(result.token_epsilon, 3, 1e-3)
    assert result.epsilon_usage == expected
    assert result.epsilon_usage < result.epsilon_budget


# --------------------------------------------------------------------------
# the backlog catch-up -- the part that fails silently if wrong
# --------------------------------------------------------------------------

def test_dprag_batch_is_fed_the_whole_backlog_at_once():
    """After a run of agreements, the k+1 batch catches up in ONE forward.

    Replaying the gap token by token would also be correct but would give up the
    saving the router exists to produce, so the batching is asserted directly.
    """
    calls = {"n": 0}

    def agree_then_disagree(rag, norag):
        # agree on 0,1,2 then disagree on 3
        consistent = calls["n"] < 3
        calls["n"] += 1
        return PrefilterDecision(consistent, int(norag.argmax()), 1.0 if consistent else 0.0)

    norag = [1, 2, 3, 4, 5]
    doc = [7, 7, 7, 7, 7]
    router, model = _build(norag, doc, agree_then_disagree, max_new_tokens=4)
    router.generate(DOCS, "q")

    wide = [(b, n) for b, n in model.model.calls if b != 2]
    # prompt call, then one catch-up carrying the three skipped tokens
    assert len(wide) == 2
    assert wide[0][1] == 3        # the prompt
    assert wide[1][1] == 3        # three backlog tokens in a single forward


def test_no_dprag_forward_happens_while_the_strategy_agrees():
    norag = [1, 2, 3]
    router, model = _build(norag, [9, 9, 9], ALWAYS_AGREE, max_new_tokens=3)
    router.generate(DOCS, "q")
    assert not [c for c in model.model.calls if c[0] != 2]


def test_aggregation_sees_all_streams():
    norag = [1, 1]
    doc = [4, 4]
    router, model = _build(norag, doc, NEVER_AGREE, max_new_tokens=2)
    router.generate(DOCS, "q")
    assert model.aggregator.seen, "the aggregator was never reached"
    for seen in model.aggregator.seen:
        # k+1: the public prior plus one row per document.
        assert seen.shape[0] == 1 + len(DOCS)


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

def test_zero_document_query_still_generates():
    """DP retrieval returning nothing must not need special-casing at the call site."""
    norag = [1, 2, 3]
    router, _ = _build(norag, norag, ALWAYS_AGREE, n_docs=0, max_new_tokens=3)
    result = router.generate([], "q")
    assert result.n_documents == 0
    assert result.emitted == norag


def test_eos_stops_generation_and_is_not_counted():
    norag = [1, EOS, 3, 4]
    router, _ = _build(norag, norag, ALWAYS_AGREE, max_new_tokens=4)
    result = router.generate(DOCS, "q")
    assert result.emitted == [1]
    assert len(result.decisions) == 1
    assert EOS not in result.emitted


def test_agreement_on_the_very_first_position():
    norag = [3, 3, 3]
    router, model = _build(norag, [8, 8, 8], ALWAYS_AGREE, max_new_tokens=1)
    result = router.generate(DOCS, "q")
    assert result.emitted == [3]
    assert result.paid_positions == []


def test_records_are_all_the_same_length():
    calls = {"n": 0}

    def alternating(rag, norag):
        consistent = calls["n"] % 2 == 0
        calls["n"] += 1
        return PrefilterDecision(consistent, int(norag.argmax()), 0.0)

    norag = [1, 2, 3, 4]
    router, _ = _build(norag, [6, 6, 6, 6], alternating, max_new_tokens=4)
    result = router.generate(DOCS, "q")
    assert len(result.decisions) == len(result.emitted) == len(result.norag_argmax)


def test_result_carries_the_accounting_caveat():
    """The assumption must travel with the number, not live only in a document."""
    norag = [1]
    router, _ = _build(norag, norag, ALWAYS_AGREE, max_new_tokens=1)
    result = router.generate(DOCS, "q")
    assert "Stage 4.3" in result.epsilon_accounting
    assert "assumes" in result.epsilon_accounting


# --------------------------------------------------------------------------
# epsilon composition helper
# --------------------------------------------------------------------------

def test_composing_zero_steps_costs_nothing():
    assert _composed_epsilon(0.2, 0, 1e-3) == 0.0


def test_composition_grows_with_steps():
    a = _composed_epsilon(0.2, 4, 1e-3)
    b = _composed_epsilon(0.2, 16, 1e-3)
    assert 0 < a < b
