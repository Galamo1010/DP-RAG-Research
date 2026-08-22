"""Tests for prompt construction and retrieval seeding (no GPU, no model)."""

import numpy as np
import pytest

from dprag import prompts
from dprag.config import MODELS
from dprag.dual_instance import NORAG_ROW, RAG_ROW
from dprag.pup_vector_store import PUPVectorStore, PUPVectorStoreConfig


# --------------------------------------------------------------------------
# prompts: the point is that both callers get the SAME NoRAG instance
# --------------------------------------------------------------------------

def test_dprag_and_dual_instance_share_one_norag_definition():
    """Stage 2 compares NoRAG against RAG; both batches must mean the same NoRAG.

    This used to be two copies kept in step by hand -- the exact thing that can
    drift without failing loudly.
    """
    question = "Why do I have a headache?"
    docs = ["doc one", "doc two"]
    dprag_row0 = prompts.dprag_chat_batch(docs, question)[0]
    dual_row0 = prompts.dual_instance_batch(docs, question)[NORAG_ROW]
    assert dprag_row0 == dual_row0


def test_dprag_batch_is_k_plus_one_rows():
    docs = ["a", "b", "c"]
    batch = prompts.dprag_chat_batch(docs, "q")
    assert len(batch) == len(docs) + 1


def test_dprag_row_zero_holds_no_document():
    docs = ["secret patient record"]
    row0 = prompts.dprag_chat_batch(docs, "q")[0]
    assert "secret patient record" not in str(row0)


def test_each_dprag_stream_holds_exactly_one_document():
    docs = ["alpha", "beta"]
    batch = prompts.dprag_chat_batch(docs, "q")
    assert "alpha" in str(batch[1]) and "beta" not in str(batch[1])
    assert "beta" in str(batch[2]) and "alpha" not in str(batch[2])


def test_rag_instance_holds_every_document():
    docs = ["alpha", "beta", "gamma"]
    rag_row = prompts.dual_instance_batch(docs, "q")[RAG_ROW]
    for doc in docs:
        assert doc in str(rag_row)


def test_rag_instance_collapses_to_norag_when_retrieval_returns_nothing():
    """DP retrieval can legitimately return zero documents."""
    assert prompts.all_documents_chat([], "q") == prompts.norag_chat("q")


def test_every_conversation_carries_the_question():
    question = "Is this contagious?"
    for row in prompts.dprag_chat_batch(["d"], question) + prompts.dual_instance_batch(["d"], question):
        assert any(m["role"] == "user" and question in m["content"] for m in row)


def test_every_row_opens_with_a_system_message():
    """The property MODELS is selected against: row 0 is always a system message.

    This is the half that can be checked offline. The other half -- whether a
    listed model's chat template actually accepts that role -- cannot be, because
    a chat template is a mutable file in a model repository that is only known
    once downloaded. Mistral's changed in July 2024; Gemma gained system-role
    support between versions 2 and 4.

    A name blacklist used to stand in for that check ("gemma" must not appear in
    MODELS). It was removed with ADR 0004: it blocked google/gemma-4-12B-it, whose
    template gives `system` its own turn, while it would have waved through any
    other model that rejects the role. Guarding a behaviour by spelling is worse
    than not guarding it, because it reads as protection.

    Verification belongs where the real tokenizer is: see the preflight check
    described in ADR 0004.
    """
    rows = prompts.dprag_chat_batch(["d"], "q") + prompts.dual_instance_batch(["d"], "q")
    for row in rows:
        assert row[0]["role"] == "system"


def test_model_set_is_three_distinct_models():
    """Stage 3.2's cross-model phase needs all three slots filled (ADR 0004)."""
    assert len(MODELS) == 3
    assert len(set(MODELS)) == 3


def test_summary_batch_mirrors_the_chat_batch_shape():
    docs = ["a", "b"]
    assert len(prompts.summary_batch(docs, "topic")) == len(docs) + 1


# --------------------------------------------------------------------------
# seeding: retrieval must be reproducible
# --------------------------------------------------------------------------

class _FakeStore(PUPVectorStore):
    """A store with the embedding model stubbed out, so seeding is testable on CPU.

    Only `encode` needs replacing: the sampling under test is pure numpy/random
    once scores exist.
    """

    def __init__(self, config, scores):
        super().__init__(config)
        self._scores = scores
        self.store = [f"doc{i}" for i in range(len(scores))]

    def pup_retrieve(self, query: str) -> list[str]:
        scores = self._scores
        if self.differential_pivacy:
            threshold = self._exp_mechanism_top_p_threshold(scores)
        else:
            threshold = self._non_dp_top_p_threshold(scores)
        pairs = sorted(zip(self.store, scores), key=lambda x: x[1], reverse=True)
        retrieved = [d for d, s in pairs if s > threshold]
        return self._py_rng.sample(retrieved, min(len(retrieved), self.max_retrieve))


def _scores(n=60, seed=0):
    return np.random.default_rng(seed).uniform(-0.4, 0.75, size=n)


def _config(seed):
    return PUPVectorStoreConfig(top_p=0.02, epsilon=0.2, max_retrieve=10, seed=seed)


def test_same_seed_retrieves_the_same_documents():
    """The pain this fixes: a re-run drew different documents, so a single
    configuration could not be re-run without redoing the whole sweep."""
    scores = _scores()
    first = _FakeStore(_config(42), scores).pup_retrieve("q")
    second = _FakeStore(_config(42), scores).pup_retrieve("q")
    assert first == second


def test_different_seeds_retrieve_differently():
    """Seeding must not have collapsed the mechanism into something constant."""
    scores = _scores()
    runs = {tuple(_FakeStore(_config(s), scores).pup_retrieve("q")) for s in range(12)}
    assert len(runs) > 1


def test_repeated_retrieval_within_a_seeded_store_still_varies():
    """Each draw advances the generator: the mechanism stays random within a run,
    it is only the run as a whole that is reproducible."""
    scores = _scores()
    store = _FakeStore(_config(42), scores)
    draws = [tuple(store.pup_retrieve("q")) for _ in range(12)]
    assert len(set(draws)) > 1


def test_a_seeded_store_does_not_disturb_the_global_rng():
    """A seeded store must not make unrelated code deterministic as a side effect."""
    scores = _scores()
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    _FakeStore(_config(42), scores).pup_retrieve("q")
    assert np.random.random() == before


def test_unseeded_keeps_the_original_global_rng_behaviour():
    scores = _scores()
    store = _FakeStore(PUPVectorStoreConfig(top_p=0.02, epsilon=0.2, max_retrieve=10), scores)
    assert store.seed is None
    assert store._np_rng is np.random
    store.pup_retrieve("q")   # must not raise


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_reproducibility_holds_across_seeds(seed):
    scores = _scores()
    assert (
        _FakeStore(_config(seed), scores).pup_retrieve("q")
        == _FakeStore(_config(seed), scores).pup_retrieve("q")
    )
