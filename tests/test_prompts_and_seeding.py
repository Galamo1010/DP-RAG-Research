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


def test_chat_template_date_is_pinned():
    """Llama's template injects today's date unless the caller supplies one.

    Left to itself it renders `Today Date: <today>` into the system message, so
    the same conversation tokenises differently on different days and a seeded
    run is not reproducible across dates -- exactly what ADR 0002 claims it is.
    Measured on Llama-3.2-1B: three token values change and generation diverges
    after roughly 60 characters, under greedy decoding and seeded sampling alike.

    Pinned here rather than at each call site so it cannot be applied to some and
    forgotten at others. Qwen and Phi templates ignore the keyword.
    """
    assert prompts.TEMPLATE_KWARGS["date_string"] == prompts.DATE_STRING
    assert prompts.DATE_STRING, "an empty date still renders 'Today Date: ', which no model saw in training"


def test_router_passes_the_pinned_date_to_the_tokenizer():
    """The pin has to survive the call path that Stage 3 actually uses."""
    import torch
    from dprag.router import Router

    seen = {}

    class _Tok:
        eos_token_id = 0
        def apply_chat_template(self, conversations, **kwargs):
            seen.update(kwargs)
            n = len(conversations)
            return {"input_ids": torch.zeros((n, 3), dtype=torch.long),
                    "attention_mask": torch.ones((n, 3), dtype=torch.long)}

    class _Model:
        tokenizer = _Tok()
        model = None

    router = Router(_Model(), lambda a, b: None, None)
    router._tokenize(prompts.norag_chat("q"))
    assert seen.get("date_string") == prompts.DATE_STRING


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


# --------------------------------------------------------------------------
# retrieval must be a function of (query, seed), not of call order
# --------------------------------------------------------------------------

def _tiny_store(seed=42):
    """A store with hand-made embeddings: no model, no corpus, no GPU."""
    import torch

    from dprag.pup_vector_store import PUPVectorStore, PUPVectorStoreConfig

    store = PUPVectorStore(PUPVectorStoreConfig(
        top_p=0.02, epsilon=0.2, max_retrieve=5, seed=seed))
    for i in range(40):
        store.add(f"document {i}")
    # Deterministic stand-in for the encoder: distinct, normalised vectors.
    vecs = torch.eye(40, 8)[:, :8].clone()
    vecs[:, 0] = torch.linspace(0.1, 1.0, 40)
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    store._embeddings = vecs
    store.encode = lambda text: vecs[len(text) % 40].unsqueeze(0)
    return store


def test_the_same_query_retrieves_the_same_documents_across_runs():
    """The property Stage 3.2's comparison depends on.

    Each configuration runs its own sweep, so without this the generator has
    advanced by the time the second one asks about a given query and it draws a
    different document set. Measured on the real corpus at 0.234 mean Jaccard
    overlap -- zero of 173 queries saw the same documents -- which makes every
    quality difference a mixture of the strategy and the evidence.
    """
    question = "Is this contagious?"

    first = _tiny_store()
    first.reseed_for(question)
    a = first.pup_retrieve(question)

    # A second store that has already served other queries, as the second
    # configuration's sweep would have.
    second = _tiny_store()
    for other in ("something else", "and another", "a third"):
        second.reseed_for(other)
        second.pup_retrieve(other)
    second.reseed_for(question)
    b = second.pup_retrieve(question)

    assert set(a) == set(b), "same query and seed must give the same documents"


def test_without_reseeding_call_order_changes_the_draw():
    """Pins the failure the reseed exists to prevent, so it cannot come back."""
    question = "Is this contagious?"

    fresh = _tiny_store()
    a = fresh.pup_retrieve(question)

    used = _tiny_store()
    for other in ("something else", "and another", "a third"):
        used.pup_retrieve(other)
    b = used.pup_retrieve(question)

    assert set(a) != set(b), (
        "if these now match, sequential draws became order-independent and this "
        "test no longer pins anything -- check before deleting it"
    )


def test_different_queries_still_draw_differently():
    """Reseeding must not collapse retrieval into one fixed answer."""
    store = _tiny_store()
    draws = []
    for q in ("first question", "second question here", "a third one entirely"):
        store.reseed_for(q)
        draws.append(tuple(sorted(store.pup_retrieve(q))))
    assert len(set(draws)) > 1


def test_an_unseeded_store_is_left_alone():
    """seed=None keeps the original global-RNG behaviour, which ADR 0002 preserved."""
    store = _tiny_store(seed=None)
    before = store._py_rng
    store.reseed_for("anything")
    assert store._py_rng is before
