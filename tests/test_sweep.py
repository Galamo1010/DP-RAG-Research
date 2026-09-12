"""Tests for the sweep's checkpointing, its baseline strategy, and substitution.

The checkpoint is what stands between a crash at query 137 and losing fifteen
hours, so it is tested for the failure it exists to survive: a process killed
mid-write.

The substitution hook is tested for a quieter failure. It feeds the counterfactual
control, and if `docs` and `retrieved_docs` were recorded the wrong way round the
run would complete, the report would print, and the conclusion drawn from it would
be inverted with nothing downstream able to notice.
"""

import json
from types import SimpleNamespace

import torch

from dprag import sweep


class _Logits:
    """Minimal stand-in for the two logit vectors a strategy receives."""

    def __init__(self, top: int, size: int = 8):
        self.vec = torch.zeros(size)
        self.vec[top] = 1.0

    def argmax(self):
        return torch.tensor(self.vec.argmax())


def test_never_agree_always_falls_through_to_the_paid_path():
    """The baseline IS plain DPRAG, produced by routing rather than a second code
    path (ADR 0003). If this ever agreed, the baseline would quietly become a
    treatment."""
    for rag_top, norag_top in ((0, 0), (0, 3), (5, 5)):
        decision = sweep.NEVER_AGREE(_Logits(rag_top).vec, _Logits(norag_top).vec)
        assert decision.consistent is False


def test_never_agree_still_reports_the_norag_token():
    """Even on the paid path the decision carries NoRAG's argmax, because the
    trace records it per position and Stage 4.3 attacks that sequence."""
    decision = sweep.NEVER_AGREE(_Logits(1).vec, _Logits(6).vec)
    assert decision.token_id == 6


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------

def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep.paths, "results_dir", lambda: tmp_path)
    return tmp_path


def test_checkpoint_round_trips(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    rows = [{"query": "a", "by_strategy": {}}, {"query": "b", "by_strategy": {}}]
    sweep._save_checkpoint("run", rows)
    assert sweep._load_checkpoint("run") == rows


def test_missing_checkpoint_is_not_an_error():
    assert sweep._load_checkpoint("never-written") == []


def test_truncated_checkpoint_is_discarded_rather_than_trusted(tmp_path, monkeypatch):
    """A checkpoint written while the process was killed can be half a file.

    Recomputing costs time; parsing it half-way costs a results file that looks
    complete and is not.
    """
    _isolate(tmp_path, monkeypatch)
    path = sweep._checkpoint_path("run")
    path.write_text('[{"query": "a"}, {"que', encoding="utf-8")
    assert sweep._load_checkpoint("run") == []


def test_saving_leaves_no_partial_file_behind(tmp_path, monkeypatch):
    """The write goes through a temporary file and an atomic replace, so a kill
    cannot leave the real checkpoint truncated."""
    _isolate(tmp_path, monkeypatch)
    sweep._save_checkpoint("run", [{"query": "a"}])
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_checkpoint_is_hidden_and_named_after_its_run(tmp_path, monkeypatch):
    """It sits beside the results, so it must not be mistaken for one --
    run_record.load_all globs *.json and a stray file would be read as a result."""
    _isolate(tmp_path, monkeypatch)
    sweep._save_checkpoint("stage3_2_main_A_eps10", [])
    name = sweep._checkpoint_path("stage3_2_main_A_eps10").name
    assert name.startswith(".")
    assert "stage3_2_main_A_eps10" in name


def test_a_resumed_run_skips_what_the_checkpoint_already_holds(tmp_path, monkeypatch):
    """The property the whole mechanism exists for."""
    _isolate(tmp_path, monkeypatch)
    done = [{"query": "q1", "by_strategy": {}}, {"query": "q2", "by_strategy": {}}]
    sweep._save_checkpoint("run", done)

    loaded = sweep._load_checkpoint("run")
    seen = {r["query"] for r in loaded}
    remaining = [q for q in ("q1", "q2", "q3") if q not in seen]
    assert remaining == ["q3"]


def test_checkpoint_survives_non_ascii(tmp_path, monkeypatch):
    """Queries are patient questions; results elsewhere in this project are
    written with ensure_ascii=False and read back as UTF-8."""
    _isolate(tmp_path, monkeypatch)
    rows = [{"query": "醫師您好，我最近咳嗽", "by_strategy": {}}]
    sweep._save_checkpoint("run", rows)
    assert sweep._load_checkpoint("run") == rows
    raw = json.loads(sweep._checkpoint_path("run").read_text(encoding="utf-8"))
    assert raw[0]["query"].startswith("醫師")


# --------------------------------------------------------------------------
# the document-substitution hook
# --------------------------------------------------------------------------

class _Store:
    """Retrieval, reduced to the two calls the sweep makes of it."""

    def __init__(self, documents):
        self.documents = documents
        self.reseeded = []

    def reseed_for(self, query):
        self.reseeded.append(query)

    def pup_retrieve(self, query):
        return list(self.documents)


class _Result:
    emitted = [1, 2]
    n_documents = 0


def _bench(documents):
    store = _Store(documents)
    return SimpleNamespace(
        engine=SimpleNamespace(pup_vector_store=store),
        dp_model=SimpleNamespace(tokenizer=object()),
        queries=lambda: ["q1"],
    ), store


def _stub_generation(monkeypatch, seen_documents):
    """Replace everything downstream of document selection.

    The sweep's job here is choosing what generation sees and recording both
    lists; the generation itself has its own tests and needs a GPU.
    """
    class _Router:
        def __init__(self, model, strategy, config):
            pass

        def generate(self, documents, question):
            seen_documents.append(list(documents))
            return _Result()

    monkeypatch.setattr(sweep, "Router", _Router)
    monkeypatch.setattr(sweep, "trace_marks", lambda *a, **k: ([], None, None))
    # trigger_rate because the sweep prints it per query; the rest of a real
    # trace is irrelevant to which documents got chosen.
    # Only the two fields the sweep's own progress line reads. The rest of a real
    # trace is irrelevant to which documents got chosen.
    monkeypatch.setattr(sweep.trace, "strategy_trace",
                        lambda *a, **k: {"trigger_rate": 0.0, "seconds": 0.0})
    monkeypatch.setattr(sweep.trace, "check", lambda record: None)
    # Echo the documents back so a row's recorded list is readable in the test.
    monkeypatch.setattr(sweep.trace, "retrieval_trace",
                        lambda store, query, documents: list(documents))


def _run(monkeypatch, tmp_path, documents, substitute):
    _isolate(tmp_path, monkeypatch)
    bench, store = _bench(documents)
    seen = []
    _stub_generation(monkeypatch, seen)
    written = {}
    def _write(name, config, metrics, per_item=None, **kwargs):
        written["rows"] = per_item
        return tmp_path / "out.json"

    monkeypatch.setattr(sweep.run_record, "write", _write)
    exp = SimpleNamespace(seed=42, vocab_min_count=1)
    sweep.routed_sweep(bench, exp, {"A": sweep.NEVER_AGREE}, object(),
                       name="t", filename="t", substitute_documents=substitute)
    return written["rows"], seen, store


def test_without_the_hook_generation_sees_what_retrieval_found(tmp_path, monkeypatch):
    """The default path, unchanged: no hook, no extra field, no substitution."""
    rows, seen, _ = _run(monkeypatch, tmp_path, ["d1", "d2"], None)
    assert seen == [["d1", "d2"]]
    assert rows[0]["docs"] == ["d1", "d2"]
    assert "retrieved_docs" not in rows[0]


def test_the_hook_decides_what_generation_actually_sees(tmp_path, monkeypatch):
    rows, seen, _ = _run(monkeypatch, tmp_path, ["d1", "d2"],
                         lambda retrieved, question: ["x1", "x2"])
    assert seen == [["x1", "x2"]]


def test_docs_records_the_substitutes_and_retrieved_docs_the_real_ones(
        tmp_path, monkeypatch):
    """The inversion that would flip the counterfactual's conclusion in silence.

    `docs` must describe the prompt that was actually built; `retrieved_docs` must
    describe what DP retrieval found. Swapped, the relevance check in the report
    compares each list against itself and passes.
    """
    rows, _, _ = _run(monkeypatch, tmp_path, ["real1", "real2"],
                      lambda retrieved, question: ["fake1", "fake2"])
    assert rows[0]["docs"] == ["fake1", "fake2"]
    assert rows[0]["retrieved_docs"] == ["real1", "real2"]


def test_the_hook_is_handed_what_retrieval_found(tmp_path, monkeypatch):
    """It has to exclude the retrieved documents, so it has to receive them."""
    got = []
    _run(monkeypatch, tmp_path, ["d1", "d2"],
         lambda retrieved, question: got.append((list(retrieved), question)) or ["x"])
    assert got == [(["d1", "d2"], "q1")]


def test_a_zero_document_query_is_still_skipped_under_substitution(
        tmp_path, monkeypatch):
    """Whether a query has documents is a property of DP retrieval and stays one:
    substituting into an empty retrieval would manufacture evidence for a query
    the mechanism declined to answer (CONTEXT.md)."""
    rows, seen, _ = _run(monkeypatch, tmp_path, [],
                         lambda retrieved, question: ["x1", "x2"])
    assert rows == []
    assert seen == []
