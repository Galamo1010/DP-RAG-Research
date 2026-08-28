"""Tests for the sweep's checkpointing and its baseline strategy.

The checkpoint is what stands between a crash at query 137 and losing fifteen
hours, so it is tested for the failure it exists to survive: a process killed
mid-write.
"""

import json

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
