"""Unit tests for ExperimentConfig, RunRecord and paths (no GPU, no dataset)."""

import json

import pytest

from dprag import paths
from dprag.config import ExperimentConfig, MODELS
from dprag.run_record import SCHEMA_VERSION, load, load_all, write


# --------------------------------------------------------------------------
# ExperimentConfig
# --------------------------------------------------------------------------

def test_to_dict_records_every_field():
    """The whole point: no parameter can be silently left out of a record."""
    d = ExperimentConfig().to_dict()
    for key in (
        "delta", "eps_retrieval", "gen_epsilon",          # these four were the
        "alpha", "omega", "corpus_seed",                  # ones the old result
        "temperature", "max_new_tokens", "max_retrieve",  # files never recorded
        "n_docs", "n_queries", "query_seed", "seed",
        "gen_model", "embed_model", "retrieval_top_p", "embed_batch_size",
    ):
        assert key in d, f"{key} missing from the parameter snapshot"


def test_with_overrides_leaves_original_untouched():
    base = ExperimentConfig()
    swept = base.with_(temperature=0.3)
    assert swept.temperature == 0.3
    assert base.temperature == 1.0
    # everything else carries over
    assert swept.max_retrieve == base.max_retrieve


def test_config_is_frozen():
    cfg = ExperimentConfig()
    with pytest.raises(Exception):
        cfg.temperature = 0.5   # type: ignore[misc]


def test_defaults_match_what_experiments_actually_use():
    """The old module claimed MAX_RETRIEVE=128 while every experiment used 10.

    The value moved to 40 in ADR 0008 -- at 10 the DP threshold's ~83 qualifying
    documents were cut to a random ten, delivering 1.5 of the true top-10. What
    this test guards is unchanged: the number here is the number that runs.
    """
    cfg = ExperimentConfig()
    assert cfg.max_retrieve == 40
    assert cfg.temperature == 1.0
    assert cfg.delta == 1e-3
    assert cfg.eps_retrieval == 0.2


def test_max_retrieve_leaves_headroom_below_the_measured_oom_ceiling():
    """69 documents peaked at 79.4 GB of an 80 GB A100 and 73 failed outright.

    Peak memory grows about 0.71 GB per document and moves with prompt length, so
    a setting that survived three probe queries can still die at query 137 of a
    two-hundred-query run. 40 sits near 70% of capacity; anything above ~55 does
    not, and the ceiling itself is not a setting.
    """
    assert ExperimentConfig().max_retrieve <= 55


def test_model_ids_are_locally_loadable_casing():
    """The old MODELS list held lowercase OpenRouter ids that HF cannot resolve."""
    for model in MODELS:
        org, _, name = model.partition("/")
        assert org and name, f"{model} is not an org/name id"
        assert model != model.lower(), (
            f"{model} looks like a lowercase API id, not a Hugging Face repo id"
        )


# --------------------------------------------------------------------------
# RunRecord
# --------------------------------------------------------------------------

def test_write_then_load_round_trip(tmp_path):
    cfg = ExperimentConfig(n_queries=5, temperature=0.3)
    out = write(
        "unit_test_run", cfg,
        metrics={"mean_consistency": 0.87},
        per_item=[{"query": "q1", "consistency": 0.9}],
        results_dir=tmp_path,
    )
    rec = load(out)
    assert rec.name == "unit_test_run"
    assert rec.schema == SCHEMA_VERSION
    assert rec.metric("mean_consistency") == 0.87
    assert rec.param("n_queries") == 5
    assert rec.param("temperature") == 0.3
    # parameters the experiment never mentioned are still recorded
    assert rec.param("alpha") == 1.0
    assert rec.param("eps_retrieval") == 0.2
    assert len(rec.per_item) == 1


def test_record_carries_provenance(tmp_path):
    out = write("prov", ExperimentConfig(), metrics={}, results_dir=tmp_path)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert "created_at" in raw
    assert "git_commit" in raw    # None is acceptable (git may be unavailable)
    assert "git_dirty" in raw


def test_write_accepts_a_plain_dict(tmp_path):
    out = write("plain", {"foo": 1}, metrics={"bar": 2}, results_dir=tmp_path)
    assert load(out).param("foo") == 1


def test_filename_can_differ_from_run_name(tmp_path):
    out = write(
        "stage2_temperature_sweep", ExperimentConfig(), metrics={},
        filename="stage2_temperature_sweep_50q", results_dir=tmp_path,
    )
    assert out.name == "stage2_temperature_sweep_50q.json"
    assert load(out).name == "stage2_temperature_sweep"


def test_legacy_files_are_rejected_not_guessed(tmp_path):
    """Pre-restructure results lack a schema; loading must fail loudly."""
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps({"config": {}, "results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="predates"):
        load(legacy)


def test_load_all_skips_unreadable_and_sorts_newest_first(tmp_path):
    write("a", ExperimentConfig(), metrics={}, results_dir=tmp_path)
    write("b", ExperimentConfig(), metrics={}, results_dir=tmp_path)
    (tmp_path / "legacy.json").write_text('{"config": {}}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")

    records = load_all(tmp_path)
    assert {r.name for r in records} == {"a", "b"}     # the other two are skipped
    assert records == sorted(records, key=lambda r: r.created_at, reverse=True)


def test_load_all_on_missing_directory_is_empty(tmp_path):
    assert load_all(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def test_repo_root_contains_the_package():
    assert (paths.REPO_ROOT / "dprag" / "__init__.py").exists()


def test_data_paths_sit_under_data_dir():
    assert paths.HEALTHCAREMAGIC_PATH.parent == paths.DATA_DIR
    assert paths.ICLINIQ_PATH.parent == paths.DATA_DIR


def test_require_data_error_names_the_override(monkeypatch, tmp_path):
    """A missing dataset should say where it looked and how to fix it."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(paths, "HEALTHCAREMAGIC_PATH", tmp_path / "HealthCareMagic-100k.json")
    monkeypatch.setattr(paths, "ICLINIQ_PATH", tmp_path / "iCliniq-10k.json")
    with pytest.raises(FileNotFoundError, match="CHATDOCTOR_DIR"):
        paths.require_data()
