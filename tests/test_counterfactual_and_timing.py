"""Tests for the two runs that cannot be checked after the fact.

Both of these spend GPU hours and produce numbers that look reasonable whether or
not the machinery underneath them is right, which is the dangerous combination.

The counterfactual's substitution is the clearest case. If it quietly returned the
retrieved documents, or documents overlapping them, the run would finish, the
report would print, and the conclusion -- "similarity did not drop, so claim 4's
pole evidence must be withdrawn" -- would be exactly backwards. Nothing downstream
could catch it.

The timing threshold is arithmetic, but arithmetic with three branches and an
interpolation that runs in the opposite direction to intuition: a LARGER ratio
(plain DPRAG relatively slower) crosses the curve EARLIER, at a lower trigger
rate.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(stem: str):
    """Import an experiment module by path.

    Experiments are executed, not imported, so they are not on the package path;
    `main()` is guarded, so loading one costs a few constants and no model.
    """
    path = ROOT / "experiments" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


counterfactual = _load("stage3_5_counterfactual")
timing = _load("stage5_1_plain_timing")
probe = _load("stage3_argmax_probe")


# --------------------------------------------------------------------------
# the counterfactual's substitution
# --------------------------------------------------------------------------

CORPUS = [f"doctor reply {i}" for i in range(500)]


def _substitute():
    return counterfactual.make_substituter(CORPUS, seed=42)


def test_substitutes_share_no_document_with_what_was_retrieved():
    """The property the whole control rests on. An overlap would leave real
    evidence in the prompt and weaken the contrast without announcing it."""
    substitute = _substitute()
    retrieved = CORPUS[:40]
    got = substitute(retrieved, "a patient question")
    assert not set(got) & set(retrieved)


def test_substitute_count_matches_the_retrieved_count():
    """Document count drives prompt length, VRAM and the k+1 batch width. Changing
    it would change the timing and the aggregation alongside relevance, so the run
    would no longer isolate one variable."""
    substitute = _substitute()
    for n in (1, 7, 40):
        assert len(substitute(CORPUS[:n], "q")) == n


def test_substitutes_are_not_repeated_within_one_query():
    substitute = _substitute()
    got = substitute(CORPUS[:40], "q")
    assert len(set(got)) == len(got)


def test_the_same_query_always_gets_the_same_substitutes():
    """Seeded from the question, so a resumed run continues with the documents an
    uninterrupted one would have used. The checkpoint restores rows, not RNG
    state, so anything seeded from a counter would silently diverge on resume."""
    a = counterfactual.make_substituter(CORPUS, seed=42)
    b = counterfactual.make_substituter(CORPUS, seed=42)
    assert a(CORPUS[:40], "q") == b(CORPUS[:40], "q")


def test_substitutes_do_not_depend_on_how_many_queries_ran_before():
    """The failure a per-run RNG would produce: query 5's documents differing
    between a fresh run and a resumed one."""
    fresh = _substitute()
    resumed = _substitute()
    for q in ("q1", "q2", "q3", "q4"):
        fresh(CORPUS[:10], q)
    assert fresh(CORPUS[:10], "q5") == resumed(CORPUS[:10], "q5")


def test_different_queries_get_different_substitutes():
    substitute = _substitute()
    assert substitute(CORPUS[:40], "q1") != substitute(CORPUS[:40], "q2")


def test_a_corpus_too_small_to_draw_from_fails_loudly():
    """Rather than returning fewer documents, which would change the batch width
    on some queries and not others."""
    substitute = counterfactual.make_substituter(CORPUS[:10], seed=42)
    with pytest.raises(SystemExit):
        substitute(CORPUS[:10], "q")


# --------------------------------------------------------------------------
# the timing threshold
# --------------------------------------------------------------------------

# (configuration, trigger rate, seconds) -- the Phase 2 shape, Llama at eps=40.
CURVE = [
    ("baseline", 0.000, 44.9),
    ("B_k20_t0.9", 0.181, 43.1),
    ("B_k20_t0.7", 0.663, 38.1),
    ("A", 0.870, 36.5),
    ("B_k50_t0.5", 0.958, 33.2),
]


def test_threshold_interpolates_inside_the_measured_range():
    # 0.961 is exactly B_k20_t0.9's relative cost, so the crossing is its trigger
    # rate, not somewhere between the neighbours.
    crossed, _ = timing.threshold(CURVE, 43.1 / 44.9)
    assert crossed == pytest.approx(0.181, abs=1e-6)


def test_a_slower_plain_dprag_crosses_earlier():
    """The direction that reads backwards. A ratio nearer 1.0 means plain DPRAG is
    relatively slower, so a smaller trigger rate already beats it."""
    high, _ = timing.threshold(CURVE, 0.99)
    low, _ = timing.threshold(CURVE, 0.80)
    assert high < low


def test_plain_dprag_no_faster_than_the_routed_baseline_is_reported_not_crashed():
    """A real possible outcome: the pre-filter batch costs so little that plain
    DPRAG is not cheaper even at zero trigger."""
    crossed, note = timing.threshold(CURVE, 1.05)
    assert crossed is None
    assert "at or below 0%" in note


def test_plain_dprag_cheaper_than_everything_measured_is_reported_not_crashed():
    crossed, note = timing.threshold(CURVE, 0.10)
    assert crossed is None
    assert "above the measured range" in note


def test_the_anchor_is_the_zero_trigger_point_not_the_list_order():
    """The curve is sorted by trigger rate before use, so a records glob returning
    a different order cannot silently normalise by the wrong configuration."""
    shuffled = [CURVE[3], CURVE[0], CURVE[4], CURVE[1], CURVE[2]]
    ordered = sorted(shuffled, key=lambda t: t[1])
    assert timing.threshold(ordered, 0.90) == timing.threshold(CURVE, 0.90)


# --------------------------------------------------------------------------
# the probe's per-model naming
# --------------------------------------------------------------------------

def test_the_primary_model_keeps_the_filename_already_on_disk():
    """Renaming it would orphan the record every claim-4 and claim-5 number is
    quoted from."""
    assert probe.filename_for(probe.PRIMARY_MODEL) == "stage3_argmax_probe_60q_eps40"


def test_another_model_gets_its_own_file():
    name = probe.filename_for("google/gemma-4-12B-it")
    assert name == "stage3_argmax_probe_gemma-4-12B-it_60q_eps40"


def test_each_model_is_checked_against_its_own_un_probed_run():
    """The primary model's lives in Phase 2, every other model's in Phase 3.
    Comparing gemma's probe against Llama's Phase 2 would report every query as
    missing and print a verdict about nothing."""
    assert probe.comparison_stem(probe.PRIMARY_MODEL, "A") == "stage3_2_main_A_eps40"
    assert (probe.comparison_stem("google/gemma-4-12B-it", "A")
            == "stage3_3_cross_gemma-4-12B-it_A_eps40")
