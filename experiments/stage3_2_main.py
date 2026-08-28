"""Stage 3.2 phase 2 -- the comparison the whole project is for.

Plain DPRAG, strategy A, and the finalists from phase 1, across the proposal's
epsilon grid at full query count. This is where "the pre-filter saves epsilon and
does not hurt quality" is established or fails.

On the evidence so far the claim is stronger than the proposal expected. Comparing
Stage 1.2's plain-DPRAG output against Stage 2.5's routed output at the same
budget, ROUGE-L rose from 0.089 to 0.122 -- and the improvement tracks how little
epsilon was spent, monotonically, across four configurations. The mechanism is not
subtle: the free path emits a clean NoRAG token where the paid path emits a
DP-sampled one, so skipping the aggregation skips its noise. That comparison was
indicative only (different days, hence different prompts before the date pin, and
two unpaired runs). **This phase is the paired version**, one process, one prompt,
one seed, baseline and treatment side by side.

THE OBJECTION THIS PHASE CANNOT ANSWER ALONE
--------------------------------------------
"You did less DP, so of course quality improved." Nothing measured here refutes
that, because a system that ignored every document could post good ROUGE-L too. The
refutation comes from `stage3_poles.py` -- where routed output sits between a
no-document and an all-document generation -- and from the paid-position and
grounding analyses in `stage3_score.py`. Run those before quoting any quality
number from this phase.

THE BASELINE IS THE ROUTER
--------------------------
Plain DPRAG is produced by routing with a strategy that never agrees, not by a
separate path (ADR 0003). The equivalence test pins that this reproduces `dp_chat`
token for token, so baseline and treatment cannot differ by setup -- only by the
strategy, which is the point.

COST
----
Sixteen runs of two hundred queries. At the measured 14-17 s per generation that is
roughly fifteen hours, not the ten the ADR assumed at 10 s. Progress is
checkpointed per query and completed runs are skipped, so this is safe to
interrupt and safe to re-run.

    uv run python experiments/stage3_2_main.py
"""

import time

from dprag import sweep
from dprag.bench import Bench
from dprag.config import EPS_TOTAL_GRID, ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import make_strategy_b, strategy_a

EXPERIMENT = ExperimentConfig(n_queries=200)

# Named here rather than inferred, because picking them is a human reading of
# phase 1's trade-off curve. Change these after looking at the curve; the check
# below only confirms the names appear in phase 1, not that they were wise.
FINALISTS = {
    "B_k20_t0.7": make_strategy_b(20, 0.7),
    "B_k50_t0.5": make_strategy_b(50, 0.5),
}

SCREEN_RECORD = "stage3_1_screen_20q"


def build_strategies() -> dict:
    return {"baseline": sweep.NEVER_AGREE, "A": strategy_a, **FINALISTS}


def check_finalists_were_screened() -> None:
    """Fail loudly if the finalists were not among the screened configurations.

    A phase that silently compares against configurations nobody screened is worse
    than one that refuses to start (spec, 'fail loudly when a prerequisite result
    is missing').
    """
    from dprag import paths, run_record

    path = paths.results_dir() / f"{SCREEN_RECORD}.json"
    if not path.exists():
        raise SystemExit(
            f"{path.name} not found. Phase 1 selects what this phase spends "
            "fifteen hours on; run experiments/stage3_1_screen.py first.\n"
            "To proceed anyway, comment out this check and say so in the report."
        )
    screened = set(run_record.load(path).metric("strategies", []))
    missing = [name for name in FINALISTS if name not in screened]
    if missing:
        raise SystemExit(
            f"finalists {missing} do not appear in {path.name}, which screened "
            f"{sorted(screened)}.\nEither the names are wrong or the finalists "
            "were chosen without evidence."
        )


def main():
    exp = EXPERIMENT
    check_finalists_were_screened()

    strategies = build_strategies()
    total = len(EPS_TOTAL_GRID) * len(strategies)
    print(f"=== Stage 3.2 phase 2: main comparison | model={exp.gen_model} ===")
    print(f"{len(strategies)} configurations x {len(EPS_TOTAL_GRID)} budgets "
          f"x {exp.n_queries} queries | max_retrieve={exp.max_retrieve}")
    print(f"configurations: {list(strategies)}")
    print(f"budgets: {EPS_TOTAL_GRID}")
    print(f"~{total} runs; checkpointed per query, completed runs skipped\n",
          flush=True)

    bench = Bench.build(exp)
    began = time.time()

    for eps in EPS_TOTAL_GRID:
        dp_cfg = DPGenerationConfig(
            temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
            alpha=exp.alpha, omega=exp.omega, epsilon=float(eps), delta=exp.delta)
        clipping = dp_cfg.token_epsilon() * exp.temperature / 2
        print(f"--- eps={eps}  (clipping={clipping:.4f}) ---", flush=True)

        for name, strategy in strategies.items():
            print(f"  {name}", flush=True)
            sweep.routed_sweep(
                bench, exp.with_(gen_epsilon=float(eps)), {name: strategy}, dp_cfg,
                name="stage3_2_main",
                filename=f"stage3_2_main_{name}_eps{eps}",
                metrics={
                    "phase": "2 (main comparison)",
                    "epsilon": eps,
                    "configuration": name,
                    "is_baseline": name == "baseline",
                },
            )

    print(f"\nTotal {(time.time()-began)/3600:.1f} h")
    print()
    print("Next, in this order:")
    print("  1. experiments/stage3_poles.py   -- the no-document and all-document")
    print("     reference points, without which a quality gain cannot be told")
    print("     apart from simply having done less RAG")
    print("  2. experiments/stage3_score.py   -- quality, grounding, and whether")
    print("     the epsilon that was spent bought a different token")


if __name__ == "__main__":
    main()
