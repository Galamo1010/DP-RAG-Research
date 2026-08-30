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
Five configurations across two budgets: ten runs of two hundred queries. At the
measured 40.8 s per generation -- 20 queries at max_retrieve=40, not the 10 s ADR
0003 assumed at max_retrieve=10 -- that is roughly twenty-one hours. Filling in the
remaining two budgets would cost the same again; ADR 0009 records why the ends run
first. Progress is checkpointed per query and completed runs are skipped, so this
is safe to interrupt and safe to re-run.

    uv run python experiments/stage3_2_main.py
"""

import time

from dprag import sweep
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import make_strategy_b, strategy_a

EXPERIMENT = ExperimentConfig(n_queries=200)

# Three Strategy B configurations spanning the strictness range, rather than the
# two Pareto-optimal ones ADR 0003 assumed. The trade-off is the finding here, and
# a trade-off needs a spread: two points near the frontier can sit almost on top of
# each other and show no shape at all. Stage 2.5 ran exactly this spread and the
# result was a clean dose-response -- quality rose monotonically as the paid
# fraction fell -- which is the evidence for choosing it again.
#
# The strictness is the *measured* trigger rate, not tau. tau reads far looser than
# it is (k=20, tau=0.9 demands 19 of 20 tokens) and k changes behaviour on its own,
# so the labels below are what Stage 2.5 measured at max_retrieve=10.
#
# WARNING: those measurements predate ADR 0008. At max_retrieve=40 the DP
# aggregation is sharper -- Stage 1.2's sampled consistency went from 0.11 to 0.75
# on that change alone -- so these three may no longer span the range. Phase 1
# exists to check that before twenty hours are spent on it.
FINALISTS = {
    "B_k20_t0.9": make_strategy_b(20, 0.9),   # strict:  19/20, triggered 14.4%
    "B_k20_t0.7": make_strategy_b(20, 0.7),   # middle:  17/20, triggered 59.1%
    "B_k50_t0.5": make_strategy_b(50, 0.5),   # loose:   34/50, triggered 95.3%
}

# The proposal specifies eps_total in {5, 10, 20, 40}. Only the two ends run first
# (ADR 0009): they are where every existing measurement sits, so the new numbers
# are directly comparable, and quality at eps=40 was already indistinguishable
# across configurations -- the interior points are the least likely to carry
# information. Extend this list to fill the grid in.
EPSILON_GRID = [10, 40]

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
    total = len(EPSILON_GRID) * len(strategies)
    print(f"=== Stage 3.2 phase 2: main comparison | model={exp.gen_model} ===")
    print(f"{len(strategies)} configurations x {len(EPSILON_GRID)} budgets "
          f"x {exp.n_queries} queries | max_retrieve={exp.max_retrieve}")
    print(f"configurations: {list(strategies)}")
    print(f"budgets: {EPSILON_GRID}  (proposal asks for [5,10,20,40]; see ADR 0009)")
    print(f"~{total} runs; checkpointed per query, completed runs skipped\n",
          flush=True)

    bench = Bench.build(exp)
    began = time.time()

    for eps in EPSILON_GRID:
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
