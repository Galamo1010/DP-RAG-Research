"""Stage 3.2 phase 1 -- screen the twelve Strategy B configurations cheaply.

The proposal specifies a (k, tau) grid of 3 x 4 and, in the same breath, the reason
not to run it at full scale:

> 策略B的12組超參數配置若全數進行完整實驗，API呼叫量龐大；建議先以小批量查詢
> （各20筆）篩選出Pareto最優配置，再對入圍配置進行完整200筆測試，以控制總費用

So this is the proposal's own cost-control step, not a deviation from it. Twelve
configurations at twenty queries produces the trade-off curve the grid exists to
draw, and the expensive phase then runs only the configurations worth the money.

WHAT COMES OUT, AND WHAT DOES NOT
---------------------------------
Trigger rate and epsilon savings come out directly. **Quality does not** -- nothing
is scored here, as everywhere in Stage 3. Run `experiments/stage3_score.py` against
this record to get the curve with both axes.

That matters for how the finalists are picked: the choice is a human reading of the
trade-off, made after scoring, not something this script decides. Phase 2 names its
finalists explicitly and checks they appear in this record.

TWO THINGS TO EXPECT IN THE OUTPUT
----------------------------------
**tau reads looser than it is.** At k=10, tau=0.7 needs 9 of 10 top-k tokens to
match (`strategies.min_overlap_for_tau`), so several configurations will trigger
far *below* strategy A rather than above it. A reader assuming tau=0.7 means "70%
similar" will misread the whole table.

**Twenty queries is a screen, not a measurement.** Trigger rate over 20 x 128
positions is reasonably stable; anything derived per query is not. Nothing here
should be quoted as a result.

    uv run python experiments/stage3_1_screen.py
"""

import time

from dprag import sweep
from dprag.bench import Bench
from dprag.config import STRATEGY_B_K_GRID, STRATEGY_B_TAU_GRID, ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import make_strategy_b, min_overlap_for_tau, strategy_a

# The proposal's screening size, and its single budget point.
EXPERIMENT = ExperimentConfig(n_queries=20)
EPSILON = 10.0

# All twelve B configurations, plus A as the reference every reader will look for.
# A is not an upper bound on B -- ADR 0007 records why the proposal's framing of
# that was falsified -- but it is the configuration the comparison is against.
STRATEGIES = {"A": strategy_a}
for _k in STRATEGY_B_K_GRID:
    for _tau in STRATEGY_B_TAU_GRID:
        STRATEGIES[f"B_k{_k}_t{_tau}"] = make_strategy_b(_k, _tau)


def main():
    exp = EXPERIMENT
    print(f"=== Stage 3.2 phase 1: screening | model={exp.gen_model} | "
          f"{exp.n_queries} queries x {len(STRATEGIES)} configurations | "
          f"eps={EPSILON} | max_retrieve={exp.max_retrieve} ===")
    print()
    print("What each tau actually demands (k, tau -> tokens that must match):")
    for k in STRATEGY_B_K_GRID:
        row = "  k=%-3d " % k
        for tau in STRATEGY_B_TAU_GRID:
            row += "tau=%.1f:%2d/%-3d " % (tau, -(-min_overlap_for_tau(k, tau) // 1), k)
        print(row)
    print(flush=True)

    bench = Bench.build(exp)
    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=EPSILON, delta=exp.delta)

    started = time.time()
    sweep.routed_sweep(
        bench, exp.with_(gen_epsilon=EPSILON), STRATEGIES, dp_cfg,
        name="stage3_1_screen",
        filename=f"stage3_1_screen_{exp.n_queries}q",
        metrics={
            "phase": "1 (screening)",
            "epsilon": EPSILON,
            "purpose": (
                "the proposal's own cost control: draw the (k, tau) trade-off "
                "curve at small batch, then spend the large budget only on the "
                "configurations that survive it"
            ),
        },
    )
    print(f"\nTotal {(time.time()-started)/60:.1f} min")
    print()
    print("Next: score this record, read the trade-off curve, and name the")
    print("finalists at the top of experiments/stage3_2_main.py. The choice is")
    print("yours to make from the curve -- this script does not make it.")


if __name__ == "__main__":
    main()
