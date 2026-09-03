"""Phase 3: does the pre-filter's advantage survive a change of model.

Every number this project has is from Llama-3.1-8B. "Strategy A saves 78% of the
budget without losing the documents" could be a property of the method or a
property of that one model, and nothing measured so far can tell the two apart.
This runs the same three configurations on a second model so that it can.

**Llama's half is already done.** Phase 2's eps=40 records hold baseline, A and
B_k20_t0.7 at max_retrieve=40 over the same 180 queries, so this only pays for the
new model. A missing Phase 2 record is therefore fatal and is checked for up front.

**Three configurations, not five.** baseline and A are the comparison the phase
exists for; B_k20_t0.7 is here because it is the one configuration that leaks --
significantly closer to the no-document pole than baseline (lean -0.113 at eps=40)
while saving only 59% -- and whether that failure reproduces elsewhere is the
sharpest available test of the mechanism behind it.

**One budget.** eps=40, where the epsilon-productivity measure is least polluted
by noise: at eps=10 a token pushed off NoRAG's argmax by noise counts as
"productive" identically to one pushed by a document, and baseline's productivity
falls from 20.7% to 14.9% between the budgets for exactly that reason.

The comparison is within each model, never across them. Models differ in absolute
quality for reasons that have nothing to do with the pre-filter, so what transfers
(or fails to) is each configuration's improvement over its own baseline.

    # from the environment that can load the model
    PYTHONPATH=. .venv-gemma/bin/python experiments/stage3_3_crossmodel.py
    PYTHONPATH=. .venv-gemma/bin/python experiments/stage3_3_crossmodel.py <model_id>
"""

import os
import sys
import time

# Before torch is imported, or the allocator is already configured. Qwen2.5-14B
# peaks at 75.1 of 79.3 GB here: without this it fails on an allocation of 1.19 GB
# while 5.35 GB sits in reserved-but-unallocated fragments. It is a memory-manager
# setting and touches no arithmetic, so the output is unaffected.
#
# It lives here rather than in env.sh because env.sh is not in git: a reconnected
# shell, or a new pod, would silently lose it and the phase would OOM hours in.
# setdefault, so an explicit export still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dprag import paths, sweep
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import make_strategy_b, strategy_a

DEFAULT_MODEL = "google/gemma-4-12B-it"
EPSILON = 40
EXPERIMENT = ExperimentConfig(n_queries=200)

# The same three names Phase 2 used, so a reader comparing the two files is
# comparing like with like rather than decoding a second vocabulary.
CONFIGS = {
    "baseline": sweep.NEVER_AGREE,
    "A": strategy_a,
    "B_k20_t0.7": make_strategy_b(20, 0.7),
}

# What this phase compares against. Without them there is no Llama arm and the
# runs below measure a model rather than a transfer.
COMPARISON_RECORDS = [f"stage3_2_main_{name}_eps{EPSILON}" for name in CONFIGS]


def check_comparison_arm_exists() -> None:
    missing = [n for n in COMPARISON_RECORDS
               if not (paths.results_dir() / f"{n}.json").exists()]
    if missing:
        raise SystemExit(
            f"missing Phase 2 records: {missing}\n"
            "Phase 3 compares a second model against Llama at the same budget and "
            "the same configurations. Without those files there is nothing to "
            "compare to, and this would spend hours measuring one model in "
            "isolation. Run experiments/stage3_2_main.py first."
        )


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    exp = EXPERIMENT.with_(gen_model=model, gen_epsilon=float(EPSILON))
    check_comparison_arm_exists()

    short = model.split("/")[-1]
    print(f"=== Stage 3.2 phase 3: cross-model | model={model} ===")
    print(f"{len(CONFIGS)} configurations x 1 budget x {exp.n_queries} queries "
          f"| max_retrieve={exp.max_retrieve} | eps={EPSILON}")
    print(f"configurations: {list(CONFIGS)}")
    print(f"compared against: {COMPARISON_RECORDS}")
    print("checkpointed per query; completed runs are skipped\n", flush=True)

    bench = Bench.build(exp)
    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=float(EPSILON), delta=exp.delta,
    )
    print(f"clipping = {dp_cfg.token_epsilon() * exp.temperature / 2:.4f}", flush=True)

    began = time.time()
    for name, strategy in CONFIGS.items():
        print(f"--- {name} ---", flush=True)
        sweep.routed_sweep(
            bench, exp, {name: strategy}, dp_cfg,
            name="stage3_3_crossmodel",
            filename=f"stage3_3_cross_{short}_{name}_eps{EPSILON}",
            metrics={
                "phase": "3 (cross-model)",
                "epsilon": EPSILON,
                "configuration": name,
                "is_baseline": name == "baseline",
                "compared_against": f"stage3_2_main_{name}_eps{EPSILON}",
            },
        )

    print(f"\nTotal {(time.time() - began) / 3600:.1f} h")
    print()
    print("Next:")
    print(f"  1. experiments/stage3_poles.py stage3_2_main_baseline_eps{EPSILON} "
          f"{model}")
    print("     -- this model's own poles. They are model-specific, and without")
    print("     them 'the documents still shape the answer' cannot be checked here.")
    print("     The documents are reused by index, so nothing is retrieved again.")
    print("  2. experiments/stage3_score.py -- one Pareto table per model")


if __name__ == "__main__":
    main()
