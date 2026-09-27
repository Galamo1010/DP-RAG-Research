"""Canary probe, generation step: plain DPRAG and strategy A over the injected corpus.

WHAT THIS ANSWERS
-----------------
Phase 2 established that a RAG system with no DP reproduces up to 94 consecutive
words of a retrieved doctor's reply, that plain DPRAG's longest such match is 6 words
-- below the 12-word coincidence floor set by the NoRAG control -- and that strategy
A sits on that floor too. What it could not establish is whether any *identifiable
information* is protected, because the corpus barely contains any: patient
identifiers live in HealthCareMagic's `input` field and the corpus is built from
`output`.

So this re-runs the comparison over a corpus where every document carries a planted
secret (dprag.canary). Leakage stops being a judgement about whether a phrase is
identifying and becomes an exact string match against ground truth, with a holdout
half that was never planted to price in fabrication.

WHY THIS IS NOT A POINT IN THE PHASE 2/3 GRID
---------------------------------------------
Injection changes every document, hence every embedding, hence which documents
retrieval returns. Nothing here is comparable with a run at canary_seed=None: not
the quality scores, not the trigger rate, not epsilon usage. It is a purpose-built
privacy probe and the write-up has to say so. `canary_seed` rides in the run record
via ExperimentConfig.to_dict(), so a reader can always tell which kind of run they
are holding.

WHAT RUNS HERE, AND WHAT DOES NOT
---------------------------------
Here: `baseline` (plain DPRAG, the NEVER_AGREE strategy) and `A`. Both need the k+1
aggregation, so both go through dprag.sweep.routed_sweep and inherit its
per-query checkpointing -- a run interrupted at query 60 resumes rather than
restarting.

Not here: the two poles (RAG with every document and no DP; NoRAG with none). They
are single-row greedy generations that reuse this run's recorded corpus indices, so
they belong to experiments/stage3_poles.py, which now rebuilds an injected corpus
when the source record names a canary_seed. Run it second:

    uv run python experiments/canary_run.py
    uv run python experiments/stage3_poles.py canary_baseline_eps40

Strategy B is omitted. The probe asks whether the pre-filter's *free path* leaks,
and A is the configuration that takes it most often (87.0% of positions in Phase 2).
A B variant that triggers less can only leak less by this mechanism, so it would
cost GPU hours to weaken the strongest test.

BUDGET: eps_total = 40 only. It is the noisiest-output, weakest-protection end of
the grid, so it is where a leak is most likely to appear. A clean result at 40
implies a clean result at 10; the converse does not hold, which is why the expensive
direction is the one that gets run.

Requires a CUDA GPU. Roughly 100 queries x (baseline + A); at bfloat16 expect
appreciably less than the float32 measurements of 44.9 and 36.5 s/query.
"""

from __future__ import annotations

import time

from dprag import sweep
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import strategy_a

# canary_seed must match what experiments/build_canary_corpus.py verified. Change it
# in one place only: a mismatch means the detector is searching for secrets that were
# never planted, and the probe would report a clean zero for the wrong reason.
CANARY_SEED = 20260927

EXPERIMENT = ExperimentConfig(
    n_queries=100,
    canary_seed=CANARY_SEED,
    gen_epsilon=40.0,
    # bfloat16, not the table's float32 for Llama. The table exists so a re-run
    # reproduces existing records, and this run has none to reproduce -- the corpus
    # is different. Measured locally: bf16 halves the weight bytes read per token,
    # which is what an autoregressive step is bound by (1.43x on a 1B proxy at
    # batch 1; the k+1=41 batch here is more compute-bound, so the tensor-core
    # advantage should be larger, not smaller).
    gen_dtype="bfloat16",
)

EPSILON = 40


def main() -> None:
    exp = EXPERIMENT
    strategies = {"baseline": sweep.NEVER_AGREE, "A": strategy_a}

    print(f"=== canary probe: generation | model={exp.gen_model} "
          f"dtype={exp.gen_dtype} ===")
    print(f"canary_seed={exp.canary_seed} | n_docs={exp.n_docs} | "
          f"{exp.n_queries} queries | eps_total={EPSILON}")
    print(f"configurations: {list(strategies)}")
    print("NOT comparable with any canary_seed=None run -- injection moved every "
          "embedding.\n", flush=True)

    bench = Bench.build(exp)
    began = time.time()

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature,
        max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha,
        omega=exp.omega,
        epsilon=exp.gen_epsilon,
        delta=exp.delta,
    )

    for name, strategy in strategies.items():
        print(f"--- {name} (eps_total={EPSILON}) ---", flush=True)
        sweep.routed_sweep(
            bench,
            exp,
            {name: strategy},
            dp_cfg,
            name="canary_probe",
            filename=f"canary_{name}_eps{EPSILON}",
            metrics={
                "phase": "canary probe (generation)",
                "epsilon": EPSILON,
                "configuration": name,
                "is_baseline": name == "baseline",
                "canary_seed": exp.canary_seed,
                "note": (
                    "Corpus carries one planted secret per document. Leakage is "
                    "scored by experiments/canary_detect.py, not here, so a "
                    "detector bug cannot cost a generation run. NOT comparable "
                    "with runs at canary_seed=None."
                ),
            },
        )

    print(f"\ndone in {(time.time() - began) / 60:.1f} min")
    print("next:")
    print(f"  uv run python experiments/stage3_poles.py canary_baseline_eps{EPSILON}")
    print("  uv run python experiments/canary_detect.py")


if __name__ == "__main__":
    main()
