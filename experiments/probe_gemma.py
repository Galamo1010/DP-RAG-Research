"""The same probe as probe_models.py, from the environment gemma-4 can load in.

`probe_models.py` runs under the pinned environment, where gemma-4 fails at import
time -- so it can never answer whether gemma-4 *generates* at max_retrieve=40. That
question only exists inside the second venv, and this is the entry point for it.

The probing logic is imported rather than copied, so the two environments measure
the same thing in the same way. Only the model list and the output filename differ,
the latter because both records live in the same results directory.

gemma-4 has one factor the other two do not: a 262,144-token vocabulary, twice
Llama's. That grows the embedding and lm_head weights and every logit tensor the DP
aggregation touches, so its 12B parameters do not place it neatly between Llama's 8B
and Qwen's 14B. Measuring is the point.

    PYTHONPATH=. .venv-gemma/bin/python experiments/probe_gemma.py
"""

import torch

from probe_models import EPSILON, N_QUERIES, probe

from dprag import run_record
from dprag.config import ExperimentConfig

MODEL = "google/gemma-4-12B-it"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this probe only means anything on the pod")

    import transformers
    base = ExperimentConfig()
    print(f"=== gemma probe | max_retrieve={base.max_retrieve} | eps={EPSILON:.0f} "
          f"| {N_QUERIES} queries ===")
    print(f"environment: torch {torch.__version__}, transformers "
          f"{transformers.__version__}")
    print(f"model: {MODEL}\n", flush=True)

    row = probe(MODEL, base)

    print("=== verdict ===")
    print(f"  loads      : {'yes' if row['loaded'] else 'NO'}")
    print(f"  runs at 40 : "
          f"{'yes' if row['generated'] else ('OOM' if row['oom'] else 'NO')}")
    print(f"  peak       : {row.get('peak_gb', 0)} GB of {row.get('gpu_total_gb', 0)} GB")
    print(f"  seconds    : {row.get('seconds_per_query', 0)} per query")
    if row["error"]:
        print(f"\n  [{row.get('stage')}] {row['error']}")
    print()

    out = run_record.write(
        "probe_gemma", base,
        metrics={
            "probed": [MODEL],
            "n_queries": N_QUERIES,
            "epsilon": EPSILON,
            "environment": f"torch {torch.__version__} / transformers "
                           f"{transformers.__version__}",
            "note": (
                "Run from the second venv, which is the only environment that can "
                "load this model. Feasibility only: peak memory moves with prompt "
                "length, so leave headroom before committing Phase 3."
            ),
        },
        per_item=[row], filename="probe_gemma",
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
