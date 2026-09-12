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

from dprag import paths, run_record
from dprag.config import ExperimentConfig

MODEL = "google/gemma-4-12B-it"
FILENAME = "probe_gemma"


def outcome_filename(row: dict) -> str:
    """Where this run's record goes, which depends on whether it worked.

    A failed probe must not overwrite a successful one. The successful record is
    the feasibility evidence Phase 3 was committed on -- it carries the weights
    size, the peak memory and the environment string, and it is how a later pod
    learns which versions to rebuild the second venv with. A run that dies while
    downloading has none of that, and writing it over the top loses all of it.
    That happened twice on 2026-09-12, to a full-disk failure that had nothing to
    say about feasibility at all.

    So failures go to their own file. Both records survive, the successful one
    stays where every reference to it points, and the failed one is still on disk
    for whoever is debugging the pod.
    """
    return FILENAME if row.get("loaded") and row.get("generated") else f"{FILENAME}_failed"


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

    filename = outcome_filename(row)
    out = run_record.write(
        "probe_gemma", base,
        metrics={
            "probed": [MODEL],
            "n_queries": N_QUERIES,
            "epsilon": EPSILON,
            "succeeded": bool(row.get("loaded") and row.get("generated")),
            "environment": f"torch {torch.__version__} / transformers "
                           f"{transformers.__version__}",
            "note": (
                "Run from the second venv, which is the only environment that can "
                "load this model. Feasibility only: peak memory moves with prompt "
                "length, so leave headroom before committing Phase 3."
            ),
        },
        per_item=[row], filename=filename,
    )
    print(f"saved -> {out}")
    if filename != FILENAME:
        kept = paths.results_dir() / f"{FILENAME}.json"
        print(f"this run failed, so it was written beside "
              f"{FILENAME}.json rather than over it.")
        if kept.exists():
            print(f"the successful record is still at {kept.name}.")


if __name__ == "__main__":
    main()
