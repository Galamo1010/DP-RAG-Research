"""Does Qwen2.5-14B now fit at max_retrieve=40.

It did not before. The router computed lm_head over every position of the prompt
and kept only the last, which at k+1 = 41 rows and Qwen's 152,064-token vocabulary
is a 9.8 GB tensor built and discarded on every prefill -- plus whatever transient
copies the model makes of it. That is gone (`logits_to_keep=1`), and gemma-4 went
from OOM on its fifth query to running the full phase at 57.8 GB peak.

Whether the same reprieve is enough for Qwen is not something to reason about. Its
weights are about 6 GB heavier than gemma's and its KV cache deeper, and the
measurement that would settle it was lost when the earlier probe crashed before
writing its record.

**The queries are the longest in the set, not the first three.** Peak memory is set
by the longest prompt in the batch, and probing the first three is how gemma-4 was
cleared at 40 and then failed on the fifth query of a real run. If Qwen survives
these, it survives the phase.

Runs in the pinned environment: Qwen loads there, and that is where its Phase 3
runs would happen.

    uv run python experiments/probe_qwen.py
"""

import torch

from probe_models import EPSILON, N_QUERIES, probe, worst_case_queries

from dprag import run_record
from dprag.config import ExperimentConfig

MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Whose retrieval to rank by. Any finished record carries the corpus indices; the
# eps=40 baseline is the run Phase 3 compares against, so its prompts are the ones
# Qwen would actually face.
SOURCE = "stage3_2_main_baseline_eps40"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this probe only means anything on the pod")

    import transformers
    base = ExperimentConfig()
    queries = worst_case_queries(base, N_QUERIES, SOURCE)

    print(f"=== qwen probe | max_retrieve={base.max_retrieve} | eps={EPSILON:.0f} "
          f"| {len(queries)} longest-prompt queries ===")
    print(f"environment: torch {torch.__version__}, transformers "
          f"{transformers.__version__}")
    print(f"model: {MODEL}")
    print(f"prompts ranked from: {SOURCE}")
    for q in queries:
        print(f"  - {q[:78]}")
    print(flush=True)

    row = probe(MODEL, base, queries=queries)
    row["queries"] = queries
    row["query_selection"] = f"longest prompts in {SOURCE}"

    print("=== verdict ===")
    print(f"  loads      : {'yes' if row['loaded'] else 'NO'}")
    print(f"  runs at {base.max_retrieve:2d} : "
          f"{'yes' if row['generated'] else ('OOM' if row['oom'] else 'NO')}")
    print(f"  peak       : {row.get('peak_gb', 0)} GB of {row.get('gpu_total_gb', 0)} GB")
    print(f"  seconds    : {row.get('seconds_per_query', 0)} per query")
    if row["error"]:
        print(f"\n  [{row.get('stage')}] {row['error']}")
    print()
    if row["generated"]:
        print("  These were the worst prompts in the set, so the phase should hold.")
        print("  All three models then run at max_retrieve=40 and the cross-model")
        print("  comparison is not confounded by how much evidence each one saw.")
    print()

    out = run_record.write(
        "probe_qwen", base,
        metrics={
            "probed": [MODEL],
            "n_queries": N_QUERIES,
            "epsilon": EPSILON,
            "query_selection": row["query_selection"],
            "environment": f"torch {torch.__version__} / transformers "
                           f"{transformers.__version__}",
            "note": (
                "Worst-case feasibility: the longest prompts in the query set, at "
                "the max_retrieve Phase 2 and the gemma-4 phase both used."
            ),
        },
        per_item=[row], filename="probe_qwen",
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
