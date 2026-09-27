"""The argmax ablation on the models whose Phase 3 records already carry rag_argmax.

`stage3_4_argmax.py` is bound to the probe: ONE record holding three strategies,
plus an equivalence check against the Phase 2 answers the probe re-generated.
Phase 3 has neither shape -- one record per configuration, and no un-probed
counterpart to check against, because `rag_argmax` was written by the same run
that produced the answers.

So this script reshapes Phase 3 into the shape that script already reads, and
then calls ITS counting and ITS tables. The definitions of `wasted` and `missed`
stay in `dprag.trace` where they have always lived, and the two models cannot
drift apart on what the columns mean -- which is the only thing a second script
here could realistically get wrong.

WHAT IS NOT HERE
----------------
**No equivalence check.** There is nothing to compare against: these records are
the only generation this model ever had. The probe's check exists because the
probe RE-RAN Phase 2, and a divergence would have meant the recording changed
generation. That question does not arise here, and printing a vacuous "IDENTICAL"
would be worse than saying so.

**No figure.** `stage3_4_argmax.figure()` writes one fixed filename, so calling it
would overwrite the Llama panel. A cross-model version belongs in one combined
chart, not in three scripts each clobbering the same file.

**Not gemma.** Its Phase 3 records predate the field, and an argmax over logits
that no longer exist cannot be back-filled: gemma needs its own probe run
(`experiments/stage3_argmax_probe.py google/gemma-4-12B-it`, about an hour).

    uv run python experiments/stage3_4b_crossmodel_argmax.py
    uv run python experiments/stage3_4b_crossmodel_argmax.py <model_id_or_substring>
"""

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dprag import paths, run_record
from stage3_4_argmax import counts, order_key, paid_breakdown, paid_table, table

DEFAULT_MODEL = "Qwen2.5-14B-Instruct"
RECORD_GLOB = "stage3_3_cross_{model}_*.json"


def find_records(model: str) -> list[run_record.RunRecord]:
    pattern = str(paths.results_dir() / RECORD_GLOB.format(model=f"*{model}*"))
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(
            f"no Phase 3 record matching {Path(pattern).name}. Cross-model records "
            f"are written by experiments/stage3_3_crossmodel.py.")
    return [run_record.load(p) for p in matches]


def merge(records: list[run_record.RunRecord]) -> run_record.RunRecord:
    """One record per configuration -> one record per query, all configurations.

    Queries are intersected rather than unioned. A configuration that is missing a
    query would otherwise contribute no positions to its own column while the
    others did, and the per-configuration totals would stop being comparable.
    """
    shared: set[str] | None = None
    for record in records:
        queries = {row["query"] for row in record.per_item}
        shared = queries if shared is None else (shared & queries)
    if not shared:
        raise SystemExit("the records share no query")

    rows: dict[str, dict] = {q: {"query": q, "by_strategy": {}} for q in shared}
    names: list[str] = []
    for record in records:
        for row in record.per_item:
            if row["query"] not in rows:
                continue
            for name, strategy_record in row["by_strategy"].items():
                if "rag_argmax" not in strategy_record:
                    raise SystemExit(
                        f"{record.path.name} carries no rag_argmax for {name}; this "
                        f"model needs a probe run (see the module docstring).")
                rows[row["query"]]["by_strategy"][name] = strategy_record
                if name not in names:
                    names.append(name)

    first = records[0]
    return run_record.RunRecord(
        name="stage3_4b_crossmodel_argmax",
        created_at=first.created_at,
        params=first.params,
        metrics={"strategies": names, "n_queries_scored": len(rows)},
        per_item=[rows[q] for q in sorted(rows)],
        path=first.path,
    )


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model = model.split("/")[-1]
    records = find_records(model)
    merged = merge(records)

    print(f"source: {len(records)} Phase 3 records, intersected on query")
    for record in records:
        print(f"  {record.path.name}")
    print()
    print("=== equivalence: not applicable ===")
    print("These records ARE the generation; rag_argmax was recorded in the same")
    print("run. The probe's token-for-token check exists because the probe re-ran")
    print("Phase 2, and that question does not arise here.")
    print()

    totals: dict[str, dict[str, int]] = {}
    paid_totals: dict[str, dict[str, int]] = {}
    for row in merged.per_item:
        for name, strategy_record in row["by_strategy"].items():
            acc = totals.setdefault(
                name, {"positions": 0, "paid": 0, "free": 0, "wasted": 0, "missed": 0})
            for key, value in counts(strategy_record).items():
                acc[key] += value
            pacc = paid_totals.setdefault(
                name, {"paid": 0, "contested": 0, "followed_rag": 0,
                       "followed_norag": 0, "neither": 0})
            for key, value in paid_breakdown(strategy_record).items():
                pacc[key] += value

    table(merged, totals)
    paid_table(paid_totals)
    print()
    print("Read this against the same model's own baseline row, never against")
    print("Llama's: the two models ran at different precisions, in different")
    print("library versions, and their vocabularies differ by 1.2x.")


if __name__ == "__main__":
    main()
