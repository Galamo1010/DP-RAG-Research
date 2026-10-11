"""Oracle-document arm for the 20-question ordinary-RAG diagnostic.

Local: python experiments/rag_oracle.py prepare
Pod:   python experiments/rag_oracle.py generate
Local: python experiments/rag_oracle.py score

Reuses rag_diagnose's prompt and generation loop unchanged; only the arm table
and the evidence manifest differ. Decisions (author, 2026-10-12): oracle = docs
labelled "support" in results/oracle_picks.json, at most 5, in prefixed-query
rank order (then raw rank), so it matches prefix_top5 in document count. The
support criterion was "positively related to the question content", i.e.
topical relevance, not verified answer evidence. D06 is excluded: it asks about
an image, which a text-only model cannot see. The two control arms are rerun
and must be byte-identical to the 2026-10-11 run.
"""

from __future__ import annotations

import argparse

import rag_diagnose
from dprag import paths
from rag_diagnose import DEFAULT_DIR, digest, pin, read

OUT = paths.REPO_ROOT / "review" / "rag_oracle_20261012"
PICKS = paths.RESULTS_DIR / "oracle_picks.json"
EXCLUDED = {"D06": "question is about an image; text-only model cannot see it"}
LIMIT = 5
ARMS = {
    "norag_task128": ("none", "task", 128),
    "rag_task_prefix5_128": ("prefix_top5", "task", 128),
    "rag_task_oracle5_128": ("oracle_top5", "task", 128),
}
COMPARISONS = [
    ("rag_task_oracle5_128", "norag_task128", "主題相關文件 vs 無文件"),
    ("rag_task_oracle5_128", "rag_task_prefix5_128", "主題相關文件 vs 自動檢索前五"),
    ("rag_task_prefix5_128", "norag_task128", "自動檢索前五 vs 無文件（19 題重算）"),
]


def prepare() -> None:
    picks = read(PICKS)
    pool = read(DEFAULT_DIR / "oracle_candidates.json")
    retrieval = read(DEFAULT_DIR / "retrieval.json")
    if not picks["retrieval_sha256"] == pool["retrieval_sha256"] == digest(retrieval):
        raise SystemExit("oracle_picks.json was labelled on a different candidate pool")
    texts = {r["id"]: {c["index"]: c["text"] for c in r["candidates"]} for r in pool["rows"]}
    ranks = {r["id"]: r["rank"] for r in pool["rows"]}
    labels = {r["id"]: r["labels"] for r in picks["rows"]}
    rows = []
    for item in retrieval["rows"]:
        if item["id"] in EXCLUDED:
            continue
        support = {int(k) for k, v in labels[item["id"]].items() if v == "support"}
        order = ranks[item["id"]]["prefix"] + ranks[item["id"]]["raw"]
        chosen = list(dict.fromkeys(i for i in order if i in support))[:LIMIT]
        if not chosen:
            raise SystemExit(f"{item['id']} has no support document and is not excluded")
        sets = {"prefix_top5": item["document_sets"]["prefix_top5"],
                "oracle_top5": [{"index": i, "score": None, "text": texts[item["id"]][i]}
                                for i in chosen]}
        rows.append({"id": item["id"], "query": item["query"], "reference": item["reference"],
                     "document_sets": sets,
                     "oracle_in_prefix_top5": len(set(chosen) & {d["index"] for d in sets["prefix_top5"]})})
    pin(OUT / "retrieval.json", {
        "schema": 1, "picks_sha256": digest(picks), "source_retrieval_sha256": digest(retrieval),
        "excluded": EXCLUDED, "limit": LIMIT,
        "note": "oracle = author 'support' labels (topical relevance), <=5 by prefix rank",
        "rows": rows})
    for r in rows:
        print(f"{r['id']}: {len(r['document_sets']['oracle_top5'])} oracle docs, "
              f"{r['oracle_in_prefix_top5']} already in prefix top5", flush=True)


def use_oracle_arms() -> None:
    rag_diagnose.ARMS = ARMS
    rag_diagnose.render = lambda out: None  # the old review page assumes the 8-arm layout


def check_controls() -> None:
    old = {(r["id"], r["arm"]): r["emitted"] for r in read(DEFAULT_DIR / "generation.json")["rows"]}
    new = read(OUT / "generation.json")["rows"]
    controls = [r for r in new if r["arm"] != "rag_task_oracle5_128"]
    same = sum(r["emitted"] == old[(r["id"], r["arm"])] for r in controls)
    print(f"Control arms byte-identical to 2026-10-11 run: {same}/{len(controls)}", flush=True)
    if same != len(controls):
        raise SystemExit("Controls drifted from the earlier run; do not compare across runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "generate", "score"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    use_oracle_arms()
    if args.command == "generate":
        rag_diagnose.generate(OUT, "all", rag_diagnose.MODEL, args.dtype)
    else:
        import rag_diagnose_score  # local only: pulls in the BERTScore stack

        rag_diagnose_score.ARMS = ARMS
        rag_diagnose_score.COMPARISONS = COMPARISONS
        check_controls()
        rag_diagnose_score.score(OUT, True)


if __name__ == "__main__":
    main()
