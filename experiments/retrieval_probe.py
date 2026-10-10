"""Retrieval only: how top_p shapes what reaches the model.

Started as a diagnosis of the 20 zero-document queries in Stage 3.2's baseline at
eps=40 (one in ten retrieving nothing and silently running as NoRAG). That
diagnosis came back as: `_exp_mechanism_top_p_threshold` weights each candidate
threshold by the WIDTH of its score interval, and scores are clipped at
max_score=0.8, so almost every query has an empty interval above its best document
that soaks up a few percent of the mass. The per-query probability of drawing a
zero-document threshold is ~4.5% for well-matched queries and rises as the best
score falls -- 20 observed against 16.4 expected, i.e. mostly chance, not "the
corpus has no answer".

Now a sweep over top_p, because the other retrieval defect is the cap: ~87
documents clear the threshold and `max_retrieve`=40 of them are drawn AT RANDOM.
A smaller top_p raises the ideal threshold, so fewer documents qualify and the
random cap binds less often. What that costs is what this measures:

  * zero-document queries -- observed under the seed, and expected from the exact
    per-query probability, which does not depend on one lucky draw
  * how many qualify, and how often the cap has to discard
  * how many of the true top-10 are retrieved
  * what share of the retrieved documents are among the true top-40 -- the
    relevance of the Baseline's voters, which is what the DP aggregation needs

Two query sets. The official 200 (query_seed=42) reproduce the recorded pipeline;
at top_p=0.02 they must yield the exact 20 zero-document queries the baseline
record excluded, or this script is not measuring the pipeline. A disjoint dev set
of 200 exists so a top_p can be CHOSEN without choosing it on the test queries.

No generation model is loaded -- only the embedder -- so this runs on a laptop.
Document embeddings are cached under cache/, keyed by the corpus and embedder, so a
re-run with other retrieval settings takes seconds.

    uv run python experiments/retrieval_probe.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time

import numpy as np
import torch

from dprag import paths
from dprag.chatdoctor import load_corpus, load_queries
from dprag.config import ExperimentConfig
from dprag.pup_vector_store import PUPVectorStore, PUPVectorStoreConfig

TOP_P_GRID = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]   # 0.05 is a contrast point
# top_k mode: the other utility pup_retrieve offers. Same width-weighted pdf, but
# utility counts RANKS off target, so at eps=0.2 each rank costs only e^-0.1.
TOP_K_GRID = [5, 10, 20, 30, 40, 50, 60, 80]
# Lowering max_score shrinks the empty interval above each query's best document,
# which is what produces zero-document draws. Probed only around the top_k values
# that are candidates for the generation run.
MAX_SCORE_GRID = [0.75, 0.7, 0.65]
MAX_SCORE_K = [30, 40, 50]
DEV_SEED = 43
BASELINE_RECORD = "stage3_2_main_baseline_eps40.json"
TRUE_TOP = 10
RELEVANT_TOP = 40     # "relevant voter" = among the true top-max_retrieve


def zero_document_probability(store: PUPVectorStore, scores: np.ndarray) -> float:
    """P(threshold >= best score), from the same pdf the mechanism samples.

    Mirrors `_exp_mechanism_top_p_threshold` (or `_top_k_`, whichever the store
    would use) line for line up to the draw. The
    caller checks the mirror against the real draw, so a divergence shows up as a
    failed assertion rather than as a plausible wrong number.
    """
    s = np.sort(scores)
    s = np.insert(s, 0, -1)
    s = np.insert(s, len(s), 1)
    s = np.clip(s, store.min_score, store.max_score)
    if store.top_p is not None:
        w = np.exp(store.top_p_alpha * (s - store.max_score) / (store.max_score - store.min_score))
        u = -np.abs(np.sum(w) * (1 - store.top_p) - np.cumsum(w))
    else:
        u = -np.abs(len(s) - store.top_k - np.arange(len(s)))
    pdf = np.exp(store.epsilon * u[:-1] / 2) * np.diff(s)
    pdf /= np.sum(pdf)
    # pup_retrieve keeps score > threshold, so a threshold at or above the best
    # (clipped) score admits nothing.
    best = min(float(np.max(scores)), store.max_score)
    return float(np.sum(pdf[s[:-1] >= best]))


def build_store(cfg: ExperimentConfig) -> PUPVectorStore:
    store = PUPVectorStore(PUPVectorStoreConfig(
        model_id=cfg.embed_model,
        top_p=cfg.retrieval_top_p,
        top_k=cfg.retrieval_top_k,
        min_score=cfg.min_score,
        max_score=cfg.max_score,
        epsilon=cfg.eps_retrieval,
        max_retrieve=cfg.max_retrieve,
        batch_size=cfg.embed_batch_size,
        seed=cfg.seed,
    ))
    for doc in load_corpus(limit=cfg.n_docs, sample_seed=cfg.corpus_seed):
        store.add(doc)

    # Keyed by the documents themselves, not by n_docs/corpus_seed, so a change to
    # the loader or the corpus cannot silently reuse stale vectors.
    key = hashlib.sha256(
        (cfg.embed_model + "\0" + "\0".join(store.store)).encode("utf-8")
    ).hexdigest()[:16]
    cache = paths.REPO_ROOT / "cache" / f"doc_embeddings_{key}.pt"
    device = store.model.device
    if cache.exists():
        store._embeddings = torch.load(cache, map_location=device)
        print(f"Loaded {len(store.store)} doc embeddings from {cache.name}")
    else:
        started = time.time()
        torch.save(store.embeddings().cpu(), cache)
        print(f"Embedded {len(store.store)} docs in {time.time() - started:.1f}s "
              f"-> {cache.name}")
    return store


def query_sets(cfg: ExperimentConfig) -> dict[str, list[str]]:
    official = [q.query for q in load_queries(n=cfg.n_queries, seed=cfg.query_seed)]
    taken = set(official)
    pool = load_queries(n=10**9, seed=DEV_SEED)      # every query, shuffled
    dev = [q.query for q in pool if q.query not in taken][:cfg.n_queries]
    return {"official": official, "dev": dev}


def probe(store: PUPVectorStore, queries: list[str], max_retrieve: int) -> list[dict]:
    doc_emb = store.embeddings()
    rows = []
    for query in queries:
        scores = (torch.mm(store.encode(query), doc_emb.T)[0]).cpu().numpy()

        store.reseed_for(query)
        draw = (store._exp_mechanism_top_p_threshold if store.top_p is not None
                else store._exp_mechanism_top_k_threshold)
        threshold = float(draw(scores))
        store.reseed_for(query)
        retrieved = store.pup_retrieve(query)

        qualifying = int(np.sum(scores > threshold))
        assert len(retrieved) == min(qualifying, max_retrieve), \
            "mirrored threshold disagrees with pup_retrieve"

        ranked = np.argsort(scores)[::-1]
        got = {store.index[d] for d in retrieved}
        rows.append({
            "query": query,
            "top_score": float(scores[ranked[0]]),
            "threshold": threshold,
            "n_qualifying": qualifying,
            "n_retrieved": len(retrieved),
            "true_top10_hits": len(set(ranked[:TRUE_TOP].tolist()) & got),
            "top40_share": (len(set(ranked[:RELEVANT_TOP].tolist()) & got) / len(got)
                            if got else None),
            "p_zero": zero_document_probability(store, scores),
        })
    return rows


def summarise(rows: list[dict], max_retrieve: int) -> dict:
    nonzero = [r for r in rows if r["n_retrieved"] > 0]
    q = statistics.quantiles([r["n_qualifying"] for r in rows], n=4)
    return {
        "zero_observed": len(rows) - len(nonzero),
        "zero_expected": sum(r["p_zero"] for r in rows),
        "qualifying_q1_med_q3": [q[0], q[1], q[2]],
        "cap_binds": sum(r["n_qualifying"] > max_retrieve for r in rows) / len(rows),
        "mean_retrieved": statistics.mean(r["n_retrieved"] for r in rows),
        # nan, not an exception, when every query retrieved nothing
        "top10_hits": (statistics.mean(r["true_top10_hits"] for r in nonzero)
                       if nonzero else float("nan")),
        "top40_share": (statistics.mean(r["top40_share"] for r in nonzero)
                        if nonzero else float("nan")),
    }


def main() -> None:
    cfg = ExperimentConfig()
    store = build_store(cfg)
    sets = query_sets(cfg)

    results: dict = {}
    for set_name, queries in sets.items():
        print(f"\n== {set_name} ({len(queries)} queries) ==")
        print("      setting  zero obs/exp   qualifying Q1/med/Q3   cap binds  "
              "mean k  top10 hits  top40 share")
        results[set_name] = {}
        settings = ([(f"p={v}", v, None, cfg.max_score) for v in TOP_P_GRID]
                    + [(f"k={v}", None, v, cfg.max_score) for v in TOP_K_GRID]
                    + [(f"k={k},ms={m}", None, k, m)
                       for m in MAX_SCORE_GRID for k in MAX_SCORE_K])
        for label, top_p, top_k, max_score in settings:
            store.top_p, store.top_k, store.max_score = top_p, top_k, max_score
            rows = probe(store, queries, cfg.max_retrieve)
            s = summarise(rows, cfg.max_retrieve)
            results[set_name][label] = {"summary": s, "per_item": rows}
            q1, med, q3 = s["qualifying_q1_med_q3"]
            print(f"{label:>13}  {s['zero_observed']:3d} / {s['zero_expected']:5.1f}   "
                  f"{q1:6.1f} {med:6.1f} {q3:6.1f}      {s['cap_binds']:5.0%}    "
                  f"{s['mean_retrieved']:5.1f}   {s['top10_hits']:5.2f}       "
                  f"{s['top40_share']:5.1%}", flush=True)

    # --- reproduction check against the recorded baseline ---------------------
    with open(paths.results_dir() / BASELINE_RECORD, encoding="utf-8") as f:
        recorded = {r["query"] for r in json.load(f)["per_item"]}
    rows = results["official"][f"p={cfg.retrieval_top_p}"]["per_item"]
    ours_zero = {r["query"] for r in rows if r["n_retrieved"] == 0}
    recorded_zero = {r["query"] for r in rows} - recorded
    reproduced = ours_zero == recorded_zero
    print(f"\nreproduction at top_p={cfg.retrieval_top_p}: zero-document queries "
          f"probe {len(ours_zero)}, record {len(recorded_zero)}, identical: {reproduced}")

    out = paths.results_dir() / "retrieval_probe_sweep.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"params": cfg.to_dict(), "top_p_alpha": store.top_p_alpha,
                   "top_p_grid": TOP_P_GRID, "top_k_grid": TOP_K_GRID,
                   "max_score_grid": MAX_SCORE_GRID, "max_score_k": MAX_SCORE_K, "dev_seed": DEV_SEED,
                   "reproduces_baseline_zero_set": reproduced, "results": results},
                  f, ensure_ascii=False, indent=1)
    print(f"saved -> {out.name}")


if __name__ == "__main__":
    main()
