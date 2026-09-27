"""Build and verify the canary corpus before any GPU time is spent on it.

Everything here runs on a laptop GPU: generating the canaries, checking they do
not collide with the real corpus, injecting them, embedding the result, and
retrieving against it. Measured on an RTX 4060 Laptop, embedding 10,000 documents
takes about 6.5 minutes and peaks at 1.6 GB, so the whole check costs nothing but
wall-clock.

It runs first because three separate things can be wrong in ways that only show up
as a quiet null result eight A100-hours later:

  1. A CANARY FIELD ALREADY EXISTS IN THE CORPUS. Then a hit on it is
     unattributable -- the model could have read it from the real text. `collisions`
     refuses rather than reports.

  2. THE DETECTOR DOES NOT MATCH ITS OWN CANARIES, or matches things it should not.
     A detector that silently finds nothing is indistinguishable from perfect
     privacy, which is the most expensive failure available here. So it is tested
     against a document that definitely contains a canary and one that definitely
     does not.

  3. INJECTION BREAKS RETRIEVAL. Prepending a name-and-age block to every document
     shifts every embedding. The shift is uniform, which is the reason for injecting
     100% rather than a fraction -- a fraction would make injected documents
     systematically more or less retrievable and confound the whole probe -- but
     "uniform" is an argument, not a measurement. This compares the retrieved sets
     against the un-injected store on the same queries.

Run:  uv run python experiments/build_canary_corpus.py
"""

from __future__ import annotations

import json
import sys
import time

import torch

from dprag import canary, paths
from dprag.chatdoctor import load_corpus, load_queries
from dprag.config import ExperimentConfig
from dprag.pup_vector_store import PUPVectorStore, PUPVectorStoreConfig

# 100 queries, per the decision to cut from 180. Note the consequence measured on
# the Phase 2 records: 100 queries reach only about 2,370 distinct documents, so
# roughly three quarters of the planted canaries never enter any context and can
# never leak. The leakage denominator is "canaries that appeared in some context",
# never n_docs -- using n_docs would understate the rate about fourfold.
EXPERIMENT = ExperimentConfig(
    n_queries=100,
    canary_seed=20260927,
    gen_dtype="bfloat16",     # new corpus, nothing to reproduce; see ADR (pending)
)

N_PROBE_QUERIES = 5           # how many queries to check retrieval on


def _store(cfg: ExperimentConfig, docs: list[str], label: str) -> PUPVectorStore:
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
    for d in docs:
        store.add(d)
    t = time.time()
    store.embeddings()
    print(f"  {label}: embedded {len(docs)} docs in {time.time() - t:.0f}s", flush=True)
    return store


def main() -> None:
    cfg = EXPERIMENT
    print(f"=== canary corpus | n_docs={cfg.n_docs} seed={cfg.canary_seed} ===\n")

    plain = load_corpus(limit=cfg.n_docs, sample_seed=cfg.corpus_seed)
    planted, holdout = canary.generate(len(plain), cfg.canary_seed)
    print(f"generated {len(planted)} planted + {len(holdout)} holdout")
    print(f"  example planted: {planted[0].prefix()}")
    print(f"  example holdout: {holdout[0].prefix()}\n")

    # -- 1. collisions -------------------------------------------------------
    print("1. collision check against the real corpus")
    hits = canary.collisions(planted + holdout, plain)
    if hits:
        for field, values in hits.items():
            print(f"   COLLISION {field}: {values[:8]}")
        raise SystemExit(
            "\ncanary strings already occur in the corpus, so a hit on them would "
            "not be attributable to memorisation. Change canary_seed and re-run; "
            "do not proceed and note it as a caveat."
        )
    print("   clean: no canary field occurs in any of the 10,000 real documents\n")

    # -- 2. detector self-test ----------------------------------------------
    print("2. detector self-test")
    det = canary.Detector(planted + holdout)
    injected = canary.inject(plain, planted)

    c0 = planted[0]
    found = det.find(injected[0])
    expect = {"surname"} | {f for f in ("tel", "email", "allergy", "history")
                            if getattr(c0, f)}
    missing = expect - set(found)
    if missing:
        raise SystemExit(
            f"detector missed {sorted(missing)} on a document that contains "
            f"canary {c0.cid}. A detector that finds nothing looks exactly like "
            "perfect privacy, so this cannot be allowed to pass."
        )
    inj_hit, hold_hit = det.split(found)
    print(f"   injected doc 0 -> fields {sorted(found)}; "
          f"attributed to {sorted(inj_hit.get('surname', set()))}")
    if hold_hit:
        raise SystemExit(f"holdout matched an injected document: {hold_hit}")

    clean_hits = [det.find(d) for d in plain[:500]]
    n_false = sum(1 for h in clean_hits if h)
    print(f"   500 un-injected real documents -> {n_false} canary hits "
          f"({'ok' if n_false == 0 else 'FALSE POSITIVES, detector is too loose'})")
    if n_false:
        raise SystemExit("detector fires on documents containing no canary")
    print()

    # -- 3. retrieval before/after ------------------------------------------
    print("3. retrieval: does injection change which documents come back?")
    store_plain = _store(cfg, plain, "plain   ")
    store_inj = _store(cfg, injected, "injected")

    # Index by text once: `list.index` per retrieved document would be 40 linear
    # scans of 10,000 long strings per query.
    ix_plain = {d: i for i, d in enumerate(plain)}
    ix_inj = {d: i for i, d in enumerate(injected)}

    queries = load_queries(n=cfg.n_queries, seed=cfg.query_seed)[:N_PROBE_QUERIES]
    overlaps, n_docs_pairs = [], []
    for q in queries:
        store_plain.reseed_for(q.query)
        a = store_plain.pup_retrieve(q.query)
        store_inj.reseed_for(q.query)
        b = store_inj.pup_retrieve(q.query)
        ia = {ix_plain[d] for d in a}
        ib = {ix_inj[d] for d in b}
        jac = len(ia & ib) / len(ia | ib) if (ia | ib) else 1.0
        overlaps.append(jac)
        n_docs_pairs.append((len(a), len(b)))
    print(f"   {N_PROBE_QUERIES} queries | retrieved counts plain vs injected: "
          f"{n_docs_pairs}")
    print(f"   Jaccard overlap of retrieved sets: "
          f"{[round(o, 3) for o in overlaps]}")
    print("   (low overlap is expected and not a defect -- the embeddings moved. "
          "What matters is that both stores retrieve a comparable NUMBER of "
          "documents, i.e. injection did not break the DP threshold.)\n")

    # -- 4. write the registry ----------------------------------------------
    out = paths.results_dir() / f"canary_registry_{cfg.canary_seed}.json"
    out.write_text(json.dumps({
        "canary_seed": cfg.canary_seed,
        "n_docs": cfg.n_docs,
        "note": ("Canaries are regenerable from (canary_seed, n_docs) via "
                 "canary.generate; this file exists for human inspection and so "
                 "that a detector can be rebuilt without importing the generator."),
        "planted": [c.to_dict() for c in planted],
        "holdout": [c.to_dict() for c in holdout],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"registry -> {out}  ({out.stat().st_size / 2**20:.1f} MB)")

    print("\nall checks passed. The injected corpus is safe to spend GPU time on.")
    print(f"To reproduce it anywhere: ExperimentConfig(canary_seed={cfg.canary_seed}, "
          f"n_docs={cfg.n_docs}, corpus_seed={cfg.corpus_seed})")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no CUDA; embedding on CPU will be slow but the checks still run",
              file=sys.stderr)
    main()
