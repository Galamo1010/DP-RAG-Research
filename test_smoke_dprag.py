"""Smoke test: original DP-RAG (local two-layer DP build) on ChatDoctor data.

Corpus : N_DOCS HealthCareMagic doctor replies -> DP retrieval (exponential mechanism)
Queries: N_QUERIES iCliniq patient questions
Generation: local DPModel (token-level DP; k+1 parallel streams, exp mechanism)

Both privacy layers are kept, so the reported epsilon is the real end-to-end
budget (retrieval PLD composed with generation PLD).

Requires a CUDA GPU.

Run:  ./.venv/Scripts/python.exe test_smoke_dprag.py
"""

import json
import os
import time

from dp_rag_engine import DPRAGEngine
from pup_vector_store import PUPVectorStoreConfig
from dp_model import DPGenerationConfig
from chatdoctor_data import load_corpus, load_queries
import experiment_params as P

N_DOCS = 10000
N_QUERIES = 20
MAX_RETRIEVE = 10  # cap retrieved docs to keep the k+1 batch small/cheap for this smoke test
CORPUS_SEED = 7    # random-sample the corpus for better topic coverage
# GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # cached, ungated; swap for P.MODELS[0] on a bigger GPU
GEN_MODEL = "meta-llama/llama-3.1-8b-instruct"  # cached, ungated; swap for P.MODELS[0] on a bigger GPU
GEN_EPSILON = 10.0                        # generation budget per full answer

def main():
    print(f"Loading {N_DOCS} corpus docs (random sample) + {N_QUERIES} queries ...")
    corpus = load_corpus(limit=N_DOCS, sample_seed=CORPUS_SEED)
    queries = load_queries(n=N_QUERIES, seed=P.QUERY_SEED)

    engine = DPRAGEngine(
        pup_vector_store_config=PUPVectorStoreConfig(
            model_id=P.EMBED_MODEL,
            top_p=P.RETRIEVAL_TOP_P,
            epsilon=P.EPS_RETRIEVAL,
            max_retrieve=MAX_RETRIEVE,
            batch_size=P.EMBED_BATCH_SIZE,
        ),
        model_id=GEN_MODEL,
        dp_generation_config=DPGenerationConfig(
            temperature=P.TEMPERATURE,
            max_new_tokens=P.MAX_NEW_TOKENS,
            alpha=P.ALPHA,
            omega=P.OMEGA,
            epsilon=GEN_EPSILON,
        ),
    )

    print("Building vector store (embedding, please wait) ...")
    t0 = time.time()
    for doc in corpus:
        engine.add(doc)
    # Force embedding now so the timing below is generation-only.
    engine.pup_vector_store.embeddings()
    print(f"Embedded {len(corpus)} docs in {time.time() - t0:.1f}s")

    eps_total = engine.privacy_loss_distribution.get_epsilon_for_delta(P.DELTA)
    print(f"End-to-end epsilon (retrieval + generation, delta={P.DELTA}): {eps_total:.4f}\n")

    results = []
    for i, q in enumerate(queries):
        t = time.time()
        retrieved = engine.pup_retrieve(q.query)
        answer = engine.dp_chat(q.query)
        dt = time.time() - t
        print(f"[{i+1:2}/{N_QUERIES}] retrieved={len(retrieved):2}  {dt:5.1f}s")
        print(f"   Q: {q.query[:100].replace(chr(10),' ')}")
        print(f"   A: {answer[:160].replace(chr(10),' ')}")
        print(f"   ref: {q.reference[:120].replace(chr(10),' ')}\n")
        results.append({
            "query": q.query,
            "reference": q.reference,
            "answer": answer,
            "n_retrieved": len(retrieved),
        })

    os.makedirs("results", exist_ok=True)
    out = f"results/smoke_{N_DOCS}x{N_QUERIES}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_docs": N_DOCS, "n_queries": N_QUERIES, "max_retrieve": MAX_RETRIEVE,
                "eps_total": eps_total, "model": GEN_MODEL,
                "note": "two-layer DP: DP retrieval + local DP generation (token-level)",
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
