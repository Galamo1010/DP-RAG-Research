"""Smoke test: original DP-RAG (local two-layer DP build) on ChatDoctor data.

Corpus : N_DOCS HealthCareMagic doctor replies -> DP retrieval (exponential mechanism)
Queries: N_QUERIES iCliniq patient questions
Generation: local DPModel (token-level DP; k+1 parallel streams, exp mechanism)

Both privacy layers are kept, so the reported epsilon is the real end-to-end
budget (retrieval PLD composed with generation PLD).

Requires a CUDA GPU.

Run:  ./.venv/Scripts/python.exe test_smoke_dprag.py
"""

import time

from dprag.dp_rag_engine import DPRAGEngine
from dprag.pup_vector_store import PUPVectorStoreConfig
from dprag.dp_model import DPGenerationConfig
from dprag.chatdoctor import load_corpus, load_queries
from dprag.config import ExperimentConfig
from dprag import run_record

# A smoke test, so fewer queries than the 200 an ExperimentConfig defaults to.
EXPERIMENT = ExperimentConfig(n_queries=20)
DP_ON = True   # False -> non-DP baseline (no clipping/noise) to check if garbled text is DP or a bug

def main():
    exp = EXPERIMENT
    mode = "DP ON (two-layer)" if DP_ON else "DP OFF (non-private baseline)"
    print(f"=== {mode} | model={exp.gen_model} ===")
    print(f"Loading {exp.n_docs} corpus docs (random sample) + {exp.n_queries} queries ...")
    corpus = load_corpus(limit=exp.n_docs, sample_seed=exp.corpus_seed)
    queries = load_queries(n=exp.n_queries, seed=exp.query_seed)

    engine = DPRAGEngine(
        pup_vector_store_config=PUPVectorStoreConfig(
            model_id=exp.embed_model,
            top_p=exp.retrieval_top_p,
            epsilon=exp.eps_retrieval,
            max_retrieve=exp.max_retrieve,
            batch_size=exp.embed_batch_size,
            differential_pivacy=DP_ON,
        ),
        model_id=exp.gen_model,
        dp_generation_config=DPGenerationConfig(
            temperature=exp.temperature,
            max_new_tokens=exp.max_new_tokens,
            alpha=exp.alpha,
            omega=exp.omega,
            epsilon=exp.gen_epsilon,
            differential_pivacy=DP_ON,
        ),
    )

    print("Building vector store (embedding, please wait) ...")
    t0 = time.time()
    for doc in corpus:
        engine.add(doc)
    # Force embedding now so the timing below is generation-only.
    engine.pup_vector_store.embeddings()
    print(f"Embedded {len(corpus)} docs in {time.time() - t0:.1f}s")

    eps_total = engine.privacy_loss_distribution.get_epsilon_for_delta(exp.delta)
    if DP_ON:
        print(f"End-to-end epsilon (retrieval + generation, delta={exp.delta}): "
              f"{eps_total:.4f}\n")
    else:
        print(f"[DP OFF] nominal budget would be {eps_total:.4f}, but NO noise is applied this run\n")

    results = []
    for i, q in enumerate(queries):
        t = time.time()
        retrieved = engine.pup_retrieve(q.query)
        answer = engine.dp_chat(q.query)
        dt = time.time() - t
        print(f"[{i+1:2}/{exp.n_queries}] retrieved={len(retrieved):2}  {dt:5.1f}s")
        print(f"   Q: {q.query[:100].replace(chr(10),' ')}")
        print(f"   A: {answer[:160].replace(chr(10),' ')}")
        print(f"   ref: {q.reference[:120].replace(chr(10),' ')}\n")
        results.append({
            "query": q.query,
            "reference": q.reference,
            "answer": answer,
            "n_retrieved": len(retrieved),
        })

    tag = "dp" if DP_ON else "nodp"
    out = run_record.write(
        "stage1_smoke", exp,
        metrics={
            "dp_on": DP_ON,
            "eps_total": eps_total,
            "note": ("two-layer DP: DP retrieval + local DP generation (token-level)"
                     if DP_ON else
                     "DP OFF baseline: no clipping/noise; eps_total is nominal only"),
        },
        per_item=results,
        filename=f"stage1_smoke_{exp.n_docs}x{exp.n_queries}_{tag}",
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
