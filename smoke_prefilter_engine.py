"""Quick GPU smoke test for prefilter_engine (Stage 2.2).

Small corpus + short generation so it runs in well under a minute once the model
is cached. Confirms the dual-instance path works end-to-end: retrieve -> build
NoRAG/RAG instances -> generate one shared stream -> per-step strategy decisions.

This is a wiring check, NOT a measurement run (that's stage2_temperature_sweep).

Run:  uv run python smoke_prefilter_engine.py
"""

from dp_rag_engine import DPRAGEngine
from pup_vector_store import PUPVectorStoreConfig
from dp_model import DPGenerationConfig
from chatdoctor_data import load_corpus, load_queries
from prefilter_engine import run_dual_instance, make_generation_config
from prefilter_strategies import strategy_a, make_strategy_b
import experiment_params as P

N_DOCS = 500          # small: embedding a huge corpus is not what we're testing
MAX_RETRIEVE = 10
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 32   # short: this is a wiring check


def main():
    print(f"=== prefilter_engine smoke | model={GEN_MODEL} ===")
    engine = DPRAGEngine(
        pup_vector_store_config=PUPVectorStoreConfig(
            model_id=P.EMBED_MODEL, top_p=P.RETRIEVAL_TOP_P, epsilon=P.EPS_RETRIEVAL,
            max_retrieve=MAX_RETRIEVE, batch_size=P.EMBED_BATCH_SIZE,
        ),
        model_id=GEN_MODEL,
        dp_generation_config=DPGenerationConfig(epsilon=10.0),  # unused here; DPRAGEngine needs one
    )

    print(f"Embedding {N_DOCS} corpus docs ...")
    for doc in load_corpus(limit=N_DOCS, sample_seed=7):
        engine.add(doc)
    engine.pup_vector_store.embeddings()

    strategies = {
        "A": strategy_a,
        "B_k10_t0.7": make_strategy_b(10, 0.7),
        "B_k20_t0.9": make_strategy_b(20, 0.9),
    }
    cfg = make_generation_config(temperature=P.TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)

    for q in load_queries(n=2, seed=P.QUERY_SEED):
        docs = engine.pup_retrieve(q.query)
        res = run_dual_instance(engine.dp_model, docs, q.query, cfg, strategies)
        print(f"\nQ: {q.query[:80].replace(chr(10),' ')}")
        print(f"   docs={res.n_documents}  steps={res.n_steps}")
        for name in strategies:
            print(f"   {name:12} trigger={res.trigger_rate(name):.2f}  "
                  f"mean_jaccard={res.mean_score(name):.2f}")
        print(f"   NoRAG-driven text: {res.text[:120].replace(chr(10),' ')}")

    print("\nOK: dual-instance engine ran end-to-end.")


if __name__ == "__main__":
    main()
