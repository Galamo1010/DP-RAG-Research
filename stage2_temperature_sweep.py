"""Stage 2.3 -- temperature sweep to fix the temperature for strategy A / B.

The proposal (2.1 step limit) asks us to try several temperatures and pick the
one with the best quality vs trigger-rate trade-off; that temperature is then
FIXED for every strategy A / B experiment.

Design decisions (see the Stage 2 plan):

  * Retrieve ONCE per query and reuse the same document set across all
    temperatures. pup_retrieve samples its threshold with the exponential
    mechanism, so re-retrieving per temperature would confound "temperature"
    with "different documents came back".
  * 50 queries (not 200) -- this only selects a hyper-parameter; the full
    200-query measurement happens after the temperature is fixed.
  * Zero-document queries are reported SEPARATELY. When DP retrieval returns
    nothing the RAG instance collapses onto NoRAG, so every strategy trivially
    fires (trigger = 1.0); averaging those in would inflate the trigger rate
    (same issue as Stage 1's 13/200 zero-doc queries).

Quality metric: a pure-Python ROUGE-L F1 (LCS-based) against the iCliniq
reference. It needs no extra dependency or model download and is enough to RANK
temperatures. The proposal's full BERTScore / ROUGE-L toolchain stays in Stage 5.

Two things this sweep will likely show, both honest findings to report:
  * Trigger rate is fairly temperature-stable, because the strategy decision is
    temperature-invariant at a fixed context (see prefilter_strategies); it only
    moves through the sampled trajectory.
  * Quality (of the NoRAG-driven text) responds to temperature. Note this is the
    NoRAG trajectory's quality, a proxy for selection; the routed system's
    quality is measured in Stage 3.

Requires a CUDA GPU. Run:  uv run python stage2_temperature_sweep.py
"""

import json
import os
import statistics as st
import time

from dp_rag_engine import DPRAGEngine
from pup_vector_store import PUPVectorStoreConfig
from dp_model import DPGenerationConfig
from chatdoctor_data import load_corpus, load_queries
from prefilter_engine import run_dual_instance, make_generation_config
from prefilter_strategies import strategy_a, make_strategy_b
import experiment_params as P

TEMPERATURES = [0.1, 0.3, 0.5, 0.7]
N_DOCS = 10000
N_QUERIES = 50
MAX_RETRIEVE = 10
CORPUS_SEED = 7
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = P.MAX_NEW_TOKENS

STRATEGIES = {
    "A": strategy_a,
    "B_k10_t0.7": make_strategy_b(10, 0.7),
    "B_k20_t0.9": make_strategy_b(20, 0.9),
}


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """LCS-based ROUGE-L F1 on whitespace tokens (lightweight proxy)."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def main():
    print(f"=== Stage 2.3 temperature sweep | model={GEN_MODEL} | "
          f"{N_QUERIES} queries | T={TEMPERATURES} ===")
    engine = DPRAGEngine(
        pup_vector_store_config=PUPVectorStoreConfig(
            model_id=P.EMBED_MODEL, top_p=P.RETRIEVAL_TOP_P, epsilon=P.EPS_RETRIEVAL,
            max_retrieve=MAX_RETRIEVE, batch_size=P.EMBED_BATCH_SIZE,
        ),
        model_id=GEN_MODEL,
        dp_generation_config=DPGenerationConfig(epsilon=10.0),  # unused; DPRAGEngine needs one
    )

    print(f"Embedding {N_DOCS} corpus docs ...")
    t0 = time.time()
    for doc in load_corpus(limit=N_DOCS, sample_seed=CORPUS_SEED):
        engine.add(doc)
    engine.pup_vector_store.embeddings()
    print(f"Embedded in {time.time() - t0:.1f}s")

    # Retrieve ONCE per query; the same document set is reused at every temperature.
    queries = load_queries(n=N_QUERIES, seed=P.QUERY_SEED)
    fixed = []
    for q in queries:
        docs = engine.pup_retrieve(q.query)
        fixed.append({"query": q.query, "reference": q.reference, "docs": docs})
    n_zero = sum(1 for f in fixed if not f["docs"])
    print(f"Retrieved once per query; {n_zero}/{N_QUERIES} queries got 0 documents "
          f"(reported separately).\n")

    per_temperature = []
    per_query = []
    for temp in TEMPERATURES:
        cfg = make_generation_config(temperature=temp, max_new_tokens=MAX_NEW_TOKENS)
        t_start = time.time()
        triggers = {name: [] for name in STRATEGIES}   # non-zero-doc only
        rouges = []                                     # all queries
        for f in fixed:
            res = run_dual_instance(engine.dp_model, f["docs"], f["query"], cfg, STRATEGIES)
            rl = rouge_l_f1(res.text, f["reference"])
            rouges.append(rl)
            has_docs = res.n_documents > 0
            for name in STRATEGIES:
                if has_docs:
                    triggers[name].append(res.trigger_rate(name))
            per_query.append({
                "temperature": temp, "query": f["query"], "n_documents": res.n_documents,
                "n_steps": res.n_steps, "rouge_l": rl,
                "trigger": {name: res.trigger_rate(name) for name in STRATEGIES},
            })
        secs = time.time() - t_start
        row = {
            "temperature": temp,
            "n_queries": N_QUERIES,
            "n_zero_doc": n_zero,
            "mean_trigger": {name: (st.mean(v) if v else 0.0) for name, v in triggers.items()},
            "mean_rouge_l": st.mean(rouges),
            "seconds": secs,
        }
        per_temperature.append(row)
        ta = row["mean_trigger"]["A"]
        print(f"T={temp}: trigger_A={ta:.3f}  "
              + "  ".join(f"{n}={row['mean_trigger'][n]:.3f}" for n in STRATEGIES if n != "A")
              + f"  rougeL={row['mean_rouge_l']:.3f}  ({secs:.0f}s)")

    # ---- summary table ----
    print("\n=== SUMMARY (trigger on doc>0 subset; rougeL = NoRAG-driven quality proxy) ===")
    print(f"{'T':>4} | {'trig_A':>7} | {'trig_B10':>8} | {'trig_B20':>8} | {'rougeL':>7}")
    print("-" * 46)
    for row in per_temperature:
        mt = row["mean_trigger"]
        print(f"{row['temperature']:>4} | {mt['A']:>7.3f} | {mt['B_k10_t0.7']:>8.3f} | "
              f"{mt['B_k20_t0.9']:>8.3f} | {row['mean_rouge_l']:>7.3f}")
    print("\nPick the temperature balancing quality (rougeL) against trigger rate; "
          "it becomes the fixed temperature for all strategy A/B runs.")

    os.makedirs("results", exist_ok=True)
    out = f"results/stage2_temperature_sweep_{N_QUERIES}q.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": GEN_MODEL, "n_docs": N_DOCS, "n_queries": N_QUERIES,
                "max_retrieve": MAX_RETRIEVE, "max_new_tokens": MAX_NEW_TOKENS,
                "temperatures": TEMPERATURES, "n_zero_doc": n_zero,
                "quality_metric": "rouge_l_f1 (lightweight proxy; full BERTScore in Stage 5)",
            },
            "per_temperature": per_temperature,
            "per_query": per_query,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
