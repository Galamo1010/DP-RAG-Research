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

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag import run_record

# A smoke test, so fewer queries than the 200 an ExperimentConfig defaults to.
EXPERIMENT = ExperimentConfig(n_queries=20)
DP_ON = True   # False -> non-DP baseline (no clipping/noise) to check if garbled text is DP or a bug

def main():
    exp = EXPERIMENT
    mode = "DP ON (two-layer)" if DP_ON else "DP OFF (non-private baseline)"
    print(f"=== {mode} | model={exp.gen_model} ===")
    bench = Bench.build(exp, differential_privacy=DP_ON)
    queries = bench.queries()

    eps_total = bench.epsilon()
    if DP_ON:
        print(f"End-to-end epsilon (retrieval + generation, delta={exp.delta}): "
              f"{eps_total:.4f}\n")
    else:
        print(f"[DP OFF] nominal budget would be {eps_total:.4f}, but NO noise is applied this run\n")

    results = []
    for i, q in enumerate(queries):
        t = time.time()
        retrieved = bench.retrieve(q.query)
        answer = bench.engine.dp_chat(q.query)
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
