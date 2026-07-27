"""Quick GPU smoke test for prefilter_engine (Stage 2.2).

Small corpus + short generation so it runs in well under a minute once the model
is cached. Confirms the dual-instance path works end-to-end: retrieve -> build
NoRAG/RAG instances -> generate one shared stream -> per-step strategy decisions.

This is a wiring check, NOT a measurement run (that's stage2_temperature_sweep).

Run:  uv run python smoke_prefilter_engine.py
"""

from dprag.bench import Bench
from dprag.dual_instance import run_dual_instance, make_generation_config
from dprag.strategies import strategy_a, make_strategy_b
from dprag.config import ExperimentConfig

# Small and short on purpose: this checks the wiring, not the measurement.
EXPERIMENT = ExperimentConfig(n_docs=500, n_queries=2, max_new_tokens=32)


def main():
    exp = EXPERIMENT
    print(f"=== dual-instance engine smoke | model={exp.gen_model} ===")
    bench = Bench.build(exp)

    strategies = {
        "A": strategy_a,
        "B_k10_t0.7": make_strategy_b(10, 0.7),
        "B_k20_t0.9": make_strategy_b(20, 0.9),
    }
    cfg = make_generation_config(temperature=exp.temperature, max_new_tokens=exp.max_new_tokens)

    for q in bench.queries():
        docs = bench.retrieve(q.query)
        res = run_dual_instance(bench.dp_model, docs, q.query, cfg, strategies)
        print(f"\nQ: {q.query[:80].replace(chr(10),' ')}")
        print(f"   docs={res.n_documents}  steps={res.n_steps}")
        for name in strategies:
            print(f"   {name:12} trigger={res.trigger_rate(name):.2f}  "
                  f"mean_jaccard={res.mean_score(name):.2f}")
        print(f"   NoRAG-driven text: {res.text[:120].replace(chr(10),' ')}")

    print("\nOK: dual-instance engine ran end-to-end.")


if __name__ == "__main__":
    main()
