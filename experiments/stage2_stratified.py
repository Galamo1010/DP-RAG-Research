"""Stage 2.4 -- trigger rate split by how much a query depends on its documents.

The proposal (5.2, sourced from 1.2) stratifies queries by the Stage 1 consistency
rate: a query whose NoRAG and DPRAG tokens usually agree is LOW dependency (the
documents rarely change the answer), one where they disagree is HIGH dependency.
The pre-filter should fire often on the first group and rarely on the second. If
it does not, the premise is wrong and that is worth knowing before Stage 3.

The stratification is READ from the Stage 1 result rather than recomputed -- 1.2
already measured it over 200 queries, and re-deriving it here would be both a
waste of GPU time and a second, possibly disagreeing, source for the same number.

Beyond the two strata, this reports the CORRELATION between a query's Stage 1
consistency and its Stage 2 trigger rate. That is the sharper question: not
"do the buckets differ" (bucket boundaries are arbitrary) but "does the Stage 1
measurement actually predict where the pre-filter fires".

Strategy B is included only as a two-point sanity check. The full (k, τ) grid of
3 x 4 belongs to Stage 3.2, which has its own cost-control plan; running it here
would duplicate that work.

Zero-document queries are excluded from the rates and reported separately: with no
documents the RAG instance collapses onto NoRAG, so every strategy fires trivially.

Requires a CUDA GPU. Run:  uv run python experiments/stage2_stratified.py
"""

import statistics as st
import time

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dual_instance import make_generation_config, run_dual_instance
from dprag.strategies import make_strategy_b, strategy_a
from dprag import paths, run_record

# Stage 1.2's output, used for the stratification.
STAGE1_RESULT = "stage1_consistency_10000x200"

EXPERIMENT = ExperimentConfig()

STRATEGIES = {
    "A": strategy_a,
    "B_k20_t0.7": make_strategy_b(20, 0.7),   # sanity check only; grid is Stage 3.2
    "B_k20_t0.9": make_strategy_b(20, 0.9),
}


def load_dependency_index() -> dict[str, float]:
    """query text -> Stage 1 greedy consistency rate.

    Keyed by query text rather than position: both runs sample with the same
    query_seed, but matching on text means a changed sample fails loudly (a query
    simply will not be found) instead of silently pairing the wrong rows.
    """
    path = paths.RESULTS_DIR / f"{STAGE1_RESULT}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 1.2 result not found: {path}\n"
            "Run experiments/stage1_consistency.py first -- this experiment "
            "stratifies by its consistency rates.\n"
            "(The copy in results/archive/ predates the RunRecord schema and the "
            "seeding fix; see results/archive/README.md.)"
        )
    record = run_record.load(path)
    return {row["query"]: row["consistency_greedy"] for row in record.per_item}


def pearson(xs: list[float], ys: list[float]) -> float:
    """Correlation, or 0.0 when either series is constant."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else 0.0


def main():
    exp = EXPERIMENT
    print(f"=== Stage 2.4 stratified trigger rate | model={exp.gen_model} | "
          f"{exp.n_queries} queries ===")

    dependency = load_dependency_index()
    print(f"Loaded Stage 1 consistency for {len(dependency)} queries")

    bench = Bench.build(exp)
    cfg = make_generation_config(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens
    )

    rows = []
    missing = 0
    for i, q in enumerate(bench.queries()):
        if q.query not in dependency:
            missing += 1
            continue
        started = time.time()
        docs = bench.retrieve(q.query)
        res = run_dual_instance(bench.dp_model, docs, q.query, cfg, STRATEGIES)
        rows.append({
            "query": q.query,
            "consistency_greedy": dependency[q.query],
            "n_documents": res.n_documents,
            "n_steps": res.n_steps,
            "trigger": {name: res.trigger_rate(name) for name in STRATEGIES},
            "mean_jaccard": {name: res.mean_score(name) for name in STRATEGIES},
        })
        print(f"[{i+1:3}/{exp.n_queries}] k={res.n_documents:2} "
              f"consist={dependency[q.query]:.2f} "
              f"trigA={res.trigger_rate('A'):.2f}  {time.time()-started:4.1f}s")

    if missing:
        print(f"\nWARNING: {missing} queries had no Stage 1 entry and were skipped. "
              "The two runs should share query_seed and n_queries.")

    # ---- stratify on the doc>0 subset ------------------------------------
    scored = [r for r in rows if r["n_documents"] > 0]
    n_zero = len(rows) - len(scored)
    if not scored:
        raise RuntimeError("every query retrieved 0 documents; nothing to stratify")

    cutoff = st.median(r["consistency_greedy"] for r in scored)
    # High consistency == the documents rarely change the token == LOW dependency.
    low_dep = [r for r in scored if r["consistency_greedy"] >= cutoff]
    high_dep = [r for r in scored if r["consistency_greedy"] < cutoff]

    def stratum(name, group):
        return {
            "stratum": name,
            "n_queries": len(group),
            "mean_consistency": st.mean(r["consistency_greedy"] for r in group) if group else 0.0,
            "mean_trigger": {
                s: (st.mean(r["trigger"][s] for r in group) if group else 0.0)
                for s in STRATEGIES
            },
        }

    strata = [stratum("low_dependency", low_dep), stratum("high_dependency", high_dep)]
    correlation = {
        s: pearson(
            [r["consistency_greedy"] for r in scored],
            [r["trigger"][s] for r in scored],
        )
        for s in STRATEGIES
    }
    overall = {s: st.mean(r["trigger"][s] for r in scored) for s in STRATEGIES}

    # ---- report -----------------------------------------------------------
    print(f"\n=== SUMMARY (doc>0 subset: {len(scored)} queries; "
          f"{n_zero} zero-doc queries excluded) ===")
    print(f"median consistency used as the split: {cutoff:.3f}\n")
    header = f"{'stratum':>16} | {'n':>3} | {'consist':>7} | " + " | ".join(f"{s:>10}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))
    for row in strata:
        print(f"{row['stratum']:>16} | {row['n_queries']:>3} | {row['mean_consistency']:>7.3f} | "
              + " | ".join(f"{row['mean_trigger'][s]:>10.3f}" for s in STRATEGIES))
    print(f"{'overall':>16} | {len(scored):>3} | "
          f"{st.mean(r['consistency_greedy'] for r in scored):>7.3f} | "
          + " | ".join(f"{overall[s]:>10.3f}" for s in STRATEGIES))

    print("\ncorrelation(Stage 1 consistency, Stage 2 trigger rate):")
    for s, r in correlation.items():
        print(f"  {s:>12}: {r:+.3f}")
    print("\nA positive correlation supports the premise: queries whose answers do "
          "not depend on the documents are the ones the pre-filter can skip.")

    out = run_record.write(
        "stage2_stratified", exp,
        metrics={
            "split_cutoff": cutoff,
            "n_scored": len(scored),
            "n_zero_doc": n_zero,
            "n_missing_from_stage1": missing,
            "stage1_source": STAGE1_RESULT,
            "strata": strata,
            "overall_trigger": overall,
            "correlation_consistency_vs_trigger": correlation,
        },
        per_item=rows,
        filename=f"stage2_stratified_{exp.n_queries}q",
    )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
