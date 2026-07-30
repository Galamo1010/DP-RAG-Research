"""Stage 2.5 -- does a loose strategy B skip epsilon on clinically-loaded tokens?

The proposal's 2.2 step limit names the risk: a high k or a low tau raises the
trigger rate, but "錯誤的不使用ε輸出可能導致醫療安全風險" at positions carrying drug
names or doses. When the pre-filter fires, the emitted token comes from the NoRAG
instance -- the model's prior -- rather than from the retrieved documents. On a
dose, that is the difference between a number the documents support and one the
model invented.

This run compares configurations along the strict-to-loose axis and asks, for
each: of the positions where it skipped epsilon, how many sat inside a drug name
or a dose?

The number to watch is `medical_skip_rate`: of all clinically-loaded positions,
the share the strategy skipped. A strategy that fires on 80% of ordinary tokens
but 5% of drug-name tokens is behaving well. One that fires equally on both is
treating "metformin" like "the".

TWO LIMITS, BOTH DELIBERATE
---------------------------
1. Detection is a regex heuristic (dprag.medical_flags), not scispaCy. The
   proposal's Medical Entity Retention Rate is a Stage 5 instrument. Counts here
   are a signal to inspect by hand, not a measured rate.
2. Flagging a skipped position does not make it an error. The NoRAG token may be
   perfectly correct. What this finds is where a wrong skip WOULD matter, which is
   where the manual inspection at the end should look.

Requires a CUDA GPU. Run:  uv run python experiments/stage2_safety_check.py
"""

import statistics as st
import time

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dual_instance import make_generation_config, run_dual_instance
from dprag.medical_flags import flag_medical_tokens
from dprag.strategies import make_strategy_b, strategy_a
from dprag import run_record

# A small batch: this is a qualitative check, and every position gets inspected.
EXPERIMENT = ExperimentConfig(n_queries=30)

# Ordered strict -> loose, so the safety cost of loosening is readable down the table.
STRATEGIES = {
    "A": strategy_a,                          # strictest: top-1 must match
    "B_k20_t0.9": make_strategy_b(20, 0.9),   # 19 of 20 must match
    "B_k20_t0.7": make_strategy_b(20, 0.7),   # 17 of 20
    "B_k50_t0.5": make_strategy_b(50, 0.5),   # loosest: 34 of 50 -- the risk case
}

EXAMPLES_TO_KEEP = 25   # flagged skips saved verbatim for manual inspection


def main():
    exp = EXPERIMENT
    print(f"=== Stage 2.5 safety check | model={exp.gen_model} | "
          f"{exp.n_queries} queries ===")
    print("Heuristic regex detection; scispaCy Medical Entity Retention is Stage 5.\n")

    bench = Bench.build(exp)
    tokenizer = bench.dp_model.tokenizer
    cfg = make_generation_config(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens
    )

    # counters[strategy] = [skips on medical tokens, skips on ordinary tokens]
    counters = {name: {"skipped_medical": 0, "skipped_plain": 0} for name in STRATEGIES}
    totals = {"medical": 0, "plain": 0}
    examples: list[dict] = []
    per_query = []

    for i, q in enumerate(bench.queries()):
        started = time.time()
        docs = bench.retrieve(q.query)
        res = run_dual_instance(bench.dp_model, docs, q.query, cfg, STRATEGIES)
        if res.n_documents == 0:
            # No documents means the RAG instance IS the NoRAG instance, so every
            # strategy fires trivially; counting it would flatter every config.
            print(f"[{i+1:3}/{exp.n_queries}] 0 docs -- skipped")
            continue

        # Flag against what the pre-filter would emit (the NoRAG argmax stream),
        # not against what DPRAG sampled: those are the tokens at risk.
        flags, text, matches = flag_medical_tokens(tokenizer, res.norag_argmax)

        n_medical = sum(flags)
        totals["medical"] += n_medical
        totals["plain"] += len(flags) - n_medical

        for name, decisions in res.decisions.items():
            for pos, decision in enumerate(decisions):
                if pos >= len(flags) or not decision.consistent:
                    continue
                if flags[pos]:
                    counters[name]["skipped_medical"] += 1
                    if name == "B_k50_t0.5" and len(examples) < EXAMPLES_TO_KEEP:
                        lo, hi = max(0, pos - 8), min(len(res.norag_argmax), pos + 8)
                        examples.append({
                            "query": q.query[:120],
                            "position": pos,
                            "token": tokenizer.decode([res.norag_argmax[pos]]),
                            "context": tokenizer.decode(res.norag_argmax[lo:hi]),
                            "jaccard": decision.score,
                        })
                else:
                    counters[name]["skipped_plain"] += 1

        per_query.append({
            "query": q.query,
            "n_documents": res.n_documents,
            "n_steps": res.n_steps,
            "n_medical_positions": n_medical,
            "medical_matches": [m for _, _, m in matches],
            "trigger": {name: res.trigger_rate(name) for name in STRATEGIES},
        })
        print(f"[{i+1:3}/{exp.n_queries}] k={res.n_documents:2} "
              f"medical={n_medical:2}/{len(flags):3}  {time.time()-started:4.1f}s")

    if not per_query:
        raise RuntimeError("every query retrieved 0 documents; nothing to check")

    # ---- rates -------------------------------------------------------------
    summary = {}
    for name, c in counters.items():
        medical_rate = c["skipped_medical"] / totals["medical"] if totals["medical"] else 0.0
        plain_rate = c["skipped_plain"] / totals["plain"] if totals["plain"] else 0.0
        summary[name] = {
            "skipped_medical": c["skipped_medical"],
            "skipped_plain": c["skipped_plain"],
            "medical_skip_rate": medical_rate,
            "plain_skip_rate": plain_rate,
            # <1 means the strategy is more cautious on clinical tokens than on
            # ordinary ones, which is the behaviour we want.
            "relative_risk": medical_rate / plain_rate if plain_rate else 0.0,
        }

    print(f"\n=== SUMMARY ({len(per_query)} queries with documents; "
          f"{totals['medical']} clinical / {totals['plain']} ordinary positions) ===")
    header = (f"{'strategy':>12} | {'med skip':>9} | {'plain skip':>10} | {'ratio':>6}")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:>12} | {s['medical_skip_rate']:>9.3f} | "
              f"{s['plain_skip_rate']:>10.3f} | {s['relative_risk']:>6.2f}")
    print("\nratio < 1: more cautious on drug names and doses than on ordinary tokens.")
    print("ratio ~ 1: treats them alike -- the case the proposal warns about.")

    if examples:
        print(f"\n=== {len(examples)} loosest-config skips on clinical tokens "
              "(inspect these by hand) ===")
        for ex in examples[:10]:
            print(f"  token={ex['token']!r:16} jaccard={ex['jaccard']:.2f}  "
                  f"...{ex['context'].strip()[:90]}...")

    out = run_record.write(
        "stage2_safety_check", exp,
        metrics={
            "detection": "regex heuristic (dprag.medical_flags); scispaCy is Stage 5",
            "n_queries_scored": len(per_query),
            "total_medical_positions": totals["medical"],
            "total_plain_positions": totals["plain"],
            "mean_medical_per_query": st.mean(r["n_medical_positions"] for r in per_query),
            "by_strategy": summary,
            "flagged_examples": examples,
        },
        per_item=per_query,
        filename=f"stage2_safety_check_{exp.n_queries}q",
    )
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
