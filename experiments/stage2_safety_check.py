"""Stage 2.5 -- does a loose strategy B skip epsilon on clinically-loaded tokens?

The proposal's 2.2 step limit names the risk: a high k or a low tau raises the
trigger rate, but "錯誤的不使用ε輸出可能導致醫療安全風險" at positions carrying drug
names, doses or conditions. When the pre-filter fires, the emitted token comes
from the NoRAG instance -- the model's prior -- rather than from the retrieved
documents. On a dose, that is the difference between a number the documents
support and one the model invented.

WHAT CHANGED, AND WHY THE FIRST ATTEMPT WAS WRONG
-------------------------------------------------
The first version searched `norag_argmax_text`: the per-step NoRAG argmax
concatenated. Because Stage 2 sampled its trajectory, each of those argmaxes was
computed over a *different* prefix from the token printed before it, so the
concatenation is a string no system ever emits. It read "irregular irregular
irregular" and the regex duly matched artifacts. Across 200 such strings the
whole corpus of matches was one word.

This version routes for real and inspects `RoutedResult.emitted` -- the answer the
system actually produced -- classifying each clinical position by whether the
router took the free path there. That is only possible now the Stage 3.1 router
exists, which is why ADR 0003 orders the work 3.1 -> 2.5 -> 3.2.

WHAT THE NUMBERS MEAN
---------------------
`relative_risk` is the medical skip rate over the ordinary skip rate. Below 1 the
strategy is more cautious on clinical content than on ordinary words, which is the
behaviour we want; at 1 it is treating "metformin" like "the".

The ratio is normalised inside a single run, which is what makes it comparable
across strategies even though each strategy walks its own trajectory and therefore
scores against its own text. The absolute rates are not comparable across
strategies for that same reason; `n` is reported beside every ratio so a figure
resting on three positions cannot be read as a measurement.

Word-like and pattern-like results are reported apart. A vocabulary can confirm
that "hepatitis" is a word and "Totosis" is not, but nothing can confirm that
"197 mg" is a real dose, and on DP-degraded text stray digits and stray units do
land next to each other. Averaging the two would hide that half is unverifiable.

TWO LIMITS, BOTH DELIBERATE
---------------------------
1. Detection is a regex heuristic plus a corpus vocabulary (dprag.medical_flags),
   not scispaCy. The proposal's Medical Entity Retention Rate is a Stage 5
   instrument. Counts here are a signal to inspect by hand, not a measured rate.
2. Flagging a skipped position does not make it an error. The NoRAG token may be
   perfectly correct. What this finds is where a wrong skip WOULD matter, which is
   where the manual inspection at the end should look.

Requires a CUDA GPU. Run:  uv run python experiments/stage2_safety_check.py
"""

import statistics as st
import time

import torch

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.medical_flags import (
    PATTERN_KIND,
    WORD_KIND,
    build_corpus_vocabulary,
    flag_medical_tokens,
)
from dprag.router import Router
from dprag import trace
from dprag.strategies import make_strategy_b, strategy_a
from dprag import run_record

EXPERIMENT = ExperimentConfig(n_queries=200)

# Both ends of the budget grid. Noise differs roughly threefold between them, so
# a safety ratio that moves with epsilon is being shaped by DP noise, while one
# that holds is a property of the strategy. Neither reading is available from a
# single budget.
EPSILON_GRID = [10.0, 40.0]

# Ordered strict -> loose, so the safety cost of loosening is readable down the
# table. Overlap requirements from strategies.min_overlap_for_tau.
STRATEGIES = {
    "A": strategy_a,                          # strictest: top-1 must match
    "B_k20_t0.9": make_strategy_b(20, 0.9),   # 19 of 20 must match
    "B_k20_t0.7": make_strategy_b(20, 0.7),   # 17 of 20
    "B_k50_t0.5": make_strategy_b(50, 0.5),   # loosest: 34 of 50 -- the risk case
}

LOOSEST = "B_k50_t0.5"
EXAMPLES_TO_KEEP = 25   # flagged skips saved verbatim for manual inspection

KINDS = (WORD_KIND, PATTERN_KIND)


def _blank_counters():
    """skipped/total positions, split by what kind of content sits there."""
    return {
        name: {k: {"skipped": 0, "total": 0} for k in (*KINDS, "plain")}
        for name in STRATEGIES
    }


def _rates(counter: dict) -> dict:
    """Skip rates per kind, plus the ratio against ordinary positions."""
    plain = counter["plain"]
    plain_rate = plain["skipped"] / plain["total"] if plain["total"] else 0.0

    def rate(bucket):
        return bucket["skipped"] / bucket["total"] if bucket["total"] else 0.0

    combined = {
        "skipped": sum(counter[k]["skipped"] for k in KINDS),
        "total": sum(counter[k]["total"] for k in KINDS),
    }
    out = {"plain_skip_rate": plain_rate, "n_plain": plain["total"]}
    for label, bucket in (
        (WORD_KIND, counter[WORD_KIND]),
        (PATTERN_KIND, counter[PATTERN_KIND]),
        ("combined", combined),
    ):
        r = rate(bucket)
        out[f"{label}_skip_rate"] = r
        out[f"n_{label}"] = bucket["total"]
        # <1 means more caution on clinical tokens than on ordinary ones.
        out[f"relative_risk_{label}"] = r / plain_rate if plain_rate else 0.0
    return out


def run_one_budget(bench, exp, eps: float, fixed: list[dict], vocabulary) -> dict:
    """Route every query under every strategy at one epsilon, and score safety."""
    cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=eps, delta=exp.delta,
    )
    tokenizer = bench.dp_model.tokenizer
    counters = _blank_counters()
    examples: list[dict] = []
    per_query: list[dict] = []

    print(f"\n=== eps_gen={eps}  (clipping={cfg.token_epsilon() * exp.temperature / 2:.4f}) ===")
    for i, item in enumerate(fixed):
        started = time.time()
        row = {
            "query": item["query"], "n_documents": len(item["docs"]),
            "docs": item["docs_trace"],
            "by_strategy": {},
        }
        for name, strategy in STRATEGIES.items():
            # Re-seed per generation so a strategy's trajectory does not depend on
            # how much randomness the strategies before it consumed.
            torch.manual_seed(exp.seed)
            began = time.time()
            res = Router(bench.dp_model, strategy, cfg).generate(
                item["docs"], item["query"]
            )
            elapsed = time.time() - began
            marks, text, spans = flag_medical_tokens(
                tokenizer, res.emitted, vocabulary
            )
            for position, mark in enumerate(marks):
                kind = mark.kind if mark else None
                bucket = counters[name][kind or "plain"]
                bucket["total"] += 1
                if res.is_free(position):
                    bucket["skipped"] += 1
                    if (name == LOOSEST and mark is not None
                            and len(examples) < EXAMPLES_TO_KEEP):
                        lo, hi = max(0, position - 8), min(len(res.emitted), position + 8)
                        examples.append({
                            "query": item["query"][:120],
                            "position": position,
                            "kind": mark.kind,
                            "is_first": mark.is_first,
                            "token": tokenizer.decode([res.emitted[position]]),
                            "context": tokenizer.decode(res.emitted[lo:hi]),
                            "jaccard": res.decisions[position].score,
                        })
            record = trace.strategy_trace(res, marks, elapsed)
            trace.check(record)      # a violation means the router is broken
            record["n_clinical_positions"] = sum(1 for m in marks if m is not None)
            record["matches"] = [s.text for s in spans]
            row["by_strategy"][name] = record
        per_query.append(row)
        clinical = sum(r["n_clinical_positions"] for r in row["by_strategy"].values())
        print(f"[{i+1:3}/{len(fixed)}] k={row['n_documents']:2} "
              f"clinical={clinical:3} across {len(STRATEGIES)} strategies "
              f"{time.time() - started:5.1f}s")

    summary = {name: _rates(counters[name]) for name in STRATEGIES}

    print(f"\n--- SUMMARY  eps={eps}  ({len(per_query)} queries with documents) ---")
    header = (f"{'strategy':>12} | {'word':>12} | {'pattern':>12} | "
              f"{'plain':>7} | {'ratio(w)':>8}")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:>12} | {s['word_skip_rate']:>6.3f} (n={s['n_word']:>3}) | "
              f"{s['pattern_skip_rate']:>6.3f} (n={s['n_pattern']:>3}) | "
              f"{s['plain_skip_rate']:>7.3f} | {s['relative_risk_word']:>8.2f}")
    print("ratio < 1: more cautious on clinical content than on ordinary tokens.")
    print("ratio ~ 1: treats them alike -- the case the proposal warns about.")
    print("n is the denominator: a ratio built on single digits is not a measurement.")

    out = run_record.write(
        "stage2_safety_check",
        exp.with_(gen_epsilon=eps),
        metrics={
            "detection": (
                "regex heuristic + corpus vocabulary (dprag.medical_flags); "
                "scispaCy Medical Entity Retention Rate is Stage 5"
            ),
            "measured_on": "router output (RoutedResult.emitted)",
            "vocabulary_size": len(vocabulary),
            "n_queries_scored": len(per_query),
            "by_strategy": summary,
            "flagged_examples": examples,
            "mean_clinical_per_query": {
                name: st.mean(r["by_strategy"][name]["n_clinical_positions"]
                              for r in per_query)
                for name in STRATEGIES
            },
        },
        per_item=per_query,
        filename=f"stage2_safety_check_eps{int(eps)}_{exp.n_queries}q",
    )
    print(f"Saved -> {out}")
    return summary


def main():
    exp = EXPERIMENT
    print(f"=== Stage 2.5 safety check | model={exp.gen_model} | "
          f"{exp.n_queries} queries x {len(STRATEGIES)} strategies x "
          f"{len(EPSILON_GRID)} budgets ===")

    print(f"Building corpus vocabulary (min_count={exp.vocab_min_count}) ...")
    vocabulary = build_corpus_vocabulary(exp.vocab_min_count)
    print(f"  {len(vocabulary):,} words")

    bench = Bench.build(exp)

    # Retrieve ONCE per query and reuse across every strategy and budget.
    # Retrieval is stochastic, so retrieving per configuration would hand each one
    # a different document set and confound "strategy" with "different evidence".
    store = bench.engine.pup_vector_store
    fixed = []
    for q in bench.queries():
        docs = bench.retrieve(q.query)
        # Indices and similarities, not text: the corpus is reproducible from
        # n_docs + corpus_seed, and without the scores "did retrieval find
        # anything relevant?" costs a full re-embedding to ask afterwards.
        fixed.append({"query": q.query, "docs": docs,
                      "docs_trace": trace.retrieval_trace(store, q.query, docs)})
    with_docs = [f for f in fixed if f["docs"]]
    print(f"Retrieved once per query; {len(fixed) - len(with_docs)}/{len(fixed)} "
          f"got 0 documents (excluded -- the RAG instance collapses onto NoRAG "
          f"there, so every strategy fires trivially).")
    if not with_docs:
        raise RuntimeError("every query retrieved 0 documents; nothing to check")

    for eps in EPSILON_GRID:
        run_one_budget(bench, exp, eps, with_docs, vocabulary)


if __name__ == "__main__":
    main()
