"""The control that decides whether the pole evidence means anything.

Claim 4 leans partly on this: strategy A's answer is more similar to the RAG pole
than plain DPRAG's is (ROUGE-L 0.326 against 0.253), which is offered as evidence
that A skips the *right* positions rather than merely skipping a lot of them.

There is a competing explanation that the existing data cannot rule out. A emits
87% of its positions straight from the NoRAG greedy argmax, with no DP noise on
them. Both poles are greedy generations too. Two low-noise texts resemble each
other for reasons that have nothing to do with documents, so "A is closer to the
RAG pole" may be measuring cleanliness, not grounding.

B_k50_t0.5 argues against that -- it triggers at 95.8%, so it is cleaner than A,
yet sits at 0.275 against A's 0.326 -- but it is not a clean control: it is also
copying the NoRAG pole nearly verbatim (0.718 similarity, twice anything else in
the table), so it moves two things at once.

WHAT THIS RUN CHANGES, AND WHAT IT HOLDS FIXED
----------------------------------------------
Everything is held fixed except relevance. DP retrieval runs exactly as before and
decides how many documents this query gets; then those documents are replaced with
the same number drawn uniformly from the corpus, excluding the ones retrieved. The
substitutes are still de-duplicated HealthCareMagic doctor replies, so register,
length and vocabulary are unchanged. Only the connection to the question is gone.

Similarity is then measured against the REAL poles -- `stage3_poles_180q.json`,
already on disk, generated from the documents retrieval actually found. Those are
the fixed target. `experiments/stage3_score.py` globs `stage3_*` and keys poles by
model and query, so it picks this record up and grounds it against them with no
change; that is why the file is named the way it is.

HOW TO READ THE RESULT
----------------------
  similarity to the real RAG pole DROPS   -> the 0.326 came from the documents,
                                             and the cleanliness explanation is
                                             dead. Claim 4's pole evidence stands.
  similarity HOLDS                        -> the 0.326 had nothing to do with the
                                             documents. The pole evidence has to
                                             be withdrawn from claim 4.

The second outcome is a bad result and it is the reason to run this. Without it
the honest answer to "isn't that just because A's text is cleaner?" is "I do not
know".

THE CONFOUND MOVES THE WRONG WAY, WHICH HELPS
---------------------------------------------
Irrelevant documents make the RAG instance agree with the NoRAG instance more
often, so A's trigger rate should RISE here and its output should get *cleaner*
than the real-document run. If cleanliness were driving the similarity, this run
should therefore score HIGHER, not lower. A drop cannot be explained by it. The
trigger rates are printed for exactly this reason.

TWO CONFIGURATIONS, NOT ONE
---------------------------
A is what is under examination. baseline is the control on the control: it pays
epsilon at every position and still reads the documents, so its similarity should
fall too. If neither moves, the measurement is simply insensitive to documents and
the whole pole table needs rethinking rather than just claim 4.

B is not run. Nothing in claim 4 rests on it.

    uv run python experiments/stage3_5_counterfactual.py
"""

import hashlib
import random
import statistics as st

from dprag import paths, run_record, sweep
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import strategy_a

EPSILON = 40
EXPERIMENT = ExperimentConfig(n_queries=200)

CONFIGS = {
    "baseline": sweep.NEVER_AGREE,
    "A": strategy_a,
}

# What this is the counterfactual OF. Without them there is nothing to compare a
# drop against, and the run measures an unfamiliar system in isolation.
COMPARISON_RECORDS = [f"stage3_2_main_{name}_eps{EPSILON}" for name in CONFIGS]
POLES_RECORD = "stage3_poles_180q"


def check_prerequisites() -> None:
    missing = [n for n in COMPARISON_RECORDS + [POLES_RECORD]
               if not (paths.results_dir() / f"{n}.json").exists()]
    if missing:
        raise SystemExit(
            f"missing prerequisite records: {missing}\n"
            "This run is a counterfactual: its numbers mean nothing except as a "
            "drop from the real-document run, measured against the real poles. "
            "Run experiments/stage3_2_main.py and experiments/stage3_poles.py first."
        )


def make_substituter(corpus: list[str], seed: int):
    """Replace the retrieved documents with the same number of unrelated ones.

    Uniform over the corpus rather than "the least similar n". Least-similar would
    make the contrast louder, but it selects documents for being far from the
    query in embedding space, which on this corpus means picking up the ones that
    are short, malformed or off-register -- so a drop in similarity could be a
    drop in document quality rather than in relevance. Uniform sampling keeps the
    substitutes ordinary and changes exactly one thing.

    Seeded from the question text rather than from a counter, for the same reason
    `PUPVectorStore.reseed_for` is: the substitute set for a query must not depend
    on how many queries ran before it, or a resumed run gets different documents
    from an uninterrupted one. sha256 rather than hash(), whose salt changes per
    process.
    """
    def substitute(retrieved: list[str], question: str) -> list[str]:
        digest = hashlib.sha256(f"{seed}:{question}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        excluded = set(retrieved)
        pool = [d for d in corpus if d not in excluded]
        if len(pool) < len(retrieved):
            raise SystemExit(
                f"corpus has {len(pool)} documents outside the retrieved set but "
                f"{len(retrieved)} are needed; the corpus is too small to draw a "
                "disjoint substitute set."
            )
        return rng.sample(pool, len(retrieved))
    return substitute


def report(paths_written: dict[str, object]) -> None:
    """The checks that are worth seeing before the scoring run, not after."""
    print()
    print("=== did the substitution actually remove relevance? ===")
    print("Mean similarity of the documents the generation saw, against the mean")
    print("similarity of the documents retrieval found for the same queries. If")
    print("these are close, the substitution did not do its job and nothing below")
    print("is interpretable.")
    print()
    print(f"{'config':>10} {'substituted':>13} {'retrieved':>11} {'trigger':>9} "
          f"{'real-doc trigger':>18}")
    print("-" * 66)

    for name, path in paths_written.items():
        record = run_record.load(path)
        sub = [s for r in record.per_item for _, s in r["docs"]]
        real = [s for r in record.per_item for _, s in r.get("retrieved_docs", [])]
        if not real:
            # routed_sweep skips a run whose output file already exists, so this
            # is what a record written before the substitution hook looks like.
            # Saying so beats a ZeroDivisionError three hours into a session.
            print(f"{name:>10}   no retrieved_docs: this record predates the "
                  "substitution hook and is not a counterfactual run")
            continue
        trigger = st.mean(r["by_strategy"][name]["trigger_rate"]
                          for r in record.per_item)

        original = run_record.load(
            paths.results_dir() / f"stage3_2_main_{name}_eps{EPSILON}.json")
        was = st.mean(r["by_strategy"][name]["trigger_rate"]
                      for r in original.per_item)

        print(f"{name:>10} {st.mean(sub):>13.4f} {st.mean(real):>11.4f} "
              f"{trigger:>9.1%} {was:>18.1%}")

    print()
    print("A's trigger rate is EXPECTED to rise: documents that say nothing about")
    print("the question cannot pull the RAG instance away from the NoRAG one. That")
    print("makes this run's output cleaner than the real-document run, so if the")
    print("cleanliness explanation were true the pole similarity should go UP here.")
    print()
    print("Next:")
    print("  experiments/stage3_score.py -- grounds these against the REAL poles.")
    print("  Read the 'to RAG pole' column against stage3_2_main's. A drop kills")
    print("  the cleanliness explanation; no drop withdraws claim 4's pole evidence.")
    print()
    print("NOTE these records are named stage3_* so stage3_score.py finds them, so")
    print("they will also appear in stage4_1_budget.py and")
    print("stage2_5_safety_from_routed.py, whose globs are the same. They carry")
    print("counterfactual=true in their metrics. A trigger rate or epsilon saving")
    print("from this run is NOT a result about the method -- it is a result about")
    print("a system fed documents no deployment would ever retrieve.")


def main():
    check_prerequisites()
    exp = EXPERIMENT.with_(gen_epsilon=float(EPSILON))
    bench = Bench.build(exp)
    substitute = make_substituter(bench.corpus, exp.seed)

    print(f"=== counterfactual: unrelated documents | model={exp.gen_model} "
          f"| eps={EPSILON} ===")
    print(f"{len(CONFIGS)} configurations x {exp.n_queries} queries "
          f"| max_retrieve={exp.max_retrieve} | dtype={exp.gen_dtype}")
    print(f"configurations: {list(CONFIGS)}")
    print(f"compared against: {COMPARISON_RECORDS}")
    print(f"grounded against: {POLES_RECORD} (the REAL poles)")
    print("checkpointed per query; completed runs are skipped\n", flush=True)

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=float(EPSILON), delta=exp.delta,
    )

    written = {}
    for name, strategy in CONFIGS.items():
        print(f"--- {name} ---", flush=True)
        written[name] = sweep.routed_sweep(
            bench, exp, {name: strategy}, dp_cfg,
            name="stage3_5_counterfactual",
            filename=f"stage3_5_counterfactual_{name}_eps{EPSILON}",
            substitute_documents=substitute,
            metrics={
                "phase": "3.5 (counterfactual control)",
                "epsilon": EPSILON,
                "configuration": name,
                "counterfactual": True,
                "compared_against": f"stage3_2_main_{name}_eps{EPSILON}",
                "grounded_against": POLES_RECORD,
                "purpose": (
                    "Retrieved documents replaced by the same number drawn "
                    "uniformly from the corpus, everything else held fixed, to "
                    "test whether claim 4's pole similarity comes from the "
                    "documents or from strategy A's output simply being cleaner. "
                    "Trigger rates and epsilon savings from this run describe a "
                    "system fed irrelevant evidence and are not results about "
                    "the method."
                ),
            },
        )

    report(written)


if __name__ == "__main__":
    main()
