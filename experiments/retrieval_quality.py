"""Does better retrieval give plain DPRAG better answers?

`retrieval_probe.py` measured retrieval alone. At the recorded setting (top_p=0.02,
max_score=0.8) about 87 documents clear the DP threshold and 40 are drawn from them
at random, so only ~54% of the Baseline's voters are among a query's true top-40.
Switching to top_k and lowering max_score raises that past 90% while cutting
zero-document queries from ~10% to ~4%. This measures whether that reaches the
answers -- ROUGE-L and BERTScore against the iCliniq reference -- which retrieval
metrics alone cannot say.

Configurations (chosen by the author, 2026-10-04):

    C0  top_p=0.02                 the recorded setting, the control
    C1  top_k=40                   top_k alone
    C2  top_k=40, max_score=0.7    best on retrieval; C1 vs C2 isolates max_score
    C3  top_k=30, max_score=0.7    fewer, more relevant voters than C2
    C4  top_k=50, max_score=0.7    more, less relevant voters than C2

Baseline only (NEVER_AGREE), eps=40, Llama. The pre-filter is deliberately absent:
retrieval is shared by every strategy, so it is tuned on the one that has no
pre-filter to confound it, and the strategies are compared afterwards on equal
evidence.

QUERIES ARE THE DEV SET, NOT THE OFFICIAL 200
---------------------------------------------
Eighty iCliniq questions disjoint from the official sample (seed 43, same
construction as retrieval_probe.py). Choosing a retrieval setting on the queries
the thesis reports would bias every number that follows toward that setting.

Comparison is PAIRED on the queries where all five configurations retrieved at
least one document (72 of 80 by the probe). A zero-document query never reaches
generation (sweep.routed_sweep excludes it), so per-config zero rates are reported
next to the quality table rather than folded into it.

All five share one Bench, so the corpus is embedded once; only the store's
top_p/top_k/max_score change between runs, and each run's record carries the
values it ran with. Retrieval is seeded per query (ADR 0010), so the dev queries
draw the same documents here as they did in the probe.

    uv run python -u experiments/retrieval_quality.py            # pod: generate
    uv run python experiments/retrieval_quality.py --score       # local: score
    uv run python experiments/retrieval_quality.py --dry-run     # local: retrieval only

Each run checkpoints per query and completed runs are skipped, so the script is
safe to interrupt and re-run.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from dprag import paths, run_record, sweep
from dprag.chatdoctor import Query, load_queries
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.quality import bertscore_f1, rouge_l_f1

EPSILON = 40
N_DEV = 80
DEV_SEED = 43

# label -> (top_p, top_k, max_score)
CONFIGS = {
    "C0_p0.02":      (0.02, None, 0.8),
    "C1_k40":        (None, 40, 0.8),
    "C2_k40_ms0.7":  (None, 40, 0.7),
    "C3_k30_ms0.7":  (None, 30, 0.7),
    "C4_k50_ms0.7":  (None, 50, 0.7),
}

EXPERIMENT = ExperimentConfig(n_queries=N_DEV, gen_epsilon=float(EPSILON))


def dev_queries() -> list[Query]:
    official = {q.query for q in load_queries(n=200, seed=EXPERIMENT.query_seed)}
    pool = load_queries(n=10**9, seed=DEV_SEED)      # every query, shuffled
    return [q for q in pool if q.query not in official][:N_DEV]


def config_for(label: str) -> ExperimentConfig:
    top_p, top_k, max_score = CONFIGS[label]
    return EXPERIMENT.with_(retrieval_top_p=top_p, retrieval_top_k=top_k,
                            max_score=max_score)


def apply(store, cfg: ExperimentConfig) -> None:
    """Point the shared store at one configuration's retrieval settings."""
    store.top_p = cfg.retrieval_top_p
    store.top_k = cfg.retrieval_top_k
    store.max_score = cfg.max_score


def filename(label: str) -> str:
    return f"retrieval_quality_{label}_eps{EPSILON}"


# --------------------------------------------------------------------------
# Generation (pod)
# --------------------------------------------------------------------------

def generate() -> None:
    from dprag.bench import Bench

    queries = dev_queries()
    print(f"=== retrieval quality | model={EXPERIMENT.gen_model} | eps={EPSILON} | "
          f"{len(queries)} dev queries | {len(CONFIGS)} configurations ===",
          flush=True)
    bench = Bench.build(EXPERIMENT)
    dp_cfg = DPGenerationConfig(
        temperature=EXPERIMENT.temperature, max_new_tokens=EXPERIMENT.max_new_tokens,
        alpha=EXPERIMENT.alpha, omega=EXPERIMENT.omega, epsilon=float(EPSILON),
        delta=EXPERIMENT.delta)

    began = time.time()
    for label in CONFIGS:
        cfg = config_for(label)
        apply(bench.engine.pup_vector_store, cfg)
        print(f"\n--- {label}: top_p={cfg.retrieval_top_p} top_k={cfg.retrieval_top_k} "
              f"max_score={cfg.max_score} ---", flush=True)
        sweep.routed_sweep(
            bench, cfg, {"baseline": sweep.NEVER_AGREE}, dp_cfg,
            name="retrieval_quality",
            filename=filename(label),
            queries=queries,
            metrics={"configuration": label, "epsilon": EPSILON,
                     "query_set": f"dev (seed {DEV_SEED}, disjoint from official)"},
        )
        print(f"  elapsed {(time.time() - began) / 3600:.2f} h", flush=True)

    print(f"\nTotal {(time.time() - began) / 3600:.2f} h")
    print("Next, locally: uv run python experiments/retrieval_quality.py --score")


def dry_run() -> None:
    """Retrieval only, on the pod's exact code path, without the LLM.

    Builds the store the same way Bench does and reports per-configuration zero
    rates and document counts, so the numbers can be checked against
    retrieval_probe.py before any GPU hour is spent.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retrieval_probe import build_store

    store = build_store(EXPERIMENT)
    queries = dev_queries()
    print(f"{len(queries)} dev queries")
    has_docs = {}
    for label in CONFIGS:
        apply(store, config_for(label))
        counts = []
        for q in queries:
            store.reseed_for(q.query)
            counts.append(len(store.pup_retrieve(q.query)))
        has_docs[label] = [c > 0 for c in counts]
        print(f"  {label:14s} zero-doc {counts.count(0):3d}   "
              f"mean k {statistics.mean(counts):5.1f}")
    common = sum(all(v[i] for v in has_docs.values()) for i in range(len(queries)))
    print(f"queries with documents under all five: {common} / {len(queries)}")


# --------------------------------------------------------------------------
# Scoring (local)
# --------------------------------------------------------------------------

def ci95(values: list[float]) -> tuple[float, float]:
    m = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** 0.5
    return m - half, m + half


def score() -> None:
    refs = {q.query: q.reference for q in dev_queries()}
    texts: dict[str, dict[str, str]] = {}
    zeros: dict[str, int] = {}
    for label in CONFIGS:
        path = paths.results_dir() / f"{filename(label)}.json"
        if not path.exists():
            raise SystemExit(f"{path.name} missing; run generation first")
        record = run_record.load(path)
        zeros[label] = record.metric("n_zero_document_queries")
        texts[label] = {row["query"]: row["by_strategy"]["baseline"]["text"]
                        for row in record.per_item}

    common = [q for q in refs if all(q in texts[label] for label in CONFIGS)]
    print(f"paired on {len(common)} of {len(refs)} dev queries "
          f"(all five retrieved at least one document)\n")

    hyps = [texts[label][q] for label in CONFIGS for q in common]
    bert = bertscore_f1(hyps, [refs[q] for _ in CONFIGS for q in common],
                        model=EXPERIMENT.bertscore_model,
                        rescale=EXPERIMENT.bertscore_rescale, batch_size=16)
    per: dict[str, dict[str, list[float]]] = {}
    for n, label in enumerate(CONFIGS):
        chunk = bert[n * len(common):(n + 1) * len(common)]
        per[label] = {
            "rouge": [rouge_l_f1(texts[label][q], refs[q]) for q in common],
            "bert": list(chunk),
        }

    control = next(iter(CONFIGS))
    print(f"{'config':14s} {'zero':>4s}  {'ROUGE-L':>8s} {'diff vs C0 [95% CI]':>26s}"
          f"  {'BERTScore':>9s} {'diff vs C0 [95% CI]':>26s}")
    for label in CONFIGS:
        cells = [f"{label:14s} {zeros[label]:4d}"]
        for metric in ("rouge", "bert"):
            mean = statistics.mean(per[label][metric])
            if label == control:
                cells.append(f"{mean:8.4f} {'':>26s}")
                continue
            diff = [a - b for a, b in zip(per[label][metric], per[control][metric])]
            lo, hi = ci95(diff)
            cells.append(f"{mean:8.4f} {statistics.mean(diff):+8.4f} "
                         f"[{lo:+.4f}, {hi:+.4f}]")
        print("  ".join(cells))
    print("\nA difference whose CI excludes 0 is distinguishable from C0 at this n;"
          "\none whose CI spans 0 is not shown to differ -- which is not the same as"
          "\nshown to be equal.")


if __name__ == "__main__":
    if "--score" in sys.argv:
        score()
    elif "--dry-run" in sys.argv:
        dry_run()
    else:
        generate()
