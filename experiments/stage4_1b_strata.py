"""Does the pre-filter behave differently on document-dependent queries?

The proposal's Stage 4.1 step limit asks for this and says why:

> ε節省量與觸發率正相關，若測試查詢集中大量包含高RAG依賴型問題（觸發率低），
> 實際節省效果可能低於實驗室估計值；需確保測試查詢集的依賴型分布與預期實際
> 應用場景一致，必要時進行分層報告

A saving averaged over a query set only transfers to deployment if the deployment
set has a similar mix of document-dependent and document-independent questions. So
the saving has to be reported per stratum, not only as one mean.

WHY THIS REPLACES `stage2_stratified.py`'s RESULT
-------------------------------------------------
`results/stage2_stratified_200q.json` (2026-07-30) answered this at
`max_retrieve=10`, which ADR 0008 made incomparable with everything since. It also
measured a **spectator**: the system paid epsilon at every position and merely
recorded what a pre-filter would have decided, so its trigger rates are not the
rates a routed system produces (`docs/notes/router-verification.md` measured the
gap: strategy A holds up, strategy B drops 8-14 points).

Nothing needs re-generating to fix that. Three sources now line up for the first
time -- same query sample, same seed, same `max_retrieve=40`:

    Stage 1.2   per-query `consistency_greedy`  -- the dependence measure
    Stage 3     per-query trigger and epsilon   -- from a really routed system
    results/scores/  per-query BERTScore        -- so quality can be stratified too

WHICH WAY ROUND THE STRATA ARE NAMED
------------------------------------
`low_dependency` means HIGH consistency: NoRAG's argmax already matches DPRAG's at
most positions, so the documents change little and the pre-filter has room to
skip. `high_dependency` is the opposite. The names follow Stage 2.4 so the two
measurements can be read against each other; the mean consistency is printed in
each row so the direction cannot be misread.

The split is at the median, computed over queries that actually retrieved
documents. A zero-document query has nothing to depend on, its consistency is
trivially high, and CONTEXT.md requires those to be reported separately rather
than averaged in -- including them would place them all in `low_dependency` and
flatter every configuration there.

EXPECT A WEAK SIGNAL, AND REPORT IT AS ONE
------------------------------------------
Stage 2.4 found strata differing by 1-3.5% in trigger rate with correlations of
0.14-0.27, because the consistency distribution is tight. Raising `max_retrieve`
widened it only a little -- sd 0.044 to 0.061, IQR 0.844-0.891 to 0.844-0.922 --
so a large stratification effect is not expected here either. The Stage 3 spec
says so directly: "If Stage 3's cross-eps results are similarly flat, that is a
finding to report plainly rather than a bug to hunt."

THE STRATA DIFFER IN DIFFICULTY, SO QUALITY IS COMPARED WITHIN THEM
------------------------------------------------------------------
Plain DPRAG does not score equally on the two strata -- at eps=10 it reaches
BERTScore 0.0687 on low_dependency and 0.0922 on high_dependency. Whatever causes
that, it is a property of the queries and not of any pre-filter, so reading a
configuration's stratum means against each other measures the queries.

The column that controls for it is `vs base`: the configuration's BERTScore minus
plain DPRAG's, computed inside the same (model, epsilon, stratum) cell. That is
the paired quantity -- same queries, same budget, same difficulty, only the
strategy differs.

    uv run python experiments/stage4_1b_strata.py
"""

import glob
import json
import math
import statistics as st

from dprag import paths, run_record

STAGE1_RECORD = "stage1_consistency_10000x200"
RECORD_GLOB = "stage3_2_main_*"
CACHE_DIR = "scores"

LOW, HIGH = "low_dependency", "high_dependency"
CONFIG_ORDER = ["baseline", "B_k20_t0.9", "B_k20_t0.7", "A", "B_k50_t0.5"]


def order_key(name: str) -> tuple[int, str]:
    return (CONFIG_ORDER.index(name) if name in CONFIG_ORDER else len(CONFIG_ORDER),
            name)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Correlation, or None when it is not defined.

    Reported next to the stratum means because they answer the question at
    different resolutions: the strata ask whether two halves differ, the
    correlation asks whether the relationship holds query by query. A large
    stratum gap with a near-zero correlation would mean the split is doing the
    work rather than the dependence.
    """
    n = len(xs)
    if n < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return None if den == 0 else num / den


def load_dependence() -> tuple[dict[str, float], float, int]:
    """query -> consistency_greedy, the median split point, and the zero-doc count.

    Zero-document queries are dropped from the split so they cannot define it,
    and counted so the report can say how many were set aside.
    """
    matches = glob.glob(str(paths.results_dir() / f"{STAGE1_RECORD}.json"))
    if not matches:
        raise SystemExit(
            f"missing {STAGE1_RECORD}.json -- the dependence measure comes from "
            "Stage 1.2. Splitting on a Stage 3 trigger rate instead would stratify "
            "the outcome by itself.")
    record = run_record.load(matches[0])
    if int(record.param("max_retrieve", 0)) != 40:
        print(f"!! {STAGE1_RECORD} ran at max_retrieve="
              f"{record.param('max_retrieve')}, not 40. ADR 0008 makes that")
        print("!! incomparable with the Stage 3 records; the split would be drawn")
        print("!! on a different retrieval regime from the runs being split.")
    with_docs = {r["query"]: r["consistency_greedy"]
                 for r in record.per_item if r.get("n_retrieved")}
    zero_doc = len(record.per_item) - len(with_docs)
    return with_docs, st.median(with_docs.values()), zero_doc


def load_bertscore(stem: str) -> dict[str, list]:
    """strategy -> per-query BERTScore against the reference, or {}.

    Read from the cache `stage3_score.py` writes rather than recomputed: the
    encoder is a 750M model, and the point of the cache is that a second reader
    does not pay for it again.
    """
    path = paths.results_dir() / CACHE_DIR / f"{stem}.json"
    if not path.exists():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {name: s.get("reference", []) for name, s in blob.get("scores", {}).items()}


def split_points(record, name, bert, dependence, cutoff):
    """{stratum: [per-query points]} for one (record, strategy).

    One definition of the split, used both to build the baseline reference and to
    build the rows compared against it. Two copies of this loop would be two
    chances for the strata to be drawn differently on the two sides of a
    subtraction, and the result would still look plausible.
    """
    buckets = {LOW: [], HIGH: []}
    scores = bert.get(name, [])
    for i, row in enumerate(record.per_item):
        sr = row["by_strategy"].get(name)
        if sr is None or row["query"] not in dependence:
            continue
        c = dependence[row["query"]]
        buckets[LOW if c >= cutoff else HIGH].append({
            "query": row["query"],
            "consistency": c,
            "trigger": sr["trigger_rate"],
            "saved": sr["epsilon_savings"],
            "bert": scores[i] if i < len(scores) else None,
        })
    return buckets


def main():
    dependence, cutoff, zero_doc = load_dependence()
    print(f"dependence source : {STAGE1_RECORD} (max_retrieve=40)")
    print(f"queries with documents : {len(dependence)}   "
          f"zero-document, set aside : {zero_doc}")
    print(f"median consistency_greedy (the split) : {cutoff:.4f}")
    print(f"  low_dependency  = consistency >= {cutoff:.4f} (documents change little)")
    print(f"  high_dependency = consistency <  {cutoff:.4f}")
    print()

    found = sorted(glob.glob(str(paths.results_dir() / f"{RECORD_GLOB}.json")))
    if not found:
        raise SystemExit("no stage3_2_main_* records found")

    # Baselines first, so every later row can be expressed against the plain
    # DPRAG run under identical conditions rather than against the other stratum.
    baselines = {}
    for path in found:
        record = run_record.load(path)
        if not record.metric("is_baseline"):
            continue
        bert = load_bertscore(record.path.stem)
        key0 = (record.param("gen_model", "unknown"),
                float(record.param("gen_epsilon", 0.0)))
        for name in record.metric("strategies", []):
            for stratum, points in split_points(record, name, bert, dependence,
                                                cutoff).items():
                # Per query, not the stratum mean: the comparison below is
                # paired -- the same question answered by two configurations --
                # and a difference of means throws away the pairing that makes a
                # 0.03 gap resolvable at this sample size.
                baselines[(*key0, stratum)] = {
                    p["query"]: p["bert"] for p in points if p["bert"] is not None}

    print(f"{'model':>22} | {'eps':>3} | {'config':>12} | {'stratum':>15} | {'n':>4} | "
          f"{'consistency':>11} | {'trigger':>8} | {'eps saved':>9} | {'BERTScore':>17} | "
          f"{'vs base (paired)':>18} | {'corr':>6}")
    print("-" * 167)

    for path in found:
        record = run_record.load(path)
        model = record.param("gen_model", "unknown").split("/")[-1]
        eps = float(record.param("gen_epsilon", 0.0))
        bert = load_bertscore(record.path.stem)

        for name in sorted(record.metric("strategies", []), key=order_key):
            buckets = split_points(record, name, bert, dependence, cutoff)
            all_consistency = [p["consistency"] for s in buckets.values() for p in s]
            all_trigger = [p["trigger"] for s in buckets.values() for p in s]

            corr = pearson(all_consistency, all_trigger)
            for stratum in (LOW, HIGH):
                pts = buckets[stratum]
                if not pts:
                    continue
                bs = [p["bert"] for p in pts if p["bert"] is not None]
                if len(bs) > 1:
                    bm = st.mean(bs)
                    bh = 1.96 * st.stdev(bs) / math.sqrt(len(bs))
                    bert_txt = f"{bm:.4f} +-{bh:.4f}"
                else:
                    bert_txt = "n/a"
                shown = f"{corr:+.3f}" if (stratum == LOW and corr is not None) else ""
                base = baselines.get((record.param("gen_model", "unknown"), eps,
                                      stratum), {})
                paired = [p["bert"] - base[p["query"]] for p in pts
                          if p["bert"] is not None and p["query"] in base]
                if record.metric("is_baseline") or len(paired) < 2:
                    delta = "-"
                else:
                    dm = st.mean(paired)
                    dh = 1.96 * st.stdev(paired) / math.sqrt(len(paired))
                    delta = f"{dm:+.4f}+-{dh:.4f}{'*' if abs(dm) > dh else ' '}"
                print(f"{model[:22]:>22} | {eps:>3.0f} | {name:>12} | {stratum:>15} | "
                      f"{len(pts):>4} | "
                      f"{st.mean(p['consistency'] for p in pts):>11.4f} | "
                      f"{st.mean(p['trigger'] for p in pts):>7.1%} | "
                      f"{st.mean(p['saved'] for p in pts):>9.3f} | {bert_txt:>17} | "
                      f"{delta:>18} | {shown:>6}")
            print(f"{'':>22} | {'':>3} | {'':>12} | {'':>15} | {'':>4} | {'':>11} | "
                  f"{'':>8} | {'':>9} | {'':>17} | {'':>18} | {'':>6}")

    print("consistency  Stage 1.2's greedy consistency rate, the dependence measure")
    print("             the split is drawn on. Higher = the documents change less.")
    print("trigger      share of positions the routed system actually skipped.")
    print("eps saved    eps_budget - eps_usage, generation layer. Read it with the")
    print("             answer-length caveat in experiments/stage4_1_budget.py: a")
    print("             short answer leaves budget unspent without the pre-filter.")
    print("vs base      PAIRED difference against plain DPRAG: per query, this")
    print("             configuration's BERTScore minus the baseline's on the SAME")
    print("             question, averaged inside the stratum, with a 95% interval.")
    print("             Paired because the two strata are not equally easy -- the")
    print("             baseline itself scores differently on them -- so an unpaired")
    print("             comparison would measure which queries fell where. A star")
    print("             marks an interval that excludes zero -- uncorrected, so with")
    print("             sixteen cells about one star is expected by chance. The")
    print("             pattern carries the weight, not any single cell: EVERY")
    print("             high-trigger configuration is negative on high_dependency and")
    print("             none is on low_dependency, and the size tracks the trigger")
    print("             rate. That is the shape the method predicts, not a scatter of")
    print("             significant cells.")
    print("corr         Pearson correlation between per-query consistency and")
    print("             per-query trigger rate, over BOTH strata (printed once per")
    print("             configuration). It asks whether the relationship holds query")
    print("             by query, which a gap between two halves does not.")
    print()
    print("Zero-document queries are excluded from the split and from every row.")
    print("They have no documents to depend on, so their consistency is trivially")
    print("high and every strategy fires on them; including them would load the")
    print("low_dependency stratum with cases that flatter it (CONTEXT.md).")


if __name__ == "__main__":
    main()
