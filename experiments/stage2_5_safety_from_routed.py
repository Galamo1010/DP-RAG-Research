"""Stage 2.5 -- are clinically-loaded positions skipped more often than ordinary ones.

The question is the proposal's: if the pre-filter treats a drug name like an
article, the epsilon it saves is bought with the part of the answer that carries
clinical risk. It is asked by comparing two skip rates over the same answer --
clinical positions against everything else -- for configurations ordered strict
to loose.

WHY THIS REPLACES `stage2_safety_check.py`
------------------------------------------
The first attempt (`results/stage2_safety_check_30q.json`, 2026-07-30) searched
for drug names in `norag_argmax_text`, a per-step teacher-forced overlay that no
system emits, and matched artifacts like "youritisitis" rather than medicine. ADR
0003 records why: detecting entities in a routed system's output requires a
routed system, and there was not one yet. It also ran at `max_retrieve=10`,
before ADR 0008, and it found **three** clinical positions across 27 queries --
too few to support any statement at all.

That is now fixed by arithmetic rather than by GPU time. The Stage 3 records
carry `clinical`, one entry per emitted position, alongside `paid_positions`, so
the classification runs offline over answers the router really produced. Measured
at eps=40 on Llama it finds 180-240 clinical positions per configuration instead
of three.

The old script still generates; it is superseded for this measurement and kept
only as the record of how the question was first asked.

TWO SPLITS, BOTH REPORTED
-------------------------
**word-like against pattern-like.** `medical_flags` separates spans a vocabulary
can vet (drug and condition names: `amoxicillin`, `gastritis`) from spans it
cannot (doses and frequencies: `500mg`, `twice daily`). Nothing can judge whether
"197 mg" is a plausible dose, so the second kind is a less trustworthy detection
-- but a skipped dose is at least as dangerous clinically as a skipped drug name.
Merging them would let the reliable class dilute the unreliable one and hide any
difference between them. They are reported separately, with a combined line for
completeness.

**all positions against span openings.** A drug name occupies several tokens, and
after the first one the rest are forced by spelling: the model is not choosing to
say something clinical at position 4 of "amoxicillin", it is finishing a word.
`TokenMark.is_first` marks the opening. Counting every position measures exposure
(how much clinical text rode the free path); counting openings measures decisions
(how often the pre-filter let a clinical choice through unpaid). Both are
reported because they answer different questions and can disagree.

The plain-position denominator is the same in both splits -- every non-clinical
position -- so the two rates stay comparable to each other.

HOW TO READ IT
--------------
`relative risk` above 1 means clinical positions were skipped more often than
ordinary ones, which is the failure the check is looking for. Below 1 means the
pre-filter is *more* cautious at clinical positions than elsewhere.

Every rate carries a 95% Wilson interval, because several cells rest on fewer
than twenty positions and a point estimate there is not a measurement. Cells
below `THIN` positions are marked; do not quote them.

    uv run python experiments/stage2_5_safety_from_routed.py
"""

import glob
import math

from dprag import paths, run_record
from dprag.medical_flags import WORD_KIND

RECORD_GLOB = "stage3_*"
THIN = 20  # below this many positions a rate is reported but must not be quoted

# Strict to loose, by measured trigger rate rather than by tau (ADR 0009).
CONFIG_ORDER = ["baseline", "B_k20_t0.9", "B_k20_t0.7", "A", "B_k50_t0.5"]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation because the counts here are small
    and several rates sit near 1.0, where the normal interval runs past 100% and
    stops meaning anything.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(strategy_record: dict, openings_only: bool) -> dict[str, tuple[int, int]]:
    """kind -> (skipped, total) over one answer's positions.

    Kinds are `WORD_KIND`, "pattern", "clinical" (their union) and "plain".
    Skipped means the position took the free path: it was not in
    `paid_positions`, so no epsilon was spent deciding it.
    """
    clinical = strategy_record.get("clinical") or []
    paid = set(strategy_record.get("paid_positions", []))
    counts = {k: [0, 0] for k in (WORD_KIND, "pattern", "clinical", "plain")}

    for i, mark in enumerate(clinical):
        skipped = i not in paid
        if mark is None:
            counts["plain"][0] += skipped
            counts["plain"][1] += 1
            continue
        kind, is_first = mark
        if openings_only and not is_first:
            # Not counted as clinical, and not moved into the plain denominator
            # either: a continuation token is neither a clinical decision nor an
            # ordinary position, and putting it in `plain` would contaminate the
            # control with the very text under test.
            continue
        key = WORD_KIND if kind == WORD_KIND else "pattern"
        counts[key][0] += skipped
        counts[key][1] += 1
        counts["clinical"][0] += skipped
        counts["clinical"][1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def config_key(row) -> tuple[int, str, str]:
    name, source = row
    rank = CONFIG_ORDER.index(name) if name in CONFIG_ORDER else len(CONFIG_ORDER)
    return (rank, name, source)


def collect(openings_only: bool):
    """(model, eps) -> (config, source) -> kind -> (skipped, total)."""
    groups: dict[tuple[str, float], dict[tuple[str, str], dict]] = {}
    files = [p for p in sorted(glob.glob(str(paths.results_dir() / f"{RECORD_GLOB}.json")))
             if "poles" not in p]
    if not files:
        raise SystemExit("no stage3_* records found; nothing to check")

    for path in files:
        record = run_record.load(path)
        if record.path.stem.startswith("stage3_1_"):
            continue  # 19 queries; the clinical cells would be single digits
        model = record.param("gen_model", "unknown")
        eps = float(record.param("gen_epsilon", 0.0))
        for name in record.metric("strategies", []):
            cell = groups.setdefault((model, eps), {}).setdefault(
                (name, record.path.stem),
                {k: [0, 0] for k in (WORD_KIND, "pattern", "clinical", "plain")})
            for row in record.per_item:
                sr = row["by_strategy"].get(name)
                if not sr:
                    continue
                for kind, (skipped, total) in classify(sr, openings_only).items():
                    cell[kind][0] += skipped
                    cell[kind][1] += total
    return groups


def report(openings_only: bool) -> None:
    title = ("span openings only (is_first) -- clinical DECISIONS"
             if openings_only else
             "every clinical position -- clinical EXPOSURE")
    print()
    print("=" * 124)
    print(f"  {title}")
    print("=" * 124)
    print(f"{'model':>22} | {'eps':>3} | {'config':>12} | {'kind':>8} | {'skipped/total':>14} | "
          f"{'skip rate (95% CI)':>26} | {'plain':>7} | {'rel.risk':>8}")
    print("-" * 124)

    for (model, eps), by_config in sorted(collect(openings_only).items()):
        for key in sorted(by_config, key=config_key):
            cell = by_config[key]
            plain_skipped, plain_total = cell["plain"]
            plain_rate = plain_skipped / plain_total if plain_total else 0.0
            for kind in (WORD_KIND, "pattern", "clinical"):
                skipped, total = cell[kind]
                if not total:
                    continue
                rate = skipped / total
                lo, hi = wilson(skipped, total)
                rr = (rate / plain_rate) if plain_rate else float("nan")
                thin = " *" if total < THIN else "  "
                print(f"{model.split('/')[-1][:22]:>22} | {eps:>3.0f} | {key[0]:>12} | "
                      f"{kind:>8} | "
                      f"{skipped:>6}/{total:<7} | "
                      f"{rate:>7.1%}  [{lo:>5.1%},{hi:>6.1%}]{thin} | "
                      f"{plain_rate:>6.1%} | {rr:>8.2f}")
            print(f"{'':>22} | {'':>3} | {'':>12} | {'':>8} | {'':>14} | "
                  f"{'':>26} | {'':>7} | {'':>8}")


def main():
    print("Stage 2.5 (rewritten) -- clinical vs ordinary skip rates on ROUTED output.")
    print("Source: the Stage 3 records. No generation; nothing here needs a GPU.")
    print()
    print("The 2026-07-30 pilot this replaces found 3 clinical positions across 27")
    print("queries, at max_retrieve=10, on a teacher-forced overlay no system emits.")
    print("Its numbers are not comparable with these and should not be shown beside")
    print("them (ADR 0003, ADR 0008).")

    report(openings_only=False)
    report(openings_only=True)

    print()
    print("kind        word    = drug/condition names, vetted against a vocabulary")
    print("            pattern = doses and frequencies; no vocabulary can vet these,")
    print("                      so the detection is weaker while the clinical risk")
    print("                      of skipping one is not")
    print("            clinical= word + pattern")
    print("skip rate   share of that kind's positions emitted on the FREE path, so")
    print("            chosen by the model's prior with no epsilon spent.")
    print("plain       the same rate over non-clinical positions: the control.")
    print("rel.risk    skip rate / plain rate. Above 1 = clinical positions skipped")
    print("            MORE often than ordinary ones, which is the failure mode.")
    print(f"*           fewer than {THIN} positions in the cell. Reported for")
    print("            completeness; not quotable.")
    print()
    print("The baseline row is the arithmetic check, not a result: it never skips,")
    print("so every rate on it must read 0.0% and a non-zero one means the")
    print("classification is wrong.")


if __name__ == "__main__":
    main()
