"""The proposal's third comparison, measured at position resolution.

The proposal (line 115) asks the ablation to quantify

> 策略B在「argmax不同但top-k高度重疊」位置的額外覆蓋增益

which could not be computed at all until `rag_argmax` was recorded (commit
001f15d): the trace held what the NoRAG instance wanted and never what the RAG
instance wanted, so nothing could say whether a paid position was worth paying
for.

WHAT THE TWO COUNTS MEAN
------------------------
**wasted** -- paid positions where `rag_argmax == norag_argmax`. The documents did
not change the model's first choice, so the aggregation ran, epsilon was spent,
and the token was one the free path would have emitted anyway.

**missed** -- free positions where the two differ. The documents *would* have
changed the token and the saving threw that influence away. This is a negative
pole lean at position resolution, and unlike the lean it cannot be produced by
copying artifacts.

Together they say whether a strategy is *selecting* the document-independent
positions or merely skipping a lot of them.

STRATEGY A'S TWO ZEROES ARE TAUTOLOGICAL
----------------------------------------
A's agreement test *is* `rag_argmax == norag_argmax`, so A cannot waste (it never
pays where they agree) and cannot miss (it never skips where they differ). Both
zeroes are therefore self-checks on the recorded field rather than results: a
non-zero count on an A run means the strategy that ran and the argmax that was
recorded disagree, and the record is broken. The content of the table is how far
the other configurations fall from that line.

THE EQUIVALENCE CHECK RIDES ALONG
---------------------------------
Two changes claim to leave generation untouched: `logits_to_keep=1` (c29e499) and
the `rag_argmax` recording itself. Both were argued from the code -- the second
reads a tensor the strategy has already seen and draws no randomness -- and this
project's history says an argument from the code is not a check. `dp_chat` versus
a never-agreeing router is what caught the missing sampling warpers, and it was
run last rather than first.

So the probe's emitted tokens are compared against the Phase 2 records at the same
epsilon, query by query, token by token. Anything other than IDENTICAL puts both
"provably identical" changes in doubt, and with them every Phase 3 number taken
after them.

WHAT THIS DOES NOT COVER
------------------------
56 scored queries, one budget (eps=40), one model (Llama). It is a probe, not a
phase. The counts are position-level and therefore have large denominators --
about 6,000 positions per configuration -- but they come from one narrow slice of
the grid and the Wilson intervals describe sampling within it, not across it.

    uv run python experiments/stage3_4_argmax.py
"""

import glob
import math

from dprag import paths, run_record, trace

PROBE_GLOB = "stage3_argmax_probe_*"
PHASE2_TEMPLATE = "stage3_2_main_{config}_eps{eps:g}"
FIG_DIR = "figures"

# Strict to loose by measured trigger rate (ADR 0009), so the table reads in the
# order the configurations are discussed.
CONFIG_ORDER = ["baseline", "B_k20_t0.9", "B_k20_t0.7", "A", "B_k50_t0.5"]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Wilson rather than the normal approximation
    because several of these rates sit at or near 0 and 1, where the normal
    interval runs outside [0, 1] and stops meaning anything."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def counts(strategy_record: dict) -> dict[str, int]:
    """Positions split four ways for one answer.

    `wasted` and `missed` come from `dprag.trace`, which owns the definitions so
    that this script and any later reader cannot drift apart on what they mean.
    """
    n = len(strategy_record["emitted"])
    paid = len(strategy_record["paid_positions"])
    return {
        "positions": n,
        "paid": paid,
        "free": n - paid,
        "wasted": len(trace.wasted_paid_positions(strategy_record)),
        "missed": len(trace.missed_free_positions(strategy_record)),
    }


def load_probe():
    matches = sorted(glob.glob(str(paths.results_dir() / f"{PROBE_GLOB}.json")))
    if not matches:
        raise SystemExit(
            "no stage3_argmax_probe_* record found. This analysis needs "
            "`rag_argmax`, which only runs after commit 001f15d carry; earlier "
            "records cannot be back-filled without re-generating.")
    return run_record.load(matches[-1])


def equivalence_check(probe) -> None:
    """Compare the probe's tokens against Phase 2's, per configuration.

    A divergence here is not a small problem. It would mean `logits_to_keep=1` or
    the `rag_argmax` recording changed generation, and every Phase 2 and Phase 3
    number produced after those commits would need re-checking.
    """
    eps = float(probe.param("gen_epsilon", 0.0))
    print("=== equivalence: probe vs Phase 2, token for token ===")
    print("Both claim to be the same generation. `logits_to_keep=1` and the")
    print("rag_argmax recording were argued from the code; this is the check.")
    print()
    print(f"{'config':>12} | {'shared queries':>14} | verdict")
    print("-" * 62)

    any_missing = False
    for name in sorted(probe.metric("strategies", []), key=order_key):
        stem = PHASE2_TEMPLATE.format(config=name, eps=eps)
        matches = glob.glob(str(paths.results_dir() / f"{stem}.json"))
        if not matches:
            any_missing = True
            print(f"{name:>12} | {'-':>14} | no {stem}.json to compare against")
            continue
        main = {r["query"]: r for r in run_record.load(matches[0]).per_item}
        shared = diverged = 0
        first = None
        for row in probe.per_item:
            other = main.get(row["query"])
            if other is None or name not in other["by_strategy"]:
                continue
            shared += 1
            a = row["by_strategy"][name]["emitted"]
            b = other["by_strategy"][name]["emitted"]
            if a != b:
                diverged += 1
                if first is None:
                    at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                              min(len(a), len(b)))
                    first = (at, len(a), len(b))
        if not shared:
            print(f"{name:>12} | {0:>14} | no shared query")
        elif diverged == 0:
            print(f"{name:>12} | {shared:>14} | IDENTICAL")
        else:
            print(f"{name:>12} | {shared:>14} | DIVERGED on {diverged}, first at "
                  f"position {first[0]} (lengths {first[1]} vs {first[2]})")
    if any_missing:
        print()
        print("!! A missing Phase 2 record means that configuration was NOT checked.")
    print()


def order_key(name: str) -> tuple[int, str]:
    return (CONFIG_ORDER.index(name) if name in CONFIG_ORDER else len(CONFIG_ORDER),
            name)


def table(probe, totals) -> None:
    eps = float(probe.param("gen_epsilon", 0.0))
    n_scored = probe.metric("n_queries_scored", len(probe.per_item))
    print(f"=== where the budget went | {probe.param('gen_model')} | eps={eps:g} | "
          f"{n_scored} queries ===")
    print()
    print(f"{'config':>12} | {'positions':>9} | {'paid':>6} | "
          f"{'wasted / paid':>26} | {'free':>6} | {'missed / free':>26}")
    print("-" * 106)
    for name in sorted(totals, key=order_key):
        c = totals[name]
        wl, wh = wilson(c["wasted"], c["paid"])
        ml, mh = wilson(c["missed"], c["free"])
        wasted = (f"{c['wasted']:>5}/{c['paid']:<5} {c['wasted'] / c['paid']:>5.1%} "
                  f"[{wl:.0%},{wh:.0%}]" if c["paid"] else f"{'-':>25}")
        missed = (f"{c['missed']:>5}/{c['free']:<5} {c['missed'] / c['free']:>5.1%} "
                  f"[{ml:.0%},{mh:.0%}]" if c["free"] else f"{'-':>25}")
        print(f"{name:>12} | {c['positions']:>9} | {c['paid']:>6} | {wasted:>26} | "
              f"{c['free']:>6} | {missed:>26}")
    print()
    print("wasted / paid  epsilon spent where rag_argmax == norag_argmax: the")
    print("               documents did not change the first choice, so the budget")
    print("               bought a token the free path would have emitted anyway.")
    print("missed / free  positions skipped where the two differ: the documents")
    print("               WOULD have changed the token and the saving discarded it.")
    print()
    print("STRATEGY A'S ZEROES ARE TAUTOLOGICAL, NOT A RESULT. A's agreement test is")
    print("exactly that equality, so it can neither waste nor miss by construction.")
    print("They are the self-check on the recorded field; a non-zero value there")
    print("would mean the strategy that ran and the argmax that was recorded")
    print("disagree. What the table says is how far the others fall from that line.")


def figure(probe, totals) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for name in sorted(totals, key=order_key):
        c = totals[name]
        x = c["free"] / c["positions"] if c["positions"] else 0.0
        y = c["missed"] / c["free"] if c["free"] else 0.0
        ax.scatter([x], [y], s=110, zorder=3)
        waste = c["wasted"] / c["paid"] if c["paid"] else 0.0
        ax.annotate(f"{name}\nwastes {waste:.0%} of its spend",
                    (x, y), textcoords="offset points",
                    xytext=(-10, 6) if x > 0.6 else (10, 6),
                    ha="right" if x > 0.6 else "left", fontsize=8)
    ax.set_xlim(-0.05, 1.15)
    # Headroom for the labels, which sit above their points and were clipped by
    # the top spine at the default limits.
    top = max((c["missed"] / c["free"] if c["free"] else 0.0) for c in totals.values())
    ax.set_ylim(-0.008, max(top * 1.35, 0.02))
    ax.set_xlabel("share of positions skipped (what the strategy saves)")
    ax.set_ylabel("share of skipped positions where the documents mattered\n"
                  "(what the saving cost)")
    ax.set_title(f"Argmax ablation | {probe.param('gen_model')} | "
                 f"eps={float(probe.param('gen_epsilon', 0)):g} | "
                 f"{probe.metric('n_queries_scored', 0)} queries", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_dir = paths.results_dir() / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "stage3_4_argmax_ablation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print()
    print(f"wrote {out}")


def main():
    probe = load_probe()
    print(f"source: {probe.path.stem}")
    print()
    equivalence_check(probe)

    totals: dict[str, dict[str, int]] = {}
    for row in probe.per_item:
        for name, sr in row["by_strategy"].items():
            acc = totals.setdefault(
                name, {"positions": 0, "paid": 0, "free": 0, "wasted": 0, "missed": 0})
            for k, v in counts(sr).items():
                acc[k] += v
    if not totals:
        raise SystemExit("the probe record holds no strategies")

    table(probe, totals)
    figure(probe, totals)


if __name__ == "__main__":
    main()
