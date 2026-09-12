"""The one timing the efficiency claim is missing: plain DPRAG, no pre-filter.

The proposal's 5.1 step limit asks for a threshold:

> 前置篩選的額外推論成本需低於節省的聚合成本，否則效率反而下降；需量測觸發率
> 臨界值

which is the trigger rate at which the pre-filter starts paying for itself. Below
it the two extra pre-filter rows cost more than the skipped k+1 aggregations save.

The project cannot currently compute it. Every timing it has is of a ROUTED run --
including the baseline, which is produced by routing with a strategy that never
agrees (`sweep.NEVER_AGREE`) and therefore pays for the pre-filter batch at every
position while skipping nothing. Measured against that, the threshold question is
already answered "0%", trivially and wrongly. The denominator the proposal means
is `dp_model.dp_chat`: k+1 streams, no pre-filter batch, no router.

WHY 40 QUERIES AND WHY BOTH ARMS IN THE SAME RUN
------------------------------------------------
Per-query time is noisy -- Llama's baseline has sd 14.3 s on a mean of 44.9 s, a
coefficient of variation of 32%. Pinning a MEAN to +-5% would take 156 queries.

But the noise is shared: long documents and long answers make both arms slow on
the same query. Baseline and strategy A correlate at 0.933 across the 180 Phase 2
queries, and their per-query RATIO has a coefficient of variation of 15.5% -- 40
queries for the same +-5%. So this measures a paired ratio, not two means, and it
runs both arms on the same query back to back.

Both arms also have to run in the SAME session for a second reason. The Phase 2
curve was measured on a pod that no longer exists. A plain-DPRAG number taken on
new hardware and divided into that old curve would fold the machine's speed into
the threshold, invisibly. Dividing by a routed baseline measured here cancels it:
the threshold is read off the curve in units of "times the routed baseline", and
both sides of that ratio come from the same GPU.

What is still borrowed from Phase 2 is the curve's SHAPE -- the ratios between
configurations at different trigger rates. That is arithmetic over how many
forward passes get skipped, not a property of the machine, but it is an
assumption and it is printed with the result.

THE BYTE-EQUALITY CHECK IS NOT A FORMALITY
------------------------------------------
`sweep.NEVER_AGREE` is supposed to reproduce `dp_chat` token for token, and the
comment above it in `dprag/sweep.py` says `test_router.py` pins this. It does not:
those tests drive a fake model and pin that the never-agree path pays at every
position and that the aggregator sees k+1 streams. Necessary, but not the same
claim as "the real model emits the same tokens through both code paths". Nothing
has ever checked that on a real model.

This run does, for free, because it already generates both. If the answers differ
the timing comparison is void -- the two arms did different amounts of work -- and
so is the baseline's standing as "plain DPRAG" everywhere else in the project.

    uv run python experiments/stage5_1_plain_timing.py
"""

import statistics as st
import time

import torch

from dprag import paths, run_record, sweep
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router

EPSILON = 40
N_QUERIES = 40
EXPERIMENT = ExperimentConfig(n_queries=200)
FILENAME = f"stage5_1_plain_timing_{N_QUERIES}q_eps{EPSILON}"

# The Phase 2 curve this anchors into. Order here is documentation; the points are
# sorted by the trigger rate read out of each record, not by this list.
CURVE = [f"stage3_2_main_{name}_eps{EPSILON}" for name in
         ("baseline", "B_k20_t0.9", "B_k20_t0.7", "A", "B_k50_t0.5")]


def load_curve() -> list[tuple[str, float, float]]:
    """(configuration, mean trigger rate, mean seconds) for each Phase 2 point."""
    out = []
    for stem in CURVE:
        path = paths.results_dir() / f"{stem}.json"
        if not path.exists():
            raise SystemExit(
                f"missing {stem}.json -- the threshold is read off the Phase 2 "
                "timing curve, and without every point there is no curve to read."
            )
        record = run_record.load(path)
        names = record.metric("strategies", [])
        if len(names) != 1:
            raise SystemExit(f"{stem} holds {names}; expected exactly one strategy")
        rows = [r["by_strategy"][names[0]] for r in record.per_item]
        out.append((names[0],
                    st.mean(r["trigger_rate"] for r in rows),
                    st.mean(r["seconds"] for r in rows)))
    return sorted(out, key=lambda t: t[1])


def threshold(curve: list[tuple[str, float, float]], ratio: float):
    """Trigger rate where the routed curve crosses plain DPRAG's cost.

    `ratio` is plain DPRAG divided by the routed baseline, both measured here, so
    normalising the curve by its own trigger-rate-0 point takes the machine out of
    both sides.

    Returns (trigger_rate, note). A None trigger rate means the crossing lies
    outside the measured range, which is a result rather than a failure: the curve
    covers 0% to 95.8% and the answer can legitimately sit beyond either end.
    """
    anchor = curve[0][2]                      # trigger rate 0: the routed baseline
    points = [(t, s / anchor) for _, t, s in curve]
    if ratio >= points[0][1]:
        return None, ("plain DPRAG is no faster than the routed baseline, so the "
                      "pre-filter costs nothing to install: the threshold is at or "
                      "below 0% and every measured configuration is past it")
    if ratio < points[-1][1]:
        return None, (f"plain DPRAG is cheaper than the routed system even at "
                      f"{points[-1][0]:.1%} trigger, so no measured configuration "
                      "breaks even and the threshold is above the measured range")
    for (t0, r0), (t1, r1) in zip(points, points[1:]):
        if r1 <= ratio <= r0:
            if r0 == r1:
                return t0, "curve is flat across this interval; reporting its left edge"
            crossed = t0 + (t1 - t0) * (r0 - ratio) / (r0 - r1)
            return crossed, f"linear interpolation between {t0:.1%} and {t1:.1%}"
    return None, "no bracketing interval found"


def paired_report(rows: list[dict], curve) -> None:
    plain = [r["plain_seconds"] for r in rows]
    routed = [r["routed_seconds"] for r in rows]
    ratios = [p / q for p, q in zip(plain, routed)]
    n = len(ratios)
    mu = st.mean(ratios)
    sd = st.stdev(ratios) if n > 1 else 0.0
    half = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0

    print("=== plain DPRAG vs the routed baseline, same queries, same session ===")
    print(f"{'':>22} {'mean s':>8} {'sd':>7}")
    print(f"{'plain dp_chat':>22} {st.mean(plain):>8.1f} "
          f"{st.stdev(plain) if n > 1 else 0.0:>7.1f}")
    print(f"{'routed NEVER_AGREE':>22} {st.mean(routed):>8.1f} "
          f"{st.stdev(routed) if n > 1 else 0.0:>7.1f}")
    print()
    print(f"paired ratio plain/routed = {mu:.4f}  +-{half:.4f}  (95%, n={n})")
    print("The pre-filter batch costs whatever this ratio falls short of 1.0. Read")
    print("the ratio, not the two means: the means carry the query-to-query noise")
    print("that pairing removes.")
    print()

    identical = sum(1 for r in rows if r["identical"])
    print("=== do both code paths emit the same answer? ===")
    if identical == n:
        print(f"  {n}/{n} IDENTICAL -- the routed baseline really is plain DPRAG,")
        print("  checked against the real model for the first time.")
    else:
        print(f"  *** {n - identical}/{n} DIFFER ***")
        print("  The two arms did different work, so the ratio above is not a")
        print("  measurement of the pre-filter's cost. It also means the baseline")
        print("  used everywhere else in this project is not plain DPRAG.")
        bad = next(r for r in rows if not r["identical"])
        print(f"  first divergence on: {bad['query'][:70]}")
    print()

    anchor = curve[0][2]
    print("=== the Phase 2 curve, normalised by its own trigger-rate-0 point ===")
    print(f"{'config':>12} {'trigger':>9} {'seconds':>9} {'relative':>9}")
    for name, trigger, seconds in curve:
        print(f"{name:>12} {trigger:>9.1%} {seconds:>9.1f} {seconds / anchor:>9.3f}")
    print(f"{'plain (here)':>12} {'n/a':>9} {'':>9} {mu:>9.3f}")
    print()

    crossed, note = threshold(curve, mu)
    print("=== the proposal's trigger-rate threshold ===")
    if crossed is None:
        print(f"  outside the measured range: {note}")
    else:
        print(f"  ~{crossed:.1%} trigger rate  ({note})")
        # The interval runs the other way round: a LARGER ratio means plain DPRAG
        # is relatively slower, which the curve crosses EARLIER.
        lo, _ = threshold(curve, min(mu + half, 1.0))
        hi, _ = threshold(curve, max(mu - half, 0.0))
        if lo is not None and hi is not None:
            print(f"  carrying the ratio's interval through: {lo:.1%} to {hi:.1%}")
        else:
            print("  the ratio's interval runs off the end of the measured curve, "
                  "so only the point estimate is reportable")
    print()
    print("ASSUMPTION, not testable from this run: the curve's SHAPE -- the ratios")
    print("between configurations -- comes from Phase 2, measured on different")
    print("hardware. Only its scale is re-anchored here. The shape is set by how")
    print("many forward passes each trigger rate skips, which is arithmetic, but a")
    print("machine whose k+1 batch scaled differently would change it.")


def main():
    exp = EXPERIMENT.with_(gen_epsilon=float(EPSILON))
    curve = load_curve()                       # fail before an hour of model loading
    bench = Bench.build(exp)
    # Slice the full sample rather than asking for forty: load_queries(n=40) is not
    # guaranteed to return the first forty of load_queries(n=200), and these have to
    # be queries Phase 2 also timed.
    queries = bench.queries()[:N_QUERIES]

    print(f"=== plain DPRAG timing | model={exp.gen_model} | eps={EPSILON} ===")
    print(f"{len(queries)} queries x 2 arms | max_retrieve={exp.max_retrieve} "
          f"| dtype={exp.gen_dtype}")
    print(flush=True)

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=float(EPSILON), delta=exp.delta,
    )
    store = bench.engine.pup_vector_store
    rows: list[dict] = []
    zero_docs = 0

    for i, q in enumerate(queries):
        question = q.query if hasattr(q, "query") else q
        # Retrieval is a function of (query, seed), so both arms and Phase 2 all
        # see the same documents for this question.
        store.reseed_for(question)
        documents = store.pup_retrieve(question)
        if not documents:
            zero_docs += 1
            continue

        def run_plain():
            # Re-seeded per arm so neither arm's trajectory depends on how much
            # randomness the other consumed -- the rule routed_sweep already uses.
            torch.manual_seed(exp.seed)
            began = time.time()
            text = bench.dp_model.dp_chat(documents, question, dp_cfg)
            return time.time() - began, text

        def run_routed():
            torch.manual_seed(exp.seed)
            began = time.time()
            result = Router(bench.dp_model, sweep.NEVER_AGREE, dp_cfg).generate(
                documents, question)
            return time.time() - began, result

        # Alternate which arm goes first. Under a fixed order anything that drifts
        # within a query -- clock boost, cache state -- lands on the same arm every
        # time and gets read as the pre-filter's cost.
        if i % 2 == 0:
            plain_seconds, plain_text = run_plain()
            routed_seconds, routed = run_routed()
        else:
            routed_seconds, routed = run_routed()
            plain_seconds, plain_text = run_plain()

        rows.append({
            "query": question,
            "n_documents": len(documents),
            "plain_seconds": plain_seconds,
            "routed_seconds": routed_seconds,
            "plain_text": plain_text,
            "routed_text": routed.text,
            "routed_emitted": list(routed.emitted),
            "identical": plain_text == routed.text,
            "order": "plain_first" if i % 2 == 0 else "routed_first",
        })
        mark = "" if rows[-1]["identical"] else "  *** DIFFERS ***"
        print(f"  [{len(rows):>3}/{len(queries)}] plain {plain_seconds:6.1f}s  "
              f"routed {routed_seconds:6.1f}s  "
              f"ratio {plain_seconds / routed_seconds:.3f}{mark}", flush=True)

    if not rows:
        raise SystemExit("every query retrieved zero documents; nothing was timed")

    ratios = [r["plain_seconds"] / r["routed_seconds"] for r in rows]
    path = run_record.write(
        "stage5_1_plain_timing", exp,
        metrics={
            "n_queries_timed": len(rows),
            "n_zero_document_queries": zero_docs,
            "epsilon": EPSILON,
            "plain_seconds_mean": st.mean(r["plain_seconds"] for r in rows),
            "routed_seconds_mean": st.mean(r["routed_seconds"] for r in rows),
            "paired_ratio_mean": st.mean(ratios),
            "paired_ratio_sd": st.stdev(ratios) if len(ratios) > 1 else 0.0,
            "n_identical": sum(1 for r in rows if r["identical"]),
            "purpose": (
                "Plain DPRAG (dp_model.dp_chat, no pre-filter batch) timed against "
                "the routed never-agree baseline on the same queries in the same "
                "session, so the proposal's trigger-rate threshold can be read off "
                "the Phase 2 curve without the machine's speed folded into it."
            ),
        },
        per_item=rows, filename=FILENAME)
    print(f"\n  saved -> {path.name}  "
          f"({len(rows)} queries, {zero_docs} with 0 docs)\n")
    paired_report(rows, curve)


if __name__ == "__main__":
    main()
