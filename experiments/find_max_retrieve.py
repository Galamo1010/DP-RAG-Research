"""How many documents fit in one routed generation before the GPU runs out.

The pre-filter's value depends on the documents being relevant, and they largely
are not: at max_retrieve=10 the DP threshold admits ~83 documents (median) and ten
are drawn from them at random, so only 1.5 of the true top-10 are seen. Raising the
cap is the direct fix, and preflight showed it is nearly free in time -- 10 to 30
documents cost +16% per query, because the 128 sequential decode steps dominate and
batch width barely registers.

**VRAM is the binding constraint, not time.** max_retrieve=100 -- 101 rows -- runs
out of memory on an 80 GB A100. This finds where the wall actually is, by bisecting
between a value known to work and one known to fail.

Each probe generates one answer per query at that setting and records peak
allocation. An OOM is caught, the allocator is drained, and the search continues
downward; nothing here should leave the process unusable.

Two things worth knowing when reading the output:

* **k is capped by the DP threshold, not only by max_retrieve.** The median query
  admits ~83 documents, so settings above that change little for most queries and
  the reported k will sit below the cap.
* **Peak memory depends on prompt length**, which varies by query, so a setting
  that survives these queries can still fail on a longer one. The result is a
  practical ceiling, not a proof. Leave headroom.

    uv run python experiments/find_max_retrieve.py
"""

import time

import torch

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router
from dprag.strategies import strategy_a

# Bisection bounds. 30 survived preflight, 100 did not.
KNOWN_GOOD = 30
KNOWN_BAD = 100
# Stop when the bracket is this tight; finer than this is noise, since peak memory
# moves with prompt length anyway.
RESOLUTION = 5
# Per probe. More queries is a better worst case and a slower search; three is
# enough to catch a long-prompt query without turning this into an experiment.
QUERIES_PER_PROBE = 3


def drain():
    """Return the allocator to a clean state after a failed probe."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def probe(bench, dp_cfg, questions, limit: int) -> tuple[bool, float, float, float]:
    """Try `limit` at full generation. Returns (ok, peak_gb, seconds, mean_k)."""
    store = bench.engine.pup_vector_store
    store.max_retrieve = limit
    drain()

    seconds, ks = [], []
    try:
        for question in questions:
            docs = store.pup_retrieve(question)
            if not docs:
                continue
            torch.manual_seed(bench.config.seed)
            began = time.time()
            result = Router(bench.dp_model, strategy_a, dp_cfg).generate(docs, question)
            torch.cuda.synchronize()
            seconds.append(time.time() - began)
            ks.append(len(docs))
            del result
    except torch.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 1e9
        drain()
        return False, peak, 0.0, (sum(ks) / len(ks) if ks else 0.0)

    peak = torch.cuda.max_memory_allocated() / 1e9
    if not seconds:
        # Every query retrieved nothing: says nothing about memory either way.
        return True, peak, 0.0, 0.0
    return True, peak, sum(seconds) / len(seconds), sum(ks) / len(ks)


def main():
    exp = ExperimentConfig(n_queries=QUERIES_PER_PROBE)
    print(f"=== max_retrieve ceiling | model={exp.gen_model} | "
          f"{QUERIES_PER_PROBE} queries per probe ===", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  {total_gb:.0f} GB\n", flush=True)

    bench = Bench.build(exp)
    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=exp.gen_epsilon, delta=exp.delta)
    questions = [q.query for q in bench.queries()]

    print(f"{'max_retrieve':>12} | {'k':>5} | {'peak VRAM':>10} | "
          f"{'s/query':>8} | result", flush=True)
    print("-" * 60, flush=True)

    tried: list[tuple[int, bool, float, float, float]] = []
    lo, hi = KNOWN_GOOD, KNOWN_BAD

    # Confirm the lower bound rather than trusting it: a different transformers
    # version or a longer query can move the wall since preflight ran.
    ok, peak, secs, mean_k = probe(bench, dp_cfg, questions, lo)
    tried.append((lo, ok, peak, secs, mean_k))
    print(f"{lo:>12} | {mean_k:>5.1f} | {peak:>7.1f} GB | {secs:>7.1f}s | "
          f"{'OK' if ok else 'OOM'}", flush=True)
    if not ok:
        print("\nThe assumed-good lower bound failed. Something changed since "
              "preflight; lower KNOWN_GOOD and re-run.")
        return

    while hi - lo > RESOLUTION:
        mid = (lo + hi) // 2
        ok, peak, secs, mean_k = probe(bench, dp_cfg, questions, mid)
        tried.append((mid, ok, peak, secs, mean_k))
        print(f"{mid:>12} | {mean_k:>5.1f} | {peak:>7.1f} GB | {secs:>7.1f}s | "
              f"{'OK' if ok else 'OOM'}", flush=True)
        if ok:
            lo = mid
        else:
            hi = mid

    best = max(t for t in tried if t[1])
    print("\n" + "=" * 60)
    print(f"Highest setting that survived: max_retrieve = {best[0]}")
    print(f"  k reached {best[4]:.1f} documents, {best[2]:.1f} GB peak of "
          f"{total_gb:.0f} GB, {best[3]:.1f} s/query")
    print(f"  First failure at or above: {hi}")
    print()
    print("Peak memory moves with prompt length, and these were three queries.")
    print("Take a margin: a setting that survives here can still fail on a longer")
    print("prompt in a 200-query run, and a run that dies at query 137 costs more")
    print("than the documents it was trying to fit.")
    print()
    print("Set it in dprag/config.py (ExperimentConfig.max_retrieve) and record the")
    print("reasoning -- the retrieval quality it buys is measurable and the change")
    print("makes every earlier result incomparable, so it needs an ADR either way.")


if __name__ == "__main__":
    main()
