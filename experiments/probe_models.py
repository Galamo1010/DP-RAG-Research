"""Can the other two models run Phase 3 at all, and at what cost.

Phase 3 is ten hours of rented A100. Two things can make that ten hours worthless,
and both are answerable in twenty minutes:

**gemma-4-12B may not load.** It needs transformers 5.x, which needs
huggingface-hub 1.x, and this environment has neither. That is a *loading* failure
with a *packaging* fix.

**Qwen2.5-14B may not generate.** Llama-3.1-8B already peaks near 68 of 80 GB at
`max_retrieve=40`, because the router runs k+1 = 41 sequences in one batch. A 14B
model adds roughly 12 GB of weights on top of that. That is a *memory* failure with
a *lower max_retrieve* fix, and the fix costs comparability: a model forced down to
25 documents is not being compared on the same footing as one running 40.

The two failures need different answers, so they are reported separately rather
than as one "it didn't work".

Three queries per model is enough. Peak memory is set by the longest prompt in the
batch, not by how many queries have run, so a fourth query reveals nothing a third
did not. The per-query seconds are a bonus: Phase 3's cost is currently estimated
by multiplying Llama's measured rate by a guessed 1.7x, and this replaces the guess.

    uv run python experiments/probe_models.py
"""

import gc
import time
import traceback

import torch

from dprag import run_record
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router
from dprag.strategies import strategy_a

# The primary model is not probed: Phase 2 ran ten times on it, so its answer is
# already known and re-measuring it would cost an embedding pass for nothing.
PROBE_MODELS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-4-12B-it",
]

N_QUERIES = 3
EPSILON = 40.0


def gb(n: int) -> float:
    return n / 1024 ** 3


def probe(model_id: str, base: ExperimentConfig) -> dict:
    """Load one model, generate a few routed answers, report what broke if any."""
    exp = base.with_(gen_model=model_id)
    row: dict = {
        "model": model_id,
        "max_retrieve": exp.max_retrieve,
        "loaded": False,
        "generated": False,
        "oom": False,
        "error": None,
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    row["gpu_total_gb"] = round(gb(total), 1)

    began = time.time()
    try:
        bench = Bench.build(exp)
        # `DPModel.model` is a cached_property, so Bench.build does not actually
        # load the weights -- the first attribute access does. Touch it here or a
        # download or version failure escapes this try block and crashes the probe
        # instead of being reported as the load failure it is.
        bench.dp_model.model
    except Exception as e:
        # Loading is where the packaging problem shows up. Keep the type: a
        # transformers version error and an out-of-disk error need different fixes.
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        row["stage"] = "load"
        print(f"  LOAD FAILED  {row['error']}", flush=True)
        return row

    row["loaded"] = True
    row["load_seconds"] = round(time.time() - began, 1)
    tok = bench.dp_model.tokenizer
    row["vocab_size"] = len(tok)
    row["weights_gb"] = round(
        gb(sum(p.numel() * p.element_size() for p in bench.dp_model.model.parameters())), 1
    )
    print(f"  loaded in {row['load_seconds']:.0f}s  vocab={row['vocab_size']:,}  "
          f"weights={row['weights_gb']} GB", flush=True)

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=EPSILON, delta=exp.delta,
    )
    store = bench.engine.pup_vector_store
    router = Router(bench.dp_model, strategy_a, dp_cfg)

    seconds, ks = [], []
    try:
        for q in bench.queries(N_QUERIES):
            # Same seeding as the sweep, so the documents are the ones Phase 3
            # would actually see and the peak memory is not measured on a
            # conveniently short prompt.
            store.reseed_for(q.query)
            documents = store.pup_retrieve(q.query)
            if not documents:
                continue
            torch.manual_seed(exp.seed)
            t = time.time()
            router.generate(documents, q.query)
            seconds.append(time.time() - t)
            ks.append(len(documents))
            print(f"    k={len(documents):2d}  {seconds[-1]:5.1f}s  "
                  f"peak={gb(torch.cuda.max_memory_allocated()):.1f} GB", flush=True)
        row["generated"] = bool(seconds)
    except torch.cuda.OutOfMemoryError as e:
        row["oom"] = True
        row["stage"] = "generate"
        row["error"] = str(e)[:300]
        print(f"  OOM at max_retrieve={exp.max_retrieve}", flush=True)
    except Exception as e:
        row["stage"] = "generate"
        row["error"] = f"{type(e).__name__}: {e}"[:400]
        traceback.print_exc()

    row["peak_gb"] = round(gb(torch.cuda.max_memory_allocated()), 1)
    row["reserved_gb"] = round(gb(torch.cuda.max_memory_reserved()), 1)
    if seconds:
        row["seconds_per_query"] = round(sum(seconds) / len(seconds), 1)
        row["k_seen"] = ks

    # Drain before the next model, or the second probe measures the first model's
    # leftovers and reports an OOM that is an artefact of this script.
    del router, store, bench
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return row


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this probe only means anything on the pod")

    base = ExperimentConfig()
    print(f"=== model probe | max_retrieve={base.max_retrieve} | eps={EPSILON:.0f} "
          f"| {N_QUERIES} queries each ===")
    print(f"primary model (already known to work): {base.gen_model}")
    print(f"probing: {PROBE_MODELS}\n", flush=True)

    rows = []
    for model_id in PROBE_MODELS:
        print(f"--- {model_id} ---", flush=True)
        rows.append(probe(model_id, base))
        print(flush=True)

    print("=== verdict ===")
    print(f"{'model':>32} | {'loads':>5} | {'runs@40':>7} | {'peak GB':>7} | {'s/query':>7}")
    print("-" * 76)
    for r in rows:
        print(f"{r['model'][:32]:>32} | {'yes' if r['loaded'] else 'NO':>5} | "
              f"{('yes' if r['generated'] else ('OOM' if r['oom'] else 'NO')):>7} | "
              f"{r.get('peak_gb', 0):>7.1f} | {r.get('seconds_per_query', 0):>7.1f}")
    print()
    for r in rows:
        if r["error"]:
            print(f"{r['model']}  [{r.get('stage')}]  {r['error']}\n")

    out = run_record.write(
        "probe_models", base,
        metrics={
            "probed": PROBE_MODELS,
            "n_queries": N_QUERIES,
            "epsilon": EPSILON,
            "note": (
                "Feasibility only. A model that survives three queries can still "
                "fail on a longer prompt; peak memory moves with prompt length. "
                "Leave headroom before committing Phase 3."
            ),
        },
        per_item=rows, filename="probe_models",
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
