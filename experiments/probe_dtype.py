"""What loading in bfloat16 costs, and what it changes.

Every generation record in results/ came from a loader that passed no dtype at
all, which under transformers 4.57 means float32: 55.0 GiB of weights for
Qwen2.5-14B against the 27.5 its checkpoint actually carries. That width buys no
precision -- the checkpoint is bf16 on the hub, so float32 only zero-extends it
-- and it costs two bytes per weight, the non-tensor-core matmul path, and
possibly most of the 12.4 hours Phase 3's Qwen arm is budgeted at, since that
budget is 82.6 s/query from probe_qwen.json multiplied out.

Both halves of the trade are measured in one run:

* **cost** -- weights, peak VRAM, seconds per query, in the fields
  `probe_models.probe` already writes, so these numbers can be read directly
  beside probe_qwen.json's rather than in a private format.
* **difference** -- emitted token ids, NoRAG argmax and paid positions, in the
  fields `env_equivalence.py` already writes, so "what dtype changes" lands on
  the same scale as the already-measured "what the environment changes".

The second half is the reason this is not just a stopwatch. A three-fold
speed-up that quietly moves the answers is not a speed-up, it is a different
experiment -- and a generation-path change in this codebase does not raise, it
returns.

**One bench, two dtypes.** The store, its embeddings and the retrieved documents
are built once and reused, so retrieval is identical across the two arms by
construction rather than by seed, and the corpus is embedded once. Only the
weights are dropped and reloaded.

**float32 first**, because it is the larger of the two: if it fails to release,
the bfloat16 arm still has room to load and report that it happened.

**The queries are the longest in the set**, the same three probe_qwen.py used
and for the same reason -- peak memory is set by the longest prompt, and the
seconds per query only compare with probe_qwen.json if the prompts match.

    uv run python experiments/probe_dtype.py
    uv run python experiments/probe_dtype.py <model_id>
"""

import os

# Before torch is imported, or the allocator is already configured. The float32
# arm peaks at 75.1 of 79.3 GB, which is where fragmentation starts failing
# allocations that would otherwise fit. Same setting, same reason, as
# stage3_3_crossmodel.py; it is a memory-manager knob and touches no arithmetic.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import sys
import time

import torch

from env_equivalence import first_difference
from probe_models import gb, worst_case_queries

from dprag import run_record, trace
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router
from dprag.strategies import strategy_a

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# float32 first: see the module docstring.
DTYPES = ["float32", "bfloat16"]

N_QUERIES = 3
EPSILON = 40.0

# Whose retrieval to rank prompts by. Retrieval does not depend on the generation
# model, so a Llama record ranks Qwen's prompts correctly -- this is what
# probe_qwen.py did, and matching it is what makes the two files comparable.
SOURCE = "stage3_2_main_baseline_eps40"


def load_and_measure(bench, dtype: str) -> dict:
    """Load the weights at `dtype` and report what that cost."""
    began = time.time()
    model = bench.dp_model.model          # cached_property: this is the load
    row = {
        "gen_dtype": dtype,
        # What was asked for and what arrived are different claims. transformers
        # keeps some modules in fp32 on purpose (`_keep_in_fp32_modules`), so
        # read the answer off the parameters rather than trusting the request.
        "loaded_dtype": str(next(model.parameters()).dtype),
        "load_seconds": round(time.time() - began, 1),
        "weights_gb": round(
            gb(sum(p.numel() * p.element_size() for p in model.parameters())), 1),
    }
    print(f"  loaded in {row['load_seconds']:.0f}s  as {row['loaded_dtype']}  "
          f"weights={row['weights_gb']} GB", flush=True)
    return row


def generate(bench, exp, dp_cfg, queries: list, dtype: str) -> tuple:
    """Run the routed generation, recording what env_equivalence records."""
    store = bench.engine.pup_vector_store
    router = Router(bench.dp_model, strategy_a, dp_cfg)
    rows, seconds = [], []
    for question in queries:
        # Identical to dprag.sweep, so the documents are the ones a real run sees.
        store.reseed_for(question)
        documents = store.pup_retrieve(question)
        if not documents:
            continue
        torch.manual_seed(exp.seed)
        began = time.time()
        result = router.generate(documents, question)
        seconds.append(time.time() - began)
        rows.append({
            "gen_dtype": dtype,
            "query": question,
            "docs": trace.retrieval_trace(store, question, documents),
            "emitted": list(result.emitted),
            "norag_argmax": list(result.norag_argmax),
            "paid_positions": list(result.paid_positions),
            "text": bench.dp_model.tokenizer.decode(
                result.emitted, skip_special_tokens=True),
        })
        print(f"    k={len(documents):2d}  {seconds[-1]:5.1f}s  "
              f"{len(result.emitted):3d} tokens  "
              f"{len(result.paid_positions):3d} paid  "
              f"peak={gb(torch.cuda.max_memory_allocated()):.1f} GB", flush=True)
    return rows, seconds


def drop_weights(bench, dtype: str) -> None:
    """Release the loaded model and arm the next dtype.

    `model` is a cached_property, so removing it from the instance dict is what
    makes the next attribute access load again. Everything else on the bench --
    the store, its embeddings, the tokenizer -- is deliberately kept, which is
    what makes retrieval identical across the arms instead of merely seeded.
    """
    bench.dp_model.__dict__.pop("model", None)
    bench.dp_model.dtype = dtype
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def compare(arms: dict) -> dict:
    """Diff the two arms position by position and say what it licenses."""
    (a_name, a_rows), (b_name, b_rows) = arms.items()
    theirs = {r["query"]: r for r in b_rows}
    shared = [r for r in a_rows if r["query"] in theirs]

    print(f"\n=== {a_name} vs {b_name} ===")
    for mine in shared:
        yours = theirs[mine["query"]]
        same_docs = ([i for i, _ in mine["docs"]] == [i for i, _ in yours["docs"]])
        d_emit = first_difference(mine["emitted"], yours["emitted"])
        d_norag = first_difference(mine["norag_argmax"], yours["norag_argmax"])
        same_paid = mine["paid_positions"] == yours["paid_positions"]
        emit_note = "identical" if d_emit < 0 else f"diverge at {d_emit}"
        norag_note = "identical" if d_norag < 0 else f"diverge at {d_norag}"
        print(f"  docs {'same' if same_docs else 'DIFFER'} | "
              f"emitted {emit_note} "
              f"({len(mine['emitted'])} vs {len(yours['emitted'])} tokens) | "
              f"norag {norag_note} | "
              f"paid {'same' if same_paid else 'DIFFER'} "
              f"({len(mine['paid_positions'])} vs {len(yours['paid_positions'])})")

    verdict = {
        "n_compared": len(shared),
        "positions": sum(len(r["norag_argmax"]) for r in shared),
        "docs_identical": all(
            [i for i, _ in r["docs"]] == [i for i, _ in theirs[r["query"]]["docs"]]
            for r in shared),
        "emitted_identical": all(
            first_difference(r["emitted"], theirs[r["query"]]["emitted"]) < 0
            for r in shared),
        "norag_identical": all(
            first_difference(r["norag_argmax"],
                             theirs[r["query"]]["norag_argmax"]) < 0
            for r in shared),
        "paid_identical": all(
            r["paid_positions"] == theirs[r["query"]]["paid_positions"]
            for r in shared),
    }

    print()
    if not verdict["docs_identical"]:
        print("  VERDICT: retrieval differs, which it cannot do -- both arms share")
        print("  one store. Something in this script is wrong; nothing downstream")
        print("  of this line means anything until that is found.")
    elif verdict["emitted_identical"] and verdict["norag_identical"]:
        print(f"  VERDICT: identical across all {verdict['positions']} positions.")
        print("  On this sample, dtype changed the cost and nothing else. Three")
        print("  queries is not a proof -- it is the absence of a cheap refutation.")
    elif verdict["emitted_identical"]:
        print("  VERDICT: the answers match, but NoRAG argmax does not. The logits")
        print("  are NOT identical; the divergence happened to land where the DP")
        print("  aggregation overrode it. The pre-filter reads NoRAG argmax, so on")
        print("  a free position this would have changed the token and everything")
        print("  after it. Same shape as the environment difference already")
        print("  recorded in results/env_equivalence_*.json -- read them together.")
    else:
        print("  VERDICT: the answers differ. bfloat16 is not a free speed-up; it")
        print("  is a different experiment. Qwen's arm can still use it, but then")
        print("  the dtype must be uniform within the arm and stated beside every")
        print("  number, exactly as the cross-model dtype split has to be.")
    return verdict


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this probe only means anything on the pod")

    import transformers
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    base = ExperimentConfig(gen_model=model_id)
    queries = worst_case_queries(base, N_QUERIES, SOURCE)

    print(f"=== dtype probe | {model_id} | max_retrieve={base.max_retrieve} "
          f"| eps={EPSILON:.0f} ===")
    print(f"environment: torch {torch.__version__}, "
          f"transformers {transformers.__version__}")
    print(f"arms: {DTYPES}   prompts ranked from: {SOURCE}\n", flush=True)

    dp_cfg = DPGenerationConfig(
        temperature=base.temperature, max_new_tokens=base.max_new_tokens,
        alpha=base.alpha, omega=base.omega, epsilon=EPSILON, delta=base.delta,
    )

    torch.cuda.reset_peak_memory_stats()
    _, total = torch.cuda.mem_get_info()

    # Built at DTYPES[0]; later arms swap the weights underneath it.
    exp = base.with_(gen_dtype=DTYPES[0])
    bench = Bench.build(exp)

    cost, arms = {}, {}
    for dtype in DTYPES:
        print(f"--- {dtype} ---", flush=True)
        if bench.dp_model.dtype != dtype:
            drop_weights(bench, dtype)
        row = load_and_measure(bench, dtype)
        rows, seconds = generate(bench, exp, dp_cfg, queries, dtype)
        row["peak_gb"] = round(gb(torch.cuda.max_memory_allocated()), 1)
        row["reserved_gb"] = round(gb(torch.cuda.max_memory_reserved()), 1)
        row["gpu_total_gb"] = round(gb(total), 1)
        if seconds:
            row["seconds_per_query"] = round(sum(seconds) / len(seconds), 1)
        cost[dtype] = row
        arms[dtype] = rows
        print(flush=True)

    print("=== cost ===")
    print(f"{'dtype':>10} | {'loaded as':>16} | {'weights':>8} | {'peak':>7} | "
          f"{'s/query':>7}")
    print("-" * 62)
    for dtype, r in cost.items():
        print(f"{dtype:>10} | {r['loaded_dtype']:>16} | "
              f"{r['weights_gb']:>7.1f}G | {r['peak_gb']:>6.1f}G | "
              f"{r.get('seconds_per_query', 0):>7.1f}")

    first, second = DTYPES
    if cost[first].get("seconds_per_query") and cost[second].get("seconds_per_query"):
        speedup = cost[first]["seconds_per_query"] / cost[second]["seconds_per_query"]
        peak_ratio = cost[second]["peak_gb"] / cost[first]["peak_gb"]
        print(f"\n  {second} runs at {speedup:.2f}x the speed of {first} "
              f"and {peak_ratio:.2f}x its peak memory")

    verdict = compare(arms)

    out = run_record.write(
        "probe_dtype", exp,
        metrics={
            "arms": DTYPES,
            "cost": cost,
            "verdict": verdict,
            "n_queries": N_QUERIES,
            "epsilon": EPSILON,
            "query_selection": f"longest prompts in {SOURCE}",
            "note": (
                "params.gen_dtype names the arm the bench was BUILT at; every "
                "per_item row carries the dtype it was actually generated at. "
                "Cost is comparable with probe_qwen.json / probe_gemma.json "
                "(same fields, same worst-case prompts); the emitted/norag/paid "
                "columns are comparable with env_equivalence_*.json."
            ),
        },
        per_item=[r for rows in arms.values() for r in rows],
        filename=f"probe_dtype_{model_id.split('/')[-1]}",
    )
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
