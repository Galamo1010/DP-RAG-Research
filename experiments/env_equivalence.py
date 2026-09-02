"""Does the same generation come out the same in a different environment.

gemma-4 needs transformers 5.x and a newer torch than the pinned environment
allows, so it runs from a second venv (ADR to follow). That buys a question: if
Llama's answers differ between the two environments, then gemma's numbers sit on a
different footing from Llama's and Qwen's, and the cross-model comparison can only
be read within each model.

The ε accounting is already known to agree -- `token_epsilon()` returns bit-identical
0.5902099609375 in both -- because dp-accounting is arithmetic and never touches the
model. Generation is the part that could differ, and the honest way to describe it
is to measure it rather than to declare it.

So this generates a few routed answers with the PRIMARY model, which both
environments can load, and records the token ids. Run it once per environment; the
second run finds the first run's file and diffs them.

Three things are compared, in the order they would break:

* **retrieved document indices** -- `reseed_for` derives its state from sha256 and
  numpy's default_rng, and numpy differs between the environments too. If retrieval
  disagrees, nothing downstream can be compared at all.
* **emitted token ids** -- the answer itself, position by position.
* **NoRAG argmax** -- the free path's opinion, which is what the pre-filter reads.

    # in each environment, from the repo root
    uv run python experiments/env_equivalence.py
    PYTHONPATH=. .venv-gemma/bin/python experiments/env_equivalence.py
"""

import glob
import re

import torch

from dprag import paths, run_record, trace
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router
from dprag.strategies import strategy_a

N_QUERIES = 3
EPSILON = 40.0
PREFIX = "env_equivalence"


def environment_tag() -> str:
    """A filename-safe fingerprint, so the two runs cannot overwrite each other."""
    import transformers
    parts = [f"torch{torch.__version__}", f"tf{transformers.__version__}"]
    return re.sub(r"[^0-9a-zA-Z.]+", "-", "_".join(parts))


def generate(bench, exp, dp_cfg) -> list[dict]:
    store = bench.engine.pup_vector_store
    router = Router(bench.dp_model, strategy_a, dp_cfg)
    rows = []
    for q in bench.queries(N_QUERIES):
        # Identical to dprag.sweep, so what is compared here is what a real run does.
        store.reseed_for(q.query)
        documents = store.pup_retrieve(q.query)
        if not documents:
            continue
        torch.manual_seed(exp.seed)
        result = router.generate(documents, q.query)
        rows.append({
            "query": q.query,
            "docs": trace.retrieval_trace(store, q.query, documents),
            "emitted": list(result.emitted),
            "norag_argmax": list(result.norag_argmax),
            "paid_positions": list(result.paid_positions),
            "text": bench.dp_model.tokenizer.decode(
                result.emitted, skip_special_tokens=True),
        })
        print(f"  k={len(documents):2d}  {len(result.emitted):3d} tokens", flush=True)
    return rows


def first_difference(a: list, b: list) -> int:
    """Index of the first mismatch, or -1 when one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def compare(this_tag: str, rows: list[dict]) -> None:
    others = [
        p for p in sorted(glob.glob(str(paths.results_dir() / f"{PREFIX}_*.json")))
        if this_tag not in p
    ]
    if not others:
        print("\nNo other environment's record found yet. Run this again from the")
        print("second environment and it will compare the two.")
        return

    for path in others:
        other = run_record.load(path)
        theirs = {r["query"]: r for r in other.per_item}
        print(f"\n=== against {other.path.stem} ===")
        for r in rows:
            mine, yours = r, theirs.get(r["query"])
            if yours is None:
                print(f"  query missing in the other record: {r['query'][:50]}")
                continue
            same_docs = [i for i, _ in mine["docs"]] == [i for i, _ in yours["docs"]]
            d_emit = first_difference(mine["emitted"], yours["emitted"])
            d_norag = first_difference(mine["norag_argmax"], yours["norag_argmax"])
            print(f"  docs {'same' if same_docs else 'DIFFER'} | "
                  f"emitted {'identical' if d_emit < 0 else f'diverge at {d_emit}'} "
                  f"({len(mine['emitted'])} vs {len(yours['emitted'])} tokens) | "
                  f"norag {'identical' if d_norag < 0 else f'diverge at {d_norag}'}")

        # A single verdict line, because this exists to answer one question.
        all_same = all(
            [i for i, _ in r["docs"]] == [i for i, _ in theirs[r["query"]]["docs"]]
            and first_difference(r["emitted"], theirs[r["query"]]["emitted"]) < 0
            for r in rows if r["query"] in theirs
        )
        print()
        if all_same:
            print("  VERDICT: byte-identical. The environments are interchangeable")
            print("  for generation, so gemma's results need no separate caveat.")
        else:
            print("  VERDICT: they differ. gemma's absolute numbers cannot be listed")
            print("  beside Llama's and Qwen's; only each model's improvement over")
            print("  its own baseline is comparable. Say so in the report.")


def main():
    exp = ExperimentConfig()
    tag = environment_tag()
    print(f"=== environment equivalence | model={exp.gen_model} | {tag} ===")
    print(f"{N_QUERIES} queries, max_retrieve={exp.max_retrieve}, eps={EPSILON:.0f}\n",
          flush=True)

    bench = Bench.build(exp)
    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=EPSILON, delta=exp.delta,
    )
    print(f"token_epsilon = {dp_cfg.token_epsilon()!r}", flush=True)

    rows = generate(bench, exp, dp_cfg)
    out = run_record.write(
        PREFIX, exp,
        metrics={
            "environment": tag,
            "token_epsilon": dp_cfg.token_epsilon(),
            "n_queries": len(rows),
            "note": (
                "Same model, same seeds, different library versions. Compared by "
                "running this in each environment; the second run diffs the first."
            ),
        },
        per_item=rows, filename=f"{PREFIX}_{tag}",
    )
    print(f"\nsaved -> {out.name}")
    compare(tag, rows)


if __name__ == "__main__":
    main()
