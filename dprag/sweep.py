"""Running one routed configuration over a query set, and surviving the attempt.

Stage 3.2's main phase is sixteen runs of two hundred queries -- fifteen hours or
so on rented hardware. Three things follow from that number, and this module
exists for them.

**A crash must not cost the whole run.** Progress is checkpointed per query, so a
process that dies at query 137 resumes at 138. The project has already lost half an
hour twice to a path bug that surfaced only at the final write; at fifteen hours the
same class of mistake is unaffordable.

**Retrieval happens once per query, not once per strategy.** DP retrieval is
stochastic, so calling it per configuration would hand each one a different document
set and confound "strategy" with "different evidence".

**Nothing is scored here.** The sweep records generated text and per-position
traces; ROUGE-L, BERTScore and everything else run afterwards from the saved output.
That separation is deliberate: BERTScore currently cannot load its model at all, and
a scoring dependency must never be able to destroy a fifteen-hour generation run.
Scoring is post hoc anyway -- it needs the text and the reference and nothing else.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from . import paths, run_record, trace
from .router import Router
from .strategies import PrefilterDecision, Strategy

# The baseline is plain DPRAG, produced by routing with a strategy that never
# agrees rather than by a second code path. `test_router.py` pins that this
# reproduces dp_chat token for token, so the baseline and the treatment cannot
# drift apart in setup (ADR 0003).
NEVER_AGREE: Strategy = lambda rag, norag: PrefilterDecision(
    False, int(norag.argmax()), 0.0
)


def _checkpoint_path(filename: str) -> Path:
    return paths.results_dir() / f".{filename}.checkpoint.json"


def _load_checkpoint(filename: str) -> list[dict[str, Any]]:
    path = _checkpoint_path(filename)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A checkpoint written during a kill can be truncated. Losing it costs
        # recomputation; trusting it costs a corrupt result file.
        print(f"  checkpoint {path.name} unreadable, starting over")
        return []


def _save_checkpoint(filename: str, rows: list[dict[str, Any]]) -> None:
    path = _checkpoint_path(filename)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    tmp.replace(path)      # atomic: a kill mid-write cannot truncate the real file


def routed_sweep(
    bench,
    exp,
    strategies: dict[str, Strategy],
    dp_config,
    *,
    name: str,
    filename: str,
    queries: list | None = None,
    metrics: dict[str, Any] | None = None,
    checkpoint_every: int = 10,
    on_query: Callable[[int, dict], None] | None = None,
) -> Path:
    """Route every query under every strategy, record everything, score nothing.

    Returns the path of the written RunRecord. If one already exists the sweep is
    skipped, which is what makes a phase script safe to re-run after a partial
    failure: completed runs are not repeated.
    """
    out_path = paths.results_dir() / f"{filename}.json"
    if out_path.exists():
        print(f"  {out_path.name} exists, skipping")
        return out_path

    store = bench.engine.pup_vector_store
    tokenizer = bench.dp_model.tokenizer
    questions = queries if queries is not None else bench.queries()

    rows = _load_checkpoint(filename)
    done = {r["query"] for r in rows}
    if rows:
        print(f"  resuming from checkpoint: {len(rows)} queries already done")

    zero_docs = 0
    for i, q in enumerate(questions):
        question = q.query if hasattr(q, "query") else q
        if question in done:
            continue

        # Make retrieval a function of (query, seed) rather than of how many
        # retrievals have happened. Without this, each configuration in a phase
        # runs its own sweep, the generator advances between them, and the second
        # one to ask about a query draws different documents from the first --
        # measured at 0.234 mean Jaccard overlap, with zero of 173 queries seeing
        # the same set. Every quality comparison then confounds the strategy with
        # the evidence, which is the one thing the phase must not do.
        store.reseed_for(question)
        documents = store.pup_retrieve(question)
        if not documents:
            # DP retrieval legitimately returning nothing: the RAG instance
            # collapses onto NoRAG and every strategy fires trivially, so these
            # are counted and excluded rather than averaged in (CONTEXT.md).
            zero_docs += 1
            continue

        row: dict[str, Any] = {
            "query": question,
            "n_documents": len(documents),
            "docs": trace.retrieval_trace(store, question, documents),
            "by_strategy": {},
        }
        for strategy_name, strategy in strategies.items():
            # Re-seed per generation so a strategy's trajectory does not depend on
            # how much randomness the strategies before it happened to consume.
            torch.manual_seed(exp.seed)
            began = time.time()
            result = Router(bench.dp_model, strategy, dp_config).generate(
                documents, question
            )
            elapsed = time.time() - began

            marks, _, _ = trace_marks(tokenizer, result, exp)
            record = trace.strategy_trace(result, marks, elapsed)
            trace.check(record)     # a violation means the router is broken
            row["by_strategy"][strategy_name] = record

        rows.append(row)
        # Printed every query, not every checkpoint. A twenty-hour phase whose log
        # stays blank for ninety minutes is indistinguishable from a hung one, and
        # this project has already spent an hour and a half watching a silent file.
        # Run with `python -u` or the buffering hides it anyway.
        triggers = " ".join(
            "%s=%.2f" % (n, r["trigger_rate"]) for n, r in row["by_strategy"].items()
        )
        print("[%3d/%3d] k=%2d  %.1fs  %s"
              % (len(rows), len(questions), row["n_documents"],
                 sum(r["seconds"] for r in row["by_strategy"].values()), triggers),
              flush=True)
        if on_query:
            on_query(i, row)
        if len(rows) % checkpoint_every == 0:
            _save_checkpoint(filename, rows)

    _save_checkpoint(filename, rows)

    summary = {
        "n_queries_scored": len(rows),
        "n_zero_document_queries": zero_docs,
        "strategies": list(strategies),
        "scored": False,
        "note": (
            "Generation only. Quality, grounding and epsilon-productivity are "
            "computed from this file by experiments/stage3_score.py, so a scoring "
            "failure cannot cost a generation run."
        ),
        **(metrics or {}),
    }
    path = run_record.write(name, exp, metrics=summary, per_item=rows, filename=filename)
    _checkpoint_path(filename).unlink(missing_ok=True)
    print(f"  saved -> {path.name}  ({len(rows)} queries, {zero_docs} with 0 docs)")
    return path


def trace_marks(tokenizer, result, exp):
    """Clinical marks over a routed result, using the configured vocabulary.

    Kept here so every sweep flags positions the same way and with the same
    threshold; `vocab_min_count` is in ExperimentConfig precisely because the
    medical rates move with it.
    """
    from .medical_flags import build_corpus_vocabulary, flag_medical_tokens

    vocabulary = build_corpus_vocabulary(exp.vocab_min_count)
    return flag_medical_tokens(tokenizer, result.emitted, vocabulary)
