"""What one routed generation leaves behind, position by position.

Stage 2.5 was analysed three times and answered none of the questions put to it,
because the results file held only per-answer summaries. Whether the medical skip
rate survives correction for tokenisation, whether epsilon went on drug names or
on "Thanks for your query", and both halves of the proposal's Stage 4 all need to
know what happened at each of the 128 positions -- and that was computed during
the run and discarded.

The proposal is explicit about needing it. Stage 4.2 asks for "逐步驟記錄每個
token 位置的路徑決策（使用ε/不使用ε）與累積 ε_usage"; Stage 4.3 asks to
"擷取這些位置的 NoRAG argmax token 組成子序列". Neither is reconstructible from a
trigger rate.

This module owns that record so every experiment writes the same shape. It sits
below the experiments and above both `router` and `medical_flags`, depending on
each while neither depends on it -- routing has no business knowing about a
medical heuristic, and `run_record` stays a generic schema layer.

REDUNDANCY THAT EARNS ITS PLACE
-------------------------------
`norag_argmax` is stored for every position although at free positions it must
equal `emitted` -- that is what "skipped" means. Storing both makes the identity
checkable rather than assumed, and this project's characteristic bug is the kind
that produces a plausible wrong answer instead of an exception. `check` turns
that redundancy into an assertion; a violation means the router is broken.

`clinical` is stored dense, one entry per position, rather than as a list of the
few positions that are clinical. It is 99% nulls and costs a few hundred KB per
run. What it buys is that all three sequences have the same length, so a truncated
record cannot pass unnoticed.
"""

from __future__ import annotations

from typing import Any

from .medical_flags import TokenMark


def strategy_trace(result, marks: list[TokenMark | None], seconds: float) -> dict[str, Any]:
    """One strategy's outcome on one query, at full position resolution.

    `result` is a router.RoutedResult; `marks` comes from
    medical_flags.flag_medical_tokens over `result.emitted`.
    """
    n = len(result.emitted)
    return {
        "trigger_rate": result.trigger_rate,
        "epsilon_usage": result.epsilon_usage,
        "epsilon_savings": result.epsilon_savings,
        "seconds": round(seconds, 2),
        "text": result.text,
        "emitted": list(result.emitted),
        "norag_argmax": list(result.norag_argmax[:n]),
        "paid_positions": list(result.paid_positions),
        "clinical": [None if m is None else [m.kind, m.is_first] for m in marks[:n]],
    }


def retrieval_trace(store, query: str, documents: list[str]) -> list[list]:
    """Which corpus documents were retrieved, and how similar each one was.

    Indices rather than text: the corpus is `load_corpus(limit=n_docs,
    sample_seed=corpus_seed)` and both parameters are already in the run record,
    so an index resolves back to the document while costing a hundredth of the
    space. The similarity rides along because the question it answers -- did DP
    retrieval find anything relevant? -- otherwise costs a full re-embedding of
    the corpus to ask later.

    Encoding is deterministic and consumes no randomness, so calling this beside
    `pup_retrieve` does not disturb the retrieval draw.
    """
    if not documents:
        return []
    import torch

    scores = torch.mm(store.encode(query), store.embeddings().transpose(0, 1))[0]
    out = []
    for doc in documents:
        i = store.index[doc]
        out.append([int(i), round(float(scores[i]), 4)])
    return out


def check(record: dict[str, Any]) -> None:
    """Assert the invariants a correct routed run must satisfy.

    Cheap enough to run on every record as it is written. Each failure names a
    specific breakage rather than leaving a plausible-looking number in place.
    """
    n = len(record["emitted"])
    if not (len(record["norag_argmax"]) == len(record["clinical"]) == n):
        raise ValueError(
            f"trace lengths disagree: emitted={n} "
            f"norag_argmax={len(record['norag_argmax'])} "
            f"clinical={len(record['clinical'])} -- a sequence was truncated"
        )

    paid = set(record["paid_positions"])
    for i in range(n):
        if i not in paid and record["emitted"][i] != record["norag_argmax"][i]:
            raise ValueError(
                f"position {i} was free but emitted {record['emitted'][i]} while "
                f"NoRAG's argmax was {record['norag_argmax'][i]}; a skipped "
                "position must emit the NoRAG token, so the router is wrong"
            )

    if n:
        expected = 1.0 - len(paid) / n
        if abs(expected - record["trigger_rate"]) > 1e-9:
            raise ValueError(
                f"trigger_rate {record['trigger_rate']} disagrees with "
                f"{len(paid)}/{n} paid positions ({expected}) -- accounting is off"
            )


def free_positions(record: dict[str, Any]) -> list[int]:
    """Positions that took the free path. Stage 4.2's x-axis."""
    paid = set(record["paid_positions"])
    return [i for i in range(len(record["emitted"])) if i not in paid]


def norag_subsequence(record: dict[str, Any]) -> list[int]:
    """The NoRAG tokens emitted at agreeing positions.

    The proposal's Stage 4.3 attacks exactly this sequence: if the tokens chosen
    without spending epsilon still carry membership signal, the routing decision
    is leaking on its own.
    """
    return [record["emitted"][i] for i in free_positions(record)]
