"""Stage 2.1 -- RAG/NoRAG pre-filter decision functions (策略A / 策略B).

Pure, stateless and GPU-free. Each function takes the two logit vectors for ONE
autoregressive position and decides whether that position is "document
independent" (consistent). When it is, the router may emit the NoRAG argmax
token and spend ZERO privacy budget; when it is not, the position is handed to
the unmodified DPRAG aggregation.

    策略A (strategy_a):  argmax(logit_RAG) == argmax(logit_NoRAG)
    策略B (strategy_b):  Jaccard(top-k(logit_RAG), top-k(logit_NoRAG)) >= tau   (Eq1)

Both emit the NoRAG argmax when they fire, exactly as the proposal specifies.

Interchangeable interface -- the router only ever needs

    (logit_rag, logit_norag) -> PrefilterDecision

`make_strategy_b(k, tau)` binds B's hyper-parameters so its signature matches A's,
letting the router swap strategies by replacing a function pointer.

TWO PROPERTIES WORTH KNOWING (both are covered by test_prefilter_strategies.py):

1. TEMPERATURE INVARIANCE. Dividing logits by T > 0 is monotonic, so it changes
   neither the argmax nor the top-k index set. Both strategies are therefore
   invariant to sampling temperature at a fixed position/context. Temperature
   only changes which trajectory gets sampled, i.e. which contexts are visited.

2. B IS NOT A SUBSET OF A. Two distributions can share the same top-k *set* while
   ordering it differently, so strategy B (even at tau = 1.0) can fire where
   strategy A does not. B's coverage is therefore not upper-bounded by A's.

TIE-BREAK: argmax ties resolve to the lowest token index (torch.argmax's
documented behaviour). Ties in real LLM logits are vanishingly rare, but the
behaviour is pinned by a test so the routing decision stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class PrefilterDecision:
    """One position's routing decision.

    consistent: True  -> document-independent, emit `token_id`, spend no epsilon.
                False -> hand the position to the DPRAG aggregation.
    token_id:   the NoRAG argmax (what to emit when consistent).
    score:      diagnostic. Strategy A reports 1.0/0.0; strategy B reports the
                Jaccard similarity, which is what the (k, tau) sweep analyses.
    """
    consistent: bool
    token_id: int
    score: float


# The signature every strategy must expose so the router can swap them freely.
Strategy = Callable[[Tensor, Tensor], PrefilterDecision]


def _validate(logit_rag: Tensor, logit_norag: Tensor) -> None:
    if logit_rag.ndim != 1 or logit_norag.ndim != 1:
        raise ValueError(
            f"expected 1-D logit vectors for a single position, got "
            f"{tuple(logit_rag.shape)} and {tuple(logit_norag.shape)}"
        )
    if logit_rag.shape != logit_norag.shape:
        raise ValueError(
            f"logit vectors must share a vocabulary size, got "
            f"{logit_rag.shape[0]} vs {logit_norag.shape[0]}"
        )


def strategy_a(logit_rag: Tensor, logit_norag: Tensor) -> PrefilterDecision:
    """策略A -- strict token comparison."""
    _validate(logit_rag, logit_norag)
    argmax_norag = int(torch.argmax(logit_norag))
    consistent = int(torch.argmax(logit_rag)) == argmax_norag
    return PrefilterDecision(
        consistent=consistent,
        token_id=argmax_norag,
        score=1.0 if consistent else 0.0,
    )


def jaccard_topk(logit_rag: Tensor, logit_norag: Tensor, k: int) -> float:
    """Jaccard similarity of the two top-k token sets (proposal Eq1).

    k is clamped to the vocabulary size, so an over-large k degrades to
    "compare the whole vocabulary", which always yields 1.0.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k_eff = min(k, logit_rag.shape[0])
    set_rag = set(torch.topk(logit_rag, k_eff).indices.tolist())
    set_norag = set(torch.topk(logit_norag, k_eff).indices.tolist())
    return len(set_rag & set_norag) / len(set_rag | set_norag)


def strategy_b(
    logit_rag: Tensor, logit_norag: Tensor, k: int, tau: float
) -> PrefilterDecision:
    """策略B -- top-k logit Jaccard comparison."""
    _validate(logit_rag, logit_norag)
    similarity = jaccard_topk(logit_rag, logit_norag, k)
    return PrefilterDecision(
        consistent=similarity >= tau,
        token_id=int(torch.argmax(logit_norag)),
        score=similarity,
    )


def make_strategy_b(k: int, tau: float) -> Strategy:
    """Bind (k, tau) so strategy B matches strategy A's signature."""
    def _bound(logit_rag: Tensor, logit_norag: Tensor) -> PrefilterDecision:
        return strategy_b(logit_rag, logit_norag, k=k, tau=tau)

    _bound.__name__ = f"strategy_b_k{k}_tau{tau}"
    return _bound


def min_overlap_for_tau(k: int, tau: float) -> float:
    """Smallest top-k overlap m that satisfies Jaccard >= tau.

    Both sets have size k, so |union| = 2k - m and Jaccard = m / (2k - m).
    Solving m / (2k - m) >= tau gives m >= 2*k*tau / (1 + tau). Useful for
    reporting what a given tau actually demands (e.g. k=20, tau=0.9 needs
    m >= 18.9, i.e. 19 of the 20 tokens must match).
    """
    return 2 * k * tau / (1 + tau)
