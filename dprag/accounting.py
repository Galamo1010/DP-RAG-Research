"""Composing the per-token privacy budget into an end-to-end epsilon.

This lived inside `router.py` as a private helper, because for a long time the
router was the only thing that needed it: compose `token_epsilon` over the paid
positions, subtract from the budget, done.

Stage 4 needs the same arithmetic from two more places. 4.1 has to add the
retrieval layer, which the router never accounted for. 4.2 has to evaluate the
composition at every position rather than once at the end. Both are analysis
scripts, and `experiments/` may not reach into a library module's private names
(CONTEXT.md: dependencies run experiments -> dprag, and the seam has to be a
public one). Re-deriving PLD composition in an analysis script was the
alternative, and it is the shape this project's characteristic bug takes: two
copies of the same arithmetic, one of which is quietly wrong while both produce
plausible numbers.

WHAT THE ROUTER DOES NOT COUNT
------------------------------
`RoutedResult.epsilon_usage` is the **generation layer alone**. DP retrieval
costs `eps_retrieval` once per query (CONTEXT.md, proposal Eq3) and no routed run
has ever included it, so every epsilon figure reported so far is smaller than the
end-to-end quantity the proposal's Stage 4.1 asks for. `two_layer_epsilon` is
what closes that gap, and it composes rather than adds: epsilon values from two
mechanisms do not sum under PLD accounting except in the loose basic-composition
bound.
"""

from __future__ import annotations

from functools import lru_cache


def _single(epsilon: float):
    from dp_accounting.pld.common import DifferentialPrivacyParameters
    from dp_accounting.pld.privacy_loss_distribution import from_privacy_parameters

    return from_privacy_parameters(DifferentialPrivacyParameters(epsilon=epsilon))


@lru_cache(maxsize=None)
def composed_epsilon(token_epsilon: float, steps: int, delta: float) -> float:
    """Epsilon after composing `steps` copies of `token_epsilon`, via PLD.

    Cached because a 200-query run composes the same handful of step counts over
    and over, and each composition is a loop over PLD objects. Stage 4.2 leans on
    the cache much harder than the router does: it asks for every step count from
    0 to 128, for every query, and the second query onwards is free.
    """
    if steps <= 0:
        return 0.0
    from dp_accounting.pld.privacy_loss_distribution import identity

    pld = identity()
    single = _single(token_epsilon)
    for _ in range(steps):
        pld = pld.compose(single)
    return pld.get_epsilon_for_delta(delta)


@lru_cache(maxsize=None)
def two_layer_epsilon(
    token_epsilon: float, steps: int, delta: float, eps_retrieval: float
) -> float:
    """End-to-end epsilon: DP retrieval composed with `steps` paid generation steps.

    The proposal's Eq3 accounting (CONTEXT.md, "two-layer accounting"): retrieval
    is one exponential-mechanism draw per query costing `eps_retrieval`, and
    generation costs `token_epsilon` at each position that took the paid path.

    Composed, not summed. Summing is the basic-composition bound and is loose
    exactly where this project lives -- many small epsilons -- so it would
    overstate the cost and understate the saving. Reporting a larger number is
    the safe direction to be wrong in, but it is still wrong, and the proposal
    names PLD specifically.
    """
    if steps <= 0:
        return _single(eps_retrieval).get_epsilon_for_delta(delta) if eps_retrieval else 0.0

    from dp_accounting.pld.privacy_loss_distribution import identity

    pld = identity()
    single = _single(token_epsilon)
    for _ in range(steps):
        pld = pld.compose(single)
    if eps_retrieval:
        pld = pld.compose(_single(eps_retrieval))
    return pld.get_epsilon_for_delta(delta)


def cumulative_epsilon(
    paid_positions: list[int] | set[int],
    n_positions: int,
    token_epsilon: float,
    delta: float,
    eps_retrieval: float = 0.0,
) -> list[float]:
    """Epsilon consumed from the start of generation up to each position.

    Stage 4.2's y-axis: `out[t]` is the budget spent after emitting position `t`,
    so a free position leaves the curve flat and a paid one steps it up. Returned
    for every position rather than only the paid ones, because the flat stretches
    are the point -- a curve sampled only where it moves is a straight line again.

    `eps_retrieval` shifts the whole curve up by the query's one retrieval draw
    when given, so the curve starts at the retrieval cost rather than at zero.
    Pass 0.0 to plot the generation layer alone.
    """
    paid = set(paid_positions)
    out: list[float] = []
    spent = 0
    for t in range(n_positions):
        if t in paid:
            spent += 1
        out.append(two_layer_epsilon(token_epsilon, spent, delta, eps_retrieval))
    return out
