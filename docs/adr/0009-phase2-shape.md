# ADR 0009 — What Phase 2 actually runs

Date: 2026-08-30
Status: Accepted. Revises the phase-2 shape in [ADR 0003](0003-stage3-scope-and-routing.md) §2.

## Context

ADR 0003 planned phase 2 as `{baseline, A, 2 finalists} x eps in {5,10,20,40} x 200
queries` and costed it at roughly eleven hours, using ~10 s per generation measured
at `max_retrieve=10`.

Two things have changed since.

**Generation costs four times what that assumed.** Measured over 20 queries at
`max_retrieve=40` ([ADR 0008](0008-max-retrieve-40.md)): **40.8 s per generation**.
The original sixteen runs would now cost about 33 hours rather than 11.

**Two Pareto-optimal finalists is the wrong selection rule for what this phase has
to show.** ADR 0007 redefined the third comparison as a trade-off rather than a
bound, and a trade-off needs a spread. Two points chosen for being near the
frontier can sit almost on top of each other and reveal no shape at all — which is
exactly what happened when strategy A and B(k=50, tau=0.5) turned out to be
indistinguishable in quality (0.1215 against 0.1192, needing 614 queries to
separate).

Stage 2.5 happened to run a spread instead, and it produced the clearest result the
project has: quality rose **monotonically** as the paid fraction fell, across four
configurations from 85.6% paid down to 4.7%. That dose-response is the evidence for
the mechanism, and it exists only because the configurations spanned the range.

## Decision

**Five configurations**: `baseline`, `A`, and three Strategy B settings spanning
strictness rather than clustering on the frontier —

| | required overlap | trigger at max_retrieve=10 |
|---|---|---|
| B (k=20, tau=0.9) | 19/20 | 14.4% |
| B (k=20, tau=0.7) | 17/20 | 59.1% |
| B (k=50, tau=0.5) | 34/50 | 95.3% |

**Two budgets first**: `eps_total in {10, 40}`, the ends of the proposal's grid.
Ten runs, about 21 hours.

### Why the ends rather than the full grid

Every existing measurement in this project sits at 10 or 40, so these are the
values that can be compared against what is already known. They also bracket the
range, which is what a trade-off needs.

The interior points are the least likely to carry information. At eps=40 **none of
the four configurations were distinguishable in quality** — all six pairwise
comparisons overlapped — while at eps=10 five of six separated. Whatever structure
exists appears at the noisy end, and 5 and 20 sit between two points that already
bound it.

This is a deferral, not a cancellation: `EPSILON_GRID` in
`experiments/stage3_2_main.py` is one line, and completed runs are skipped, so
filling the grid in later re-runs nothing.

### Why strictness is read off the measured trigger rate, not tau

`tau` reads far looser than it is — at k=20, tau=0.9 demands 19 of 20 tokens — and
`k` changes behaviour independently of it: B(k=50, tau=0.5) requires a *lower*
fraction than B(k=20, tau=0.7) yet triggers far more often. The strictness ordering
is empirical.

## Consequences

**A deviation from the proposal to declare.** The proposal specifies
`eps_total in {5,10,20,40}`. Two of the four are deferred, for the reason above and
under the cost control the proposal itself authorises. The report should say which
budgets were run and why, not present the grid as complete.

**The three configurations were chosen on stale evidence.** Those trigger rates
were measured at `max_retrieve=10`, before ADR 0008. That change made the DP
aggregation markedly sharper — Stage 1.2's *sampled* consistency went from 0.11 to
0.75 on it alone — so the three may no longer span the range at 40. **Phase 1
exists to check that before twenty-one hours are committed**, and the selection
moves if the spread has collapsed.

**Filling in eps 5 and 20 costs another 21 hours.** Decide it against what the ends
show rather than in advance.
