# ADR 0010 — Retrieval is seeded per query, not per run

Date: 2026-08-31
Status: Accepted. Amends [ADR 0002](0002-seeded-retrieval.md). Invalidates the first
three Phase 2 runs.

## Context

ADR 0002 gave `PUPVectorStore` store-local generators so a run could be reproduced.
That solves reproducibility *within* a run and says nothing about comparability
*between* runs — a distinction this project did not notice until it had spent six
hours on the wrong side of it.

Phase 2 runs one sweep per configuration, so each gets its own result file and can
be skipped or resumed independently. Each sweep calls `pup_retrieve` for every
query, and the generators advance with every call. The second configuration to ask
about a query therefore draws a **different document set** from the first.

Measured across the three completed runs at eps=10:

| | |
|---|---|
| Queries where baseline and A retrieved the same documents | **0 of 173** |
| Mean Jaccard overlap of the two document sets | **0.234** |

Every quality comparison between configurations was thus a mixture of *the
strategy* and *the evidence* — the one confound the phase exists to exclude, and
one `sweep.py`'s own docstring claims to prevent. That claim was true within a
single sweep, where all strategies share one retrieval, which is how Stage 2.5 and
Phase 1 were written. Splitting Phase 2 into one sweep per configuration, for
resumability, silently broke it.

## Decision

`PUPVectorStore.reseed_for(query)` sets both generators from
`sha256(f"{seed}:{query}")`, and `sweep.routed_sweep` calls it before every
retrieval. Retrieval becomes a function of `(query, seed)` alone: identical across
configurations, across runs, and across processes.

`hash()` is not used — Python salts it per process, so the guarantee would hold
only inside one run, which is precisely the case that already worked.

**The alternative was to put every configuration back into one sweep**, as Stage
2.5 does. That restores shared retrieval but gives up per-configuration result
files, skip-on-exists, and per-run checkpointing — the things that make a
twenty-three hour phase interruptible. Seeding per query keeps both.

**This does not touch the privacy guarantee.** ADR 0002's argument applies
unchanged: ε-DP is a property of the mechanism's output *distribution* over its own
randomness, and fixing which draw is observed does not alter that distribution. The
same caveat also applies unchanged — a seeded output must never be presented as a
DP-protected release.

## Consequences

**The first three Phase 2 runs are void.** baseline, A and B_k20_t0.9 at eps=10 —
about six hours of A100 — are moved to `results/archive/` with an
`_orderdependent` suffix. They are kept because their *within-run* figures
(trigger rate, epsilon usage, epsilon productivity) do not depend on cross-run
comparison and remain readable, but no quality comparison may be drawn from them.

**One number from them is worth carrying forward** as an indication, since it
survives the confound in direction if not in magnitude: plain DPRAG scored ROUGE-L
0.1199 at `max_retrieve=40` against 0.0893 at 10. The +36% advantage the pre-filter
appeared to hold in the Stage 2.5 pilot came from the baseline being crippled by
near-uniform sampling at k=10, not from the pre-filter being better. Expect the
Phase 2 claim to be non-inferiority — the proposal's original wording — rather than
superiority. The re-run will settle it properly.

**Two tests pin the behaviour.** One asserts that a store which has already served
other queries retrieves the same documents for a given query as a fresh one. The
other asserts the *opposite* holds without reseeding, so that if sequential draws
ever become order-independent the test fails loudly rather than passing vacuously.

**The general lesson is worth keeping.** Seeding for reproducibility and seeding
for comparability are different requirements, and satisfying the first hides how
easily the second breaks. Any experiment that draws randomness per item, and
compares configurations item by item, has to key that randomness to the item.
