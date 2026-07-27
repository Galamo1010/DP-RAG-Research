# ADR 0002 — Seed retrieval and generation, without weakening the DP claim

Date: 2026-07-25
Status: Accepted

## Context

`PUPVectorStore.pup_retrieve` draws twice from a random source: the
exponential-mechanism score threshold (`np.random.choice`) and the truncation to
`max_retrieve` (`random.sample`). Both used the global RNG with no seed, so two
runs of the same experiment retrieved different documents.

This has already cost real time. During the Stage 2.3 temperature sweep we wanted
to add T=1.0 to four temperatures already measured. We could not: a fill-in run
would have drawn a different document set, so that row would not have been
comparable with the others. All five temperatures were re-run instead, about 30
minutes of A100 time to obtain one row.

The cost grows. Stage 3 runs 14 configurations across 3 models and 4 ε_total
values. Comparing Strategy A against Strategy B is only meaningful if both see the
same retrieved documents — otherwise the difference between them is confounded
with the difference between their document sets.

The obvious objection: **differential privacy comes from randomness. Does fixing
the seed destroy the guarantee?**

## Decision

Add `seed` to `ExperimentConfig`. `Bench.build` applies it to both sources of
randomness: retrieval, through `PUPVectorStoreConfig(seed=...)`, and generation
sampling, through `torch.manual_seed`. A seeded `PUPVectorStore` uses store-local
generators (`np.random.default_rng`, `random.Random`) so it never perturbs global
RNG state that other code may depend on. `seed=None` preserves the original
global-RNG behaviour exactly.

## On the privacy question

An ε-DP mechanism is a claim about a **distribution**: for adjacent datasets D and
D′, `Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D′) ∈ S]`, where the probability is over the
mechanism's internal randomness. Seeding does not alter that distribution. It
fixes *which draw* we observe, the way reporting a specific sample does not change
the distribution it came from. The exponential mechanism in `pup_retrieve` is
unchanged: same utility function, same ε, same sampling distribution.

What seeding *does* change is who can predict the draw. An adversary who knows the
seed can reproduce the output exactly, so the observed release carries no
protection for that adversary. The distinction to hold onto:

- **Measuring the mechanism** (what this project does): seeding is correct and
  standard, and it is what makes results reproducible.
- **Releasing an output** (deployment): the seed must be unpredictable, or the
  guarantee is vacuous against anyone who knows it.

The report must therefore state that seeds are experiment-control only, and must
never present a fixed-seed output as a DP-protected release. This is written into
the docstrings of `PUPVectorStore` and `ExperimentConfig.seed` so the caveat sits
next to the code rather than only in a document.

## Consequences

**Single configurations become re-runnable.** A missing row can be filled in
without redoing a sweep — exactly the operation that was impossible in Stage 2.3.

**Cross-configuration comparison becomes controlled.** Strategy A and Strategy B
can be run against identical document sets, so a measured difference is
attributable to the strategy.

**Retrieval becomes testable.** `tests/test_prompts_and_seeding.py` can assert
reproducibility without a GPU. Those tests also pin two properties that matter:
different seeds still produce different draws (seeding did not collapse the
mechanism into something constant), and repeated retrieval within one seeded store
still varies (the mechanism stays random inside a run; it is the run as a whole
that is reproducible).

**The archived pre-seed results cannot be reproduced.** Runs in `results/archive/`
predate this change; `results/archive/README.md` records that they are kept as a
development record and are to be re-run rather than migrated.
