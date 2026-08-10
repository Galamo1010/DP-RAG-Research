# Verifying the Stage 3.1 router: two silent bugs and how they surfaced

Date: 2026-08-09
Related: [ADR 0003](../adr/0003-stage3-scope-and-routing.md), [Stage 3 spec](../specs/stage3-router-and-comparisons.md)

The router replaces `generate()` with a hand-written autoregressive loop, because
deciding per position whether to run the k+1 aggregation is not something
`generate()` can express. Reimplementing a decoding loop means reimplementing
everything `generate()` quietly does, and this is a record of what was missed,
how each omission surfaced, and which measurements had to be withdrawn.

The through-line: **none of these bugs raised an exception.** Every symptom was
"the answer is a bit worse", and DP noise makes answers worse by design, so the
two were indistinguishable by inspection. That is what made the diagnosis take
three rounds instead of one.

## What was built

Two batches over one shared emitted sequence. The pre-filter batch (2 rows: NoRAG
and RAG) advances every step, because a decision is needed at every step. The
DPRAG batch (k+1 rows) advances only when the strategy disagrees and an
aggregation is actually required; tokens emitted while it is idle accumulate in a
backlog and are fed in a single forward when it is next needed.

Unit tests covered this against a fake model on CPU: routing decisions, epsilon
accounting, backlog bookkeeping, and two equivalence properties (a strategy that
never agrees pays everywhere; one that always agrees pays nowhere and emits the
NoRAG stream). Those tests passed throughout, including while both bugs were live.

## Bug 1 — position_ids not derived from the attention mask

**What.** The loop called `model(...)` directly, bypassing
`prepare_inputs_for_generation`, which is where `generate()` computes
`position_ids` from the attention mask. The batches are left-padded and the rows
differ enormously in length — the NoRAG prompt is a few dozen tokens, the RAG
prompt carries ten concatenated documents — so a model falling back to a shared
`arange` places the padded rows at the wrong rotary positions.

**How it surfaced.** By reasoning, before any GPU run. The fake model ignores
`position_ids` entirely, so no test could have caught it; writing the smoke
prompted the question "what else does generate() do that I am not doing?".

**Fix.** Compute `positions = cumsum(mask) - 1`, masked to 1 where the mask is 0,
sliced to the new tokens — the same derivation `generate()` uses. A test now pins
that a padded row and an unpadded row receive different position vectors.

## Bug 2 — the sampling warpers were never applied

This is the substantial one.

**Symptom.** Routed output was fluent-looking gibberish assembled from rare and
non-English tokens: `ContentType`, `pragma`, `gameState`, `πισ`, `őség`, `запр`,
punctuated by long runs of commas.

**First wrong conclusion.** The gibberish was read as a *trajectory feedback
loop*: a DP-sampled token lands, the context degrades, the two instances diverge,
more positions get paid, more noise arrives. A controlled comparison seemed to
support it — same query, same documents, NoRAG-driven versus routed — giving a
trigger-rate gap of −0.137 for strategy A and −0.195 for strategy B. Those numbers
were reported as a finding. **They were an artifact and have been withdrawn.**

**The contradiction that broke it.** The feedback story predicts that a larger
budget means less noise, a cleaner context, and a narrower gap. Running
eps_gen ∈ {10, 40} — clipping 0.158 against 0.435, roughly threefold — the gap did
not narrow; it widened slightly. Worse, the generated text at the two budgets was
*byte-identical* for the first sample. Changing the noise scale threefold cannot
leave the output unchanged if noise is what shapes it.

**The decisive diagnostic.** A strategy that never agrees pays at every position,
which *is* plain DPRAG. Running that alongside `dp_model.dp_chat` puts the same
aggregation, the same config and the same documents through two different
machines — the router's loop and upstream `generate()`. They diverged: the router
produced gibberish, `dp_chat` produced English. That localised the fault to the
paid path and ruled out the prompts, the retrieval, the cache and the aggregator,
all of which are shared.

**Cause.** `GenerationConfig` defaults to `top_k=50`, and `DPGenerationConfig`
inherits it, so `generate()` applies `TopKLogitsWarper(50)` — sampling only from
the fifty highest-scoring tokens, which are the ones the documents support. The
hand-written loop applied temperature and nothing else, sampling the full
vocabulary.

**Why that is catastrophic here, and why it hid.** The aggregated score spans
roughly ±(k × clipping) ≈ ±1.58. Across a 128k vocabulary a softmax over that
range concentrates almost nothing: measured top-1 probability 2.6e-05, against
7.8e-06 for a uniform draw. Sampling was therefore close to picking a token at
random, and most of a 128k vocabulary is rare or non-English — hence the
particular flavour of the gibberish. It also explains the byte-identical outputs
across budgets: the distribution is near-uniform at both, so the same RNG draw
lands in the same place. After the fix, top-1 probability is 0.0200 over 50
candidates.

The unit tests could not have caught it either: the fake generation config had no
`top_k` field at all. The gap in the test double sat exactly where the bug was.

**Fix.** A `sampling_warpers()` function builds the warper list from the config —
temperature, top_k, top_p, min_p, typical_p — in `generate()`'s order, and the
paid path runs the aggregated scores through it before sampling. Temperature moved
into the warpers, so the manual division was removed to avoid applying it twice.
Six tests now pin the behaviour, including one that sets `top_k=1` to make the
paid path deterministic and assert it emits the aggregated argmax.

**Verification.** With the fix, the never-agreeing router and `dp_chat` produce
byte-identical text. That is the strongest available check: the same input through
two independent implementations of the same decoding rule.

## Corrected measurements

Nine queries with documents, 64 tokens, k = 10, Llama-3.1-8B. Same query and same
retrieved documents on both sides; only the driver differs.

| strategy | NoRAG-driven | routed, eps=10 | routed, eps=40 |
|---|---|---|---|
| A | 0.868 | 0.842 (−0.026) | 0.851 (−0.017) |
| B (k=20, τ=0.7) | 0.705 | 0.569 (−0.136) | 0.628 (−0.078) |

The NoRAG-driven column reproduces Stage 2.4 (0.875 and 0.745), which is what
makes the comparison trustworthy: the setup matches the earlier measurement, so
the remaining difference is attributable to the trajectory.

**What this licenses.**

- **Strategy A is close to trajectory-independent.** A 2–3 point drop means Stage
  2.4's measurement of A stands essentially as reported.
- **Strategy B is not.** It loses 8–14 points, which is coherent: B compares an
  entire top-k set, and set membership is more sensitive to context quality than a
  single argmax. Stage 2's B figures are optimistic and must be labelled as
  spectator measurements when reported.
- **A larger budget raises the trigger rate** (B: 0.569 → 0.628). Less noise, a
  cleaner context, more agreement, fewer paid positions. Epsilon savings are
  therefore non-linear in epsilon, which Stage 5's Pareto analysis should expect.
- **Strategy A routes around DP at 84–85% of positions**, against the 87% ceiling
  Stage 1.2 measured. This is the project's first measurement of the contribution
  actually running rather than being simulated alongside a run that paid anyway.

**Status of these numbers.** A smoke: 9 queries at 64 tokens. Stage 3 measures at
200 queries and 128 tokens. Expect B's gap to grow there, since a longer sequence
gives trajectory divergence more room to accumulate.

## What to do differently

**Put the upstream-equivalence check first.** A degenerate configuration that
collapses the new machinery onto the old one — here, a strategy that never agrees
— gives a byte-for-byte comparison against a known-good implementation. It was
run last and found the bug immediately. Run first, it would have saved three
rounds of GPU time and one withdrawn finding.

**Test doubles inherit the author's blind spots.** The fake config omitted
`top_k`, so the tests asserted correctness of a loop the real config would never
run. When faking a collaborator whose defaults matter, copy the defaults rather
than the fields currently in use.

**A prediction that fails is worth more than one that succeeds.** The feedback
hypothesis was wrong, but testing it produced the byte-identical-across-budgets
observation, and that was the thread that unravelled the real cause. The
diagnostic value came from the prediction being sharp enough to be contradicted.
