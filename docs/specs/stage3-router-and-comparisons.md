# Spec — Stage 3: pre-filter router and the three comparisons

Status: ready-for-agent
Date: 2026-08-02
Related: [ADR 0003](../adr/0003-stage3-scope-and-routing.md), [CONTEXT.md](../../CONTEXT.md)

## Problem Statement

Everything measured so far has been hypothetical. Stage 1.2 ran plain DPRAG and
recorded, alongside it, what a pre-filter *would* have decided — giving a
consistency rate of 0.874. Stage 2.3 and 2.4 ran the two instances and recorded
what each strategy *would* have triggered on. In every one of those runs the
system still spent ε on all 128 positions. Nothing has ever actually skipped a
position.

So the project's central claim — that a RAG/NoRAG pre-filter cuts ε consumption
without hurting the answer — cannot yet be supported. There is no routed output
to score for quality, no ε_usage that differs from ε_budget, and no way to check
whether skipping is dangerous at clinically-loaded positions, because the text a
routed system would produce does not exist.

The first attempt at the Stage 2.5 safety check ran into exactly this. Lacking a
routed output, it searched for drug names in the per-step NoRAG argmax overlay —
a string no system emits — and matched artifacts like "youritisitis" rather than
medicine.

A second problem is cost. The full comparison grid the proposal specifies is
168 runs of 200 queries, which at the measured throughput is roughly 92 hours of
A100 time. That is not affordable, so the comparisons need a plan that keeps the
scientific claims intact on a fraction of the budget.

## Solution

Build the router: the module that, at each autoregressive step, asks the strategy
whether the RAG instance and the NoRAG instance agree, and takes one of two paths.
On agreement it emits the NoRAG token and charges nothing. On disagreement it runs
the unmodified DPRAG aggregation over the k+1 streams and charges token_ε. The
answer that comes out is a real routed output, and the ε it consumed is a real
number smaller than the budget.

Because a skipped position costs nothing, the k+1 streams fall behind the emitted
sequence. Rather than running them anyway, the router remembers the tokens it
emitted while they were idle and feeds the whole backlog in a single forward pass
the next time an aggregation is needed. Skipping therefore saves wall-clock time
as well as privacy budget, which is what makes the efficiency claim measurable.

With a routed output in hand, the deferred safety check becomes answerable: find
the clinically-loaded spans in the text the router actually produced, and see how
many of them were emitted on the free path — that is, chosen by the model's prior
rather than by the retrieved documents.

The comparisons then run in three phases: screen the twelve Strategy B
configurations cheaply to find the few worth pursuing, run the headline
comparison at full query count across the ε_total grid on the primary model, and
confirm the result generalises on two further models.

## User Stories

1. As a researcher, I want a router that emits the NoRAG token when the strategy agrees, so that positions the documents do not influence cost no privacy budget.
2. As a researcher, I want the router to fall through to the unmodified DPRAG aggregation when the strategy disagrees, so that the ε-DP guarantee at those positions is inherited rather than re-derived.
3. As a researcher, I want the router to skip the k+1 aggregation entirely on agreement rather than compute and discard it, so that the efficiency dimension reflects a real saving rather than an artifact of how I implemented it.
4. As a researcher, I want the k+1 streams to catch up on the backlog of emitted tokens in one forward pass, so that skipping stays cheap no matter how long a run of agreements is.
5. As a researcher, I want the routed output to be a single coherent answer, so that quality can be scored against the iCliniq reference the way any other answer would be.
6. As a researcher, I want the router to report which positions consumed budget, so that ε_usage is derived from what happened rather than estimated.
7. As a researcher, I want ε_usage composed over paid positions only, so that ε savings can be computed as budget minus usage per the proposal's accounting.
8. As a researcher, I want the router to accept whichever strategy I hand it, so that comparing Strategy A against a Strategy B configuration means changing an argument rather than editing the router.
9. As a researcher, I want the router to accept the model as a dependency, so that its logic can be exercised on a laptop against a fake model instead of costing GPU time per iteration.
10. As a researcher, I want a test proving that a strategy which never agrees produces token-for-token the same output as plain DPRAG, so that I know the backlog machinery has not corrupted the baseline path.
11. As a researcher, I want a test proving that a strategy which always agrees produces the pure NoRAG greedy sequence and charges nothing, so that the two degenerate cases bracket the router's behaviour.
12. As a researcher, I want a test proving ε_usage matches the number of paid positions, so that an accounting error cannot hide behind a plausible-looking number.
13. As a researcher, I want the router to retrieve once and build both the k+1 streams and the RAG instance from that same document set, so that the pre-filter and the aggregation are reasoning about identical evidence.
14. As a researcher, I want the router to handle a zero-document query without special-casing at the call site, so that DP retrieval returning nothing does not crash a 200-query run at query 137.
15. As a researcher, I want the router to reuse the existing prompt builders, so that its NoRAG instance is the same NoRAG instance Stage 1 and Stage 2 measured.
16. As a researcher, I want the router's runs seeded through the existing configuration, so that a comparison can be re-run or extended one configuration at a time.
17. As a researcher, I want every routed run recorded with the full parameter snapshot, so that a number in the final report can be traced to the settings that produced it.
18. As a researcher, I want each recorded run to carry a note stating that ε accounting assumes the routing decisions are themselves free, so that the caveat travels with the number instead of living only in a document.
19. As a researcher, I want to score routed answers with BERTScore as well as ROUGE-L, so that "quality did not drop" rests on semantic similarity and not only on word overlap that DP noise disturbs by design.
20. As a researcher, I want the quality functions to live in the library rather than inside one experiment, so that every Stage 3 phase scores answers the same way.
21. As a researcher, I want to screen all twelve Strategy B configurations at low query count first, so that I spend the large budget only on configurations that survive.
22. As a researcher, I want the screening phase to produce the trigger-rate against quality trade-off curve, so that the (k, τ) grid yields a result even though it is never run at full scale.
23. As a researcher, I want the headline comparison to cover the full ε_total grid on the primary model, so that the claim holds across privacy budgets rather than at one convenient point.
24. As a researcher, I want plain DPRAG run as a baseline under identical settings, so that ε savings and any quality change are measured against a like-for-like control rather than against an earlier run.
25. As a researcher, I want a cross-model phase on two further models, so that the finding is a property of the method and not of one model's quirks.
26. As a researcher, I want each phase to write results in the shared record format, so that Stage 5 can load every run through one reader.
27. As a researcher, I want the safety check to run on the router's actual output, so that the clinical spans it inspects are ones a system really produced.
28. As a researcher, I want the safety check to report how often clinically-loaded positions were emitted on the free path, so that I can see whether a loose configuration is treating a drug name like an article.
29. As a researcher, I want the safety check to compare configurations from strict to loose, so that the safety cost of loosening τ is readable rather than inferred.
30. As a researcher, I want flagged positions saved with surrounding context, so that the qualitative inspection the proposal asks for has something to inspect.
31. As a researcher, I want the entity heuristic to stop matching the word "diagnosis" as a diagnosis, so that a false positive does not send me reading positions with no clinical content.
32. As a researcher, I want the entity heuristic to cover conditions as well as drugs and doses, so that it matches the three categories the proposal names.
33. As a researcher, I want zero-document queries excluded from trigger and safety rates and reported separately, so that trivially-firing queries do not flatter every configuration.
34. As a researcher, I want runs to fail loudly when a prerequisite result is missing, so that a phase does not silently compare against stale data.
35. As a reviewer, I want the report to state the measured cost that made the full grid infeasible, so that the reduced scope reads as a considered decision rather than an omission.
36. As a reviewer, I want to see that the router's correctness was established by equivalence tests before any GPU time was spent, so that the reported numbers rest on verified machinery.
37. As a reviewer, I want the ε accounting assumption stated wherever ε_usage appears, so that I am not left to discover that routing decisions depend on the private corpus.
38. As a future maintainer, I want the router's two degenerate behaviours pinned by tests, so that a later optimisation cannot quietly change what the system emits.

## Implementation Decisions

### The router module

A new module owns the routing loop. Its interface is one call: given the retrieved
documents and the query, return a routed result carrying the emitted token ids,
the decoded answer, the per-position strategy decisions, the indices of positions
that consumed budget, and both ε_usage and ε_budget.

It is constructed with three things: a model, a strategy function, and a
generation configuration. The model arriving as a constructor argument is what
allows the loop to be exercised against a fake on CPU; this mirrors the existing
dual-instance entry point, which already takes its model as a parameter.

The loop maintains two batches over one shared emitted sequence:

- the **pre-filter batch** — two rows, the NoRAG instance and the RAG instance —
  advanced at every step, because a decision is needed at every step;
- the **DPRAG batch** — the k+1 streams — advanced only when an aggregation is
  actually required.

Tokens emitted while the DPRAG batch is idle accumulate in a backlog. When a
disagreement occurs, the backlog is fed to that batch as a single multi-token
forward, bringing its cache up to date, and the aggregation proceeds. Feeding a
chunk rather than replaying each step individually is what keeps skipping cheap.

Both batches are built from the same retrieval result, through the existing prompt
builders, so the router's NoRAG instance is identical to the one Stage 1 measured.
A zero-document query degrades naturally: the RAG instance collapses onto the
NoRAG instance, as the prompt builder already handles.

Rejected alternatives are recorded in ADR 0003: computing every row every step and
merely declining to charge (would make the routed system strictly slower than the
baseline, so the efficiency result would be an artifact), and simulating routing
over a plain DPRAG trajectory (the routed trajectory is a mixture of both paths,
so quality could not be scored honestly).

### ε accounting

token_ε is unchanged: it is still the value binary-searched so that composing
max_new_tokens steps reaches the generation budget. The router simply composes it
over the paid positions only, and ε savings is budget minus usage.

Re-solving token_ε for the number of paid positions — trading savings for lower
per-step noise — is not possible, because the paid count is not known until
generation has finished.

Every routed run records that ε accounting counts paid positions only and assumes
the routing decisions are themselves free. That assumption is real: the RAG
instance holds the private documents, so which positions are skipped is a function
of private data. The proposal places this outside its formal scope and tests it
empirically in Stage 4.3.

### Quality scoring

A small library module exposes BERTScore F1 and the existing LCS-based ROUGE-L F1.
The ROUGE-L implementation currently lives inside the temperature-sweep experiment
and moves here unchanged, so earlier numbers stay comparable. BERTScore adds a
package dependency and a one-time model download; per-answer scoring cost is
negligible beside generation. Medical Entity Retention Rate remains a Stage 5
instrument.

### Experiment phases

Three experiments, all reading their settings from the shared configuration object
and writing through the shared run record.

**Screening.** Twelve Strategy B configurations at 20 queries, primary model,
single ε_total. Produces the (k, τ) trade-off curve and selects the finalists.
Roughly 40 minutes.

**Headline comparison.** Plain DPRAG, Strategy A, and the finalists across
ε_total ∈ {5, 10, 20, 40} at 200 queries on the primary model. This is where the
central claim is established. Roughly 11 hours.

**Cross-model.** Plain DPRAG, Strategy A, and the best Strategy B configuration at
one ε_total on the two remaining models. Roughly 3 hours.

The baseline is produced by running the router with a strategy that never agrees,
rather than by a separate code path. The equivalence test guarantees this matches
plain DPRAG, and it removes any doubt that baseline and treatment differ in setup.

### Safety check

Rewritten to consume routed output. For each query, the router generates an
answer; clinically-loaded spans are located in that answer; each such position is
classified by whether it was emitted on the free path or the paid path. The
reported quantity is the share of clinical positions that were skipped, compared
against the share of ordinary positions skipped, for configurations ordered strict
to loose. Flagged positions are saved with context for manual inspection.

### Entity heuristic corrections

Two defects surfaced when the heuristic was run over real generated text. The word
"diagnosis" matches the condition suffix rule and must be excluded. Conditions are
not covered at all, though the proposal names them alongside drugs and doses;
condition suffixes are added, with the same guard against short prefixes that
already prevents "April" matching a drug rule.

### Model set

The third comparison model is fixed at Mistral-7B-Instruct-v0.3, closing the open
item from ADR 0001. This yields three sizes across three architectures, all
accepting a system role, which the single prompt format requires.

## Testing Decisions

A good test here exercises behaviour that is visible at a module's interface and
would still be meaningful if the implementation were rewritten. Tests that assert
on internal bookkeeping — how the backlog list is stored, how many forward passes
occurred — are avoided, because those are the details most likely to change and
least likely to be wrong in a way that matters.

**The router is tested against a fake model on CPU.** The fake returns scripted
logits, so the loop can be driven through chosen agreement patterns without
loading weights. This is the central testing decision: the router is the most
intricate code in the project — two caches advancing at different rates, with
errors that produce plausible-looking wrong answers rather than exceptions — and
it must not be debugged at ten seconds per query on rented hardware.

Three properties carry most of the weight:

- With a strategy that never agrees, the emitted sequence equals plain DPRAG's
  token for token, and ε_usage equals ε_budget. This is what proves the backlog
  catch-up has not corrupted the aggregation path.
- With a strategy that always agrees, the emitted sequence equals the pure NoRAG
  greedy sequence, no position is marked paid, and ε_usage is zero.
- With a mixed pattern, the paid positions are exactly the disagreements and
  ε_usage equals token_ε composed that many times.

The two degenerate cases bracket the behaviour: any error in cache synchronisation
shows up as a divergence from one or the other. Further cases cover a zero-document
query, agreement on the very first position, and a run that ends before
max_new_tokens.

**The quality module is tested as pure functions**, following the existing tests
for the entity heuristic: known-identical text scores 1.0, disjoint text scores
0.0, and empty input does not raise. BERTScore assertions stay coarse — ordering
and bounds rather than exact values — since it depends on a downloaded model.

**The entity heuristic gains regression cases** for the two defects found in real
text: "diagnosis" must not match, and the artifacts produced by incoherent text
motivated moving detection onto routed output. Existing false-positive traps
(cholesterol, April) stay pinned, since loosening the rules would silently
reintroduce them.

**Experiment scripts are not unit tested.** They are thin sequencing over tested
modules, and their real verification is a small-scale run whose output is
inspected — the pattern already used for the smoke checks.

Prior art: the strategy tests establish the style for pure-function testing
including tie-breaks and boundaries; the configuration and record tests establish
round-trip testing through a temporary directory; the entity tests establish the
fake-collaborator pattern that the router's fake model extends.

## Out of Scope

- **MIA and the ε-path attack.** Whether the routing decision sequence leaks
  membership is Stage 4.3. Stage 3 records the assumption; it does not test it.
- **Formal DP analysis of the pre-filter.** The proposal explicitly excludes
  deriving a new guarantee for the pre-filter layer. Stage 3 inherits DPRAG's
  guarantee on the paid path and claims nothing further.
- **Medical Entity Retention Rate.** The scispaCy instrument belongs to Stage 5;
  Stage 3 uses the regex heuristic and labels it as such.
- **Pareto analysis across all dimensions.** Stage 5 combines privacy, quality and
  efficiency. Stage 3 supplies two of the three.
- **The full 168-run grid.** Deliberately not run; see ADR 0003.
- **Re-running Stage 1.1's DP-versus-non-DP comparison.** Its archived results
  predate the record schema, but nothing in Stage 3 depends on them.
- **Replacing the existing dual-instance entry point.** Stages 2.3 and 2.4 depend
  on it and their results stand. It shares prompt construction with the router, so
  the two cannot drift.
- **Multiple seeds per configuration.** Error bars would multiply the budget;
  single-seed runs are reported as such.

## Further Notes

**Execution order is forced.** The safety check needs routed output, so the router
must exist first. Screening then narrows the grid before the expensive phase.

**Watch for a weak stratification result repeating.** Stage 2.4 found only a
1–3.5% trigger-rate difference between dependency strata, with correlations below
0.3, because the consistency distribution is tight. If Stage 3's cross-ε results
are similarly flat, that is a finding to report plainly rather than a bug to hunt.

**Temperature stays at 1.0.** The sweep showed the pre-filter is insensitive to it,
so it is held at the Stage 1 baseline value, keeping the consistency rate and the
trigger rates directly comparable.

**Expect Strategy B to trigger less than Strategy A**, not more. A τ of 0.7 at
k = 10 demands nine of ten tokens match, which is stricter than matching one
argmax. B is nonetheless not a subset of A, since two distributions can share a
top-k set while ordering it differently.

**Publication note.** This spec would normally be filed on the issue tracker with
the `ready-for-agent` label. No tracker is configured for this project and the
GitHub CLI is not installed, so it lives in the repository beside the ADRs. Filing
it later needs no change to the content.
