# ADR 0003 — Stage 3 scope, routing implementation, and the ε-accounting caveat

Date: 2026-08-02
Status: Accepted (design agreed; implementation not started)

## Context

Stage 3 deploys the strategy A/B functions into DPRAG as a real router and runs
the three comparisons. Two facts, measured rather than assumed, shaped the design.

**The proposal's grid does not fit in any plausible budget.** 3.2 specifies
(1 strategy A + 12 strategy B configs + 1 ablation) × 3 models × ε_total ∈
{5,10,20,40} = 168 runs of 200 queries. At the measured ~10 s/query for a routed
run that is **~92 hours of A100 (~$185)**. The proposal's own 3.2 step limit
anticipates this and prescribes the fix: screen at small batch first.

**Batch width is not the bottleneck; sequence length is.** A k+1 = 11-row DPRAG
batch runs ~7.1 s/query and a 2-row pre-filter batch ~7.7 s/query. The 128
sequential decode steps dominate, not the width of each step. This is what makes
conditional routing worth implementing: skipping the k+1 aggregation saves real
time, and catching its KV cache up later in one multi-token forward is cheap.

## Decisions

### 1. True conditional routing

Hand-written autoregressive loop. Each step runs the 2-row pre-filter batch and
calls the strategy. On agreement the NoRAG argmax is emitted, the k+1 aggregation
is **not run at all**, and the token is added to a pending list. On disagreement
the pending tokens are fed to the k+1 batch in a single forward — bringing its KV
cache up to date — and the aggregation produces the token.

Rejected: computing every row every step and merely not charging ε. That is much
simpler, but it adds 2 rows to the baseline's k+1, so the routed system would be
*strictly slower* than plain DPRAG and the efficiency dimension the proposal asks
about (3.1 step limit: confirm 整體效益為正) would be negative by construction —
the measurement would be an artifact of the implementation, not a finding.

Also rejected: post-hoc simulation over a pure DPRAG trajectory. Cheapest, but the
routed system emits a *mixture* of NoRAG and DP-aggregated tokens, so its
trajectory is neither pure trajectory. Quality could not be measured honestly.

### 2. Three-phase budget (~15 h A100, ~$30)

| Phase | Runs | Cost |
|---|---|---|
| 1. Screen: 12 B configs × 20 queries, Llama, ε=10 | 12 | ~40 min |
| 2. Main: {baseline, A, 2 finalists} × ε ∈ {5,10,20,40} × 200 queries, Llama | 16 | ~11 h |
| 3. Cross-model: {baseline, A, best B} × ε=10 × 200 queries × Qwen, Mistral | 6 | ~3 h |

Phase 1 also produces the (k, τ) trade-off curve, which is the point of the 12
configs; running them all at 200 queries would buy precision on configurations
most of which will be discarded.

### 3. BERTScore alongside ROUGE-L

Stage 3 is where "saves ε without hurting quality" is established, so quality
cannot rest on the lightweight proxy used to pick a temperature in 2.3. ROUGE-L
measures lexical overlap and DP noise changes wording by design, so a semantic
metric is needed next to it. Medical Entity Retention Rate stays in Stage 5.

### 4. Third model: Mistral-7B-Instruct-v0.3

Closing the TODO left by [ADR 0001](0001-replace-gemma.md). Gives 7B/8B/14B across
three different architectures, all accepting a system role. Larger models were
considered and rejected: 2026's frontier open models are mostly MoE with 119B–397B
*total* parameters, and MoE saves compute, not VRAM, so they do not fit an 80 GB
A100. A dense 32B would fit but leaves ~14 GB for the KV cache of 11 long-prompt
rows, runs 3–4× slower (budget → 50 h), and would confound architecture with model
size in the cross-model comparison.

## The ε-accounting caveat

ε_usage is composed over paid positions only (proposal Eq5), and this ADR records
the assumption that entails, because it is invisible in the resulting number.

**The routing decision is computed from the private documents.** The RAG instance
holds every retrieved document, so *which* positions are skipped is itself a
function of private data. Charging zero ε at those positions is therefore only
sound if the sequence of routing decisions leaks nothing on its own.

The proposal is aware of this and places it outside the formal scope ("本專題不對
前置篩選層本身推導新的差分隱私保護保證"), testing it empirically instead via the
不使用ε路徑專項 MIA in Stage 4.3. A second, related subtlety: T_paid is
data-dependent, and adaptive composition needs more care than composing a fixed
number of steps.

Consequently, no Stage 3 output may describe ε_usage as an unconditional
guarantee. Runs record

    epsilon_accounting: "paid positions only; assumes routing decisions
                         are themselves free — tested empirically in Stage 4.3"

so the assumption travels with the number rather than living only in a document.

## Consequences

**2.5 is blocked on 3.1 and that is the right order.** The first attempt searched
for drug names in `norag_argmax_text`, a per-step teacher-forced overlay rather
than any system's output, and duly matched artifacts ("youritisitis",
"TOTOSOSOSIS"). Detecting entities in the router's real output requires the router.
Execution order: 3.1 → 2.5 → 3.2.

**Deviation from the proposal to declare.** The full grid is not run. The report
should state the measured cost that made it infeasible and the screening protocol
used instead — which the proposal itself recommends.

**The router is the most intricate code in the project.** Two KV caches advancing
at different rates is easy to get subtly wrong, so the ε accounting and the
routing decisions get unit tests against a fake model, on CPU, before any GPU time
is spent.
