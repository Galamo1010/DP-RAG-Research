# Domain language

Shared vocabulary for this project. Use these terms in code, comments and the
report; where a term has a Chinese name in the proposal (專題計畫書), both are given
so the two documents stay readable against each other.

Project: **基於 Top-K Logit 相似度之差分隱私生成改良研究** — cutting the ε that DP-RAG
spends by skipping the generation positions where the retrieved documents make no
difference.

## Data

**Corpus** — the private document set the system may retrieve from: 110,513 unique
doctor replies (the `output` field of HealthCareMagic-100k). This is what DP
protects and what a membership-inference attack targets. Patient questions from
that file are *not* in the corpus.

**Query** — a patient question (`input` from iCliniq-10k), paired with the real
doctor's answer (`answer_icliniq`) as the **reference** for quality scoring. Queries
come from a different platform than the corpus, so a query never appears in it.

## The two DP layers

**DP retrieval** — `PUPVectorStore.pup_retrieve`. Samples a score *threshold* with
the exponential mechanism instead of taking a fixed top-k, so whether any one
document was retrieved is itself noisy. Costs **ε_retrieval** (0.2) once per query.

**DP generation** — `DPModel` + `DPLogitsAggregator`. Costs **ε_generation**, spent
per token.

**Two-layer accounting** — end-to-end ε is the PLD composition of the two layers
(proposal Eq3), read off with `get_epsilon_for_delta(δ)`.

**Clipping (Δ)** — the exponential mechanism's sensitivity bound, always *derived*,
never set by hand: `Δ = token_ε × temperature / 2` (Eq4). `token_ε` is itself
reverse-solved by binary search so that composing `max_new_tokens` steps lands on
the generation budget.

## Instances and streams

**NoRAG instance** — the model given the query and **no documents**. Appears twice:
as row 0 of DPRAG's k+1 streams (the *public prior*, weighted by ω), and as row 0
of the pre-filter's 2-row batch. Both come from `prompts.norag_chat`, and they must
stay identical or the Stage 2 comparison is comparing two different things.

**RAG instance** — the model given the query and **every retrieved document
concatenated** (proposal 3.1). One prompt, not k. Distinct from a DPRAG stream.

**DPRAG stream** — one of the k rows that each carry exactly one document. The k+1
rows are aggregated per token by clipping and the exponential mechanism.

**Public prior** — another name for the NoRAG stream in the aggregation, where it
enters with weight ω.

## The contribution

**Pre-filter (前置篩選)** — the layer this project adds ahead of the DP aggregation.
At each position it compares the RAG and NoRAG instances; if they agree, the
position is document-independent, so the NoRAG token is emitted and **no ε is
spent**. Disagreement falls through to unmodified DPRAG.

**Strategy A (策略A)** — agreement means `argmax(logit_RAG) == argmax(logit_NoRAG)`.

**Strategy B (策略B)** — agreement means `Jaccard(top-k(RAG), top-k(NoRAG)) ≥ τ`
(Eq1). Note that τ reads looser than it is: k=10, τ=0.7 requires 9 of the 10 tokens
to match. B is *not* a superset of A — two distributions can share a top-k set while
ordering it differently, so B can fire where A does not.

**Consistency rate (一致率)** — Stage 1.2's measurement: over one generated sequence,
the fraction of positions where NoRAG's argmax matches DPRAG's. The **greedy**
variant (argmax vs argmax, ≈0.87 measured) is the one that corresponds to what a
pre-filter decides; the **sampled** variant (argmax vs the sampled token, ≈0.12) is
dominated by temperature sampling noise and understates the opportunity. This rate
is the theoretical ceiling on ε savings.

**Trigger rate** — the fraction of positions a strategy actually routes around DP.
Distinct from the consistency rate: the ceiling is what is available, the trigger
rate is what a given strategy claims.

**ε savings** — `ε_budget − ε_usage` (Eq5); only positions that fall through to DP
consume budget.

**Zero-document query** — DP retrieval legitimately returning nothing. The RAG
instance then collapses onto NoRAG and every strategy trivially fires, so these
queries are reported separately rather than averaged in.

## Code structure

**Library vs experiment** — the one structural rule: `dprag/` is imported and has no
`main()`; `experiments/` is executed and produces results. Dependencies run
`experiments → dprag`, never the reverse. Stage numbers name *experiments*, because
a stage is a point in time; they never name library modules.

**Bench** — `Bench.build(cfg)`: a configured engine with the corpus embedded and both
RNGs seeded. What an experiment needs before it can measure anything.

**ExperimentConfig** — every parameter one run depends on. `to_dict()` feeds the run
record, so a parameter cannot be silently left unrecorded.

**RunRecord** — the one result schema, plus git commit and parameter snapshot.
`load_all()` is how Stage 5 reads every run uniformly.

**Seed** — reproducibility only. It fixes retrieval's two draws and generation
sampling so a single configuration can be re-run. It says nothing about privacy: the
DP guarantee is a property of the mechanism's output distribution, so a seeded
output must never be described as a DP-protected release (see
`docs/adr/0002-seeded-retrieval.md`).
