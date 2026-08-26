# ADR 0006 — MedQA is not used

Date: 2026-08-26
Status: Accepted. Records a decision taken while the proposal was being drafted.

## Context

The proposal names MedQA **once**, in Stage 3.2's 核心任務:

> 三組比較均在相同模型（…）、資料集（**ChatDoctor、MedQA**）、ε_total ∈ {5,10,20,40} 設定下進行

The nearly identical sentence in 比較對象 names ChatDoctor alone, and 需求環境 (三)
資料集 — the section that has to say how a dataset is split into corpus, queries
and references — describes ChatDoctor in full and never mentions MedQA. The
repository has no trace of it either.

So the document already carries this decision in two places out of three. The
author's account is that the intention to drop MedQA existed while drafting and
simply never reached the page. That makes this a record of an old decision rather
than a new deviation, and it is written down now because the alternative is
arriving at an oral defence with a dataset named in the plan, absent from the
work, and unexplained.

## Decision

MedQA is not used. ChatDoctor is the only dataset.

### Why it does not fit

MedQA is USMLE-style multiple choice: a clinical vignette, four options, and a
single letter as ground truth, retrieved against a corpus of 18 medical
textbooks. Three things break against this project's architecture, and the third
is fatal on its own.

**The DP semantics do not hold.** Differential privacy here protects membership:
whether a particular document is in the corpus. The corpus would be published
textbooks. There is nothing to protect and nothing for the Stage 4.3 membership
attack to attack.

**The quality axis cannot run.** ROUGE-L and BERTScore compare two passages of
text. The reference is one letter.

**The contribution has nothing to route.** The whole method is a per-position
decision over a generated sequence — 128 positions, each asking whether the
retrieved documents change the next token. A single-letter answer is one
position. The pre-filter would have almost nothing to filter.

### What carries the weight instead

The two purposes a second dataset might have served are already covered without
one:

**Generalisation** is carried by the cross-model phase. Three models across three
architectures answers "is this a property of the method or of one model", which is
the same question a second dataset would answer along a different axis. ADR 0003
already budgets for it.

**An objective correctness signal** — the one thing MedQA offers that nothing else
here does, since ROUGE-L and BERTScore both measure similarity to *one* valid
answer rather than correctness — is Stage 5's Medical Entity Retention Rate. Does
the routed answer still name the drug the doctor named? That uses data already in
hand and the entity detection already written, and it does not require a corpus
that DP cannot meaningfully protect.

## Consequences

**The report explains a resolved inconsistency, not a cut feature.** The proposal
disagrees with itself about the dataset; this records which side the work took and
why, and the reasoning is structural rather than budgetary.

**No second experimental track.** Adding MedQA would have doubled Stage 3.2 —
separate loading, prompts and scoring — for a track whose privacy framing does not
hold.

**The correctness gap is real until Stage 5 closes it.** Between now and MERR, the
project has no metric that distinguishes "the answer is still correct" from "the
answer is still similar". That is a limitation to state, not to hide behind the
cross-model phase.
