# ADR 0008 — max_retrieve raised from 10 to 40

Date: 2026-08-29
Status: Accepted. Invalidates the comparability of every result produced before it.

## Context

`pup_retrieve` samples a score threshold with the exponential mechanism, takes every
document above it, and then — because the k+1 batch cannot be unbounded — draws
`max_retrieve` of them **at random**:

```python
retrieved = [doc for doc, score in doc_score_pairs if score > score_threshold]
return self._py_rng.sample(retrieved, min(len(retrieved), self.max_retrieve))
```

The random draw is upstream's, not this project's, and the paper describes neither
it nor the cap: it says the threshold "is then used to select **all** the documents
whose similarity is above" it. The cap is an engineering limit that the write-up
never had to make.

`max_retrieve = 10` was this project's own choice, and `config.py` justified it on
cost alone — nobody had measured what it did to retrieval. Measured over the 200
queries the experiments use:

| | |
|---|---|
| Documents clearing the DP threshold | median **83** (Q1 67, Q3 107, max 175) |
| Kept at max_retrieve=10 | 10, drawn at random |
| **Discarded** | **88% of qualifying documents, median** |
| Queries needing no discard | 7 of 188 |
| **True top-10 documents actually retrieved** | **1.51 of 10** |

So the RAG instance was reasoning over a random eighth of the qualifying evidence,
and usually not the best eighth. One query asking about a wife's early pregnancy
after two miscarriages retrieved, first, a reply about chronic back pain.

**This matters beyond retrieval quality.** The pre-filter fires when the RAG and
NoRAG instances agree, and irrelevant documents give the RAG instance nothing to
disagree with. Part of the measured 82.5% trigger rate and 71.5% epsilon saving is
therefore attributable to retrieval failing rather than to positions genuinely not
depending on documents.

## Decision

`max_retrieve = 40`.

### Why not higher

Bisecting for the out-of-memory ceiling on an 80 GB A100, three queries per probe:

| max_retrieve | k reached | peak VRAM | s/query | |
|---|---|---|---|---|
| 30 | 28.0 | 52.1 GB | 43.8 | OK |
| 65 | 65.0 | 77.0 GB | 59.9 | OK |
| **69** | **69.0** | **79.4 GB** | 67.4 | **OK — 93% of capacity** |
| 73 | — | 70.1 GB | — | OOM |
| 82 | 53.5 | 73.0 GB | — | OOM |

Peak memory grows at roughly **0.71 GB per document**, so 40 lands near 59 GB —
about 70% of capacity. The ceiling of 69 is unusable: those were three queries, and
peak memory moves with prompt length. A 200-query run that dies at query 137 costs
far more than the documents it was trying to fit.

### Why not stay at 10

The retrieval defect is the largest known weakness in the results, and Phase 2 will
carry it into every number it produces. 1.51 of 10 becomes roughly 4.9 of 10 —
about 3.2x more of the best-matching evidence actually reaching the model.

## Consequences

**Every earlier result becomes incomparable.** Stage 1.2's consistency ceiling of
0.874, Stage 2.3's temperature sweep, Stage 2.4's stratification and Stage 2.5's
safety check were all measured at max_retrieve=10. They are pilot measurements from
here on.

**Stage 1.2 must be re-run before Stage 3.2 can cite a ceiling.** "Strategy A
triggers at 82.5%, against the 87.4% ceiling" is two different rulers unless both
are measured under the same retrieval. That re-run also picks up the date pin from
ADR 0005, so it becomes the first result comparable with everything that follows.

**Generation costs roughly twice as much time.** Interpolating the probe, 40
documents cost around 48 s/query against 22.7 at 10 — the k+1 batch is wider and the
pre-filter's RAG instance concatenates four times as much text. Phase 2 moves from
roughly 13 hours to roughly 23.

**The trigger rate is expected to FALL.** Better documents mean more positions where
they change the answer, so more disagreement and more epsilon spent. If the headline
number drops after this change, that is the correction working: the old number was
partly measuring retrieval failure. Report both and say why they differ.

**One measurement is unresolved.** The same configuration timed 26.4 s/query in
preflight and 43.8 s/query in this probe — 1.66x apart, with the transformers
upgrade (4.46.1 to 4.57.6) as the obvious suspect. Three queries each, so warm-up
dominates both. Before committing to a twenty-hour phase, take a steady-state
timing over ~20 queries; the difference is ten hours of rented GPU.

**The random subsample stays.** Taking the top-k after thresholding would use raw
scores again, and the threshold's noise is what protects which documents were
selected. Raising the cap narrows the gap between what qualifies and what is seen
without touching the mechanism.
