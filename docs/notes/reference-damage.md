# The ChatDoctor references are damaged, and it does not matter (for ROUGE-L)

Date: 2026-08-26
Related: [ADR 0006](../adr/0006-no-medqa.md), `dprag/quality.py`

Stage 3.2 scores answers against iCliniq's doctor replies. Those replies are
damaged. This records how badly, what was expected to follow from that, and the
measurement that showed the expectation was wrong.

## What the damage is

Measured over the 200-query sample the experiments actually use (`query_seed=42`):

| | count | share |
|---|---|---|
| Truncated mid-sentence with `ChatDoctor` spliced in | 33 | 16.5% |
| References an attachment the model never sees | 25 | 12.5% |
| Both | 3 | 1.5% |
| **Undamaged** | **139** | **69.5%** |

The truncation eats content, frequently a drug name:

```
Do you have a dan ChatDoctor.                    <- "dandruff"
nothing to do with anesthesia ChatDoctor.        <- sentence cut
Keep well hy ChatDoctor.                          <- "hydrated"
Sudafed (Pseudoephe ChatDoctor.                   <- "Pseudoephedrine"
```

The attachment cases are a different problem: the doctor is describing an MRI or a
doppler study the model cannot see, so no answer can match.

Detection is heuristic: `<content word> ChatDoctor`, with common function words
excluded so that ordinary phrases like "Welcome to ChatDoctor forum" do not count.
It will miss truncations that inserted nothing, and it will occasionally flag a
sentence that legitimately names the platform.

## The expectation, and why it was wrong

The obvious reading is that damaged references depress the quality scores, so the
honest fix is to report quality on the undamaged subset alongside the full set,
and the gap between them quantifies the contamination.

That was tested retroactively on the Stage 2.5 runs — 12 days after they were
generated, on saved text, with no GPU. ROUGE-L, all queries versus undamaged
queries:

| strategy | all (n=184) | undamaged (n=129) | difference |
|---|---|---|---|
| A | 0.1215 | 0.1232 | +0.0016 |
| B_k20_t0.9 | 0.0927 | 0.0941 | +0.0014 |
| B_k20_t0.7 | 0.1091 | 0.1099 | +0.0007 |
| B_k50_t0.5 | 0.1192 | 0.1214 | +0.0022 |

Roughly 1–2% relative, in the direction predicted but nowhere near the magnitude
assumed. The reason is that ROUGE-L is *already* low: model and doctor share few
word sequences whatever happens, so removing a fragment from an already-poor
overlap barely moves the mean.

**The report should say the damage was measured and found not to matter, rather
than excluding 30% of the sample.** That is both the stronger claim and the one
that needs no defence against selection bias — the excluded attachment cases skew
towards imaging-related presentations, which is not a random 12.5%.

## What is still open

**BERTScore is untested here.** It matches semantically, so a missing drug name
may cost it more than it costs a subsequence measure. Re-run this comparison once
the package is available; if the gap is small there too, this limitation comes off
the list entirely.

## The transferable part

Quality scoring is entirely post hoc. It needs the generated text, the reference,
and nothing else — no model, no GPU, no re-run. Any decision about how to group,
filter or weight the scoring can therefore be deferred indefinitely, **provided
the generated text is saved**. It is (`text`, and now `emitted` via `dprag/trace.py`).

The corollary is the useful one: do not spend GPU time settling a scoring question.
Generate once, save the text, and argue about the metric afterwards.
