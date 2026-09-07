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

## Resolved: BERTScore loses no more than ROUGE-L (2026-09-05)

The question left open above was whether a *semantic* metric is hurt more than a
subsequence metric when the reference is missing a drug name. It is not.

Re-measured on the Stage 3 records at eps=40, all queries versus the undamaged
subset, both metrics on the same answers:

| record | ROUGE all | clean | diff | BERT all | clean | diff |
|---|---|---|---|---|---|---|
| baseline (Llama) | 0.1200 | 0.1209 | +0.0009 | 0.0792 | 0.0825 | +0.0033 |
| A (Llama) | 0.1210 | 0.1213 | +0.0002 | 0.0688 | 0.0714 | +0.0027 |
| B_k20_t0.7 (Llama) | 0.1191 | 0.1196 | +0.0004 | 0.0700 | 0.0714 | +0.0014 |
| B_k20_t0.9 (Llama) | 0.1201 | 0.1209 | +0.0008 | 0.0835 | 0.0855 | +0.0020 |
| B_k50_t0.5 (Llama) | 0.1210 | 0.1223 | +0.0013 | 0.0608 | 0.0642 | +0.0034 |
| baseline (gemma) | 0.0606 | 0.0637 | +0.0031 | −0.2508 | −0.2419 | +0.0090 |
| A (gemma) | 0.0850 | 0.0890 | +0.0040 | −0.1763 | −0.1698 | +0.0065 |
| B_k20_t0.7 (gemma) | 0.0635 | 0.0648 | +0.0013 | −0.2320 | −0.2290 | +0.0030 |

Across every record: ROUGE-L moves by **+0.0002 to +0.0040**, BERTScore by
**+0.0014 to +0.0090**. Both are in the direction predicted and both are small.
BERTScore's absolute movement is larger, but so is its scale after rescaling, and
the ordering of configurations is unchanged in every cell.

**So the conclusion above stands for both metrics: the damage was measured and
found not to matter, and the sample is not filtered.** This limitation comes off
the list.

Two things changed since the measurement at the top of this note:

**The detector is now in code.** `dprag.chatdoctor.reference_damage` returns the
set of damage kinds (`TRUNCATED`, `ATTACHMENT`), so the split is reproducible and
`experiments/stage3_score.py` prints the comparison on every run. The original
table was computed ad hoc and its function-word list was never saved, which is why
the counts differ slightly: this detector finds 34 truncated-only / 24
attachment-only / 4 both / 138 undamaged, against the 33 / 25 / 3 / 139 above —
within one in every cell. **A measurement whose code was not kept is a measurement
you no longer have**, and that is the transferable lesson from this note.

**The kinds are returned separately rather than as one boolean**, because they
break a score for different reasons: truncation removes content (so a semantic
metric might have lost more, which is the question this section answers), whereas
an attachment reference is intact text describing an image the model never sees —
there the *task* is impossible, not the text broken.

## The transferable part

Quality scoring is entirely post hoc. It needs the generated text, the reference,
and nothing else — no model, no GPU, no re-run. Any decision about how to group,
filter or weight the scoring can therefore be deferred indefinitely, **provided
the generated text is saved**. It is (`text`, and now `emitted` via `dprag/trace.py`).

The corollary is the useful one: do not spend GPU time settling a scoring question.
Generate once, save the text, and argue about the metric afterwards.
