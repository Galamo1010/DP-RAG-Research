"""Scoring a generated answer against the doctor's reply.

Stage 3.2 has to establish that the pre-filter "saves epsilon without hurting
quality". One metric cannot support that sentence, because the two ways it can
fail look identical to a lexical measure.

WHY TWO METRICS
---------------
**ROUGE-L** counts the longest common subsequence of words. "Take ibuprofen for
pain" against "Use Advil to reduce ache" scores zero: same advice, no shared
words. That matters here more than usual, because the exponential mechanism
samples a token at every paid position -- **changing the wording is what DP
does**. A low ROUGE-L therefore conflates "DP made the answer wrong" with "DP
made the answer say it differently", and those call for opposite conclusions.

**BERTScore** embeds each token in context and greedily matches every candidate
token to its most similar reference token, so `ibuprofen` and `Advil` score high.
It answers "does it mean the same?" where ROUGE-L answers "does it use the same
words?". Neither alone separates a wrong answer from a rephrased one.

The ROUGE-L implementation is moved here unchanged from the Stage 2.3 temperature
sweep. Not touched -- altering it would make the 0.109-0.114 that sweep measured
incomparable with anything scored later.

WHAT NEITHER METRIC FIXES
-------------------------
17.7% of the iCliniq reference answers are truncated mid-sentence with the string
"ChatDoctor" spliced in, frequently eating the drug name ("Sudafed (Pseudoephe
ChatDoctor."). Both metrics score against damaged text and both are depressed by
it. The penalty is constant across configurations, so *relative* comparisons hold
and absolute quality claims do not. Say so where the numbers appear.

COST
----
BERTScore is not part of the generation run. It reads finished answers, so it
scores offline -- on a laptop GPU, or on CPU -- and its model size does not enter
the A100 budget. That is why the model here is the one the BERTScore authors
recommend for best human correlation rather than the smaller default.
"""

from __future__ import annotations

# The BERTScore authors recommend this over the package default (roberta-large)
# for correlation with human judgement. Recorded in ExperimentConfig, because a
# BERTScore number is uninterpretable without knowing which model produced it --
# the same reason vocab_min_count lives there.
DEFAULT_BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """LCS-based ROUGE-L F1 on whitespace tokens (lightweight proxy)."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def bertscore_f1(
    hypotheses: list[str],
    references: list[str],
    *,
    model: str = DEFAULT_BERTSCORE_MODEL,
    rescale: bool = True,
    batch_size: int = 32,
    device: str | None = None,
) -> list[float]:
    """BERTScore F1 for each (hypothesis, reference) pair, in order.

    Batched rather than per-answer because a forward pass over 200 answers at
    once costs a fraction of 200 separate calls; the asymmetry with
    `rouge_l_f1`'s signature reflects that, rather than being an oversight.

    `rescale` maps raw scores against the package's random-pairing baseline. Raw
    BERTScore sits in a narrow band -- two unrelated English medical passages
    still score above 0.8 -- so without rescaling the differences between
    configurations hide in the third decimal place.

    Pairs where either side is empty score 0.0 without being sent to the model,
    matching `rouge_l_f1` and avoiding an unclear failure on empty input.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"got {len(hypotheses)} hypotheses and {len(references)} references; "
            "they are matched pairwise and must be the same length"
        )

    scores = [0.0] * len(hypotheses)
    scorable = [
        i for i, (h, r) in enumerate(zip(hypotheses, references))
        if h.strip() and r.strip()
    ]
    if not scorable:
        return scores

    # Imported here so this module can be imported -- and rouge_l_f1 used --
    # without the package installed. The tests rely on that.
    try:
        from bert_score import score as _score
    except ImportError as exc:
        raise ImportError(
            "bertscore_f1 needs the `bert-score` package: uv sync (it is in "
            "pyproject.toml). rouge_l_f1 does not."
        ) from exc

    try:
        _, _, f1 = _score(
            [hypotheses[i] for i in scorable],
            [references[i] for i in scorable],
            model_type=model,
            lang="en",
            rescale_with_baseline=rescale,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
    except (ValueError, FileNotFoundError) as exc:
        # The usual cause is that the package ships no rescaling baseline for
        # this model. Better to say so than to silently drop the rescaling and
        # return numbers on a different scale from every other run.
        raise RuntimeError(
            f"BERTScore failed for model_type={model!r} with "
            f"rescale_with_baseline={rescale}. If the baseline file is missing, "
            "either pick a model that ships one or set rescale=False -- and "
            "record which, because the two are not comparable."
        ) from exc

    for i, value in zip(scorable, f1.tolist()):
        scores[i] = float(value)
    return scores
