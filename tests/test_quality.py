"""Tests for the quality metrics (no GPU, no model download).

ROUGE-L is covered fully because it is pure. BERTScore is covered only where it
can be without the package -- the pairing contract and the empty-input path --
and the rest is left to the preflight on a machine that has the model.
"""

import pytest

from dprag import quality
from dprag.quality import bertscore_f1, rouge_l_f1


# --------------------------------------------------------------------------
# ROUGE-L -- moved from the temperature sweep and deliberately unchanged
# --------------------------------------------------------------------------

def test_identical_text_scores_one():
    assert rouge_l_f1("take two tablets daily", "take two tablets daily") == 1.0


def test_disjoint_text_scores_zero():
    assert rouge_l_f1("aaa bbb", "ccc ddd") == 0.0


def test_case_is_ignored():
    assert rouge_l_f1("Take Two", "take two") == 1.0


@pytest.mark.parametrize("hyp,ref", [("", "something"), ("something", ""), ("", "")])
def test_empty_input_scores_zero_rather_than_raising(hyp, ref):
    assert rouge_l_f1(hyp, ref) == 0.0


def test_partial_overlap_lands_between():
    score = rouge_l_f1("take two tablets daily", "take three tablets daily")
    assert 0.0 < score < 1.0


def test_subsequence_need_not_be_contiguous():
    """LCS, not substring: the shared words may be interrupted."""
    assert rouge_l_f1("take the red tablets", "take tablets") > 0.5


def test_paraphrase_scores_zero_which_is_why_bertscore_exists():
    """The failure this module's second metric answers.

    Same clinical advice, no shared vocabulary. DP sampling rewords by design,
    so a lexical metric cannot separate "the answer got worse" from "the answer
    got rephrased" -- and Stage 3.2's whole claim rests on that distinction.
    """
    assert rouge_l_f1("Take ibuprofen for pain", "Use Advil to reduce ache") == 0.0


# --------------------------------------------------------------------------
# BERTScore -- what is checkable offline
# --------------------------------------------------------------------------

def test_mismatched_lengths_are_rejected():
    """Scoring is pairwise, so a length mismatch is a caller bug, not something
    to silently truncate."""
    with pytest.raises(ValueError, match="matched pairwise"):
        bertscore_f1(["a", "b"], ["a"])


def test_all_empty_pairs_return_zeros_without_loading_a_model():
    """Also the reason this can be tested at all: nothing is imported until
    there is something to score."""
    assert bertscore_f1(["", "  "], ["x", ""]) == [0.0, 0.0]


def test_no_pairs_returns_no_scores():
    assert bertscore_f1([], []) == []


def test_the_recommended_model_is_the_default():
    """The BERTScore authors recommend this over the package default; the choice
    is recorded in ExperimentConfig because scores from different encoders are
    not comparable."""
    from dprag.config import ExperimentConfig

    assert quality.DEFAULT_BERTSCORE_MODEL == "microsoft/deberta-xlarge-mnli"
    assert ExperimentConfig().bertscore_model == quality.DEFAULT_BERTSCORE_MODEL
    assert "bertscore_model" in ExperimentConfig().to_dict()
    assert "bertscore_rescale" in ExperimentConfig().to_dict()
