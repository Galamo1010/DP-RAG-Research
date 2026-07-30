"""Tests for the Stage 2.5 medical-token heuristic (no GPU, no tokenizer download)."""

import pytest

from dprag.medical_flags import (
    find_medical_spans,
    flag_medical_tokens,
    token_char_spans,
)


def _matched(text: str) -> list[str]:
    return [m for _, _, m in find_medical_spans(text)]


# --------------------------------------------------------------------------
# what should be flagged
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Take 500mg twice daily", ["500mg", "twice daily"]),
    ("Give 2.5 ml of the syrup", ["2.5 ml"]),
    ("Start amoxicillin now", ["amoxicillin"]),
    ("Prescribed metformin", ["metformin"]),
    ("Use omeprazole 20 mg", ["omeprazole", "20 mg"]),
    ("atorvastatin lowers cholesterol", ["atorvastatin"]),
    ("BP was 120/80 mmHg", ["120/80 mmHg"]),
    ("Take it po tds", ["po", "tds"]),
])
def test_flags_doses_and_drug_names(text, expected):
    assert _matched(text) == expected


def test_flags_drug_names_by_class_suffix_not_a_fixed_list():
    """The point of suffix matching: drugs never seen before still flag."""
    for drug in ("clarithromycin", "esomeprazole", "candesartan", "rosuvastatin"):
        assert _matched(f"prescribe {drug} today"), drug


# --------------------------------------------------------------------------
# what should NOT be flagged (false positives are the expensive failure here:
# they would send you inspecting positions that carry no clinical content)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "I have had a headache since Monday",
    "Please consult your doctor for advice",
    "The pain is worse when I bend over",
    "in the morning",          # bare "in" must not match the -vir/-in family
    "I saw a video",           # "vir" inside a word, not a suffix
    "It happened 3 times",     # a count with no time unit
])
def test_ordinary_prose_is_not_flagged(text):
    assert _matched(text) == []


def test_short_words_do_not_match_suffixes():
    assert _matched("vir nib mab") == []


@pytest.mark.parametrize("word", ["cholesterol", "April", "sterol"])
def test_known_false_positive_traps(word):
    """Each of these hit a suffix rule during development.

    cholesterol/sterol matched a bare -terol (intended for salbuterol); April
    matched -pril. They are pinned because loosening the suffix list would
    silently bring them back, and a false positive sends you inspecting positions
    that carry no clinical content.
    """
    assert _matched(f"the {word} result") == []


# --------------------------------------------------------------------------
# token-level flagging, including the sub-word case this exists to handle
# --------------------------------------------------------------------------

class _FakeTokenizer:
    """Decodes by concatenation, so tests can pin sub-word behaviour without a model."""

    def __init__(self, pieces: list[str]):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def test_token_spans_cover_the_text_contiguously():
    tok = _FakeTokenizer(["Take ", "500", "mg"])
    spans = token_char_spans(tok, [0, 1, 2])
    assert spans == [(0, 5), (5, 8), (8, 10)]


def test_subword_fragments_of_a_drug_name_are_all_flagged():
    """'metformin' arriving as met+form+in is the case token-level matching misses."""
    tok = _FakeTokenizer(["Take ", "met", "form", "in", " now"])
    flags, text, matches = flag_medical_tokens(tok, [0, 1, 2, 3, 4])
    assert text == "Take metformin now"
    assert [m for _, _, m in matches] == ["metformin"]
    assert flags == [False, True, True, True, False]


def test_dose_split_across_number_and_unit_flags_both():
    tok = _FakeTokenizer(["Take ", "500", "mg", " daily"])
    flags, _, _ = flag_medical_tokens(tok, [0, 1, 2, 3])
    assert flags[1] and flags[2]
    assert not flags[0]


def test_no_medical_content_flags_nothing():
    tok = _FakeTokenizer(["I ", "have ", "a ", "headache"])
    flags, text, matches = flag_medical_tokens(tok, [0, 1, 2, 3])
    assert matches == []
    assert flags == [False] * 4


def test_empty_input():
    assert flag_medical_tokens(_FakeTokenizer([]), []) == ([], "", [])


def test_flags_align_one_to_one_with_tokens():
    tok = _FakeTokenizer(["Give ", "amox", "icillin", " 500", "mg"])
    ids = [0, 1, 2, 3, 4]
    flags, _, _ = flag_medical_tokens(tok, ids)
    assert len(flags) == len(ids)
