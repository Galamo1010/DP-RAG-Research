"""Tests for the Stage 2.5 medical heuristic (no GPU, no tokenizer, no corpus).

The vocabulary arrives as a parameter rather than being loaded, so these run
against hand-made word sets instead of 112,165 doctor replies.
"""

import pytest

from dprag.medical_flags import (
    PATTERN_KIND,
    WORD_KIND,
    find_medical_spans,
    flag_medical_tokens,
    token_char_spans,
)


def _matched(text: str, vocabulary=None) -> list[str]:
    return [s.text for s in find_medical_spans(text, vocabulary)]


def _kinds(text: str) -> list[str]:
    return [s.kind for s in find_medical_spans(text)]


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


@pytest.mark.parametrize("condition", [
    "gastritis", "hepatitis", "cirrhosis", "neuropathy", "fibromyalgia",
    "hysterectomy", "hyperplasia", "hypertrophy", "thalassemia", "hematuria",
])
def test_flags_conditions(condition):
    """The proposal names conditions alongside drugs and doses; they were missing."""
    assert _matched(f"consistent with {condition}") == [condition]


def test_conditions_the_prefix_guard_cannot_reach_are_listed_by_name():
    """'an|emia' and 'a|trophy' have too short a stem for the >=3-letter guard."""
    for condition in ("anemia", "atrophy", "asthma", "diabetes"):
        assert _matched(f"history of {condition}") == [condition], condition


# --------------------------------------------------------------------------
# what should NOT be flagged
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


@pytest.mark.parametrize("word", [
    "diagnosis",    # 13,955 occurrences in the corpus -- the top -osis word
    "prognosis",
    "homeopathy",
    "sympathy",
    "nostalgia",
])
def test_ordinary_words_shaped_like_conditions_are_blocked(word):
    """A vocabulary cannot catch these: they are real words, just not conditions.

    Blocking them by name is the only guard, which is why the list was built by
    frequency-ranking corpus words against the condition rules.
    """
    assert _matched(f"the {word} was clear") == []


# --------------------------------------------------------------------------
# the vocabulary filter: what separates a real term from a DP-noise artifact
# --------------------------------------------------------------------------

@pytest.mark.parametrize("artifact", [
    "Totosis", "Sonopathy", "Alleralgia", "Cerderitis", "Hepitis", "iboprofen",
])
def test_vocabulary_rejects_noise_artifacts_the_shape_rules_accept(artifact):
    """Every one of these was produced by DP noise and matched on shape alone.

    Without a vocabulary the morphological rules take them; with one they are
    rejected, because none occurs in 112,165 real doctor replies. This pins the
    difference the vocabulary makes, which is the whole reason it exists.
    """
    text = f"suggestive of {artifact} here"
    assert _matched(text) == [artifact]                  # shape alone accepts
    assert _matched(text, vocabulary=frozenset()) == []  # vocabulary rejects


def test_vocabulary_keeps_words_it_contains():
    vocabulary = frozenset({"hepatitis"})
    assert _matched("likely hepatitis", vocabulary) == ["hepatitis"]


def test_vocabulary_never_filters_doses():
    """No word list can judge '197 mg'; pattern matches must survive an empty one."""
    assert _matched("take 197 mg daily", vocabulary=frozenset()) == ["197 mg"]


# --------------------------------------------------------------------------
# the two kinds are labelled, because only one of them can be vetted
# --------------------------------------------------------------------------

def test_word_and_pattern_matches_are_labelled_separately():
    assert _kinds("give amoxicillin 500 mg") == [WORD_KIND, PATTERN_KIND]


def test_spans_are_ordered_and_do_not_overlap():
    spans = find_medical_spans("start metformin 500mg twice daily then aspirin")
    assert [s.text for s in spans] == ["metformin", "500mg", "twice daily", "aspirin"]
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end <= later.start


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
    kinds, text, spans = flag_medical_tokens(tok, [0, 1, 2, 3, 4])
    assert text == "Take metformin now"
    assert [s.text for s in spans] == ["metformin"]
    assert kinds == [None, WORD_KIND, WORD_KIND, WORD_KIND, None]


def test_dose_split_across_number_and_unit_flags_both():
    tok = _FakeTokenizer(["Take ", "500", "mg", " daily"])
    kinds, _, _ = flag_medical_tokens(tok, [0, 1, 2, 3])
    assert kinds[1] == PATTERN_KIND and kinds[2] == PATTERN_KIND
    assert kinds[0] is None


def test_token_flags_respect_the_vocabulary():
    """The filter has to reach per-token labels, not only the span list."""
    tok = _FakeTokenizer(["seems ", "like ", "Tot", "osis"])
    ids = [0, 1, 2, 3]
    assert flag_medical_tokens(tok, ids)[0] == [None, None, WORD_KIND, WORD_KIND]
    assert flag_medical_tokens(tok, ids, frozenset())[0] == [None] * 4


def test_no_medical_content_flags_nothing():
    tok = _FakeTokenizer(["I ", "have ", "a ", "headache"])
    kinds, text, spans = flag_medical_tokens(tok, [0, 1, 2, 3])
    assert spans == []
    assert kinds == [None] * 4


def test_empty_input():
    assert flag_medical_tokens(_FakeTokenizer([]), []) == ([], "", [])


def test_flags_align_one_to_one_with_tokens():
    tok = _FakeTokenizer(["Give ", "amox", "icillin", " 500", "mg"])
    ids = [0, 1, 2, 3, 4]
    kinds, _, _ = flag_medical_tokens(tok, ids)
    assert len(kinds) == len(ids)
