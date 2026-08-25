"""Heuristic flagging of clinically-loaded token positions.

Stage 2.5 asks a safety question about the loose end of strategy B: when the
pre-filter skips ε at a position, and that position turns out to be part of a drug
name, a dose or a condition, the medical content came from the model's prior
rather than from the retrieved documents. Those are the positions where a wrong
skip could matter clinically.

TWO KINDS OF MATCH, BECAUSE ONLY ONE CAN BE VETTED
--------------------------------------------------
**Word-like** matches are drug and condition names. They are single tokens, so a
vocabulary can be asked whether the word exists at all.

**Pattern-like** matches are doses and frequencies ("500 mg", "twice daily",
"po tds"). They are structural, not lexical: no vocabulary can say whether
"197 mg" is a real dose. On DP-degraded text a stray number and a stray unit will
occasionally land next to each other and match.

The two are reported separately for exactly that reason. A combined rate hides
that half of it cannot be checked.

TWO FILTERS, DOING DIFFERENT JOBS
---------------------------------
The morphological rules match *shape*, so they flag anything shaped like a drug.
Two different things get past them and each needs its own guard:

* **Ordinary English that looks clinical.** "diagnosis" ends in -osis (13,955
  occurrences in the corpus), "sympathy" in -pathy, "nostalgia" in -algia. A
  vocabulary cannot help here — these are real words. They are blocked by name in
  `_NOT_CONDITIONS`, which was assembled by frequency-ranking every corpus word
  that matches the condition rules rather than by guesswork.

* **DP-noise artifacts shaped like medicine.** "Totosis", "Sonopathy",
  "Alleralgia", "Hepitis", "iboprofen". These are not words at all, so a
  vocabulary rejects them cleanly: measured over 112,165 real doctor replies,
  every one of them occurs zero times. That is what `build_corpus_vocabulary`
  exists for, and why `find_medical_spans` takes a vocabulary rather than
  hard-coding one — passing None gives the morphological rules alone, which is
  what the unit tests pin.

Measured cost of the vocabulary filter: on real doctor replies 95% of matches
survive, the 5% lost being drugs too rare to clear the frequency threshold
(Beclometasone, Olmesartan, Temazepam). On plain-DPRAG output at ε=10 half the
matches are dropped, and on inspection every dropped one is an artifact.

WHAT THIS IS NOT
----------------
A regex heuristic, deliberately. The proposal's real instrument is scispaCy's
Medical Entity Retention Rate, and that belongs to Stage 5. Treat the counts here
as a warning signal to inspect by hand, never as a measured entity rate.

WHY SPANS RATHER THAN TOKENS
----------------------------
Tokenizers split words: "metformin" can arrive as "met" + "form" + "in", and
"500mg" as "500" + "mg". Testing each token's own text against a word list would
miss every fragment. So the text is reconstructed, entities are matched over the
whole string, and each token is flagged by whether its character span overlaps a
match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

WORD_KIND = "word"
PATTERN_KIND = "pattern"

# Drug-name endings. Chosen because each marks a pharmacological class, so they
# generalise past any fixed list of brand names.
#
# Two rules learned from the tests:
#   * A suffix needs >=3 preceding letters, or "April" matches -pril. Three, not
#     four, because "met|formin" only has three.
#   * Short or common endings are spelled out in their specific drug forms.
#     Bare "terol" flags cholesterol; bare "vir"/"nib"/"mab" flag ordinary words.
_DRUG_SUFFIXES = [
    "cillin", "mycin", "cycline", "oxacin", "prazole", "azole", "statin",
    "sartan", "pril", "olol", "dipine", "parin", "formin", "caine", "zepam",
    "profen", "codone", "morphone", "tidine", "triptan", "sone", "solone",
    "dronate", "tinib", "navir", "umab", "imab", "buterol", "meterol",
]

# Drugs whose names do not follow a class suffix, so no rule reaches them.
_DRUG_NAMES = [
    "insulin", "aspirin", "paracetamol", "acetaminophen", "warfarin", "cocaine",
    "morphine", "codeine", "prednisone", "levothyroxine", "amoxicillin",
    "azithromycin", "ranitidine", "salbutamol", "albuterol", "ibuprofen",
]

# Condition endings. The proposal names conditions alongside drugs and doses, and
# they were previously not covered at all. Same >=3-letter prefix guard as the
# drug rules.
_CONDITION_SUFFIXES = [
    "itis", "osis", "emia", "aemia", "pathy", "algia", "ectomy", "otomy",
    "plasia", "trophy", "penia", "megaly", "uria",
]

# Conditions the suffix rules cannot reach: either the stem is too short for the
# >=3-letter guard ("an|emia", "a|trophy") or the word carries no class suffix.
_CONDITION_NAMES = [
    "anemia", "anaemia", "atrophy", "asthma", "diabetes", "hypertension",
    "eczema", "psoriasis", "migraine", "pneumonia", "sepsis", "ulcer",
]

# Words that match a condition suffix but are not conditions. Assembled by
# frequency-ranking every corpus word matching `_CONDITION_SUFFIXES`, not by
# guesswork -- the first three are what that ranking actually surfaced:
#   diagnosis 13,955 | prognosis 973 | homeopathy 319
# The rest are ordinary English that the same rules would catch in other text.
_NOT_CONDITIONS = {
    "diagnosis", "prognosis", "homeopathy", "naturopathy", "osteopathy",
    "sympathy", "empathy", "apathy", "antipathy", "telepathy",
    "nostalgia", "academia", "bohemia", "osmosis", "symbiosis", "hypnosis",
    "metamorphosis",
}

# Doses, strengths and administration routes/frequencies.
_UNITS = r"(?:mg|mcg|ug|µg|g|kg|ml|mL|l|L|IU|iu|units?|tabs?|tablets?|capsules?|drops?|tsp|tbsp|puffs?)"
_FREQUENCIES = r"(?:bd|tds|od|bid|tid|qid|qd|prn|hs|po|iv|im|sos|stat)"

# Word-like: a vocabulary can vet the whole token.
WORDLIKE_PATTERN = re.compile(
    r"\b[A-Za-z]{3,}(?:" + "|".join(_DRUG_SUFFIXES + _CONDITION_SUFFIXES) + r")\b"
    r"|\b(?:" + "|".join(_DRUG_NAMES + _CONDITION_NAMES) + r")\b",
    re.IGNORECASE,
)

# Pattern-like: structural, so no vocabulary applies.
PATTERNLIKE_PATTERN = re.compile(
    # 500mg, 2.5 ml, 10 units
    rf"\b\d+(?:\.\d+)?\s*{_UNITS}\b"
    # mg/dl, mg/kg, ml/hr
    rf"|\b{_UNITS}\s*/\s*(?:{_UNITS}|dl|hr|day|min)\b"
    # twice daily, three times a day
    r"|\b(?:once|twice|thrice|\d+\s*times?)\s+(?:a\s+|per\s+)?(?:daily|day|week|month|hour)\b"
    # dosing abbreviations, as standalone words
    rf"|\b{_FREQUENCIES}\b"
    # blood pressure and similar readings
    r"|\b\d+\s*/\s*\d+\s*(?:mmHg|mm\s*Hg)\b",
    re.IGNORECASE,
)


class TokenMark(NamedTuple):
    """What a single token position carries, clinically.

    `is_first` marks the token that opens its span, and it is the field the
    safety analysis turns on. A multi-token drug name -- "Folliculitis" arriving
    as F+ol+lic+ul+itis -- contributes five clinical positions, but only the
    first is a decision: once "Follicul" is committed, English spelling leaves no
    alternative to "itis", so both instances agree there whatever the documents
    say. Counting all five inflates the medical skip rate against the ordinary
    one, which is mostly single-token words. Serialises to ["word", true].
    """

    kind: str        # WORD_KIND or PATTERN_KIND
    is_first: bool


@dataclass(frozen=True)
class MedicalSpan:
    """One clinically-loaded phrase located in a text."""

    start: int
    end: int
    text: str
    kind: str          # WORD_KIND or PATTERN_KIND


@lru_cache(maxsize=4)
def build_corpus_vocabulary(min_count: int = 3) -> frozenset[str]:
    """Lower-cased words appearing at least `min_count` times in the corpus.

    The corpus is 112,165 real doctor replies, which makes it a medical
    vocabulary in its own right -- and the only one available offline. The
    threshold is what separates a word from a one-off typo: at 1 a doctor's slip
    would admit the same slip when a DP-noise token reproduces it, while at 10
    real but uncommon conditions start to fall out ("orchitis" occurs 7 times).
    Three sits between those, and it is recorded as `ExperimentConfig.vocab_min_count`
    because the medical rates depend on it.

    Note for the report: this vocabulary is built from the *private* corpus. That
    is sound because the detector is a measurement instrument applied post hoc by
    the researcher, not part of the released system -- no information reaches an
    adversary through it. Stated here so the caveat sits next to the code.
    """
    from .chatdoctor import load_corpus

    counts: dict[str, int] = {}
    word = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
    for document in load_corpus():
        for token in word.findall(document):
            key = token.lower()
            counts[key] = counts.get(key, 0) + 1
    return frozenset(w for w, n in counts.items() if n >= min_count)


def find_medical_spans(
    text: str, vocabulary: frozenset[str] | set[str] | None = None
) -> list[MedicalSpan]:
    """Clinically-loaded phrases in `text`, in order and non-overlapping.

    `vocabulary`: if given, a word-like match is kept only when the matched word
    appears in it. Pattern-like matches are never filtered -- no vocabulary can
    judge "197 mg". Passing None applies the morphological rules alone, which is
    the behaviour the unit tests pin and the baseline the filter is measured
    against.
    """
    candidates = [
        MedicalSpan(m.start(), m.end(), m.group(), WORD_KIND)
        for m in WORDLIKE_PATTERN.finditer(text)
        if m.group().lower() not in _NOT_CONDITIONS
        and (vocabulary is None or m.group().lower() in vocabulary)
    ] + [
        MedicalSpan(m.start(), m.end(), m.group(), PATTERN_KIND)
        for m in PATTERNLIKE_PATTERN.finditer(text)
    ]

    # Two independent scans can overlap where one regex would have taken the
    # leftmost-longest match. Resolve the same way: earliest start wins, longer
    # wins a tie, and anything overlapping an accepted span is dropped.
    candidates.sort(key=lambda s: (s.start, -(s.end - s.start)))
    spans: list[MedicalSpan] = []
    for span in candidates:
        if not spans or span.start >= spans[-1].end:
            spans.append(span)
    return spans


def token_char_spans(tokenizer, token_ids: list[int]) -> list[tuple[int, int]]:
    """Each token's character span in the decoded text.

    Decodes incrementally rather than per token, because decoding a token alone
    loses the leading-space and word-piece handling that gives it its real extent.
    """
    spans: list[tuple[int, int]] = []
    previous = 0
    for i in range(len(token_ids)):
        text = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        spans.append((previous, len(text)))
        previous = len(text)
    return spans


def flag_medical_tokens(
    tokenizer,
    token_ids: list[int],
    vocabulary: frozenset[str] | set[str] | None = None,
) -> tuple[list[TokenMark | None], str, list[MedicalSpan]]:
    """Per-token clinical marks for a generated sequence.

    Returns (marks, decoded_text, spans). `marks[i]` is None or a TokenMark, so a
    caller can separate three things that get conflated when a position is just a
    bool: whether it is clinical, whether the evidence for that is vettable
    (word-like) or not (pattern-like), and whether it is the position where the
    model actually chose to say this rather than a continuation forced by
    spelling.
    """
    if not token_ids:
        return [], "", []
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    spans = find_medical_spans(text, vocabulary)
    if not spans:
        return [None] * len(token_ids), text, []

    marks: list[TokenMark | None] = []
    opened: set[tuple[int, int]] = set()
    for start, end in token_char_spans(tokenizer, token_ids):
        # A token counts if any part of it lies inside a span. Empty-width tokens
        # (some special tokens decode to nothing) never match. Word-like wins a
        # tie, being the vetted kind.
        overlapping = [
            s for s in spans if start < end and start < s.end and end > s.start
        ]
        if not overlapping:
            marks.append(None)
            continue
        span = next((s for s in overlapping if s.kind == WORD_KIND), overlapping[0])
        key = (span.start, span.end)
        marks.append(TokenMark(span.kind, key not in opened))
        opened.add(key)
    return marks, text, spans
