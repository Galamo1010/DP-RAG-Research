"""Heuristic flagging of clinically-loaded token positions.

Stage 2.5 asks a safety question about the loose end of strategy B: when the
pre-filter skips ε at a position, and that position turns out to be part of a drug
name or a dose, the medical content came from the model's prior rather than from
the retrieved documents. Those are the positions where a wrong skip could matter
clinically.

WHAT THIS IS NOT
----------------
This is a regex heuristic, deliberately. The proposal's real instrument is
scispaCy's Medical Entity Retention Rate, and that belongs to Stage 5. Using it
here would pull a Stage 5 dependency forward to answer a question that only needs
"are we anywhere near drug names and doses?". Treat the counts below as a warning
signal to inspect by hand, never as a measured entity rate.

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

# Doses, strengths and administration routes/frequencies.
_UNITS = r"(?:mg|mcg|ug|µg|g|kg|ml|mL|l|L|IU|iu|units?|tabs?|tablets?|capsules?|drops?|tsp|tbsp|puffs?)"
_FREQUENCIES = r"(?:bd|tds|od|bid|tid|qid|qd|prn|hs|po|iv|im|sos|stat)"

_PATTERNS = [
    # 500mg, 2.5 ml, 10 units
    rf"\b\d+(?:\.\d+)?\s*{_UNITS}\b",
    # mg/dl, mg/kg, ml/hr
    rf"\b{_UNITS}\s*/\s*(?:{_UNITS}|dl|hr|day|min)\b",
    # twice daily, three times a day
    r"\b(?:once|twice|thrice|\d+\s*times?)\s+(?:a\s+|per\s+)?(?:daily|day|week|month|hour)\b",
    # dosing abbreviations, as standalone words
    rf"\b{_FREQUENCIES}\b",
    # blood pressure and similar readings
    r"\b\d+\s*/\s*\d+\s*(?:mmHg|mm\s*Hg)\b",
    # drug names by class suffix
    r"\b[A-Za-z]{3,}(?:" + "|".join(_DRUG_SUFFIXES) + r")\b",
    # drugs with no class suffix to match on
    r"\b(?:" + "|".join(_DRUG_NAMES) + r")\b",
]

MEDICAL_PATTERN = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def find_medical_spans(text: str) -> list[tuple[int, int, str]]:
    """Character spans of clinically-loaded phrases: (start, end, matched text)."""
    return [(m.start(), m.end(), m.group()) for m in MEDICAL_PATTERN.finditer(text)]


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


def flag_medical_tokens(tokenizer, token_ids: list[int]) -> tuple[list[bool], str, list[tuple[int, int, str]]]:
    """Per-token flags for "this token sits inside a drug name or a dose".

    Returns (flags, decoded_text, matched_spans) so a caller can both count and
    show what was matched -- the counts are only interpretable next to the text
    that produced them.
    """
    if not token_ids:
        return [], "", []
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    matches = find_medical_spans(text)
    if not matches:
        return [False] * len(token_ids), text, []

    spans = token_char_spans(tokenizer, token_ids)
    flags = []
    for start, end in spans:
        # A token counts if any part of it lies inside a match. Empty-width tokens
        # (some special tokens decode to nothing) never match.
        overlaps = start < end and any(start < m_end and end > m_start for m_start, m_end, _ in matches)
        flags.append(overlaps)
    return flags, text, matches
