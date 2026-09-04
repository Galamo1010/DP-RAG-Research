"""ChatDoctor data loader for the DP-RAG pipeline.

Replaces the synthetic `medical_dirichlet_documents` (test_data.py) with the
real ChatDoctor datasets required by the proposal:

    corpus  (private, FAISS + MIA target) = HealthCareMagic-100k  "output"  (doctor replies)
    queries (disjoint from corpus)         = iCliniq-10k           "input"   (patient questions)
                                             + "answer_icliniq" kept as the reference
                                               answer for BERTScore / ROUGE-L.

Query set and corpus come from different sources, so non-overlap is guaranteed.
"""

import json
import random
import re
from dataclasses import dataclass

# dprag.paths is the single owner of layout knowledge; ask it rather than
# computing a relative path here (that broke once already when this file moved).
from .paths import HEALTHCAREMAGIC_PATH, ICLINIQ_PATH, require_data


@dataclass
class Query:
    query: str          # patient question (iCliniq "input")
    reference: str      # real doctor answer (iCliniq "answer_icliniq"), for quality metrics


# --------------------------------------------------------------------------
# Reference damage
# --------------------------------------------------------------------------
# The iCliniq reference answers are damaged by the dataset's own construction,
# and the quality scores are computed against them. Measured over the 200-query
# sample the experiments use (query_seed=42): 16.5% truncated, 12.5% referring to
# an attachment, 1.5% both, 69.5% clean. See docs/notes/reference-damage.md.
#
# This lives here rather than in `quality` because it is a property of the
# DATASET, not of a metric: the same damaged reference depresses ROUGE-L and
# BERTScore alike, and a future third metric would inherit it too.

TRUNCATED = "truncated"
ATTACHMENT = "attachment"

# Function words are excluded so that ordinary sentences naming the platform
# ("Welcome to ChatDoctor forum", "Thanks for consulting ChatDoctor") are not
# counted as truncations. A truncation cuts mid-content, so the word before the
# splice is a content word -- often a half-eaten drug name ("Pseudoephe").
_FUNCTION_WORDS = frozenset({
    "to", "at", "on", "in", "for", "with", "from", "of", "the", "a", "an",
    "and", "or", "but", "is", "are", "was", "were", "be", "been", "using",
    "use", "used", "visit", "consult", "consulting", "contact", "ask",
    "asking", "join", "joining", "welcome", "thanks", "thank", "regards",
    "team", "here", "this", "that", "our", "your", "my", "we", "you", "i",
})

_TRUNCATION_RE = re.compile(r"\b([A-Za-z][A-Za-z'-]*)\s+ChatDoctor\b", re.IGNORECASE)
_ATTACHMENT_RE = re.compile(r"attachment removed to protect (?:patient )?identity",
                            re.IGNORECASE)


def reference_damage(reference: str) -> set[str]:
    """Which kinds of damage this reference carries. Empty set means clean.

    Two kinds, returned separately rather than as one boolean, because they
    break a quality score for different reasons and may not affect a semantic
    metric equally:

    `TRUNCATED` -- a sentence is cut, often mid-word, with the string
    "ChatDoctor" spliced in, and what gets eaten is frequently the drug name
    ("Sudafed (Pseudoephe ChatDoctor."). Content is MISSING, so a metric that
    matches meaning may be hurt more than one matching word subsequences.

    `ATTACHMENT` -- the doctor is describing an image the model never sees. The
    reference is intact; it is the TASK that is impossible, so no answer can
    score well and the ceiling is unreachable rather than the text being broken.

    Heuristic, and deliberately conservative in one direction: it will miss a
    truncation that spliced nothing in, and it will occasionally flag a sentence
    that legitimately names the platform after a content word. Both are stated
    wherever the resulting rates are reported.
    """
    kinds = set()
    for match in _TRUNCATION_RE.finditer(reference):
        if match.group(1).lower() not in _FUNCTION_WORDS:
            kinds.add(TRUNCATED)
            break
    if _ATTACHMENT_RE.search(reference):
        kinds.add(ATTACHMENT)
    return kinds


def load_corpus(limit: int | None = None, sample_seed: int | None = None) -> list[str]:
    """Doctor replies from HealthCareMagic-100k -> private corpus documents.

    `limit` caps the number of documents (useful for smoke tests; the full set is
    ~112k and embedding it into the vector store is the heavy part).
    `sample_seed`: if set, `limit` docs are drawn as a RANDOM sample (reproducible)
    instead of taking the first `limit`. Random sampling gives much better topic
    coverage of the corpus for a given size.
    """
    require_data()
    with open(HEALTHCAREMAGIC_PATH, encoding="utf-8") as f:
        data = json.load(f)
    docs = [row["output"].strip() for row in data if row.get("output", "").strip()]
    # De-duplicate while preserving order (PUPVectorStore also dedupes on add()).
    seen: set[str] = set()
    unique = [d for d in docs if not (d in seen or seen.add(d))]
    if limit is None:
        return unique
    if sample_seed is not None:
        return random.Random(sample_seed).sample(unique, min(limit, len(unique)))
    return unique[:limit]


def load_queries(n: int = 200, seed: int = 42) -> list[Query]:
    """Sample `n` patient questions from iCliniq-10k with a fixed seed.

    Keeps the real doctor answer (answer_icliniq) as the evaluation reference.
    """
    require_data()
    with open(ICLINIQ_PATH, encoding="utf-8") as f:
        data = json.load(f)
    rows = [
        Query(query=row["input"].strip(), reference=row.get("answer_icliniq", "").strip())
        for row in data
        if row.get("input", "").strip()
    ]
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def main():
    corpus = load_corpus(limit=5)
    queries = load_queries(n=3)
    print(f"corpus docs (showing 2 of full {len(load_corpus())}):")
    for d in corpus[:2]:
        print("  -", d[:100].replace("\n", " "))
    print(f"\nsampled queries ({len(queries)}):")
    for q in queries:
        print("  Q:", q.query[:90].replace("\n", " "))
        print("  ref:", q.reference[:90].replace("\n", " "))


if __name__ == "__main__":
    main()
