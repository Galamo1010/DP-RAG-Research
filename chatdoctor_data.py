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
import os
import random
from dataclasses import dataclass

# ChatDoctor-main sits next to the dp-rag folder.
_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "ChatDoctor-main")
)
HEALTHCAREMAGIC_PATH = os.path.join(_DATA_DIR, "HealthCareMagic-100k.json")
ICLINIQ_PATH = os.path.join(_DATA_DIR, "iCliniq-10k.json")


@dataclass
class Query:
    query: str          # patient question (iCliniq "input")
    reference: str      # real doctor answer (iCliniq "answer_icliniq"), for quality metrics


def load_corpus(limit: int | None = None, sample_seed: int | None = None) -> list[str]:
    """Doctor replies from HealthCareMagic-100k -> private corpus documents.

    `limit` caps the number of documents (useful for smoke tests; the full set is
    ~112k and embedding it into the vector store is the heavy part).
    `sample_seed`: if set, `limit` docs are drawn as a RANDOM sample (reproducible)
    instead of taking the first `limit`. Random sampling gives much better topic
    coverage of the corpus for a given size.
    """
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
