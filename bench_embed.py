"""Benchmark: embed the ENTIRE medical dataset and time it.

Run with:  uv run python bench_embed.py
Optionally override the batch size:  uv run python bench_embed.py 64

First run downloads the dataset (sarus-tech/medical_dirichlet_phi3) and the
embedding model. It reports how long it takes to embed every document.
"""

import sys
import time

import torch
from termcolor import cprint

from pup_vector_store import PUPVectorStore, PUPVectorStoreConfig
from test_data import medical_dirichlet_documents


def _peak_memory_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    cprint("Loading full medical dataset...", "cyan")
    t0 = time.time()
    docs = medical_dirichlet_documents()
    cprint(f"  {len(docs)} documents loaded in {time.time() - t0:.1f}s", "green")

    store = PUPVectorStore(PUPVectorStoreConfig(top_p=0.02, epsilon=0.5, batch_size=batch_size))
    device = next(store.model.parameters()).device
    cprint(f"  device = {device} | batch_size = {batch_size}", "yellow")

    cprint("Adding documents to the store...", "cyan")
    for doc in docs:
        store.add(doc)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cprint("Embedding ALL documents...", "cyan")
    t0 = time.time()
    embeddings = store.embeddings()
    dt = time.time() - t0

    n = len(docs)
    cprint(f"\n=== Result ===", "white")
    cprint(f"  documents      : {n}", "white")
    cprint(f"  embeddings     : {tuple(embeddings.shape)}", "white")
    cprint(f"  total time     : {dt:.1f}s", "green")
    cprint(f"  throughput     : {n / dt:.1f} docs/s", "green")
    mem = _peak_memory_mb()
    if mem is not None:
        cprint(f"  peak memory    : {mem:.0f} MB", "yellow")


if __name__ == "__main__":
    main()
