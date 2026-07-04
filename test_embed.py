"""Quick check for the embedding / retrieval part (pup_vector_store).

Run with:  uv run python test_embed.py

It does NOT download the medical dataset — it uses locally generated fake
documents, so it's a fast way to confirm that:
  1. the embedding model loads (on GPU if available, else CPU),
  2. mini-batch encoding gives the SAME result as encoding everything at once,
  3. peak memory stays bounded,
  4. DP retrieval returns sensible documents.
"""

import time

import torch
from termcolor import cprint

from pup_vector_store import PUPVectorStore, PUPVectorStoreConfig
from test_data import hair_color_documents


def _peak_memory_mb() -> float | None:
    """Best-effort peak memory reading (CUDA if present, else RSS via psutil)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def test_device_and_load():
    cprint("\n[1] Loading embedding model...", "cyan")
    store = PUPVectorStore(PUPVectorStoreConfig(top_p=0.02, epsilon=0.5))
    device = next(store.model.parameters()).device
    cprint(f"    model_id = {store.model_id}", "white")
    cprint(f"    device   = {device}", "green")
    cprint(f"    batch_size = {store.batch_size}", "white")
    return store


def test_embeddings_shape(store: PUPVectorStore, docs: list[str]):
    cprint("\n[2] Encoding documents (mini-batched)...", "cyan")
    for doc in docs:
        store.add(doc)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    embeddings = store.embeddings()
    dt = time.time() - t0
    cprint(f"    {len(docs)} docs -> embeddings shape {tuple(embeddings.shape)} in {dt:.2f}s", "green")
    # Normalized embeddings must have unit L2 norm.
    norms = torch.linalg.norm(embeddings, dim=1)
    cprint(f"    L2 norms in [{norms.min():.4f}, {norms.max():.4f}] (should be ~1.0)", "white")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "embeddings are not unit-normalized"
    mem = _peak_memory_mb()
    if mem is not None:
        cprint(f"    peak memory ~ {mem:.0f} MB", "yellow")


def test_batching_equivalence(docs: list[str]):
    """Batched encoding must match single-shot encoding (numerically)."""
    cprint("\n[3] Checking batched vs single-shot equivalence...", "cyan")
    small = docs[:64]

    store_batched = PUPVectorStore(PUPVectorStoreConfig(top_p=0.02, batch_size=8))
    store_single = PUPVectorStore(PUPVectorStoreConfig(top_p=0.02, batch_size=10_000))

    emb_batched = store_batched.encode(small)
    emb_single = store_single.encode(small)

    max_diff = (emb_batched - emb_single).abs().max().item()
    cprint(f"    max abs difference = {max_diff:.2e}", "white")
    assert max_diff < 1e-3, "batched and single-shot embeddings differ too much!"
    cprint("    OK: batch_size does not change the result.", "green")


def test_retrieval(store: PUPVectorStore):
    cprint("\n[4] DP retrieval demo...", "cyan")
    query = "What color is the person's hair?"
    retrieved = store.pup_retrieve(query)
    cprint(f"    query: {query}", "blue")
    cprint(f"    retrieved {len(retrieved)} documents (showing up to 5):", "green")
    for doc in retrieved[:5]:
        cprint(f"      - {doc}", "grey")


if __name__ == "__main__":
    docs = hair_color_documents(n=200)
    store = test_device_and_load()
    test_embeddings_shape(store, docs)
    test_batching_equivalence(docs)
    test_retrieval(store)
    cprint("\nAll embedding checks passed.\n", "green")
