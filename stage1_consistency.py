"""Stage 1.2 -- baseline "document-independent position" analysis.

For every autoregressive step of the ORIGINAL DPRAG generation we record, WITHOUT
touching the privacy aggregation logic (計畫書 1.2 requirement):

  * NoRAG argmax token   -- argmax of the public-prior stream (row 0 of the k+1
                            streams, i.e. scores[0] in DPLogitsAggregator._dp_call)
  * DPRAG output token    -- the token DPRAG actually emits (sampled, temperature fixed)
  * DPRAG greedy token    -- argmax of the aggregated distribution (deterministic,
                            low-variance reference)

Per query, the consistency rate = fraction of positions where NoRAG argmax matches
the DPRAG token. Averaged over >=200 queries this is the THEORETICAL UPPER BOUND on
the epsilon a RAG/NoRAG pre-filter could save (the project's contribution).

Non-invasive: two pass-through LogitsProcessors are placed around the unmodified
DPLogitsAggregator, so the DP generation itself is byte-for-byte the baseline.

Requires a CUDA GPU. Run:  uv run python stage1_consistency.py
"""

import json
import os
import statistics as st
import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from dp_rag_engine import DPRAGEngine
from pup_vector_store import PUPVectorStoreConfig
from dp_model import DPGenerationConfig
from chatdoctor_data import load_corpus, load_queries
import experiment_params as P

N_DOCS = 10000
N_QUERIES = 200          # proposal requires >= 200 for a stable consistency rate
MAX_RETRIEVE = 10
CORPUS_SEED = 7
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GEN_EPSILON = 10.0


class _NoRagRecorder(LogitsProcessor):
    """Runs BEFORE the aggregator; sees the raw [k+1, vocab] logits.

    Row 0 is the NoRAG / public-prior stream (see DPLogitsAggregator._dp_call).
    Pass-through: returns scores unchanged so the aggregator behaves exactly as baseline.
    """
    def __init__(self):
        self.norag_argmax: list[int] = []

    def __call__(self, input_ids, scores):
        self.norag_argmax.append(int(scores[0].argmax()))
        return scores


class _GreedyRecorder(LogitsProcessor):
    """Runs AFTER the aggregator; sees the aggregated [1, vocab] distribution."""
    def __init__(self):
        self.dprag_greedy: list[int] = []

    def __call__(self, input_ids, scores):
        self.dprag_greedy.append(int(scores[0].argmax()))
        return scores


def _build_messages(docs: list[str], question: str) -> list[list[dict]]:
    # Identical to DPModel.dp_chat so the measured baseline is faithful.
    return [
        [
            {'role': 'system', 'content': 'You give a short response based on a predefined set documents.'},
            {'role': 'user', 'content': f'{question}'},
        ]
    ] + [
        [
            {'role': 'system', 'content': f'You give a short responses based on this document or a predefined set of similar documents.\nDocument:\n"{doc}"'},
            {'role': 'user', 'content': f'{question}'},
        ]
        for doc in docs
    ]


def _generate_with_recording(dp_model, docs, question, cfg):
    """Replicates DPModel.dp_chat but with recorders + returns emitted token ids."""
    messages = _build_messages(docs, question)
    model_inputs = dp_model.tokenizer.apply_chat_template(
        messages, tokenize=True, padding=True, return_tensors='pt',
        return_dict=True, add_generation_prompt=True, continue_final_message=False,
    ).to('cuda')
    input_len = model_inputs['input_ids'].shape[-1]

    pre = _NoRagRecorder()
    post = _GreedyRecorder()
    aggregator = dp_model.dp_logits_aggregator(cfg)   # unmodified baseline aggregator
    processors = LogitsProcessorList([pre, aggregator, post])

    with torch.no_grad():
        out = dp_model.model.generate(
            **model_inputs, generation_config=cfg, logits_processor=processors,
        )
    emitted = out[0, input_len:].tolist()

    # Drop a trailing EOS so it doesn't count as a (mis)matched position.
    eos = dp_model.tokenizer.eos_token_id
    if emitted and emitted[-1] == eos:
        emitted = emitted[:-1]

    text = dp_model.tokenizer.decode(emitted, skip_special_tokens=True)
    return emitted, pre.norag_argmax, post.dprag_greedy, text


def _consistency(a: list[int], b: list[int]) -> tuple[int, int]:
    n = min(len(a), len(b))
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches, n


def main():
    print(f"=== Stage 1.2 consistency | model={GEN_MODEL} | {N_QUERIES} queries ===")
    corpus = load_corpus(limit=N_DOCS, sample_seed=CORPUS_SEED)
    queries = load_queries(n=N_QUERIES, seed=P.QUERY_SEED)

    cfg = DPGenerationConfig(
        temperature=P.TEMPERATURE, max_new_tokens=P.MAX_NEW_TOKENS,
        alpha=P.ALPHA, omega=P.OMEGA, epsilon=GEN_EPSILON, delta=P.DELTA,
    )
    engine = DPRAGEngine(
        pup_vector_store_config=PUPVectorStoreConfig(
            model_id=P.EMBED_MODEL, top_p=P.RETRIEVAL_TOP_P, epsilon=P.EPS_RETRIEVAL,
            max_retrieve=MAX_RETRIEVE, batch_size=P.EMBED_BATCH_SIZE,
        ),
        model_id=GEN_MODEL, dp_generation_config=cfg,
    )

    print(f"Embedding {N_DOCS} corpus docs ...")
    t0 = time.time()
    for doc in corpus:
        engine.add(doc)
    engine.pup_vector_store.embeddings()
    print(f"Embedded in {time.time() - t0:.1f}s\n")

    per_query = []
    for i, q in enumerate(queries):
        t = time.time()
        docs = engine.pup_retrieve(q.query)
        emitted, norag, greedy, text = _generate_with_recording(
            engine.dp_model, docs, q.query, cfg,
        )
        m_out, n_out = _consistency(norag, emitted)      # proposal-literal (sampled)
        m_grd, n_grd = _consistency(norag, greedy)       # deterministic reference
        cr_out = m_out / n_out if n_out else 0.0
        cr_grd = m_grd / n_grd if n_grd else 0.0
        # Decode the token streams so the rates are eyeball-checkable. Note the
        # NoRAG/greedy "text" is a per-step overlay on DPRAG's shared prefix
        # (teacher-forced), not an independent autoregressive generation.
        tok = engine.dp_model.tokenizer
        norag_text = tok.decode(norag, skip_special_tokens=True)
        greedy_text = tok.decode(greedy, skip_special_tokens=True)
        per_query.append({
            "query": q.query,
            "n_retrieved": len(docs),
            "n_tokens": n_out,
            "consistency_sampled": cr_out,
            "consistency_greedy": cr_grd,
            "dprag_text": text,                  # DPRAG's actual (sampled) output
            "norag_argmax_text": norag_text,     # NoRAG top token per step, decoded
            "dprag_greedy_text": greedy_text,    # DPRAG aggregated argmax per step, decoded
        })
        print(f"[{i+1:3}/{N_QUERIES}] k={len(docs):2} tok={n_out:3} "
              f"consist(sampled)={cr_out:.2f} consist(greedy)={cr_grd:.2f} "
              f"{time.time()-t:4.1f}s")

    # ---- aggregate: the theoretical epsilon-savings upper bound ----
    cs = [r["consistency_sampled"] for r in per_query]
    cg = [r["consistency_greedy"] for r in per_query]
    summary = {
        "n_queries": len(per_query),
        "mean_consistency_sampled": st.mean(cs),
        "median_consistency_sampled": st.median(cs),
        "mean_consistency_greedy": st.mean(cg),
        "median_consistency_greedy": st.median(cg),
    }
    print("\n=== SUMMARY (consistency rate = theoretical epsilon-savings upper bound) ===")
    print(f"  sampled : mean={summary['mean_consistency_sampled']:.3f}  "
          f"median={summary['median_consistency_sampled']:.3f}")
    print(f"  greedy  : mean={summary['mean_consistency_greedy']:.3f}  "
          f"median={summary['median_consistency_greedy']:.3f}")

    os.makedirs("results", exist_ok=True)
    out = f"results/stage1_consistency_{N_DOCS}x{N_QUERIES}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": GEN_MODEL, "n_docs": N_DOCS, "n_queries": N_QUERIES,
                "max_retrieve": MAX_RETRIEVE, "temperature": P.TEMPERATURE,
                "max_new_tokens": P.MAX_NEW_TOKENS, "gen_epsilon": GEN_EPSILON,
            },
            "summary": summary,
            "per_query": per_query,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
