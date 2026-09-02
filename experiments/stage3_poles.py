"""Two reference points the routed output can be located against.

The strongest objection to this project's result is one sentence:

> You skipped differential privacy at 82% of positions and the answer got better.
> Of course it did. You did less DP.

Answering it needs evidence that the documents still shape the answer -- that the
pre-filter skips where they do not matter rather than skipping their influence
away. Quality metrics cannot supply that: ROUGE-L and BERTScore ask whether the
answer resembles the doctor's, and a system that ignored every document could score
well on both.

So this generates the two ends of the axis, using objects CONTEXT.md already
defines:

    NoRAG instance   question only, no documents      -- no document influence
    RAG instance     question plus every document     -- full document influence,
                                                        no privacy machinery

Where the routed output sits between them is the answer. Close to the RAG pole and
the documents are still doing work; close to the NoRAG pole and the method is
merely not doing RAG.

BOTH POLES ARE INDEPENDENT OF STRATEGY AND EPSILON, so this runs once and every
configuration is compared against it. Two single-row generations per query, no k+1
aggregation: around fifty minutes for two hundred queries, against the fifteen
hours the main phase costs.

WHY THE DOCUMENTS COME FROM A PREVIOUS RUN
------------------------------------------
DP retrieval is stochastic. Retrieving again here would hand the poles a different
document set from the one the routed run saw, and the comparison would be
confounded by exactly the thing it is trying to isolate. Instead the corpus indices
recorded by `trace.retrieval_trace` are read back and the same documents
reconstructed -- which is why those indices are stored rather than the text.

WHY THE POLES DECODE GREEDILY
-----------------------------
The router emits `scores[NORAG_ROW].argmax()` at every position the pre-filter
skips, so 87% of a strategy-A answer is greedy NoRAG text. Sampling the poles
would measure that answer against a *differently drawn* NoRAG answer, and most of
the gap would be the draw rather than the documents.

The failure that matters is the one in the worst case. Suppose a configuration
ignores the documents entirely and reproduces the greedy NoRAG answer exactly.
Greedy poles report a similarity near 1.0 and the objection is conceded, which is
the honest outcome. Sampled poles report a much lower one -- not because the
documents did anything, but because the two texts diverge on the sampling -- and
that reads as "the answer is not NoRAG". The metric would confirm document
influence precisely where there is none.

So both poles decode greedily, matching the router's free path. `temperature` is
therefore unset: it does nothing under argmax, and passing it only invites a
warning. The RAG pole still receives every document; only the token rule changed.

    uv run python experiments/stage3_poles.py [source_record_name]
"""

import sys
import time

import torch

from dprag import paths, prompts, run_record
from dprag.bench import Bench
from dprag.chatdoctor import load_corpus
from dprag.config import ExperimentConfig
from transformers import GenerationConfig

# Whose retrieval to mirror. Any record written by dprag.sweep carries the corpus
# indices; the baseline run is the natural choice because every other
# configuration in that phase saw the same documents.
DEFAULT_SOURCE = "stage3_2_main_baseline_eps10"

EXPERIMENT = ExperimentConfig()


def generate_one(dp_model, conversation, generation_config) -> str:
    """One plain generation from one conversation. No DP, no k+1 batch."""
    tokenizer = dp_model.tokenizer
    model_inputs = tokenizer.apply_chat_template(
        [conversation],
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        continue_final_message=False,
        **prompts.TEMPLATE_KWARGS,
    ).to("cuda")
    input_len = model_inputs["input_ids"].shape[-1]
    if generation_config.pad_token_id is None:
        generation_config.pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        out = dp_model.model.generate(
            **model_inputs, generation_config=generation_config
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    exp = EXPERIMENT

    source_path = paths.results_dir() / f"{source}.json"
    if not source_path.exists():
        raise SystemExit(
            f"{source_path.name} not found.\n"
            "The poles must reuse the documents a routed run actually retrieved, "
            "or the comparison is confounded by different evidence. Run a phase "
            "first, or pass a record name:\n"
            "    uv run python experiments/stage3_poles.py <record_name>"
        )
    source_record = run_record.load(source_path)

    # The corpus is reproducible from these two parameters, which is what makes
    # stored indices sufficient. Take them from the source record rather than the
    # local config, so a mismatch cannot pass silently.
    n_docs = source_record.param("n_docs", exp.n_docs)
    corpus_seed = source_record.param("corpus_seed", exp.corpus_seed)
    if (n_docs, corpus_seed) != (exp.n_docs, exp.corpus_seed):
        print(f"note: rebuilding the source run's corpus (n_docs={n_docs}, "
              f"corpus_seed={corpus_seed}), which differs from this config")
    corpus = load_corpus(limit=n_docs, sample_seed=corpus_seed)

    print(f"=== Stage 3 poles | source={source} | model={exp.gen_model} ===")
    print(f"{len(source_record.per_item)} queries, documents reused by index\n",
          flush=True)

    bench = Bench.build(exp)
    # Greedy, not sampled -- see WHY THE POLES DECODE GREEDILY above. Built here
    # rather than via make_generation_config, which the Stage 2 scripts share and
    # which must keep sampling.
    cfg = GenerationConfig(do_sample=False, max_new_tokens=exp.max_new_tokens)

    rows = []
    for i, item in enumerate(source_record.per_item):
        question = item["query"]
        documents = [corpus[index] for index, _ in item.get("docs", [])]
        if not documents:
            continue

        started = time.time()
        # Same seed for both poles and for the routed run they will be compared
        # against, so a difference between them is the prompt and not the draw.
        torch.manual_seed(exp.seed)
        norag_text = generate_one(bench.dp_model, prompts.norag_chat(question), cfg)
        torch.manual_seed(exp.seed)
        rag_text = generate_one(
            bench.dp_model, prompts.all_documents_chat(documents, question), cfg
        )

        rows.append({
            "query": question,
            "n_documents": len(documents),
            "norag_text": norag_text,
            "rag_text": rag_text,
        })
        if (i + 1) % 25 == 0:
            print(f"[{i+1:3}/{len(source_record.per_item)}] "
                  f"{time.time()-started:4.1f}s", flush=True)

    out = run_record.write(
        "stage3_poles",
        exp,
        metrics={
            "source_record": source,
            "n_queries": len(rows),
            "poles": "norag_text = no documents; rag_text = all documents, no DP",
            "note": (
                "Reference points for locating routed output. Both are independent "
                "of strategy and epsilon, so every configuration compares against "
                "this one file. Documents reused by corpus index from the source "
                "record, so the comparison is not confounded by a different draw."
            ),
        },
        per_item=rows,
        filename=f"stage3_poles_{len(rows)}q",
    )
    print(f"\nSaved -> {out}")
    print("Locate routed output against these with experiments/stage3_score.py.")


if __name__ == "__main__":
    main()
