"""Canary probe on a laptop: does the secret come out of a RAG system with no DP?

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is a feasibility probe. It runs the two arms that need only ONE row of
generation -- RAG with every retrieved document and no DP, and NoRAG with none --
so the whole pipeline can be exercised end to end before any GPU is rented.

It is NOT the measurement, for one reason that cannot be engineered away:
`Llama-3.1-8B-Instruct` needs 16 GB of weights at bfloat16 and an 8 GB laptop card
cannot hold it. So this runs `Llama-3.2-1B-Instruct` -- same family, same chat
template, same tokenizer lineage, and therefore the closest available stand-in, but
an eighth of the parameters. Verbatim copying is not size-invariant: a smaller model
follows instructions less well and may parrot its context more, or retrieve from a
long context worse and parrot it less. The direction of that bias is unknown, so a
number from here indicates and does not measure.

The DP arms (plain DPRAG and strategy A) are absent because they need the k+1 = 41
row batch, which does not fit either. They stay on the A100.

WHAT IT DOES ESTABLISH, THOUGH
------------------------------
1. Whether canaries reach the output at all. If a no-DP RAG system on this corpus
   never reproduces a planted secret, the probe's premise is wrong and that is worth
   knowing before spending money, not after.
2. Whether the holdout stays at zero on real generated text -- the gate that
   licenses reading every later number. A detector that over-matches shows up here.
3. That retrieval, injection, generation and detection actually compose.

RETRIEVAL HERE MATCHES THE A100 RUN EXACTLY
-------------------------------------------
`PUPVectorStore.reseed_for` makes retrieval a function of (query, seed) alone
(ADR 0010), and the corpus is a function of (n_docs, corpus_seed, canary_seed). None
of those depend on the generator. So the documents this run sees are the same ones
`canary_run.py` will see, and a later comparison is about the model, not the
evidence.

WHY NoRAG RUNS TOO, DESPITE BEING A "POLE"
------------------------------------------
It costs one short single-row generation per query and it is the coincidence floor.
Without it, "the surname appeared in the output" has no baseline: the floor is what
separates reproduction from a model writing something that happens to collide. The
holdout covers fabrication of canary-shaped strings; NoRAG covers everything else
the model says on its own. They are different controls and both are nearly free.

Run:  uv run python experiments/canary_local_probe.py [n_queries] [model]
"""

from __future__ import annotations

import sys
import time

import torch
from transformers import GenerationConfig

from dprag import canary, prompts, run_record, trace
from dprag.chatdoctor import load_corpus, load_queries
from dprag.config import ExperimentConfig
from dprag.dp_model import DPModel
from dprag.pup_vector_store import PUPVectorStore, PUPVectorStoreConfig

CANARY_SEED = 20260927
LOCAL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

EXPERIMENT = ExperimentConfig(
    n_queries=100,
    canary_seed=CANARY_SEED,
    gen_dtype="bfloat16",
)


def generate_one(dp_model, conversation, cfg) -> str:
    """One greedy single-row generation. Same signature as stage3_poles.generate_one.

    Takes a DPModel rather than a bare model and tokenizer. The first version loaded
    both with AutoModelForCausalLM/AutoTokenizer and crashed on "the tokenizer does
    not have a padding token" -- because DPModel.tokenizer is the thing that sets
    `pad_token = eos_token` AND `padding_side='left'`. Patching the pad token here
    would have fixed the crash and left the padding side wrong, which for a
    single-row generation happens not to matter and would therefore never have
    surfaced. Going through DPModel means this probe and the A100 path tokenise
    identically by construction instead of by coincidence.
    """
    tokenizer = dp_model.tokenizer
    inputs = tokenizer.apply_chat_template(
        [conversation],
        tokenize=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
        continue_final_message=False,
        **prompts.TEMPLATE_KWARGS,
    ).to("cuda")
    input_len = inputs["input_ids"].shape[-1]
    if cfg.pad_token_id is None:
        cfg.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = dp_model.model.generate(**inputs, generation_config=cfg)
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True)


def main() -> None:
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else EXPERIMENT.n_queries
    model_id = sys.argv[2] if len(sys.argv) > 2 else LOCAL_MODEL
    exp = EXPERIMENT.with_(n_queries=n_queries, gen_model=model_id)

    print(f"=== canary local probe | {model_id} | {exp.gen_dtype} ===")
    print(f"canary_seed={exp.canary_seed}  n_docs={exp.n_docs}  "
          f"max_retrieve={exp.max_retrieve}  n_queries={n_queries}")
    print("FEASIBILITY PROBE, NOT THE MEASUREMENT -- 1B stand-in for the 8B that "
          "does not fit in 8 GB.\n", flush=True)

    # -- corpus -------------------------------------------------------------
    plain = load_corpus(limit=exp.n_docs, sample_seed=exp.corpus_seed)
    planted, holdout = canary.generate(len(plain), exp.canary_seed)
    corpus = canary.inject(plain, planted)
    det = canary.Detector(planted + holdout)

    store = PUPVectorStore(PUPVectorStoreConfig(
        model_id=exp.embed_model,
        top_p=exp.retrieval_top_p,
        top_k=exp.retrieval_top_k,
        min_score=exp.min_score,
        max_score=exp.max_score,
        epsilon=exp.eps_retrieval,
        max_retrieve=exp.max_retrieve,
        batch_size=exp.embed_batch_size,
        seed=exp.seed,
    ))
    for doc in corpus:
        store.add(doc)
    t0 = time.time()
    store.embeddings()
    print(f"embedded {len(corpus)} injected docs in {time.time() - t0:.0f}s\n",
          flush=True)

    # -- model --------------------------------------------------------------
    # DPModel, not a bare from_pretrained: it owns the tokenizer configuration the
    # rest of the project generates under. See generate_one.
    dp_model = DPModel(model_id=model_id, dtype=exp.gen_dtype)
    gen_cfg = GenerationConfig(do_sample=False, max_new_tokens=exp.max_new_tokens)

    # -- run ----------------------------------------------------------------
    queries = load_queries(n=n_queries, seed=exp.query_seed)
    rows, zero_doc = [], 0
    began = time.time()
    for i, q in enumerate(queries):
        store.reseed_for(q.query)
        documents = store.pup_retrieve(q.query)
        if not documents:
            zero_doc += 1
            continue
        torch.manual_seed(exp.seed)
        rag = generate_one(dp_model,
                           prompts.all_documents_chat(documents, q.query), gen_cfg)
        torch.manual_seed(exp.seed)
        norag = generate_one(dp_model,
                             prompts.norag_chat(q.query), gen_cfg)
        rows.append({
            "query": q.query,
            "n_documents": len(documents),
            # Same [index, similarity] shape every sweep writes, so canary_detect
            # and the review-sheet builder read this record the way they read any
            # other. Encoding consumes no randomness, so asking for the scores
            # here does not disturb the retrieval draw.
            "docs": trace.retrieval_trace(store, q.query, documents),
            "norag_text": norag,
            "rag_text": rag,
        })
        if (i + 1) % 10 == 0:
            print(f"[{i + 1:3}/{len(queries)}] {time.time() - began:5.0f}s  "
                  f"{len(rows)} with documents", flush=True)

    print(f"\n{len(rows)} queries generated, {zero_doc} zero-document "
          f"({(time.time() - began) / 60:.1f} min)\n")

    # -- detection ----------------------------------------------------------
    exposed = {planted[i].cid for r in rows for i, _ in r["docs"]}
    print(f"denominator: {len(exposed)} of {exp.n_docs} planted canaries reached "
          f"a context ({100 * len(exposed) / exp.n_docs:.1f}%)\n")

    summary = {}
    for arm, key in (("RAG_noDP", "rag_text"), ("NoRAG", "norag_text")):
        inj_fields, hold_fields, examples = {}, {}, []
        for r in rows:
            hits = det.find(r[key])
            if not hits:
                continue
            inj, hold = det.split(hits)
            for f, cids in inj.items():
                inj_fields.setdefault(f, set()).update(cids)
            for f, cids in hold.items():
                hold_fields.setdefault(f, set()).update(cids)
            if inj:
                own = {planted[i].cid for i, _ in r["docs"]}
                leaked = {c for cids in inj.values() for c in cids}
                examples.append((r, {f: sorted(c) for f, c in inj.items()},
                                 sorted(leaked - own)))
        summary[arm] = (inj_fields, hold_fields, examples)

    print("1. HOLDOUT (never planted; must be 0)")
    total_hold = 0
    for arm, (_, hold, _) in summary.items():
        n = sum(len(c) for c in hold.values())
        total_hold += n
        print(f"   {arm:<10} {n:>4}" + (f"  {dict((f, sorted(c)[:3]) for f, c in hold.items())}" if n else ""))
    print("   clean\n" if not total_hold else
          "   *** NOT ZERO -- the detector over-matches or the model fabricates. "
          "Nothing below is interpretable. ***\n")

    print(f"2. INJECTED hits, of {len(exposed)} canaries that reached a context")
    print(f"   {'arm':<10}" + "".join(f"{f:>10}" for f in canary.FIELDS) + f"{'any':>8}")
    for arm, (inj, _, _) in summary.items():
        any_cids = set().union(*inj.values()) if inj else set()
        print(f"   {arm:<10}" + "".join(f"{len(inj.get(f, ())):>10}" for f in canary.FIELDS)
              + f"{len(any_cids):>8}")
    print()
    for arm, (inj, _, ex) in summary.items():
        any_cids = set().union(*inj.values()) if inj else set()
        print(f"   {arm:<10} {len(any_cids)}/{len(exposed)} canaries leaked "
              f"({100 * len(any_cids) / max(1, len(exposed)):.2f}%) "
              f"across {len(ex)} of {len(rows)} queries")

    print("\n3. EXAMPLES")
    for arm, (_, _, ex) in summary.items():
        if not ex:
            print(f"   {arm}: none")
            continue
        for r, fields, elsewhere in ex[:4]:
            print(f"\n   [{arm}] fields={fields}")
            print(f"     query : {r['query'][:100]}")
            print(f"     output: {r['rag_text'][:240]}")
            if elsewhere:
                print(f"     !! {elsewhere} were not in this query's context -- "
                      "index misalignment, not cross-query leakage")

    out = run_record.write(
        "canary_local_probe",
        exp,
        metrics={
            "phase": "canary probe (local feasibility)",
            "canary_seed": exp.canary_seed,
            "n_exposed_canaries": len(exposed),
            "n_zero_document_queries": zero_doc,
            "arms": {a: {"injected": {f: sorted(c) for f, c in inj.items()},
                         "holdout": {f: sorted(c) for f, c in hold.items()}}
                     for a, (inj, hold, _) in summary.items()},
            "note": (
                "FEASIBILITY PROBE on a 1B stand-in, not the measurement: the 8B "
                "model does not fit in 8 GB. No DP arms (the k+1 batch does not "
                "fit either). Retrieval is identical to what canary_run.py will "
                "see, because reseed_for keys on (query, seed) alone."
            ),
        },
        per_item=rows,
        filename=f"canary_local_probe_{model_id.split('/')[-1]}_{len(rows)}q",
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    main()
