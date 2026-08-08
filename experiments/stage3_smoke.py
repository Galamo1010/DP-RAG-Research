"""Stage 3.1 -- router verification, and the trajectory question it raised.

Two jobs.

MECHANICAL CHECKS. The router's unit tests run against a fake model, so they
establish the routing logic and the epsilon accounting but say nothing about how a
real transformers KV cache behaves. Two properties stay unverified until this runs:
feeding a backlog of N tokens in ONE forward must leave the cache where N single
forwards would, and position_ids derived from a left-padded mask must place each
row where generate() would. Both fail silently -- the cache ends up conditioned on
a subtly wrong prefix and the answer merely comes out worse, which is
indistinguishable from DP noise.

THE TRAJECTORY COMPARISON. The first smoke run turned up something the mechanical
checks were not looking for: strategy B_k20_t0.7 triggered on 16% of positions,
against the 74.5% Stage 2.4 measured for the same configuration. The likely cause
is that Stage 2 drove generation with the NoRAG instance, so its context stayed
clean, whereas the router walks the routed trajectory -- and once a DP-sampled
token lands, the context degrades, the two instances diverge further, more
positions get paid, and more noise arrives. A feedback loop.

If that is right, every trigger rate measured so far is optimistic, which matters
for Stage 3's predictions. So this measures both drivers on the SAME query with the
SAME retrieved documents, isolating the trajectory as the only difference.

Requires a CUDA GPU. Run:  uv run python experiments/stage3_smoke.py
"""

import statistics as st

import torch

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.dual_instance import make_generation_config, run_dual_instance
from dprag.router import Router, _Stream
from dprag.strategies import PrefilterDecision, make_strategy_b, strategy_a
from dprag import prompts

# 10k corpus so retrieval mostly returns documents; the 500-doc first run left
# most queries at k=0, where every strategy fires trivially and says nothing.
EXPERIMENT = ExperimentConfig(n_docs=10000, n_queries=10, max_new_tokens=64)

ALWAYS_AGREE = lambda rag, norag: PrefilterDecision(True, int(norag.argmax()), 1.0)
NEVER_AGREE = lambda rag, norag: PrefilterDecision(False, int(norag.argmax()), 0.0)

STRATEGIES = {
    "A": strategy_a,
    "B_k20_t0.7": make_strategy_b(20, 0.7),
}

# What Stage 2.4 measured for the same configurations, on the NoRAG trajectory.
STAGE2_TRIGGER = {"A": 0.875, "B_k20_t0.7": 0.745}


def check_chunked_catch_up(dp_model, documents, question) -> bool:
    """One forward over N tokens must equal N forwards of one token.

    The property the backlog exists to exploit. Checked against the real cache,
    because it is a statement about transformers rather than about the router.
    """
    encoded = dp_model.tokenizer.apply_chat_template(
        prompts.dprag_chat_batch(documents, question),
        tokenize=True, padding=True, return_tensors="pt",
        return_dict=True, add_generation_prompt=True, continue_final_message=False,
    )
    device = dp_model.model.device
    ids, mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
    backlog = [11, 22, 33, 44]

    chunked = _Stream(dp_model.model, ids, mask)
    chunked.advance([])
    chunked_logits = chunked.advance(backlog)

    stepwise = _Stream(dp_model.model, ids, mask)
    stepwise.advance([])
    for token in backlog:
        stepwise_logits = stepwise.advance([token])

    gap = (chunked_logits - stepwise_logits).abs().max().item()
    ok = gap < 1e-2
    print(f"  chunked vs stepwise catch-up: max|delta logit| = {gap:.5f}  "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  -> the backlog catch-up does NOT preserve the cache; fix before "
              "running any measurement.")
    return ok


def main():
    exp = EXPERIMENT
    print(f"=== Stage 3.1 router smoke | model={exp.gen_model} | "
          f"{exp.n_queries} queries x {exp.max_new_tokens} tokens ===")
    bench = Bench.build(exp)

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=exp.gen_epsilon, delta=exp.delta,
    )
    plain_cfg = make_generation_config(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens
    )

    # Retrieve ONCE per query and reuse. Retrieval is stochastic, so calling it
    # per strategy would hand each one a different document set and confound
    # "strategy" with "different evidence".
    fixed = []
    for q in bench.queries():
        fixed.append({"query": q.query, "docs": bench.retrieve(q.query)})
    with_docs = [f for f in fixed if f["docs"]]
    print(f"Retrieved once per query; {len(fixed) - len(with_docs)}/{len(fixed)} "
          f"got 0 documents (excluded below -- they fire trivially).\n")
    if not with_docs:
        raise SystemExit("every query retrieved 0 documents; nothing to measure")

    # ---- mechanical checks ------------------------------------------------
    print("--- cache mechanics ---")
    first = with_docs[0]
    if not check_chunked_catch_up(bench.dp_model, first["docs"], first["query"]):
        raise SystemExit("cache catch-up check failed")

    print("\n--- degenerate strategies, on real weights ---")
    for name, strategy in (("always_agree", ALWAYS_AGREE), ("never_agree", NEVER_AGREE)):
        torch.manual_seed(exp.seed)
        res = Router(bench.dp_model, strategy, dp_cfg).generate(
            first["docs"], first["query"]
        )
        print(f"  {name:12} paid={res.n_paid:2}/{res.n_steps:2}  "
              f"trigger={res.trigger_rate:.2f}  "
              f"eps={res.epsilon_usage:.3f}/{res.epsilon_budget:.3f}")
        if name == "always_agree":
            assert res.n_paid == 0 and res.epsilon_usage == 0.0
            assert res.emitted == res.norag_argmax[: len(res.emitted)]
            # Decisive diagnostic. Nothing was paid, so no DP noise entered: this
            # is the pure NoRAG instance speaking, seen through the router's own
            # cache handling, with a k-document RAG row padded alongside it.
            # Coherent here means the padding and position handling are sound and
            # the garbling seen in routed output is the feedback loop. Garbled
            # here means the NoRAG row is being corrupted by the length disparity
            # between the two rows, and the trigger rates above cannot be trusted.
            print(f"    (no DP noise -- pure NoRAG through the router, k="
                  f"{res.n_documents})")
            print(f"    {res.text[:160]}")

    # ---- trajectory comparison -------------------------------------------
    print(f"\n--- trajectory comparison ({len(with_docs)} queries with documents) ---")
    print("Same query, same documents; only the driver differs.\n")
    rows = []
    for i, f in enumerate(with_docs):
        row = {"query": f["query"], "k": len(f["docs"])}

        # Stage 2's method: NoRAG drives, context stays clean.
        torch.manual_seed(exp.seed)
        norag_driven = run_dual_instance(
            bench.dp_model, f["docs"], f["query"], plain_cfg, STRATEGIES
        )
        for name in STRATEGIES:
            row[f"norag_{name}"] = norag_driven.trigger_rate(name)

        # The real thing: the router walks the routed trajectory.
        for name, strategy in STRATEGIES.items():
            torch.manual_seed(exp.seed)
            routed = Router(bench.dp_model, strategy, dp_cfg).generate(
                f["docs"], f["query"]
            )
            row[f"routed_{name}"] = routed.trigger_rate
            if name == "A":
                row["text"] = routed.text

        rows.append(row)
        print(f"  [{i+1:2}/{len(with_docs)}] k={row['k']:2}  "
              + "  ".join(
                  f"{n}: NoRAG={row[f'norag_{n}']:.2f} routed={row[f'routed_{n}']:.2f}"
                  for n in STRATEGIES
              ))

    print(f"\n{'strategy':>12} | {'NoRAG-driven':>12} | {'routed':>8} | "
          f"{'delta':>7} | {'Stage 2.4':>9}")
    print("-" * 60)
    for name in STRATEGIES:
        norag_mean = st.mean(r[f"norag_{name}"] for r in rows)
        routed_mean = st.mean(r[f"routed_{name}"] for r in rows)
        print(f"{name:>12} | {norag_mean:>12.3f} | {routed_mean:>8.3f} | "
              f"{routed_mean - norag_mean:>+7.3f} | {STAGE2_TRIGGER[name]:>9.3f}")

    print("\nA large negative delta supports the feedback hypothesis: the routed")
    print("trajectory degrades, the instances diverge, and fewer positions are free.")
    print("If so, every trigger rate measured on the NoRAG trajectory is optimistic")
    print("and Stage 3's expectations need adjusting.\n")

    print("Sample routed output (strategy A):")
    for r in rows[:3]:
        print(f"  k={r['k']:2} trigger={r['routed_A']:.2f}  "
              f"{r['text'][:100].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
