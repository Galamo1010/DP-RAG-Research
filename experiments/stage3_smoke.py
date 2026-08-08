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

# The two ends of the proposal's budget grid. At eps=10 the clipping is 0.110; at
# eps=40 it is 0.295, so noise per paid token is roughly a third. Running both
# separates "the routed output degrades" from "the routed output degrades at low
# budget", which are different findings and call for different responses.
EPSILON_GRID = [10.0, 40.0]


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

    # ---- trajectory comparison, across the epsilon budget ----------------
    # The NoRAG-driven rate carries no DP noise, so it does not depend on epsilon
    # and is measured once.
    print(f"\n--- trajectory comparison ({len(with_docs)} queries with documents) ---")
    print("Same query, same documents; the driver and the budget vary.\n")

    norag_rates = {name: [] for name in STRATEGIES}
    for f in with_docs:
        torch.manual_seed(exp.seed)
        driven = run_dual_instance(
            bench.dp_model, f["docs"], f["query"], plain_cfg, STRATEGIES
        )
        for name in STRATEGIES:
            norag_rates[name].append(driven.trigger_rate(name))

    # If the feedback reading is right, a bigger budget means less noise, a
    # cleaner context, and a routed rate that climbs back toward the NoRAG one.
    # A rate that does not move would say the mechanism is something else.
    routed_rates = {e: {name: [] for name in STRATEGIES} for e in EPSILON_GRID}
    samples = {e: [] for e in EPSILON_GRID}
    for eps in EPSILON_GRID:
        cfg = DPGenerationConfig(
            temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
            alpha=exp.alpha, omega=exp.omega, epsilon=eps, delta=exp.delta,
        )
        clipping = cfg.token_epsilon() * exp.temperature / 2
        print(f"  eps_gen={eps}  (clipping={clipping:.4f})")
        for i, f in enumerate(with_docs):
            for name, strategy in STRATEGIES.items():
                torch.manual_seed(exp.seed)
                res = Router(bench.dp_model, strategy, cfg).generate(
                    f["docs"], f["query"]
                )
                routed_rates[eps][name].append(res.trigger_rate)
                if name == "A" and len(samples[eps]) < 3:
                    samples[eps].append((res.trigger_rate, res.text))
        print("    " + "  ".join(
            f"{n}: {st.mean(routed_rates[eps][n]):.3f}" for n in STRATEGIES
        ))

    header = (f"\n{'strategy':>12} | {'NoRAG-driven':>12} | "
              + " | ".join(f"{'routed eps=' + str(e):>14}" for e in EPSILON_GRID))
    print(header)
    print("-" * len(header))
    for name in STRATEGIES:
        norag_mean = st.mean(norag_rates[name])
        cells = []
        for eps in EPSILON_GRID:
            routed_mean = st.mean(routed_rates[eps][name])
            cells.append(f"{routed_mean:>7.3f} ({routed_mean - norag_mean:+.3f})")
        print(f"{name:>12} | {norag_mean:>12.3f} | " + " | ".join(cells))

    print("\nIf the gap narrows as epsilon grows, the feedback reading holds: less")
    print("noise keeps the context intact, so the two instances stay in agreement.")
    print("If the gap is flat, something other than context degradation is at work.")

    for eps in EPSILON_GRID:
        print(f"\nSample routed output (strategy A, eps_gen={eps}):")
        for trigger, text in samples[eps]:
            print(f"  trigger={trigger:.2f}  {text[:110].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
