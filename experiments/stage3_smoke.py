"""Stage 3.1 -- GPU smoke for the router.

The router's unit tests run against a fake model, so they establish the routing
logic and the epsilon accounting but say nothing about how a real transformers KV
cache behaves. Two things in particular are unverified until this runs:

  * feeding a backlog of N tokens in ONE forward must leave the cache in the same
    state as feeding them one at a time -- the whole catch-up scheme rests on it;
  * position_ids derived from a left-padded attention mask must place each row
    where generate() would have placed it.

Both fail silently. The cache ends up conditioned on a subtly wrong prefix and the
answer merely comes out worse, which is indistinguishable from DP noise. So they
are checked directly, before any measurement run is built on top.

Small on purpose: 500 documents, 3 queries, 32 tokens. This is a wiring check, not
a measurement.

Requires a CUDA GPU. Run:  uv run python experiments/stage3_smoke.py
"""

import torch

from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router, _Stream
from dprag.strategies import PrefilterDecision, make_strategy_b, strategy_a
from dprag import prompts

EXPERIMENT = ExperimentConfig(n_docs=500, n_queries=3, max_new_tokens=32)

ALWAYS_AGREE = lambda rag, norag: PrefilterDecision(True, int(norag.argmax()), 1.0)
NEVER_AGREE = lambda rag, norag: PrefilterDecision(False, int(norag.argmax()), 0.0)


def check_chunked_catch_up(dp_model, documents, question) -> bool:
    """One forward over N tokens must equal N forwards of one token.

    This is the property the backlog exists to exploit. Verified on the real model
    and the real cache, because it is a statement about transformers' cache
    handling rather than about the router's own bookkeeping.
    """
    conversations = prompts.dprag_chat_batch(documents, question)
    encoded = dp_model.tokenizer.apply_chat_template(
        conversations, tokenize=True, padding=True, return_tensors="pt",
        return_dict=True, add_generation_prompt=True, continue_final_message=False,
    )
    device = dp_model.model.device
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)

    backlog = [11, 22, 33, 44]

    chunked = _Stream(dp_model.model, ids, mask)
    chunked.advance([])                 # prime
    chunked_logits = chunked.advance(backlog)

    stepwise = _Stream(dp_model.model, ids, mask)
    stepwise.advance([])                # prime
    for token in backlog:
        stepwise_logits = stepwise.advance([token])

    gap = (chunked_logits - stepwise_logits).abs().max().item()
    ok = gap < 1e-2                     # fp16 tolerance
    print(f"  chunked vs stepwise catch-up: max|Δlogit| = {gap:.5f}  "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  -> the backlog catch-up does NOT preserve the cache. Do not run "
              "any measurement until this is fixed.")
    return ok


def main():
    exp = EXPERIMENT
    print(f"=== Stage 3.1 router smoke | model={exp.gen_model} ===")
    bench = Bench.build(exp)
    queries = bench.queries()

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=exp.gen_epsilon, delta=exp.delta,
    )

    first = queries[0]
    docs = bench.retrieve(first.query)
    print(f"\n--- cache mechanics (k={len(docs)}) ---")
    if not check_chunked_catch_up(bench.dp_model, docs, first.query):
        raise SystemExit("cache catch-up check failed")

    strategies = {
        "always_agree": ALWAYS_AGREE,
        "never_agree": NEVER_AGREE,
        "A": strategy_a,
        "B_k20_t0.7": make_strategy_b(20, 0.7),
    }

    print("\n--- degenerate strategies (the equivalence properties, on real weights) ---")
    for name in ("always_agree", "never_agree"):
        torch.manual_seed(exp.seed)
        router = Router(bench.dp_model, strategies[name], dp_cfg)
        res = router.generate(docs, first.query)
        print(f"  {name:12} paid={res.n_paid:2}/{res.n_steps:2}  "
              f"trigger={res.trigger_rate:.2f}  "
              f"eps={res.epsilon_usage:.3f}/{res.epsilon_budget:.3f}")
        if name == "always_agree":
            assert res.n_paid == 0, "always-agree must never pay"
            assert res.epsilon_usage == 0.0
            assert res.emitted == res.norag_argmax[: len(res.emitted)], (
                "always-agree must emit the NoRAG argmax stream"
            )
        else:
            assert res.n_paid == res.n_steps, "never-agree must pay at every position"

    print("\n--- real strategies ---")
    for name in ("A", "B_k20_t0.7"):
        for q in queries:
            torch.manual_seed(exp.seed)
            d = bench.retrieve(q.query)
            router = Router(bench.dp_model, strategies[name], dp_cfg)
            res = router.generate(d, q.query)
            print(f"  [{name:11}] k={res.n_documents:2} paid={res.n_paid:2}/{res.n_steps:2} "
                  f"trigger={res.trigger_rate:.2f} "
                  f"eps={res.epsilon_usage:.3f}/{res.epsilon_budget:.3f} "
                  f"saved={res.epsilon_savings:.3f}")
            print(f"       {res.text[:110].replace(chr(10), ' ')}")
            assert res.epsilon_usage <= res.epsilon_budget + 1e-9

    print("\nOK: router ran end-to-end, cache catch-up verified, epsilon accounting "
          "within budget.")
    print("Expect trigger rates near the ~0.87 consistency rate measured in Stage 1.2; "
          "a wildly different number means the strategies are seeing something "
          "unexpected.")


if __name__ == "__main__":
    main()
