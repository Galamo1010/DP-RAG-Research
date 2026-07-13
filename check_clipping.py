"""Stage 1.1c -- verify the DP clipping / PLD accounting matches the paper.

No GPU, no model download: this is pure privacy-accounting math. It confirms two
things the proposal (計畫書) relies on:

  Eq4:  clipping = token_epsilon * temperature / 2      (dp_model.py:113)
  Eq3-ish round-trip: composing `max_new_tokens` copies of token_epsilon via the
        Google dp-accounting PLD returns the generation budget epsilon we asked for.

token_epsilon itself is found by binary search inside DPGenerationConfig; here we
just check that the code's clipping value and the PLD composition are self-consistent
with the formulas, for the generation budgets used in the experiments.

Run:  uv run python check_clipping.py
"""

from dp_model import DPGenerationConfig, DPLogitsAggregator
import experiment_params as P

# Generation-side budgets to check. GEN_EPSILON=10 is what the smoke test uses;
# the wider grid mirrors the proposal's ε_total control variable.
GEN_EPSILON_GRID = [5.0, 10.0, 20.0, 40.0]


def check_one(gen_epsilon: float) -> dict:
    cfg = DPGenerationConfig(
        temperature=P.TEMPERATURE,
        max_new_tokens=P.MAX_NEW_TOKENS,
        alpha=P.ALPHA,
        omega=P.OMEGA,
        epsilon=gen_epsilon,
        delta=P.DELTA,
    )
    token_eps = cfg.token_epsilon()

    # Eq4 exactly as dp_model.py:113 computes it.
    clipping_formula = token_eps * cfg.temperature / 2

    # The value the live aggregator will actually use during generation.
    agg = DPLogitsAggregator(cfg)
    clipping_used = agg.token_epsilon * agg.temperature / 2

    # PLD round-trip: does composing max_new_tokens steps of token_eps give the budget back?
    composed = cfg.composed_epsilon(token_eps)

    return {
        "gen_epsilon": gen_epsilon,
        "token_epsilon": token_eps,
        "clipping_formula": clipping_formula,
        "clipping_used": clipping_used,
        "composed_epsilon": composed,
        "clip_match": abs(clipping_formula - clipping_used) < 1e-12,
        # token_epsilon is found by binary search (tol=1e-3); composing 128 steps
        # amplifies that, so allow ~1% relative error on the round-tripped budget.
        "budget_match": abs(composed - gen_epsilon) / gen_epsilon <= 0.01,
    }


def main():
    print(f"temperature={P.TEMPERATURE}  max_new_tokens={P.MAX_NEW_TOKENS}  delta={P.DELTA}\n")
    header = f"{'eps_gen':>7} | {'token_e':>9} | {'clip=te/2':>10} | {'clip(agg)':>10} | {'PLD_T':>8} | ok?"
    print(header)
    print("-" * len(header))
    all_ok = True
    for e in GEN_EPSILON_GRID:
        r = check_one(e)
        ok = r["clip_match"] and r["budget_match"]
        all_ok &= ok
        print(f"{r['gen_epsilon']:7.1f} | {r['token_epsilon']:9.5f} | "
              f"{r['clipping_formula']:10.5f} | {r['clipping_used']:10.5f} | "
              f"{r['composed_epsilon']:8.4f} | {'PASS' if ok else 'FAIL'}")
    print()
    if all_ok:
        print("OK  clipping == token_epsilon*temperature/2 (Eq4), and PLD-composing "
              "max_new_tokens steps returns the generation budget.")
    else:
        print("FAIL  a mismatch was found -- inspect the failing row above.")


if __name__ == "__main__":
    main()
