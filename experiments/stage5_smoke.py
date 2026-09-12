"""Smoke test for the plain-DPRAG arm: one query, four tokens, loud at each step.

`stage5_1_plain_timing.py` hung on its first generation with the GPU idle, and a
40-query run is a bad place to find out where. This does the same thing once, with
max_new_tokens=4, printing before and after every stage so the hang localises
without a debugger.

WHY THE PLAIN ARM IS THE SUSPECT AND THE ROUTED ONE IS NOT
----------------------------------------------------------
`dp_model.dp_chat` is the only code path in this project that nothing has run for
months. Every stage since has gone through `Router`, which builds its own batches
and calls the model directly. So `dp_chat` has not been exercised since
`max_retrieve` went from 10 to 40 (ADR 0008), nor since torch 2.6 / transformers
4.57. Both arms are run here, in that order, so the output says which one it is.

Read the last line printed. If it stops after "plain: generate() ..." the legacy
path is where it hangs; if both arms finish, the problem is in the timing script
rather than in either path.

    uv run python experiments/stage5_smoke.py
    uv run python experiments/stage5_smoke.py 16      # more tokens
"""

import faulthandler
import sys
import time

import torch

from dprag import sweep
from dprag.bench import Bench
from dprag.config import ExperimentConfig
from dprag.dp_model import DPGenerationConfig
from dprag.router import Router

EPSILON = 40

# A hang with the GPU idle gives no traceback on its own. This dumps the stack of
# every thread if the whole thing takes longer than it could possibly need, which
# is the one piece of evidence the failed run did not produce.
WATCHDOG_SECONDS = 600


def stamp(message: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    max_new_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    faulthandler.dump_traceback_later(WATCHDOG_SECONDS, exit=True)

    exp = ExperimentConfig(n_queries=200, gen_epsilon=float(EPSILON),
                           max_new_tokens=max_new_tokens)
    print(f"=== stage 5 smoke | model={exp.gen_model} | dtype={exp.gen_dtype} "
          f"| eps={EPSILON} | max_new_tokens={max_new_tokens} ===")
    print(f"watchdog: dumps every thread's stack and exits after "
          f"{WATCHDOG_SECONDS}s\n", flush=True)

    stamp("Bench.build (embeds the corpus; the generation model is lazy)")
    bench = Bench.build(exp)
    stamp("Bench.build done")

    stamp("DPGenerationConfig (binary-searches token_epsilon over a PLD)")
    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=float(EPSILON), delta=exp.delta,
    )
    stamp(f"DPGenerationConfig done, token_epsilon={dp_cfg.token_epsilon():.6f}")

    question = bench.queries()[0].query
    store = bench.engine.pup_vector_store
    stamp("pup_retrieve")
    store.reseed_for(question)
    documents = store.pup_retrieve(question)
    stamp(f"pup_retrieve done: {len(documents)} documents")
    if not documents:
        raise SystemExit("query 0 retrieved nothing; nothing to smoke-test")

    stamp("forcing the generation model to load (it is a cached_property, so it "
          "would otherwise load inside the first timed call)")
    _ = bench.dp_model.model
    stamp("model loaded")

    print()
    stamp(f"plain: dp_chat over {len(documents) + 1} rows -- START")
    began = time.time()
    torch.manual_seed(exp.seed)
    plain_text = bench.dp_model.dp_chat(documents, question, dp_cfg)
    plain_seconds = time.time() - began
    stamp(f"plain: dp_chat -- DONE in {plain_seconds:.1f}s")
    print(f"       {plain_text[:120]!r}")

    print()
    stamp("routed: Router(NEVER_AGREE).generate -- START")
    began = time.time()
    torch.manual_seed(exp.seed)
    result = Router(bench.dp_model, sweep.NEVER_AGREE, dp_cfg).generate(
        documents, question)
    routed_seconds = time.time() - began
    stamp(f"routed: generate -- DONE in {routed_seconds:.1f}s")
    print(f"       {result.text[:120]!r}")

    faulthandler.cancel_dump_traceback_later()

    print()
    print(f"plain  {plain_seconds:6.1f}s")
    print(f"routed {routed_seconds:6.1f}s")
    print(f"identical: {plain_text == result.text}")
    if plain_text != result.text:
        print()
        print("The two paths disagree. On four tokens that is worth knowing before")
        print("anything longer runs: the routed baseline is supposed to BE plain")
        print("DPRAG, and nothing has ever checked it against the real model.")


if __name__ == "__main__":
    main()
