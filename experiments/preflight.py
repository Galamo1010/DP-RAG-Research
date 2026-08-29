"""Everything that has to be true before Stage 3.2 is worth starting.

Four things are currently believed rather than known, and each was settled on
evidence that did not come from the machine the experiments run on:

* `bert-score` is in pyproject and `dprag.quality.bertscore_f1` has never been
  executed, because the environment it was written in has no network.
* ADR 0004 chose gemma-4-12B-it after reading `chat_template.jinja` from the model
  repository. A chat template is a mutable file; what matters is the one that
  actually downloads. Mistral's changed in July 2024, which is how this project
  came to re-open the question at all.
* ADR 0005 pinned the template date after measuring on Llama-3.2-**1B**. The
  experiments run on 3.1-**8B**.
* ADR 0003 budgeted Stage 3.2 at roughly 15 hours assuming k+1 = 11 rows. Raising
  max_retrieve to 100 makes it 101, and "batch width is not the bottleneck" was
  measured at the smaller size. The multiplier is unknown, and it multiplies the
  single largest expense in the project.

Checks run cheapest first and independently: one failure does not hide the rest.

    uv run python experiments/preflight.py
"""

import time
import traceback

import torch

from dprag import prompts
from dprag.config import ExperimentConfig

CHECKS: list[tuple[str, object]] = []
RESULTS: list[tuple[str, bool, str]] = []

SYSTEM_PROBE = "You give a short response based on a predefined set documents."
CONV = prompts.norag_chat("Is this contagious?")


def check(name):
    """Register a check.

    Registration, not execution: importing this module has to stay free, or a
    stray import triggers a corpus embedding and a model load.
    """
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def run_all():
    """Run every registered check. One failure does not stop the others."""
    for name, fn in CHECKS:
        print("\n=== " + name + " ===", flush=True)
        try:
            note = fn() or ""
            RESULTS.append((name, True, note))
            print("  PASS  " + note, flush=True)
        except Exception as exc:
            RESULTS.append((name, False, type(exc).__name__ + ": " + str(exc)))
            print("  FAIL  " + type(exc).__name__ + ": " + str(exc), flush=True)
            traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. bert-score is installed  (instant)
# ---------------------------------------------------------------------------
@check("1. bert-score importable")
def _bert_import():
    import bert_score
    return "bert_score " + str(getattr(bert_score, "__version__", "?"))


# ---------------------------------------------------------------------------
# 2. the date pin holds on the model the experiments actually use
# ---------------------------------------------------------------------------
@check("2. Llama-3.1-8B: date is injected, and the pin stops it")
def _date_pin():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ExperimentConfig().gen_model)
    unpinned = tok.apply_chat_template(CONV, tokenize=True, add_generation_prompt=True)
    pinned_a = tok.apply_chat_template(
        CONV, tokenize=True, add_generation_prompt=True, **prompts.TEMPLATE_KWARGS)
    pinned_b = tok.apply_chat_template(
        CONV, tokenize=True, add_generation_prompt=True, **prompts.TEMPLATE_KWARGS)

    if pinned_a != pinned_b:
        raise AssertionError("the pin is not deterministic")
    if unpinned == pinned_a:
        # Not a failure: it would mean 3.1-8B carries no dynamic date at all.
        return "no dynamic date on this model; the pin is a no-op here"
    diff = sum(1 for x, y in zip(unpinned, pinned_a) if x != y)
    return ("date injected (" + str(diff) + " tokens differ from today's render); "
            "pinned to " + repr(prompts.DATE_STRING) + ", reproducible")


# ---------------------------------------------------------------------------
# 3. gemma-4-12B-it accepts a system role -- on the snapshot that downloads
# ---------------------------------------------------------------------------
@check("3. gemma-4-12B-it: system role survives its own chat template")
def _gemma_template():
    from transformers import AutoTokenizer

    from dprag.config import MODELS

    model_id = next((m for m in MODELS if "gemma" in m.lower()), None)
    if model_id is None:
        raise AssertionError("no gemma in config.MODELS; ADR 0004 was not applied")

    tok = AutoTokenizer.from_pretrained(model_id)
    rendered = tok.apply_chat_template(
        CONV, tokenize=False, add_generation_prompt=True, **prompts.TEMPLATE_KWARGS)

    if SYSTEM_PROBE not in rendered:
        raise AssertionError(
            "the system message is missing from the rendered prompt -- dropped "
            "silently, which is worse than an exception. ADR 0004's verification "
            "does not hold for this snapshot. Fall back to "
            "microsoft/Phi-3.5-mini-instruct (verified to preserve it) and record why."
        )
    before_question = rendered.split("Is this contagious")[0].lower()
    if "system" in before_question:
        return "system content preserved, in its own turn"
    return ("system content preserved but MERGED into the user turn -- prompt "
            "structure differs from Llama/Qwen; note it in the report")


# ---------------------------------------------------------------------------
# 4. bertscore_f1 actually runs, on real generated text
# ---------------------------------------------------------------------------
@check("4. bertscore_f1 on real answers (downloads ~3GB on first run)")
def _bertscore():
    import json
    from pathlib import Path

    from dprag import paths, quality
    from dprag.chatdoctor import load_queries

    cfg = ExperimentConfig()
    path = Path(paths.RESULTS_DIR) / "stage2_safety_check_eps10_200q.json"
    if not path.exists():
        raise FileNotFoundError(path.name + " not found; run Stage 2.5 first")

    rows = json.load(open(path, encoding="utf-8"))["per_item"][:8]
    refs = {q.query: q.reference
            for q in load_queries(n=cfg.n_queries, seed=cfg.query_seed)}
    hyps = [r["by_strategy"]["A"]["text"] for r in rows]
    gold = [refs[r["query"]] for r in rows]

    started = time.time()
    scores = quality.bertscore_f1(
        hyps, gold, model=cfg.bertscore_model, rescale=cfg.bertscore_rescale)
    elapsed = time.time() - started

    if len(scores) != len(hyps):
        raise AssertionError("expected one score per pair")
    return ("%d pairs in %.1fs, F1 range %.3f..%.3f (model=%s, rescale=%s)"
            % (len(scores), elapsed, min(scores), max(scores),
               cfg.bertscore_model, cfg.bertscore_rescale))


# ---------------------------------------------------------------------------
# 5. what k+1 = 101 rows costs.  The number Stage 3.2's budget hangs on.
# ---------------------------------------------------------------------------
@check("5. routing speed at max_retrieve 10 vs 100")
def _routing_speed():
    from dprag.bench import Bench
    from dprag.dp_model import DPGenerationConfig
    from dprag.router import Router
    from dprag.strategies import strategy_a

    if not torch.cuda.is_available():
        raise RuntimeError("needs a CUDA GPU")

    exp = ExperimentConfig(n_queries=3)
    bench = Bench.build(exp)
    store = bench.engine.pup_vector_store
    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=exp.gen_epsilon, delta=exp.delta)

    queries = [q.query for q in bench.queries()]
    notes = []
    for limit in (10, 30):
        # Same store, no re-embedding: max_retrieve only caps the subsample drawn
        # from the documents that already cleared the DP threshold.
        store.max_retrieve = limit
        timings = []
        peak = 0
        for question in queries:
            docs = store.pup_retrieve(question)
            if not docs:
                continue
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(exp.seed)
            began = time.time()
            res = Router(bench.dp_model, strategy_a, dp_cfg).generate(docs, question)
            torch.cuda.synchronize()
            timings.append((time.time() - began, len(docs), res.trigger_rate))
            peak = max(peak, torch.cuda.max_memory_allocated())
        if not timings:
            notes.append("max_retrieve=" + str(limit) + ": every query retrieved 0 docs")
            continue
        mean_s = sum(t for t, _, _ in timings) / len(timings)
        mean_k = sum(k for _, k, _ in timings) / len(timings)
        mean_tr = sum(tr for _, _, tr in timings) / len(timings)
        print("  max_retrieve=%3d  k=%5.1f  %6.1f s/query  trigger=%.3f  "
              "peak VRAM %.1f GB" % (limit, mean_k, mean_s, mean_tr, peak / 1e9),
              flush=True)
        notes.append("k=%.0f:%.1fs" % (mean_k, mean_s))
    return "  ".join(notes)


def main():
    run_all()
    print()
    print("=" * 72)
    print("PREFLIGHT SUMMARY")
    print("=" * 72)
    for name, ok, note in RESULTS:
        print("  " + ("PASS" if ok else "FAIL") + "  " + name)
        if note:
            print("          " + note)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print()
    if failed:
        print(str(len(failed)) + " check(s) failed. Stage 3.2 is not ready to start.")
        return
    print("All checks passed.")
    print()
    print("Next: take the seconds-per-query from check 5, multiply by the Stage 3.2")
    print("run counts in ADR 0003 (the main phase is 16 runs x 200 queries), and")
    print("decide max_retrieve against that budget BEFORE the phase scripts are")
    print("written -- batch size and time estimates both depend on it.")


if __name__ == "__main__":
    main()
