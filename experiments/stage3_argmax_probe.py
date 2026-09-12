"""The proposal's third comparison: where did Strategy B's budget actually go.

The proposal asks for a specific number -- "量化策略B在「argmax不同但top-k高度
重疊」位置的額外覆蓋增益" -- and it could not be computed, because the trace
recorded what the NoRAG instance wanted and never what the RAG instance wanted.
With `rag_argmax` stored, two counts follow directly:

* **wasted** -- paid positions where `rag_argmax == norag_argmax`. The documents
  did not change the model's first choice, so the aggregation emitted what the
  free path would have emitted and the epsilon bought nothing. Strategy A cannot
  produce these; Strategy B can, because a low top-k overlap fires on tail
  disagreement even when the head agrees.
* **missed** -- free positions where `rag_argmax != norag_argmax`. The position
  was skipped to save budget, and the documents' influence on that token went
  with it. **This is what a negative pole lean looks like position by position**,
  and Llama's B_k20_t0.7 has one: lean -0.113 against baseline's -0.025.

Llama first, for two reasons. B_k20_t0.7 actually operates there -- 66.3% trigger
against 26.9% on gemma-4, where it barely fires and so has little opportunity to
waste or miss anything. And Llama's Phase 2 records exist to check against.

OTHER MODELS
------------
Qwen needs no probe: its Phase 3 records were generated after `rag_argmax` was
recorded and already carry it. gemma's were not, and the field cannot be
back-filled -- it is an argmax over logits that no longer exist -- so gemma gets a
probe of its own, checked against its Phase 3 records the way Llama's is checked
against Phase 2.

Sixty queries there too. The counts are per position, so the sample is tens of
thousands of decisions either way: resampling Qwen's 180 queries down to 56 moves
the waste rate by at most 2.0 points and the miss rate by 0.9, against gaps
between configurations of six points and more.

Expect gemma's B column to be the weak one -- at a 26.9% trigger rate only about
2,059 free positions exist for it to miss anything at, against 7,571 paid
positions for baseline. baseline and A are what that arm is really for.

WHAT THIS ALSO VERIFIES
-----------------------
Recording `rag_argmax` reads a tensor the strategy has already seen and draws no
randomness, so the generated text must be unchanged. `logits_to_keep=1` likewise.
Both claims were checked on three queries; this checks them on sixty, against
whichever record already holds this model's un-probed generation, and says so per
configuration. A mismatch means one of those "provably identical" changes was not,
and every number below is suspect.

Roughly two hours on the primary model, one on gemma-4 -- the gap is dtype, not
size. Sixty queries is enough: the counts are per position, so this is tens of
thousands of decisions, not sixty.

    uv run python experiments/stage3_argmax_probe.py
    uv run python experiments/stage3_argmax_probe.py <model_id>
"""

import sys

from dprag import paths, run_record, sweep, trace
from dprag.bench import Bench
from dprag.config import ExperimentConfig, gen_dtype_for
from dprag.dp_model import DPGenerationConfig
from dprag.strategies import make_strategy_b, strategy_a

EPSILON = 40
N_QUERIES = 60
EXPERIMENT = ExperimentConfig(n_queries=200)
PRIMARY_MODEL = EXPERIMENT.gen_model

CONFIGS = {
    "baseline": sweep.NEVER_AGREE,
    "A": strategy_a,
    "B_k20_t0.7": make_strategy_b(20, 0.7),
}


def filename_for(model: str) -> str:
    """Where this model's probe is written.

    The primary model keeps the unstamped name it already has on disk. Renaming it
    would orphan `results/stage3_argmax_probe_60q_eps40.json` and every number
    already quoted from it, and buy nothing.
    """
    if model == PRIMARY_MODEL:
        return f"stage3_argmax_probe_{N_QUERIES}q_eps{EPSILON}"
    short = model.split("/")[-1]
    return f"stage3_argmax_probe_{short}_{N_QUERIES}q_eps{EPSILON}"


def comparison_stem(model: str, config: str) -> str:
    """The record this model's probe has to reproduce token for token.

    Different phases for different models, because that is where each model's
    un-probed generation lives: the primary model's is Phase 2, every other
    model's is Phase 3. The question is the same either way -- did recording
    `rag_argmax` disturb generation.
    """
    if model == PRIMARY_MODEL:
        return f"stage3_2_main_{config}_eps{EPSILON}"
    short = model.split("/")[-1]
    return f"stage3_3_cross_{short}_{config}_eps{EPSILON}"


def verify_against_existing(record, model: str) -> None:
    """Byte-compare the regenerated answers with this model's existing ones."""
    where = "Phase 2" if model == PRIMARY_MODEL else "Phase 3"
    print(f"=== identical to {where}? ===")
    print("Recording rag_argmax and keeping one position of logits are both")
    print("supposed to leave generation untouched. If they did, every answer here")
    print(f"matches the {where} record token for token.")
    print()

    mine = {r["query"]: r for r in record.per_item}
    for name in CONFIGS:
        path = paths.results_dir() / f"{comparison_stem(model, name)}.json"
        if not path.exists():
            print(f"  {name:>11}: no {where} record to compare against")
            continue
        theirs = {r["query"]: r for r in run_record.load(path).per_item}

        same = diff = missing = 0
        first_bad = None
        for question, row in mine.items():
            other = theirs.get(question)
            if other is None:
                missing += 1
                continue
            if row["by_strategy"][name]["emitted"] == other["by_strategy"][name]["emitted"]:
                same += 1
            else:
                diff += 1
                first_bad = first_bad or question
        # "no answer differed" is not the same claim as "no answer was compared".
        # An earlier version printed IDENTICAL for both, which is the failure mode
        # this whole check exists to catch.
        if same + diff == 0:
            verdict = "*** NOTHING COMPARED ***"
        elif diff == 0:
            verdict = "IDENTICAL"
        else:
            verdict = f"*** {diff} DIFFER ***"
        absent = f", {missing} not in {where}" if missing else ""
        print(f"  {name:>11}: {same} identical, {diff} different{absent}   {verdict}")
        if first_bad:
            print(f"               first divergence on: {first_bad[:60]}")
    print()


def report(record) -> None:
    print("=== where the budget went ===")
    print("wasted  paid positions where rag_argmax == norag_argmax: the documents")
    print("        did not change the first choice, so the epsilon bought a token")
    print("        the free path would have emitted anyway.")
    print("missed  free positions where rag_argmax != norag_argmax: the documents")
    print("        would have changed the token, and skipping dropped that.")
    print()
    print(f"{'config':>11} | {'positions':>9} | {'paid':>6} | {'wasted':>15} | "
          f"{'free':>6} | {'missed':>15}")
    print("-" * 84)

    for name in CONFIGS:
        pos = paid = wasted = free = missed = 0
        for row in record.per_item:
            sr = row["by_strategy"][name]
            n = len(sr["emitted"])
            pos += n
            paid += len(sr["paid_positions"])
            free += n - len(sr["paid_positions"])
            wasted += len(trace.wasted_paid_positions(sr))
            missed += len(trace.missed_free_positions(sr))
        wp = f"{wasted} ({100 * wasted / paid:.1f}%)" if paid else "-"
        mp = f"{missed} ({100 * missed / free:.1f}%)" if free else "-"
        print(f"{name:>11} | {pos:>9} | {paid:>6} | {wp:>15} | {free:>6} | {mp:>15}")

    print()
    print("Strategy A's missed count is zero by construction -- its agreement test")
    print("IS rag_argmax == norag_argmax. A non-zero figure there would mean the")
    print("recorded argmax and the strategy that ran disagree, and nothing in this")
    print("table could be trusted.")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else PRIMARY_MODEL
    # Not the field default: this model's arm has a dtype, and running at any other
    # one makes the record incomparable with the rest of its own arm -- which is
    # what the equivalence check below would then fail on, for a reason that has
    # nothing to do with rag_argmax.
    exp = EXPERIMENT.with_(gen_model=model, gen_epsilon=float(EPSILON),
                           gen_dtype=gen_dtype_for(model))
    bench = Bench.build(exp)
    # Slice the full sample rather than asking for sixty: load_queries(n=60) is
    # not guaranteed to return the first sixty of load_queries(n=200), and these
    # have to be a subset of what this model already ran or there is nothing to
    # compare against.
    queries = bench.queries()[:N_QUERIES]

    print(f"=== argmax ablation | model={exp.gen_model} | eps={EPSILON} ===")
    print(f"{len(CONFIGS)} configurations x {len(queries)} queries "
          f"| max_retrieve={exp.max_retrieve} | dtype={exp.gen_dtype}")
    print(f"configurations: {list(CONFIGS)}")
    print(flush=True)

    dp_cfg = DPGenerationConfig(
        temperature=exp.temperature, max_new_tokens=exp.max_new_tokens,
        alpha=exp.alpha, omega=exp.omega, epsilon=float(EPSILON), delta=exp.delta,
    )
    path = sweep.routed_sweep(
        bench, exp, CONFIGS, dp_cfg,
        name="stage3_argmax_probe", filename=filename_for(model), queries=queries,
        metrics={
            "phase": "3 (argmax ablation)",
            "epsilon": EPSILON,
            "purpose": (
                "Counts the proposal's third comparison: paid positions where the "
                "documents changed nothing, and skipped positions where they would "
                "have. Needs rag_argmax, which earlier runs do not carry."
            ),
        },
    )

    record = run_record.load(path)
    print()
    verify_against_existing(record, model)
    report(record)


if __name__ == "__main__":
    main()
