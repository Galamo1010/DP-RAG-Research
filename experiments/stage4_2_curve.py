"""Stage 4.2 -- where in an answer the privacy budget actually goes.

The proposal asks for one figure:

> 逐步驟記錄每個token位置的路徑決策（使用ε/不使用ε）與累積ε_usage，生成結束後
> 繪製token位置vs. ε_usage曲線。從ChatDoctor選取代表性查詢，疊加於同一圖中：
> 文件依賴度低的查詢曲線接近水平（多數位置不使用ε），含藥名的醫療查詢在藥名位置
> 出現集中跳升；同時疊加原版DPRAG的固定斜率直線作為對照，ε_budget上限以水平虛線
> 標示，曲線末端與直線的差距即為ε_savings。

Every input is already recorded. `paid_positions` says which positions took the
paid path, `clinical` says which positions carry a drug, dose or condition, and
epsilon after n paid steps is a deterministic PLD composition. No GPU.

WHICH QUERIES, AND WHY NOT PICKED BY EYE
----------------------------------------
Two, as the proposal names two behaviours.

The **low-dependency** query is chosen by Stage 1.2's per-query
`consistency_greedy` -- how often NoRAG's argmax already matches DPRAG's. That is
an independent measurement of document dependence, taken before any routing
happened, so choosing on it does not select for the outcome being plotted.
Choosing instead by the highest trigger rate would pick the query whose curve is
flattest *because* it is flattest, which is circular.

The **clinical** query is chosen by counting span-opening clinical positions in
strategy A's own answer. There is no independent ranking available for this one:
whether an answer names drugs is a property of the answer, and the answer is what
the router produced. Stated rather than hidden.

Both are restricted to queries that retrieved documents and appear in every
configuration being drawn, so the panels compare like with like.

THE PROPOSAL EXPECTS A STRAIGHT LINE AND WILL NOT GET ONE
---------------------------------------------------------
"疊加原版DPRAG的固定斜率直線" assumes plain DPRAG spends epsilon linearly, one
token_epsilon per position. Under PLD accounting at fixed delta it does not: the
composed distribution of pure-DP mechanisms sits on a coarse lattice, so the
increment from one more paid position swings between 0.012 and 0.589 at
eps_total=40 while staying monotone. The baseline curve is therefore a rising
sawtooth that is concave overall -- 128 paid positions cost 40.0, but the first
10 already cost 5.8.

That is a property of the accounting, not of the pre-filter, and it is worth a
sentence in the report: composition is sublinear, so the *last* positions of an
answer are much cheaper than the first, and skipping early positions is worth
more than skipping late ones.

    uv run python experiments/stage4_2_curve.py
"""

import glob
import statistics as st
from functools import lru_cache

from dprag import accounting, paths, run_record
from dprag.dp_model import DPGenerationConfig
from dprag.medical_flags import WORD_KIND

# The headline budget. eps=40 rather than 10 because the sawtooth and the
# skipped-position plateaus are both easier to see when each step is larger.
RECORDS = [
    "stage3_2_main_baseline_eps40",
    "stage3_2_main_B_k20_t0.7_eps40",
    "stage3_2_main_A_eps40",
]
STAGE1_RECORD = "stage1_consistency_10000x200"
FIG_DIR = "figures"


@lru_cache(maxsize=None)
def token_epsilon_for(epsilon: float, delta: float, max_new_tokens: int,
                      temperature: float, alpha: float, omega: float) -> float:
    cfg = DPGenerationConfig(
        epsilon=epsilon, delta=delta, max_new_tokens=max_new_tokens,
        temperature=temperature, alpha=alpha, omega=omega,
    )
    return cfg.token_epsilon()


def load_records() -> list[tuple[str, object]]:
    """(configuration name, RunRecord) for each curve to draw, in RECORDS order."""
    out = []
    for stem in RECORDS:
        matches = glob.glob(str(paths.results_dir() / f"{stem}.json"))
        if not matches:
            raise SystemExit(
                f"missing {stem}.json -- Stage 4.2 draws the strategies against the "
                "baseline, so a partial set would silently plot fewer curves")
        record = run_record.load(matches[0])
        names = record.metric("strategies", [])
        if len(names) != 1:
            raise SystemExit(f"{stem} holds {names}; expected exactly one strategy")
        out.append((names[0], record))
    return out


def clinical_opens(strategy_record: dict, word_like_only: bool) -> list[int]:
    """Positions where a clinical span begins.

    Span openings rather than every clinical position: a drug name spans several
    tokens, and the continuation tokens are forced by spelling once the first one
    is out. What the figure is about is where the model *chose* to say something
    clinical, which is the opening.
    """
    out = []
    for i, mark in enumerate(strategy_record.get("clinical") or []):
        if not mark:
            continue
        kind, is_first = mark
        if not is_first:
            continue
        if word_like_only and kind != WORD_KIND:
            continue
        out.append(i)
    return out


def pick_queries(records) -> dict[str, str]:
    """{'low dependency': query, 'clinical': query}. See the module docstring."""
    shared = None
    for _, record in records:
        here = {r["query"] for r in record.per_item if r.get("n_documents")}
        shared = here if shared is None else (shared & here)
    if not shared:
        raise SystemExit("no query with documents is present in every record")

    stage1 = glob.glob(str(paths.results_dir() / f"{STAGE1_RECORD}.json"))
    if not stage1:
        raise SystemExit(
            f"missing {STAGE1_RECORD}.json -- the low-dependency query is chosen on "
            "Stage 1.2's consistency, and picking it from the routed run instead "
            "would select on the outcome being plotted")
    consistency = {r["query"]: r["consistency_greedy"]
                   for r in run_record.load(stage1[0]).per_item
                   if r["query"] in shared}
    if not consistency:
        raise SystemExit("Stage 1.2 and the routed runs share no query")
    low_dependency = max(consistency, key=consistency.get)

    # The clinical pick reads strategy A's answers -- the last record listed.
    name, record = records[-1]
    counts = {r["query"]: len(clinical_opens(r["by_strategy"][name], word_like_only=True))
              for r in record.per_item
              if r["query"] in shared and name in r["by_strategy"]}
    clinical = max(counts, key=counts.get)

    print(f"low-dependency query : Stage 1.2 consistency_greedy = "
          f"{consistency[low_dependency]:.4f} (max over {len(consistency)} shared queries)")
    print(f"clinical query       : {counts[clinical]} word-like clinical spans in "
          f"the {name} answer (max over {len(counts)} shared queries)")
    print(f"                       median across those queries = "
          f"{st.median(counts.values()):.1f}")
    return {"low dependency": low_dependency, "clinical": clinical}


def curves_for(records, query: str):
    """[(config name, cumulative epsilon per position, clinical opens, budget)]."""
    out = []
    for name, record in records:
        row = next((r for r in record.per_item if r["query"] == query), None)
        if row is None:
            continue
        sr = row["by_strategy"][name]
        delta = float(record.param("delta", 1e-3))
        token_eps = token_epsilon_for(
            float(record.param("gen_epsilon", 0.0)), delta,
            int(record.param("max_new_tokens", 128)),
            float(record.param("temperature", 1.0)),
            float(record.param("alpha", 1.0)),
            float(record.param("omega", 0.01)),
        )
        n = len(sr["emitted"])
        cumulative = accounting.cumulative_epsilon(
            sr["paid_positions"], n, token_eps, delta, eps_retrieval=0.0)
        budget = accounting.composed_epsilon(
            token_eps, int(record.param("max_new_tokens", 128)), delta)
        out.append((name, cumulative, clinical_opens(sr, word_like_only=True),
                    set(sr["paid_positions"]), budget))
    return out


def main():
    records = load_records()
    picks = pick_queries(records)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(picks), figsize=(7.2 * len(picks), 4.8),
                             squeeze=False)
    budget = None
    for ax, (label, query) in zip(axes[0], picks.items()):
        series = curves_for(records, query)
        for name, cumulative, opens, paid, budget in series:
            line, = ax.plot(range(len(cumulative)), cumulative, lw=1.6, label=name)
            # A clinical span that opened on the paid path is a filled marker; one
            # that opened on the free path is hollow. The second kind is the
            # safety question Stage 2.5 asks, visible here per position.
            for i in opens:
                if i >= len(cumulative):
                    continue
                ax.plot([i], [cumulative[i]], marker="o", ms=5.5,
                        color=line.get_color(),
                        mfc=line.get_color() if i in paid else "white",
                        mec=line.get_color(), zorder=4)
        if budget is not None:
            ax.axhline(budget, ls="--", lw=1, color="grey")
            # Right-aligned: the legend sits top-left, and a budget label placed
            # there was drawn over it.
            ax.text(0.995, budget, f"eps_budget = {budget:.2f} ",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8, color="grey")
        ax.set_xlabel("token position")
        ax.set_ylabel("cumulative epsilon (generation layer)")
        ax.set_title(f"{label}\n{query[:78]}...", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Stage 4.2  cumulative epsilon by token position  |  markers = "
                 "clinical spans (filled = paid, hollow = skipped)", fontsize=11)
    fig.tight_layout()
    out_dir = paths.results_dir() / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "stage4_2_epsilon_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print()
    print("Curves end at different x: each configuration generates its own "
          "answer and stops at its own EOS, so 'distance from the budget "
          "line at the end of the curve' compares answers of different "
          "lengths. Stage 4.1 carries the mean length beside its epsilon "
          "figures for the same reason.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
