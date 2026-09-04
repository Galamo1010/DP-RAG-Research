"""Stage 4.1 -- what the budget was actually spent on, per configuration.

The proposal's 4.1 asks for two-layer accounting and one scatter plot:

> 第一層記錄每次檢索消耗的ε_retrieval；第二層記錄生成時使用ε路徑的累積消耗；
> 兩層合併得到ε_usage，與ε_budget的差即為ε_savings，對各配置分別計算後繪製
> trigger_rate vs ε_savings散佈圖

Everything needed is already on disk. No GPU, no regeneration: `paid_positions`
is recorded per query, and epsilon is a deterministic function of how many there
were, so the whole of 4.1 is arithmetic over the Stage 3 records.

TWO SCATTER PLOTS, NOT ONE
--------------------------
The proposal says "對各配置分別計算後繪製", which is one point per configuration
-- ten points for Llama, three for gemma. That plot answers "which configuration
saves how much", and it is the one Stage 5's Pareto analysis consumes.

It also hides the spread completely. A configuration whose queries all save 70%
and one whose queries split between 20% and 95% land on the same dot. So the
per-query scatter is drawn as well, and the two are read together: the summary
plot for the claim, the per-query plot for whether the claim is a description of
the queries or an average over two different behaviours.

THE RETRIEVAL LAYER, AND WHY ITS COLUMN IS UNSTABLE
---------------------------------------------------
No routed run has ever counted `eps_retrieval`. `RoutedResult.epsilon_usage` is
the generation layer alone, so every epsilon figure this project has reported so
far is short of the end-to-end quantity 4.1 asks for.

`dprag.accounting.two_layer_epsilon` composes the missing layer in. It is
reported here as a separate column rather than replacing the generation figure,
because composing it turns out to be badly behaved: the composed PLD of pure-DP
mechanisms is supported on a coarse lattice, so `get_epsilon_for_delta` returns a
lattice point, and adding a 0.2-epsilon mechanism moves that point sometimes by
0.16 and sometimes by exactly zero. Measured at eps_total=40:

    paid=15  generation 7.6678  two-layer 7.8492  (+0.1814)
    paid=19  generation 9.6232  two-layer 9.6232  (+0.0000)
    paid=21  generation 10.0124 two-layer 10.1745 (+0.1621)

Those zeroes are not a bug in the composition; they are what tight PLD accounting
does at this lattice spacing. But a table where the retrieval layer costs nothing
at one paid count and 0.16 at the next is not something to hand a reader without
saying so, which is why both columns are printed and neither is called *the*
answer here.

    uv run python experiments/stage4_1_budget.py
"""

import glob
import statistics as st
from functools import lru_cache

from dprag import accounting, paths, run_record
from dprag.dp_model import DPGenerationConfig

RECORD_GLOB = "stage3_*"
FIG_DIR = "figures"

# Drawn in strictness order so a legend reads top to bottom the way the
# configurations are discussed. Anything not listed falls to the end.
CONFIG_ORDER = ["baseline", "B_k20_t0.9", "B_k20_t0.7", "A", "B_k50_t0.5"]


@lru_cache(maxsize=None)
def token_epsilon_for(epsilon: float, delta: float, max_new_tokens: int,
                      temperature: float, alpha: float, omega: float) -> float:
    """The per-step epsilon the run used, re-derived from its recorded parameters.

    Not stored in the records, but it is a pure function of parameters that are,
    and `DPGenerationConfig` owns the binary search that defines it. Re-deriving
    it here rather than reimplementing keeps one definition of the quantity.
    Cached because constructing that config composes 128 PLDs.
    """
    cfg = DPGenerationConfig(
        epsilon=epsilon, delta=delta, max_new_tokens=max_new_tokens,
        temperature=temperature, alpha=alpha, omega=omega,
    )
    return cfg.token_epsilon()


def config_key(key: tuple[str, str]) -> tuple[int, str, str]:
    name, source = key
    rank = CONFIG_ORDER.index(name) if name in CONFIG_ORDER else len(CONFIG_ORDER)
    return (rank, name, source)


def is_screening(series: dict) -> bool:
    """Whether this row came from the 19-query screen rather than a full run.

    Screening rows measure the same quantities at a fifth of the query count, so
    they belong on the plot but must not be read as if they carried the same
    weight. Asked of the record name rather than of n, because a main run that
    was interrupted would also be short and is still a main run.
    """
    return series["source"].startswith("stage3_1_")


def collect() -> dict[tuple[str, float], dict[tuple[str, str], dict]]:
    """(model, eps_total) -> (configuration, source record) -> per-query series.

    Grouped the way `stage3_score.py` groups its Pareto tables, and for the same
    reason: a configuration at eps=40 saves more than one at eps=10 whatever the
    strategy does, and a different model is a different scale entirely, so
    pooling either axis produces a comparison of the conditions rather than of
    the pre-filter.

    The source record is part of the key, and this is not cosmetic. Strategy A
    appears both in the 19-query screening run and in the 180-query main run at
    eps=10; merging them silently reported n=199 for a quantity measured twice at
    very different precision, on different query samples. They are kept apart and
    the screening rows carry their own n, exactly as `stage3_score.py` does.
    """
    groups: dict[tuple[str, float], dict[tuple[str, str], dict]] = {}
    files = [p for p in sorted(glob.glob(str(paths.results_dir() / f"{RECORD_GLOB}.json")))
             if "poles" not in p]
    if not files:
        raise SystemExit("no stage3_* records found; nothing to account for")

    for path in files:
        record = run_record.load(path)
        model = record.param("gen_model", "unknown")
        eps_total = float(record.param("gen_epsilon", 0.0))
        delta = float(record.param("delta", 1e-3))
        eps_retrieval = float(record.param("eps_retrieval", 0.0))
        token_eps = token_epsilon_for(
            eps_total, delta,
            int(record.param("max_new_tokens", 128)),
            float(record.param("temperature", 1.0)),
            float(record.param("alpha", 1.0)),
            float(record.param("omega", 0.01)),
        )

        for name in record.metric("strategies", []):
            series = groups.setdefault((model, eps_total), {}).setdefault(
                (name, record.path.stem),
                {"trigger": [], "usage": [], "savings": [], "two_layer": [],
                 "length": [], "n_queries": 0, "config": name,
                 "source": record.path.stem,
                 "token_epsilon": token_eps, "eps_retrieval": eps_retrieval}
            )
            for row in record.per_item:
                sr = row["by_strategy"].get(name)
                if not sr:
                    continue
                paid = len(sr.get("paid_positions", []))
                series["length"].append(len(sr.get("emitted", [])))
                series["trigger"].append(sr["trigger_rate"])
                series["usage"].append(sr["epsilon_usage"])
                series["savings"].append(sr["epsilon_savings"])
                series["two_layer"].append(
                    accounting.two_layer_epsilon(token_eps, paid, delta, eps_retrieval))
                series["n_queries"] += 1
    return groups


def table(groups) -> None:
    print(f"{'model':>26} | {'eps':>4} | {'config':>21} | {'n':>4} | {'trigger':>8} | "
          f"{'eps_gen':>8} | {'saved':>8} | {'saved %':>8} | {'len':>5} | {'eps_2layer':>10}")
    print("-" * 133)
    for (model, eps_total), by_config in sorted(groups.items()):
        for row_key in sorted(by_config, key=config_key):
            s = by_config[row_key]
            name = s["config"] + (" (screen)" if is_screening(s) else "")
            trig = st.mean(s["trigger"])
            usage = st.mean(s["usage"])
            saved = st.mean(s["savings"])
            budget = usage + saved
            two = st.mean(s["two_layer"])
            print(f"{model.split('/')[-1][:26]:>26} | {eps_total:>4.0f} | {name:>21} | "
                  f"{s['n_queries']:>4} | {trig:>7.1%} | {usage:>8.3f} | {saved:>8.3f} | "
                  f"{saved / budget:>7.1%} | {st.mean(s['length']):>5.1f} | "
                  f"{two:>10.3f}")
    print()
    print("eps_gen     mean end-of-answer epsilon over the generation layer only --")
    print("            what the router records and every earlier number reported.")
    print("saved       eps_budget - eps_gen, the proposal's Eq5, generation layer.")
    print("len         mean emitted positions. READ THIS COLUMN BEFORE `saved`:")
    print("            eps_budget is composed over max_new_tokens, so an answer")
    print("            that stops early leaves budget unspent without the")
    print("            pre-filter doing anything. For the baseline, which pays at")
    print("            every position it generates, correlation(length, saved) is")
    print("            -1.000 exactly -- its whole `saved` figure is early EOS. A")
    print("            strategy's own contribution is the part above the baseline")
    print("            row in the same (model, eps) block.")
    print("eps_2layer  DP retrieval composed in (proposal Eq3). Read the module")
    print("            docstring before quoting it: the composition is exact but")
    print("            lands on a coarse lattice, so the retrieval layer's")
    print("            contribution is not resolvable query by query.")


def figures(groups) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display on a pod, and none needed to write a file
    import matplotlib.pyplot as plt

    out_dir = paths.results_dir() / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(groups)

    # --- one point per configuration: the plot the proposal asks for ---------
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        model, eps_total = key
        for row_key in sorted(groups[key], key=config_key):
            s = groups[key][row_key]
            x, y = st.mean(s["trigger"]), st.mean(s["savings"])
            if is_screening(s):
                # Same quantity, a fifth of the queries. Drawn hollow and left
                # unlabelled so it reads as context for the full runs, not as a
                # result standing beside them.
                ax.scatter([x], [y], s=45, marker="x", color="grey", zorder=2,
                           linewidths=1.1)
                continue
            ax.scatter([x], [y], s=90, zorder=3)
            ax.annotate(f"{s['config']}\n{y:.2f} ({y / (y + st.mean(s['usage'])):.0%})",
                        (x, y), textcoords="offset points",
                        xytext=(-9, -4) if x > 0.75 else (9, -4),
                        ha="right" if x > 0.75 else "left", fontsize=8)
        ax.axhline(eps_total, ls="--", lw=1, color="grey")
        ax.text(0.01, eps_total, f" eps_budget = {eps_total:g}", va="bottom",
                fontsize=8, color="grey")
        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(0, eps_total * 1.12)
        ax.set_xlabel("trigger rate (share of positions routed around DP)")
        ax.set_ylabel("epsilon saved (generation layer)")
        ax.set_title(f"{model.split('/')[-1]}  |  eps_total = {eps_total:g}", fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("Stage 4.1  trigger rate vs epsilon saved -- one point per "
                 "configuration (grey x = 19-query screen)", fontsize=11)
    fig.tight_layout()
    per_config = out_dir / "stage4_1_savings_by_config.png"
    fig.savefig(per_config, dpi=150)
    plt.close(fig)

    # --- one point per query: what the summary plot averages over -----------
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        model, eps_total = key
        for row_key in sorted(groups[key], key=config_key):
            s = groups[key][row_key]
            if is_screening(s):
                continue  # 19 queries would read as a thin band beside 180
            ax.scatter(s["trigger"], s["savings"], s=7, alpha=0.35, label=s["config"])
        ax.axhline(eps_total, ls="--", lw=1, color="grey")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, eps_total * 1.12)
        ax.set_xlabel("trigger rate (this query)")
        ax.set_ylabel("epsilon saved (this query)")
        ax.set_title(f"{model.split('/')[-1]}  |  eps_total = {eps_total:g}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, markerscale=2, loc="lower right")
    fig.suptitle("Stage 4.1  the same axes, one point per query -- the spread the "
                 "summary plot hides", fontsize=11)
    fig.tight_layout()
    per_query = out_dir / "stage4_1_savings_by_query.png"
    fig.savefig(per_query, dpi=150)
    plt.close(fig)

    print()
    print(f"wrote {per_config}")
    print(f"wrote {per_query}")


def main():
    groups = collect()
    table(groups)
    figures(groups)


if __name__ == "__main__":
    main()
