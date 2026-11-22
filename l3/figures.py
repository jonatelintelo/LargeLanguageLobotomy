"""Stage 3's figures of one model's expert ranking.

scores_figure     every local expert's ensemble score, sorted, and the share of the total positive score held by the
                  top x% of local experts: how concentrated the ranking is
agreement_figure  the Jaccard overlap between the ensemble members' own top-x% expert sets against x, the overlap of
                  their positively scored sets, and the overlap two random sets of the same size would have
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from l3 import attribution  # noqa: E402

COLOR, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 10}


def scores_figure(ranked, model, path):
    """`ranked` is stage 3's ranking: ((layer, expert), score) for every local expert, highest first."""
    scores = np.array([score for _, score in ranked])
    n, n_pos = len(scores), int((scores > 0).sum())
    x = 100 * np.arange(1, n + 1) / n
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
        fig.suptitle(model)
        left.plot(x, scores, color=COLOR, linewidth=2)
        left.axhline(0, color=MUTED, linewidth=1)
        left.axvline(100 * n_pos / n, color=MUTED, linewidth=1, linestyle=":")
        left.annotate(f"{n_pos} of {n} positive ({100 * n_pos / n:.1f}%)", xy=(100 * n_pos / n, 0),
                      xytext=(6, 12), textcoords="offset points", color=MUTED, fontsize=9)
        left.set(title="Ensemble score per local expert, sorted", xlabel="Rank (% of local experts)",
                 ylabel="Score (member s.d. units, averaged)", xlim=(0, 100))
        mass = np.cumsum(np.clip(scores, 0, None))
        if mass[-1] > 0:
            share = 100 * mass / mass[-1]
            right.plot(x, share, color=COLOR, linewidth=2)
            for p in (1, 5, 10, 20):
                k = max(1, round(p / 100 * n)) - 1
                right.plot([x[k]], [share[k]], marker="o", markersize=6, color=COLOR, markeredgecolor=SURFACE,
                           markeredgewidth=1.5)
                right.annotate(f"top {p}%: {share[k]:.0f}%", xy=(x[k], share[k]), xytext=(8, -12),
                               textcoords="offset points", color=MUTED, fontsize=9)
        right.set(title="Share of the positive score held by the top experts", xlabel="Top x% of local experts",
                  ylabel="Share of the total positive score (%)", xlim=(0, 60), ylim=(0, 101))
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)


def agreement_figure(agreement, model, path):
    """`agreement` is attribution.member_agreement's result."""
    top, curve, pos = agreement["top_percent"], agreement["curve"], agreement["positive"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        chance = np.linspace(0, 100, 201)
        ax.plot(chance, 100 * attribution.chance_jaccard(chance / 100), color=MUTED, linewidth=1.5, linestyle="--",
                label="Chance: two random sets of that size")
        ax.fill_between(top, [100 * c["min"] for c in curve], [100 * c["max"] for c in curve], color=COLOR,
                        alpha=0.18, linewidth=0)
        ax.plot(top, [100 * c["mean"] for c in curve], color=COLOR, linewidth=2,
                label="Top x%: mean over member pairs (band: min to max)")
        ax.plot([100 * pos["share"]], [100 * pos["mean"]], marker="o", markersize=8, color=COLOR,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                label=f"Each member's positively scored experts: {100 * pos['mean']:.0f}% (min {100 * pos['min']:.0f}%)")
        ax.set(title=f"{model}: agreement between the ensemble members",
               xlabel="Top x% of local experts, by each member's own ranking", ylabel="Jaccard overlap (%)",
               xlim=(0, 100), ylim=(0, 101))
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
