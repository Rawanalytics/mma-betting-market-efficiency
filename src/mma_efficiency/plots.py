"""Publication figures (light theme, GitHub-README friendly).

Style follows a small design system: recessive hairline grid, thin marks,
series colors reserved for marks (text always in ink tokens), direct labels
on the points the story is about, a legend whenever there are >= 2 series.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- palette (validated: see README methodology notes) ---------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1 = "#2a78d6"  # blue
S2 = "#eb6834"  # orange
S3 = "#1baf7a"  # aqua
S4 = "#eda100"  # yellow

STRATEGY_COLORS = [S1, S2, S3, S4]


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _fig_caption(fig, text: str, y: float = -0.01) -> None:
    fig.text(0.01, y, text, ha="left", va="top", fontsize=8.5, color=MUTED)


# ---------------------------------------------------------------------------
def plot_calibration(
    tab: pd.DataFrame,
    path: str,
    prob_label: str = "de-vigged",
    title: str = "The market is well calibrated in the middle — and bent at the tails",
    caption: str = "Deciles of the de-vigged implied probability; buckets with fewer than "
                   "50 bets omitted from the plot (all buckets in results/calibration_buckets.csv).",
) -> None:
    """Calibration curve: market implied probability vs realized win rate."""
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 6.2))

    tab = tab[tab["n"] >= 50].reset_index(drop=True)  # tiny buckets -> CSV only

    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.4, zorder=1,
            label="Perfect calibration")
    ax.errorbar(
        tab["implied"], tab["actual"],
        yerr=[tab["actual"] - tab["actual_lo"], tab["actual_hi"] - tab["actual"]],
        fmt="o-", color=S1, lw=2, ms=7.5, capsize=3, capthick=1.2,
        elinewidth=1.2, ecolor=INK_2,
        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
        label=f"UFC market ({prob_label}), 95% Wilson CI",
    )

    # Direct labels on the two tails -- the story of the chart.
    lo_row = tab.iloc[0]
    hi_row = tab.iloc[-1]
    ax.annotate(
        f"Longshots overpriced:\npriced {lo_row['implied']:.0%}, win {lo_row['actual']:.0%}",
        xy=(lo_row["implied"], lo_row["actual"]), xytext=(0.24, 0.04),
        fontsize=9.5, color=INK_2,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkB=4),
    )
    ax.annotate(
        f"Favorites underpriced:\npriced {hi_row['implied']:.0%}, win {hi_row['actual']:.0%}",
        xy=(hi_row["implied"], hi_row["actual"]), xytext=(0.98, 0.50),
        fontsize=9.5, color=INK_2, ha="right",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkB=4),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, zorder=0)
    _despine(ax)
    ax.set_xlabel("Market implied win probability (vig removed)")
    ax.set_ylabel("Actual win rate")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9.5)
    _fig_caption(fig, caption)
    fig.savefig(path)
    plt.close(fig)


def plot_returns_by_band(tab: pd.DataFrame, path: str) -> None:
    """Mean flat-bet return by American-odds band, favorites -> underdogs."""
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    t = tab.iloc[::-1].reset_index(drop=True)  # dogs at bottom -> favorites top
    y = np.arange(len(t))
    vals = t["mean_return"] * 100

    null_ret = float(tab["vig_null_return"].mean() * 100)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=5)
    ax.axvline(null_ret, color=MUTED, lw=1.2, zorder=5)
    ax.annotate(
        "efficient market:\nlose only the vig",
        xy=(null_ret, len(t) - 2.45), xytext=(-21, len(t) - 1.75),
        fontsize=8.5, color=MUTED, ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkB=3),
    )

    ax.barh(y, vals, height=0.62, color=S1, zorder=3,
            edgecolor=SURFACE, linewidth=1.0)
    err_lo = vals - t["ci_lo"] * 100
    err_hi = t["ci_hi"] * 100 - vals
    ax.errorbar(vals, y, xerr=[err_lo, err_hi], fmt="none",
                ecolor=INK_2, elinewidth=1.2, capsize=3, capthick=1.2, zorder=4)

    # Direct labels where the story is: the two heavy-underdog bands and the
    # heavy-favorite bands. Placed above the bar end so they never collide
    # with the CI whisker or the axis.
    for i, row in t.iterrows():
        if row["band"] in ("+500 or longer", "+300 to +499", "-799 to -500", "-800 or shorter"):
            v = row["mean_return"] * 100
            ax.text(v, i + 0.42, f"{v:+.1f}%", va="bottom", ha="center",
                    fontsize=9.5, color=INK, fontweight="semibold")

    ax.set_yticks(y, t["band"])
    ax.set_xlabel("Mean return per 1-unit flat bet (%), 95% cluster-bootstrap CI")
    ax.set_title("Flat-betting heavy underdogs lost ~30 cents per dollar")
    ax.grid(True, axis="x", zorder=0)
    ax.grid(False, axis="y")
    _despine(ax)
    _fig_caption(
        fig,
        "One bet per listed price (both sides of every fight). "
        "Under an efficient market with proportional vig, every band's expected return is the gray line.",
    )
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative_pnl(pnl: pd.DataFrame, path: str) -> None:
    """Cumulative flat-stake profit over time for the headline strategies."""
    _style()
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    strategies = list(dict.fromkeys(pnl["strategy"]))
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)

    ends = []
    for i, strat in enumerate(strategies):
        sub = pnl[pnl["strategy"] == strat]
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        dates = pd.to_datetime(sub["date"])
        ax.plot(dates, sub["cum_profit"], lw=2, color=color,
                solid_capstyle="round", zorder=3, label=strat)
        ends.append((dates.iloc[-1], sub["cum_profit"].iloc[-1], strat, color))

    # Direct end labels (relief for low-contrast hues), collision-nudged.
    ends.sort(key=lambda e: e[1])
    y_min, y_max = ax.get_ylim()
    min_gap = (y_max - y_min) * 0.045
    placed = []
    for x, yv, strat, color in ends:
        y_lab = yv
        if placed and y_lab - placed[-1] < min_gap:
            y_lab = placed[-1] + min_gap
        placed.append(y_lab)
        ax.annotate(
            f"{strat}  {yv:+,.0f}u",
            xy=(x, yv), xytext=(10, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=INK_2,
            annotation_clip=False,
        )
        ax.plot([x], [yv], "o", ms=7, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                clip_on=False)

    ax.set_xlim(right=pnl["date"].max() + pd.Timedelta(days=40))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Cumulative profit (units, 1-unit flat stakes)")
    ax.set_title("Sixteen years of flat-betting every price in the band")
    ax.grid(True, axis="y", zorder=0)
    ax.grid(False, axis="x")
    _despine(ax)
    ax.legend(loc="lower left", fontsize=9, ncols=2)
    fig.subplots_adjust(right=0.72)
    fig.savefig(path)
    plt.close(fig)


def plot_slope_stability(rows: list[dict], path: str) -> None:
    """Calibration slope (with 95% CI) by era; slope 1 = efficient."""
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))

    y = np.arange(len(rows))[::-1]
    slopes = np.array([r["slope"] for r in rows])
    los = np.array([r["slope"] - 1.96 * r["slope_se"] for r in rows])
    his = np.array([r["slope"] + 1.96 * r["slope_se"] for r in rows])

    ax.axvline(1.0, color=MUTED, lw=1.4, zorder=2)
    ax.text(1.005, -0.38, "slope = 1: prices match outcomes",
            fontsize=8.5, color=MUTED, va="bottom", ha="left")

    ax.errorbar(slopes, y, xerr=[slopes - los, his - slopes], fmt="o",
                color=S1, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                ecolor=INK_2, elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=3)
    for yi, r in zip(y, rows):
        ax.text(r["slope"], yi + 0.22, f"{r['slope']:.2f}", ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="semibold")

    ax.set_ylim(-0.55, len(rows) - 0.35)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlabel("Logistic calibration slope on de-vigged probabilities (95% CI, fight-clustered)")
    ax.set_title("The mispricing is strongest in recent years")
    ax.grid(True, axis="x", zorder=0)
    ax.grid(False, axis="y")
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, wf: dict, path: str) -> None:
    """Favorite-cutoff sweep: flat-bet ROI with cluster-bootstrap CIs, plus
    the walk-forward out-of-sample verdict for the pre-registered rule."""
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    x = np.arange(len(sweep))
    roi = sweep["roi"] * 100

    lo_err = roi - sweep["ci_lo"] * 100
    hi_err = sweep["ci_hi"] * 100 - roi
    y_top = float((sweep["ci_hi"] * 100).max())
    y_bot = float((sweep["ci_lo"] * 100).min())
    ax.set_ylim(y_bot - 0.6, y_top + 2.1)  # headroom for the verdict text

    ax.axhline(0, color=MUTED, lw=1.4, zorder=2)
    vig = float(sweep["vig_null"].mean() * 100)
    ax.axhline(vig, color=BASELINE, lw=1.2, zorder=2)
    ax.text(-0.35, vig - 0.25, "efficient market (lose the vig)",
            fontsize=8.5, color=MUTED, ha="left", va="top")

    ax.errorbar(x, roi, yerr=[lo_err, hi_err],
                fmt="o-", lw=2, color=S1, ms=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, ecolor=INK_2, elinewidth=1.3,
                capsize=3.5, capthick=1.3, zorder=3)

    # Ring the rule the walk-forward test selected on training data only, and
    # state its out-of-sample verdict in the free top-left corner.
    hit = np.where(sweep["cutoff"].to_numpy() == wf["chosen_cutoff"])[0]
    if len(hit):
        i = int(hit[0])
        ax.plot([i], [roi.iloc[i]], "o", ms=16, mfc="none", mec=INK_2,
                mew=1.4, zorder=4)
        ax.text(
            -0.35, y_top + 1.95,
            f"Walk-forward: rule '{wf['chosen_rule']}' (ringed) picked on "
            f"pre-{str(wf['train_end'])[:4]} data only;\n"
            f"out of sample it returned {wf['test_roi']*100:+.1f}% "
            f"[{wf['test_roi_ci'][0]*100:+.1f}, {wf['test_roi_ci'][1]*100:+.1f}] "
            f"per bet on {wf['test_n_bets']:,} bets — "
            f"beats the vig, but p(ROI>0) = {wf['test_p_roi_gt0']:.2f}",
            fontsize=9, color=INK_2, ha="left", va="top",
        )

    ax.set_xticks(x, [f"≤ {int(c)}" for c in sweep["cutoff"]])
    ax.set_xlabel("Rule: flat-bet every favorite at least this heavy (American odds)")
    ax.set_ylabel("Return per 1-unit bet (%), 95% cluster-bootstrap CI")
    ax.set_title("Betting heavy favorites beats the vig — but not zero, reliably")
    ax.grid(True, axis="y", zorder=0)
    ax.grid(False, axis="x")
    _despine(ax)
    _fig_caption(fig, "Full-sample sweep shown for context; the annotated verdict is the "
                      "walk-forward test, whose rule was chosen on the training era only.")
    fig.savefig(path)
    plt.close(fig)


def plot_fee_scenarios(tab: pd.DataFrame, path: str) -> None:
    """Out-of-sample ROI of the walk-forward favorites rule under each
    execution scenario (sportsbook baseline vs Kalshi price/fee assumptions),
    with fight-cluster bootstrap CIs."""
    _style()
    t = tab.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(t))[::-1]
    roi = t["roi"] * 100

    ax.axvline(0, color=MUTED, lw=1.4, zorder=2)
    ax.errorbar(
        roi, y,
        xerr=[roi - t["ci_lo"] * 100, t["ci_hi"] * 100 - roi],
        fmt="o", color=S1, ms=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
        ecolor=INK_2, elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=3,
    )
    for yi, (_, row) in zip(y, t.iterrows()):
        p = row["p_roi_gt0"]
        p_txt = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        ax.text(row["roi"] * 100, yi + 0.24,
                f"{row['roi']*100:+.1f}%  ({p_txt})",
                ha="center", va="bottom", fontsize=8.8, color=INK,
                fontweight="semibold")

    labels = [
        f"{row['label']}\nfee {row['mean_fee_pct_of_stake']:.1%} of stake"
        if row["mean_fee_per_contract"] > 0 else row["label"]
        for _, row in t.iterrows()
    ]
    ax.set_ylim(-0.55, len(t) - 0.2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Out-of-sample ROI per bet (%), 95% cluster-bootstrap CI; p = one-sided p(ROI ≤ 0)")
    ax.set_title("The same bets, net of prediction-market costs")
    ax.grid(True, axis="x", zorder=0)
    ax.grid(False, axis="y")
    _despine(ax)
    _fig_caption(
        fig,
        "Walk-forward rule (favorites ≤ −300, chosen on 2010–2017) staked on 2018–2026, n = 975 bets.\n"
        "Kalshi rows are counterfactual: fees follow the official schedule "
        "(taker 0.07·P·(1−P), maker 0.0175·P·(1−P) per contract; no settlement fee); "
        "prices assume the exchange mid tracks the de-vigged sportsbook consensus.",
        y=-0.04,
    )
    fig.savefig(path)
    plt.close(fig)


def plot_ml_benchmark(res: pd.DataFrame, path: str) -> None:
    """Out-of-sample log-loss difference vs the market, with bootstrap CIs."""
    _style()
    models = res[res["model"] != "Market (de-vigged odds)"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    y = np.arange(len(models))[::-1]

    ax.axvline(0, color=MUTED, lw=1.4, zorder=2)
    ax.set_ylim(-0.55, len(models) - 0.45)
    ax.text(0.0004, -0.38, "0 = exactly as good as the closing line",
            fontsize=8.5, color=MUTED, va="bottom", ha="left")

    d = models["logloss_diff"]
    ax.errorbar(d, y,
                xerr=[d - models["diff_ci_lo"], models["diff_ci_hi"] - d],
                fmt="o", color=S1, ms=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, ecolor=INK_2, elinewidth=1.4,
                capsize=3.5, capthick=1.4, zorder=3)

    ax.set_yticks(y, models["model"])
    ax.set_xlabel("Test-set log-loss vs market  (negative = model beats the market)")
    ax.set_title("Public fighter stats do not beat the closing line")
    ax.grid(True, axis="x", zorder=0)
    ax.grid(False, axis="y")
    _despine(ax)
    _fig_caption(fig, "Chronological split: fit on fights to 2021, evaluated on 2022–2026. "
                      "95% paired-bootstrap CIs over test fights.", y=-0.14)
    fig.savefig(path)
    plt.close(fig)
