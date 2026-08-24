"""Economic tests: flat-bet returns by odds band and simple strategy backtests.

The economic version of the efficiency null: if de-vigged prices are true
probabilities and the margin is applied proportionally, a flat 1-unit bet on
*any* side has expected return 1/r - 1 (r = overround), regardless of price.
Favorite-longshot bias shows up as expected returns that *fall* as the odds
get longer.

All confidence intervals resample whole fights (cluster bootstrap), because
the two sides of a fight are mechanically anticorrelated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = [
    "ODDS_BANDS",
    "returns_by_band",
    "returns_trend_test",
    "backtest_rules",
    "cumulative_pnl",
]

# American-odds bands, favorites -> underdogs. Cut points sit on half-integers
# so integer odds land unambiguously in the band their label names.
ODDS_BANDS = [
    (-np.inf, -799.5, "-800 or shorter"),
    (-799.5, -499.5, "-799 to -500"),
    (-499.5, -299.5, "-499 to -300"),
    (-299.5, -199.5, "-299 to -200"),
    (-199.5, -139.5, "-199 to -140"),
    (-139.5, 99.5, "-139 to +99"),
    (99.5, 139.5, "+100 to +139"),
    (139.5, 199.5, "+140 to +199"),
    (199.5, 299.5, "+200 to +299"),
    (299.5, 499.5, "+300 to +499"),
    (499.5, np.inf, "+500 or longer"),
]


def band_label(odds: pd.Series) -> pd.Series:
    lab = pd.Series(pd.NA, index=odds.index, dtype="object")
    for lo, hi, name in ODDS_BANDS:
        lab[(odds >= lo) & (odds < hi)] = name
    return lab


def _cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 4000,
    seed: int = 7,
    alpha: float = 0.05,
):
    """Bootstrap CI for the mean of `values`, resampling cluster ids."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    # sum and count of values per cluster
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    # bootstrap p-value against the vig-only null is computed by the caller
    return float(lo), float(hi), boot_means


def returns_by_band(
    sides: pd.DataFrame, n_boot: int = 4000, seed: int = 7
) -> pd.DataFrame:
    """Mean flat-bet return per odds band with cluster-bootstrap CIs and the
    efficient-market benchmark 1/r - 1 for that band."""
    d = sides.copy()
    d["band"] = band_label(d["odds"])

    rows = []
    for _, _, name in ODDS_BANDS:
        sub = d[d["band"] == name]
        if len(sub) == 0:
            continue
        lo, hi, boot = _cluster_bootstrap_mean(
            sub["ret"].to_numpy(), sub["fight_id"].to_numpy(), n_boot=n_boot, seed=seed
        )
        null_ret = float((1.0 / sub["overround"]).mean() - 1.0)
        mean_ret = float(sub["ret"].mean())
        # two-sided bootstrap p-value for H0: E[ret] = null_ret
        shifted = boot - boot.mean() + null_ret
        p_boot = float(
            min(1.0, 2 * min((shifted >= mean_ret).mean(), (shifted <= mean_ret).mean()))
        )
        rows.append(
            {
                "band": name,
                "n_bets": int(len(sub)),
                "mean_odds": float(sub["odds"].mean()),
                "implied_raw": float(sub["p_raw"].mean()),
                "implied_devig": float(sub["p_basic"].mean()),
                "win_rate": float(sub["won"].mean()),
                "mean_return": mean_ret,
                "ci_lo": lo,
                "ci_hi": hi,
                "vig_null_return": null_ret,
                "p_vs_vig_null": p_boot,
            }
        )
    return pd.DataFrame(rows)


def returns_trend_test(sides: pd.DataFrame) -> dict:
    """OLS of flat-bet return on raw implied probability with fight-clustered
    SEs. A positive slope = shorter prices are better bets = favorite-longshot
    bias in the economic sense."""
    X = sm.add_constant(sides[["p_raw"]].rename(columns={"p_raw": "implied_prob"}))
    res = sm.OLS(sides["ret"].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": sides["fight_id"].to_numpy()}
    )
    return {
        "slope": float(res.params["implied_prob"]),
        "slope_se": float(res.bse["implied_prob"]),
        "t_stat": float(res.tvalues["implied_prob"]),
        "p_value": float(res.pvalues["implied_prob"]),
        "n_sides": int(res.nobs),
    }


# ---------------------------------------------------------------------------
# Strategy backtests
# ---------------------------------------------------------------------------

def backtest_rules(sides: pd.DataFrame, n_boot: int = 4000, seed: int = 11) -> pd.DataFrame:
    """Flat 1-unit stake on every side matching each rule, in chronological
    order. Reports ROI, bootstrap CI, win rate, and max drawdown.

    These are *diagnostics of historical mispricing*, not betting advice: they
    use closing lines, ignore limits/fees, and assume every listed price was
    available.
    """
    rules = {
        "All favorites": sides["is_favorite"],
        "All underdogs": sides["is_underdog"],
        "Heavy favorites (-300 or shorter)": sides["odds"] <= -300,
        "Moderate favorites (-299 to -105)": (sides["odds"] > -300) & (sides["odds"] < -100),
        "Moderate underdogs (+100 to +299)": (sides["odds"] >= 100) & (sides["odds"] < 300),
        "Heavy underdogs (+300 or longer)": sides["odds"] >= 300,
        "Extreme underdogs (+500 or longer)": sides["odds"] >= 500,
    }
    rows = []
    for name, mask in rules.items():
        sub = sides[mask].sort_values("date")
        if len(sub) == 0:
            continue
        lo, hi, _ = _cluster_bootstrap_mean(
            sub["ret"].to_numpy(), sub["fight_id"].to_numpy(), n_boot=n_boot, seed=seed
        )
        pnl = sub["ret"].cumsum()
        peak = pnl.cummax()
        rows.append(
            {
                "strategy": name,
                "n_bets": int(len(sub)),
                "total_staked": float(len(sub)),
                "total_profit": float(sub["ret"].sum()),
                "roi": float(sub["ret"].mean()),
                "roi_ci_lo": lo,
                "roi_ci_hi": hi,
                "win_rate": float(sub["won"].mean()),
                "max_drawdown": float((pnl - peak).min()),
            }
        )
    return pd.DataFrame(rows)


def cumulative_pnl(sides: pd.DataFrame, rules: dict[str, pd.Series]) -> pd.DataFrame:
    """Cumulative profit series (1-unit flat stakes) for each rule, indexed by
    bet number with the date carried along for plotting."""
    frames = []
    for name, mask in rules.items():
        sub = sides[mask].sort_values("date")
        frames.append(
            pd.DataFrame(
                {
                    "strategy": name,
                    "date": sub["date"].to_numpy(),
                    "bet_number": np.arange(1, len(sub) + 1),
                    "cum_profit": sub["ret"].cumsum().to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
