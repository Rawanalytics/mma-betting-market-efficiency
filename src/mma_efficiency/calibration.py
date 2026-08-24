"""Calibration analysis and formal tests of the favorite-longshot bias.

Unit of observation is one *side* of a fight (one price a bettor could take).
The two sides of a fight are mirror images, so every test here either
clusters standard errors by fight or resamples whole fights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

__all__ = [
    "wilson_ci",
    "calibration_table",
    "logistic_calibration_test",
    "bucket_binomial_tests",
]

DEFAULT_BINS = np.round(np.arange(0.0, 1.0001, 0.1), 10)


def wilson_ci(k, n, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion (vectorized)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return center - half, center + half


def _dedupe_within_bucket(sides: pd.DataFrame, bucket: pd.Series) -> pd.DataFrame:
    """If both sides of the same fight fall in the same bucket (possible only
    around 0.5), keep one of them; otherwise the mirrored pair (y, 1-y) would
    artificially pin the bucket mean at 0.5 and understate its variance.

    Which side is kept alternates deterministically with the fight id. Always
    keeping the same corner would be biased: red corners win ~58% of
    pick'em-priced fights (a matchmaking/corner-assignment effect), so a
    keep-red rule would paint a spurious kink at 0.5. Alternating restores the
    'random side of an evenly priced fight' interpretation (~50%)."""
    tmp = sides.assign(_bucket=bucket.to_numpy())
    grp = tmp.groupby(["fight_id", "_bucket"], observed=True)
    rank = grp.cumcount()
    size = grp["fight_id"].transform("size")
    keep = (size == 1) | (rank == (tmp["fight_id"].to_numpy() % 2))
    return tmp[keep.to_numpy()]


def calibration_table(
    sides: pd.DataFrame,
    prob_col: str = "p_basic",
    bins: np.ndarray = DEFAULT_BINS,
    equal_count: int | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Group sides into implied-probability buckets and compare the market's
    mean implied probability with the realized win frequency (Wilson CI).

    Binning is fixed-width over `bins` by default. Pass ``equal_count=k`` to
    use k quantile buckets instead -- every bucket holds (nearly) the same
    number of bets, grouped by nearest market probability, which equalizes
    statistical power across the curve and adds resolution where prices
    cluster.
    """
    if equal_count is not None:
        bucket = pd.qcut(sides[prob_col], q=equal_count, duplicates="drop")
    else:
        bucket = pd.cut(sides[prob_col], bins=bins, right=False)
    deduped = _dedupe_within_bucket(sides, bucket)

    g = deduped.groupby("_bucket", observed=True)
    tab = g.agg(
        n=("won", "size"),
        wins=("won", "sum"),
        implied=(prob_col, "mean"),
        implied_raw=("p_raw", "mean"),
        mean_odds=("odds", "mean"),
    ).reset_index(names="bucket")
    tab["actual"] = tab["wins"] / tab["n"]
    lo, hi = wilson_ci(tab["wins"], tab["n"], alpha=alpha)
    tab["actual_lo"] = lo
    tab["actual_hi"] = hi
    tab["gap"] = tab["actual"] - tab["implied"]
    return tab


def logistic_calibration_test(
    sides: pd.DataFrame, prob_col: str = "p_basic"
) -> dict:
    """Mincer-Zarnowitz-style calibration test on the logit scale.

    Fits  P(win) = sigmoid(a + b * logit(p_hat))  with standard errors
    clustered by fight, and Wald-tests the joint null of perfect calibration
    H0: a = 0, b = 1.

    Interpretation: b > 1 means outcomes are *more* extreme than prices --
    favorites win even more often than the market says (the classic
    favorite-longshot bias); b < 1 means prices are too extreme.

    Note: for de-vigged probability columns the two-sided (mirrored) design
    forces the intercept to 0 by construction, so the joint Wald test is
    effectively a 1-degree-of-freedom test of the slope; ``slope_p_vs_1`` is
    the primary statistic to quote.
    """
    d = sides.dropna(subset=[prob_col]).copy()
    logit_p = np.log(d[prob_col] / (1 - d[prob_col]))
    X = sm.add_constant(pd.DataFrame({"logit_p": logit_p}))
    model = sm.GLM(d["won"].to_numpy(), X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d["fight_id"].to_numpy()})

    wald = res.wald_test("(const = 0), (logit_p = 1)", scalar=True)
    a, b = res.params["const"], res.params["logit_p"]
    se_a, se_b = res.bse["const"], res.bse["logit_p"]
    return {
        "prob_col": prob_col,
        "n_sides": int(len(d)),
        "n_fights": int(d["fight_id"].nunique()),
        "intercept": float(a),
        "intercept_se": float(se_a),
        "slope": float(b),
        "slope_se": float(se_b),
        "slope_z_vs_1": float((b - 1) / se_b),
        "slope_p_vs_1": float(2 * stats.norm.sf(abs((b - 1) / se_b))),
        "joint_wald_stat": float(wald.statistic),
        "joint_wald_p": float(wald.pvalue),
    }


def bucket_binomial_tests(
    sides: pd.DataFrame,
    prob_col: str = "p_basic",
    bins: np.ndarray = DEFAULT_BINS,
    equal_count: int | None = None,
) -> pd.DataFrame:
    """Exact two-sided binomial test per bucket of H0: win rate equals the
    bucket's mean de-vigged implied probability, with Benjamini-Hochberg
    control of the false discovery rate across buckets."""
    tab = calibration_table(sides, prob_col=prob_col, bins=bins, equal_count=equal_count)
    pvals = [
        stats.binomtest(int(w), int(n), p0).pvalue
        for w, n, p0 in zip(tab["wins"], tab["n"], tab["implied"])
    ]
    tab["binom_p"] = pvals
    tab["binom_p_bh"] = _benjamini_hochberg(np.array(pvals))
    return tab


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values)."""
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(ranked, 1.0)
    return q
