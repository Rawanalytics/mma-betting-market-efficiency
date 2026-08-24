import numpy as np
import pandas as pd
import pytest

from mma_efficiency.calibration import (
    _benjamini_hochberg,
    calibration_table,
    logistic_calibration_test,
    wilson_ci,
)
from mma_efficiency.returns import band_label, returns_by_band


def make_synthetic_sides(n_fights: int = 4000, slope: float = 1.0, seed: int = 0):
    """Synthetic two-sided market with true prob = sigmoid(slope * logit(p)).

    slope = 1 -> perfectly calibrated market; slope > 1 -> favorite-longshot
    bias (favorites win more often than priced).
    """
    rng = np.random.default_rng(seed)
    p_market = rng.uniform(0.15, 0.85, n_fights)
    logit = np.log(p_market / (1 - p_market))
    p_true = 1 / (1 + np.exp(-slope * logit))
    won_a = rng.random(n_fights) < p_true

    rows = []
    for i in range(n_fights):
        for p, won in ((p_market[i], won_a[i]), (1 - p_market[i], not won_a[i])):
            dec = 1 / (p * 1.05)  # 5% proportional vig
            rows.append(
                {
                    "fight_id": i,
                    "p_raw": p * 1.05,
                    "p_basic": p,
                    "p_shin": p,
                    "odds": -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p,
                    "dec": dec,
                    "overround": 1.05,
                    "won": int(won),
                    "ret": dec - 1 if won else -1.0,
                }
            )
    return pd.DataFrame(rows)


def test_wilson_ci_contains_point_estimate():
    lo, hi = wilson_ci(np.array([30]), np.array([100]))
    assert lo < 0.3 < hi
    assert 0 <= lo and hi <= 1


def test_wilson_ci_narrows_with_n():
    lo1, hi1 = wilson_ci(np.array([30]), np.array([100]))
    lo2, hi2 = wilson_ci(np.array([300]), np.array([1000]))
    assert (hi2 - lo2) < (hi1 - lo1)


def test_benjamini_hochberg_bounds_and_order():
    p = np.array([0.001, 0.01, 0.04, 0.5, 0.9])
    q = _benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1)
    # BH preserves the ordering of p-values
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_logistic_test_recovers_calibrated_market():
    sides = make_synthetic_sides(n_fights=8000, slope=1.0, seed=1)
    res = logistic_calibration_test(sides, "p_basic")
    assert res["slope"] == pytest.approx(1.0, abs=3.5 * res["slope_se"])
    assert res["joint_wald_p"] > 0.01


def test_logistic_test_detects_favorite_longshot_bias():
    sides = make_synthetic_sides(n_fights=8000, slope=1.35, seed=2)
    res = logistic_calibration_test(sides, "p_basic")
    assert res["slope"] > 1.1
    assert res["slope_p_vs_1"] < 0.01


def test_calibration_table_mirrored_buckets():
    sides = make_synthetic_sides(n_fights=3000, slope=1.0, seed=3)
    tab = calibration_table(sides, "p_basic")
    assert tab["n"].sum() == len(sides) - _n_deduped(sides)
    assert np.all(tab["actual_lo"] <= tab["actual"])
    assert np.all(tab["actual"] <= tab["actual_hi"])


def _n_deduped(sides):
    # rows removed because both sides of a fight share a bucket
    bucket = pd.cut(sides["p_basic"], bins=np.arange(0, 1.0001, 0.1), right=False)
    return len(sides) - sides.assign(b=bucket).drop_duplicates(["fight_id", "b"]).shape[0]


def test_calibration_table_equal_count_bins():
    sides = make_synthetic_sides(n_fights=3000, slope=1.0, seed=6)
    tab = calibration_table(sides, "p_basic", equal_count=12)
    assert len(tab) == 12
    # quantile bins hold near-equal counts (the within-fight dedupe can only
    # thin the bucket(s) straddling 0.5)
    assert tab["n"].min() >= 0.7 * tab["n"].max()
    # implied probability is monotone across quantile bins
    assert np.all(np.diff(tab["implied"]) > 0)
    lo, hi = tab["actual_lo"], tab["actual_hi"]
    assert np.all(lo <= tab["actual"]) and np.all(tab["actual"] <= hi)


def test_returns_by_band_efficient_market_matches_vig_null():
    sides = make_synthetic_sides(n_fights=8000, slope=1.0, seed=4)
    tab = returns_by_band(sides, n_boot=500, seed=4)
    # In a calibrated market with proportional vig every band's true expected
    # return is 1/1.05 - 1 = -4.76%; empirical means should sit inside CIs
    # that cover the null for the vast majority of bands.
    covered = ((tab["vig_null_return"] >= tab["ci_lo"]) & (tab["vig_null_return"] <= tab["ci_hi"]))
    assert covered.mean() > 0.7


def test_band_label_edges():
    odds = pd.Series([-2000.0, -800.0, -300.0, -140.0, 100.0, 140.0, 499.0, 500.0, 1200.0])
    labels = band_label(odds)
    assert labels.iloc[0] == "-800 or shorter"
    assert labels.iloc[1] == "-800 or shorter"  # -800 is inside its named band
    assert labels.iloc[2] == "-499 to -300"
    assert labels.iloc[3] == "-199 to -140"
    assert labels.iloc[4] == "+100 to +139"
    assert labels.iloc[5] == "+140 to +199"
    assert labels.iloc[6] == "+300 to +499"
    assert labels.iloc[7] == "+500 or longer"
    assert not labels.isna().any()
