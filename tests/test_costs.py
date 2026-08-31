import numpy as np
import pandas as pd
import pytest

from mma_efficiency.costs import (
    EXECUTION_SCENARIOS,
    KALSHI_MAKER_RATE,
    KALSHI_TAKER_RATE,
    breakeven_friction,
    fee_scenario_table,
    kalshi_fee,
    scenario_returns,
)


def make_bets(n=400, seed=0, q_extra=0.02):
    """Synthetic heavy-favorite bets: de-vigged mid ~U(0.75, 0.90), true win
    prob = mid + q_extra, raw price = mid * 1.04 (proportional 4% overround
    share)."""
    rng = np.random.default_rng(seed)
    p_basic = rng.uniform(0.75, 0.90, n)
    p_raw = np.minimum(p_basic * 1.04, 0.99)
    won = (rng.random(n) < p_basic + q_extra).astype(int)
    return pd.DataFrame(
        {
            "fight_id": np.arange(n),
            "p_basic": p_basic,
            "p_raw": p_raw,
            "won": won,
        }
    )


def test_kalshi_fee_values_match_schedule():
    # peak of the parabola: 1.75 cents taker, 0.4375 cents maker at 50 cents
    assert kalshi_fee(0.5) == pytest.approx(0.0175)
    assert kalshi_fee(0.5, rate=KALSHI_MAKER_RATE) == pytest.approx(0.004375)
    # a -300 favorite bought at its 75-cent de-vigged mid
    assert kalshi_fee(0.75) == pytest.approx(0.07 * 0.75 * 0.25)


def test_kalshi_fee_shape():
    p = np.linspace(0.0, 1.0, 101)
    f = kalshi_fee(p)
    assert f[0] == 0.0 and f[-1] == pytest.approx(0.0)
    assert np.max(f) == pytest.approx(0.0175)
    # symmetric and concave: fee falls toward the extremes
    assert np.allclose(f, f[::-1])
    assert kalshi_fee(0.9) < kalshi_fee(0.75) < kalshi_fee(0.5)


def test_book_close_scenario_equals_flat_bet_return():
    bets = make_bets()
    ret = scenario_returns(bets, "book_close")
    # flat 1-unit bet at american odds == won/p_raw - 1
    expected = bets["won"].to_numpy() / bets["p_raw"].to_numpy() - 1.0
    assert np.allclose(ret, expected)


def test_scenario_cost_ordering():
    """More friction -> lower ROI, bet by bet."""
    bets = make_bets()
    r_maker = scenario_returns(bets, "kalshi_mid_maker")
    r_taker = scenario_returns(bets, "kalshi_mid_taker")
    r_sp1 = scenario_returns(bets, "kalshi_spread1_taker")
    r_sp2 = scenario_returns(bets, "kalshi_spread2_taker")
    r_raw = scenario_returns(bets, "kalshi_raw_taker")
    won = bets["won"].to_numpy() == 1
    # on winning bets the payout strictly falls with cost; on losers all lose
    for a, b in [(r_maker, r_taker), (r_taker, r_sp1), (r_sp1, r_sp2)]:
        assert np.all(a[won] > b[won])
        assert np.all(a[~won] == -1.0) or np.all(a[~won] > -1.0)  # bounded below by -1
    assert r_raw[won].mean() < r_taker[won].mean()


def test_kalshi_beats_book_at_same_underlying_price():
    """At the same fair mid, exchange fees cost less than the book's margin
    for heavy favorites: mid-taker ROI must exceed the sportsbook baseline."""
    bets = make_bets()
    roi_book = scenario_returns(bets, "book_close").mean()
    roi_kalshi = scenario_returns(bets, "kalshi_mid_taker").mean()
    assert roi_kalshi > roi_book


def test_fee_scenario_table_structure():
    bets = make_bets()
    tab = fee_scenario_table(bets, n_boot=300, seed=1)
    assert set(tab["scenario"]) == set(EXECUTION_SCENARIOS)
    assert np.all(tab["ci_lo"] <= tab["roi"]) and np.all(tab["roi"] <= tab["ci_hi"])
    assert np.all((tab["p_roi_gt0"] >= 0) & (tab["p_roi_gt0"] <= 1))
    # baseline has no fee; kalshi scenarios do
    base = tab.set_index("scenario")
    assert base.loc["book_close", "mean_fee_per_contract"] == 0.0
    assert base.loc["kalshi_mid_taker", "mean_fee_per_contract"] > 0.0


def test_breakeven_friction_consistency():
    bets = make_bets(n=20000, seed=3, q_extra=0.03)
    b = breakeven_friction(bets)
    # by construction the edge vs mid is ~3 points
    assert b["edge_vs_mid"] == pytest.approx(0.03, abs=0.01)
    # breakeven friction approx edge/mid
    assert b["breakeven_total_friction_pct_of_mid"] == pytest.approx(
        b["edge_vs_mid"] / b["mean_mid"], abs=1e-9
    )
    # residual for spread = edge minus the fee, in cents
    assert b["residual_for_spread_cents"] == pytest.approx(
        100 * (b["edge_vs_mid"] - b["kalshi_taker_fee_at_mid"]), abs=1e-9
    )
