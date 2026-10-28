"""Prediction-market execution costs: would the edge survive Kalshi's fees?

The rest of the repo measures mispricing against *sportsbook closing lines*,
where the bettor's friction is the bookmaker margin (~3.5% overround) baked
into the price. Event-contract exchanges price the same fights differently:
there is no vig inside the price — a YES contract on the favorite trades at
some price pi in (0, 1) and pays $1 — and the friction is instead (a) an
explicit trading fee charged on execution and (b) the bid-ask spread.

Kalshi's fee schedule (kalshi.com/docs/kalshi-fee-schedule.pdf, effective
July 7, 2026) charges, per contract, on execution only (settlement is free):

    taker:  fee = 0.07   * P * (1 - P)     [rounded up, cap at extreme prices]
    maker:  fee = 0.0175 * P * (1 - P)

The fee is parabolic in price: it peaks at P = $0.50 (1.75 cents/contract
taker) and shrinks toward the extremes — at the heavy-favorite prices this
study cares about (P ~ 0.75-0.90) the taker fee is 0.6-1.3 cents per
contract, i.e. 0.7-1.75% of the stake. As a fraction of the amount staked the
taker fee is exactly 0.07*(1-P): it *falls* as the favorite gets heavier,
whereas the sportsbook's effective margin does not.

Because no historical exchange prices exist for UFC fights over 2010-2026
(Kalshi's UFC markets launched in 2025), execution there is modeled under
explicit counterfactual price assumptions, from optimistic to worst-case:

* mid  — the contract trades at the de-vigged sportsbook consensus
         probability (proportional method, `p_basic`). This is the standard
         "prediction-market mid tracks the books' consensus" assumption.
* mid + spread — same, but the buyer crosses a 1- or 2-cent half-spread.
* raw  — worst case: the exchange price embeds the same margin as the book
         (`p_raw`), so the trader pays book price *and* the exchange fee.

Mechanics of one modeled bet: stake $1 on the favorite by buying
N = 1 / (pi + fee(pi)) YES contracts at price pi; at settlement collect
$N if the favorite wins, else $0.  Per-dollar return = won/(pi+fee) - 1.
Under "price = p_raw, fee = 0" this reduces *exactly* to the sportsbook
flat-bet return used everywhere else in the repo (won/p_raw - 1), which is
unit-tested — the scenarios are strict generalizations of the baseline.

The field module scores these price assumptions against live Kalshi quotes
collected going forward; see ``field.py`` and ``data/live_quotes.csv``.

All CIs and p-values use the same fight-cluster bootstrap as the rest of the
repo. Fees follow the exact formula without per-order rounding (rounding adds
at most one cent per order, negligible at any realistic order size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import _cluster_bootstrap_mean

__all__ = [
    "KALSHI_TAKER_RATE",
    "KALSHI_MAKER_RATE",
    "kalshi_fee",
    "EXECUTION_SCENARIOS",
    "scenario_returns",
    "fee_scenario_table",
    "breakeven_friction",
]

KALSHI_TAKER_RATE = 0.07     # fee schedule effective 2026-07-07
KALSHI_MAKER_RATE = 0.0175


def kalshi_fee(price, rate: float = KALSHI_TAKER_RATE, multiplier: float = 1.0):
    """Kalshi trading fee per contract at `price` dollars (exact formula,
    no per-order rounding). Charged on execution; settlement is free."""
    price = np.asarray(price, dtype=float)
    return rate * multiplier * price * (1.0 - price)


# name -> (label, price_fn(sides_row_df) -> price array, fee rate)
# price functions receive the bets DataFrame and return the contract price.
EXECUTION_SCENARIOS: dict[str, dict] = {
    "book_close": dict(
        label="Sportsbook, closing line (baseline)",
        price=lambda d: d["p_raw"].to_numpy(),
        fee_rate=0.0,
    ),
    "kalshi_mid_taker": dict(
        label="Kalshi taker at de-vigged mid",
        price=lambda d: d["p_basic"].to_numpy(),
        fee_rate=KALSHI_TAKER_RATE,
    ),
    "kalshi_spread1_taker": dict(
        label="Kalshi taker, mid + 1c half-spread",
        price=lambda d: np.minimum(d["p_basic"].to_numpy() + 0.01, 0.99),
        fee_rate=KALSHI_TAKER_RATE,
    ),
    "kalshi_spread2_taker": dict(
        label="Kalshi taker, mid + 2c half-spread",
        price=lambda d: np.minimum(d["p_basic"].to_numpy() + 0.02, 0.99),
        fee_rate=KALSHI_TAKER_RATE,
    ),
    "kalshi_mid_maker": dict(
        label="Kalshi maker at de-vigged mid",
        price=lambda d: d["p_basic"].to_numpy(),
        fee_rate=KALSHI_MAKER_RATE,
    ),
    "kalshi_raw_taker": dict(
        label="Kalshi taker at book price (worst case)",
        price=lambda d: d["p_raw"].to_numpy(),
        fee_rate=KALSHI_TAKER_RATE,
    ),
}


def scenario_returns(bets: pd.DataFrame, scenario: str) -> np.ndarray:
    """Per-dollar returns of staking $1 per bet under an execution scenario.

    Buy N = 1/(pi + fee) contracts at price pi; collect N if the side wins.
    """
    spec = EXECUTION_SCENARIOS[scenario]
    price = np.asarray(spec["price"](bets), dtype=float)
    fee = kalshi_fee(price, rate=spec["fee_rate"]) if spec["fee_rate"] else 0.0
    cost = price + fee
    won = bets["won"].to_numpy().astype(float)
    return won / cost - 1.0


def fee_scenario_table(
    bets: pd.DataFrame,
    scenarios=tuple(EXECUTION_SCENARIOS),
    n_boot: int = 4000,
    seed: int = 17,
) -> pd.DataFrame:
    """ROI (mean per-dollar return) of the given bets under each execution
    scenario, with fight-cluster bootstrap 95% CI and one-sided p(ROI <= 0)."""
    rows = []
    for name in scenarios:
        spec = EXECUTION_SCENARIOS[name]
        ret = scenario_returns(bets, name)
        lo, hi, boot = _cluster_bootstrap_mean(
            ret, bets["fight_id"].to_numpy(), n_boot=n_boot, seed=seed
        )
        roi = float(ret.mean())
        shifted = boot - boot.mean()  # bootstrap distribution under H0: ROI = 0
        price = np.asarray(spec["price"](bets), dtype=float)
        fee = kalshi_fee(price, rate=spec["fee_rate"]) if spec["fee_rate"] else np.zeros_like(price)
        rows.append(
            {
                "scenario": name,
                "label": spec["label"],
                "n_bets": int(len(bets)),
                "mean_price": float(price.mean()),
                "mean_fee_per_contract": float(np.mean(fee)),
                "mean_fee_pct_of_stake": float(np.mean(fee / (price + fee))),
                "roi": roi,
                "ci_lo": lo,
                "ci_hi": hi,
                "p_roi_gt0": float((shifted >= roi).mean()),
                "total_profit_per_unit": float(ret.sum()),
            }
        )
    return pd.DataFrame(rows)


def breakeven_friction(bets: pd.DataFrame) -> dict:
    """How much total friction (spread + fee, as a fraction of the de-vigged
    mid) would exactly erase the edge?

    With win rate q and cost c per contract (price + fee), ROI = q/c - 1 = 0
    at c = q. The de-vigged mid is p_basic, so the tolerable friction is
    q_hat / mean(p_basic) - 1 per dollar of mid price. Reported alongside the
    two components it must cover at the mean price: the Kalshi taker fee and
    any half-spread.
    """
    q_hat = float(bets["won"].mean())
    mid = float(bets["p_basic"].mean())
    raw = float(bets["p_raw"].mean())
    fee_at_mid = float(kalshi_fee(bets["p_basic"].to_numpy()).mean())
    return {
        "n_bets": int(len(bets)),
        "win_rate": q_hat,
        "mean_mid": mid,
        "mean_raw": raw,
        "edge_vs_mid": q_hat - mid,
        "edge_vs_raw": q_hat - raw,
        "breakeven_total_friction_pct_of_mid": q_hat / mid - 1.0,
        "kalshi_taker_fee_at_mid": fee_at_mid,
        "kalshi_taker_fee_at_mid_pct": fee_at_mid / mid,
        "residual_for_spread_cents": (q_hat - mid - fee_at_mid) * 100.0,
    }
