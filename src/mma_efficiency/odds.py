"""Odds conversions and vig (overround) removal.

All functions are vectorized over numpy arrays / pandas Series.

Conventions
-----------
* American odds: negative for favorites (e.g. -300 = risk 300 to win 100),
  positive for underdogs (e.g. +250 = risk 100 to win 250).
* Decimal odds: total payout per 1 unit staked (stake included), so a
  winning 1-unit bet at decimal odds d returns d (profit d - 1).
* "Raw" implied probability: the break-even probability embedded in the
  price, 1 / decimal_odds. Raw probabilities of the two sides of a fight
  sum to more than 1; the excess is the bookmaker margin (vig/overround).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "american_to_prob",
    "prob_to_american",
    "devig_proportional",
    "devig_power",
    "devig_shin",
    "overround",
]


def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dec = np.where(odds > 0, 1.0 + odds / 100.0, 1.0 + 100.0 / np.abs(odds))
    return dec


def decimal_to_american(dec):
    """Convert decimal odds to American odds."""
    dec = np.asarray(dec, dtype=float)
    return np.where(dec >= 2.0, (dec - 1.0) * 100.0, -100.0 / (dec - 1.0))


def american_to_prob(odds):
    """Raw implied (break-even) probability from American odds."""
    return 1.0 / american_to_decimal(odds)


def prob_to_american(p):
    """American odds at which probability p is the break-even probability."""
    p = np.asarray(p, dtype=float)
    return np.where(p >= 0.5, -100.0 * p / (1.0 - p), 100.0 * (1.0 - p) / p)


def overround(p_a, p_b):
    """Bookmaker overround r = p_a + p_b (raw implied probs). r - 1 is the vig."""
    return np.asarray(p_a, dtype=float) + np.asarray(p_b, dtype=float)


def devig_proportional(p_a, p_b):
    """Remove the vig by proportional scaling (a.k.a. 'basic'/'multiplicative').

    pi_i = p_i / (p_a + p_b).  This is the standard baseline in the market
    efficiency literature. Under an efficient market with proportional vig,
    a flat bet on either side has expected return 1/r - 1 (identical for
    favorites and underdogs), which is what makes return-by-odds-band tests
    interpretable.
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    r = p_a + p_b
    return p_a / r, p_b / r


def devig_power(p_a, p_b, tol=1e-12, max_iter=200):
    """Remove the vig with the 'power' method: pi_i = p_i**k with k chosen
    so the de-vigged probabilities sum to 1.

    Solved per-pair by bisection on k in [0.25, 4]. Vectorized.
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)

    lo = np.full_like(p_a, 0.25)
    hi = np.full_like(p_a, 4.0)

    def s(k):
        return p_a**k + p_b**k - 1.0

    # s is decreasing in k; ensure bracket
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = s(mid)
        hi = np.where(val < 0, mid, hi)
        lo = np.where(val >= 0, mid, lo)
        if np.nanmax(np.abs(hi - lo)) < tol:
            break
    k = 0.5 * (lo + hi)
    return p_a**k, p_b**k


def _shin_probs_given_z(p, r, z):
    """Shin (1993) implied true probability for one side, given insider
    fraction z, raw prob p and overround r (two-outcome market)."""
    return (np.sqrt(z**2 + 4.0 * (1.0 - z) * (p**2) / r) - z) / (2.0 * (1.0 - z))


def devig_shin(p_a, p_b, tol=1e-12, max_iter=200):
    """Remove the vig with Shin's (1993) method for a two-outcome market.

    Shin's model assumes part of the betting volume comes from insiders who
    know the outcome; the bookmaker shades longshot prices most in response.
    Relative to proportional scaling, Shin probabilities shift mass toward
    the favorite, so it is the natural robustness check for
    favorite-longshot bias results.

    The insider fraction z >= 0 is solved per fight by bisection so that the
    two Shin probabilities sum to 1 (at z = 0 the formula reduces to
    normalizing each raw probability by the square root of the booksum).
    See Shin (1993) and Strumbelj (2014) for the closed form used here.

    Returns (pi_a, pi_b, z).
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    r = p_a + p_b

    lo = np.zeros_like(p_a)
    hi = np.full_like(p_a, 0.5)

    def s(z):
        return (
            _shin_probs_given_z(p_a, r, z)
            + _shin_probs_given_z(p_b, r, z)
            - 1.0
        )

    # s(z) is decreasing in z: s(0) >= 0 (probs normalized by r of squares
    # overshoot), and s(0.5) < 0 for realistic prices.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = s(mid)
        lo = np.where(val >= 0, mid, lo)
        hi = np.where(val < 0, mid, hi)
        if np.nanmax(np.abs(hi - lo)) < tol:
            break
    z = 0.5 * (lo + hi)
    return _shin_probs_given_z(p_a, r, z), _shin_probs_given_z(p_b, r, z), z
