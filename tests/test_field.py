import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from mma_efficiency.field import (
    HEAVY_FAVORITE_CUTOFF,
    load_quotes,
    quote_gap_table,
    summarize_gaps,
)
from mma_efficiency.odds import american_to_prob

# Repo root, so the real-data test below works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


def make_quotes(gap_cents: float = 2.0, spread_cents: float = 1.0) -> pd.DataFrame:
    """Synthetic snapshot built backwards from a known gap.

    Start from a book consensus (-400/+320 and -150/+130), de-vig it, then
    place the Kalshi favorite mid exactly `gap_cents` above the de-vigged
    probability with a `spread_cents` bid-ask spread. quote_gap_table must
    recover the gap we planted.
    """
    rows = []
    for fav_odds, dog_odds in ((-400.0, 320.0), (-150.0, 130.0)):
        fav_raw = american_to_prob(fav_odds)
        dog_raw = american_to_prob(dog_odds)
        fav_devig = fav_raw / (fav_raw + dog_raw)
        mid = fav_devig + gap_cents / 100.0
        half = spread_cents / 200.0
        rows.append(
            {
                "snapshot_date": "2026-01-01",
                "card_date": "2026-01-04",
                "event": "synthetic",
                "favorite": "A",
                "underdog": "B",
                "kalshi_fav_bid": mid - half,
                "kalshi_fav_ask": mid + half,
                "kalshi_dog_bid": 1 - mid - half,
                "kalshi_dog_ask": 1 - mid + half,
                # best == worst so the consensus is exact, not a range
                "book_fav_best": fav_odds,
                "book_fav_worst": fav_odds,
                "book_dog_best": dog_odds,
                "book_dog_worst": dog_odds,
            }
        )
    df = pd.DataFrame(rows)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["card_date"] = pd.to_datetime(df["card_date"])
    return df


def test_gap_recovers_planted_values():
    gaps = quote_gap_table(make_quotes(gap_cents=2.0, spread_cents=1.0))
    # mid sits 2 cents above de-vigged consensus; ask adds the half-spread
    assert np.allclose(gaps["gap_mid_cents"], 2.0, atol=1e-9)
    assert np.allclose(gaps["gap_ask_cents"], 2.5, atol=1e-9)
    assert np.allclose(gaps["kalshi_spread_cents"], 1.0, atol=1e-9)


def test_heavy_favorite_flag_uses_consensus_cutoff():
    gaps = quote_gap_table(make_quotes())
    # -400 qualifies at the -300 cutoff, -150 does not
    flags = gaps.sort_values("consensus_fav_raw")["is_heavy_favorite"].tolist()
    assert flags == [False, True]
    assert HEAVY_FAVORITE_CUTOFF == -300.0


def test_consensus_midpoint_in_probability_space():
    """A best/worst range must average in probability space, where the
    midpoint of -500/-700 is well-defined even though American odds are not
    linear in probability."""
    q = make_quotes()
    q.loc[0, "book_fav_best"] = -500.0
    q.loc[0, "book_fav_worst"] = -700.0
    gaps = quote_gap_table(q)
    expected = 0.5 * (american_to_prob(-500.0) + american_to_prob(-700.0))
    assert gaps.loc[0, "consensus_fav_raw"] == pytest.approx(expected)


def test_summary_counts_and_means():
    gaps = quote_gap_table(make_quotes(gap_cents=1.5))
    s = summarize_gaps(gaps)
    assert s["n_fights"] == 2 and s["n_cards"] == 1
    assert s["n_heavy_favorites"] == 1
    assert s["mean_gap_mid_cents_all"] == pytest.approx(1.5, abs=1e-9)
    # with best == worst the consensus booksum is the true overround
    assert s["mean_booksum"] > 1.0


def test_load_quotes_validates_real_file():
    """The committed snapshot file loads, passes validation, and contains
    both pilot cards with at least a few strategy-relevant favorites."""
    quotes = load_quotes(REPO_ROOT / "data" / "live_quotes.csv")
    gaps = quote_gap_table(quotes)
    s = summarize_gaps(gaps)
    assert s["n_cards"] >= 2
    assert s["n_heavy_favorites"] >= 4
    # pilot reading: exchange asks sit above the de-vigged consensus, but by
    # cents, not by the full vig share
    assert 0.0 < s["mean_gap_ask_cents_heavy"] < 4.0
