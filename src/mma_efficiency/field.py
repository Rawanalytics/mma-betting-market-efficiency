"""Field evidence: hand-collected Kalshi quotes vs. the sportsbook consensus.

The costs module prices exchange execution under *assumed* prices, because no
historical exchange quotes exist for the 2010-2026 sample (Kalshi listed its
first UFC markets in 2025). This module works on the small dataset I am
collecting going forward to test that assumption directly: paired snapshots
of (a) Kalshi's live order book for every fight on an upcoming card and
(b) the multi-book sportsbook lines for the same fights at the same time.

The quantity of interest is the *gap* between what a taker actually pays on
the exchange and the de-vigged sportsbook consensus:

    gap_ask = kalshi_fav_ask - devig(consensus mid of the books)

mapped onto the execution scenarios of ``costs.py``: a gap near 0 is the
"taker at the mid" scenario (historically +4.5%/bet), ~1 cent matches
"mid + 1c" (+3.3%), ~2 cents matches "mid + 2c" (+2.1%), and a gap the size
of the full vig share (~3-4 cents on a heavy favorite) is the worst case
where no edge survives.

Data format (``data/live_quotes.csv``), one row per fight:

    snapshot_date, card_date, event, favorite, underdog,
    kalshi_fav_bid/ask, kalshi_dog_bid/ask,
    book_fav_best/worst, book_dog_best/worst,   (American odds)
    horizon                                     ("midweek" or "close")

Two collection horizons, flagged by the ``horizon`` column:

* ``midweek`` rows are live snapshots taken by hand several days before the
  card: Kalshi's displayed order book vs. the aggregator's current
  best/worst book prices, read at the same time.
* ``close`` rows are reconstructed closing quotes: the Kalshi side is the
  last pre-fight hourly candle (1-2 hours before the recorded market close,
  from the exchange's public candlestick history), and the book side is the
  aggregator's closing range for each fighter. On a binary exchange the two
  sides share one order book (a bid on one side *is* an ask on the other),
  so the underdog columns of ``close`` rows are the exact complement of the
  favorite's book, which avoids pairing quotes read minutes apart.

Honest limitations, stated once here and repeated in the paper:

* The book consensus is approximated from the best/worst price range shown
  by a multi-book aggregator (midpoint in probability space), not from the
  full list of quotes.
* The "closing" exchange quote is the last hourly candle before the fight,
  not the final tick, and the book closing range aggregates books that stop
  updating at slightly different times.
* Four cards across two horizons is a pilot, not a sample. The target is
  20-30 cards per horizon before treating the mean gap as an estimate
  rather than an illustration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .odds import american_to_prob

__all__ = [
    "HEAVY_FAVORITE_CUTOFF",
    "load_quotes",
    "quote_gap_table",
    "summarize_gaps",
]

# Same strategy trigger as the walk-forward rule: consensus price of -300 or
# shorter marks the heavy favorites the historical edge lives on.
HEAVY_FAVORITE_CUTOFF = -300.0


def load_quotes(path: str | Path = "data/live_quotes.csv") -> pd.DataFrame:
    """Load the snapshot file and sanity-check its columns and ranges."""
    df = pd.read_csv(path, parse_dates=["snapshot_date", "card_date"])
    required = {
        "favorite", "underdog",
        "kalshi_fav_bid", "kalshi_fav_ask", "kalshi_dog_bid", "kalshi_dog_ask",
        "book_fav_best", "book_fav_worst", "book_dog_best", "book_dog_worst",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"live_quotes.csv is missing columns: {missing}")
    prices = df[["kalshi_fav_bid", "kalshi_fav_ask", "kalshi_dog_bid", "kalshi_dog_ask"]]
    if ((prices <= 0) | (prices >= 1)).any().any():
        raise ValueError("Kalshi prices must lie strictly inside (0, 1) dollars")
    # Horizon flag: early snapshot files predate the column, so default to
    # the midweek label they were collected under.
    if "horizon" not in df.columns:
        df["horizon"] = "midweek"
    bad = set(df["horizon"].unique()) - {"midweek", "close"}
    if bad:
        raise ValueError(f"unknown horizon labels: {bad}")
    return df


def _consensus_prob(best: pd.Series, worst: pd.Series) -> np.ndarray:
    """Raw implied probability of the consensus book price for one side.

    The aggregator reports each side's best and worst American price across
    books; the consensus is approximated as the midpoint of the two implied
    probabilities. Midpointing in probability space (rather than in American
    odds) keeps the approximation well-behaved when the range crosses the
    +100/-100 discontinuity.
    """
    return 0.5 * (american_to_prob(best) + american_to_prob(worst))


def quote_gap_table(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per-fight comparison of Kalshi prices with the de-vigged consensus.

    Adds, per fight: the consensus raw probabilities and booksum, the
    de-vigged (proportional) favorite probability, the favorite's Kalshi mid
    and spread, and the two headline gaps in cents per contract:

    * ``gap_ask_cents`` - what a taker pays above the de-vigged consensus;
    * ``gap_mid_cents`` - where the exchange mid sits relative to it.

    ``is_heavy_favorite`` flags fights whose consensus favorite price is
    -300 or shorter, the subset the walk-forward strategy would trade.
    """
    q = quotes.copy()

    fav_raw = _consensus_prob(q["book_fav_best"], q["book_fav_worst"])
    dog_raw = _consensus_prob(q["book_dog_best"], q["book_dog_worst"])
    booksum = fav_raw + dog_raw
    fav_devig = fav_raw / booksum  # proportional de-vig, as everywhere else

    kalshi_mid = 0.5 * (q["kalshi_fav_bid"] + q["kalshi_fav_ask"])

    q["consensus_fav_raw"] = fav_raw
    q["consensus_booksum"] = booksum
    q["consensus_fav_devig"] = fav_devig
    q["kalshi_fav_mid"] = kalshi_mid
    q["kalshi_spread_cents"] = 100 * (q["kalshi_fav_ask"] - q["kalshi_fav_bid"])
    q["gap_ask_cents"] = 100 * (q["kalshi_fav_ask"] - fav_devig)
    q["gap_mid_cents"] = 100 * (kalshi_mid - fav_devig)
    # Consensus American price of the favorite, for the strategy flag. The
    # probability-space midpoint is converted back through the raw implied
    # probability rather than averaging American odds directly.
    q["is_heavy_favorite"] = fav_raw >= american_to_prob(HEAVY_FAVORITE_CUTOFF)
    return q


def summarize_gaps(gaps: pd.DataFrame) -> dict:
    """Headline numbers for the paper: mean gaps overall and for the
    heavy-favorite subset the strategy would actually trade.

    When the table carries a ``horizon`` column the same statistics are also
    reported per horizon under ``by_horizon``, since the economically
    relevant comparison (what does a taker pay *at close*, when the
    historical edge is defined?) should not be diluted by midweek rows.
    """
    heavy = gaps[gaps["is_heavy_favorite"]]
    out = {
        "n_fights": int(len(gaps)),
        "n_cards": int(gaps["card_date"].nunique()),
        "n_heavy_favorites": int(len(heavy)),
        "mean_gap_ask_cents_all": float(gaps["gap_ask_cents"].mean()),
        "mean_gap_mid_cents_all": float(gaps["gap_mid_cents"].mean()),
        "mean_gap_ask_cents_heavy": float(heavy["gap_ask_cents"].mean()) if len(heavy) else float("nan"),
        "mean_gap_mid_cents_heavy": float(heavy["gap_mid_cents"].mean()) if len(heavy) else float("nan"),
        "mean_spread_cents": float(gaps["kalshi_spread_cents"].mean()),
        "mean_booksum": float(gaps["consensus_booksum"].mean()),
    }
    if "horizon" in gaps.columns and gaps["horizon"].nunique() > 1:
        out["by_horizon"] = {
            h: summarize_gaps(sub.drop(columns=["horizon"]))
            for h, sub in gaps.groupby("horizon", sort=True)
        }
    return out
