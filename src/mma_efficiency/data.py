"""Download and prepare the fight-level dataset.

Primary source: the "Ultimate UFC Dataset" (Kaggle: mdabbert/ultimate-ufc-dataset,
mirrored on GitHub at shortlikeafox/ultimate_ufc_dataset, Apache-2.0), which
combines UFC fight results with closing moneyline odds from 2010 onward.

The pipeline is promotion-agnostic: any MMA fight table with the columns
produced by :func:`build_fight_table` can be appended and analyzed identically.
Free, clean historical odds simply do not exist in public form for
Bellator / PFL / ONE, so the empirical sample here is UFC.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .odds import (
    american_to_decimal,
    american_to_prob,
    devig_proportional,
    devig_shin,
)

DATA_URL = (
    "https://raw.githubusercontent.com/shortlikeafox/"
    "ultimate_ufc_dataset/main/ufc-master.csv"
)

def download_raw(dest: str | Path = "data/ufc-master.csv", url: str = DATA_URL) -> Path:
    """Download the raw CSV (skipped if it already exists). Returns the path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    # Validate before writing: must parse as CSV with the expected columns.
    probe = pd.read_csv(io.BytesIO(payload), nrows=5)
    required = {"R_fighter", "B_fighter", "R_odds", "B_odds", "date", "Winner"}
    missing = required - set(probe.columns)
    if missing:
        raise ValueError(f"Downloaded file is missing expected columns: {missing}")
    dest.write_bytes(payload)
    return dest


def load_raw(path: str | Path = "data/ufc-master.csv") -> pd.DataFrame:
    """Load the raw CSV, downloading it first if necessary."""
    path = Path(path)
    if not path.exists():
        download_raw(path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_fight_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per fight with market probabilities and outcome.

    Keeps fights that have moneyline odds for both corners and a decisive
    winner (draws and no-contests are dropped for calibration work; they are
    fewer than 0.15% of bouts and are treated as pushes in the backtest
    module, where the distinction matters).
    """
    df = raw.copy()
    df = df.dropna(subset=["R_odds", "B_odds"])
    df = df[np.isfinite(df["R_odds"]) & np.isfinite(df["B_odds"])]

    out = pd.DataFrame(
        {
            "fight_id": np.arange(len(df)),
            "raw_index": df.index.to_numpy(),
            "date": df["date"].to_numpy(),
            "red": df["R_fighter"].to_numpy(),
            "blue": df["B_fighter"].to_numpy(),
            "weight_class": df["weight_class"].to_numpy(),
            "gender": df["gender"].to_numpy(),
            "title_bout": df["title_bout"].astype(bool).to_numpy(),
            "winner": df["Winner"].to_numpy(),
            "r_odds": df["R_odds"].astype(float).to_numpy(),
            "b_odds": df["B_odds"].astype(float).to_numpy(),
        }
    )

    out["decisive"] = out["winner"].isin(["Red", "Blue"])

    # Decimal odds and raw implied probabilities
    out["r_dec"] = american_to_decimal(out["r_odds"])
    out["b_dec"] = american_to_decimal(out["b_odds"])
    out["r_p_raw"] = american_to_prob(out["r_odds"])
    out["b_p_raw"] = american_to_prob(out["b_odds"])
    out["overround"] = out["r_p_raw"] + out["b_p_raw"]

    # Keep only two-sided books with a plausible margin (guards against data
    # entry errors such as both sides heavy favorites).
    ok = (out["overround"] > 1.0) & (out["overround"] < 1.25)
    out = out[ok].copy()

    # De-vigged probabilities
    out["r_p_basic"], out["b_p_basic"] = devig_proportional(
        out["r_p_raw"], out["b_p_raw"]
    )
    r_shin, b_shin, z = devig_shin(out["r_p_raw"], out["b_p_raw"])
    out["r_p_shin"] = r_shin
    out["b_p_shin"] = b_shin
    out["shin_z"] = z

    out["year"] = pd.to_datetime(out["date"]).dt.year
    return out.reset_index(drop=True)


def side_table(fights: pd.DataFrame, decisive_only: bool = True) -> pd.DataFrame:
    """Long format: one row per *side of a fight* (the natural unit for a
    bettor — every price offered is one potential bet).

    Columns: fight_id, date, side ('red'/'blue'), american odds, decimal odds,
    raw/de-vigged probabilities, won (0/1), is_favorite.
    Two rows of the same fight are mirror images, so all inference on this
    table clusters standard errors / bootstraps by fight_id.
    """
    f = fights[fights["decisive"]] if decisive_only else fights

    rows = []
    for side, opp in (("red", "blue"), ("blue", "red")):
        s = pd.DataFrame(
            {
                "fight_id": f["fight_id"],
                "date": f["date"],
                "year": f["year"],
                "weight_class": f["weight_class"],
                "gender": f["gender"],
                "title_bout": f["title_bout"],
                "side": side,
                "fighter": f[side],
                "odds": f[f"{side[0]}_odds"],
                "dec": f[f"{side[0]}_dec"],
                "p_raw": f[f"{side[0]}_p_raw"],
                "p_basic": f[f"{side[0]}_p_basic"],
                "p_shin": f[f"{side[0]}_p_shin"],
                "overround": f["overround"],
                "won": (f["winner"] == side.capitalize()).astype(int),
            }
        )
        rows.append(s)
    long = pd.concat(rows, ignore_index=True)
    # Favorite = the side with the shorter price *within the fight* (raw probs
    # of both sides can exceed 0.5 once the vig is included, so comparing to
    # 0.5 would misclassify near-pick'em fights). Exact pick'ems (~2% of
    # fights) have no favorite: both sides get False.
    opp_p_raw = np.concatenate(
        [f["b_p_raw"].to_numpy(), f["r_p_raw"].to_numpy()]  # red rows, blue rows
    )
    long["is_favorite"] = long["p_raw"].to_numpy() > opp_p_raw
    long["is_underdog"] = long["p_raw"].to_numpy() < opp_p_raw
    # flat 1-unit bet return: dec - 1 on a win, -1 on a loss
    long["ret"] = np.where(long["won"] == 1, long["dec"] - 1.0, -1.0)
    return long.sort_values(["date", "fight_id"]).reset_index(drop=True)


def load_fights(path: str | Path = "data/ufc-master.csv") -> pd.DataFrame:
    """Convenience: raw csv -> fight table."""
    return build_fight_table(load_raw(path))
