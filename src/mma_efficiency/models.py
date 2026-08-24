"""ML benchmark: can public pre-fight information beat the market?

This is a *semi-strong efficiency* test, not a fight-prediction exercise.
If closing odds already impound all public information, then models built on
public fighter statistics should not out-predict the de-vigged market
probability out of sample. We test that directly:

* one observation per fight, oriented favorite-vs-underdog (so the arbitrary
  red/blue corner assignment cannot leak information);
* features are favorite-minus-underdog differentials of career stats known
  before the bout, computed from the per-corner columns (the dataset's own
  ``*_dif`` columns mix R-B and B-R orientation, so we do not use them);
* chronological train/test split -- no shuffling, no leakage from the future;
* the yardstick is out-of-sample log loss / Brier score against the market's
  own de-vigged probability, with a paired bootstrap over test fights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "build_model_table",
    "run_benchmark",
    "edge_backtest",
]

# (feature name, red column, blue column) -- all career-to-date, pre-fight.
FEATURE_SPECS = [
    ("age", "R_age", "B_age"),
    ("height_cm", "R_Height_cms", "B_Height_cms"),
    ("reach_cm", "R_Reach_cms", "B_Reach_cms"),
    ("wins", "R_wins", "B_wins"),
    ("losses", "R_losses", "B_losses"),
    ("win_streak", "R_current_win_streak", "B_current_win_streak"),
    ("lose_streak", "R_current_lose_streak", "B_current_lose_streak"),
    ("longest_win_streak", "R_longest_win_streak", "B_longest_win_streak"),
    ("rounds_fought", "R_total_rounds_fought", "B_total_rounds_fought"),
    ("title_bouts", "R_total_title_bouts", "B_total_title_bouts"),
    ("ko_wins", "R_win_by_KO/TKO", "B_win_by_KO/TKO"),
    ("sub_wins", "R_win_by_Submission", "B_win_by_Submission"),
    ("sig_str_landed", "R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed"),
    ("sig_str_pct", "R_avg_SIG_STR_pct", "B_avg_SIG_STR_pct"),
    ("td_landed", "R_avg_TD_landed", "B_avg_TD_landed"),
    ("td_pct", "R_avg_TD_pct", "B_avg_TD_pct"),
    ("sub_att", "R_avg_SUB_ATT", "B_avg_SUB_ATT"),
]


def build_model_table(raw: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """One row per decisive fight, oriented favorite (by raw price) vs underdog.

    Columns: date, y (1 = favorite won), p_market (favorite's de-vigged basic
    probability), logit_p_market, and ``d_<feature>`` = favorite minus
    underdog for each career stat.
    """
    f = fights[fights["decisive"]].copy()
    stats = raw.loc[f["raw_index"]]

    red_is_fav = (f["r_p_raw"].to_numpy() >= f["b_p_raw"].to_numpy())

    out = pd.DataFrame(
        {
            "fight_id": f["fight_id"].to_numpy(),
            "date": f["date"].to_numpy(),
            "year": f["year"].to_numpy(),
            "title_bout": f["title_bout"].astype(int).to_numpy(),
            "y": np.where(
                red_is_fav, f["winner"].to_numpy() == "Red", f["winner"].to_numpy() == "Blue"
            ).astype(int),
            "p_market": np.where(
                red_is_fav, f["r_p_basic"].to_numpy(), f["b_p_basic"].to_numpy()
            ),
            "fav_odds": np.where(red_is_fav, f["r_odds"].to_numpy(), f["b_odds"].to_numpy()),
            "dog_odds": np.where(red_is_fav, f["b_odds"].to_numpy(), f["r_odds"].to_numpy()),
            "fav_dec": np.where(red_is_fav, f["r_dec"].to_numpy(), f["b_dec"].to_numpy()),
            "dog_dec": np.where(red_is_fav, f["b_dec"].to_numpy(), f["r_dec"].to_numpy()),
            "fav_p_raw": np.where(red_is_fav, f["r_p_raw"].to_numpy(), f["b_p_raw"].to_numpy()),
            "dog_p_raw": np.where(red_is_fav, f["b_p_raw"].to_numpy(), f["r_p_raw"].to_numpy()),
        }
    )
    out["logit_p_market"] = logit(out["p_market"])

    for name, r_col, b_col in FEATURE_SPECS:
        r = pd.to_numeric(stats[r_col], errors="coerce").to_numpy()
        b = pd.to_numeric(stats[b_col], errors="coerce").to_numpy()
        out[f"d_{name}"] = np.where(red_is_fav, r - b, b - r)

    return out


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def _paired_bootstrap_logloss(
    y: np.ndarray, p_model: np.ndarray, p_market: np.ndarray,
    n_boot: int = 4000, seed: int = 3,
) -> dict:
    """Paired bootstrap over test fights of (model log loss - market log loss).
    Negative diff = model beats market."""
    p_model = np.clip(p_model, 1e-6, 1 - 1e-6)
    p_market = np.clip(p_market, 1e-6, 1 - 1e-6)
    li_model = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    li_market = -(y * np.log(p_market) + (1 - y) * np.log(1 - p_market))
    d = li_model - li_market
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "logloss_diff": float(d.mean()),
        "diff_ci_lo": float(lo),
        "diff_ci_hi": float(hi),
        "p_model_better": float((boot >= 0).mean()),  # one-sided
    }


def run_benchmark(
    model_table: pd.DataFrame, split_date: str = "2022-01-01", seed: int = 3
) -> tuple[pd.DataFrame, dict]:
    """Fit the model suite on fights before `split_date`, evaluate after.

    Returns (metrics table, dict of test-set probability vectors by model).
    """
    feat_cols = [c for c in model_table.columns if c.startswith("d_")] + ["title_bout"]
    split = pd.Timestamp(split_date)
    train = model_table[model_table["date"] < split]
    test = model_table[model_table["date"] >= split]
    y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()

    def lr_pipeline():
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=2000)),
            ]
        )

    def hgb():
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.06,
            max_iter=400,
            l2_regularization=1.0,
            random_state=seed,
        )

    X_stats_tr, X_stats_te = train[feat_cols], test[feat_cols]
    X_both_tr = train[feat_cols + ["logit_p_market"]]
    X_both_te = test[feat_cols + ["logit_p_market"]]

    preds: dict[str, np.ndarray] = {}
    preds["Market (de-vigged odds)"] = test["p_market"].to_numpy()

    m = lr_pipeline().fit(X_stats_tr, y_tr)
    preds["Stats only (logistic)"] = m.predict_proba(X_stats_te)[:, 1]

    m = lr_pipeline().fit(X_both_tr, y_tr)
    preds["Stats + market (logistic)"] = m.predict_proba(X_both_te)[:, 1]

    m = hgb().fit(X_both_tr, y_tr)
    preds["Stats + market (grad. boosting)"] = m.predict_proba(X_both_te)[:, 1]

    rows = []
    market_p = preds["Market (de-vigged odds)"]
    for name, p in preds.items():
        row = {"model": name, "n_test": int(len(test)), **_metrics(y_te, p)}
        if name != "Market (de-vigged odds)":
            row.update(_paired_bootstrap_logloss(y_te, p, market_p, seed=seed))
        rows.append(row)
    return pd.DataFrame(rows), {"preds": preds, "test": test.reset_index(drop=True)}


def edge_backtest(
    test: pd.DataFrame, p_model_fav: np.ndarray, thresholds=(0.0, 0.02, 0.05, 0.10),
    n_boot: int = 4000, seed: int = 5,
) -> pd.DataFrame:
    """Bet 1 unit on whichever side the model gives positive expected value
    against the *raw* (vig included) price, at various minimum-edge thresholds.

    Expected value per unit on side s: p_model_s * dec_s - 1. This is the
    economically meaningful test: to profit you must beat the price with the
    vig, not the de-vigged probability. ROI comes with a bootstrap 95% CI and
    a one-sided p-value against ROI <= 0 (bets are one per fight, so plain
    resampling is valid here).
    """
    rng = np.random.default_rng(seed)
    rows = []
    y = test["y"].to_numpy()
    for thr in thresholds:
        ev_fav = p_model_fav * test["fav_dec"].to_numpy() - 1.0
        ev_dog = (1 - p_model_fav) * test["dog_dec"].to_numpy() - 1.0
        take_fav = (ev_fav > thr) & (ev_fav >= ev_dog)
        take_dog = (ev_dog > thr) & (ev_dog > ev_fav)
        ret = np.zeros(len(test))
        ret[take_fav] = np.where(
            y[take_fav] == 1, test["fav_dec"].to_numpy()[take_fav] - 1.0, -1.0
        )
        ret[take_dog] = np.where(
            y[take_dog] == 0, test["dog_dec"].to_numpy()[take_dog] - 1.0, -1.0
        )
        n_bets = int(take_fav.sum() + take_dog.sum())
        mask = take_fav | take_dog
        picked = ret[mask]
        if n_bets:
            idx = rng.integers(0, n_bets, size=(n_boot, n_bets))
            boot = picked[idx].mean(axis=1)
            ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
            shifted = boot - boot.mean()
            p_gt0 = float((shifted >= picked.mean()).mean())
        else:
            ci_lo = ci_hi = p_gt0 = np.nan
        rows.append(
            {
                "min_edge": thr,
                "n_bets": n_bets,
                "share_underdog_bets": float(take_dog.sum() / max(n_bets, 1)),
                "roi": float(picked.mean()) if n_bets else np.nan,
                "roi_ci_lo": float(ci_lo),
                "roi_ci_hi": float(ci_hi),
                "p_roi_gt0": p_gt0,
                "total_profit": float(picked.sum()),
            }
        )
    return pd.DataFrame(rows)
