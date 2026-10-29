#!/usr/bin/env python3
"""Run the full MMA market-efficiency analysis end to end.

Usage:  python run_analysis.py [--data data/ufc-master.csv]

Downloads the dataset if needed, runs every test, writes tables to results/,
figures to figures/, and prints a summary. Every stage is deterministic
(fixed bootstrap seeds), so a clean checkout regenerates the committed
results byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mma_efficiency import calibration, costs, data, exploit, field, models, plots, returns  # noqa: E402

# Era boundaries for the stability check: three spans of similar size that
# track the sport's growth from niche to mainstream.
ERAS = [
    ("2010–2015", "2010-01-01", "2016-01-01"),
    ("2016–2020", "2016-01-01", "2021-01-01"),
    ("2021–2026", "2021-01-01", "2027-01-01"),
]


def main() -> None:
    """Execute all nine stages: data prep, calibration tests, returns by
    band, strategy backtests, the walk-forward exploitability test, Kalshi
    fee scenarios, the live-quote field check, the ML benchmark, and the
    publication figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "ufc-master.csv"))
    ap.add_argument("--split-date", default="2022-01-01",
                    help="train/test split for the ML benchmark")
    args = ap.parse_args()

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    print("[1/9] Loading data ...")
    raw = data.load_raw(args.data)
    fights = data.build_fight_table(raw)
    sides = data.side_table(fights)
    print(f"      {len(fights):,} fights with two-sided odds "
          f"({fights['date'].min():%Y-%m-%d} to {fights['date'].max():%Y-%m-%d}), "
          f"{int(fights['decisive'].sum()):,} decisive, "
          f"mean overround {fights['overround'].mean():.4f}")

    summary: dict = {
        "n_fights": int(len(fights)),
        "n_decisive": int(fights["decisive"].sum()),
        "date_min": str(fights["date"].min().date()),
        "date_max": str(fights["date"].max().date()),
        "mean_overround": float(fights["overround"].mean()),
        "mean_shin_z": float(fights["shin_z"].mean()),
    }

    # --------------------------------------------------------- calibration
    print("[2/9] Calibration tests ...")
    cal_tab = calibration.bucket_binomial_tests(sides, "p_basic")
    cal_tab.to_csv(results_dir / "calibration_buckets.csv", index=False)
    # Robustness view: quantile bins with ~500 bets each, grouped by nearest
    # market price, so every point on the curve has equal statistical power.
    n_eq = max(10, len(sides) // 500)
    cal_eq = calibration.bucket_binomial_tests(sides, "p_basic", equal_count=n_eq)
    cal_eq.to_csv(results_dir / "calibration_buckets_equal_count.csv", index=False)

    logit_tests = {
        pc: calibration.logistic_calibration_test(sides, pc)
        for pc in ("p_raw", "p_basic", "p_shin")
    }
    summary["logistic_calibration"] = logit_tests
    t = logit_tests["p_basic"]
    print(f"      de-vigged slope = {t['slope']:.3f} (SE {t['slope_se']:.3f}), "
          f"H0 slope=1 p = {t['slope_p_vs_1']:.4f}")

    era_rows = [dict(label=f"All fights ({summary['date_min'][:4]}–{summary['date_max'][:4]})",
                     **logit_tests["p_basic"])]
    for label, start, end in ERAS:
        mask = (sides["date"] >= pd.Timestamp(start)) & (sides["date"] < pd.Timestamp(end))
        era_rows.append(dict(label=label,
                             **calibration.logistic_calibration_test(sides[mask], "p_basic")))
    summary["era_slopes"] = [
        {k: r[k] for k in ("label", "slope", "slope_se", "slope_p_vs_1", "n_fights")}
        for r in era_rows
    ]

    # -------------------------------------------------------------- returns
    print("[3/9] Returns by odds band ...")
    band_tab = returns.returns_by_band(sides)
    band_tab.to_csv(results_dir / "returns_by_band.csv", index=False)
    trend = returns.returns_trend_test(sides)
    summary["returns_trend"] = trend
    print(f"      return-vs-price slope = {trend['slope']:.3f} "
          f"(t = {trend['t_stat']:.2f}, p = {trend['p_value']:.2g})")

    # ------------------------------------------------------------- backtest
    print("[4/9] Strategy backtests ...")
    bt = returns.backtest_rules(sides)
    bt.to_csv(results_dir / "backtest_strategies.csv", index=False)
    pnl_rules = {
        "All favorites": sides["is_favorite"],
        "All underdogs": sides["is_underdog"],
        "Heavy favorites (-300 or shorter)": sides["odds"] <= -300,
        "Heavy underdogs (+300 or longer)": sides["odds"] >= 300,
    }
    pnl = returns.cumulative_pnl(sides, pnl_rules)

    # ------------------------------------------------------- exploitability
    print("[5/9] Exploitability: walk-forward test of a pre-registered rule ...")
    sweep_fav = exploit.threshold_sweep(sides)
    sweep_dog = exploit.threshold_sweep(sides, cutoffs=exploit.UNDERDOG_CUTOFFS)
    pd.concat([sweep_fav, sweep_dog]).to_csv(results_dir / "threshold_sweep.csv", index=False)
    wf = exploit.walk_forward_rule(sides)
    summary["walk_forward"] = wf
    print(f"      rule '{wf['chosen_rule']}' picked on pre-{wf['train_end'][:4]} data; "
          f"out-of-sample ROI {wf['test_roi']:+.2%} on {wf['test_n_bets']:,} bets "
          f"(95% CI [{wf['test_roi_ci'][0]:+.2%}, {wf['test_roi_ci'][1]:+.2%}], "
          f"p(ROI>0) one-sided = {wf['test_p_roi_gt0']:.3f})")

    # ------------------------------------------------ prediction-market costs
    print("[6/9] Prediction-market execution: Kalshi fee scenarios ...")
    split = pd.Timestamp(wf["train_end"])
    wf_bets = sides[(sides["date"] >= split) & (sides["odds"] <= wf["chosen_cutoff"])]
    fee_tab = costs.fee_scenario_table(wf_bets, seed=17)
    fee_tab.insert(0, "sample", "walk_forward_oos")
    full_bets = sides[sides["odds"] <= wf["chosen_cutoff"]]
    fee_tab_full = costs.fee_scenario_table(full_bets, seed=19)
    fee_tab_full.insert(0, "sample", "full_sample")
    pd.concat([fee_tab, fee_tab_full]).to_csv(
        results_dir / "kalshi_fee_scenarios.csv", index=False
    )
    summary["kalshi_fees"] = {
        "fee_schedule": {
            "source": "kalshi.com/docs/kalshi-fee-schedule.pdf (effective 2026-07-07)",
            "taker": "0.07 * C * P * (1-P), rounded up; charged on execution only",
            "maker": "0.0175 * C * P * (1-P); settlement, membership, ACH: free",
        },
        "walk_forward_oos": fee_tab.drop(columns="sample").to_dict("records"),
        "breakeven": costs.breakeven_friction(wf_bets),
    }
    k = fee_tab.set_index("scenario")
    print(f"      OOS rule ROI: book {k.loc['book_close','roi']:+.2%} | "
          f"Kalshi mid taker {k.loc['kalshi_mid_taker','roi']:+.2%} "
          f"(p = {k.loc['kalshi_mid_taker','p_roi_gt0']:.3f}) | "
          f"mid+1c {k.loc['kalshi_spread1_taker','roi']:+.2%} "
          f"(p = {k.loc['kalshi_spread1_taker','p_roi_gt0']:.3f}) | "
          f"mid+2c {k.loc['kalshi_spread2_taker','roi']:+.2%} "
          f"(p = {k.loc['kalshi_spread2_taker','p_roi_gt0']:.3f})")

    # ----------------------------------------------------- field evidence
    # The fee scenarios above rest on assumptions about the prices the
    # exchange actually offers. This stage scores those assumptions against
    # the live Kalshi-vs-books snapshots collected in data/live_quotes.csv.
    # It is skipped gracefully if the snapshot file is absent (e.g. a user
    # who deleted it to run the purely historical study).
    quotes_path = ROOT / "data" / "live_quotes.csv"
    if quotes_path.exists():
        print("[7/9] Field evidence: live Kalshi quotes vs de-vigged consensus ...")
        gaps = field.quote_gap_table(field.load_quotes(quotes_path))
        gaps.to_csv(results_dir / "field_quote_gaps.csv", index=False)
        fs = field.summarize_gaps(gaps)
        summary["field_evidence"] = fs
        print(f"      {fs['n_fights']} fights on {fs['n_cards']} cards; "
              f"heavy favorites (n = {fs['n_heavy_favorites']}): "
              f"ask sits {fs['mean_gap_ask_cents_heavy']:+.1f}c above the "
              f"de-vigged consensus (mid {fs['mean_gap_mid_cents_heavy']:+.1f}c); "
              f"mean spread {fs['mean_spread_cents']:.1f}c")
        for h, sub in fs.get("by_horizon", {}).items():
            print(f"      [{h}] {sub['n_fights']} fights / {sub['n_cards']} cards: "
                  f"heavy-favorite ask gap {sub['mean_gap_ask_cents_heavy']:+.2f}c "
                  f"(mid {sub['mean_gap_mid_cents_heavy']:+.2f}c), "
                  f"spread {sub['mean_spread_cents']:.1f}c, "
                  f"booksum {sub['mean_booksum']:.3f}")
    else:
        print("[7/9] Field evidence: data/live_quotes.csv not found, skipping.")
        gaps = None

    # ------------------------------------------------------------------- ML
    print("[8/9] ML benchmark (public stats vs the closing line) ...")
    model_table = models.build_model_table(raw, fights)
    ml_res, ml_extra = models.run_benchmark(model_table, split_date=args.split_date)
    ml_res.to_csv(results_dir / "ml_benchmark.csv", index=False)
    best = ml_res.loc[ml_res["model"] == "Stats + market (logistic)"].iloc[0]
    print(f"      best model log-loss diff vs market = {best['logloss_diff']:+.4f} "
          f"(95% CI [{best['diff_ci_lo']:+.4f}, {best['diff_ci_hi']:+.4f}])")
    edge = models.edge_backtest(
        ml_extra["test"], ml_extra["preds"]["Stats + market (logistic)"]
    )
    edge.to_csv(results_dir / "ml_edge_backtest.csv", index=False)
    summary["ml_benchmark"] = ml_res.drop(columns=["p_model_better"]).to_dict("records")

    # -------------------------------------------------------------- figures
    print("[9/9] Figures ...")
    plots.plot_calibration(cal_tab, figures_dir / "calibration.png")
    plots.plot_calibration(
        cal_eq, figures_dir / "calibration_equal_count.png",
        title="Equal-count bins tell the same story",
        caption=f"{len(cal_eq)} quantile bins of the de-vigged implied probability "
                f"(~{int(cal_eq['n'].mean())} bets each), grouped by nearest market price "
                "(results/calibration_buckets_equal_count.csv).",
    )
    plots.plot_returns_by_band(band_tab, figures_dir / "returns_by_band.png")
    plots.plot_cumulative_pnl(pnl, figures_dir / "cumulative_pnl.png")
    plots.plot_slope_stability(summary["era_slopes"], figures_dir / "slope_stability.png")
    plots.plot_threshold_sweep(sweep_fav, wf, figures_dir / "exploitability.png")
    plots.plot_fee_scenarios(fee_tab, figures_dir / "fees_scenarios.png")
    if gaps is not None:
        plots.plot_field_gaps(gaps, figures_dir / "field_gaps.png")
    plots.plot_ml_benchmark(ml_res, figures_dir / "ml_benchmark.png")

    with open(results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------------------------------- headline
    def wavg(sub: pd.DataFrame, col: str) -> float:
        """Bet-count-weighted average of a returns_by_band column."""
        return float((sub[col] * sub["n_bets"]).sum() / sub["n_bets"].sum())

    hu = band_tab[band_tab["band"].isin(["+300 to +499", "+500 or longer"])]
    hf = band_tab[band_tab["band"].isin(["-799 to -500", "-800 or shorter"])]
    print("\n================ HEADLINE RESULTS ================")
    print(f"Calibration slope (de-vigged): {t['slope']:.3f} > 1, p = {t['slope_p_vs_1']:.4f}")
    print(f"Heavy underdogs (+300+): priced {wavg(hu, 'implied_devig'):.1%}, "
          f"won {wavg(hu, 'win_rate'):.1%}; flat-bet return {wavg(hu, 'mean_return'):+.1%}")
    print(f"Heavy favorites (-500 or shorter): flat-bet return {wavg(hf, 'mean_return'):+.1%} "
          f"(vig-only benchmark {band_tab['vig_null_return'].mean():+.1%})")
    print(f"Exploitable edge, out of sample: {wf['test_roi']:+.1%} "
          f"(beats the vig, not significantly > 0: p = {wf['test_p_roi_gt0']:.2f})")
    if gaps is not None:
        fs = summary["field_evidence"]
        bh = fs.get("by_horizon", {})
        if {"midweek", "close"} <= set(bh):
            print(f"Field check: Kalshi heavy-favorite asks sit "
                  f"{bh['midweek']['mean_gap_ask_cents_heavy']:+.1f}c over the de-vigged "
                  f"consensus midweek but {bh['close']['mean_gap_ask_cents_heavy']:+.1f}c "
                  f"at close ({fs['n_cards']} pilot cards)")
        else:
            print(f"Field check: Kalshi heavy-favorite asks {fs['mean_gap_ask_cents_heavy']:+.1f}c "
                  f"vs de-vigged consensus over {fs['n_cards']} pilot cards")
    print("Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
