#!/usr/bin/env python3
"""Run the full MMA market-efficiency analysis end to end.

Usage:  python run_analysis.py [--data data/ufc-master.csv]

Downloads the dataset if needed, runs every test, writes tables to results/,
figures to figures/, and prints a summary.
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

from mma_efficiency import calibration, data, exploit, models, plots, returns  # noqa: E402

ERAS = [
    ("2010–2015", "2010-01-01", "2016-01-01"),
    ("2016–2020", "2016-01-01", "2021-01-01"),
    ("2021–2026", "2021-01-01", "2027-01-01"),
]


def main() -> None:
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
    print("[1/6] Loading data ...")
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
    print("[2/6] Calibration tests ...")
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
    print("[3/6] Returns by odds band ...")
    band_tab = returns.returns_by_band(sides)
    band_tab.to_csv(results_dir / "returns_by_band.csv", index=False)
    trend = returns.returns_trend_test(sides)
    summary["returns_trend"] = trend
    print(f"      return-vs-price slope = {trend['slope']:.3f} "
          f"(t = {trend['t_stat']:.2f}, p = {trend['p_value']:.2g})")

    # ------------------------------------------------------------- backtest
    print("[4/6] Strategy backtests ...")
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
    print("[5/7] Exploitability: walk-forward test of a pre-registered rule ...")
    sweep_fav = exploit.threshold_sweep(sides)
    sweep_dog = exploit.threshold_sweep(sides, cutoffs=exploit.UNDERDOG_CUTOFFS)
    pd.concat([sweep_fav, sweep_dog]).to_csv(results_dir / "threshold_sweep.csv", index=False)
    wf = exploit.walk_forward_rule(sides)
    summary["walk_forward"] = wf
    print(f"      rule '{wf['chosen_rule']}' picked on pre-{wf['train_end'][:4]} data; "
          f"out-of-sample ROI {wf['test_roi']:+.2%} on {wf['test_n_bets']:,} bets "
          f"(95% CI [{wf['test_roi_ci'][0]:+.2%}, {wf['test_roi_ci'][1]:+.2%}], "
          f"p(ROI>0) one-sided = {wf['test_p_roi_gt0']:.3f})")

    # ------------------------------------------------------------------- ML
    print("[6/7] ML benchmark (public stats vs the closing line) ...")
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
    print("[7/7] Figures ...")
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
    plots.plot_ml_benchmark(ml_res, figures_dir / "ml_benchmark.png")

    with open(results_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ------------------------------------------------------------- headline
    def wavg(sub: pd.DataFrame, col: str) -> float:
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
    print("Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
