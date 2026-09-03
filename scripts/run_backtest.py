"""Run component backtests against the baselines and print score tables.

Requires a season-level parquet (one row per batter-season) with columns:
batter, season, pa, k, bb, hr, ab, xb_points, bip, hits_in_play, and
optionally age. Build it from the PA-level data (see src/eval/backtest.py
docstring for the schema each component needs).

Usage:
    python scripts/run_backtest.py --seasons data/parquet/hitter_seasons_agg.parquet \
        --component k_rate --train-through 2023

    # roadmap 0.2 sweep: every split from 2019→2020 through 2024→2025
    python scripts/run_backtest.py --seasons ... --component k_rate --sweep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.eval import HITTER_COMPONENTS, backtest, calibration, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=Path, required=True)
    parser.add_argument("--component", choices=sorted(HITTER_COMPONENTS),
                        required=True)
    parser.add_argument("--train-through", type=int, default=None)
    parser.add_argument("--sweep", action="store_true",
                        help="run all splits 2019→2020 through 2024→2025")
    parser.add_argument("--min-trials", type=int, default=100)
    parser.add_argument("--calibration-model", default="marcel")
    args = parser.parse_args()

    seasons = pd.read_parquet(args.seasons)
    splits = (
        [(y, y + 1) for y in range(2019, 2025)]
        if args.sweep else [(args.train_through, args.train_through + 1)]
    )
    if splits[0][0] is None:
        parser.error("--train-through or --sweep is required")

    all_scores = []
    for train_through, predict in splits:
        results = backtest(
            args.component, train_through, predict,
            seasons=seasons, min_trials=args.min_trials,
        )
        s = score(results).assign(train_through=train_through, predict_year=predict)
        all_scores.append(s)
        print(f"\n=== {args.component}: train ≤ {train_through} → {predict} ===")
        print(s.to_string(index=False))
        if not args.sweep:
            print(f"\nCalibration ({args.calibration_model}):")
            print(calibration(results, args.calibration_model).to_string(index=False))

    if args.sweep:
        combined = pd.concat(all_scores, ignore_index=True)
        print("\n=== Mean across splits ===")
        print(
            combined.groupby("model")[["log_loss", "mae", "rmse"]]
            .mean().sort_values("mae").to_string()
        )


if __name__ == "__main__":
    main()
