"""Score the preseason 2026 projections — ours and the public systems —
against 2026 actuals, on the same players (roadmap 0.3 / accuracy page).

Providers: our Bayesian preseason files (data/projections/*_2026.parquet,
generated ~Apr 10), Steamer / ZiPS / Depth Charts as captured Apr 9
(data/projections/comparison_2026.parquet), Marcel and league average
fit on 2015-2025 season totals.

Usage:
    python scripts/score_2026_projections.py [--min-trials 150]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.eval import backtest, frame_provider, score
from src.eval.baselines import BASELINES

ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ["k_rate", "bb_rate", "hr_rate", "iso", "babip"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-trials", type=int, default=150)
    args = parser.parse_args()

    seasons = pd.read_parquet(ROOT / "data/parquet/hitter_seasons_api.parquet")
    public = pd.read_parquet(ROOT / "data/projections/comparison_2026.parquet")
    tables = []
    for c in COMPONENTS:
        ours = pd.read_parquet(ROOT / f"data/projections/{c}_projections_2026.parquet")
        providers = {
            "bayes_preseason": frame_provider(ours, pred_col=f"projected_{c}"),
            "steamer": frame_provider(public, pred_col=f"stea_{c}"),
            "zips": frame_provider(public, pred_col=f"zips_{c}"),
            "depth_charts": frame_provider(public, pred_col=f"dept_{c}"),
            "marcel": BASELINES["marcel"],
            "league_average": BASELINES["league_average"],
        }
        results = backtest(c, 2025, 2026, seasons=seasons, providers=providers,
                           min_trials=args.min_trials)
        tables.append(score(results))
    out = pd.concat(tables).reset_index(drop=True)
    print(out.round(5).to_string(index=False))

    ranks = (out.groupby("component")
             .apply(lambda g: g.set_index("model")["mae"].rank(), include_groups=False)
             .unstack(0))
    print("\nMAE rank by component (1 = best):")
    print(ranks.assign(mean_rank=ranks.mean(1)).sort_values("mean_rank").round(1).to_string())


if __name__ == "__main__":
    main()
