"""How wide the strength distribution actually is, week by week.

The number every claim in [docs/parameter-uncertainty.md](../docs/parameter-uncertainty.md)
rests on: at each weekly as-of date of each backtested season, the implied
standard deviation of a club's talent win% under the posterior the 60-game
ballast defines, and — for scale — how many wins that is over the schedule the
club still has to play. Reads only the cached schedules, so it costs nothing
next to the projection stage.

Usage:
    python scripts/report_strength_width.py [--json-out P]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.run_team_backtest import DEFAULT_SEASONS, OUT_DIR, season_paths
from src.eval import team_season as ts
from src.eval.team_backtest import BUCKET_LABELS, SEASON_BUCKETS
from src.sim.strength import regressed_strength, strength_distribution


def widths(seasons, out_dir: Path, step_days: int = 7) -> pd.DataFrame:
    rows = []
    for season in seasons:
        paths = season_paths(season, out_dir)
        if not paths["schedule"].exists():
            continue
        sched = pd.read_parquet(paths["schedule"])
        teams = pd.read_parquet(paths["teams"])
        for as_of in ts.weekly_cutoffs(sched, teams, step_days=step_days):
            split = ts.split_season_at(sched, teams, as_of, season)
            point = regressed_strength(split.standings)
            for name, sampling in (("posterior", "posterior"),
                                   ("bootstrap", "bootstrap")):
                d = strength_distribution(point, split.played, split.standings,
                                          sampling=sampling)
                sd = d.talent_sd()
                left = split.club_games_remaining().reindex(sd.index)
                rows.append({
                    "season": season, "as_of": as_of, "width": name,
                    "season_fraction": split.games_played
                    / (split.games_played + split.games_remaining),
                    "games_played_per_club":
                        float(d.sampling.games.mean()),
                    "club_games_remaining": float(left.mean()),
                    "talent_sd": float(sd.mean()),
                    "wins_sd": float((sd * left).mean()),
                })
    out = pd.DataFrame(rows)
    out["bucket"] = pd.cut(
        out["season_fraction"],
        bins=[b[0] for b in SEASON_BUCKETS] + [SEASON_BUCKETS[-1][1]],
        labels=BUCKET_LABELS, right=False, include_lowest=True)
    return out


def summary(w: pd.DataFrame) -> pd.DataFrame:
    return (w.groupby(["width", "bucket"], observed=True)
            .agg(n=("talent_sd", "size"),
                 games_played=("games_played_per_club", "mean"),
                 games_left=("club_games_remaining", "mean"),
                 talent_sd=("talent_sd", "mean"),
                 wins_sd=("wins_sd", "mean"))
            .reset_index())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    w = widths(list(DEFAULT_SEASONS), args.out_dir)
    s = summary(w)
    print(s.round(4).to_string(index=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            json.loads(s.to_json(orient="records")), indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
