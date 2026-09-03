"""Build the pitcher season table the station A pitcher provider trains on.

    python scripts/build_pitcher_seasons.py --start 2015 --end 2026

Writes `data/parquet/pitcher_seasons_api.parquet` (gitignored, rebuildable),
the pitcher twin of `hitter_seasons_api.parquet`: one row per pitcher-season
in the schema `src/eval/pitchers.py` reads.

    pitcher, season, bf, k, bb, hbp, bbhbp, hr, ab, h, sf, outs,
    bip, hits_in_play, age

The counts come from the Stats API season-pitching endpoint, the same feed
station E's starter rates already use (`fetch_season_pitching`), so the two
stations cannot disagree about what a pitcher did. Responses are cached under
`data/cache/statsapi/`, so a rebuild is offline once the first run has fetched.

The derived columns use the standard identities: BIP = AB - K - HR + SF and
hits in play = H - HR, matching `src/eval/intraseason.aggregate_pa` so a
season built from the API table and one aggregated from PA-level Statcast
rows mean the same thing.

`age` is the Chadwick register's age as of June 30 (`src/data/birthdates.py`),
the project's age of record — the pitching endpoint's own `age` field is the
player's age *today*, not during the season, so it is useless for a
historical row.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import fetch_season_pitching

OUT_PARQUET = ROOT / "data/parquet/pitcher_seasons_api.parquet"
COLUMNS = ["pitcher", "season", "bf", "k", "bb", "hbp", "bbhbp", "hr", "ab",
           "h", "sf", "outs", "bip", "hits_in_play", "age"]

logger = logging.getLogger("build_pitcher_seasons")


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived count columns and force the counts to int64."""
    out = df.copy()
    for column in ["bf", "k", "bb", "hbp", "hr", "ab", "h", "sf"]:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")
    out["outs"] = pd.to_numeric(out.get("outs", 0), errors="coerce").fillna(0.0).astype(float)
    out["bbhbp"] = out["bb"] + out["hbp"]
    out["bip"] = out["ab"] - out["k"] - out["hr"] + out["sf"]
    out["hits_in_play"] = out["h"] - out["hr"]
    return out


def add_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Chadwick seasonal age (June 30) per pitcher-season; NaN where unknown."""
    from src.data.birthdates import load_birthdates, seasonal_age

    out = df.copy()
    try:
        birthdates = load_birthdates()
    except Exception as exc:                                   # noqa: BLE001
        logger.warning("no birthdate register (%s); ages left empty", exc)
        out["age"] = np.nan
        return out
    out["age"] = seasonal_age(birthdates, out["pitcher"].to_numpy(),
                              out["season"].to_numpy())
    return out


def build(start: int, end: int, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for season in range(start, end + 1):
        df = fetch_season_pitching(season, refresh=refresh)
        if df.empty:
            logger.warning("%s: no pitcher seasons returned", season)
            continue
        frames.append(df)
        logger.info("%s: %d pitcher seasons, %d BF", season, len(df),
                    int(df["bf"].sum()))
    if not frames:
        raise SystemExit("no seasons fetched")
    table = add_ages(derive(pd.concat(frames, ignore_index=True)))
    return table[COLUMNS].sort_values(["season", "pitcher"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--out", type=Path, default=OUT_PARQUET)
    p.add_argument("--refresh", action="store_true",
                   help="re-pull the cached Stats API responses")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    table = build(args.start, args.end, refresh=args.refresh)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {len(table)} pitcher-seasons, "
          f"{table['season'].min()}-{table['season'].max()}, "
          f"age coverage {table['age'].notna().mean():.1%}")
    print(table.groupby("season").agg(pitchers=("pitcher", "nunique"),
                                      bf=("bf", "sum")).to_string())


if __name__ == "__main__":
    main()
