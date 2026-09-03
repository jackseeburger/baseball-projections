"""Reconstruct the pre-first-pitch close of every settled Kalshi player prop.

The moneyline twin of this script is `backfill_market_closes.py`; this one
walks the seven prop series instead and writes one row per *contract* (a
player, a stat and a line) rather than per game:

    data/parquet/prop_closes_<season>.parquet   (gitignored)

Then `scripts/props_exam.py` prices those closes against a Marcel-with-partial
rate model and runs the fill-aware P&L on them.

Cost: one candlestick request per settled contract that traded, and Kalshi has
no bulk endpoint. A month of props is ~30k requests at ~12/s. `--start` and
`--end` bound the window; `--append` merges into an existing file so a run can
be taken in bites.

Usage:
    python scripts/backfill_prop_closes.py --season 2026
    python scripts/backfill_prop_closes.py --start 2026-08-01 --end 2026-09-02
    python scripts/backfill_prop_closes.py --stats hr k --workers 4
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.mlb_stats_api import fetch_schedule
from src.market import backfill, kalshi, players

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_props")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--start", help="earliest market close (YYYY-MM-DD)")
    ap.add_argument("--end", help="latest market close (YYYY-MM-DD)")
    ap.add_argument("--stats", nargs="+", default=None,
                    help=f"subset of {sorted(set(kalshi.PROP_SERIES.values()))}")
    ap.add_argument("--min-volume", type=float, default=1.0,
                    help="skip contracts that never traded (default 1)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=Path("data/parquet"))
    ap.add_argument("--append", action="store_true",
                    help="merge into the existing parquet instead of replacing it")
    ap.add_argument("--count-only", action="store_true",
                    help="list settled markets and stop before the candlesticks")
    args = ap.parse_args()

    series = kalshi.PROP_SERIES
    if args.stats:
        wanted = {s if s.startswith("prop_") else f"prop_{s}" for s in args.stats}
        series = {k: v for k, v in series.items() if v in wanted}
        if not series:
            raise SystemExit(f"no prop series matches {args.stats}")

    markets = backfill.kalshi_settled_props(
        args.season, series=series, start=args.start, end=args.end,
        min_volume=args.min_volume)
    logger.info("%d settled prop contracts that traded", len(markets))
    if args.count_only:
        return

    rows = backfill.kalshi_prop_closes(args.season, markets=markets,
                                       workers=args.workers)
    logger.info("%d contracts with a pre-first-pitch close", len(rows))

    schedule = fetch_schedule(f"{args.season}-03-01", f"{args.season}-11-15")
    schedule = schedule[schedule["game_type"] == "R"]
    resolver = players.NameResolver(args.season)
    df = backfill.prop_frame(rows, schedule, resolver=resolver)
    logger.info("player ids: %d/%d rows resolved, %d names unresolved",
                int(df["player_id"].notna().sum()), len(df), len(resolver.misses))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"prop_closes_{args.season}.parquet"
    if args.append and out.exists():
        old = pd.read_parquet(out)
        df = (pd.concat([old, df], ignore_index=True)
              .drop_duplicates(subset=["venue", "market_id"], keep="last")
              .sort_values(["game_date", "game_pk", "prop_stat", "player_id", "prop_line"])
              .reset_index(drop=True))
    df.to_parquet(out, index=False)

    print(coverage(df).to_string())
    print(f"\n{len(df)} rows, {df['game_pk'].nunique()} games, "
          f"{df['player_id'].nunique()} players, "
          f"{df['game_date'].min()} .. {df['game_date'].max()} → {out}")


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Markets, games, players and settlement rate per prop stat."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby("prop_stat").agg(
        markets=("market_id", "nunique"),
        games=("game_pk", "nunique"),
        players=("player_id", "nunique"),
        lines=("prop_line", "nunique"),
        mean_p_over=("p_over_close", "mean"),
        over_rate=("over_hit", "mean"),
        median_min_before=("minutes_before_pitch", "median"),
        median_volume=("volume_total", "median"),
    ).sort_values("markets", ascending=False)


if __name__ == "__main__":
    main()
