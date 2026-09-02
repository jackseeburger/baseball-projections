"""Reconstruct the pre-game closing price for every settled game of a season.

Issue #6 part 3. Writes one row per (venue, game_pk):

    data/parquet/market_closes_<season>.parquet
    s3://<bucket>/market/market_closes_<season>.parquet   (unless --no-upload)

Then `scripts/backtest_game_odds.py --market data/parquet/market_closes_2026.parquet`
scores the simulator's per-game probabilities against the market on the
same games.

Usage:
    python scripts/backfill_market_closes.py --season 2026
    python scripts/backfill_market_closes.py --season 2026 --venues kalshi --no-upload
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
from src.market import backfill

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--venues", nargs="+", default=["kalshi", "polymarket"])
    ap.add_argument("--out-dir", type=Path, default=Path("data/parquet"))
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    schedule = fetch_schedule(f"{args.season}-03-01", f"{args.season}-11-15")
    schedule = schedule[schedule["game_type"] == "R"]

    rows: list[dict] = []
    if "kalshi" in args.venues:
        k = backfill.kalshi_closes(args.season)
        logger.info("kalshi: %d games with a pre-game close", len(k))
        rows += k
    if "polymarket" in args.venues:
        p = backfill.polymarket_closes(args.season)
        logger.info("polymarket: %d games with a pre-game close", len(p))
        rows += p

    df = backfill.to_frame(rows, schedule)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"market_closes_{args.season}.parquet"
    df.to_parquet(out, index=False)

    summary = df.groupby("venue").agg(
        games=("game_pk", "nunique"),
        first=("game_date", "min"), last=("game_date", "max"),
        median_min_before=("minutes_before_pitch", "median"),
        mean_p_home=("p_home_close", "mean"),
    )
    print(summary.to_string())
    both = df.pivot_table(index="game_pk", columns="venue", values="p_home_close")
    if {"kalshi", "polymarket"} <= set(both.columns):
        both = both.dropna()
        print(f"\n{len(both)} games priced by both venues; "
              f"mean |kalshi - polymarket| = {(both['kalshi'] - both['polymarket']).abs().mean():.4f}, "
              f"corr = {both['kalshi'].corr(both['polymarket']):.3f}")
    print(f"\nwrote {len(df)} rows → {out}")

    if args.no_upload:
        return
    from src.data.r2 import bucket, get_s3_client
    key = f"market/market_closes_{args.season}.parquet"
    get_s3_client().upload_file(str(out), bucket(), key)
    print(f"uploaded s3://{bucket()}/{key}")


if __name__ == "__main__":
    main()
