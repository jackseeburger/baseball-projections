"""Archive the pre-game price *path* for every settled Kalshi game market.

The closes archive (`backfill_market_closes.py`) keeps one quote per game, and
one quote can only price a **taker**: the only trade it can simulate is
crossing that quote. The maker exam asks a different question — *would a
resting limit order have been filled?* — and that depends on where the price
travelled before first pitch, not on where it ended. Kalshi publishes hourly
OHLC per market (traded price high/low/open/close, the bid and ask at the end
of the hour, and the hour's volume), so the path is reconstructable after the
fact; this script freezes it before it is needed.

Writes one row per (market, hour) over the 24 hours before first pitch:

    data/market/kalshi_candles_<season>.parquet

Kalshi rate-limits bursts of candlestick calls, so requests are paced, 429s
back off with Retry-After (`src/market/http.get_json`), a market that fails is
logged and skipped rather than killing the run, and progress is checkpointed
to the parquet every `--checkpoint-every` markets. Re-running resumes: markets
already in the file are skipped unless `--refresh` is passed.

Usage:
    python scripts/backfill_kalshi_candles.py --season 2026
    python scripts/backfill_kalshi_candles.py --season 2026 --pace 0.5 --limit 50
    python scripts/backfill_kalshi_candles.py --season 2026 --refresh
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.market import backfill

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("candles")

ROOT = Path(__file__).resolve().parent.parent


def load_existing(path: Path) -> pd.DataFrame:
    """Whatever a previous (possibly killed) run already archived."""
    if not path.exists():
        return pd.DataFrame(columns=backfill.CANDLE_COLUMNS)
    return pd.read_parquet(path)


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--closes", type=Path, default=None,
                    help="market_closes_<season>.parquet (one row per venue+game)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default data/market/kalshi_candles_<season>.parquet")
    ap.add_argument("--pace", type=float, default=0.25,
                    help="seconds between candlestick calls")
    ap.add_argument("--hours-before", type=int, default=backfill.CANDLE_HOURS_BEFORE,
                    help="how far before first pitch to archive")
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="write partial progress every N markets")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N markets (a smoke test)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch markets already in the output file")
    args = ap.parse_args()

    closes_path = args.closes or ROOT / f"data/parquet/market_closes_{args.season}.parquet"
    out_path = args.out or ROOT / f"data/market/kalshi_candles_{args.season}.parquet"

    closes = pd.read_parquet(closes_path)
    kalshi_rows = closes[closes["venue"] == "kalshi"].copy()
    logger.info("%d Kalshi markets in %s", len(kalshi_rows), closes_path)

    existing = load_existing(out_path)
    done = set() if args.refresh else set(existing["market_id"].astype(str))
    if done:
        logger.info("%d markets already archived; resuming", len(done))
    todo = kalshi_rows[~kalshi_rows["market_id"].astype(str).isin(done)]
    if args.limit is not None:
        todo = todo.head(args.limit)
    logger.info("fetching %d markets at %.2fs pacing", len(todo), args.pace)

    base = existing.to_dict("records")
    collected: list[dict] = []
    state = {"n": 0}

    def checkpoint(market_id: str, rows: list[dict]) -> None:
        """Fold each market in as it lands so a killed run keeps its work."""
        collected.extend(rows)
        state["n"] += 1
        if state["n"] % args.checkpoint_every:
            return
        merged = backfill.candle_frame(base + collected)
        write(merged, out_path)
        logger.info("checkpoint: %d markets, %d rows → %s",
                    state["n"], len(merged), out_path)

    _, failures = backfill.kalshi_candle_archive(
        todo, pace_seconds=args.pace, hours_before=args.hours_before,
        skip_markets=done, on_market=checkpoint)

    df = backfill.candle_frame(base + collected)
    write(df, out_path)

    n_markets = df["market_id"].nunique()
    per_market = df.groupby("market_id").size()
    traded = df[df["volume"] > 0]
    print(f"\n{n_markets} markets, {len(df)} hourly candles "
          f"({per_market.mean():.1f} per market, min {per_market.min()}, "
          f"max {per_market.max()})")
    print(f"{len(traded)} candles with volume ({100 * len(traded) / max(len(df), 1):.1f}%), "
          f"{traded['volume'].sum():,.0f} contracts traded pre-game")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    if failures:
        print(f"\n{len(failures)} markets failed and were skipped:")
        for t in failures[:20]:
            print(f"  {t}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print("re-run the command to retry them (finished markets are skipped)")


if __name__ == "__main__":
    main()
