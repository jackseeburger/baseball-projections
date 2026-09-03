"""Reconstruct the pre-first-pitch close of every settled Kalshi player prop.

The moneyline twin of this script is `backfill_market_closes.py`; this one
walks the seven prop series instead and writes one row per *contract* (a
player, a stat and a line) rather than per game:

    data/market/prop_closes_<season>.parquet          the close (committed)
    data/market/kalshi_prop_candles_<season>.parquet  the hourly path (committed)

Then `scripts/props_exam.py` prices those closes against a Marcel-with-partial
rate model and runs the fill-aware P&L on them, taker and maker.

**Both files are committed, and the candles are why.** Kalshi serves hourly
candlesticks only while a market is young enough; a month from now this path
cannot be reconstructed at any price. The closes were gitignored once and did
not survive a container restart, which cost a full re-fetch — 30k requests —
so the archive now lives where the repository keeps it.

Cost: one candlestick request per settled contract that traded, and Kalshi has
no bulk endpoint. A month of props is ~65k requests at ~19/s on eight threads.
`--start` and `--end` bound the window; progress is checkpointed every
`--checkpoint-every` markets under `--checkpoint-dir` (gitignored), so a run
killed halfway resumes where it stopped instead of starting over.

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

ROOT = Path(__file__).resolve().parent.parent


class Checkpoint:
    """Partial progress on disk, so a killed run resumes instead of restarting.

    Every `every` markets the closes and candles fetched since the last write
    go to a new numbered pair of part files and the tickers go to `done.txt`.
    Parts are never rewritten, so a checkpoint costs the size of one batch
    rather than the size of the archive, and an interrupted run loses at most
    the last batch. The closes are small enough to hold in memory as they
    accumulate; the candles are not (a month is well over a million rows), so
    they are read back off the parts at the end.
    """

    def __init__(self, directory: Path, every: int = 2000):
        self.dir = directory
        self.every = every
        self.dir.mkdir(parents=True, exist_ok=True)
        self.done_path = self.dir / "done.txt"
        self.rows: list[dict] = []          # every close row, this run and prior
        self.part = 0
        self._rows: list[dict] = []         # since the last flush
        self._candles: list[dict] = []
        self._tickers: list[str] = []

    def done(self) -> set[str]:
        """Tickers a previous run already fetched, close or no close."""
        if not self.done_path.exists():
            return set()
        return {t for t in self.done_path.read_text().split("\n") if t}

    def resume(self) -> None:
        """Load the closes a previous run checkpointed."""
        parts = sorted(self.dir.glob("closes_*.parquet"))
        self.part = len(parts)
        frames = [pd.read_parquet(p) for p in parts]
        frames = [f for f in frames if len(f)]
        self.rows = pd.concat(frames, ignore_index=True).to_dict("records") \
            if frames else []

    def on_market(self, ticker: str, row: dict | None, candles: list[dict]) -> None:
        self._tickers.append(ticker)
        if row is not None:
            self._rows.append(row)
        self._candles.extend(candles)
        if len(self._tickers) >= self.every:
            self.flush()

    def flush(self) -> None:
        if not self._tickers:
            return
        self.part += 1
        tag = f"{self.part:05d}"
        pd.DataFrame(self._rows, columns=None if self._rows else ["market_id"]) \
            .to_parquet(self.dir / f"closes_{tag}.parquet", index=False)
        pd.DataFrame(self._candles, columns=backfill.CANDLE_COLUMNS) \
            .to_parquet(self.dir / f"candles_{tag}.parquet", index=False)
        with self.done_path.open("a") as fh:
            fh.write("\n".join(self._tickers) + "\n")
        logger.info("checkpoint %s: +%d markets, +%d closes, +%d candles",
                    tag, len(self._tickers), len(self._rows), len(self._candles))
        self.rows.extend(self._rows)
        self._rows, self._candles, self._tickers = [], [], []

    def candles(self) -> list[dict]:
        """Every archived candle, read back off the parts."""
        frames = [pd.read_parquet(p) for p in sorted(self.dir.glob("candles_*.parquet"))]
        frames = [f for f in frames if len(f)]
        if not frames:
            return []
        return pd.concat(frames, ignore_index=True).to_dict("records")


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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/market")
    ap.add_argument("--candle-hours", type=int, default=backfill.CANDLE_HOURS_BEFORE,
                    help="how far before first pitch to archive the hourly path")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="default data/cache/prop_backfill_<season> (gitignored)")
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                    help="write partial progress every N markets")
    ap.add_argument("--restart", action="store_true",
                    help="ignore the checkpoint and fetch every market again")
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

    ckpt_dir = args.checkpoint_dir or ROOT / f"data/cache/prop_backfill_{args.season}"
    ckpt = Checkpoint(ckpt_dir, every=args.checkpoint_every)
    done = set()
    if not args.restart:
        ckpt.resume()
        done = ckpt.done()
    if done:
        logger.info("%d markets already fetched (%d closes); resuming",
                    len(done), len(ckpt.rows))
    todo = [m for m in markets if m["ticker"] not in done]

    backfill.kalshi_prop_closes(args.season, markets=todo, workers=args.workers,
                                candle_hours=args.candle_hours,
                                on_market=ckpt.on_market)
    ckpt.flush()
    rows, candles = ckpt.rows, ckpt.candles()
    logger.info("%d contracts with a pre-first-pitch close, %d hourly candles",
                len(rows), len(candles))

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
    df.to_parquet(out, index=False, compression="gzip")

    candle_out = args.out_dir / f"kalshi_prop_candles_{args.season}.parquet"
    cf = backfill.prop_candle_frame(candles, df)
    if args.append and candle_out.exists():
        old = pd.read_parquet(candle_out)
        cf = (pd.concat([old, cf], ignore_index=True)
              .drop_duplicates(subset=["market_id", "end_period_ts"], keep="last")
              .sort_values(["game_pk", "market_id", "end_period_ts"])
              .reset_index(drop=True))
    cf.to_parquet(candle_out, index=False, compression="gzip")

    print(coverage(df).to_string())
    print(f"\n{len(df)} rows, {df['game_pk'].nunique()} games, "
          f"{df['player_id'].nunique()} players, "
          f"{df['game_date'].min()} .. {df['game_date'].max()} → {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")
    traded = cf[cf["volume"] > 0] if len(cf) else cf
    print(f"{len(cf)} hourly candles over {cf['market_id'].nunique() if len(cf) else 0} "
          f"markets ({len(traded)} with volume) → {candle_out} "
          f"({candle_out.stat().st_size / 1e6:.2f} MB)")


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
