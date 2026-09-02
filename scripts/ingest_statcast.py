"""Pull a Statcast season from Baseball Savant into R2, and build PA outcomes.

The season archive in R2 ended at 2025, so Bayesian refits saw nothing from
the current year. This script closes that gap and can be re-run daily: it
refetches the season (or a date window) and overwrites the year's files.

    python scripts/ingest_statcast.py --season 2026
    python scripts/ingest_statcast.py --season 2026 --since 2026-08-01
    python scripts/ingest_statcast.py --season 2026 --no-upload --work-dir /tmp/sc

Writes, for season Y:
    s3://<bucket>/statcast/statcast_Y.parquet        pitch level
    s3://<bucket>/pa_outcomes/pa_outcomes_Y.parquet  one row per plate appearance

The PA file is what the Modal training functions read; getting it onto the
Modal volume is a separate step that must run from GitHub Actions, because
Modal's client speaks gRPC and cloud sessions cannot.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq

from src.data.pa_outcomes_pipeline import process_year
from src.data.r2 import bucket, get_s3_client
from src.data.statcast_savant import fetch_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_to_parquet(season: int, start: date | None, end: date | None,
                     raw_dir: Path, chunk_days: int) -> Path:
    """Fetch month by month, writing each to disk, then combine.

    Monthly pieces keep peak memory near one month of pitches rather than a
    whole season.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = raw_dir / f"_parts_{season}"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir()

    start = start or date(season, 3, 1)
    end = end or min(date(season, 11, 15), date.today())
    total = 0
    cursor = start
    while cursor <= end:
        nxt = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        month_end = min(nxt - timedelta(days=1), end)
        df = fetch_season(season, start=cursor, end=month_end, chunk_days=chunk_days)
        if len(df):
            df.to_parquet(parts_dir / f"{cursor:%Y%m}.parquet", index=False)
            total += len(df)
            logger.info("%s: %d pitches (running total %d)", f"{cursor:%B %Y}", len(df), total)
        cursor = nxt

    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no Statcast pitches returned for {season} {start}..{end}")

    out = raw_dir / f"statcast_{season}.parquet"
    table = pq.read_table(parts)
    pq.write_table(table, out)
    shutil.rmtree(parts_dir)
    logger.info("wrote %s (%d pitches, %.1f MB)", out, table.num_rows, out.stat().st_size / 1e6)
    return out


def upload(path: Path, key: str) -> None:
    s3 = get_s3_client()
    s3.upload_file(str(path), bucket(), key)
    logger.info("uploaded s3://%s/%s (%.1f MB)", bucket(), key, path.stat().st_size / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--since", type=iso_date, help="start date (default: March 1)")
    ap.add_argument("--until", type=iso_date, help="end date (default: today)")
    ap.add_argument("--chunk-days", type=int, default=3)
    ap.add_argument("--work-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--no-upload", action="store_true", help="build files but skip R2")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="reuse an existing statcast_<season>.parquet in --work-dir")
    args = ap.parse_args()

    raw = args.work_dir
    raw_path = raw / f"statcast_{args.season}.parquet"
    if args.skip_fetch:
        if not raw_path.exists():
            raise SystemExit(f"--skip-fetch given but {raw_path} does not exist")
        logger.info("reusing %s", raw_path)
    else:
        raw_path = fetch_to_parquet(args.season, args.since, args.until, raw, args.chunk_days)

    pa = process_year(args.season, data_dir=str(raw))
    if pa.empty:
        raise SystemExit(f"no plate appearances built for {args.season}")
    pa_dir = Path("data/parquet/pa_outcomes")
    pa_dir.mkdir(parents=True, exist_ok=True)
    pa_path = pa_dir / f"pa_outcomes_{args.season}.parquet"
    pa.to_parquet(pa_path, index=False)

    games = pa["game_pk"].nunique() if "game_pk" in pa.columns else float("nan")
    logger.info("%d PAs across %s games; K%%=%.3f BB%%=%.3f HR%%=%.4f",
                len(pa), games, pa["is_k"].mean(), pa["is_bb"].mean(), pa["is_hr"].mean())

    if args.no_upload:
        logger.info("--no-upload: left files in %s and %s", raw_path, pa_path)
        return
    upload(raw_path, f"statcast/statcast_{args.season}.parquet")
    upload(pa_path, f"pa_outcomes/pa_outcomes_{args.season}.parquet")


if __name__ == "__main__":
    main()
