"""Pull Statcast from Baseball Savant into R2, and build PA outcomes.

The season archive in R2 ended at 2025, so Bayesian refits saw nothing from
the current year. This script closes that gap and can be re-run daily.

Re-downloading the whole season every night was the original approach (~70
CSV exports, ~15 minutes) but is wasteful once the season file already
exists: with no `--since`, the script now reads the existing
`statcast_<season>.parquet` from R2, resumes from `RESUME_OVERLAP_DAYS`
days before its latest `game_date` (Savant posts corrections and a chunk
boundary can land mid-game, so the existing dedup on
`(game_pk, at_bat_number, pitch_number)` handles the short overlap), fetches
only that tail, and merges it into the existing season file. If no season
file exists yet in R2, it falls back to a full build from March 1.

An explicit `--since` (or `--full`, which ignores whatever is in R2) is the
deliberate full-rebuild path — it always fetches from that date instead of
resuming, and is not merged against R2's copy: it fetches then overwrites,
same as `--full`. `--full` alone reruns the season default window.

    python scripts/ingest_statcast.py --season 2026                 # incremental, resumes from R2
    python scripts/ingest_statcast.py --season 2026 --since 2026-08-01   # explicit rebuild from a date
    python scripts/ingest_statcast.py --season 2026 --full          # full rebuild, March 1 on
    python scripts/ingest_statcast.py --season 2026 --no-upload --work-dir /tmp/sc

Writes, for season Y:
    s3://<bucket>/statcast/statcast_Y.parquet        pitch level, whole season
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

# Resuming from the existing R2 file's max game_date, minus this many days.
# Savant posts corrections to already-published games, and a chunk boundary
# can land mid-game, so the resume window overlaps the last few days already
# on file rather than starting exactly where it left off. The existing
# dedup on (game_pk, at_bat_number, pitch_number) — keeping the freshly
# fetched copy of a duplicate over the stale one — absorbs the overlap.
RESUME_OVERLAP_DAYS = 3

MERGE_KEYS = ("game_pk", "at_bat_number", "pitch_number")


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


def resolve_resume_since(season: int, raw_dir: Path, s3=None) -> tuple[date, Path | None]:
    """Where to resume an incremental (no `--since`) run from.

    Downloads the existing `statcast_<season>.parquet` from R2, if any, and
    returns the date `RESUME_OVERLAP_DAYS` before its latest `game_date`
    plus the local path it was downloaded to (so it can be merged back in
    later). If R2 has no season file yet, returns March 1 and `None` — a
    full build, logged as such.
    """
    s3 = s3 or get_s3_client()
    key = f"statcast/statcast_{season}.parquet"
    existing_path = raw_dir / f"statcast_{season}_r2existing.parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket(), key, str(existing_path))
    except Exception as exc:                        # noqa: BLE001 — any failure means "not there"
        logger.info("no existing s3://%s/%s in R2 (%s) — full build from %s",
                    bucket(), key, exc, date(season, 3, 1))
        return date(season, 3, 1), None

    max_date = pd.to_datetime(
        pq.read_table(existing_path, columns=["game_date"]).column("game_date").to_pandas()
    ).max().date()
    since = max_date - timedelta(days=RESUME_OVERLAP_DAYS)
    logger.info("existing %s: max game_date %s — resuming from %s (%d-day overlap)",
                key, max_date, since, RESUME_OVERLAP_DAYS)
    return since, existing_path


def merge_with_existing(existing_path: Path, fetched_path: Path, out_path: Path) -> Path:
    """Concat the existing season file with a freshly fetched tail, dedup, sort.

    The fetched copy of an overlapping pitch wins over the stale one, since
    Savant may have posted a correction since the existing file was written.
    """
    existing = pq.read_table(existing_path).to_pandas()
    fetched = pq.read_table(fetched_path).to_pandas()
    combined = pd.concat([existing, fetched], ignore_index=True)
    keys = [k for k in MERGE_KEYS if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    if "game_date" in combined.columns:
        combined = combined.sort_values("game_date")
    combined = combined.reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    return out_path


def upload(path: Path, key: str) -> None:
    s3 = get_s3_client()
    s3.upload_file(str(path), bucket(), key)
    logger.info("uploaded s3://%s/%s (%.1f MB)", bucket(), key, path.stat().st_size / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=date.today().year)
    ap.add_argument("--since", type=iso_date,
                    help="start date; forces a full (non-incremental) fetch from this date, "
                         "not merged against R2 (default: resume from R2, or March 1 if absent)")
    ap.add_argument("--until", type=iso_date, help="end date (default: today)")
    ap.add_argument("--full", action="store_true",
                    help="ignore R2 and fetch the whole season from March 1 (or --since); "
                         "alias for the explicit-rebuild path with no date override")
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
    elif args.since is not None or args.full:
        # Deliberate full-rebuild path: fetch the requested window from
        # Savant and overwrite R2's copy outright, no merge.
        mode = f"explicit --since {args.since}" if args.since is not None else "--full rebuild"
        logger.info("mode: %s — fetching from scratch, no R2 merge", mode)
        raw_path = fetch_to_parquet(args.season, args.since, args.until, raw, args.chunk_days)
        logger.info("fetched %d rows (full rebuild)", pq.read_table(raw_path).num_rows)
    else:
        since, existing_path = resolve_resume_since(args.season, raw)
        if existing_path is None:
            logger.info("mode: full build (no existing R2 file) — fetching from %s", since)
            raw_path = fetch_to_parquet(args.season, since, args.until, raw, args.chunk_days)
            logger.info("fetched %d rows (full build)", pq.read_table(raw_path).num_rows)
        else:
            logger.info("mode: incremental — fetching from %s", since)
            fetched_path = fetch_to_parquet(args.season, since, args.until, raw, args.chunk_days)
            fetched_rows = pq.read_table(fetched_path).num_rows
            raw_path = merge_with_existing(existing_path, fetched_path, raw_path)
            total_rows = pq.read_table(raw_path).num_rows
            logger.info("fetched %d rows, merged into %d total rows for the season",
                        fetched_rows, total_rows)

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
