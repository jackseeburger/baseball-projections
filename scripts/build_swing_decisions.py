"""Build the monthly swing-decision artifact from the Statcast archive.

Reads `statcast_<year>.parquet` (pulled from R2 into --raw-dir) and writes one
small committed parquet of swing-decision counts per batter, per calendar
month — pitches in and out of the zone, swings at each, contact and whiffs at
each, called strikes taken in the zone. See `src/data/swing_decisions.py` for
what a bucket contains, how the zone is decided, and why the grain is a month.

    # pull the archive (1.4 GB for 2015-2026), then:
    python scripts/build_swing_decisions.py --seasons 2015 2026

    # the nightly refresh of the current season only
    python scripts/build_swing_decisions.py --update-season 2026 --download

The output is the input to every swing-decision feature in `src/eval/swing.py`,
and it is committed precisely so the download does not have to be repeated.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.swing_decisions import (
    DEFAULT_PATH,
    build_year,
    save_monthly,
    write_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ROOT = Path(__file__).resolve().parent.parent


def download_archive(years: list[int], raw_dir: Path) -> None:
    """Fetch any missing `statcast_<year>.parquet` from R2."""
    from src.data.r2 import bucket, get_s3_client

    s3, b = get_s3_client(), bucket()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        path = raw_dir / f"statcast_{year}.parquet"
        if path.exists():
            continue
        logging.info("downloading statcast_%d.parquet", year)
        s3.download_file(b, f"statcast/statcast_{year}.parquet", str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2015, 2026],
                    metavar=("FIRST", "LAST"))
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / DEFAULT_PATH)
    ap.add_argument("--download", action="store_true",
                    help="pull missing seasons from R2 first")
    ap.add_argument(
        "--update-season", type=int, default=None,
        help="rebuild only this season's rows in --out and leave every other "
             "season's rows untouched — the nightly refresh of the current "
             "season from statcast/statcast_<year>.parquet on R2, rather than "
             "re-aggregating twelve seasons of Statcast every night.")
    args = ap.parse_args()

    if args.update_season is not None:
        year = args.update_season
        if args.download:
            download_archive([year], args.raw_dir)
        fresh = build_year(year, args.raw_dir)
        logging.info("%d: %d buckets", year, len(fresh))
        if args.out.exists():
            existing = pd.read_parquet(args.out)
            existing = existing[existing["season"] != year]
            out = pd.concat([existing, fresh], ignore_index=True)
        else:
            out = fresh
        out = out.sort_values(["season", "month", "batter"], ignore_index=True)
        path = save_monthly(out, args.out)
        meta = write_meta(args.out, seasons_built=[year])
        logging.info("wrote %s (%d rows total, %d for %d, %.1f MB); %s",
                     path, len(out), len(fresh), year,
                     path.stat().st_size / 1e6, meta)
        return

    years = list(range(args.seasons[0], args.seasons[1] + 1))
    if args.download:
        download_archive(years, args.raw_dir)

    frames = []
    for year in years:
        g = build_year(year, args.raw_dir)
        logging.info("%d: %d buckets", year, len(g))
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["season", "month", "batter"], ignore_index=True)
    path = save_monthly(out, args.out)
    meta = write_meta(args.out, seasons_built=years)
    logging.info("wrote %s (%d rows, %.1f MB); %s", path, len(out),
                 path.stat().st_size / 1e6, meta)


if __name__ == "__main__":
    main()
