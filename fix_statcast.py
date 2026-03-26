#!/usr/bin/env python3
"""Re-download partial/missing Statcast years and rebuild aggregates."""
import sys, os
sys.path.insert(0, '.')

import pandas as pd
from src.data.statcast_pipeline import (
    fetch_statcast_season, aggregate_hitter_statcast, aggregate_pitcher_statcast,
    build_statcast_pipeline
)
from src.config import RAW_DIR, PARQUET_DIR
from src.utils.helpers import setup_logging

logger = setup_logging("fix_statcast")

# Check what we have vs what's expected
expected_pitches = {
    # Full season should be ~700-760K pitches for a normal year
    2015: 700000,  # Got 380K (partial — 2 of 4 chunks failed)
    2016: 700000,  # Got 726K ✓
    2017: 700000,  # Got 735K ✓
    2018: 700000,  # Got 130K (partial — 3 of 4 chunks failed)
    2019: 700000,  # Got 763K ✓
    2020: 250000,  # COVID shortened season, got 130K (partial)
    2021: 700000,  # Got 535K (partial — 1 chunk failed)
    2022: 700000,  # Got 775K ✓
    2023: 700000,  # Got 774K ✓
    2024: 0,       # Missing entirely
    2025: 0,       # Missing entirely
}

# Determine which years need re-download
redownload = []
for year in range(2015, 2026):
    raw_file = RAW_DIR / f"statcast_{year}.parquet"
    if not raw_file.exists():
        logger.info(f"{year}: MISSING — will download")
        redownload.append(year)
    else:
        df = pd.read_parquet(raw_file)
        threshold = expected_pitches.get(year, 700000)
        if year == 2020:
            threshold = 200000  # COVID year
        if len(df) < threshold * 0.8:  # Less than 80% of expected
            logger.info(f"{year}: PARTIAL ({len(df):,} pitches, expected ~{threshold:,}) — will re-download")
            redownload.append(year)
        else:
            logger.info(f"{year}: OK ({len(df):,} pitches)")

if not redownload:
    logger.info("All years look complete!")
else:
    logger.info(f"\nRe-downloading {len(redownload)} years: {redownload}")
    for year in redownload:
        raw_file = RAW_DIR / f"statcast_{year}.parquet"
        # Remove old partial data
        if raw_file.exists():
            os.remove(raw_file)
        
        df = fetch_statcast_season(year)
        if not df.empty:
            df.to_parquet(raw_file, index=False)
            logger.info(f"  {year}: saved {len(df):,} pitches")
        else:
            logger.warning(f"  {year}: no data returned!")

# Now rebuild the aggregate parquet files from all raw data
logger.info("\nRebuilding season aggregates from all raw data...")
all_hitters = []
all_pitchers = []
for year in range(2015, 2026):
    raw_file = RAW_DIR / f"statcast_{year}.parquet"
    if raw_file.exists():
        df = pd.read_parquet(raw_file)
        h = aggregate_hitter_statcast(df, year)
        p = aggregate_pitcher_statcast(df, year)
        all_hitters.append(h)
        all_pitchers.append(p)
        logger.info(f"  {year}: {len(h)} hitters, {len(p)} pitchers")

hitter_df = pd.concat(all_hitters, ignore_index=True)
pitcher_df = pd.concat(all_pitchers, ignore_index=True)

hitter_df.to_parquet(PARQUET_DIR / "statcast_hitters.parquet", index=False)
pitcher_df.to_parquet(PARQUET_DIR / "statcast_pitchers.parquet", index=False)

logger.info(f"\nDone! Statcast hitters: {len(hitter_df):,} player-seasons, pitchers: {len(pitcher_df):,} player-seasons")
