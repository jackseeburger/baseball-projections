#!/usr/bin/env python3
"""
Master pipeline — runs all data acquisition and projection steps in order.

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --skip-statcast  # Skip slow Statcast download
    python run_pipeline.py --force      # Force re-download everything
    python run_pipeline.py --marcel-only    # Just run Marcel on existing data
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import setup_logging

logger = setup_logging("pipeline")


def main():
    args = set(sys.argv[1:])
    force = "--force" in args
    skip_statcast = "--skip-statcast" in args
    marcel_only = "--marcel-only" in args
    
    start = time.time()
    
    if not marcel_only:
        # Step 1: Historical data (FanGraphs)
        logger.info("=" * 60)
        logger.info("STEP 1: Historical Data Pipeline")
        logger.info("=" * 60)
        from src.data.historical_pipeline import build_historical_pipeline
        hitters, pitchers = build_historical_pipeline(force_refetch=force)
        logger.info(f"  Hitters: {len(hitters):,} seasons")
        logger.info(f"  Pitchers: {len(pitchers):,} seasons")
        
        # Step 2: Statcast data (slow — downloads from Baseball Savant)
        if not skip_statcast:
            logger.info("=" * 60)
            logger.info("STEP 2: Statcast Pipeline")
            logger.info("=" * 60)
            from src.data.statcast_pipeline import build_statcast_pipeline
            sc_h, sc_p = build_statcast_pipeline(force_refetch=force)
            logger.info(f"  Statcast hitters: {len(sc_h):,} seasons")
            logger.info(f"  Statcast pitchers: {len(sc_p):,} seasons")
        else:
            logger.info("Skipping Statcast pipeline (--skip-statcast)")
        
        # Step 3: Park factors
        logger.info("=" * 60)
        logger.info("STEP 3: Park Factors")
        logger.info("=" * 60)
        from src.data.park_factors import build_park_factors
        pf = build_park_factors(hitter_df=hitters, force_rebuild=force)
        logger.info(f"  Park factors: {len(pf)} team-years")
    
    # Step 4: Marcel projections
    logger.info("=" * 60)
    logger.info("STEP 4: Marcel Projections")
    logger.info("=" * 60)
    from src.models.marcel import run_marcel
    h_proj, p_proj = run_marcel(force_rebuild=force)
    
    # Summary
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info("=" * 60)
    
    if not h_proj.empty:
        print(f"\n--- Top 15 Hitter Projections by WAR ---")
        top = h_proj.nlargest(15, 'war')[['name', 'team', 'age', 'pa', 'avg', 'hr', 'woba', 'wrc_plus', 'war']]
        pd_opts = {'display.max_columns': 20, 'display.width': 120}
        import pandas as pd
        for k, v in pd_opts.items():
            pd.set_option(k, v)
        print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"))
    
    if not p_proj.empty:
        print(f"\n--- Top 15 Pitcher Projections by WAR ---")
        top = p_proj.nlargest(15, 'war')[['name', 'team', 'age', 'role', 'ip', 'era', 'fip', 'k_9', 'whip', 'war']]
        print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"))


if __name__ == "__main__":
    main()
