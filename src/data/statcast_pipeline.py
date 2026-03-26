"""
Statcast data pipeline — downloads and processes pitch-level and season-aggregate
Statcast data from Baseball Savant via pybaseball.

Outputs:
  - statcast_hitters.parquet: Hitter season aggregates with Statcast metrics
  - statcast_pitchers.parquet: Pitcher season aggregates with Statcast metrics
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pybaseball import (
    statcast,
    batting_stats,
    pitching_stats,
)

from src.config import (
    STATCAST_START_YEAR,
    CURRENT_YEAR,
    PARQUET_DIR,
    RAW_DIR,
)
from src.utils.helpers import setup_logging

logger = setup_logging("statcast_pipeline")


def fetch_statcast_season(year: int) -> pd.DataFrame:
    """Fetch full-season Statcast pitch-level data for a given year."""
    logger.info(f"Fetching Statcast pitch-level data for {year}...")
    # pybaseball statcast() fetches from Baseball Savant
    # We fetch in small 2-week chunks to avoid timeouts and CSV parse errors
    import datetime
    
    season_start = datetime.date(year, 3, 20)  # Spring training ends ~late March
    season_end = datetime.date(year, 11, 5)     # World Series typically ends early Nov
    
    dfs = []
    chunk_start = season_start
    chunk_days = 14  # 2-week chunks for reliability
    
    while chunk_start < season_end:
        chunk_end = min(chunk_start + datetime.timedelta(days=chunk_days - 1), season_end)
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")
        
        try:
            df = statcast(start_dt=start_str, end_dt=end_str)
            if df is not None and len(df) > 0:
                dfs.append(df)
                logger.debug(f"  {start_str} to {end_str}: {len(df):,} pitches")
        except Exception as e:
            logger.warning(f"  Chunk {start_str} to {end_str} failed: {e}")
            # Retry with even smaller chunks (1 week)
            mid = chunk_start + datetime.timedelta(days=chunk_days // 2)
            for sub_start, sub_end in [(chunk_start, mid - datetime.timedelta(days=1)), (mid, chunk_end)]:
                try:
                    df = statcast(start_dt=sub_start.strftime("%Y-%m-%d"), end_dt=sub_end.strftime("%Y-%m-%d"))
                    if df is not None and len(df) > 0:
                        dfs.append(df)
                except Exception as e2:
                    logger.warning(f"  Sub-chunk {sub_start} to {sub_end} also failed: {e2}")
        
        chunk_start = chunk_end + datetime.timedelta(days=1)
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"  Got {len(combined):,} pitches for {year}")
        return combined
    else:
        logger.warning(f"  No data for {year}")
        return pd.DataFrame()


def aggregate_hitter_statcast(pitch_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate pitch-level Statcast data into hitter season stats.
    
    Key metrics computed:
    - exit_velo_avg, exit_velo_max: Batted ball velocity
    - launch_angle_avg: Average launch angle
    - barrel_pct: Barrel rate (optimal exit velo + launch angle)
    - hard_hit_pct: % of batted balls >= 95 mph
    - sweet_spot_pct: % of batted balls with 8-32 degree launch angle
    - xba, xslg, xwoba: Expected stats from Statcast
    - whiff_pct: Swing-and-miss rate
    - chase_pct: Out-of-zone swing rate
    - sprint_speed: Average sprint speed
    """
    if pitch_df.empty:
        return pd.DataFrame()
    
    # Filter to batted ball events for batted ball metrics
    batted = pitch_df[pitch_df['type'] == 'X'].copy()
    
    # Hitter-level aggregation
    hitter_stats = []
    
    # Group by batter
    for batter_id, group in pitch_df.groupby('batter'):
        batter_batted = batted[batted['batter'] == batter_id]
        
        stats = {
            'player_id': int(batter_id),
            'year': year,
            'pitches_seen': len(group),
        }
        
        # Batted ball metrics (only if enough batted balls)
        if len(batter_batted) >= 30:
            stats['exit_velo_avg'] = batter_batted['launch_speed'].mean()
            stats['exit_velo_max'] = batter_batted['launch_speed'].max()
            stats['launch_angle_avg'] = batter_batted['launch_angle'].mean()
            
            # Hard hit: >= 95 mph
            valid_speed = batter_batted['launch_speed'].dropna()
            if len(valid_speed) > 0:
                stats['hard_hit_pct'] = (valid_speed >= 95).mean()
            
            # Barrel: optimal combination (95+ mph, 26-30 degree LA at peak)
            if 'barrel' in batter_batted.columns:
                stats['barrel_pct'] = batter_batted['barrel'].mean() if batter_batted['barrel'].notna().any() else np.nan
            
            # Sweet spot: 8-32 degrees
            valid_angle = batter_batted['launch_angle'].dropna()
            if len(valid_angle) > 0:
                stats['sweet_spot_pct'] = ((valid_angle >= 8) & (valid_angle <= 32)).mean()
        
        # Expected stats
        for col in ['estimated_ba_using_speedangle', 'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle']:
            short_name = col.replace('estimated_', 'x').replace('_using_speedangle', '')
            if col in batter_batted.columns:
                vals = batter_batted[col].dropna()
                stats[short_name] = vals.mean() if len(vals) > 0 else np.nan
        
        # Plate discipline
        swings = group[group['description'].isin([
            'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
            'foul_tip', 'foul_bunt', 'bunt_foul_tip', 'missed_bunt'
        ])]
        whiffs = group[group['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
        
        if len(swings) > 0:
            stats['whiff_pct'] = len(whiffs) / len(swings)
        
        # Chase rate (swings outside zone)
        out_of_zone = group[group['zone'].isin([11, 12, 13, 14]) | group['zone'].isna()]
        oz_swings = out_of_zone[out_of_zone['description'].isin([
            'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
            'foul_tip', 'foul_bunt'
        ])]
        if len(out_of_zone) > 0:
            stats['chase_pct'] = len(oz_swings) / len(out_of_zone)
        
        hitter_stats.append(stats)
    
    return pd.DataFrame(hitter_stats)


def aggregate_pitcher_statcast(pitch_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Aggregate pitch-level Statcast data into pitcher season stats.
    
    Key metrics:
    - avg_velocity, max_velocity: Fastball velocity
    - spin_rate_avg: Average spin rate
    - extension_avg: Release extension
    - induced_vert_break, horz_break: Pitch movement
    - xba_against, xslg_against, xwoba_against: Expected stats against
    - whiff_pct, chase_pct: Swing-and-miss, chase rates
    - stuff_plus_proxy: Crude stuff metric (K-related indicators)
    """
    if pitch_df.empty:
        return pd.DataFrame()
    
    batted = pitch_df[pitch_df['type'] == 'X'].copy()
    pitcher_stats = []
    
    for pitcher_id, group in pitch_df.groupby('pitcher'):
        pitcher_batted = batted[batted['pitcher'] == pitcher_id]
        
        stats = {
            'player_id': int(pitcher_id),
            'year': year,
            'pitches_thrown': len(group),
        }
        
        # Velocity (fastballs: FF, SI, FC)
        fastballs = group[group['pitch_type'].isin(['FF', 'SI', 'FC'])]
        if len(fastballs) > 0:
            stats['avg_fastball_velo'] = fastballs['release_speed'].mean()
            stats['max_fastball_velo'] = fastballs['release_speed'].max()
        
        # Overall velocity
        stats['avg_velocity'] = group['release_speed'].mean()
        
        # Spin rate
        if 'release_spin_rate' in group.columns:
            stats['spin_rate_avg'] = group['release_spin_rate'].mean()
        
        # Extension
        if 'release_extension' in group.columns:
            stats['extension_avg'] = group['release_extension'].mean()
        
        # Pitch movement
        if 'pfx_x' in group.columns:
            stats['horz_break_avg'] = group['pfx_x'].mean()
        if 'pfx_z' in group.columns:
            stats['vert_break_avg'] = group['pfx_z'].mean()
        
        # Expected stats against
        if len(pitcher_batted) >= 30:
            for col in ['estimated_ba_using_speedangle', 'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle']:
                short_name = col.replace('estimated_', 'x').replace('_using_speedangle', '_against')
                if col in pitcher_batted.columns:
                    vals = pitcher_batted[col].dropna()
                    stats[short_name] = vals.mean() if len(vals) > 0 else np.nan
        
        # Whiff rate
        swings = group[group['description'].isin([
            'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
            'foul_tip', 'foul_bunt'
        ])]
        whiffs = group[group['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
        if len(swings) > 0:
            stats['whiff_pct'] = len(whiffs) / len(swings)
        
        # Chase rate
        out_of_zone = group[group['zone'].isin([11, 12, 13, 14]) | group['zone'].isna()]
        oz_swings = out_of_zone[out_of_zone['description'].isin([
            'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip'
        ])]
        if len(out_of_zone) > 0:
            stats['chase_pct'] = len(oz_swings) / len(out_of_zone)
        
        # Pitch mix (% of each pitch type)
        pitch_counts = group['pitch_type'].value_counts(normalize=True)
        for pt in ['FF', 'SI', 'SL', 'CH', 'CU', 'FC', 'KC', 'FS', 'ST']:
            stats[f'pct_{pt.lower()}'] = pitch_counts.get(pt, 0.0)
        
        pitcher_stats.append(stats)
    
    return pd.DataFrame(pitcher_stats)


def build_statcast_pipeline(
    start_year: int = STATCAST_START_YEAR,
    end_year: int = CURRENT_YEAR,
    force_refetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Statcast pipeline: fetch, aggregate, save to parquet.
    
    Returns (hitter_df, pitcher_df) with season-level Statcast aggregates.
    """
    hitter_output = PARQUET_DIR / "statcast_hitters.parquet"
    pitcher_output = PARQUET_DIR / "statcast_pitchers.parquet"
    
    # Check cache
    if not force_refetch and hitter_output.exists() and pitcher_output.exists():
        logger.info("Loading cached Statcast parquet files...")
        return pd.read_parquet(hitter_output), pd.read_parquet(pitcher_output)
    
    all_hitters = []
    all_pitchers = []
    
    for year in tqdm(range(start_year, end_year + 1), desc="Statcast years"):
        # Check for cached raw data
        raw_file = RAW_DIR / f"statcast_{year}.parquet"
        
        if raw_file.exists() and not force_refetch:
            logger.info(f"Loading cached raw Statcast for {year}...")
            pitch_df = pd.read_parquet(raw_file)
        else:
            pitch_df = fetch_statcast_season(year)
            if not pitch_df.empty:
                pitch_df.to_parquet(raw_file, index=False)
        
        if not pitch_df.empty:
            hitter_agg = aggregate_hitter_statcast(pitch_df, year)
            pitcher_agg = aggregate_pitcher_statcast(pitch_df, year)
            all_hitters.append(hitter_agg)
            all_pitchers.append(pitcher_agg)
    
    # Combine all years
    hitter_df = pd.concat(all_hitters, ignore_index=True) if all_hitters else pd.DataFrame()
    pitcher_df = pd.concat(all_pitchers, ignore_index=True) if all_pitchers else pd.DataFrame()
    
    # Save
    if not hitter_df.empty:
        hitter_df.to_parquet(hitter_output, index=False)
        logger.info(f"Saved {len(hitter_df):,} hitter-seasons to {hitter_output}")
    
    if not pitcher_df.empty:
        pitcher_df.to_parquet(pitcher_output, index=False)
        logger.info(f"Saved {len(pitcher_df):,} pitcher-seasons to {pitcher_output}")
    
    return hitter_df, pitcher_df


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    hitters, pitchers = build_statcast_pipeline(force_refetch=force)
    print(f"\nStatcast Pipeline Complete!")
    print(f"  Hitters: {len(hitters):,} player-seasons")
    print(f"  Pitchers: {len(pitchers):,} player-seasons")
