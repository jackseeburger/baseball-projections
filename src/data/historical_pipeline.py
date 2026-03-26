"""
Historical data pipeline — downloads season-level batting and pitching stats
from FanGraphs via pybaseball.

Outputs:
  - hitter_seasons.parquet: Full batting stats (2000-present) with advanced metrics
  - pitcher_seasons.parquet: Full pitching stats (2000-present) with advanced metrics
  
These form the backbone of Marcel and all projection models.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

from pybaseball import batting_stats, pitching_stats, playerid_lookup, cache

from src.config import (
    HISTORICAL_START_YEAR,
    CURRENT_YEAR,
    PARQUET_DIR,
)
from src.utils.helpers import setup_logging

logger = setup_logging("historical_pipeline")

# Enable pybaseball caching to avoid re-downloading
cache.enable()


# Columns we want to keep and standardize for hitters
HITTER_COLS = {
    # Identifiers
    'IDfg': 'fg_id',
    'Name': 'name',
    'Team': 'team',
    'Age': 'age',
    'Season': 'year',
    # Counting stats
    'G': 'g',
    'PA': 'pa',
    'AB': 'ab',
    'H': 'h',
    '2B': 'doubles',
    '3B': 'triples',
    'HR': 'hr',
    'R': 'r',
    'RBI': 'rbi',
    'SB': 'sb',
    'CS': 'cs',
    'BB': 'bb',
    'SO': 'so',
    'HBP': 'hbp',
    'SF': 'sf',
    'SH': 'sh',
    'GDP': 'gdp',
    'IBB': 'ibb',
    # Rate stats
    'AVG': 'avg',
    'OBP': 'obp',
    'SLG': 'slg',
    'wOBA': 'woba',
    'wRC+': 'wrc_plus',
    # Batted ball
    'BABIP': 'babip',
    'ISO': 'iso',
    'GB%': 'gb_pct',
    'FB%': 'fb_pct',
    'LD%': 'ld_pct',
    'HR/FB': 'hr_fb',
    # Plate discipline
    'BB%': 'bb_pct',
    'K%': 'k_pct',
    'O-Swing%': 'o_swing_pct',
    'Z-Swing%': 'z_swing_pct',
    'Swing%': 'swing_pct',
    'O-Contact%': 'o_contact_pct',
    'Z-Contact%': 'z_contact_pct',
    'Contact%': 'contact_pct',
    'SwStr%': 'swstr_pct',
    # Value
    'WAR': 'war',
    'Off': 'off',
    'Def': 'def_value',
    'BsR': 'bsr',
}

# Columns for pitchers
PITCHER_COLS = {
    'IDfg': 'fg_id',
    'Name': 'name',
    'Team': 'team',
    'Age': 'age',
    'Season': 'year',
    # Counting
    'W': 'w',
    'L': 'l',
    'SV': 'sv',
    'HLD': 'hld',
    'G': 'g',
    'GS': 'gs',
    'IP': 'ip',
    'H': 'h',
    'R': 'r',
    'ER': 'er',
    'HR': 'hr',
    'BB': 'bb',
    'SO': 'so',
    'HBP': 'hbp',
    # Rate stats
    'ERA': 'era',
    'FIP': 'fip',
    'xFIP': 'xfip',
    'SIERA': 'siera',
    'WHIP': 'whip',
    'K/9': 'k_9',
    'BB/9': 'bb_9',
    'HR/9': 'hr_9',
    'K%': 'k_pct',
    'BB%': 'bb_pct',
    'K-BB%': 'k_bb_pct',
    # Batted ball
    'BABIP': 'babip',
    'GB%': 'gb_pct',
    'FB%': 'fb_pct',
    'LD%': 'ld_pct',
    'HR/FB': 'hr_fb',
    'LOB%': 'lob_pct',
    # Value
    'WAR': 'war',
}


def fetch_batting_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch season batting stats from FanGraphs for a range of years."""
    logger.info(f"Fetching batting stats {start_year}-{end_year}...")
    
    all_dfs = []
    for year in tqdm(range(start_year, end_year + 1), desc="Batting stats"):
        try:
            df = batting_stats(year, qual=0)  # qual=0 = no minimum PA filter
            if df is not None and len(df) > 0:
                df['Season'] = year
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"  Failed for {year}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  Got {len(combined):,} hitter-seasons")
    return combined


def fetch_pitching_stats(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch season pitching stats from FanGraphs for a range of years."""
    logger.info(f"Fetching pitching stats {start_year}-{end_year}...")
    
    all_dfs = []
    for year in tqdm(range(start_year, end_year + 1), desc="Pitching stats"):
        try:
            df = pitching_stats(year, qual=0)
            if df is not None and len(df) > 0:
                df['Season'] = year
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"  Failed for {year}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  Got {len(combined):,} pitcher-seasons")
    return combined


def clean_and_standardize(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename columns to standard names, keeping only the ones we need."""
    # Only keep columns that exist in the data
    available = {k: v for k, v in col_map.items() if k in df.columns}
    missing = set(col_map.keys()) - set(available.keys())
    if missing:
        logger.debug(f"  Missing columns (will be NaN): {missing}")
    
    result = df[list(available.keys())].rename(columns=available)
    
    # Convert percentage strings to floats if needed
    pct_cols = [c for c in result.columns if c.endswith('_pct')]
    for col in pct_cols:
        if result[col].dtype == object:
            result[col] = result[col].str.rstrip('%').astype(float) / 100
    
    return result


def build_historical_pipeline(
    start_year: int = HISTORICAL_START_YEAR,
    end_year: int = CURRENT_YEAR,
    force_refetch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full historical data pipeline.
    
    Returns (hitter_df, pitcher_df) with standardized season stats.
    """
    hitter_output = PARQUET_DIR / "hitter_seasons.parquet"
    pitcher_output = PARQUET_DIR / "pitcher_seasons.parquet"
    
    # Check cache
    if not force_refetch and hitter_output.exists() and pitcher_output.exists():
        logger.info("Loading cached historical parquet files...")
        return pd.read_parquet(hitter_output), pd.read_parquet(pitcher_output)
    
    # Fetch data
    raw_batting = fetch_batting_stats(start_year, end_year)
    raw_pitching = fetch_pitching_stats(start_year, end_year)
    
    # Clean and standardize
    hitter_df = clean_and_standardize(raw_batting, HITTER_COLS) if not raw_batting.empty else pd.DataFrame()
    pitcher_df = clean_and_standardize(raw_pitching, PITCHER_COLS) if not raw_pitching.empty else pd.DataFrame()
    
    # Add derived fields
    if not hitter_df.empty:
        # Singles
        hitter_df['singles'] = hitter_df['h'] - hitter_df['doubles'] - hitter_df['triples'] - hitter_df['hr']
        # K% and BB% from counting stats (more reliable than FG's sometimes)
        hitter_df['k_rate'] = np.where(hitter_df['pa'] > 0, hitter_df['so'] / hitter_df['pa'], np.nan)
        hitter_df['bb_rate'] = np.where(hitter_df['pa'] > 0, hitter_df['bb'] / hitter_df['pa'], np.nan)
        
        hitter_df.to_parquet(hitter_output, index=False)
        logger.info(f"Saved {len(hitter_df):,} hitter-seasons to {hitter_output}")
    
    if not pitcher_df.empty:
        # Innings to outs (IP is in X.1, X.2 format where .1 = 1/3 inning)
        pitcher_df['outs'] = np.floor(pitcher_df['ip']) * 3 + (pitcher_df['ip'] % 1) * 10
        pitcher_df['outs'] = pitcher_df['outs'].astype(int)
        
        pitcher_df.to_parquet(pitcher_output, index=False)
        logger.info(f"Saved {len(pitcher_df):,} pitcher-seasons to {pitcher_output}")
    
    return hitter_df, pitcher_df


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    hitters, pitchers = build_historical_pipeline(force_refetch=force)
    print(f"\nHistorical Pipeline Complete!")
    print(f"  Hitters: {len(hitters):,} player-seasons")
    print(f"  Pitchers: {len(pitchers):,} player-seasons")
    if not hitters.empty:
        print(f"  Years: {hitters['year'].min()}-{hitters['year'].max()}")
        print(f"  Columns: {list(hitters.columns)}")
