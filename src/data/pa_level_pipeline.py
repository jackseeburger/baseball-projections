"""
Plate-appearance level data pipeline for Bayesian models.

Converts pitch-level Statcast data into one row per PA with binary outcomes
and context features. This is the primary input for hierarchical Bayesian models
where each PA is a Bernoulli trial.

Each PA row includes:
  - Outcomes: K, BB, HBP, HR, hit, single, double, triple, in_play_out
  - Batted ball (when applicable): exit_velo, launch_angle, barrel, hard_hit
  - Context: batter_id, pitcher_id, year, month, park, bat_side, pitch_hand
  - Count at PA end, pitches seen in PA
  - Platoon split indicator

Outputs:
  - pa_outcomes.parquet: One row per PA (~6M+ rows for 2015-2025)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import (
    STATCAST_START_YEAR,
    CURRENT_YEAR,
    PARQUET_DIR,
    RAW_DIR,
)
from src.utils.helpers import setup_logging

logger = setup_logging("pa_level_pipeline")


# Event types that end a PA
PA_ENDING_EVENTS = {
    # Strikeouts
    'strikeout', 'strikeout_double_play',
    # Walks
    'walk',
    # HBP
    'hit_by_pitch',
    # In-play outcomes
    'single', 'double', 'triple', 'home_run',
    'field_out', 'force_out', 'grounded_into_double_play',
    'fielders_choice', 'fielders_choice_out',
    'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'sac_bunt_double_play',
    'double_play', 'triple_play',
    'field_error',
    # Other PA-ending
    'catcher_interf',
}


def process_year_to_pa(year: int) -> pd.DataFrame:
    """Convert a year's pitch-level data to PA-level outcomes.
    
    Args:
        year: Season year to process
        
    Returns:
        DataFrame with one row per plate appearance
    """
    raw_file = RAW_DIR / f"statcast_{year}.parquet"
    if not raw_file.exists():
        logger.warning(f"No raw Statcast file for {year}")
        return pd.DataFrame()
    
    logger.info(f"Processing {year} pitch data to PA level...")
    pitches = pd.read_parquet(raw_file)
    
    if pitches.empty:
        return pd.DataFrame()
    
    # Filter to PA-ending pitches only (each = one PA)
    # The 'events' column is non-null only on the last pitch of a PA
    pa_pitches = pitches[pitches['events'].notna() & pitches['events'].isin(PA_ENDING_EVENTS)].copy()
    
    logger.info(f"  {len(pitches):,} pitches → {len(pa_pitches):,} plate appearances")
    
    # Also compute pitches-per-PA by counting all pitches in each at_bat
    if 'at_bat_number' in pitches.columns and 'game_pk' in pitches.columns:
        pitch_counts = pitches.groupby(['game_pk', 'at_bat_number']).size().reset_index(name='pitches_in_pa')
        pa_pitches = pa_pitches.merge(pitch_counts, on=['game_pk', 'at_bat_number'], how='left')
    
    # Build PA outcome DataFrame
    pa = pd.DataFrame()
    
    # Identifiers
    pa['batter_id'] = pa_pitches['batter'].astype(int).values
    pa['pitcher_id'] = pa_pitches['pitcher'].astype(int).values
    pa['year'] = year
    pa['game_pk'] = pa_pitches['game_pk'].values
    
    if 'game_date' in pa_pitches.columns:
        pa['game_date'] = pd.to_datetime(pa_pitches['game_date'].values)
        pa['month'] = pa['game_date'].dt.month
    
    # Context
    if 'home_team' in pa_pitches.columns:
        pa['home_team'] = pa_pitches['home_team'].values
    if 'away_team' in pa_pitches.columns:
        pa['away_team'] = pa_pitches['away_team'].values
    if 'stand' in pa_pitches.columns:
        pa['bat_side'] = pa_pitches['stand'].values  # R or L
    if 'p_throws' in pa_pitches.columns:
        pa['pitch_hand'] = pa_pitches['p_throws'].values  # R or L
    
    # Platoon advantage (same-side = disadvantage, opposite = advantage)
    if 'stand' in pa_pitches.columns and 'p_throws' in pa_pitches.columns:
        pa['platoon_adv'] = (pa_pitches['stand'].values != pa_pitches['p_throws'].values).astype(int)
    
    if 'at_bat_number' in pa_pitches.columns:
        pa['at_bat_number'] = pa_pitches['at_bat_number'].values
    
    if 'pitches_in_pa' in pa_pitches.columns:
        pa['pitches_in_pa'] = pa_pitches['pitches_in_pa'].values
    
    # Binary outcomes (the core data for Bayesian models)
    events = pa_pitches['events'].values
    
    pa['is_k'] = np.isin(events, ['strikeout', 'strikeout_double_play']).astype(np.int8)
    pa['is_bb'] = (events == 'walk').astype(np.int8)
    pa['is_hbp'] = (events == 'hit_by_pitch').astype(np.int8)
    pa['is_hit'] = np.isin(events, ['single', 'double', 'triple', 'home_run']).astype(np.int8)
    pa['is_single'] = (events == 'single').astype(np.int8)
    pa['is_double'] = (events == 'double').astype(np.int8)
    pa['is_triple'] = (events == 'triple').astype(np.int8)
    pa['is_hr'] = (events == 'home_run').astype(np.int8)
    
    # Ball in play (not K, BB, HBP)
    pa['is_bip'] = (~np.isin(events, [
        'strikeout', 'strikeout_double_play', 'walk', 'hit_by_pitch', 'catcher_interf'
    ])).astype(np.int8)
    
    # In-play out
    pa['is_inplay_out'] = (pa['is_bip'] & ~pa['is_hit']).astype(np.int8)
    
    # Sac fly/bunt (productive outs)
    pa['is_sac'] = np.isin(events, [
        'sac_fly', 'sac_bunt', 'sac_fly_double_play', 'sac_bunt_double_play'
    ]).astype(np.int8)
    
    # GIDP
    pa['is_gidp'] = np.isin(events, [
        'grounded_into_double_play', 'double_play'
    ]).astype(np.int8)
    
    # Batted ball data (only available on balls in play)
    if 'launch_speed' in pa_pitches.columns:
        pa['exit_velo'] = pa_pitches['launch_speed'].values
    if 'launch_angle' in pa_pitches.columns:
        pa['launch_angle'] = pa_pitches['launch_angle'].values
    
    # Hard hit (>= 95 mph) — NaN for non-BIP
    if 'launch_speed' in pa_pitches.columns:
        speed = pa_pitches['launch_speed'].values
        hard_hit = pd.array(np.where(pd.isna(speed), pd.NA, (speed >= 95).astype(int)), dtype='Int8')
        pa['is_hard_hit'] = hard_hit
    
    # Barrel — NaN for non-BIP
    if 'barrel' in pa_pitches.columns:
        barrel_vals = pa_pitches['barrel'].values
        pa['is_barrel'] = pd.array(np.where(pd.isna(barrel_vals), pd.NA, barrel_vals), dtype='Int8')
    
    # Batted ball type — NaN for non-BIP
    if 'bb_type' in pa_pitches.columns:
        bb_type = pa_pitches['bb_type'].values
        for bb_name, bb_val in [('is_ground_ball', 'ground_ball'), ('is_line_drive', 'line_drive'),
                                 ('is_fly_ball', 'fly_ball'), ('is_popup', 'popup')]:
            pa[bb_name] = pd.array(np.where(pd.isna(bb_type), pd.NA, (bb_type == bb_val).astype(int)), dtype='Int8')
    
    # Expected stats from Statcast
    for col, new_name in [
        ('estimated_ba_using_speedangle', 'xba'),
        ('estimated_slg_using_speedangle', 'xslg'),
        ('estimated_woba_using_speedangle', 'xwoba'),
    ]:
        if col in pa_pitches.columns:
            pa[new_name] = pa_pitches[col].values
    
    # Pitch info on the final pitch
    if 'release_speed' in pa_pitches.columns:
        pa['last_pitch_velo'] = pa_pitches['release_speed'].values
    if 'pitch_type' in pa_pitches.columns:
        pa['last_pitch_type'] = pa_pitches['pitch_type'].values
    
    # Count at end of PA
    if 'balls' in pa_pitches.columns:
        pa['final_balls'] = pa_pitches['balls'].values
    if 'strikes' in pa_pitches.columns:
        pa['final_strikes'] = pa_pitches['strikes'].values
    
    # Outs when PA occurred
    if 'outs_when_up' in pa_pitches.columns:
        pa['outs_when_up'] = pa_pitches['outs_when_up'].values
    
    # Runners on base
    if 'on_1b' in pa_pitches.columns:
        pa['runner_1b'] = pa_pitches['on_1b'].notna().astype(np.int8).values
    if 'on_2b' in pa_pitches.columns:
        pa['runner_2b'] = pa_pitches['on_2b'].notna().astype(np.int8).values
    if 'on_3b' in pa_pitches.columns:
        pa['runner_3b'] = pa_pitches['on_3b'].notna().astype(np.int8).values
    
    # Score differential (for leverage context)
    if 'bat_score' in pa_pitches.columns and 'fld_score' in pa_pitches.columns:
        pa['score_diff'] = (pa_pitches['bat_score'] - pa_pitches['fld_score']).values
    
    # Inning
    if 'inning' in pa_pitches.columns:
        pa['inning'] = pa_pitches['inning'].values
    if 'inning_topbot' in pa_pitches.columns:
        pa['is_home'] = (pa_pitches['inning_topbot'] == 'Bot').astype(np.int8).values
    
    return pa


def build_pa_level_data(
    start_year: int = STATCAST_START_YEAR,
    end_year: int = CURRENT_YEAR,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build the full PA-level dataset from raw Statcast pitch data.
    
    Returns:
        DataFrame with ~6M+ rows (one per PA, 2015-2025)
    """
    output_path = PARQUET_DIR / "pa_outcomes.parquet"
    
    if not force_rebuild and output_path.exists():
        logger.info("Loading cached PA-level data...")
        return pd.read_parquet(output_path)
    
    all_pa = []
    for year in tqdm(range(start_year, end_year + 1), desc="PA-level processing"):
        pa_df = process_year_to_pa(year)
        if not pa_df.empty:
            all_pa.append(pa_df)
    
    if not all_pa:
        logger.error("No PA data generated!")
        return pd.DataFrame()
    
    combined = pd.concat(all_pa, ignore_index=True)
    
    # Optimize dtypes for storage
    int8_cols = [c for c in combined.columns if c.startswith('is_') or c in ['platoon_adv', 'runner_1b', 'runner_2b', 'runner_3b']]
    for col in int8_cols:
        if col in combined.columns:
            combined[col] = combined[col].astype('Int8')  # nullable int8
    
    # Save
    combined.to_parquet(output_path, index=False)
    
    # Summary
    n_pa = len(combined)
    n_batters = combined['batter_id'].nunique()
    n_pitchers = combined['pitcher_id'].nunique()
    years = f"{combined['year'].min()}-{combined['year'].max()}"
    
    logger.info(f"\nPA-Level Dataset Complete:")
    logger.info(f"  Total PAs: {n_pa:,}")
    logger.info(f"  Unique batters: {n_batters:,}")
    logger.info(f"  Unique pitchers: {n_pitchers:,}")
    logger.info(f"  Years: {years}")
    logger.info(f"  Columns: {len(combined.columns)}")
    logger.info(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    
    # Outcome rates
    logger.info(f"\n  K rate:  {combined['is_k'].mean():.3f}")
    logger.info(f"  BB rate: {combined['is_bb'].mean():.3f}")
    logger.info(f"  HR rate: {combined['is_hr'].mean():.3f}")
    logger.info(f"  Hit rate:{combined['is_hit'].mean():.3f}")
    logger.info(f"  BABIP:   {combined.loc[combined['is_bip']==1, 'is_hit'].mean():.3f}")
    
    return combined


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    pa_df = build_pa_level_data(force_rebuild=force)
    
    print(f"\n{'='*60}")
    print(f"PA-Level Data Summary")
    print(f"{'='*60}")
    print(f"Total PAs: {len(pa_df):,}")
    print(f"Columns: {list(pa_df.columns)}")
    print(f"\nOutcome rates:")
    for col in ['is_k', 'is_bb', 'is_hbp', 'is_hit', 'is_hr', 'is_single', 'is_double', 'is_triple']:
        if col in pa_df.columns:
            print(f"  {col:15s}: {pa_df[col].mean():.4f}")
