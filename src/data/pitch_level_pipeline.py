"""
Pitch-Level Pipeline for Pitcher Stuff Model

Processes raw Statcast data into a clean pitch-level dataset optimized for
the hierarchical Bayesian Stuff model. Each row = one pitch with physical
characteristics and outcome.

The Stuff model learns: given pitch physical properties (velocity, movement,
spin, release point, location) → what is the expected whiff rate, called
strike rate, and overall "stuff" quality per pitch type?

Output columns:
- Identifiers: pitcher, batter, game_pk, game_date, game_year, at_bat_number,
               pitch_number
- Pitch type: pitch_type, pitch_name
- Physical characteristics (model inputs):
    - release_speed (velocity)
    - pfx_x, pfx_z (horizontal/vertical movement in inches)
    - release_spin_rate, spin_axis
    - release_pos_x, release_pos_y, release_pos_z (release point)
    - release_extension
    - arm_angle
    - ax, ay, az (acceleration components)
    - effective_speed
- Location: plate_x, plate_z, zone, sz_top, sz_bot
- Outcome (model targets):
    - is_whiff, is_called_strike, is_ball, is_foul, is_in_play
    - is_swing, is_contact (derived)
    - description (raw Statcast description)
- Batted ball (when in play): launch_speed, launch_angle, bb_type
- Context: stand (L/R), p_throws (L/R), balls, strikes, outs_when_up, inning
"""

import pandas as pd
import numpy as np
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pitch descriptions and their classifications
WHIFF_DESCRIPTIONS = {
    'swinging_strike', 'swinging_strike_blocked', 'missed_bunt',
}

CALLED_STRIKE_DESCRIPTIONS = {
    'called_strike',
}

FOUL_DESCRIPTIONS = {
    'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip',
}

BALL_DESCRIPTIONS = {
    'ball', 'blocked_ball', 'intent_ball', 'pitchout',
}

IN_PLAY_DESCRIPTIONS = {
    'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score',
}

SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | FOUL_DESCRIPTIONS | IN_PLAY_DESCRIPTIONS

# Physical characteristic columns needed for the Stuff model
STUFF_FEATURES = [
    'release_speed',
    'pfx_x', 'pfx_z',               # horizontal/vertical movement
    'release_spin_rate', 'spin_axis',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'release_extension',
    'arm_angle',
    'ax', 'ay', 'az',               # acceleration
    'effective_speed',
    'plate_x', 'plate_z',           # location at plate
    'zone',                          # strike zone region (1-9 strike, 11-14 ball)
    'sz_top', 'sz_bot',             # batter-specific zone boundaries
]


def process_year(year: int, data_dir: str = 'data/raw') -> pd.DataFrame:
    """Process a single year of Statcast data into pitch-level observations."""
    filepath = os.path.join(data_dir, f'statcast_{year}.parquet')
    if not os.path.exists(filepath):
        logger.warning(f'No data file for {year}: {filepath}')
        return pd.DataFrame()

    logger.info(f'Loading {year} Statcast data...')
    df = pd.read_parquet(filepath)
    logger.info(f'  {len(df):,} pitches loaded')

    # Filter to regular season
    df = df[df['game_type'] == 'R'].copy()
    logger.info(f'  {len(df):,} regular season pitches')

    # Drop pitches with no pitch type (rare tracking failures)
    df = df[df['pitch_type'].notna() & (df['pitch_type'] != '')].copy()
    # Also drop intentional balls and pitchouts (not real "stuff")
    df = df[~df['pitch_type'].isin(['PO', 'IN'])].copy()
    logger.info(f'  {len(df):,} after filtering nulls/pitchouts')

    # --- Outcome flags ---
    desc = df['description']
    df['is_whiff'] = desc.isin(WHIFF_DESCRIPTIONS).astype(np.int8)
    df['is_called_strike'] = desc.isin(CALLED_STRIKE_DESCRIPTIONS).astype(np.int8)
    df['is_foul'] = desc.isin(FOUL_DESCRIPTIONS).astype(np.int8)
    df['is_ball'] = desc.isin(BALL_DESCRIPTIONS).astype(np.int8)
    df['is_in_play'] = desc.isin(IN_PLAY_DESCRIPTIONS).astype(np.int8)
    df['is_swing'] = desc.isin(SWING_DESCRIPTIONS).astype(np.int8)
    df['is_contact'] = (df['is_swing'] & ~df['is_whiff']).astype(np.int8)
    df['is_strike'] = (df['is_whiff'] | df['is_called_strike'] | df['is_foul']).astype(np.int8)

    # --- Select columns ---
    output_cols = [
        # Identifiers
        'pitcher', 'batter', 'game_pk', 'game_date', 'game_year',
        'at_bat_number', 'pitch_number',
        # Pitch type
        'pitch_type', 'pitch_name',
        # Pitcher/batter info
        'p_throws', 'stand',
        # Physical characteristics (stuff model inputs)
        *STUFF_FEATURES,
        # Outcome flags (stuff model targets)
        'is_whiff', 'is_called_strike', 'is_foul', 'is_ball', 'is_in_play',
        'is_swing', 'is_contact', 'is_strike',
        'description',
        # Batted ball (only populated on contact)
        'launch_speed', 'launch_angle', 'bb_type',
        # Count / game state
        'balls', 'strikes', 'outs_when_up', 'inning',
        # Player metadata
        'player_name',  # pitcher name
    ]

    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()

    # --- Drop rows where core physical features are all null ---
    # (tracking failures where we have no pitch data)
    core_features = ['release_speed', 'pfx_x', 'pfx_z']
    core_mask = result[core_features].notna().all(axis=1)
    result = result[core_mask].copy()
    logger.info(f'  {len(result):,} pitches with valid tracking data')

    return result


def build_pitch_dataset(
    years: list[int] = None,
    data_dir: str = 'data/raw',
    output_path: str = 'data/parquet/pitch_level',
) -> pd.DataFrame:
    """Build the full pitch-level dataset across all years."""

    if years is None:
        files = sorted(glob.glob(os.path.join(data_dir, 'statcast_*.parquet')))
        years = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]

    logger.info(f'Building pitch-level dataset for years: {years}')

    # Process one year at a time and write per-year parquet files to avoid OOM
    # on memory-constrained systems (2GB VPS). Output is a directory of
    # year-partitioned parquet files that can be read with:
    #   pd.read_parquet('data/parquet/pitch_level/')
    os.makedirs(output_path, exist_ok=True)
    total_pitches = 0
    years_processed = 0

    for year in sorted(years):
        year_df = process_year(year, data_dir)
        if len(year_df) == 0:
            continue

        year_path = os.path.join(output_path, f'pitch_level_{year}.parquet')
        year_df.to_parquet(year_path, index=False)

        total_pitches += len(year_df)
        years_processed += 1
        mb = os.path.getsize(year_path) / 1e6
        logger.info(f'  Saved {year}: {len(year_df):,} pitches ({mb:.1f} MB)')
        logger.info(f'  Running total: {total_pitches:,} pitches from {years_processed} seasons')

        # Free memory
        del year_df
        import gc; gc.collect()

    if total_pitches == 0:
        raise ValueError('No pitch data generated!')

    total_mb = sum(
        os.path.getsize(os.path.join(output_path, f)) / 1e6
        for f in os.listdir(output_path) if f.endswith('.parquet')
    )
    logger.info(f'\nTotal: {total_pitches:,} pitches across {years_processed} seasons')
    logger.info(f'  Saved to {output_path}/ ({total_mb:.1f} MB total)')


if __name__ == '__main__':
    build_pitch_dataset()
