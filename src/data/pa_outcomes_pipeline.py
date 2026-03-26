"""
Plate Appearance Outcomes Pipeline

Rolls up pitch-level Statcast data into PA-level observations for
hierarchical Bayesian hitter models. Each row = one plate appearance
with the final outcome and contextual features.

Output columns:
- Identifiers: batter, pitcher, game_pk, game_date, game_year, at_bat_number
- Outcome: event (strikeout, walk, single, double, triple, home_run, etc.)
- Outcome flags: is_k, is_bb, is_hit, is_hr, is_single, is_double, is_triple,
                  is_out, is_in_play, reached_base
- Batted ball (when in play): launch_speed, launch_angle, hit_distance,
                               bb_type (ground_ball, fly_ball, line_drive, popup)
- Context: stand (L/R), p_throws (L/R), balls, strikes (final count),
           outs_when_up, inning, bat_score_diff, on_1b, on_2b, on_3b
- Game info: home_team, away_team
- Pitch summary: pitches_seen (count of pitches in PA),
                  avg_release_speed, max_release_speed
- Expected stats: estimated_woba (xwOBA), estimated_ba (xBA)
- wOBA: woba_value, woba_denom
"""

import pandas as pd
import numpy as np
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PA-ending events that we want to capture
# Events column is only populated on the final pitch of each PA
PA_EVENTS = {
    # Strikeouts
    'strikeout', 'strikeout_double_play',
    # Walks / HBP
    'walk', 'hit_by_pitch', 'intent_walk',
    # Hits
    'single', 'double', 'triple', 'home_run',
    # Outs
    'field_out', 'grounded_into_double_play', 'force_out',
    'fielders_choice', 'fielders_choice_out', 'double_play',
    'field_error', 'sac_fly', 'sac_bunt', 'sac_fly_double_play',
    'sac_bunt_double_play', 'triple_play',
    # Other
    'catcher_interf',
}

# Events that count as reaching base
REACHED_BASE_EVENTS = {
    'single', 'double', 'triple', 'home_run',
    'walk', 'hit_by_pitch', 'intent_walk',
    'field_error', 'catcher_interf',
}

HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}


def process_year(year: int, data_dir: str = 'data/raw') -> pd.DataFrame:
    """Process a single year of Statcast data into PA-level outcomes."""
    filepath = os.path.join(data_dir, f'statcast_{year}.parquet')
    if not os.path.exists(filepath):
        logger.warning(f'No data file for {year}: {filepath}')
        return pd.DataFrame()

    logger.info(f'Loading {year} Statcast data...')
    df = pd.read_parquet(filepath)
    logger.info(f'  {len(df):,} pitches loaded')

    # Filter to regular season only
    df = df[df['game_type'] == 'R'].copy()
    logger.info(f'  {len(df):,} regular season pitches')

    # Identify the last pitch of each PA (where events is populated)
    pa_endings = df[df['events'].notna() & df['events'].isin(PA_EVENTS)].copy()
    logger.info(f'  {len(pa_endings):,} plate appearances identified')

    # --- Pitch-level aggregations per PA ---
    # Pre-compute boolean columns to avoid slow lambda aggregations
    swing_descs = {
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip',
        'foul_bunt', 'hit_into_play', 'hit_into_play_no_out',
        'hit_into_play_score', 'missed_bunt'
    }
    whiff_descs = {'swinging_strike', 'swinging_strike_blocked', 'foul_tip'}

    df['_is_swing'] = df['description'].isin(swing_descs).astype(np.int8)
    df['_is_whiff'] = df['description'].isin(whiff_descs).astype(np.int8)
    df['_is_cstr'] = (df['description'] == 'called_strike').astype(np.int8)

    pa_group = df.groupby(['game_pk', 'at_bat_number', 'batter'])
    pitch_aggs = pa_group.agg(
        pitches_seen=('pitch_type', 'count'),
        avg_release_speed=('release_speed', 'mean'),
        max_release_speed=('release_speed', 'max'),
        swings=('_is_swing', 'sum'),
        whiffs=('_is_whiff', 'sum'),
        called_strikes=('_is_cstr', 'sum'),
    ).reset_index()

    # Merge PA endings with pitch aggregations
    pa_df = pa_endings.merge(
        pitch_aggs,
        on=['game_pk', 'at_bat_number', 'batter'],
        how='left'
    )

    # --- Build outcome flags ---
    event = pa_df['events']
    pa_df['is_k'] = event.isin({'strikeout', 'strikeout_double_play'}).astype(np.int8)
    pa_df['is_bb'] = event.isin({'walk', 'intent_walk'}).astype(np.int8)
    pa_df['is_hbp'] = (event == 'hit_by_pitch').astype(np.int8)
    pa_df['is_single'] = (event == 'single').astype(np.int8)
    pa_df['is_double'] = (event == 'double').astype(np.int8)
    pa_df['is_triple'] = (event == 'triple').astype(np.int8)
    pa_df['is_hr'] = (event == 'home_run').astype(np.int8)
    pa_df['is_hit'] = event.isin(HIT_EVENTS).astype(np.int8)
    pa_df['is_in_play'] = pa_df['bb_type'].notna().astype(np.int8)
    pa_df['reached_base'] = event.isin(REACHED_BASE_EVENTS).astype(np.int8)
    pa_df['is_out'] = (~event.isin(REACHED_BASE_EVENTS)).astype(np.int8)

    # --- Baserunner state as binary flags ---
    pa_df['runner_on_1b'] = pa_df['on_1b'].notna().astype(np.int8)
    pa_df['runner_on_2b'] = pa_df['on_2b'].notna().astype(np.int8)
    pa_df['runner_on_3b'] = pa_df['on_3b'].notna().astype(np.int8)

    # --- Select and rename final columns ---
    output_cols = [
        # Identifiers
        'batter', 'pitcher', 'game_pk', 'game_date', 'game_year', 'at_bat_number',
        # Event
        'events',
        # Outcome flags
        'is_k', 'is_bb', 'is_hbp', 'is_hit', 'is_hr', 'is_single', 'is_double',
        'is_triple', 'is_in_play', 'is_out', 'reached_base',
        # Batted ball data (null when not in play)
        'launch_speed', 'launch_angle', 'hit_distance_sc', 'bb_type',
        'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle',
        'estimated_woba_using_speedangle',
        # Context
        'stand', 'p_throws', 'balls', 'strikes', 'outs_when_up',
        'inning', 'inning_topbot', 'bat_score_diff',
        'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
        # Game info
        'home_team', 'away_team',
        # Pitch summary
        'pitches_seen', 'avg_release_speed', 'max_release_speed',
        'swings', 'whiffs', 'called_strikes',
        # Value
        'woba_value', 'woba_denom',
    ]

    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in pa_df.columns]
    result = pa_df[output_cols].copy()

    # Rename for clarity
    result = result.rename(columns={
        'events': 'event',
        'hit_distance_sc': 'hit_distance',
        'estimated_ba_using_speedangle': 'xba',
        'estimated_slg_using_speedangle': 'xslg',
        'estimated_woba_using_speedangle': 'xwoba',
    })

    logger.info(f'  Output: {len(result):,} PAs with {result.columns.size} columns')
    return result


def build_pa_dataset(
    years: list[int] = None,
    data_dir: str = 'data/raw',
    output_path: str = 'data/parquet/pa_outcomes',
):
    """Build the full PA-level dataset across all years.
    
    Writes per-year parquet files to avoid OOM on memory-constrained systems.
    Read with: pd.read_parquet('data/parquet/pa_outcomes/')
    """

    if years is None:
        files = sorted(glob.glob(os.path.join(data_dir, 'statcast_*.parquet')))
        years = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in files]

    logger.info(f'Building PA dataset for years: {years}')
    os.makedirs(output_path, exist_ok=True)
    total_pas = 0
    years_processed = 0

    for year in sorted(years):
        year_df = process_year(year, data_dir)
        if len(year_df) == 0:
            continue

        year_path = os.path.join(output_path, f'pa_outcomes_{year}.parquet')
        year_df.to_parquet(year_path, index=False)

        total_pas += len(year_df)
        years_processed += 1
        mb = os.path.getsize(year_path) / 1e6
        logger.info(f'  Saved {year}: {len(year_df):,} PAs ({mb:.1f} MB)')
        logger.info(f'  Running total: {total_pas:,} PAs from {years_processed} seasons')

        # Free memory
        del year_df
        import gc; gc.collect()

    if total_pas == 0:
        raise ValueError('No PA data generated!')

    total_mb = sum(
        os.path.getsize(os.path.join(output_path, f)) / 1e6
        for f in os.listdir(output_path) if f.endswith('.parquet')
    )
    logger.info(f'\nTotal: {total_pas:,} plate appearances across {years_processed} seasons')
    logger.info(f'  Saved to {output_path}/ ({total_mb:.1f} MB total)')


if __name__ == '__main__':
    build_pa_dataset()
