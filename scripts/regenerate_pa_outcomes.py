#!/usr/bin/env python3
"""Regenerate PA outcomes parquet files from SQLite DB.

Fixes two issues with original data:
1. Duplicate rows: pitch-level rows where event='0' leaked through (events IS NOT NULL
   but IS empty/placeholder). Fix: also filter events != ''
2. Missing inning_topbot column needed by the model.

Each PA has exactly ONE row (the final pitch where events IS NOT NULL AND events != '').
"""

import sqlite3
import pandas as pd
import os
import time

# The real DB is in the baseball-projections project
DB_PATH = os.path.expanduser('~/projects/baseball-projections/data/statcast_local.db')
OUTPUT_DIR = os.path.expanduser('~/projects/baseball-pa-k-model/data/parquet/pa_outcomes')
PARQUET_DIR = os.path.expanduser('~/projects/baseball-pa-k-model/data/parquet')
YEARS = list(range(2015, 2026))

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

conn = sqlite3.connect(DB_PATH)

# Check table name
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
table_names = [t[0] for t in tables]
log(f"Available tables: {table_names}")

# Figure out the table name
if 'statcast_pitches' in table_names:
    TABLE = 'statcast_pitches'
elif 'statcast' in table_names:
    TABLE = 'statcast'
else:
    raise RuntimeError(f"No statcast table found! Tables: {table_names}")

# Verify inning_topbot column exists
cols = [c[1] for c in conn.execute(f"PRAGMA table_info([{TABLE}])").fetchall()]
log(f"Table: {TABLE}, columns include inning_topbot: {'inning_topbot' in cols}")

total_pa = 0
for year in YEARS:
    t0 = time.time()
    log(f"Processing {year}...")
    
    query = f"""
        SELECT batter, pitcher, game_pk, game_date, game_year, at_bat_number,
               events, stand, p_throws, balls, strikes, outs_when_up, inning,
               inning_topbot, home_team, away_team
        FROM {TABLE}
        WHERE game_year = {year} 
          AND events IS NOT NULL 
          AND events != ''
          AND events != '0'
        ORDER BY game_pk, at_bat_number
    """
    
    df = pd.read_sql(query, conn)
    
    # Rename events -> event
    df = df.rename(columns={'events': 'event'})
    
    # Add flags
    df['is_k'] = df['event'].isin(['strikeout', 'strikeout_double_play']).astype('int8')
    df['is_bb'] = df['event'].isin(['walk', 'intent_walk']).astype('int8')
    df['is_hbp'] = (df['event'] == 'hit_by_pitch').astype('int8')
    df['is_hit'] = df['event'].isin(['single', 'double', 'triple', 'home_run']).astype('int8')
    df['is_hr'] = (df['event'] == 'home_run').astype('int8')
    
    # Cast types
    for col in ['batter', 'pitcher', 'game_year']:
        df[col] = df[col].astype('int32')
    
    out_path = os.path.join(OUTPUT_DIR, f'pa_outcomes_{year}.parquet')
    df.to_parquet(out_path, index=False)
    
    mb = os.path.getsize(out_path) / 1e6
    total_pa += len(df)
    elapsed = time.time() - t0
    topbot_vals = df['inning_topbot'].unique().tolist() if 'inning_topbot' in df.columns else 'MISSING'
    log(f"  {year}: {len(df):,} PAs ({mb:.1f} MB) K%={df['is_k'].mean():.3f} topbot={topbot_vals} [{elapsed:.1f}s]")
    del df

log(f"\nTotal PAs: {total_pa:,}")

# Regenerate park factors
log("Regenerating park_factors.parquet...")
t0 = time.time()
all_dfs = []
for year in YEARS:
    fp = os.path.join(OUTPUT_DIR, f'pa_outcomes_{year}.parquet')
    if os.path.exists(fp):
        d = pd.read_parquet(fp, columns=['home_team', 'game_year', 'is_k', 'is_hit', 'is_hr'])
        all_dfs.append(d)

all_pa = pd.concat(all_dfs, ignore_index=True)
del all_dfs

home = all_pa.groupby(['home_team', 'game_year']).agg(
    home_pa=('is_k', 'count'),
    home_k_rate=('is_k', 'mean'),
    home_hit_rate=('is_hit', 'mean'),
    home_hr_rate=('is_hr', 'mean'),
).reset_index()

lg = all_pa.groupby('game_year').agg(
    lg_k_rate=('is_k', 'mean'),
    lg_hit_rate=('is_hit', 'mean'),
    lg_hr_rate=('is_hr', 'mean'),
).reset_index()

pf = home.merge(lg, on='game_year')
pf['k_park_factor'] = pf['home_k_rate'] / pf['lg_k_rate']
pf['hit_park_factor'] = pf['home_hit_rate'] / pf['lg_hit_rate']
pf['hr_park_factor'] = pf['home_hr_rate'] / pf['lg_hr_rate']
pf = pf.rename(columns={'home_team': 'team'})
pf = pf[['team', 'game_year', 'home_pa', 'k_park_factor', 'hit_park_factor', 'hr_park_factor']]
pf.to_parquet(os.path.join(PARQUET_DIR, 'park_factors.parquet'), index=False)
log(f"  Park factors: {len(pf)} team-year rows [{time.time()-t0:.1f}s]")
del all_pa

# Player metadata
log("Regenerating player_metadata.parquet...")
t0 = time.time()
meta_dfs = []
for year in YEARS:
    fp = os.path.join(OUTPUT_DIR, f'pa_outcomes_{year}.parquet')
    if os.path.exists(fp):
        d = pd.read_parquet(fp, columns=['batter', 'stand', 'game_year'])
        meta_dfs.append(d)

meta_all = pd.concat(meta_dfs, ignore_index=True)
del meta_dfs

meta = meta_all.groupby(['batter', 'stand']).agg(
    first_year=('game_year', 'min'),
    last_year=('game_year', 'max'),
    total_pa=('game_year', 'count'),
).reset_index()
meta.to_parquet(os.path.join(PARQUET_DIR, 'player_metadata.parquet'), index=False)
log(f"  Player metadata: {len(meta)} batter-stand combos [{time.time()-t0:.1f}s]")

conn.close()
log("Done!")
