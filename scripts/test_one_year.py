import sqlite3, pandas as pd, os, time

DB_PATH = os.path.expanduser('~/projects/baseball-projections/data/statcast_local.db')
conn = sqlite3.connect(DB_PATH)

t = time.time()
query = """
    SELECT batter, pitcher, game_pk, game_date, game_year, at_bat_number,
           events, stand, p_throws, balls, strikes, outs_when_up, inning,
           home_team, away_team
    FROM statcast_pitches
    WHERE events IS NOT NULL AND game_year = 2024
    ORDER BY game_pk, at_bat_number
"""
df = pd.read_sql(query, conn)
print(f"Read {len(df)} rows in {time.time()-t:.1f}s")
print(df.dtypes)
print(df.head())
conn.close()
