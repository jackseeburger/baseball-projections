import os
"""Verify R2 parquet data matches SQLite source."""
import sqlite3
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.r2 import get_s3_client  # noqa: E402
import pyarrow.parquet as pq

s3 = get_s3_client()

conn = sqlite3.connect('data/statcast_local.db')

years = range(2015, 2026)
total_parquet = 0
total_sqlite = 0
all_good = True

for year in years:
    # SQLite count
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM statcast_pitches WHERE game_date LIKE '{year}%'")
    sqlite_count = cur.fetchone()[0]
    
    # Parquet metadata (just read the footer, not the whole file)
    key = f'statcast/statcast_{year}.parquet'
    obj = s3.get_object(Bucket='baseball-data', Key=key)
    buf = io.BytesIO(obj['Body'].read())
    pf = pq.ParquetFile(buf)
    parquet_count = pf.metadata.num_rows
    parquet_cols = pf.metadata.num_columns
    
    match = "✅" if sqlite_count == parquet_count else "❌"
    if sqlite_count != parquet_count:
        all_good = False
    
    print(f"{year}: SQLite={sqlite_count:,}  Parquet={parquet_count:,}  Cols={parquet_cols}  {match}")
    total_parquet += parquet_count
    total_sqlite += sqlite_count

conn.close()

print(f"\nTotal: SQLite={total_sqlite:,}  Parquet={total_parquet:,}")
if all_good:
    print("✅ ALL YEARS MATCH — data integrity verified!")
else:
    print("❌ MISMATCH DETECTED")
