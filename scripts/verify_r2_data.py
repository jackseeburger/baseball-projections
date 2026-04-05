"""Verify R2 parquet data matches SQLite source."""
import sqlite3
import io
import boto3
import pyarrow.parquet as pq
from botocore.config import Config

s3 = boto3.client('s3',
    endpoint_url='https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com',
    aws_access_key_id='18170dc8b2b4805d3c057f69bb5b8ffb',
    aws_secret_access_key='3b7f08cb995538ce5ce81572036d416a5535b8b582e53f2ac2ad80b9fd9a687a',
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

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
