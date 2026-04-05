"""Export statcast_pitches from SQLite → year-partitioned Parquets → R2.

Uses chunked reads (50K rows) + ParquetWriter to avoid OOM on 2GB RAM.
Writes to /tmp then uploads (BytesIO still needs full file in memory).
"""
import sqlite3
import os
import sys
import gc
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from botocore.config import Config

# R2 config
R2_ENDPOINT = 'https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com'
R2_ACCESS_KEY = '18170dc8b2b4805d3c057f69bb5b8ffb'
R2_SECRET_KEY = '3b7f08cb995538ce5ce81572036d416a5535b8b582e53f2ac2ad80b9fd9a687a'
BUCKET = 'baseball-data'
DB_PATH = 'data/statcast_local.db'
CHUNK_SIZE = 50_000

s3 = boto3.client('s3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

# Check existing uploads
existing = set()
try:
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix='statcast/')
    for obj in response.get('Contents', []):
        name = obj['Key'].split('/')[-1]
        if name.startswith('statcast_') and name.endswith('.parquet'):
            existing.add(name.replace('statcast_', '').replace('.parquet', ''))
except:
    pass

print(f"Already uploaded: {sorted(existing) or 'none'}", flush=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Get column names
cur.execute('PRAGMA table_info(statcast_pitches)')
columns = [row[1] for row in cur.fetchall()]

# Get years
cur.execute("SELECT DISTINCT substr(game_date, 1, 4) FROM statcast_pitches ORDER BY 1")
years = [r[0] for r in cur.fetchall()]
print(f"Years: {years}", flush=True)

for year in years:
    if year in existing:
        print(f"[{year}] SKIP", flush=True)
        continue
    
    local_path = f'/tmp/statcast_{year}.parquet'
    print(f"[{year}] Exporting...", flush=True)
    
    # Count rows first
    cur.execute(f"SELECT COUNT(*) FROM statcast_pitches WHERE game_date LIKE '{year}%'")
    total_rows = cur.fetchone()[0]
    
    # Read in chunks, write with ParquetWriter
    cur.execute(f"SELECT * FROM statcast_pitches WHERE game_date LIKE '{year}%' ORDER BY game_date")
    
    writer = None
    written = 0
    
    while True:
        rows = cur.fetchmany(CHUNK_SIZE)
        if not rows:
            break
        
        # Build arrow table from chunk
        arrays = {}
        for i, col in enumerate(columns):
            arrays[col] = [row[i] for row in rows]
        table = pa.table(arrays)
        
        if writer is None:
            writer = pq.ParquetWriter(local_path, table.schema, compression='snappy')
        
        writer.write_table(table)
        written += len(rows)
        print(f"  [{year}] {written:,}/{total_rows:,} rows", flush=True)
        
        del rows, arrays, table
        gc.collect()
    
    if writer:
        writer.close()
    
    file_size = os.path.getsize(local_path)
    print(f"[{year}] Parquet: {file_size / 1e6:.1f} MB, uploading...", flush=True)
    
    # Upload
    r2_key = f'statcast/statcast_{year}.parquet'
    s3.upload_file(local_path, BUCKET, r2_key)
    print(f"[{year}] ✓ Uploaded", flush=True)
    
    # Delete local file immediately
    os.remove(local_path)
    gc.collect()

conn.close()

# Final verification
print(f"\n=== VERIFICATION ===", flush=True)
response = s3.list_objects_v2(Bucket=BUCKET, Prefix='statcast/')
total_size = 0
for obj in sorted(response.get('Contents', []), key=lambda x: x['Key']):
    mb = obj['Size'] / 1e6
    total_size += obj['Size']
    print(f"  {obj['Key']}: {mb:.1f} MB")
print(f"Total: {total_size / 1e6:.0f} MB", flush=True)
print("DONE!", flush=True)
