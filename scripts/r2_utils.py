"""Utilities for reading Statcast data from Cloudflare R2.

Usage:
    from scripts.r2_utils import load_statcast, get_s3_client

    # Load all years
    df = load_statcast()

    # Load specific years
    df = load_statcast(years=[2023, 2024, 2025])

    # Load with DuckDB (zero-copy, faster for queries)
    import duckdb
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    for stmt in duckdb_settings():
        conn.execute(stmt)
    result = conn.execute("SELECT * FROM read_parquet('s3://baseball-data/statcast/*.parquet')").df()
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.r2 import bucket, duckdb_settings, get_s3_client  # noqa: E402

BUCKET = bucket()


def load_statcast(years=None, columns=None):
    """Load Statcast data from R2 as a pandas DataFrame.
    
    Args:
        years: List of years to load (default: all 2015-2025)
        columns: List of columns to load (default: all 119)
    
    Returns:
        pd.DataFrame with requested data
    """
    if years is None:
        years = list(range(2015, 2026))
    
    s3 = get_s3_client()
    frames = []
    
    for year in years:
        key = f'statcast/statcast_{year}.parquet'
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        buf = io.BytesIO(obj['Body'].read())
        df = pd.read_parquet(buf, columns=columns)
        frames.append(df)
        print(f"  Loaded {year}: {len(df):,} rows")
    
    result = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(result):,} rows × {len(result.columns)} columns")
    return result


def list_r2_files(prefix=''):
    """List files in the R2 bucket."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    files = []
    for obj in response.get('Contents', []):
        files.append({
            'key': obj['Key'],
            'size_mb': obj['Size'] / 1e6,
            'modified': obj['LastModified']
        })
    return files


def upload_to_r2(local_path, r2_key):
    """Upload a file to R2."""
    s3 = get_s3_client()
    s3.upload_file(local_path, BUCKET, r2_key)
    print(f"Uploaded {local_path} → s3://{BUCKET}/{r2_key}")
