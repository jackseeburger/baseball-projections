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
    conn.execute(f"SET s3_endpoint='{R2_ENDPOINT.replace('https://','')}'")
    conn.execute(f"SET s3_access_key_id='{R2_ACCESS_KEY}'")
    conn.execute(f"SET s3_secret_access_key='{R2_SECRET_KEY}'")
    conn.execute("SET s3_region='auto'")
    conn.execute("SET s3_url_style='path'")
    result = conn.execute("SELECT * FROM read_parquet('s3://baseball-data/statcast/*.parquet')").df()
"""
import os
import io
import pandas as pd
import boto3
from botocore.config import Config

# R2 configuration — reads from env vars with fallbacks
R2_ENDPOINT = os.getenv('R2_ENDPOINT_URL', 'https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com')
R2_ACCESS_KEY = os.getenv('R2_ACCESS_KEY_ID', '18170dc8b2b4805d3c057f69bb5b8ffb')
R2_SECRET_KEY = os.getenv('R2_SECRET_ACCESS_KEY', '3b7f08cb995538ce5ce81572036d416a5535b8b582e53f2ac2ad80b9fd9a687a')
BUCKET = os.getenv('R2_BUCKET_NAME', 'baseball-data')


def get_s3_client():
    """Get a boto3 S3 client configured for R2."""
    return boto3.client('s3',
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )


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
