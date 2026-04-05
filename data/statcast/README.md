# Statcast Data

Year-partitioned Parquet files stored in **Cloudflare R2**.

**Bucket:** `s3://baseball-data/statcast/`
**Format:** Parquet (Snappy compression)
**Source:** `statcast_local.db` → exported April 2026

## Files

| File | Rows | Size |
|------|------|------|
| statcast_2015.parquet | 712,844 | 102.5 MB |
| statcast_2016.parquet | 726,275 | 105.4 MB |
| statcast_2017.parquet | 735,954 | 125.2 MB |
| statcast_2018.parquet | 734,567 | 122.5 MB |
| statcast_2019.parquet | 763,198 | 126.9 MB |
| statcast_2020.parquet | 280,398 | 47.0 MB |
| statcast_2021.parquet | 765,733 | 127.2 MB |
| statcast_2022.parquet | 775,330 | 127.5 MB |
| statcast_2023.parquet | 774,038 | 135.7 MB |
| statcast_2024.parquet | 760,248 | 145.7 MB |
| statcast_2025.parquet | 749,091 | 146.2 MB |
| **Total** | **7,777,676** | **1.31 GB** |

## Access

### From Modal (production)
```python
# boto3 via r2-baseball secret
s3 = get_s3_client()
s3.download_file('baseball-data', 'statcast/statcast_2024.parquet', '/tmp/2024.parquet')

# Or DuckDB direct query
conn.execute("SELECT * FROM read_parquet('s3://baseball-data/statcast/*.parquet')")
```

### From VPS (local development)
```python
import duckdb
conn = duckdb.connect()
# Configure R2 endpoint (see .env)
df = conn.execute("SELECT * FROM read_parquet('s3://baseball-data/statcast/statcast_2024.parquet')").df()
```

## Schema
119 columns — pitch-level Statcast data from 2015-2025.
Key columns: `game_date`, `batter`, `pitcher`, `events`, `description`,
`launch_speed`, `launch_angle`, `plate_x`, `plate_z`, etc.
