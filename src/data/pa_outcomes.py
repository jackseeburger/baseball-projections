"""Loader for the PA-level outcome parquets (`pa_outcomes/pa_outcomes_<year>.parquet` in R2).

One row per plate appearance, written by `src/data/pa_outcomes_pipeline.py`.
The intra-season backtest (`src/eval/intraseason.py`) needs these because the
season-level table has no dates: you cannot ask "what did we know on July 1"
of a frame whose finest grain is a season.

Files land in `data/parquet/` (gitignored) and are cached there — the first
call downloads, later calls read the local copy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PA_OUTCOMES_PREFIX = "pa_outcomes/"
LOCAL_DIR = Path("data/parquet")

# Columns the intra-season aggregation actually needs. Reading only these
# keeps a 157k-row season under a few MB in memory.
# `pitcher` rides along because the pitcher side of station A
# (`src/eval/pitchers.py`) aggregates the very same rows by the other id on
# them — a batter faced is a plate appearance seen from the mound.
REQUIRED_COLUMNS = [
    "batter", "pitcher", "game_pk", "game_date", "game_year", "event",
    "is_k", "is_bb", "is_hbp", "is_hit", "is_hr", "is_single",
    "is_double", "is_triple",
]


def r2_key(year: int) -> str:
    return f"{PA_OUTCOMES_PREFIX}pa_outcomes_{year}.parquet"


def local_path(year: int, data_dir: str | Path = LOCAL_DIR) -> Path:
    return Path(data_dir) / f"pa_outcomes_{year}.parquet"


def available_years(prefix: str = PA_OUTCOMES_PREFIX) -> list[int]:
    """Years with a PA-outcomes parquet in R2. Empty list if R2 is unreachable."""
    from src.data.r2 import bucket, get_s3_client

    client = get_s3_client()
    years: list[int] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            stem = Path(obj["Key"]).stem
            tail = stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                years.append(int(tail))
    return sorted(set(years))


def download(year: int, data_dir: str | Path = LOCAL_DIR, refresh: bool = False) -> Path:
    """Fetch one year's PA parquet from R2 into `data_dir`; returns the path."""
    from src.data.r2 import bucket, get_s3_client

    path = local_path(year, data_dir)
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s -> %s", r2_key(year), path)
    get_s3_client().download_file(bucket(), r2_key(year), str(path))
    return path


def load_pa_outcomes(
    year: int,
    data_dir: str | Path = LOCAL_DIR,
    columns: list[str] | None = REQUIRED_COLUMNS,
    allow_download: bool = True,
) -> pd.DataFrame:
    """PA-level outcomes for one season, cached locally.

    `game_date` comes back as a pandas datetime so cutoffs can be compared
    without string-format assumptions.
    """
    path = local_path(year, data_dir)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"{path} not found and allow_download=False — run "
                f"`python -c \"from src.data.pa_outcomes import download; download({year})\"`"
            )
        download(year, data_dir)
    df = pd.read_parquet(path, columns=columns)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df
