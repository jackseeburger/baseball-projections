"""Real player birthdates from the Chadwick Bureau register.

Replaces the `birth_year = first_year - 23` estimate (roadmap 0.1). The
register (github.com/chadwickbureau/register) carries key_mlbam alongside
birth year/month/day, so it joins directly onto Statcast `batter` ids.

Age convention: a player's seasonal age is their age as of June 30 of the
season, per the roadmap and standard baseball reference practice.

Usage:
    from src.data.birthdates import load_birthdates, seasonal_age

    bd = load_birthdates()                      # cached parquet or download
    age = seasonal_age(bd, batter_ids, season)  # float ages, NaN if unknown
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import PARQUET_DIR

logger = logging.getLogger(__name__)

REGISTER_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/register"
    "/master/data/people-{shard}.csv"
)
REGISTER_SHARDS = "0123456789abcdef"
REGISTER_COLUMNS = [
    "key_mlbam", "name_first", "name_last",
    "birth_year", "birth_month", "birth_day",
    "mlb_played_first", "mlb_played_last",
]
BIRTHDATES_PARQUET = PARQUET_DIR / "birthdates.parquet"
# Age as of June 30; used when month/day are missing (register has year only).
SEASONAL_AGE_MONTH, SEASONAL_AGE_DAY = 6, 30


def fetch_register(timeout: int = 60) -> pd.DataFrame:
    """Download the Chadwick register and return rows that have an MLBAM id.

    The register is sharded into 16 CSVs (people-0 .. people-f). Only players
    with a known MLBAM id are kept since Statcast keys on it.
    """
    frames = []
    for shard in REGISTER_SHARDS:
        url = REGISTER_URL.format(shard=shard)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            usecols=REGISTER_COLUMNS,
            dtype={"key_mlbam": "Int64", "birth_year": "Int64",
                   "birth_month": "Int64", "birth_day": "Int64",
                   "mlb_played_first": "Int64", "mlb_played_last": "Int64"},
        )
        frames.append(df[df["key_mlbam"].notna()])
        logger.info(f"register shard {shard}: {len(frames[-1])} rows with MLBAM id")
    people = pd.concat(frames, ignore_index=True)
    people = parse_register(people)
    logger.info(
        f"Chadwick register: {len(people)} players with MLBAM id, "
        f"{people['birth_year'].notna().sum()} with birth year"
    )
    return people


def parse_register(people: pd.DataFrame) -> pd.DataFrame:
    """Normalize register rows to the birthdates schema.

    Output columns: batter (int64 MLBAM id), name_first, name_last,
    birth_year, birth_month, birth_day (nullable ints),
    mlb_played_first, mlb_played_last.
    Duplicate MLBAM ids keep the row with the most complete birthdate.
    """
    df = people.copy()
    df = df[df["key_mlbam"].notna()]
    df["batter"] = df["key_mlbam"].astype("int64")
    completeness = (
        df["birth_year"].notna().astype(int)
        + df["birth_month"].notna().astype(int)
        + df["birth_day"].notna().astype(int)
    )
    df = (
        df.assign(_complete=completeness)
        .sort_values("_complete", ascending=False)
        .drop_duplicates("batter", keep="first")
        .drop(columns=["_complete", "key_mlbam"])
        .reset_index(drop=True)
    )
    return df[["batter", "name_first", "name_last",
               "birth_year", "birth_month", "birth_day",
               "mlb_played_first", "mlb_played_last"]]


def load_birthdates(
    cache_path: Path = BIRTHDATES_PARQUET,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load birthdates from the local parquet cache, downloading if absent."""
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    people = fetch_register()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    people.to_parquet(cache_path, index=False)
    logger.info(f"wrote {cache_path} ({len(people)} rows)")
    return people


def seasonal_age(
    birthdates: pd.DataFrame,
    batter_ids: pd.Series | np.ndarray,
    season: pd.Series | np.ndarray | int,
) -> np.ndarray:
    """Age as of June 30 of `season` for each batter id; NaN when unknown.

    Uses the full birthdate when month/day are present; falls back to
    June-30-of-birth-year (i.e., integer season - birth_year) when the
    register has year only.
    """
    bd = birthdates.set_index("batter")
    ids = pd.Series(np.asarray(batter_ids))
    season = pd.Series(np.broadcast_to(np.asarray(season), ids.shape))

    by = ids.map(bd["birth_year"]).astype("float64")
    bm = ids.map(bd["birth_month"]).astype("float64").fillna(SEASONAL_AGE_MONTH)
    bday = ids.map(bd["birth_day"]).astype("float64").fillna(SEASONAL_AGE_DAY)

    # Fractional year of birth, and of the June 30 reference point.
    birth_frac = by + (bm - 1) / 12.0 + (bday - 1) / 365.25
    ref_frac = season + (SEASONAL_AGE_MONTH - 1) / 12.0 + (SEASONAL_AGE_DAY - 1) / 365.25
    return (ref_frac - birth_frac).to_numpy()


def birth_year_map(
    birthdates: pd.DataFrame,
    fallback_first_year: pd.Series | None = None,
    fallback_offset: int = 23,
) -> pd.Series:
    """batter -> birth_year Series, with the legacy first_year-23 fallback.

    `fallback_first_year` is a batter-indexed Series of debut years used for
    ids missing from the register (retired ids, data errors). Callers should
    log how many fell through — a high count means the join is broken.
    """
    known = birthdates.set_index("batter")["birth_year"].dropna().astype("int64")
    if fallback_first_year is None:
        return known
    fallback = (fallback_first_year - fallback_offset).astype("int64")
    combined = known.reindex(fallback.index).fillna(fallback)
    n_missing = int(combined.isna().sum() + (~fallback.index.isin(known.index)).sum())
    if n_missing:
        logger.warning(
            f"birthdates: {n_missing} of {len(fallback)} batters not in register; "
            f"using first_year - {fallback_offset} estimate for them"
        )
    return combined.astype("int64")


def build_batter_birth_years(
    batter_first_year: pd.Series,
    birthdates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the `batter_birth_years.parquet` table the Modal volume expects.

    Schema: columns [batter, birth_year]. Real register values where known,
    legacy estimate otherwise, so existing Modal training functions pick up
    corrected ages with no code change.
    """
    if birthdates is None:
        birthdates = load_birthdates()
    combined = birth_year_map(birthdates, fallback_first_year=batter_first_year)
    return combined.rename("birth_year").rename_axis("batter").reset_index()
