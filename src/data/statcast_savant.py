"""Direct Statcast pull from Baseball Savant — no credentials, no pybaseball.

The season archive in R2 stopped at 2025 (uploaded before the 2026 season
started), so refits trained on nothing from the current year. This module
is the incremental pull that keeps it current.

Savant's CSV export truncates very large result sets, so requests are
chunked by date. A chunk that comes back suspiciously full is split and
retried rather than silently losing pitches.
"""
from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
# Savant caps an export around 25k rows; stay well under it per chunk.
ROW_CAP = 24_000
DEFAULT_CHUNK_DAYS = 3


def _params(start: date, end: date, season: int) -> dict:
    return {
        "all": "true",
        "type": "details",
        "player_type": "batter",
        "hfSea": f"{season}|",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
    }


def fetch_range(start: date, end: date, season: int | None = None,
                timeout: float = 180.0, session: requests.Session | None = None) -> pd.DataFrame:
    """One Savant CSV export. Raises on HTTP error or unparseable body."""
    season = season or start.year
    sess = session or requests
    r = sess.get(CSV_URL, params=_params(start, end, season), timeout=timeout,
                 headers={"User-Agent": "baseball-projections/0.1"})
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    if not text.strip() or text.lstrip().startswith("<"):
        raise ValueError(f"non-CSV response for {start}..{end}")
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    # Savant sometimes returns a lone header row for empty ranges.
    return df if len(df) else df.iloc[0:0]


def fetch_season(season: int, start: date | None = None, end: date | None = None,
                 chunk_days: int = DEFAULT_CHUNK_DAYS,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Every pitch in a date range, chunked and concatenated.

    Defaults to the full season window (March 1 through the earlier of
    November 15 and today).
    """
    start = start or date(season, 3, 1)
    end = end or min(date(season, 11, 15), date.today())
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    frames, cursor = [], start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        df = _fetch_chunk(cursor, chunk_end, season, chunk_days, session)
        if len(df):
            frames.append(df)
        logger.info("%s..%s → %d pitches", cursor, chunk_end, len(df))
        cursor = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Savant can return a pitch twice when a chunk boundary lands mid-game.
    keys = [k for k in ("game_pk", "at_bat_number", "pitch_number") if k in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    return out


def _fetch_chunk(start: date, end: date, season: int, chunk_days: int,
                 session, depth: int = 0) -> pd.DataFrame:
    """Fetch one window, halving it if the result looks truncated."""
    df = fetch_range(start, end, season, session=session)
    if len(df) < ROW_CAP or start == end or depth > 4:
        if len(df) >= ROW_CAP:
            logger.warning("%s..%s hit the row cap and cannot be split further; "
                           "pitches may be missing", start, end)
        return df
    mid = start + (end - start) / 2
    logger.info("%s..%s returned %d rows (cap %d) — splitting",
                start, end, len(df), ROW_CAP)
    left = _fetch_chunk(start, mid, season, chunk_days, session, depth + 1)
    right = _fetch_chunk(mid + timedelta(days=1), end, season, chunk_days, session, depth + 1)
    return pd.concat([left, right], ignore_index=True)
