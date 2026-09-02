"""MLB Stats API (statsapi.mlb.com) ingest — no credentials required.

Three feeds:
  * season hitting stats  → the season-level table the backtest harness scores
  * standings             → Phase 2 simulator input
  * schedule              → remaining games for the season Monte Carlo

Player ids are MLBAM ids, identical to Statcast `batter`, so everything joins
to the Bayesian components and the Chadwick birthdates with no crosswalk.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import PARQUET_DIR

logger = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
SEASONS_PARQUET = PARQUET_DIR / "hitter_seasons_api.parquet"

# Stats API field → our column
HITTING_FIELDS = {
    "plateAppearances": "pa",
    "atBats": "ab",
    "hits": "h",
    "doubles": "doubles",
    "triples": "triples",
    "homeRuns": "hr",
    "strikeOuts": "k",
    "baseOnBalls": "bb",
    "hitByPitch": "hbp",
    "sacFlies": "sf",
}


def _get(path: str, **params) -> dict:
    resp = requests.get(f"{BASE}/{path}", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_season_hitting(season: int, page_size: int = 500) -> pd.DataFrame:
    """One row per player-team split for a season; counts summed later."""
    rows, offset = [], 0
    while True:
        data = _get(
            "stats", stats="season", group="hitting", season=season,
            sportId=1, playerPool="all", limit=page_size, offset=offset,
        )
        stats = data.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        if not splits:
            break
        for s in splits:
            row = {"batter": s["player"]["id"], "season": season,
                   "age": s["stat"].get("age")}
            for api_field, col in HITTING_FIELDS.items():
                row[col] = s["stat"].get(api_field, 0)
            rows.append(row)
        offset += page_size
        total = stats[0].get("totalSplits", 0)
        if offset >= total:
            break
    df = pd.DataFrame(rows)
    logger.info(f"{season}: {len(df)} player-team splits")
    return df


def build_seasons_table(
    start: int, end: int, cache_path: Path = SEASONS_PARQUET,
    refresh: bool = False,
) -> pd.DataFrame:
    """Season-level table in the backtest harness schema.

    Columns: batter, season, age, pa, ab, k, bb, hr, xb_points,
    bip, hits_in_play. Traded players' team splits are summed.
    """
    if cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        if cached["season"].min() <= start and cached["season"].max() >= end:
            return cached[cached["season"].between(start, end)]

    frames = [fetch_season_hitting(season) for season in range(start, end + 1)]
    df = pd.concat(frames, ignore_index=True)
    count_cols = list(HITTING_FIELDS.values())
    agg = (
        df.groupby(["batter", "season"], as_index=False)
        .agg({**{c: "sum" for c in count_cols}, "age": "first"})
    )
    # Derived harness columns
    agg["xb_points"] = agg["doubles"] + 2 * agg["triples"] + 3 * agg["hr"]
    agg["bip"] = agg["ab"] - agg["k"] - agg["hr"] + agg["sf"]
    agg["hits_in_play"] = agg["h"] - agg["hr"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(cache_path, index=False)
    logger.info(f"wrote {cache_path}: {len(agg)} player-seasons "
                f"{agg['season'].min()}-{agg['season'].max()}")
    return agg


def fetch_standings(season: int) -> pd.DataFrame:
    """Current standings, one row per team (Phase 2.1)."""
    data = _get("standings", leagueId="103,104", season=season)
    rows = []
    for record in data.get("records", []):
        division_id = record.get("division", {}).get("id")
        for tr in record.get("teamRecords", []):
            rows.append({
                "team_id": tr["team"]["id"],
                "team": tr["team"]["name"],
                "division_id": division_id,
                "wins": tr["wins"],
                "losses": tr["losses"],
                "win_pct": float(tr["winningPercentage"]),
                "games_back": tr.get("gamesBack"),
                "runs_scored": tr.get("runsScored"),
                "runs_allowed": tr.get("runsAllowed"),
            })
    return pd.DataFrame(rows)


def fetch_schedule(start_date: str, end_date: str) -> pd.DataFrame:
    """Games between two ISO dates, one row per game (Phase 2.1)."""
    data = _get("schedule", sportId=1, startDate=start_date, endDate=end_date)
    rows = []
    for date in data.get("dates", []):
        for g in date.get("games", []):
            rows.append({
                "game_pk": g["gamePk"],
                "date": date["date"],
                "game_datetime": g.get("gameDate"),
                "status": g["status"]["abstractGameState"],
                "game_type": g.get("gameType"),
                "home_id": g["teams"]["home"]["team"]["id"],
                "away_id": g["teams"]["away"]["team"]["id"],
                "home_score": g["teams"]["home"].get("score"),
                "away_score": g["teams"]["away"].get("score"),
            })
    return pd.DataFrame(rows)
