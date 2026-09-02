"""MLB Stats API (statsapi.mlb.com) ingest — no credentials required.

Feeds:
  * season hitting stats  → the season-level table the backtest harness scores
  * standings             → Phase 2 simulator input
  * schedule              → remaining games for the season Monte Carlo
  * probable starters     → station E starting-pitcher term
  * season / game-log pitching stats → the pitcher rates that term is built on

Player ids are MLBAM ids, identical to Statcast `batter`, so everything joins
to the Bayesian components and the Chadwick birthdates with no crosswalk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_DIR, PARQUET_DIR

logger = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
SEASONS_PARQUET = PARQUET_DIR / "hitter_seasons_api.parquet"
# Raw API JSON lands here so a walk-forward backtest can be re-run offline.
# Gitignored (data/cache/); delete a file or pass refresh=True to re-pull.
STATSAPI_CACHE = DATA_DIR / "cache" / "statsapi"

PITCHING_FIELDS = {
    "battersFaced": "bf",
    "strikeOuts": "k",
    "baseOnBalls": "bb",
    "hitBatsmen": "hbp",
    "homeRuns": "hr",
    "earnedRuns": "er",
    "gamesStarted": "gs",
}

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


def _get_cached(cache_key: str, path: str, refresh: bool = False, **params) -> dict:
    """`_get` with the response JSON cached under data/cache/statsapi/.

    Only ever cache *settled* facts (a past date's schedule, a finished
    appearance). Anything about a game that has not started yet will go stale.
    """
    cache_file = STATSAPI_CACHE / f"{cache_key}.json"
    if cache_file.exists() and not refresh:
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            logger.warning(f"corrupt cache {cache_file}; re-fetching")
    data = _get(path, **params)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def _ip_to_outs(innings) -> float:
    """Stats API innings-pitched strings ("5.2" = 5 innings + 2 outs) → outs."""
    if innings is None:
        return 0.0
    whole, _, frac = str(innings).partition(".")
    try:
        return float(whole or 0) * 3 + float(frac[:1] or 0)
    except ValueError:
        return 0.0


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


# ─── Station E: probable starters and pitcher rates ───

def fetch_probables(start_date: str, end_date: str, refresh: bool = False) -> pd.DataFrame:
    """One row per game_pk with the home/away starting pitcher ids.

    Columns: game_pk, date, game_type, home_sp_id, away_sp_id (nullable Int64).

    Walk-forward honesty: `probablePitcher` is a *pre-game* announcement field,
    but for a date already in the past the API serves the pitcher who actually
    started. Verified on 2026-07-18 (16/16 games match the boxscore's starter).
    That is the same information the market had at its close, which is a median
    15 minutes before first pitch (docs/market-benchmark-2026.md) — so scoring a
    model that uses it against the exchanges' closes is a fair comparison. It is
    *not* a simulation of picking games the morning before, where late scratches
    would cost you a little.

    Requests are chunked by month and cached; the API returns the whole range in
    one call but a month is a friendlier cache unit.
    """
    months = pd.period_range(start=start_date, end=end_date, freq="M")
    rows = []
    for period in months:
        lo = max(period.start_time.date(), pd.Timestamp(start_date).date())
        hi = min(period.end_time.date(), pd.Timestamp(end_date).date())
        data = _get_cached(
            f"probables_{lo}_{hi}", "schedule", refresh=refresh,
            sportId=1, startDate=str(lo), endDate=str(hi),
            hydrate="probablePitcher",
        )
        for day in data.get("dates", []):
            for g in day.get("games", []):
                teams = g.get("teams", {})
                rows.append({
                    "game_pk": g["gamePk"],
                    "date": day["date"],
                    "game_type": g.get("gameType"),
                    "home_sp_id": (teams.get("home", {}).get("probablePitcher") or {}).get("id"),
                    "away_sp_id": (teams.get("away", {}).get("probablePitcher") or {}).get("id"),
                })
    df = pd.DataFrame(rows, columns=["game_pk", "date", "game_type",
                                     "home_sp_id", "away_sp_id"])
    for col in ("home_sp_id", "away_sp_id"):
        df[col] = df[col].astype("Int64")
    logger.info(f"probables {start_date}..{end_date}: {len(df)} games, "
                f"{int((df['home_sp_id'].notna() & df['away_sp_id'].notna()).sum())} with both")
    return df


def fetch_season_pitching(season: int, page_size: int = 1000,
                          refresh: bool = False) -> pd.DataFrame:
    """Season pitching totals for every pitcher, one row per player-team split.

    Columns: pitcher, season, bf, k, bb, hbp, hr, er, gs, outs.
    """
    rows, offset = [], 0
    while True:
        data = _get_cached(
            f"pitching_season_{season}_{offset}", "stats", refresh=refresh,
            stats="season", group="pitching", season=season, sportId=1,
            playerPool="all", limit=page_size, offset=offset,
        )
        stats = data.get("stats", [])
        splits = stats[0].get("splits", []) if stats else []
        if not splits:
            break
        for s in splits:
            row = {"pitcher": s["player"]["id"], "season": season,
                   "outs": _ip_to_outs(s["stat"].get("inningsPitched"))}
            for api_field, col in PITCHING_FIELDS.items():
                row[col] = s["stat"].get(api_field, 0) or 0
            rows.append(row)
        offset += page_size
        if offset >= stats[0].get("totalSplits", 0):
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(["pitcher", "season"], as_index=False).sum(numeric_only=True)
    logger.info(f"{season}: {len(agg)} pitcher seasons")
    return agg


def fetch_pitcher_game_logs(pitcher_ids, season: int,
                            refresh: bool = False) -> pd.DataFrame:
    """Per-appearance pitching lines for `pitcher_ids` in `season`.

    Columns: pitcher, season, date, game_pk, game_type, bf, k, bb, hbp, hr,
    er, gs, outs. One row per appearance — the caller filters to `date <` the
    game being predicted, which is what keeps the backtest walk-forward.
    """
    rows = []
    for pid in sorted({int(p) for p in pitcher_ids}):
        data = _get_cached(
            f"pitching_gamelog_{season}_{pid}", f"people/{pid}/stats",
            refresh=refresh, stats="gameLog", group="pitching", season=season,
        )
        stats = data.get("stats", [])
        for s in (stats[0].get("splits", []) if stats else []):
            row = {"pitcher": pid, "season": season, "date": s.get("date"),
                   "game_pk": (s.get("game") or {}).get("gamePk"),
                   "game_type": s.get("gameType"),
                   "outs": _ip_to_outs(s["stat"].get("inningsPitched"))}
            for api_field, col in PITCHING_FIELDS.items():
                row[col] = s["stat"].get(api_field, 0) or 0
            rows.append(row)
    cols = ["pitcher", "season", "date", "game_pk", "game_type", "outs",
            *PITCHING_FIELDS.values()]
    df = pd.DataFrame(rows, columns=cols)
    logger.info(f"{season} game logs: {len(df)} appearances "
                f"for {df['pitcher'].nunique() if len(df) else 0} pitchers")
    return df
