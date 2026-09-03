"""MLB Stats API (statsapi.mlb.com) ingest — no credentials required.

Feeds:
  * season hitting stats  → the season-level table the backtest harness scores
  * standings             → Phase 2 simulator input
  * schedule              → remaining games for the season Monte Carlo
  * probable starters     → station E starting-pitcher term
  * season / game-log pitching stats → the pitcher rates that term is built on
  * rosters with IL status, hitter game logs → station B playing time
  * transactions (IL placements, activations, options, recalls) → the dated
    spells station B's return-time distribution is estimated from

Player ids are MLBAM ids, identical to Statcast `batter`, so everything joins
to the Bayesian components and the Chadwick birthdates with no crosswalk.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_DIR, PARQUET_DIR

logger = logging.getLogger(__name__)

BASE = "https://statsapi.mlb.com/api/v1"
# The live feed (boxscore, play-by-play, posted lineup) lives on v1.1 only.
BASE_V11 = "https://statsapi.mlb.com/api/v1.1"
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
    # Pitches thrown. Present on every game-log split the API serves for 2025
    # and 2026 and on the season totals; it is the workload number a manager
    # actually counts, and `src.sim.reliever_usage` reads it to decide who is
    # available tonight. Absent or zero on a stray split, that module falls
    # back to batters faced times the league's pitches per batter faced.
    "numberOfPitches": "pitches",
    # Hits, at-bats and sacrifice flies allowed. Station E's FIP term does not
    # want them — FIP is deliberately blind to balls in play — but two other
    # consumers do: the pitcher rate provider's BABIP-against component
    # (`src/eval/pitchers.py`) and the team-defence residual
    # (`src/sim/defence.py`), both of which need BIP = AB - K - HR + SF and
    # hits in play = H - HR. The API has carried all three on every season
    # split since 2015, and they are re-parsed from the same cached responses
    # the rest of the pitching line comes from, so adding them costs no fetch.
    "hits": "h",
    "atBats": "ab",
    "sacFlies": "sf",
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


# A cold season is ~4,000 requests and `_fetch_many` runs them a dozen at a
# time; at that rate the odd connection is dropped mid-handshake or served a
# 429/503. Those are transient and retrying fixes them, while a 404 for a
# player who never pitched is not and must still raise on the first try.
RETRY_STATUS = (429, 500, 502, 503, 504)
RETRIES = 4
RETRY_BACKOFF = 1.5


def _get(path: str, base: str = BASE, **params) -> dict:
    """One Stats API call, retried through transient network and server errors."""
    last = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(f"{base}/{path}", params=params, timeout=60)
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{resp.status_code} for {path}",
                                         response=resp)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout,
                json.JSONDecodeError) as exc:
            last = exc
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in RETRY_STATUS:
                raise
            last = exc
        if attempt < RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise last


def _get_cached(cache_key: str, path: str, refresh: bool = False,
                base: str = BASE, **params) -> dict:
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
    data = _get(path, base=base, **params)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def _fetch_many(ids, build, workers: int = 1) -> list:
    """`build(id)` for every id, results in the order given.

    One player-season game log is one request, and a caller that wants the
    whole league wants about fifteen hundred of them: the nightly odds job
    needs every pitcher's appearances (for the pen, the rotation and who is
    available) and every hitter's plate appearances (for the club's shares)
    *today*, not eventually. `workers > 1` runs them through a small thread
    pool — the requests are independent GETs and the cache writes go to
    distinct files — which is the difference between an eleven-minute fetch
    and an eighty-second one. The order of the returned list never depends on
    it, so the frame this builds is byte-identical either way.
    """
    ids = list(ids)
    if workers and int(workers) > 1 and len(ids) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            return list(pool.map(build, ids))
    return [build(i) for i in ids]


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
    """Games between two ISO dates, one row per game (Phase 2.1).

    `venue_id` / `venue_name` are the ballpark the game is played in, which is
    the home club's park for all but the handful of neutral-site games a season
    and is what `src.sim.park` keys its run multipliers on. Taken from the
    schedule rather than from a static park table so a club that moves, a
    London series and a temporary home all come out right. `day_night` comes
    free on the same response; it and the venue are the two pre-game
    circumstances no station models, and the learned challenger of station E
    reads both (`src/sim/game_features.py`). All are `.get`-guarded, so a
    caller serving a trimmed fixture keeps working and simply gets a null.
    """
    data = _get("schedule", sportId=1, startDate=start_date, endDate=end_date)
    rows = []
    for date in data.get("dates", []):
        for g in date.get("games", []):
            venue = g.get("venue") or {}
            rows.append({
                "game_pk": g["gamePk"],
                "date": date["date"],
                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),
                "game_datetime": g.get("gameDate"),
                "status": g["status"]["abstractGameState"],
                "game_type": g.get("gameType"),
                "home_id": g["teams"]["home"]["team"]["id"],
                "away_id": g["teams"]["away"]["team"]["id"],
                "home_score": g["teams"]["home"].get("score"),
                "away_score": g["teams"]["away"].get("score"),
                "venue_id": (g.get("venue") or {}).get("id"),
                "day_night": g.get("dayNight"),
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

    Columns: pitcher, season, bf, k, bb, hbp, hr, er, gs, pitches, outs.
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
                            refresh: bool = False,
                            workers: int = 1) -> pd.DataFrame:
    """Per-appearance pitching lines for `pitcher_ids` in `season`.

    Columns: pitcher, season, date, game_pk, game_type, team, bf, k, bb, hbp,
    hr, er, gs, pitches, outs. One row per appearance — the caller filters to
    `date <` the game being predicted, which is what keeps the backtest
    walk-forward.

    `team` is the club he pitched for *that day*, which is what the bullpen
    model needs (a reliever traded in July belongs to one pen before the
    deadline and another after it, and only the appearances themselves say
    when the line moved).

    `workers` fetches that many logs at a time (see `_fetch_many`); the frame
    is assembled in pitcher-id order either way.
    """
    rows = []
    ids = sorted({int(p) for p in pitcher_ids})
    payloads = _fetch_many(ids, lambda pid: _get_cached(
        f"pitching_gamelog_{season}_{pid}", f"people/{pid}/stats",
        refresh=refresh, stats="gameLog", group="pitching", season=season,
    ), workers=workers)
    for pid, data in zip(ids, payloads):
        stats = data.get("stats", [])
        for s in (stats[0].get("splits", []) if stats else []):
            row = {"pitcher": pid, "season": season, "date": s.get("date"),
                   "game_pk": (s.get("game") or {}).get("gamePk"),
                   "game_type": s.get("gameType"),
                   "team": (s.get("team") or {}).get("id"),
                   "outs": _ip_to_outs(s["stat"].get("inningsPitched"))}
            for api_field, col in PITCHING_FIELDS.items():
                row[col] = s["stat"].get(api_field, 0) or 0
            rows.append(row)
    cols = ["pitcher", "season", "date", "game_pk", "game_type", "team",
            "outs", *PITCHING_FIELDS.values()]
    df = pd.DataFrame(rows, columns=cols)
    logger.info(f"{season} game logs: {len(df)} appearances "
                f"for {df['pitcher'].nunique() if len(df) else 0} pitchers")
    return df


# ─── Station B: rosters, IL status, and per-hitter plate appearances ───

# `position.type` on a roster entry. Everything that is not a Pitcher can
# come to the plate; "Hitter" is the designated hitter and "Two-Way Player"
# is Ohtani-shaped.
PITCHER_POSITION_TYPE = "Pitcher"


def _dedupe_roster(df: pd.DataFrame, keys) -> pd.DataFrame:
    """One row per player, keeping the most-available status.

    The roster endpoint occasionally returns a player twice for the same date
    — an option and a recall that both landed that day give an `A` row and an
    `RM` row (seen on 2026-07-01 and 2026-08-01, 2-6 players a day) — and a
    player traded that morning can briefly appear on both clubs' 40-mans.
    Left as-is he would be counted twice in his team's PA share. `A` wins,
    then the injured list, then everything else.
    """
    if df.empty:
        return df
    rank = df["status_code"].astype(str).map(
        lambda c: 0 if c == "A" else (1 if c.startswith("D") else 2))
    return (df.assign(_rank=rank).sort_values("_rank", kind="stable")
            .drop_duplicates(subset=list(keys), keep="first")
            .drop(columns="_rank").sort_index().reset_index(drop=True))


def fetch_teams(season: int, refresh: bool = False) -> pd.DataFrame:
    """The 30 MLB clubs for a season: team_id, abbrev, name, division_id."""
    data = _get_cached(f"teams_{season}", "teams", refresh=refresh,
                       sportId=1, season=season)
    rows = [{
        "team_id": t["id"],
        "abbrev": t.get("abbreviation"),
        "team": t.get("name"),
        "division_id": (t.get("division") or {}).get("id"),
    } for t in data.get("teams", [])]
    return pd.DataFrame(rows, columns=["team_id", "abbrev", "team", "division_id"])


def fetch_team_roster(team_id: int, date: str, roster_type: str = "40Man",
                      refresh: bool = False) -> pd.DataFrame:
    """One club's roster **as of `date`**, with each player's status that day.

    Columns: batter, team_id, date, status_code, status, position_code,
    position_type, is_hitter, note.

    `rosterType=40Man` with a historical `date` is the useful call: unlike
    `rosterType=active` it *keeps* the unavailable players and labels them, so
    the injured list falls out of the same request with no need for the
    transactions feed. Status codes seen in 2026: `A` active, `D7`/`D10`/
    `D15`/`D60` injured lists, `RM` optioned to the minors, `PL` paternity,
    `RL` restricted, `SU` suspended. Verified walk-forward: Aaron Judge is
    `A` on 2026-05-01 and `D60` on 2026-08-01, so the endpoint is answering
    "as of that date", not "today".

    Only past dates are cached — a roster for today or later is still moving.
    """
    is_past = pd.Timestamp(date).normalize() < pd.Timestamp.today().normalize()
    params = dict(rosterType=roster_type, date=date)
    if is_past:
        data = _get_cached(f"roster_{roster_type}_{team_id}_{date}",
                           f"teams/{team_id}/roster", refresh=refresh, **params)
    else:
        data = _get(f"teams/{team_id}/roster", **params)
    rows = []
    for entry in data.get("roster", []):
        position = entry.get("position") or {}
        status = entry.get("status") or {}
        rows.append({
            "batter": entry["person"]["id"],
            "team_id": team_id,
            "date": date,
            "status_code": status.get("code"),
            "status": status.get("description"),
            "position_code": position.get("code"),
            "position_type": position.get("type"),
            "is_hitter": position.get("type") != PITCHER_POSITION_TYPE,
            "note": entry.get("note"),
        })
    df = pd.DataFrame(rows, columns=[
        "batter", "team_id", "date", "status_code", "status",
        "position_code", "position_type", "is_hitter", "note"])
    return _dedupe_roster(df, ("batter", "team_id"))


def fetch_rosters(team_ids, date: str, roster_type: str = "40Man",
                  refresh: bool = False) -> pd.DataFrame:
    """`fetch_team_roster` over many clubs, concatenated and deduped."""
    frames = [fetch_team_roster(int(t), date, roster_type=roster_type, refresh=refresh)
              for t in team_ids]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df = _dedupe_roster(df, ("batter",))
    if len(df):
        logger.info(f"rosters {date}: {len(df)} players over {df['team_id'].nunique()} "
                    f"clubs, {int(df['is_hitter'].sum())} hitters, "
                    f"{int((df['status_code'] == 'A').sum())} active")
    return df


def fetch_team_hitting_game_logs(team_ids, season: int,
                                 refresh: bool = False) -> pd.DataFrame:
    """Team-level hitting line per game: team_id, date, game_pk, pa, runs.

    This is the denominator for station B — a team's plate appearances per
    game. Taking it from the team's own log rather than summing player logs
    means a hitter who has since been released or traded away still counts
    toward the games he played in, which is what makes the per-hitter shares
    normalize honestly.

    Caveat: the team gameLog splits carry no `gameType` field (the player
    logs do), but `season=YYYY` already restricts them to the regular season.
    """
    rows = []
    for tid in sorted({int(t) for t in team_ids}):
        data = _get_cached(
            f"team_hitting_gamelog_{season}_{tid}", f"teams/{tid}/stats",
            refresh=refresh, stats="gameLog", group="hitting",
            season=season, sportId=1,
        )
        stats = data.get("stats", [])
        for s in (stats[0].get("splits", []) if stats else []):
            rows.append({
                "team_id": tid,
                "season": season,
                "date": s.get("date"),
                "game_pk": (s.get("game") or {}).get("gamePk"),
                "pa": s["stat"].get("plateAppearances", 0) or 0,
                "runs": s["stat"].get("runs", 0) or 0,
            })
    df = pd.DataFrame(rows, columns=["team_id", "season", "date", "game_pk", "pa", "runs"])
    logger.info(f"{season} team hitting logs: {len(df)} team-games "
                f"for {df['team_id'].nunique() if len(df) else 0} clubs")
    return df


def fetch_hitter_game_logs(player_ids, season: int,
                           refresh: bool = False,
                           workers: int = 1) -> pd.DataFrame:
    """Per-game hitting lines for `player_ids` in `season`.

    Columns: batter, season, date, game_pk, game_type, team_id, pa, ab, plus
    the rest of HITTING_FIELDS. One row per game played — the caller filters
    to `date <` the cutoff, which is what keeps station B walk-forward.

    Cached per player-season under data/cache/statsapi/. The season is still
    in progress, so a cached log is only complete up to the day it was
    pulled; pass `refresh=True` (or `--refresh` on the build script) to
    re-pull. Same convention as `fetch_pitcher_game_logs`, `workers` included.
    """
    rows = []
    ids = sorted({int(p) for p in player_ids})
    payloads = _fetch_many(ids, lambda pid: _get_cached(
        f"hitting_gamelog_{season}_{pid}", f"people/{pid}/stats",
        refresh=refresh, stats="gameLog", group="hitting", season=season,
    ), workers=workers)
    for pid, data in zip(ids, payloads):
        stats = data.get("stats", [])
        for s in (stats[0].get("splits", []) if stats else []):
            row = {"batter": pid, "season": season, "date": s.get("date"),
                   "game_pk": (s.get("game") or {}).get("gamePk"),
                   "game_type": s.get("gameType"),
                   "team_id": (s.get("team") or {}).get("id")}
            for api_field, col in HITTING_FIELDS.items():
                row[col] = s["stat"].get(api_field, 0) or 0
            rows.append(row)
    cols = ["batter", "season", "date", "game_pk", "game_type", "team_id",
            *HITTING_FIELDS.values()]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        # gameType "R" is the regular season; spring/exhibition rows would
        # otherwise inflate an early-season share.
        df = df[df["game_type"].isna() | (df["game_type"] == "R")].reset_index(drop=True)
    logger.info(f"{season} hitting logs: {len(df)} player-games "
                f"for {df['batter'].nunique() if len(df) else 0} hitters")
    return df


# ─── Station B: transactions (injured-list and option spells) ───

TRANSACTION_COLUMNS = [
    "transaction_id", "player_id", "date", "effective_date", "resolution_date",
    "type_code", "type_desc", "description", "from_team_id", "to_team_id",
    "season",
]


def fetch_transactions(season: int, refresh: bool = False) -> pd.DataFrame:
    """Every roster transaction in a season, dated.

    Columns: TRANSACTION_COLUMNS. One row per transaction, sorted by date.

    `GET /transactions?startDate=&endDate=&sportId=1` is the only feed that
    carries the *dates* of an injured-list stint. The 40-man roster endpoint
    says a hitter is `D60` today; it cannot say when he went on the list or
    which list he went on, and both are what a return-time distribution has to
    be conditioned on (`src/projections/il_returns.py`).

    The event type is in `type_code` (`OPT` optioned, `CU` recalled, `SE`
    selected, `TR` trade, ...) except for the injured list, which the API files
    under the single code `SC` ("status change") and distinguishes only in the
    English `description`: "placed CF X on the 10-day injured list",
    "activated CF X from the 10-day injured list", "transferred RHP Y from the
    15-day injured list to the 60-day injured list". `il_returns.parse_events`
    owns that parsing.

    Requests are chunked by month and cached under data/cache/statsapi/. A
    finished season never changes; the current month of a running season does,
    so pass `refresh=True` (or `--refresh` on the build script) to re-pull —
    the same convention as the game logs.
    """
    rows = []
    for period in pd.period_range(start=f"{season}-01-01", end=f"{season}-12-31",
                                  freq="M"):
        lo, hi = period.start_time.date(), period.end_time.date()
        data = _get_cached(
            f"transactions_{lo}_{hi}", "transactions", refresh=refresh,
            sportId=1, startDate=str(lo), endDate=str(hi),
        )
        for t in data.get("transactions", []):
            person = t.get("person") or {}
            if not person.get("id"):
                continue
            rows.append({
                "transaction_id": t.get("id"),
                "player_id": person["id"],
                "date": t.get("date"),
                "effective_date": t.get("effectiveDate"),
                "resolution_date": t.get("resolutionDate"),
                "type_code": t.get("typeCode"),
                "type_desc": t.get("typeDesc"),
                "description": t.get("description"),
                "from_team_id": (t.get("fromTeam") or {}).get("id"),
                "to_team_id": (t.get("toTeam") or {}).get("id"),
                "season": season,
            })
    df = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    if len(df):
        df = (df.drop_duplicates(subset="transaction_id")
              .sort_values(["date", "transaction_id"]).reset_index(drop=True))
        for col in ("from_team_id", "to_team_id"):
            df[col] = df[col].astype("Int64")
    logger.info(f"{season} transactions: {len(df)} rows for "
                f"{df['player_id'].nunique() if len(df) else 0} players")
    return df


# ─── Station E: posted lineups and batter rates ───

# The live feed is ~750 KB per game; `fields` trims it to ~9 KB, which is what
# makes caching a whole season of lineups (~1,800 games) reasonable.
_LINEUP_FIELDS = ("gamePk,liveData,boxscore,teams,home,away,players,person,id,"
                  "battingOrder")


def posted_lineup(team_box: dict) -> list[int]:
    """The nine batters a club *started*, in batting order, from a boxscore.

    Read from the per-player `battingOrder` codes, not from the team-level
    `battingOrder` array. The codes are slot x 100 for a starter ("300" is the
    number-three hitter) and slot x 100 + n for the nth player to take over
    that slot ("301" is the pinch hitter who batted third). The team-level
    array is always nine long, but it holds the *last* occupant of each slot,
    so in roughly one slot per team per game it is a substitute.

    That distinction is the whole ballgame for a walk-forward backtest: who
    pinch-hit is a fact about how the game went (managers rest regulars in
    blowouts and hit for them when trailing), so the ending lineup leaks the
    result backwards. Measured on 2025, the ending lineup's distance from a
    club's own norm correlates **-0.06** with the runs that club scored —
    backwards — while the posted lineup does not. Only the "x00" entries were
    knowable before first pitch.

    Returns [] when the boxscore has no batting order at all (postponed games,
    feeds without a boxscore).
    """
    slots: dict[int, int] = {}
    for key, player in (team_box.get("players") or {}).items():
        code = player.get("battingOrder")
        if not code or not str(code).endswith("00"):
            continue
        pid = (player.get("person") or {}).get("id")
        if pid is None and key.startswith("ID"):
            pid = key[2:]
        try:
            slots[int(code) // 100] = int(pid)
        except (TypeError, ValueError):
            continue
    return [slots[s] for s in sorted(slots)] if len(slots) == 9 else []


def fetch_lineups(game_pks, refresh: bool = False,
                  pace: float = 0.02, workers: int = 1) -> pd.DataFrame:
    """Posted batting order for each game, one row per lineup slot.

    Columns: game_pk, side ("home"/"away"), slot (1-9), batter (MLBAM id).

    Source is the live feed's boxscore, `liveData.boxscore.teams.{home,away}`,
    reduced to the nine starters by `posted_lineup` (see there for why the
    team-level `battingOrder` array is the wrong thing to read).

    Walk-forward honesty: the starters are the card the club posted, typically
    2-4 hours before first pitch and always before the exchange of cards. The
    one thing this cannot see is a **late scratch** after the card went public,
    which the backfill silently absorbs. The exchanges' closes knew about those
    too — the median Kalshi close is 15 minutes before first pitch
    (docs/market-benchmark-2026.md) — so scoring against those closes is fair,
    exactly the argument `fetch_probables` makes for the starting pitcher. It
    is *not* a simulation of forecasting the morning before.

    Games with no posted lineup are simply absent, and the caller falls back to
    a lineup-free model.

    `pace` sleeps between *uncached* requests; a season is ~1,800 of them, and
    `workers` fetches that many at a time (`_fetch_many`).
    """
    def one(pk: int) -> dict:
        cache_file = STATSAPI_CACHE / f"lineup_{pk}.json"
        fresh = not cache_file.exists() or refresh
        try:
            data = _get_cached(f"lineup_{pk}", f"game/{pk}/feed/live",
                               refresh=refresh, base=BASE_V11,
                               fields=_LINEUP_FIELDS)
        except requests.HTTPError as exc:      # pragma: no cover - network
            logger.warning(f"lineup {pk}: {exc}")
            return {}
        if fresh and pace:
            time.sleep(pace)
        return data

    rows = []
    pks = sorted({int(p) for p in game_pks})
    for pk, data in zip(pks, _fetch_many(pks, one, workers=workers)):
        teams = ((data.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("home", "away"):
            for slot, batter in enumerate(posted_lineup(teams.get(side) or {}), start=1):
                rows.append({"game_pk": pk, "side": side, "slot": slot,
                             "batter": int(batter)})
    df = pd.DataFrame(rows, columns=["game_pk", "side", "slot", "batter"])
    logger.info(f"lineups: {df['game_pk'].nunique() if len(df) else 0} games, "
                f"{len(df)} slots")
    return df


def fetch_batter_game_logs(batter_ids, season: int, refresh: bool = False,
                           pace: float = 0.02) -> pd.DataFrame:
    """Per-game hitting lines for `batter_ids` in `season`.

    Columns: batter, season, date, game_pk, game_type, plus the counting
    columns in HITTING_FIELDS. One row per game — the caller filters to
    `date <` the game being predicted, which is what keeps the backtest
    walk-forward (`src.sim.lineups.games_before`).
    """
    rows = []
    for bid in sorted({int(b) for b in batter_ids}):
        cache_file = STATSAPI_CACHE / f"hitting_gamelog_{season}_{bid}.json"
        fresh = not cache_file.exists() or refresh
        try:
            data = _get_cached(
                f"hitting_gamelog_{season}_{bid}", f"people/{bid}/stats",
                refresh=refresh, stats="gameLog", group="hitting", season=season)
        except requests.HTTPError as exc:      # pragma: no cover - network
            logger.warning(f"hitting game log {bid}: {exc}")
            continue
        if fresh and pace:
            time.sleep(pace)
        stats = data.get("stats", [])
        for s in (stats[0].get("splits", []) if stats else []):
            row = {"batter": bid, "season": season, "date": s.get("date"),
                   "game_pk": (s.get("game") or {}).get("gamePk"),
                   "game_type": s.get("gameType")}
            for api_field, col in HITTING_FIELDS.items():
                row[col] = s["stat"].get(api_field, 0) or 0
            rows.append(row)
    cols = ["batter", "season", "date", "game_pk", "game_type",
            *HITTING_FIELDS.values()]
    df = pd.DataFrame(rows, columns=cols)
    logger.info(f"{season} hitting game logs: {len(df)} games for "
                f"{df['batter'].nunique() if len(df) else 0} batters")
    return df
