"""Station B-pitchers — assemble the inputs the workload backtest is scored on.

    # pull one season's appearances, rosters, club games and transactions
    python3 scripts/build_pitcher_workload.py --fetch --season 2024

    # every season the backtest uses
    python3 scripts/build_pitcher_workload.py --fetch --all

Everything lands in `data/workload/` as parquet, committed, because these are
~1,400 Stats API calls a season and the backtest has to be reproducible without
them. The cache under `data/cache/statsapi/` is the first line; these files are
the second.

Per season:

    pitcher_appearances_{season}.parquet   pitcher, date, game_pk, team, bf,
                                           outs, gs, k, bb, hr — one row per
                                           regular-season appearance
    pitcher_rosters_{season}.parquet       pitcher, team_id, cutoff,
                                           status_code — the 40-man snapshot
                                           at each walk-forward cutoff, with
                                           the injured/optioned labelled
    team_games_{season}.parquet            team_id, date, game_pk — the club
                                           games either side of a cutoff
    transactions_{season}.parquet          the dated roster moves the
                                           injured-list return-time
                                           distribution is fitted from

The cutoffs are the 1st and the 15th of May through September, which is what
"weekly or biweekly as-of dates" costs here: nine per season, five seasons,
and every method scored at every one of them.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import DATA_DIR
from src.data.mlb_stats_api import (
    fetch_pitcher_game_logs,
    fetch_schedule,
    fetch_season_pitching,
    fetch_team_hitting_game_logs,
    fetch_team_roster,
    fetch_teams,
    fetch_transactions,
)

WORKLOAD_DIR = DATA_DIR / "workload"

# Seasons the backtest scores on. 2020 is left out everywhere: a 60-game
# season has no rest-of-season horizon worth projecting and its injured-list
# spells censor at a different length from every other year's.
SCORE_SEASONS = (2022, 2023, 2024, 2025, 2026)
# Seasons the injured-list return-time distribution may be fitted from. Each
# score season reads the three before it, minus 2020.
FIT_SEASONS = (2019, 2021, 2022, 2023, 2024, 2025)
# Seasons whose appearance log is pulled purely so the `last_season` baseline
# has something to read in the first scored season.
PRIOR_ONLY_SEASONS = (2021,)
ALL_SEASONS = tuple(sorted(set(SCORE_SEASONS) | set(FIT_SEASONS)))

# Walk-forward as-of dates: the 1st and the 15th, May through September. A
# cutoff whose horizon is shorter than `MIN_HORIZON_DAYS` is dropped, which is
# what trims 2026 (whose data ends in early September).
CUTOFF_DAYS = ("05-01", "05-15", "06-01", "06-15", "07-01", "07-15",
               "08-01", "08-15", "09-01")
MIN_HORIZON_DAYS = 14

logger = logging.getLogger("build_pitcher_workload")


def cutoffs_for(season: int, score_end: str) -> list[str]:
    """The as-of dates for a season, dropping any with too short a horizon."""
    end = pd.Timestamp(score_end)
    out = []
    for day in CUTOFF_DAYS:
        c = pd.Timestamp(f"{season}-{day}")
        if (end - c).days >= MIN_HORIZON_DAYS:
            out.append(c.date().isoformat())
    return out


def _path(kind: str, season: int) -> Path:
    return WORKLOAD_DIR / f"{kind}_{season}.parquet"


# --- the fetch ---------------------------------------------------------

def fetch_season(season: int, refresh: bool = False, workers: int = 12) -> dict:
    """Pull and write every frame the backtest needs for one season."""
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    teams = fetch_teams(season, refresh=refresh)
    team_ids = teams["team_id"].tolist()

    # Club games. The denominator of every appearance rate, and the source of
    # both "games played before the cutoff" and "games left after it".
    team_logs = fetch_team_hitting_game_logs(team_ids, season, refresh=refresh)
    team_games = team_logs.loc[:, ["team_id", "date", "game_pk"]].copy()
    team_games.to_parquet(_path("team_games", season), index=False)
    score_end = str(pd.to_datetime(team_games["date"]).max().date())
    logger.info("%s: %d club games, last %s", season, len(team_games), score_end)

    # Every pitcher who threw a pitch in the season, and his appearances.
    season_pitching = fetch_season_pitching(season, refresh=refresh)
    ids = sorted(season_pitching["pitcher"].astype(int))
    logs = fetch_pitcher_game_logs(ids, season, refresh=refresh, workers=workers)
    logs = logs[logs["game_type"] == "R"]
    keep = ["pitcher", "date", "game_pk", "team", "bf", "outs", "gs",
            "k", "bb", "hr", "pitches"]
    appearances = logs.loc[:, [c for c in keep if c in logs.columns]].copy()
    appearances.to_parquet(_path("pitcher_appearances", season), index=False)
    logger.info("%s: %d appearances for %d pitchers", season,
                len(appearances), appearances["pitcher"].nunique())

    # The 40-man snapshot at each cutoff, pitchers only. This is what says who
    # is on a staff at all, and who is hurt or optioned on the morning the
    # projection is made.
    cutoffs = cutoffs_for(season, score_end)
    frames = []
    for cutoff in cutoffs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rosters = list(pool.map(
                lambda t: fetch_team_roster(int(t), cutoff, refresh=refresh),
                team_ids))
        roster = pd.concat(rosters, ignore_index=True)
        roster = roster[~roster["is_hitter"]]
        frames.append(pd.DataFrame({
            "pitcher": roster["batter"].astype("int64"),
            "team_id": roster["team_id"].astype("int64"),
            "cutoff": cutoff,
            "status_code": roster["status_code"].astype(str),
        }))
        logger.info("%s roster %s: %d pitchers, %d active", season, cutoff,
                    len(frames[-1]), int((frames[-1]["status_code"] == "A").sum()))
    rosters = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["pitcher", "team_id", "cutoff", "status_code"])
    # A pitcher traded that morning can appear on two clubs' 40-mans; keep the
    # one whose status is the most available, exactly as the roster fetcher does.
    rank = rosters["status_code"].map(
        lambda c: 0 if c == "A" else (1 if c.startswith("D") else 2))
    rosters = (rosters.assign(_rank=rank)
               .sort_values(["cutoff", "_rank"], kind="stable")
               .drop_duplicates(subset=["cutoff", "pitcher"], keep="first")
               .drop(columns="_rank").reset_index(drop=True))
    rosters.to_parquet(_path("pitcher_rosters", season), index=False)

    # The schedule, for games remaining as the projection would have seen it.
    schedule = fetch_schedule(f"{season}-03-01", f"{season}-11-15")
    schedule = schedule[schedule["game_type"] == "R"]
    schedule.loc[:, ["date", "game_pk", "home_id", "away_id"]].to_parquet(
        _path("schedule", season), index=False)

    tx = fetch_transactions(season, refresh=refresh)
    tx.to_parquet(_path("transactions", season), index=False)
    logger.info("%s: %d transactions", season, len(tx))
    return {"season": season, "score_end": score_end, "cutoffs": cutoffs,
            "appearances": len(appearances), "rosters": len(rosters)}


def fetch_appearances_only(season: int, refresh: bool = False,
                           workers: int = 12) -> int:
    """Just the appearance log, for a season nothing is scored at.

    2021 is here for one reason: it is the prior season the `last_season`
    baseline reads when 2022 is scored, and without it that baseline would be
    a column of zeros in 2022 — indistinguishable from the no-model floor, and
    an unfairly easy thing to beat.
    """
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(fetch_season_pitching(season, refresh=refresh)["pitcher"].astype(int))
    logs = fetch_pitcher_game_logs(ids, season, refresh=refresh, workers=workers)
    logs = logs[logs["game_type"] == "R"]
    keep = ["pitcher", "date", "game_pk", "team", "bf", "outs", "gs",
            "k", "bb", "hr", "pitches"]
    out = logs.loc[:, [c for c in keep if c in logs.columns]].copy()
    out.to_parquet(_path("pitcher_appearances", season), index=False)
    logger.info("%s: %d appearances (prior-season baseline only)", season, len(out))
    return len(out)


def fetch_transactions_only(season: int, refresh: bool = False) -> int:
    """A fit-only season needs its transactions and nothing else."""
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tx = fetch_transactions(season, refresh=refresh)
    tx.to_parquet(_path("transactions", season), index=False)
    logger.info("%s: %d transactions (fit-only season)", season, len(tx))
    return len(tx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.fetch:
        ap.error("nothing to do; pass --fetch")
    seasons = ALL_SEASONS if args.all else (args.season,)
    for season in seasons:
        if season in SCORE_SEASONS:
            print(fetch_season(season, refresh=args.refresh, workers=args.workers))
        else:
            fetch_transactions_only(season, refresh=args.refresh)
            if season in PRIOR_ONLY_SEASONS:
                fetch_appearances_only(season, refresh=args.refresh,
                                       workers=args.workers)


if __name__ == "__main__":
    main()
