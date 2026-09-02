"""Station B — build (and score) projected rest-of-season plate appearances.

    # today's projection -> data/parquet/playing_time_ros.parquet
    python3 scripts/build_playing_time.py --cutoff 2026-09-02

    # walk-forward score: three methods x two cutoffs vs realized PA
    python3 scripts/build_playing_time.py --score

The math lives in `src/projections/playing_time.py` (pure functions over
DataFrames). This file is only the fetch/assemble layer: it pulls 40-man
rosters as of the cutoff, team and player hitting game logs, and the
remaining schedule from the Stats API, then hands frames to the model.

First run pulls ~700 player game logs and takes a few minutes; everything
lands in data/cache/statsapi/ (gitignored) so re-runs are instant. The season
is still in progress, so pass --refresh to re-pull stale logs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PARQUET_DIR
from src.data.mlb_stats_api import (
    fetch_hitter_game_logs,
    fetch_rosters,
    fetch_schedule,
    fetch_season_hitting,
    fetch_team_hitting_game_logs,
    fetch_teams,
)
from src.projections.playing_time import (
    METHODS,
    is_active,
    is_injured,
    project_playing_time,
    team_pa_per_game,
    walk_forward_scores,
)

SEASON = 2026
# Last day of the 2026 regular season (roadmap calendar anchors).
SEASON_END = "2026-09-27"
# Walk-forward cutoffs and the day realized PA is measured through.
SCORE_CUTOFFS = ("2026-07-01", "2026-08-01")
SCORE_END = "2026-09-02"
OUT_PARQUET = PARQUET_DIR / "playing_time_ros.parquet"

logger = logging.getLogger("build_playing_time")


def games_remaining(schedule: pd.DataFrame, cutoff: str, end: str) -> pd.DataFrame:
    """Regular-season games each club still has to play in [cutoff, end]."""
    dates = pd.to_datetime(schedule["date"]).dt.normalize()
    window = schedule[(dates >= pd.Timestamp(cutoff)) & (dates <= pd.Timestamp(end))]
    window = window[window["game_type"] == "R"]
    counts = (pd.concat([window["home_id"], window["away_id"]])
              .value_counts().rename_axis("team_id")
              .reset_index(name="games_remaining"))
    return counts


def hitter_universe(rosters: pd.DataFrame, season_hitting: pd.DataFrame) -> list[int]:
    """Every hitter who could plausibly appear in the projection or the score.

    Two sources, unioned:
      * non-pitchers on any of the fetched 40-man snapshots. The snapshot at a
        date carries optioned and injured players too, so taking the union
        across the cutoff dates *and* the scoring end date covers call-ups who
        only debut after a cutoff.
      * anyone with a plate appearance in the season at all — this catches the
        hitter who was up in July, designated for assignment in August, and is
        on nobody's 40-man by the scoring date. His PA are real; leaving him
        out would hand every method free credit for not projecting him.
    """
    ids = set(rosters.loc[rosters["is_hitter"], "batter"].astype(int))
    played = season_hitting.groupby("batter")["pa"].sum()
    return sorted(ids | set(played[played > 0].index.astype(int)))


def load(cutoffs, refresh: bool = False):
    """Pull every frame the model needs. Returns (rosters_by_date, logs, team_logs)."""
    teams = fetch_teams(SEASON, refresh=refresh)
    team_ids = teams["team_id"].tolist()
    rosters = {d: fetch_rosters(team_ids, d, refresh=refresh) for d in cutoffs}
    all_rosters = pd.concat(rosters.values(), ignore_index=True)
    batters = hitter_universe(all_rosters, fetch_season_hitting(SEASON))
    logger.info(f"fetching game logs for {len(batters)} hitters")
    logs = fetch_hitter_game_logs(batters, SEASON, refresh=refresh)
    team_logs = fetch_team_hitting_game_logs(team_ids, SEASON, refresh=refresh)
    hitters = {d: r[r["is_hitter"]].reset_index(drop=True) for d, r in rosters.items()}
    return hitters, logs, team_logs, teams


def build(cutoff: str, refresh: bool = False) -> pd.DataFrame:
    rosters, logs, team_logs, teams = load([cutoff], refresh=refresh)
    schedule = fetch_schedule(cutoff, SEASON_END)
    remaining = games_remaining(schedule, cutoff, SEASON_END)
    roster = rosters[cutoff]
    proj = project_playing_time(roster, logs, remaining, cutoff,
                                team_logs=team_logs, method="last_30")

    active = roster["status_code"].map(is_active)
    n_active, n_il = int(active.sum()), int(roster["status_code"].map(is_injured).sum())
    seen = set(logs.loc[pd.to_datetime(logs["date"]) < pd.Timestamp(cutoff), "batter"])
    n_no_history = int((active & ~roster["batter"].isin(seen)).sum())
    logger.info(f"{len(roster)} hitters on 40-man rosters: {n_active} active, "
                f"{n_il} on the IL, {len(roster) - n_active - n_il} optioned or "
                f"otherwise unavailable, {n_no_history} active with no prior PA "
                f"(bench default)")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    proj.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}: {len(proj)} hitters, "
          f"{int((proj['projected_pa_ros'] > 0).sum())} with PA > 0")

    ppg = team_pa_per_game(team_logs, cutoff).set_index("team_id")["pa_per_game"]
    summary = (proj.groupby("team_id")
               .agg(hitters=("batter", "size"),
                    projected=("projected_pa_ros", lambda s: (s > 0).sum()),
                    games_remaining=("games_remaining", "first"),
                    team_pa_ros=("projected_pa_ros", "sum"),
                    top_pa=("projected_pa_ros", "max"))
               .join(teams.set_index("team_id")["abbrev"])
               .join(ppg.rename("pa_per_game")))
    summary = summary.sort_values("abbrev")
    print(f"\nPer-team summary (cutoff {cutoff}, through {SEASON_END}):")
    print(summary.round(1).to_string())
    return proj


def score(refresh: bool = False) -> pd.DataFrame:
    cutoffs = list(SCORE_CUTOFFS)
    rosters, logs, team_logs, _ = load(cutoffs + [SCORE_END], refresh=refresh)
    schedule = fetch_schedule(min(cutoffs), SCORE_END)
    remaining = {c: games_remaining(schedule, c, SCORE_END) for c in cutoffs}
    # The scoring horizon is cutoff -> SCORE_END, not cutoff -> end of season,
    # so projections and realizations cover the same games.
    table = walk_forward_scores(
        {c: rosters[c] for c in cutoffs}, logs, team_logs, remaining,
        SCORE_END, methods=METHODS)
    print(f"\nWalk-forward: projected at each cutoff from data strictly before it, "
          f"scored on realized PA through {SCORE_END}.")
    print(table.round(3).to_string(index=False))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", default=pd.Timestamp.today().date().isoformat(),
                        help="projection date, YYYY-MM-DD (default: today)")
    parser.add_argument("--score", action="store_true",
                        help=f"run the walk-forward evaluation at {SCORE_CUTOFFS}")
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull cached Stats API responses")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.score:
        score(refresh=args.refresh)
    else:
        build(args.cutoff, refresh=args.refresh)


if __name__ == "__main__":
    main()
