"""Build the strictly-pre-game feature table one season at a time.

    python scripts/build_game_features.py --seasons 2015-2026
    python scripts/build_game_features.py --seasons 2026 --workers 12

One parquet per season under `data/features/`, which is the checkpoint: a
season already on disk is skipped unless `--refresh-table` is given, so a run
that dies halfway through 2019 costs one season, not twelve.

Everything is fetched through `src/data/mlb_stats_api.py`, whose responses are
cached under `data/cache/statsapi/` (gitignored), so the second run of a season
is offline and byte-identical. The first run of a cold season is API-bound:
about 750 pitcher game logs, 650 hitter game logs and 2,400 posted lineups,
which `--workers` fetches in parallel exactly the way the nightly odds job
does (`_fetch_many`); the pacing convention — `pace` between *uncached*
requests only — is the one already in the fetchers.

The features themselves, and the walk-forward cut that makes them honest, are
`src/sim/game_features.py`; this script is the fetching around it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PARQUET_DIR
from src.data.mlb_stats_api import (
    SEASONS_PARQUET, build_seasons_table, fetch_hitter_game_logs, fetch_lineups,
    fetch_pitcher_game_logs, fetch_probables, fetch_schedule,
    fetch_season_pitching,
)
from src.sim import game_features as gf
from src.sim import game_model as gm
from src.sim.teams import fetch_teams

FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
# Marcel wants the two completed seasons before the one being built, and the
# season-level hitting table in the repository starts at 2015. Extending it
# backwards into its own file rather than refreshing the shared one keeps
# every other station's numbers exactly where they are: the rows for 2015+ are
# the same rows, byte for byte, and only 2013-2014 are new.
HITTER_SEASONS_EXT = PARQUET_DIR / "hitter_seasons_2013_2026.parquet"
PRIOR_SEASONS = 2


def season_path(season: int) -> Path:
    return FEATURES_DIR / f"game_features_{season}.parquet"


def hitter_seasons(start: int, end: int) -> pd.DataFrame:
    """Season-level hitting counts covering `start..end`, extending the shared table.

    `build_seasons_table` keeps one parquet and refetches the whole range
    whenever the range it is asked for is not covered — which for a 2015 build
    (Marcel wants 2013 and 2014) would rewrite the file every other station
    reads. So the older seasons go in a file of their own and are concatenated
    with the shared one, which leaves the shared rows untouched.
    """
    base = pd.read_parquet(SEASONS_PARQUET) if SEASONS_PARQUET.exists() \
        else pd.DataFrame(columns=["batter", "season"])
    have = int(base["season"].min()) if len(base) else end + 1
    if start >= have:
        return base[base["season"].between(start, end)]
    if not HITTER_SEASONS_EXT.exists():
        older = build_seasons_table(start, have - 1,
                                    cache_path=FEATURES_DIR / "_hitter_seasons_pre.parquet")
        ext = pd.concat([older, base], ignore_index=True)
        ext = ext.drop_duplicates(subset=["batter", "season"], keep="last")
        HITTER_SEASONS_EXT.parent.mkdir(parents=True, exist_ok=True)
        ext.to_parquet(HITTER_SEASONS_EXT, index=False)
    ext = pd.read_parquet(HITTER_SEASONS_EXT)
    return ext[ext["season"].between(start, end)]


def completed_games(season: int) -> pd.DataFrame:
    """Regular-season games that finished with a winner, in schedule order.

    The same population `scripts/backtest_game_odds.py` scores: status Final,
    game type R, and no tie (a suspended game called level has no home_win).
    """
    sched = fetch_schedule(f"{season}-03-01", f"{season}-11-15")
    scored = sched[sched["status"] == "Final"].dropna(
        subset=["home_score", "away_score"]).copy()
    scored = scored[scored["home_score"] != scored["away_score"]]
    scored = scored[scored["game_type"] == "R"].copy()
    scored["home_win"] = scored["home_score"] > scored["away_score"]
    return scored.sort_values("date").reset_index(drop=True)


def build_season(season: int, *, workers: int, min_games: int,
                 verbose: bool = True) -> pd.DataFrame:
    """Fetch a season and turn it into one feature row per completed game."""
    t0 = time.time()
    teams = fetch_teams(season)
    scored = completed_games(season)

    probables = fetch_probables(f"{season}-03-01", f"{season}-11-15")
    probables = probables.dropna(subset=["home_sp_id", "away_sp_id"])
    probables = probables[probables["game_pk"].isin(scored["game_pk"])]
    pmap = {int(r.game_pk): (int(r.home_sp_id), int(r.away_sp_id))
            for r in probables.itertuples(index=False)}

    lineups = fetch_lineups(scored["game_pk"], workers=workers)
    cards: dict[int, dict] = {}
    if len(lineups):
        for (pk, side), grp in lineups.sort_values("slot").groupby(["game_pk", "side"]):
            cards.setdefault(int(pk), {})[side] = [int(b) for b in grp["batter"]]
    cards = {pk: sides for pk, sides in cards.items()
             if len(sides.get("home", [])) == 9 and len(sides.get("away", [])) == 9}

    # The batter universe is every hitter who has appeared in a posted lineup,
    # which is what `scripts/backtest_game_odds.build_c_context` uses and covers
    # 99.98% of the league's plate appearances.
    batters = sorted({b for sides in cards.values() for ids in sides.values()
                      for b in ids})
    h_logs = fetch_hitter_game_logs(batters, season, workers=workers)
    pitchers = fetch_season_pitching(season)["pitcher"]
    p_logs = fetch_pitcher_game_logs(pitchers, season, workers=workers)
    p_logs = p_logs[p_logs["game_type"] == "R"]

    prior_p = pd.concat([fetch_season_pitching(y)
                         for y in range(season - PRIOR_SEASONS, season)],
                        ignore_index=True)
    prior_h = hitter_seasons(season - PRIOR_SEASONS, season - 1)
    inputs = gm.ChainInputs.from_logs(season, p_logs, h_logs, prior_p, prior_h)
    if verbose:
        print(f"{season}: {len(scored)} games, {len(pmap)} with probables, "
              f"{len(cards)} with a full card, {len(batters)} batters, "
              f"{p_logs['pitcher'].nunique()} pitchers "
              f"({time.time() - t0:.0f}s fetching)", flush=True)

    def progress(date: str, n_rows: int) -> None:
        if verbose and date.endswith(("-01", "-15")):
            print(f"  {date}: {n_rows} rows ({time.time() - t0:.0f}s)", flush=True)

    feats = gf.season_features(scored, teams["team_id"].to_numpy(), inputs,
                               probables=pmap, cards=cards, min_games=min_games,
                               progress=progress)
    if verbose:
        print(f"{season}: {len(feats)} feature rows in {time.time() - t0:.0f}s",
              flush=True)
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2015-2026",
                        help="a season, a comma list, or an inclusive range")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel Stats API fetches (the nightly uses 8)")
    parser.add_argument("--min-games", type=int, default=20,
                        help="skip dates until every club has this many games, "
                             "the same cut scripts/backtest_game_odds.py makes")
    parser.add_argument("--refresh-table", action="store_true",
                        help="rebuild a season already on disk")
    parser.add_argument("--out-dir", type=Path, default=FEATURES_DIR)
    args = parser.parse_args()

    seasons: list[int] = []
    for piece in str(args.seasons).split(","):
        if "-" in piece:
            lo, hi = piece.split("-")
            seasons += list(range(int(lo), int(hi) + 1))
        else:
            seasons.append(int(piece))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = args.out_dir / f"game_features_{season}.parquet"
        if path.exists() and not args.refresh_table:
            print(f"{season}: {path.name} exists, skipping", flush=True)
            continue
        feats = build_season(season, workers=args.workers,
                             min_games=args.min_games)
        if not len(feats):
            print(f"{season}: no rows, nothing written", flush=True)
            continue
        feats.to_parquet(path, index=False)
        print(f"wrote {path} ({len(feats)} rows)", flush=True)


if __name__ == "__main__":
    main()
