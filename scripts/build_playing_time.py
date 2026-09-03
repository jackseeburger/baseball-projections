"""Station B — build (and score) projected rest-of-season plate appearances.

    # today's projection -> data/parquet/playing_time_ros.parquet
    python3 scripts/build_playing_time.py --cutoff 2026-09-02

    # walk-forward score: five methods x two cutoffs vs realized PA
    python3 scripts/build_playing_time.py --score

    # the 2025 selection curve the blend's parameters were chosen from
    python3 scripts/build_playing_time.py --sweep --season 2025

    # the injured-list return-time distribution, fitted on 2023-2025
    python3 scripts/build_playing_time.py --il-table

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
    fetch_transactions,
)
from src.projections import il_returns
from src.projections.playing_time import (
    BLEND_ANCHOR_GAMES,
    BLEND_WEIGHT_LONG,
    BLEND_WEIGHT_SHORT,
    METHODS,
    PRODUCTION_METHOD,
    USE_IL_RETURNS,
    logistic_from_anchors,
    absolute_errors,
    horizon_weight,
    is_active,
    is_injured,
    paired_difference,
    project_playing_time,
    score_projection,
    team_pa_per_game,
    walk_forward_projections,
    walk_forward_scores,
)

SEASON = 2026
# Last day of the 2026 regular season (roadmap calendar anchors).
SEASON_END = "2026-09-27"
OUT_PARQUET = PARQUET_DIR / "playing_time_ros.parquet"

# Walk-forward cutoffs and the day realized PA is measured through, per season.
#
# 2026 is the *evaluation* season: two cutoffs, 63 and 32 days of horizon,
# scored through the day the station was built. 2025 is the *selection*
# season -- the blend's two parameters are chosen there and nowhere else. It
# carries a denser grid of cutoffs on purpose: the quantity being fitted is a
# function of the horizon, so it needs more than two horizons to be identified,
# and the 2026 pair (63 and 32 days) has to sit inside the fitted range rather
# than off the end of it. The 2025 grid spans 15 to 107 days.
SCORE_SEASONS = {
    2026: {
        "cutoffs": ("2026-07-01", "2026-08-01"),
        "score_end": "2026-09-02",
    },
    2025: {
        "cutoffs": ("2025-06-15", "2025-07-01", "2025-07-15", "2025-08-01",
                    "2025-08-15", "2025-09-01", "2025-09-15"),
        "score_end": "2025-09-30",
    },
}
SCORE_CUTOFFS = SCORE_SEASONS[SEASON]["cutoffs"]
SCORE_END = SCORE_SEASONS[SEASON]["score_end"]

# Constant blend weights the sweep traces MAE against at each cutoff.
SWEEP_WEIGHTS = tuple(i / 20 for i in range(21))

# Last day of each regular season, from the schedule feed. A spell still open
# on that day is where the return-time distribution censors: he did not come
# back, but the season ran out before we could find out whether he would.
REGULAR_SEASON_END = {
    2022: "2022-10-05", 2023: "2023-10-02", 2024: "2024-09-30",
    2025: "2025-09-28", 2026: "2026-09-27",
}

# The return-time distribution is estimated on the three seasons *before* the
# one being projected and never on the season itself: 2023-2025 for the 2026
# table, 2022-2024 for the 2025 one. Three seasons is about 3,000 injured-list
# stints and 4,500 options, which is enough for a day-by-day curve per list
# type and recent enough to reflect the current lists (the 15-day position-
# player list became a 10-day list in 2020, and the 2023-2025 curves already
# show that).
IL_FIT_SEASONS = 3

logger = logging.getLogger("build_playing_time")


def il_fit_seasons(season: int = SEASON) -> tuple[int, ...]:
    """The seasons the return-time distribution for `season` is fitted on."""
    return tuple(range(season - IL_FIT_SEASONS, season))


def il_survival(season: int = SEASON, refresh: bool = False) -> pd.DataFrame:
    """The Kaplan-Meier return-time table for projecting `season`."""
    seasons = il_fit_seasons(season)
    tx = pd.concat([fetch_transactions(s, refresh=refresh) for s in seasons],
                   ignore_index=True)
    table = il_returns.fit(tx, REGULAR_SEASON_END)
    logger.info(f"return-time distribution fitted on {seasons}: "
                f"{table['type'].nunique()} spell types, {len(table)} rows")
    return table


def active_fractions(roster: pd.DataFrame, events: pd.DataFrame,
                     table: pd.DataFrame, cutoff: str, horizon_end: str):
    """Expected share of the horizon each unavailable hitter is back for."""
    horizon_days = (pd.Timestamp(horizon_end) - pd.Timestamp(cutoff)).days
    return il_returns.expected_active_fractions(roster, events, table, cutoff,
                                                horizon_days)


def production_fractions(roster: pd.DataFrame, cutoff: str, horizon_end: str,
                         season: int = SEASON, refresh: bool = False):
    """The expected-return fractions `PRODUCTION_METHOD` needs, or None.

    The one place that says how the *live* projection is built, so the ROS
    document the site serves and this script's own `--cutoff` build cannot
    drift apart. Returns None when the flag is off, which is exactly what
    `project_playing_time` wants in order to fall back to the hard roster
    gate.
    """
    if not USE_IL_RETURNS:
        return None
    events = il_returns.parse_events(fetch_transactions(season, refresh=refresh))
    fractions = active_fractions(roster, events, il_survival(season, refresh),
                                 cutoff, horizon_end)
    logger.info(f"expected returns: {len(fractions)} unavailable hitters dated, "
                f"mean expected active fraction "
                f"{fractions['active_fraction'].mean():.3f}")
    return fractions


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


def load(cutoffs, season: int = SEASON, refresh: bool = False):
    """Pull every frame the model needs. Returns (rosters_by_date, logs, team_logs)."""
    teams = fetch_teams(season, refresh=refresh)
    team_ids = teams["team_id"].tolist()
    rosters = {d: fetch_rosters(team_ids, d, refresh=refresh) for d in cutoffs}
    all_rosters = pd.concat(rosters.values(), ignore_index=True)
    batters = hitter_universe(all_rosters, fetch_season_hitting(season))
    logger.info(f"fetching game logs for {len(batters)} hitters")
    logs = fetch_hitter_game_logs(batters, season, refresh=refresh)
    team_logs = fetch_team_hitting_game_logs(team_ids, season, refresh=refresh)
    hitters = {d: r[r["is_hitter"]].reset_index(drop=True) for d, r in rosters.items()}
    return hitters, logs, team_logs, teams


def build(cutoff: str, refresh: bool = False) -> pd.DataFrame:
    rosters, logs, team_logs, teams = load([cutoff], refresh=refresh)
    schedule = fetch_schedule(cutoff, SEASON_END)
    remaining = games_remaining(schedule, cutoff, SEASON_END)
    roster = rosters[cutoff]
    fractions = production_fractions(roster, cutoff, SEASON_END, refresh=refresh)
    proj = project_playing_time(roster, logs, remaining, cutoff,
                                team_logs=team_logs, method=PRODUCTION_METHOD,
                                active_fractions=fractions)

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


def _frames_for(season: int, refresh: bool = False):
    """Rosters, logs, team logs, schedule-derived games remaining for a season."""
    cfg = SCORE_SEASONS[season]
    cutoffs, score_end = list(cfg["cutoffs"]), cfg["score_end"]
    rosters, logs, team_logs, _ = load(cutoffs + [score_end], season=season,
                                       refresh=refresh)
    schedule = fetch_schedule(min(cutoffs), score_end)
    # The scoring horizon is cutoff -> score_end, not cutoff -> end of season,
    # so projections and realizations cover the same games.
    remaining = {c: games_remaining(schedule, c, score_end) for c in cutoffs}
    rosters = {c: rosters[c] for c in cutoffs}
    # The expected-return fractions for `blend_il`, one frame per cutoff. The
    # survival table comes from the seasons *before* this one and the spell
    # dates from this season's transactions strictly before each cutoff, so
    # nothing a projection could not have known enters here.
    events = il_returns.parse_events(fetch_transactions(season, refresh=refresh))
    table = il_survival(season, refresh=refresh)
    fractions = {c: active_fractions(rosters[c], events, table, c, score_end)
                 for c in cutoffs}
    return rosters, logs, team_logs, remaining, score_end, fractions


def score(season: int = SEASON, refresh: bool = False) -> pd.DataFrame:
    """The headline table: every method at every cutoff, plus the paired tests."""
    rosters, logs, team_logs, remaining, score_end, fractions = _frames_for(
        season, refresh)
    table = walk_forward_scores(rosters, logs, team_logs, remaining, score_end,
                                methods=METHODS,
                                active_fractions_by_cutoff=fractions)
    print(f"\nWalk-forward: projected at each cutoff from data strictly before it, "
          f"scored on realized PA through {score_end}.")
    print(table.round(3).to_string(index=False))

    print(f"\nExpected returns fitted on {il_fit_seasons(season)}, applied at each "
          f"cutoff (blend_il):")
    for cutoff in sorted(fractions):
        f = fractions[cutoff]
        by_type = (f.groupby("spell_type")
                   .agg(n=("active_fraction", "size"),
                        median_elapsed=("elapsed_days", "median"),
                        mean_fraction=("active_fraction", "mean")))
        n_out = int((rosters[cutoff]["status_code"] != "A").sum())
        print(f"  {cutoff}: {len(f)} of {n_out} unavailable hitters dated")
        print("    " + by_type.round(3).to_string().replace("\n", "\n    "))

    # MAE differences hitter by hitter. The methods saw the same players and
    # the same season, so the paired SE is the honest one — most of the
    # variance in either MAE is common and cancels.
    rows = []
    for cutoff, projections, real, universe in walk_forward_projections(
            rosters, logs, team_logs, remaining, score_end, methods=METHODS,
            active_fractions_by_cutoff=fractions):
        errors = {m: absolute_errors(p, real, universe=universe)
                  for m, p in projections.items()}
        horizon = (pd.Timestamp(score_end) - pd.Timestamp(cutoff)).days
        for focus, others in (("blend", ("last_30", "season_share", "uniform")),
                              ("blend_il", ("blend", "last_30", "season_share"))):
            for other in others:
                d = paired_difference(errors[focus], errors[other])
                rows.append({"cutoff": str(cutoff), "horizon_days": horizon,
                             "method": focus, "vs": other, "n": d["n"],
                             "mean_mae_diff": d["mean"], "se": d["se"], "t": d["t"]})
    paired = pd.DataFrame(rows)
    print("\nPaired per-hitter MAE difference (the method minus the one it is "
          "compared against; negative means the method is better):")
    print(paired.round(3).to_string(index=False))
    return table


def sweep(season: int = SEASON, refresh: bool = False) -> pd.DataFrame:
    """MAE against a *constant* blend weight at each cutoff, and the fit.

    This is the selection procedure, and it is run on 2025 only. At each
    cutoff the horizon is fixed, so tracing MAE against a constant `w` gives
    the curve `MAE_cutoff(w)`; the best `w` at that cutoff is the argmin. Two
    parameters (midpoint, scale) are then chosen to minimize the mean over
    cutoffs of `MAE_cutoff(w(h))`, with `MAE_cutoff` interpolated between the
    swept grid points and `h` the club-median games remaining at that cutoff.
    Nothing about 2026 enters.
    """
    rosters, logs, team_logs, remaining, score_end, _ = _frames_for(season, refresh)
    rows, curves, horizons = [], {}, {}
    for cutoff, projections, real, universe in walk_forward_projections(
            rosters, logs, team_logs, remaining, score_end,
            methods=("season_share", "last_30"), blend_weights=SWEEP_WEIGHTS):
        h = float(remaining[cutoff]["games_remaining"].median())
        horizons[cutoff] = h
        maes = {}
        for name, proj in projections.items():
            mae = score_projection(proj, real, universe=universe)["mae"]
            if name.startswith("blend@"):
                maes[float(name.split("@")[1])] = mae
            rows.append({"cutoff": str(cutoff), "horizon_days":
                         (pd.Timestamp(score_end) - pd.Timestamp(cutoff)).days,
                         "games_remaining": h, "method": name, "mae": mae})
        curves[cutoff] = pd.Series(maes).sort_index()

    table = pd.DataFrame(rows)
    print(f"\n{season} selection sweep: MAE against a constant blend weight, "
          f"scored through {score_end}.")
    grid = (table[table["method"].str.startswith("blend@")]
            .assign(w=lambda d: d["method"].str.split("@").str[1].astype(float))
            .pivot(index="w", columns="cutoff", values="mae"))
    print(grid.round(2).to_string())

    best = pd.DataFrame([{
        "cutoff": str(c), "games_remaining": horizons[c],
        "horizon_days": (pd.Timestamp(score_end) - pd.Timestamp(c)).days,
        "best_w": float(curve.idxmin()), "best_mae": float(curve.min()),
        "mae_at_w1": float(curve.loc[1.0]), "mae_at_w0": float(curve.loc[0.0]),
    } for c, curve in curves.items()]).sort_values("games_remaining")
    print("\nBest constant weight per cutoff (the selection curve):")
    print(best.round(3).to_string(index=False))

    w_short, w_long, loss = _fit_logistic(curves, horizons)
    midpoint, scale = logistic_from_anchors(w_short, w_long)
    h_short, h_long = BLEND_ANCHOR_GAMES
    print(f"\nLogistic fit on {season} only: w({h_short:.0f} games)={w_short:.2f}, "
          f"w({h_long:.0f} games)={w_long:.2f} "
          f"(midpoint {midpoint:.0f} games, scale {scale:.0f}); "
          f"mean excess over each cutoff's own best weight {100 * (loss - 1):.2f}%")
    print("  w(h) at the fitted parameters: " + ", ".join(
        f"h={int(h)} -> {horizon_weight(h, midpoint, scale):.2f}"
        for h in (15, 28, 45, 55, 75, 95)))
    print(f"  in use: w={BLEND_WEIGHT_SHORT} / {BLEND_WEIGHT_LONG} at "
          f"{h_short:.0f} / {h_long:.0f} games")
    return table


def _fit_logistic(curves: dict, horizons: dict):
    """Grid-search the blend's two anchor weights against the swept curves.

    The parameters are searched as `w` at 30 and at 90 games remaining (with
    `w(30) > w(90)`, so the curve is decreasing) rather than as a midpoint and
    a scale, because the anchors are bounded, interpretable and identified
    while the midpoint of the fitted curve is not.

    The objective is the mean over cutoffs of `MAE(w(h)) / min_w MAE(w)` — each
    cutoff's MAE measured against the best that cutoff could have done with a
    constant weight. Mean raw MAE would be the wrong objective: MAE scales with
    the horizon (about 80 PA at three months, 8 at two weeks), so a raw-MAE fit
    is a fit to the longest cutoff alone, which is precisely the horizon
    dependence being estimated.
    """
    import numpy as np

    grid = np.arange(0.03, 0.98, 0.01)
    best = (None, None, float("inf"))
    for w_short in grid:
        for w_long in grid:
            if w_long > w_short - 0.005:
                continue
            midpoint, scale = logistic_from_anchors(float(w_short), float(w_long))
            excess = []
            for cutoff, curve in curves.items():
                w = float(horizon_weight(horizons[cutoff], midpoint, scale))
                mae = float(np.interp(w, curve.index.to_numpy(float),
                                      curve.to_numpy(float)))
                excess.append(mae / float(curve.min()))
            mean = float(np.mean(excess))
            if mean < best[2]:
                best = (round(float(w_short), 2), round(float(w_long), 2), mean)
    return best


# Days the printed return-time table reports the survival curve at. The whole
# curve is a few hundred rows per type; these are the ones worth reading.
IL_TABLE_DAYS = (7, 10, 14, 21, 30, 45, 60, 90)


def il_table(season: int = SEASON, refresh: bool = False) -> pd.DataFrame:
    """Print the return-time distribution the projection reads.

    `S(d)` is the probability a spell of that type is still running after `d`
    days; `P(back within a month)` reads the same curve conditionally for a
    hitter who went on the list today.
    """
    seasons = il_fit_seasons(season)
    tx = pd.concat([fetch_transactions(s, refresh=refresh) for s in seasons],
                   ignore_index=True)
    spells = pd.concat(
        [il_returns.build_spells(il_returns.parse_events(tx[tx["season"] == s]),
                                 REGULAR_SEASON_END[s]) for s in seasons],
        ignore_index=True)
    table = il_returns.survival_table(spells)
    counts = spells.groupby("type").agg(
        spells=("returned", "size"), returns=("returned", "sum"),
        median_days=("exit_day", "median"))
    counts["censored"] = counts["spells"] - counts["returns"]
    print(f"\nSpells from {seasons}, censored at the end of each regular season:")
    print(counts.to_string())

    curve = pd.DataFrame(
        {f"S({d})": {t: float(il_returns.survival_at(table, t, d))
                     for t in sorted(table["type"].unique())}
         for d in IL_TABLE_DAYS})
    curve["P(back <= 30d | out now)"] = [
        1.0 - float(il_returns.survival_at(table, t, 30))
        for t in curve.index]
    print("\nSurvival S(d) — the chance a spell is still running after d days:")
    print(curve.round(3).to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", default=pd.Timestamp.today().date().isoformat(),
                        help="projection date, YYYY-MM-DD (default: today)")
    parser.add_argument("--score", action="store_true",
                        help=f"run the walk-forward evaluation at {SCORE_CUTOFFS}")
    parser.add_argument("--sweep", action="store_true",
                        help="trace MAE against a constant blend weight and "
                             "refit the horizon logistic (selection: 2025)")
    parser.add_argument("--season", type=int, default=SEASON,
                        choices=sorted(SCORE_SEASONS),
                        help="season to score or sweep (default: %(default)s)")
    parser.add_argument("--il-table", action="store_true",
                        help="print the injured-list / option return-time "
                             "distribution the projection reads")
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull cached Stats API responses")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.il_table:
        il_table(season=args.season, refresh=args.refresh)
    elif args.sweep:
        sweep(season=args.season, refresh=args.refresh)
    elif args.score:
        score(season=args.season, refresh=args.refresh)
    else:
        build(args.cutoff, refresh=args.refresh)


if __name__ == "__main__":
    main()
