"""Park factors — the run environment of the ballpark the game is played in.

Every rate in the chain above this module is either a *record of what
happened* (station D's season-to-date runs scored and allowed) or a
*component estimate that cannot see the ballpark* (station C's bottom-up
rebuild: FIP for the staff, linear weights for the bats). Neither one knows
that Coors Field turns a 4.4-run game into a 5.5-run game and that Petco does
the reverse, and the difference between those two facts is part of the
residual docs/playoff-odds-validation.md names: Atlanta allows 4.02 runs a
game while its components say 4.38.

This module supplies the missing multiplier, and it does two separate jobs
with it:

1. **Neutralise the top-down half.** A club's season-to-date runs are the sum
   of its talent and the parks it has played in, half of them its own. Divide
   the totals by the club's *park exposure* — the games-weighted mean factor
   over the games it has actually played — before regressing them toward the
   league, and what comes out is a park-neutral rate directly comparable with
   station C's bottom-up estimate.

2. **Put tonight's park back on.** Both clubs' runs scored and allowed are
   multiplied by the factor of the venue the game is at, before Pythagenpat.
   That is not cosmetic: Pythagenpat's exponent rises with the run
   environment, so the same run *ratio* converts to a more extreme win
   probability in a high-scoring park, which is the real effect a park has on
   who wins a game.

## How the factor is built

The classic home/road split, pooled over the completed seasons *before* the
one being predicted:

    factor(v) = (runs per game in games played at v)
              / (runs per game in the road games of the clubs whose home v is)

Both halves count *both* clubs' runs, so this is a run-environment multiplier
and not a statement about either team. The denominator is the same clubs away
from home, which is what controls for the quality of the teams that happen to
play there most: a good pitching staff depresses the numerator and the
denominator alike.

Then regressed toward 1 with `ballast` games of ballast:

    factor = (n · raw + ballast · 1) / (n + ballast)

`ballast = 0` is the raw split and an infinite ballast is no park term at all,
so the sweep that chooses it is a clean nesting — `--park-ballast inf`
reproduces the model without the term exactly.

Finally the factors are renormalised so the games-weighted league mean is
exactly 1. Every other term in the chain is centred on the league by
construction and this one is too: park can redistribute the league's runs
across venues, never move the league's total.

## Leakage

The factors for season Y are built from seasons Y−1 and Y−2 and nothing else,
so no game of the season being scored can inform them — a stronger guard than
the strictly-before-the-date rule the rest of the chain uses, and the one the
ticket asks for. The *exposure* half does read the current season, but only
games strictly before the date, the same cut every other frame gets.
`tests/test_sim/test_park.py` pins both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Completed seasons pooled into a factor. Two, matching the Marcel horizon the
# rate models use: one season of a park is about 81 games of a two-team run
# environment, which is a noisy thing, and parks do change (a fence moves, a
# humidor goes in), so a long pool is not free either.
PRIOR_SEASONS = 2

# Games of ballast on the raw home/road split. Chosen walk-forward on the 2025
# season only, over {0, 100, 200, 400, 800, inf} — see
# docs/market-benchmark-2026.md. `inf` is the model with no park term in it.
BALLAST_GAMES = 200.0

GAME_COLS = ["game_pk", "date", "venue_id", "home_id", "away_id", "runs"]


def completed_venue_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games with a final score, one row per game, with the venue.

    `schedule` is `mlb_stats_api.fetch_schedule`. A frame with no `venue_id`
    column falls back to the home club's id as the venue key, which is exact
    for every game except the handful played at a neutral site each year and
    keeps older callers (and the synthetic fixtures) working.
    """
    if schedule is None or len(schedule) == 0:
        return pd.DataFrame(columns=GAME_COLS)
    df = schedule
    if "status" in df.columns:
        df = df[df["status"] == "Final"]
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return pd.DataFrame(columns=GAME_COLS)
    venue = (pd.to_numeric(df["venue_id"], errors="coerce")
             if "venue_id" in df.columns else pd.Series(np.nan, index=df.index))
    home = pd.to_numeric(df["home_id"], errors="coerce").astype("int64")
    out = pd.DataFrame({
        "game_pk": pd.to_numeric(df["game_pk"], errors="coerce").astype("int64"),
        "date": df["date"].astype(str),
        "venue_id": venue.fillna(home).astype("int64"),
        "home_id": home,
        "away_id": pd.to_numeric(df["away_id"], errors="coerce").astype("int64"),
        "runs": (pd.to_numeric(df["home_score"], errors="coerce")
                 + pd.to_numeric(df["away_score"], errors="coerce")).astype(float),
    })
    return out.reset_index(drop=True)


def _home_clubs(games: pd.DataFrame) -> dict:
    """{venue_id: {the clubs that host there}} — read off the games themselves.

    No venue-to-club table is needed or wanted: a club that moved parks, a
    neutral site and a temporary home all come out right, because the only
    question asked is "who was the home team in the games played here".
    """
    out: dict[int, set] = {}
    for v, h in zip(games["venue_id"], games["home_id"]):
        out.setdefault(int(v), set()).add(int(h))
    return out


def run_factors(games: pd.DataFrame, ballast: float = BALLAST_GAMES) -> dict:
    """{venue_id: run multiplier}, regressed toward 1 and centred on the league.

    `games` is `completed_venue_games` over the *prior* seasons. A venue whose
    home clubs played no road games in the pool (a one-off neutral site) gets
    exactly 1.0, as does every venue when `ballast` is infinite.
    """
    if games is None or len(games) == 0:
        return {}
    if not np.isfinite(float(ballast)):
        return {int(v): 1.0 for v in pd.unique(games["venue_id"])}
    hosts = _home_clubs(games)
    raw, weight = {}, {}
    for venue, clubs in hosts.items():
        at = games[games["venue_id"] == venue]
        away = games[games["away_id"].isin(clubs) & (games["venue_id"] != venue)]
        if len(at) == 0 or len(away) == 0:
            raw[venue], weight[venue] = 1.0, 0.0
            continue
        raw[venue] = float(at["runs"].mean() / away["runs"].mean())
        weight[venue] = float(len(at))
    b = float(ballast)
    reg = {v: (weight[v] * raw[v] + b) / (weight[v] + b) for v in raw}
    # Centre on the league: the games-weighted mean factor is exactly 1, so the
    # term redistributes the league's runs across venues and cannot move the
    # league's own run environment (the anchor every other term is centred on).
    total = sum(weight[v] for v in reg)
    if total > 0:
        mean = sum(weight[v] * reg[v] for v in reg) / total
        if mean > 0:
            reg = {v: f / mean for v, f in reg.items()}
    return {int(v): float(f) for v, f in reg.items()}


def factor(venue_id, factors: dict) -> float:
    """The multiplier for one venue; 1.0 for anything the pool never saw."""
    if venue_id is None or not factors:
        return 1.0
    try:
        return float(factors.get(int(venue_id), 1.0))
    except (TypeError, ValueError):
        return 1.0


def team_exposure(games: pd.DataFrame, factors: dict, team_ids=None) -> pd.Series:
    """team_id → the mean park factor of the games it has played.

    `games` is `completed_venue_games` cut to the games *before* the date being
    predicted (the caller does the cut; this function is pure). Roughly
    `(own park + the mean of the road parks) / 2`, and it is the divisor that
    turns a club's season-to-date runs into a park-neutral rate.

    Normalised so the games-weighted mean exposure is exactly 1, which keeps
    the league's total runs unchanged by the neutralisation: what one club
    gains another gives back. A club with no games yet gets 1.0.
    """
    ids = None if team_ids is None else [int(t) for t in team_ids]
    if games is None or len(games) == 0:
        return pd.Series(1.0, index=pd.Index(ids or [], name="team_id"), dtype=float)
    f = games["venue_id"].map(lambda v: factor(v, factors)).astype(float)
    long = pd.concat([
        pd.DataFrame({"team_id": games["home_id"].astype("int64"), "f": f.to_numpy()}),
        pd.DataFrame({"team_id": games["away_id"].astype("int64"), "f": f.to_numpy()}),
    ], ignore_index=True)
    agg = long.groupby("team_id")["f"].agg(["sum", "size"])
    mean = float(agg["sum"].sum() / max(float(agg["size"].sum()), 1.0))
    exposure = agg["sum"] / agg["size"]
    if mean > 0:
        exposure = exposure / mean
    if ids is not None:
        exposure = exposure.reindex(ids).fillna(1.0)
    exposure.index.name = "team_id"
    return exposure.astype(float)


def neutral_run_rates(rs, ra, games, regress_games: float,
                      exposure: pd.Series | None = None) -> pd.DataFrame:
    """Park-neutral runs scored / allowed per game, regressed toward the league.

    The same regression `strength.regressed_run_rates` and the backtest's
    `team_rates` apply, with one change: the club's *totals* are divided by its
    park exposure before the league ballast is added, rather than the finished
    rate being divided afterwards. That matters — dividing the finished rate
    would divide the ballast's league-average share too, and pull a Coors club
    below the league instead of to it.

    `exposure = None` (or all ones) reproduces the un-neutralised rates
    exactly, which is what makes the park sweep a clean nesting.
    """
    rs = pd.to_numeric(rs, errors="coerce").astype(float)
    ra = pd.to_numeric(ra, errors="coerce").astype(float)
    g = pd.to_numeric(games, errors="coerce").astype(float)
    lg_rs = float(rs.sum()) / max(float(g.sum()), 1.0)
    lg_ra = float(ra.sum()) / max(float(g.sum()), 1.0)
    if exposure is not None:
        e = pd.to_numeric(exposure, errors="coerce").reindex(rs.index).fillna(1.0)
        e = e.where(e > 0, 1.0).astype(float)
        rs, ra = rs / e, ra / e
    b = float(regress_games)
    out = pd.DataFrame({
        "rs_pg": (rs + b * lg_rs) / (g + b),
        "ra_pg": (ra + b * lg_ra) / (g + b),
    })
    out.index.name = "team_id"
    return out


def apply_factor(rs9: float, ra9: float, venue_factor: float) -> tuple[float, float]:
    """Both of one club's rates scaled into tonight's park.

    Symmetric by construction — a park is a fact about the game, not about
    either club — so the *ratio* of the two is untouched and only the run
    environment moves. Pythagenpat then converts that ratio at a higher
    exponent, which is where a park earns its place in a win probability.
    """
    f = float(venue_factor)
    return float(rs9) * f, float(ra9) * f


def fetch_prior_games(season: int,
                      prior_seasons: int = PRIOR_SEASONS) -> pd.DataFrame:
    """The completed seasons before `season`, in `completed_venue_games` shape.

    The only function here that touches the network, and it is one schedule
    call per prior season (a few hundred kilobytes each, no per-player
    fetching). Walk-forward by construction: a game of `season` cannot reach
    these numbers because no game of `season` is in the pool. The pool is
    returned rather than the finished factors so a caller can sweep the ballast
    without paying for the fetch again — the same shape `ChainInputs` carries.
    """
    from src.data.mlb_stats_api import fetch_schedule

    frames = []
    for year in range(int(season) - int(prior_seasons), int(season)):
        sched = fetch_schedule(f"{year}-03-01", f"{year}-11-15")
        frames.append(completed_venue_games(sched))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=GAME_COLS))


def fetch_prior_factors(season: int, prior_seasons: int = PRIOR_SEASONS,
                        ballast: float = BALLAST_GAMES) -> dict:
    """`fetch_prior_games` + `run_factors`, for a caller that wants the table."""
    return run_factors(fetch_prior_games(season, prior_seasons), ballast=ballast)


__all__ = ["PRIOR_SEASONS", "BALLAST_GAMES", "completed_venue_games",
           "run_factors", "factor", "team_exposure", "neutral_run_rates",
           "apply_factor", "fetch_prior_games", "fetch_prior_factors"]
