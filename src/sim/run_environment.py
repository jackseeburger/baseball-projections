"""Station C — the team run environment, built bottom-up from the players.

Stations D and E both start from the same top-down number: the club's
season-to-date runs scored and allowed, regressed toward the league with a
60-game ballast (`strength.regressed_run_rates`). That number is a *record of
what happened*, and it does not know the roster changed. A club that sold at
the deadline, lost its best hitter to the IL in August, or is running out
September call-ups carries its June self around for weeks.

This module builds the other half of the estimate — what the players who will
actually play are worth:

    runs scored per game  = league RS/G
                          + PA/game · Σ_i  (hitter i's PA share)
                                         · (his runs above average per PA)

    runs allowed per game = (5.5/9) · rotation FIP RA/9   (top-5 by starts,
                                                           weighted by starts)
                          + (3.5/9) · bullpen FIP RA/9    (workload-weighted)

The hitter rates are station A's Marcel-style component rates
(`src.sim.lineups`), the PA shares are station B's trailing-window shares
(`src.projections.playing_time`), and the pitcher rates are the same
Marcel/FIP machinery station E prices starters and pens with
(`src.sim.starters`, `src.sim.bullpen`). Nothing new is estimated here; C is
the *assembly*.

Both halves are centred on the league, exactly as every station E term is: a
club whose hitters are all league average scores `lg_rs9`, and a club whose
rotation and pen are league average allows `lg_ra9`. So the bottom-up estimate
never shifts the league's run environment, only redistributes it.

## Why it is blended rather than swapped in

A pure bottom-up estimate is park-, defense-, baserunning- and
sequencing-neutral. FIP does not know about Coors or about the best
defensive alignment in the league; linear weights do not know a club runs the
bases badly. All of that *is* in the top-down number, measured. So swapping
C in wholesale would throw away real information to gain roster awareness —
which is the mistake `starters.py` documents having made and scored (an
absolute-level blend of FIP and team RA came in worse than no pitcher at all).

Hence:

    RS_C = w · RS_bottom_up + (1 − w) · RS_pythag60
    RA_C = w · RA_bottom_up + (1 − w) · RA_pythag60

with `w` chosen walk-forward on a season the model is not scored on. `w = 0`
reproduces the production model *exactly*, which makes the sweep a clean
nesting: any gain is the roster information, and only that. Station E's
starter / lineup / bullpen terms then apply on top as the same deltas they
already are, so `pythag_C_sp` is directly comparable to `pythag_60_sp`.

## Park

Not applied. `src/data/park_factors.py` computes team-season factors from a
league-wide hitter-season table with a documented approximation (no home/road
splits), keyed by team abbreviation and year rather than team id and date, and
it is not wired to anything the simulator reads. More to the point, park is
already inside the top-down half of the blend, where it was measured; a park
factor would only be needed if C were swapped in whole. Left out, and said so.

Everything here is a pure function over DataFrames, so it unit-tests without a
network. The as-of-date assembly lives in `scripts/backtest_game_odds.py`
alongside the station E terms, which is where the walk-forward guarantee is
enforced: every frame handed in is filtered to rows strictly before the date
being predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projections.playing_time import MAX_PA_SHARE, cap_shares, window_pa_by_team
from src.sim.starters import GAME_IP, MIN_RA9, STARTER_IP

# The trailing window that defines "who plays for this club, and how much".
# `None` is the season to date. Swept on 2025 only, against station B's own
# 30-day primary window and 15 and 60 days ({15, 30, 60, season} x {30, 45,
# season} days of starts, 12 cells spanning 0.00027 Brier); the season-to-date
# share won it, by 0.00007 over 30 days. Longer wins here and shorter wins for
# station B because the two answer different questions: B forecasts one
# hitter's next month, C wants the club's *average* batter, and averaging over
# more plate appearances is worth more than reacting a week sooner. See
# docs/market-benchmark-2026.md.
SHARE_WINDOW_DAYS = None

# A rotation is five men. Sixth starters, openers and bulk relievers exist, but
# their innings are the pen's in this decomposition — `bullpen.pen_window`
# already prices every non-start.
ROTATION_TOP_N = 5

# How many trailing days of starts define the rotation. `None` is the whole
# season to date. A window rather than the season for the same reason the pen
# uses one: a starter traded away in July is not in this rotation in August,
# and the season-to-date start count would keep him there until October. Thirty
# days is about six turns, so a five-man rotation is identified from ~30 starts.
# Swept on 2025 only ({30, 45, season}); 30 won at every share window.
ROTATION_WINDOW_DAYS = 30

# How much of the bottom-up estimate to use. Chosen walk-forward on 2025 only
# — see docs/market-benchmark-2026.md for the curve, which is an inverted U
# with an interior minimum: 0 / .25 / .5 / .75 / 1 score +0.00000 / -0.00047 /
# -0.00067 / -0.00059 / -0.00022 paired Brier against `pythag_60_sp`. Half and
# half, i.e. neither half of the estimate is worth much more than the other.
# `0.0` is the production model exactly.
BLEND_WEIGHT = 0.5

SHARE_COLS = ["team_id", "batter", "pa", "share"]
ROTATION_COLS = ["team", "pitcher", "starts"]


# ─── runs scored: hitters × playing time ───

def team_pa_shares(hitter_logs: pd.DataFrame, as_of,
                   window_days: int | None = SHARE_WINDOW_DAYS,
                   cap: float = MAX_PA_SHARE) -> pd.DataFrame:
    """Who bats for each club as of `as_of`, and for what share of its PA.

    `hitter_logs` is per-game hitting lines with `batter, team_id, date, pa`
    (`mlb_stats_api.fetch_hitter_game_logs`). Only games strictly before
    `as_of` are read — that is the leakage guard, and it is the same one
    station B applies: a game played *on* the date being predicted has not
    happened when the line is priced.

    Membership and usage come from the same window, deliberately. There is no
    roster call: a hitter belongs to the club he has been taking plate
    appearances for, which is walk-forward by construction, handles a trade the
    day it happens, and drops an injured player as his window empties. What it
    cannot do is know about a call-up before his first game — no leakage-free
    source can.

    Shares are capped at one lineup slot (`playing_time.cap_shares`) and
    re-normalised within the club, so they sum to 1 per team.

    Returns `team_id, batter, pa, share`.
    """
    win = window_pa_by_team(hitter_logs, as_of, window_days)
    if win.empty:
        return pd.DataFrame(columns=SHARE_COLS)
    win = win[win["pa"] > 0].reset_index(drop=True)
    if win.empty:
        return pd.DataFrame(columns=SHARE_COLS)
    totals = win.groupby("team_id")["pa"].transform("sum")
    raw = win["pa"] / totals.where(totals > 0, 1.0)
    win["share"] = cap_shares(raw, win["team_id"], cap=cap)
    return win.loc[:, SHARE_COLS]


def team_rs9(shares: pd.DataFrame, runs_lookup: dict, lg_rs9: float,
             pa_per_game: float) -> pd.Series:
    """Runs per game each club's *projected* batters are worth.

        lg_rs9 + PA/game · Σ_i share_i · runs_above_average_per_PA(i)

    `runs_lookup` is `lineups.batter_runs_lookup` — runs above an average plate
    appearance, from the Marcel-regressed component rates, centred so a
    league-average hitter is exactly 0. A hitter with no history is absent from
    the lookup and contributes 0, i.e. he is treated as league average, the
    same convention `lineups.lineup_r9` uses.

    Because the shares sum to 1 within a club and the weights are centred, a
    club of nine league-average hitters returns exactly `lg_rs9`: the term
    redistributes runs across clubs and cannot move the league.

    Returns a Series indexed by team_id.
    """
    if shares.empty:
        return pd.Series(dtype=float, name="rs9")
    raa = shares["batter"].map(lambda b: runs_lookup.get(int(b), 0.0)).astype(float)
    per_team = (shares["share"].astype(float) * raa).groupby(
        shares["team_id"].astype("int64")).sum()
    out = float(lg_rs9) + float(pa_per_game) * per_team
    out.index.name = "team_id"
    return out.astype(float).rename("rs9")


# ─── runs allowed: rotation + bullpen ───

def start_appearances(logs: pd.DataFrame) -> pd.DataFrame:
    """Starts from a pitching game-log frame: `pitcher, team, date`.

    The mirror of `bullpen.relief_appearances` — that one keeps `gs == 0`, this
    one keeps `gs == 1`, so between them every appearance is priced exactly
    once. `team` is the club he started for *that day*, which is what makes a
    July trade move a starter from one rotation to the other on the right date.
    """
    cols = ["pitcher", "team", "date"]
    if logs.empty:
        return pd.DataFrame(columns=cols)
    keep = logs[pd.to_numeric(logs.get("gs", 0), errors="coerce").fillna(0) == 1]
    out = pd.DataFrame({
        "pitcher": pd.to_numeric(keep["pitcher"], errors="coerce").astype("int64"),
        "team": pd.to_numeric(keep["team"], errors="coerce").astype("Int64"),
        "date": keep["date"].astype(str),
    })
    return out.dropna(subset=["team"]).reset_index(drop=True)


def rotation_window(starts: pd.DataFrame, as_of,
                    days: int | None = ROTATION_WINDOW_DAYS,
                    top_n: int = ROTATION_TOP_N) -> pd.DataFrame:
    """Each club's rotation as of `as_of`: `team, pitcher, starts`.

    The `top_n` pitchers by starts made for that club in the `days` calendar
    days strictly before `as_of` (`days=None` for the season to date). Strictly
    before, always: a start made on the date being predicted is the game being
    predicted.

    Ties are broken by the most recent start, so between two men with the same
    count the one still in the rotation wins.
    """
    if starts is None or starts.empty:
        return pd.DataFrame(columns=ROTATION_COLS)
    past = starts[starts["date"].astype(str) < str(as_of)]
    if days is not None:
        lo = str((pd.Timestamp(str(as_of)) - pd.Timedelta(days=int(days))).date())
        past = past[past["date"].astype(str) >= lo]
    if past.empty:
        return pd.DataFrame(columns=ROTATION_COLS)
    agg = past.groupby(["team", "pitcher"], as_index=False).agg(
        starts=("date", "size"), last=("date", "max"))
    agg = agg.sort_values(["team", "starts", "last"], ascending=[True, False, False])
    out = agg.groupby("team", as_index=False).head(top_n)
    return out.loc[:, ROTATION_COLS].reset_index(drop=True)


def rotation_ra9(rotation: pd.DataFrame, ra9_lookup: dict,
                 lg_ra9: float) -> dict:
    """{team_id: starts-weighted FIP runs allowed per 9} for the rotation.

    `ra9_lookup` is `starters.starter_ra9_lookup` — the same Marcel-weighted,
    league-regressed FIP the station E starter term uses, so the rotation slot
    and the announced starter are priced on one scale. A pitcher with no
    history is scored at `lg_ra9`.

    Weighting by starts rather than equally is what makes the number a
    *rotation*: an ace who has taken 9 of a club's last 30 turns counts for
    more than the spot starter who took one.
    """
    if rotation is None or len(rotation) == 0:
        return {}
    out = {}
    for team, grp in rotation.groupby("team"):
        w = grp["starts"].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        r = np.array([float(ra9_lookup.get(int(p), lg_ra9)) for p in grp["pitcher"]],
                     dtype=float)
        out[int(team)] = float(np.dot(w, r) / w.sum())
    return out


def team_ra9(rot_ra9: dict, pen_ra9: dict, lg_ra9: float,
             team_ids=None, starter_ip: float = STARTER_IP,
             game_ip: float = GAME_IP) -> pd.Series:
    """Runs allowed per game the staff is worth: rotation + pen, by innings.

        (5.5/9) · rotation RA/9 + (3.5/9) · bullpen RA/9

    the same 5.5/3.5 split `starters.py` and `bullpen.py` partition the game
    with, so a club whose rotation and pen are both league average returns
    exactly `lg_ra9`. Either half missing (a club with no starts or no relief
    appearances in the window yet) falls back to `lg_ra9` for that half only.

    Returns a Series indexed by team_id.
    """
    ids = sorted({int(t) for t in (team_ids if team_ids is not None
                                   else set(rot_ra9) | set(pen_ra9))})
    w = float(starter_ip) / float(game_ip)
    vals = {t: w * float(rot_ra9.get(t, lg_ra9))
            + (1.0 - w) * float(pen_ra9.get(t, lg_ra9)) for t in ids}
    out = pd.Series(vals, dtype=float, name="ra9")
    out.index.name = "team_id"
    return out.clip(lower=MIN_RA9)


# ─── the blend ───

def blend_run_env(bottom_up: pd.DataFrame, top_down: pd.DataFrame,
                  weight: float = BLEND_WEIGHT) -> pd.DataFrame:
    """`weight · bottom_up + (1 − weight) · top_down`, on rs_pg / ra_pg.

    Both frames are indexed by team_id with `rs_pg` and `ra_pg` columns.
    `top_down` is station D's regressed run rates
    (`strength.regressed_run_rates`, or the backtest's walk-forward
    equivalent) and is the fallback for anything the bottom-up half could not
    estimate: a team missing from `bottom_up`, or an NaN in it, keeps its
    top-down rate for that column.

    `weight = 0` returns `top_down` unchanged — the production model, exactly —
    which is what makes the weight sweep a nesting rather than a family of
    unrelated models.
    """
    w = float(weight)
    out = top_down.loc[:, ["rs_pg", "ra_pg"]].astype(float).copy()
    if w == 0.0 or bottom_up is None or len(bottom_up) == 0:
        return out
    bu = bottom_up.reindex(out.index)
    for col in ("rs_pg", "ra_pg"):
        b = pd.to_numeric(bu[col], errors="coerce") if col in bu.columns else np.nan
        out[col] = np.where(pd.isna(b), out[col], w * b + (1.0 - w) * out[col])
    return out


def league_constant_rates(team_ids, lg_rs9: float, lg_ra9: float) -> pd.DataFrame:
    """The shrinkage control: every club league average, top and bottom.

    Blended in at `weight` this is exactly "the production rates shrunk
    `weight` of the way to the league", with no player information in it at
    all. It exists because the bottom-up estimate is *less spread out* than
    season-to-date run differential — FIP and linear weights are heavily
    regressed component rates — so any blend of the two compresses the league,
    and `pythag_60` is known to be overconfident in its tails
    (docs/market-benchmark-2026.md). Without this control a gain from plain
    shrinkage would be indistinguishable from a gain from knowing the roster.

    `scripts/backtest_game_odds.py --c-control league` scores it.
    """
    idx = pd.Index(sorted({int(t) for t in team_ids}), name="team_id")
    return pd.DataFrame({"rs_pg": float(lg_rs9), "ra_pg": float(lg_ra9)}, index=idx)


def bottom_up_rates(rs9: pd.Series, ra9: pd.Series,
                    team_ids=None) -> pd.DataFrame:
    """Assemble the two halves into a `rs_pg` / `ra_pg` frame for `blend_run_env`."""
    idx = pd.Index(sorted({int(t) for t in (team_ids if team_ids is not None
                                            else set(rs9.index) | set(ra9.index))}),
                   name="team_id")
    return pd.DataFrame({
        "rs_pg": pd.to_numeric(rs9, errors="coerce").reindex(idx),
        "ra_pg": pd.to_numeric(ra9, errors="coerce").reindex(idx),
    }, index=idx)
