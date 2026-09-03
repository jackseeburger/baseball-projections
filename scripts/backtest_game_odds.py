"""Day-to-day backtest of per-game win probabilities (walk-forward).

For every completed game of the season, rebuild each team's strength from
ONLY the games played before that date, predict P(home wins), and score
against the actual result. This is the honest test of the simulator's
per-game engine — the playoff-odds table mostly reflects banked standings,
but per-game probabilities are where a strength model actually earns edge.

Metrics: Brier score and log loss (lower is better). Baselines:
    home_constant   — P(home) = HFA prior, ignores teams entirely
    win_pct_log5    — raw season win% into log5 + HFA (no regression)
    pythag_60       — the production model (regressed Pythagenpat, 60 games)
    pythag_{k}      — ballast sweep to see what the data prefers
    pythag_60_sp    — pythag_60 with each side's runs allowed re-weighted
                      toward its announced starting pitcher (src/sim/starters)
    pythag_60_sp_lu — pythag_60_sp with each side's runs *scored* re-weighted
                      toward its posted lineup (src/sim/lineups)
    pythag_60_sp_lu_bp
                    — ...and the 3.5 relief innings re-weighted toward the
                      bullpen that is actually available (src/sim/bullpen)
    pythag_60_sp_lu_bpa
                    — the same, with the binary "worked three days running"
                      exclusion replaced by a pitch-count availability weight
                      per reliever (src/sim/reliever_usage)
    pythag_C        — station C: the team's runs scored / allowed rebuilt
                      bottom-up from the hitters who are actually playing and
                      the rotation + pen that are actually pitching, blended
                      with the top-down regressed rates (src/sim/run_environment)
    pythag_C_sp     — pythag_C with the same starting-pitcher delta pythag_60_sp
                      applies, so the two are directly comparable
    pythag_C_sp_bpa — pythag_C_sp with the pitch-count availability delta on the
                      3.5 relief innings, the one term at a time on the best
                      model there is
    pythag_C_sp_bpa_ip
                    — the same, splitting the game between starter and pen at
                      *this* starter's expected innings instead of a flat 5.5

Usage:
    python scripts/backtest_game_odds.py --season 2026 --min-games 20
    python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
        --market data/parquet/market_closes_2026.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.mlb_stats_api import (
    build_seasons_table, fetch_batter_game_logs, fetch_hitter_game_logs,
    fetch_lineups, fetch_pitcher_game_logs, fetch_probables, fetch_schedule,
    fetch_season_pitching,
)
from src.sim import bullpen as bp_model
from src.sim import lineups as lu_model
from src.sim import reliever_usage as ru_model
from src.sim import run_environment as rn_model
from src.sim import starters as sp_model
from src.sim.season import from_schedule
from src.sim.strength import HFA_PRIOR, home_win_prob, pythagenpat
from src.sim.teams import fetch_teams

SP_MODEL = "pythag_60_sp"
LU_MODEL = "pythag_60_sp_lu"
BP_MODEL = "pythag_60_sp_lu_bp"
BPA_MODEL = "pythag_60_sp_lu_bpa"
C_MODEL = "pythag_C"
C_SP_MODEL = "pythag_C_sp"
C_SP_BPA_MODEL = "pythag_C_sp_bpa"
C_SP_BPA_IP_MODEL = "pythag_C_sp_bpa_ip"
SP_BALLAST_GAMES = 60.0  # the production team-strength ballast this model extends
MIN_R9 = 0.5              # Pythagenpat needs positive run rates on both sides
# How much of the lineup's distance from its baseline to apply, and what to
# measure it against. Both chosen walk-forward on the 2025 season and never on
# a 2026 score; the 2025 curve is flat for any weight from 0.25 to 0.75
# (docs/market-benchmark-2026.md).
LU_WEIGHT = 0.5
LU_BASELINE = "team"
# Bullpen availability: what counts as a pen and what counts as used up. Both
# chosen walk-forward on 2025 only (docs/market-benchmark-2026.md).
BP_BASELINE = bp_model.BASELINE
# The pitch-count availability weight that replaces that binary rule: two pitch
# thresholds and what the available pen is measured against, all three chosen
# walk-forward on 2025 only (docs/market-benchmark-2026.md).
BPA_HARD_1D = ru_model.HARD_1D_PITCHES
BPA_HARD_2D = ru_model.HARD_2D_PITCHES
BPA_TAPER = ru_model.TAPER_PITCHES
BPA_BASELINE = ru_model.BASELINE
# How hard to regress a starter's own innings per start toward the flat 5.5
# when his expected innings, not 5.5, set the starter/bullpen split. Chosen
# walk-forward on 2025 only (docs/market-benchmark-2026.md).
SP_IP_BALLAST = sp_model.IP_BALLAST_STARTS
# Station C: how much of the bottom-up run environment to use, what trailing
# window defines a club's hitters and their plate-appearance shares, and how
# many days of starts define a rotation. All three chosen walk-forward on 2025
# only (docs/market-benchmark-2026.md); `--c-weight 0` reproduces pythag_60 and
# pythag_60_sp exactly.
C_WEIGHT = rn_model.BLEND_WEIGHT
C_SHARE_WINDOW = rn_model.SHARE_WINDOW_DAYS or 0   # 0 = the season to date
C_ROTATION_DAYS = rn_model.ROTATION_WINDOW_DAYS


def team_totals(games: pd.DataFrame, team_ids) -> pd.DataFrame:
    """Runs scored/allowed, wins/losses per team from a completed-games frame."""
    home = games.groupby("home_id").agg(rs=("home_score", "sum"), ra=("away_score", "sum"),
                                        w=("home_win", "sum"), g=("home_win", "size"))
    away = games.groupby("away_id").agg(rs=("away_score", "sum"), ra=("home_score", "sum"),
                                        w=("home_win", lambda x: (~x).sum()), g=("home_win", "size"))
    tot = home.add(away, fill_value=0).reindex(team_ids).fillna(0)
    return tot


def team_rates(tot: pd.DataFrame, regress_games: float) -> pd.DataFrame:
    """Runs scored/allowed per game, regressed toward league average."""
    lg_rs = tot["rs"].sum() / max(tot["g"].sum(), 1)
    lg_ra = tot["ra"].sum() / max(tot["g"].sum(), 1)
    return pd.DataFrame({
        "rs_pg": (tot["rs"] + regress_games * lg_rs) / (tot["g"] + regress_games),
        "ra_pg": (tot["ra"] + regress_games * lg_ra) / (tot["g"] + regress_games),
    })


def strengths(tot: pd.DataFrame, regress_games: float) -> pd.Series:
    rates = team_rates(tot, regress_games)
    return pd.Series({t: pythagenpat(r["rs_pg"], r["ra_pg"], 1.0)
                      for t, r in rates.iterrows()})


def walk_forward(completed: pd.DataFrame, team_ids, min_games: int,
                 ballasts: list[float], sp_ctx: dict | None = None,
                 lu_ctx: dict | None = None,
                 bp_ctx: dict | None = None,
                 c_ctx: dict | None = None) -> pd.DataFrame:
    """One row per scored game with each model's P(home).

    Every quantity used on a date is rebuilt from games, pitcher appearances
    batter games and relief appearances strictly before that date.
    """
    completed = completed.sort_values("date").reset_index(drop=True)
    rows = []
    lu_history: dict[int, list[float]] = {}
    for date, day in completed.groupby("date", sort=True):
        before = completed[completed["date"] < date]
        tot = team_totals(before, team_ids) if len(before) else None
        if tot is None or tot["g"].min() < min_games:
            continue
        hfa_obs = (HFA_PRIOR * 2000 + before["home_win"].sum()) / (2000 + len(before))
        wp = (tot["w"] / tot["g"]).clip(0.2, 0.8)
        s_by_k = {k: strengths(tot, k) for k in ballasts}
        sp_day = starter_day_context(tot, date, sp_ctx) if sp_ctx else None
        lu_day = (lineup_day_context(tot, date, day, lu_ctx, lu_history)
                  if lu_ctx and sp_day is not None else None)
        bp_day = (bullpen_day_context(tot, date, bp_ctx, sp_day)
                  if bp_ctx and lu_day is not None else None)
        c_day = (run_env_day_context(tot, date, c_ctx, sp_day, lu_day, bp_day)
                 if c_ctx and bp_day is not None else None)
        for g in day.itertuples(index=False):
            row = {"date": date, "game_pk": int(g.game_pk),
                   "home_id": g.home_id, "away_id": g.away_id,
                   "home_win": bool(g.home_win),
                   "home_constant": hfa_obs,
                   "win_pct_log5": float(home_win_prob(wp[g.home_id], wp[g.away_id], hfa_obs))}
            for k, s in s_by_k.items():
                row[f"pythag_{int(k)}"] = float(home_win_prob(s[g.home_id], s[g.away_id], hfa_obs))
            if sp_day is not None:
                p_sp, flags = starter_game_prob(g, sp_day, hfa_obs)
                row[SP_MODEL] = p_sp if p_sp is not None else row[f"pythag_{int(SP_BALLAST_GAMES)}"]
                row.update(flags)
            if lu_day is not None:
                p_lu, flags = lineup_game_prob(g, sp_day, lu_day, hfa_obs, lu_ctx)
                row[LU_MODEL] = p_lu if p_lu is not None else row[SP_MODEL]
                row.update(flags)
            if bp_day is not None:
                p_bp, flags = lineup_game_prob(g, sp_day, lu_day, hfa_obs, lu_ctx,
                                               bp_day, bp_ctx)
                row[BP_MODEL] = p_bp if p_bp is not None else row[LU_MODEL]
                row.update(flags)
                p_bpa, flags = lineup_game_prob(g, sp_day, lu_day, hfa_obs, lu_ctx,
                                                bp_day, bp_ctx, pen_kind="available")
                row[BPA_MODEL] = p_bpa if p_bpa is not None else row[LU_MODEL]
                row.update(flags)
            if c_day is not None:
                row.update(run_env_game_probs(g, c_day, sp_day, hfa_obs,
                                              bp_day, bp_ctx))
            rows.append(row)
        if lu_day is not None:
            update_lineup_history(day, lu_ctx, lu_history)
    return pd.DataFrame(rows)


# ─── station E starting-pitcher term ───

def starter_day_context(tot: pd.DataFrame, date: str, sp_ctx: dict) -> dict:
    """Everything the starter model needs for one slate, built from the past only.

    `starters.rate_table` does the pitcher half (rates from appearances
    strictly before `date`); the team half is the same regressed run rates
    `pythag_60` uses. The live nightly job calls the same two functions for a
    single date — see `scripts/run_playoff_odds.py`.
    """
    # The current run environment enters through lg_ra9, which anchors the FIP
    # constant to season-to-date league runs per game.
    lg_ra9 = float(tot["ra"].sum() / max(tot["g"].sum(), 1))
    return {
        "sp_ra9": sp_model.rate_table(sp_ctx, date, lg_ra9,
                                      ballast=sp_ctx["ballast"]),
        "lg_ra9": lg_ra9,
        "team": team_rates(tot, SP_BALLAST_GAMES),
        "probables": sp_ctx["probables"],
        "starter_ip": sp_ctx["starter_ip"],
    }


def starter_game_prob(g, day: dict, hfa: float):
    """P(home) with each side's runs allowed blended toward its starter.

    Returns (probability or None when a starter is unknown, diagnostic flags).
    """
    sp_ids = day["probables"].get(int(g.game_pk))
    flags = {"sp_fallback": sp_ids is None, "sp_no_history": 0}
    if sp_ids is None:
        return None, flags
    p_home, no_history = sp_model.game_home_prob(
        day["team"], g.home_id, g.away_id, sp_ids, day["sp_ra9"],
        day["lg_ra9"], hfa, starter_ip=day["starter_ip"])
    flags["sp_no_history"] = no_history
    return p_home, flags


def build_sp_context(season: int, scored: pd.DataFrame, ballast: float,
                     starter_ip: float, prior_seasons: int = 2) -> dict:
    """Fetch probables + pitcher counts once for the whole backtest.

    `prior_seasons` completed seasons plus the current one are Marcel-weighted,
    matching the 5/4/3 weights in `src.sim.starters`.
    """
    probables = fetch_probables(f"{season}-03-01", f"{season}-11-15")
    probables = probables.dropna(subset=["home_sp_id", "away_sp_id"])
    probables = probables[probables["game_pk"].isin(scored["game_pk"])]
    pmap = {int(r.game_pk): (int(r.home_sp_id), int(r.away_sp_id))
            for r in probables.itertuples(index=False)}

    inputs = sp_model.rate_inputs(
        season, {p for ids in pmap.values() for p in ids},
        prior_seasons=prior_seasons)
    return {**inputs, "probables": pmap,
            "ballast": ballast, "starter_ip": starter_ip}


# ─── station E posted-lineup term ───

def lineup_day_context(tot: pd.DataFrame, date: str, day: pd.DataFrame,
                       lu_ctx: dict, history: dict) -> dict:
    """Everything the lineup model needs for one slate, built from the past only.

    `raa` holds each posted lineup's runs *above league average* per nine
    innings, which is scale-free, so a club's baseline can be accumulated
    across dates whose league run environment differs slightly.
    """
    current = lu_model.games_before(lu_ctx["game_logs"], date)
    counts = pd.concat([lu_ctx["prior_counts"], current], ignore_index=True)
    # League rates come from the completed prior seasons only: the current-season
    # logs we hold cover posted-lineup batters, not the whole population, so
    # pooling them would bias the regression target. The current run environment
    # enters through lg_rs9, which is season-to-date league runs per game.
    lg = lu_ctx["league"]
    rates = lu_model.marcel_rates(counts, lu_ctx["season"], lg,
                                  ballast=lu_ctx["ballast"])
    lookup = lu_model.batter_runs_lookup(rates, lg)
    def raa9(ids):
        return lu_model.lineup_r9(ids, lookup, 0.0, lu_ctx["pa_per_game"])

    raa, missing = {}, {}
    for g in day.itertuples(index=False):
        ids = lu_ctx["lineups"].get(int(g.game_pk))
        if ids is None:
            continue
        raa[int(g.game_pk)] = {side: raa9(ids[side]) for side in ("home", "away")}
        missing[int(g.game_pk)] = sum(int(b) not in lookup
                                      for side in ("home", "away") for b in ids[side])
    # The club's own baseline is its recent cards re-scored with *today's*
    # rates, not the numbers those games were given at the time — see
    # `lu_model.team_lineup_baseline` for why that drift matters.
    window = lu_ctx["baseline_window"]
    baseline = {int(t): lu_model.team_lineup_baseline(
        [raa9(ids) for ids in history.get(int(t), [])[-window:]], 0.0,
        lu_ctx["baseline_ballast"]) for t in tot.index}
    return {"raa": raa, "no_history": missing, "baseline": baseline,
            # Station C prices *every* hitter on the club, not just the nine
            # posted, so it reuses this date's rates rather than rebuilding them.
            "runs_lookup": lookup,
            "lg_rs9": float(tot["rs"].sum() / max(tot["g"].sum(), 1))}


def pen_delta_ra9(ra9: float, team_id, pid, bp_day: dict, bp_ctx: dict,
                  kind: str, relief_ip: float | None = None) -> tuple[float, float]:
    """Apply the 3.5-inning bullpen delta to one side's runs-allowed rate.

    `kind` picks which reading of "the pen that is available" is used:

      * `"whole"` — `src/sim/bullpen`: the whole pen with the men who worked
        three calendar days running dropped outright.
      * `"available"` — `src/sim/reliever_usage`: every man in the pen weighted
        by his trailing workload *times* a pitch-count availability weight, so
        a heavy outing last night costs a fraction of an arm rather than all or
        nothing.

    Both are the same delta form and the same 3.5/9 share, so the pair isolates
    the reading and nothing else. Returns (adjusted rate, how far the pen used
    sits from the club's whole pen — the diagnostic that says how much the
    availability reading actually moved).
    """
    lg_ra9 = bp_day["lg_ra9"]
    avail, full = bp_day["pen"].get(int(team_id), (lg_ra9, lg_ra9))
    if kind == "available":
        now = bp_day["available"](int(team_id), int(pid))
        base = bp_day["lg_bpa_ra9"] if bp_ctx["bpa_baseline"] == "league" else full
    else:
        now = avail
        base = bp_day["lg_pen_ra9"] if bp_ctx["baseline"] == "league" else full
    ip = bp_ctx["relief_ip"] if relief_ip is None else relief_ip
    out = bp_model.blend_bullpen_team(now, ra9, base, relief_ip=ip)
    return float(out), abs(float(now) - float(full))


def lineup_game_prob(g, sp_day: dict, lu_day: dict, hfa: float, lu_ctx: dict,
                     bp_day: dict | None = None, bp_ctx: dict | None = None,
                     pen_kind: str = "whole"):
    """P(home) with both sides' starter, posted lineup and (optionally) pen.

    The three terms stack additively on the club's own regressed run rates —
    the lineup moves runs scored, the starter moves the 5.5 innings he covers
    and the bullpen the other 3.5 — so each is a delta and none of them can
    re-regress a club toward the league on its own.

    Returns (probability or None when the starter or lineup is unknown,
    diagnostic flags).
    """
    pk = int(g.game_pk)
    raa = lu_day["raa"].get(pk)
    sp_ids = sp_day["probables"].get(pk)
    fallback = raa is None or sp_ids is None
    tag = "bp" if pen_kind == "whole" else "bpa"
    flags = ({f"{tag}_fallback": fallback, f"{tag}_short": 0, f"{tag}_shift": 0.0}
             if bp_day is not None else
             {"lu_fallback": fallback,
              "lu_no_history": lu_day["no_history"].get(pk, 0) if raa else 0})
    if fallback:
        return None, flags
    team, lg_ra9 = sp_day["team"], sp_day["lg_ra9"]
    league_baseline = lu_ctx["baseline"] == "league"
    strength = {}
    for side, team_id, pid in (("home", g.home_id, sp_ids[0]),
                               ("away", g.away_id, sp_ids[1])):
        ra9 = sp_model.blend_starter_team(sp_day["sp_ra9"].get(pid, lg_ra9),
                                          team.loc[team_id, "ra_pg"], lg_ra9,
                                          starter_ip=sp_day["starter_ip"])
        if bp_day is not None:
            ra9, shift = pen_delta_ra9(ra9, team_id, pid, bp_day, bp_ctx, pen_kind)
            flags[f"{tag}_short"] += int(shift > 1e-9)
            flags[f"{tag}_shift"] += shift
        base = 0.0 if league_baseline else lu_day["baseline"].get(int(team_id), 0.0)
        rs9 = lu_model.blend_lineup_team(raa[side], team.loc[team_id, "rs_pg"],
                                         base, weight=lu_ctx["weight"])
        strength[side] = pythagenpat(max(float(rs9), MIN_R9),
                                     max(float(ra9), MIN_R9), 1.0)
    return float(home_win_prob(strength["home"], strength["away"], hfa)), flags


def update_lineup_history(day: pd.DataFrame, lu_ctx: dict, history: dict) -> None:
    """Bank each club's posted cards after its games are scored.

    The nine ids are stored, not the runs number, so the baseline can be
    re-scored on each later date's rates. Only dates the backtest actually
    scores contribute, so a club's baseline starts empty on the first scored
    date and is regressed toward league average until it fills in —
    walk-forward, never using a lineup from a game that has not been played.
    """
    for g in day.itertuples(index=False):
        ids = lu_ctx["lineups"].get(int(g.game_pk))
        if ids is None:
            continue
        history.setdefault(int(g.home_id), []).append(ids["home"])
        history.setdefault(int(g.away_id), []).append(ids["away"])


def build_lu_context(season: int, scored: pd.DataFrame, ballast, weight: float,
                     baseline: str, baseline_ballast: float, pa_per_game: float,
                     baseline_window: int = lu_model.BASELINE_WINDOW_GAMES,
                     prior_seasons: int = 2) -> dict:
    """Fetch posted lineups + batter counts once for the whole backtest."""
    lineups = fetch_lineups(scored["game_pk"])
    by_game: dict[int, dict[str, list[int]]] = {}
    for (pk, side), grp in lineups.sort_values("slot").groupby(["game_pk", "side"]):
        by_game.setdefault(int(pk), {})[side] = [int(b) for b in grp["batter"]]
    by_game = {pk: sides for pk, sides in by_game.items()
               if len(sides.get("home", [])) == 9 and len(sides.get("away", [])) == 9}

    prior = build_seasons_table(season - prior_seasons, season - 1)
    prior_counts = lu_model.normalize_counts(prior)

    batters = {b for sides in by_game.values() for ids in sides.values() for b in ids}
    logs = fetch_batter_game_logs(batters, season)
    logs = logs[logs["game_type"] == "R"]
    game_logs = lu_model.normalize_counts(logs)
    game_logs["date"] = logs["date"].to_numpy()
    return {"lineups": by_game, "prior_counts": prior_counts, "game_logs": game_logs,
            "league": lu_model.league_rates(prior_counts), "season": season,
            "ballast": ballast, "weight": weight, "baseline": baseline,
            "baseline_ballast": baseline_ballast, "pa_per_game": pa_per_game,
            "baseline_window": baseline_window}


# ─── station E bullpen-availability term ───

def bullpen_day_context(tot: pd.DataFrame, date: str, bp_ctx: dict,
                        sp_day: dict) -> dict:
    """Each club's available and whole bullpen for one slate, from the past only.

    The rates are the same Marcel/FIP machinery the rotation is priced with
    (`src.sim.starters`), rebuilt here over *every* pitcher rather than only
    the announced starters, because the pen is mostly men who never start.
    """
    lg = bp_ctx["league"]
    lg_ra9 = sp_day["lg_ra9"]
    counts = pd.concat([bp_ctx["prior_counts"],
                        sp_model.appearances_before(bp_ctx["game_logs"], date)],
                       ignore_index=True)
    rates = sp_model.marcel_rates(counts, bp_ctx["season"], lg,
                                  ballast=bp_ctx["ballast"])
    ra9 = sp_model.starter_ra9_lookup(rates, lg, lg_ra9)

    relief = bp_ctx["relief"]
    pens = bp_model.pen_window(relief, date, days=bp_ctx["roster_days"])
    out = bp_model.unavailable(relief, date, days=bp_ctx["rest_days"],
                               min_days=bp_ctx["rest_min_days"])
    # The pitch-count reading of the same question. `usage` is *every*
    # appearance, starts included — an opener's arm is as tired as a
    # reliever's — while pen membership and the workload weights stay with the
    # relief appearances above.
    weights = ru_model.availability(bp_ctx["usage"], date,
                                    hard_1d=bp_ctx["hard_1d"],
                                    hard_2d=bp_ctx["hard_2d"],
                                    taper=bp_ctx["taper"])
    pen, frames = {}, {}
    for team_id, grp in pens.groupby("team"):
        frames[int(team_id)] = grp
        pen[int(team_id)] = (bp_model.pen_ra9(grp, ra9, lg_ra9, exclude=out),
                             bp_model.pen_ra9(grp, ra9, lg_ra9))

    def available(team_id: int, starter_id: int) -> float:
        """The club's availability-weighted pen rate, minus tonight's starter.

        Per game rather than per day because the exclusion is the announced
        starter, who is on the roster but is not in the pen behind himself —
        he only ever appears in `frames` at all if he relieved inside the
        window, which is the swingman case.
        """
        grp = frames.get(int(team_id))
        if grp is None:
            return lg_ra9
        return ru_model.available_pen_ra9(grp, ra9, lg_ra9, weights,
                                          exclude=(int(starter_id),))

    return {"pen": pen, "lg_pen_ra9": bp_model.league_pen_ra9(pens, ra9, lg_ra9),
            "available": available,
            # How deep each starter has been going, regressed toward the flat
            # 5.5 — the workload split the `_ip` model uses in place of it.
            "sp_ip": sp_model.expected_starter_ip(bp_ctx["start_ip"], date,
                                                  ballast=bp_ctx["ip_ballast"]),
            # The pieces `available` closes over, so a caller sweeping the two
            # pitch thresholds can rebuild it without paying for the rates
            # again (scripts/sweep_reliever_usage.py).
            "frames": frames, "pens": pens, "weights": weights,
            "lg_bpa_ra9": ru_model.league_available_pen_ra9(pens, ra9, lg_ra9,
                                                            weights),
            # Station C prices the rotation off the same table.
            "ra9": ra9, "lg_ra9": lg_ra9, "n_out": len(out),
            "n_limited": sum(1 for w in weights.values() if w < 1.0)}


def build_bp_context(season: int, ballast, baseline: str, roster_days: int,
                     rest_days: int, rest_min_days: int, relief_ip: float,
                     league: dict, prior_counts: pd.DataFrame,
                     bpa_baseline: str = BPA_BASELINE,
                     hard_1d: float = BPA_HARD_1D,
                     hard_2d: float = BPA_HARD_2D,
                     taper: float = BPA_TAPER,
                     ip_ballast: float = SP_IP_BALLAST) -> dict:
    """Fetch every pitcher's appearances once for the whole backtest.

    `sp_ctx` already holds the prior-season pitching totals and league rates —
    they are the same pool — so only the current season's game logs are new,
    and they have to cover the whole pitcher population rather than just the
    announced starters.
    """
    ids = fetch_season_pitching(season)["pitcher"]
    logs = fetch_pitcher_game_logs(ids, season)
    logs = logs[logs["game_type"] == "R"]
    game_logs = sp_model.normalize_counts(logs)
    game_logs["date"] = logs["date"].to_numpy()
    return {"game_logs": game_logs, "prior_counts": prior_counts,
            "relief": bp_model.relief_appearances(logs),
            # Every appearance with its pitch count, starts included: the
            # workload half of the availability weight (src/sim/reliever_usage).
            "usage": ru_model.appearance_pitches(logs),
            # The other half of the same appearances: station C's rotation,
            # and how many innings each of those starts actually lasted.
            "starts": rn_model.start_appearances(logs),
            "start_ip": sp_model.start_innings(logs), "league": league,
            "season": season, "ballast": ballast, "baseline": baseline,
            "roster_days": roster_days, "rest_days": rest_days,
            "rest_min_days": rest_min_days, "relief_ip": relief_ip,
            "bpa_baseline": bpa_baseline, "hard_1d": hard_1d,
            "hard_2d": hard_2d, "taper": taper, "ip_ballast": ip_ballast}


# ─── station C: the bottom-up team run environment ───

def run_env_day_context(tot: pd.DataFrame, date: str, c_ctx: dict,
                        sp_day: dict, lu_day: dict, bp_day: dict) -> dict:
    """Each club's blended runs scored / allowed for one slate, from the past only.

    Every input is a station already scored on its own:

      * hitter rates    `lu_day["runs_lookup"]` — station A/E's Marcel component
                        rates, built from games strictly before `date`
      * playing time    trailing-window plate-appearance shares by club, station
                        B's window and its one-lineup-slot cap, from games
                        strictly before `date`
      * rotation        the top-5 by starts in the trailing window, priced with
                        the same FIP table the announced starter is
      * bullpen         `bp_day["pen"]`'s *whole* pen (the availability news is
                        station E's term, not C's), workload-weighted

    The result is blended with the top-down regressed rates at `weight`, so
    `weight = 0` hands back exactly what `pythag_60` uses.
    """
    top_down = team_rates(tot, SP_BALLAST_GAMES)
    lg_ra9, lg_rs9 = sp_day["lg_ra9"], lu_day["lg_rs9"]

    shares = rn_model.team_pa_shares(c_ctx["hitter_logs"], date,
                                     window_days=c_ctx["share_window"])
    rs9 = rn_model.team_rs9(shares, lu_day["runs_lookup"], lg_rs9,
                            c_ctx["pa_per_game"])

    rotation = rn_model.rotation_window(c_ctx["starts"], date,
                                        days=c_ctx["rotation_days"],
                                        top_n=c_ctx["rotation_top_n"])
    rot_ra9 = rn_model.rotation_ra9(rotation, bp_day["ra9"], lg_ra9)
    pen_ra9 = {t: full for t, (_avail, full) in bp_day["pen"].items()}
    ra9 = rn_model.team_ra9(rot_ra9, pen_ra9, lg_ra9, team_ids=top_down.index,
                            starter_ip=sp_day["starter_ip"])

    if c_ctx.get("control") == "league":
        # The shrinkage control: same blend, no player information in it.
        bottom_up = rn_model.league_constant_rates(top_down.index, lg_rs9, lg_ra9)
    else:
        bottom_up = rn_model.bottom_up_rates(rs9, ra9, team_ids=top_down.index)
    blended = rn_model.blend_run_env(bottom_up, top_down, c_ctx["weight"])
    return {
        "team": blended,
        "lg_ra9": lg_ra9,
        "starter_ip": sp_day["starter_ip"],
        # Diagnostics: clubs the bottom-up half could not price and that
        # therefore kept their top-down rate for that column.
        "rs_missing": {int(t) for t in top_down.index if t not in set(rs9.index)},
        "ra_missing": {int(t) for t in top_down.index
                       if int(t) not in rot_ra9 or int(t) not in pen_ra9},
    }


def run_env_game_probs(g, c_day: dict, sp_day: dict, hfa: float,
                       bp_day: dict | None = None,
                       bp_ctx: dict | None = None) -> dict:
    """P(home) for `pythag_C`, `pythag_C_sp` and `pythag_C_sp_bpa`, plus diagnostics.

    `pythag_C` is the blended run environment straight into Pythagenpat + log5
    + HFA — no starter, so it is the station D comparison. `pythag_C_sp` adds
    exactly the delta `pythag_60_sp` adds, on top of C's runs allowed, so the
    pair isolates what the bottom-up rebuild is worth with the pitcher held
    fixed. When no probable is posted it falls back to `pythag_C`.

    Note that the starter is added as the same *delta from league average* it
    is everywhere else, on top of a runs-allowed rate that already contains
    half a rotation term over the same 5.5 innings. That is a mild
    double-count — and it is exactly the one `pythag_60_sp` already makes
    against a team RA/9 that contains the club's whole rotation, which is what
    keeps the two directly comparable. Replacing C's rotation slot with
    tonight's starter instead of adding to it is a cleaner construction and is
    untested (docs/market-benchmark-2026.md).
    """
    team, lg_ra9 = c_day["team"], c_day["lg_ra9"]
    sp_ids = sp_day["probables"].get(int(g.game_pk))
    talent, talent_sp, talent_bpa, talent_ip = {}, {}, {}, {}
    shift, ip_used = 0.0, 0.0
    for side, team_id, i in (("home", g.home_id, 0), ("away", g.away_id, 1)):
        rs9 = max(float(team.loc[team_id, "rs_pg"]), MIN_R9)
        ra9 = max(float(team.loc[team_id, "ra_pg"]), MIN_R9)
        talent[side] = pythagenpat(rs9, ra9, 1.0)
        if sp_ids is not None:
            ra9_sp = sp_model.blend_starter_team(
                sp_day["sp_ra9"].get(int(sp_ids[i]), lg_ra9), ra9, lg_ra9,
                starter_ip=c_day["starter_ip"])
            talent_sp[side] = pythagenpat(rs9, max(float(ra9_sp), MIN_R9), 1.0)
            if bp_day is not None:
                ra9_bpa, d = pen_delta_ra9(float(ra9_sp), team_id, sp_ids[i],
                                           bp_day, bp_ctx, "available")
                shift += d
                talent_bpa[side] = pythagenpat(rs9, max(ra9_bpa, MIN_R9), 1.0)
                # ...and the same two deltas with the game split at this
                # starter's own expected innings instead of the flat 5.5.
                ip = bp_day["sp_ip"].get(int(sp_ids[i]), c_day["starter_ip"])
                ip_used += ip
                ra9_ip = sp_model.blend_starter_team(
                    sp_day["sp_ra9"].get(int(sp_ids[i]), lg_ra9), ra9, lg_ra9,
                    starter_ip=ip)
                ra9_ip, _ = pen_delta_ra9(float(ra9_ip), team_id, sp_ids[i],
                                          bp_day, bp_ctx, "available",
                                          relief_ip=sp_model.GAME_IP - ip)
                talent_ip[side] = pythagenpat(rs9, max(ra9_ip, MIN_R9), 1.0)
    p_c = float(home_win_prob(talent["home"], talent["away"], hfa))
    p_c_sp = (p_c if sp_ids is None else
              float(home_win_prob(talent_sp["home"], talent_sp["away"], hfa)))
    missing = sum(int(t) in c_day["rs_missing"] or int(t) in c_day["ra_missing"]
                  for t in (g.home_id, g.away_id))
    out = {C_MODEL: p_c, C_SP_MODEL: p_c_sp,
           "c_sp_fallback": sp_ids is None, "c_partial": missing}
    if bp_day is not None:
        out[C_SP_BPA_MODEL] = (p_c_sp if sp_ids is None else
                               float(home_win_prob(talent_bpa["home"],
                                                   talent_bpa["away"], hfa)))
        out[C_SP_BPA_IP_MODEL] = (p_c_sp if sp_ids is None else
                                  float(home_win_prob(talent_ip["home"],
                                                      talent_ip["away"], hfa)))
        out["c_bpa_shift"] = shift
        out["c_starter_ip"] = ip_used
    return out


def build_c_context(season: int, lu_ctx: dict, bp_ctx: dict, weight: float,
                    share_window: int | None, rotation_days: int | None,
                    rotation_top_n: int, pa_per_game: float,
                    control: str = "none") -> dict:
    """Fetch the club-attributed hitting logs station C needs, once.

    `lu_ctx["game_logs"]` cannot be reused for this: `fetch_batter_game_logs`
    drops `team_id`, and station C's whole point is *which club* a hitter has
    been taking his plate appearances for. `fetch_hitter_game_logs` reads the
    same cached responses and keeps the team, so the extra fetch is free after
    the lineup context has run.

    The batter universe is every hitter who has appeared in a posted lineup
    this season (`lu_ctx`) — 661 men on 2025, 637 on 2026. Measured against
    the clubs' own team hitting logs that covers **99.98%** of all plate
    appearances taken in 2026 (99.99% in 2025), worst club 99.75%: what it
    misses is the pinch hitter who never started a game all year. Shares are
    normalised within the club, so a fraction of a percent of missing plate
    appearances moves a share by a fraction of a percent.
    """
    batters = pd.unique(lu_ctx["game_logs"]["batter"]) if len(lu_ctx["game_logs"]) else []
    logs = fetch_hitter_game_logs(batters, season)
    logs = logs.loc[:, ["batter", "team_id", "date", "pa"]].dropna(subset=["team_id"])
    return {"hitter_logs": logs, "starts": bp_ctx["starts"], "weight": weight,
            "share_window": share_window, "rotation_days": rotation_days,
            "rotation_top_n": rotation_top_n, "pa_per_game": pa_per_game,
            "control": control}


def join_market(preds: pd.DataFrame, closes: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add one column per venue (P(home) at the close) and keep only games
    every venue priced, so all models are scored on the same population."""
    wide = closes.pivot_table(index="game_pk", columns="venue", values="p_home_close")
    wide.columns = [f"{v}_close" for v in wide.columns]
    joined = preds.merge(wide, left_on="game_pk", right_index=True, how="inner")
    market_models = list(wide.columns)
    joined = joined.dropna(subset=market_models)
    return joined, market_models


def paired_t_line(df: pd.DataFrame, model: str, base: str) -> str:
    """Paired t on the per-game Brier difference (model - base).

    Brier is a mean of per-game squared errors, so the same games scored by two
    models are a paired sample and the difference has a standard error. A
    negative t means `model` is better; |t| ~ 2 is where the gain stops being
    inside one standard error.
    """
    y = df["home_win"].astype(float).to_numpy()
    d = (df[model].to_numpy() - y) ** 2 - (df[base].to_numpy() - y) ** 2
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    t = d.mean() / se if se else np.nan
    return (f"\npaired Brier {model} - {base}: {d.mean():+.5f} "
            f"(se {se:.5f}, t = {t:+.2f}, n = {n})")


def model_names(*groups) -> list[str]:
    """The model columns to score, in order, each named exactly once.

    The scored set is assembled in pieces — the ballast sweep, then whichever
    of the station models were built, then one column per market venue — and
    the venue columns are appended to `models` when `--market` is joined *and*
    kept in a separate list for the JSON payload's `market_models`. Adding the
    two together at the end therefore scored, printed and wrote every venue
    twice; downstream the site had to dedupe by model name to draw a table.
    Assemble the list through here instead, so a name can only appear once
    however many times it is contributed.
    """
    seen, names = set(), []
    for name in [n for group in groups for n in group]:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def score(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    y = df["home_win"].astype(float).to_numpy()
    out = []
    for m in models:
        p = np.clip(df[m].to_numpy(), 1e-6, 1 - 1e-6)
        out.append({"model": m,
                    "brier": float(np.mean((p - y) ** 2)),
                    "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                    "mean_p_home": float(p.mean())})
    return pd.DataFrame(out).sort_values("brier").reset_index(drop=True)


def score_payload(preds: pd.DataFrame, models: list[str], venues: list[str], *,
                  generated_at: str, season: int, min_games: int,
                  market_file: str | None, sp_fallback_games: int | None,
                  sp_no_history_slots: int | None) -> dict:
    """The document `--json-out` writes: the printed table plus its provenance.

    `preds` is the market-joined common-game set when `--market` was given and
    the full walk-forward set otherwise; either way it is the population the
    table names. One row per model, each model exactly once — `venues` is
    already inside `models` by the time this is called, and the site reads the
    rows straight into a ranked table, so a repeated name would be a repeated
    row on the page.
    """
    table = score(preds, model_names(models, venues))
    return {
        "generated_at": generated_at,
        "season": season,
        "min_games": min_games,
        "n_games": int(len(preds)),
        "first_date": str(preds["date"].min()),
        "last_date": str(preds["date"].max()),
        "market_file": market_file,
        "market_models": list(venues),
        "realized_home_win_rate": float(preds["home_win"].mean()),
        "scores": json.loads(table.to_json(orient="records")),
        "sp_fallback_games": sp_fallback_games,
        "sp_no_history_slots": sp_no_history_slots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--min-games", type=int, default=20,
                        help="skip dates until every team has this many games")
    parser.add_argument("--ballasts", default="0,30,60,100,160")
    parser.add_argument("--no-starters", action="store_true",
                        help="skip the pythag_60_sp starting-pitcher model")
    parser.add_argument("--sp-ballast", type=float, default=None,
                        help="override the per-component regression ballasts in "
                             "src/sim/starters with one batters-faced number "
                             "(the defaults are stabilization points from the "
                             "literature; nothing here is fit to this season)")
    parser.add_argument("--starter-ip", type=float, default=sp_model.STARTER_IP,
                        help="innings the starter is assumed to cover")
    parser.add_argument("--no-lineups", action="store_true",
                        help="skip the pythag_60_sp_lu posted-lineup model")
    parser.add_argument("--lu-ballast", type=float, default=None,
                        help="override the per-component regression ballasts in "
                             "src/sim/lineups with one number (the defaults are "
                             "stabilization points from the literature)")
    parser.add_argument("--lu-weight", type=float, default=LU_WEIGHT,
                        help="how much of the lineup's delta from its baseline "
                             "to apply to the team's runs scored")
    parser.add_argument("--lu-baseline", choices=("league", "team"),
                        default=LU_BASELINE,
                        help="measure the posted lineup against league average "
                             "or against the club's own typical lineup")
    parser.add_argument("--lu-baseline-ballast", type=float,
                        default=lu_model.BASELINE_BALLAST_GAMES,
                        help="league-average games of ballast on a club's own "
                             "lineup baseline")
    parser.add_argument("--no-bullpen", action="store_true",
                        help="skip the pythag_60_sp_lu_bp bullpen-availability model")
    parser.add_argument("--bp-baseline", choices=("league", "team"),
                        default=BP_BASELINE,
                        help="measure the available pen against the league's "
                             "relievers or against the club's own whole pen")
    parser.add_argument("--bp-roster-days", type=int,
                        default=bp_model.ROSTER_WINDOW_DAYS,
                        help="trailing days of relief work that define a pen")
    parser.add_argument("--bp-rest-days", type=int, default=bp_model.REST_DAYS,
                        help="how many calendar days back the rest rule looks")
    parser.add_argument("--bp-rest-min-days", type=int,
                        default=bp_model.REST_MIN_DAYS,
                        help="days worked inside that window that make a "
                             "reliever unavailable (2 = back-to-back)")
    parser.add_argument("--bpa-baseline", choices=("league", "team"),
                        default=BPA_BASELINE,
                        help="measure the availability-weighted pen against the "
                             "league's relievers on the same weights, or "
                             "against the club's own whole pen (availability "
                             "news only)")
    parser.add_argument("--bpa-hard-1d", type=float, default=BPA_HARD_1D,
                        help="pitches thrown yesterday that rule a reliever out")
    parser.add_argument("--bpa-hard-2d", type=float, default=BPA_HARD_2D,
                        help="pitches thrown over the last two days that rule a "
                             "reliever out")
    parser.add_argument("--bpa-taper", type=float, default=BPA_TAPER,
                        help="recency-discounted pitch load at which a reliever "
                             "who is still usable would be scored at zero")
    parser.add_argument("--sp-ip-ballast", type=float, default=SP_IP_BALLAST,
                        help="league-average starts of ballast on a starter's "
                             "own innings per start, for the model that splits "
                             "the game there instead of at 5.5")
    parser.add_argument("--relief-ip", type=float, default=bp_model.RELIEF_IP,
                        help="innings the bullpen is assumed to cover")
    parser.add_argument("--no-run-env", action="store_true",
                        help="skip the station C pythag_C / pythag_C_sp models")
    parser.add_argument("--c-weight", type=float, default=C_WEIGHT,
                        help="how much of the bottom-up run environment to use "
                             "(0 = the production top-down rates exactly)")
    parser.add_argument("--c-share-window", type=int, default=C_SHARE_WINDOW,
                        help="trailing days of plate appearances that define a "
                             "club's hitters and their PA shares "
                             "(0 = the season to date)")
    parser.add_argument("--c-rotation-days", type=int, default=C_ROTATION_DAYS,
                        help="trailing days of starts that define a rotation "
                             "(0 = the season to date)")
    parser.add_argument("--c-control", choices=("none", "league"), default="none",
                        help="replace the bottom-up half with league average, "
                             "so pythag_C is pythag_60 shrunk --c-weight of the "
                             "way to the league and nothing else — the control "
                             "that says whether C's gain is roster information "
                             "or plain shrinkage")
    parser.add_argument("--c-rotation-top-n", type=int,
                        default=rn_model.ROTATION_TOP_N,
                        help="how many starters make a rotation")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the per-game prediction frame here (parquet) "
                             "for follow-up analysis")
    parser.add_argument("--market", type=Path, default=None,
                        help="market_closes parquet from scripts/backfill_market_closes.py; "
                             "scores each venue's pre-game close as a model on the common games")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="also write the printed score table to PATH as JSON "
                             "(the market-joined table when --market is given)")
    args = parser.parse_args()
    ballasts = [float(b) for b in args.ballasts.split(",")]

    teams = fetch_teams(args.season)
    sched = fetch_schedule(f"{args.season}-03-01", f"{args.season}-10-15")
    state = from_schedule(sched, teams)
    # from_schedule drops scores; re-attach for run totals
    scored = sched[sched["status"] == "Final"].dropna(subset=["home_score", "away_score"])
    scored = scored[scored["home_score"] != scored["away_score"]].copy()
    scored["home_win"] = scored["home_score"] > scored["away_score"]
    scored = scored[scored["game_type"] == "R"]

    sp_ctx = None
    if not args.no_starters:
        sp_ctx = build_sp_context(
            args.season, scored,
            sp_model.BALLAST_BF if args.sp_ballast is None else args.sp_ballast,
            args.starter_ip)

    lu_ctx = None
    if sp_ctx is not None and not args.no_lineups:
        lu_ctx = build_lu_context(
            args.season, scored,
            lu_model.BALLAST if args.lu_ballast is None else args.lu_ballast,
            args.lu_weight, args.lu_baseline, args.lu_baseline_ballast,
            # Plate appearances per team-game, taken from the league's own
            # batters-faced-per-inning rather than assumed.
            pa_per_game=sp_ctx["league"]["bf_per_ip"] * 9.0)

    bp_ctx = None
    if lu_ctx is not None and not args.no_bullpen:
        bp_ctx = build_bp_context(
            args.season,
            sp_model.BALLAST_BF if args.sp_ballast is None else args.sp_ballast,
            args.bp_baseline, args.bp_roster_days, args.bp_rest_days,
            args.bp_rest_min_days, args.relief_ip,
            sp_ctx["league"], sp_ctx["prior_counts"],
            bpa_baseline=args.bpa_baseline, hard_1d=args.bpa_hard_1d,
            hard_2d=args.bpa_hard_2d, taper=args.bpa_taper,
            ip_ballast=args.sp_ip_ballast)

    c_ctx = None
    if bp_ctx is not None and not args.no_run_env:
        c_ctx = build_c_context(
            args.season, lu_ctx, bp_ctx, args.c_weight,
            args.c_share_window if args.c_share_window > 0 else None,
            args.c_rotation_days if args.c_rotation_days > 0 else None,
            args.c_rotation_top_n, lu_ctx["pa_per_game"], args.c_control)

    preds = walk_forward(scored, teams["team_id"].to_numpy(), args.min_games,
                         ballasts, sp_ctx, lu_ctx, bp_ctx, c_ctx)
    models = ["home_constant", "win_pct_log5"] + [f"pythag_{int(k)}" for k in ballasts]
    if sp_ctx is not None:
        models.append(SP_MODEL)
    if lu_ctx is not None:
        models.append(LU_MODEL)
    if bp_ctx is not None:
        models += [BP_MODEL, BPA_MODEL]
    if c_ctx is not None:
        models += [C_MODEL, C_SP_MODEL]
        if bp_ctx is not None:
            models += [C_SP_BPA_MODEL, C_SP_BPA_IP_MODEL]
    print(f"{len(preds)} games scored (from the date every team had {args.min_games}+ games)\n")
    print(score(preds, models).round(4).to_string(index=False))
    if sp_ctx is not None:
        print(f"\n{SP_MODEL}: {int(preds['sp_fallback'].sum())} of {len(preds)} games fell back "
              f"to pythag_{int(SP_BALLAST_GAMES)} for a missing starter; "
              f"{int(preds['sp_no_history'].sum())} starter slots had no history "
              f"(scored at league average).")
    if lu_ctx is not None:
        print(f"{LU_MODEL}: {int(preds['lu_fallback'].sum())} of {len(preds)} games fell back "
              f"to {SP_MODEL} for a missing lineup; "
              f"{int(preds['lu_no_history'].sum())} lineup slots had no history "
              f"(scored at league average). "
              f"weight={args.lu_weight}, baseline={args.lu_baseline}.")
    if bp_ctx is not None:
        print(f"{BP_MODEL}: {int(preds['bp_short'].sum())} of {2 * len(preds)} club-games "
              f"took the mound with a reliever unavailable; "
              f"baseline={args.bp_baseline}, pen={args.bp_roster_days}d, "
              f"rest={args.bp_rest_min_days}/{args.bp_rest_days}d, "
              f"relief_ip={args.relief_ip}.")
        print(f"{BPA_MODEL}: {int(preds['bpa_short'].sum())} of {2 * len(preds)} "
              f"club-games took the mound with a pen short of full availability, "
              f"mean shift {preds['bpa_shift'].sum() / (2 * len(preds)):.3f} runs "
              f"per nine; baseline={args.bpa_baseline}, "
              f"hard={args.bpa_hard_1d:.0f}/{args.bpa_hard_2d:.0f} pitches, "
              f"taper={args.bpa_taper:.0f}.")
        print(f"{C_SP_BPA_IP_MODEL}: mean expected start "
              f"{preds['c_starter_ip'].sum() / (2 * len(preds)):.2f} innings "
              f"against the flat {sp_model.STARTER_IP}; "
              f"ip_ballast={args.sp_ip_ballast:.0f} starts.")
    if c_ctx is not None:
        print(f"{C_SP_MODEL}: {int(preds['c_sp_fallback'].sum())} of {len(preds)} games "
              f"fell back to {C_MODEL} for a missing starter; "
              f"{int(preds['c_partial'].sum())} of {2 * len(preds)} club-games had no "
              f"bottom-up estimate for one half and kept the top-down rate. "
              f"weight={args.c_weight}, "
              f"shares={str(args.c_share_window) + 'd' if args.c_share_window > 0 else 'season'}, "
              f"control={args.c_control}, "
              f"rotation=top{args.c_rotation_top_n}/"
              f"{args.c_rotation_days if args.c_rotation_days > 0 else 'season'}.")

    if args.market is not None:
        preds, market_models = join_market(preds, pd.read_parquet(args.market))
        models = model_names(models, market_models)
        print(f"\n{len(preds)} games also priced by every venue in {args.market.name} — "
              f"the market is the bar (docs/architecture.md §0):\n")
        print(score(preds, models).round(5).to_string(index=False))
        if sp_ctx is not None:
            print(f"\n  of these, {int(preds['sp_fallback'].sum())} fell back to "
                  f"pythag_{int(SP_BALLAST_GAMES)} for a missing starter.")
        if lu_ctx is not None:
            print(f"  of these, {int(preds['lu_fallback'].sum())} fell back to "
                  f"{SP_MODEL} for a missing lineup.")
        if bp_ctx is not None:
            print(f"  of these, {int(preds['bp_short'].sum())} of "
                  f"{2 * len(preds)} club-games were a reliever short, and "
                  f"{int(preds['bpa_short'].sum())} were short of full "
                  f"availability on the pitch-count reading (mean shift "
                  f"{preds['bpa_shift'].sum() / (2 * len(preds)):.3f} runs/9).")
        if c_ctx is not None:
            print(f"  of these, {int(preds['c_sp_fallback'].sum())} fell back to "
                  f"{C_MODEL} for a missing starter and {int(preds['c_partial'].sum())} "
                  f"of {2 * len(preds)} club-games kept a top-down half.")

    # Paired comparisons on the scored population: is the new term's gain real?
    pairs = []
    if sp_ctx is not None:
        pairs.append((SP_MODEL, "pythag_60"))
    if lu_ctx is not None:
        pairs += [(LU_MODEL, SP_MODEL), (LU_MODEL, "pythag_60")]
    if bp_ctx is not None:
        pairs += [(BP_MODEL, LU_MODEL), (BP_MODEL, SP_MODEL),
                  (BPA_MODEL, BP_MODEL), (BPA_MODEL, LU_MODEL)]
    if c_ctx is not None:
        # The gate: C with the starter on top against the same model without
        # the bottom-up rebuild, and C alone against the production model.
        pairs += [(C_SP_MODEL, SP_MODEL), (C_MODEL, "pythag_60")]
        if bp_ctx is not None:
            # The reliever-availability gate: the one new term against the best
            # model there is.
            pairs += [(C_SP_BPA_MODEL, C_SP_MODEL),
                      (C_SP_BPA_IP_MODEL, C_SP_BPA_MODEL)]
    for model, base in pairs:
        print(paired_t_line(preds, model, base))

    # Calibration of the production model, and of the challengers
    for model in ["pythag_60"] + ([SP_MODEL] if sp_ctx is not None else []) + \
                 ([LU_MODEL] if lu_ctx is not None else []) + \
                 ([BP_MODEL] if bp_ctx is not None else []) + \
                 ([C_SP_MODEL] if c_ctx is not None else []) + \
                 ([C_SP_BPA_MODEL] if c_ctx is not None and bp_ctx is not None else []):
        buckets = pd.cut(preds[model], [0, .4, .45, .5, .55, .6, .65, 1.0])
        cal = preds.groupby(buckets, observed=True).agg(n=("home_win", "size"),
                                                        predicted=(model, "mean"),
                                                        realized=("home_win", "mean"))
        print(f"\nCalibration ({model}):")
        print(cal.round(3).to_string())

    if args.json_out is not None:
        from datetime import datetime, timezone
        payload = score_payload(
            preds, models,
            list(market_models) if args.market is not None else [],
            generated_at=datetime.now(timezone.utc).isoformat(),
            season=args.season, min_games=args.min_games,
            market_file=args.market.name if args.market is not None else None,
            sp_fallback_games=(int(preds["sp_fallback"].sum())
                               if sp_ctx is not None else None),
            sp_no_history_slots=(int(preds["sp_no_history"].sum())
                                 if sp_ctx is not None else None))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(args.out, index=False)
        print(f"\nwrote {len(preds)} rows x {len(models)} models → {args.out}")


if __name__ == "__main__":
    main()
