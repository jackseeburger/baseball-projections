"""Station B-pitchers — projected rest-of-season batters faced, and the
baselines it has to beat.

The pitcher line the site serves is a rate times a workload, exactly as the
hitter line is (`ros.py` x `playing_time.py`). The rates cleared a gate; the
workload never had one. `pitcher_ros.projected_batters_faced` is stamped
`structural` for that reason — it is arithmetic on recent usage that nobody
had ever scored against an alternative. This module is the missing half: the
same quantity, several honest ways of estimating it, and a walk-forward
harness that scores all of them on the same pitchers.

The unit is deliberately a parameter. `unit="bf"` projects batters faced,
which is what the site multiplies the rates by; `unit="outs"` projects outs,
whose third is innings. They are the same model over a different column, so
innings are scored independently rather than being read off batters faced
through a league constant.

**The methods.** Every one of them maps a cutoff to `pitcher, projected` and
is scored on what the pitcher actually did from the cutoff to the end of the
scored window.

    zero              nobody pitches again. The no-model floor: MAE here is
                      the mean rest-of-season workload, and any method that
                      cannot beat it is worse than silence.
    last_season       last season's total, prorated to the games left. The
                      only baseline that uses no current-season information.
    season_rate       season-to-date workload per club game x games left. The
                      "rate times games remaining" extrapolation, with no
                      role, no regression and no roster gate.
    recent_rate       the same over the trailing 30 days only.
    structural        the projection the site serves today, bit for bit
                      (`pitcher_ros.projected_batters_faced`).
    structural_nogate the same with the injured-list fractions removed, which
                      isolates what station B's return-time distribution is
                      worth on the served model.
    blend             role read off starts rather than a batters-faced
                      threshold; appearance rate blended across the trailing
                      window and the season with a horizon weight; both halves
                      regressed toward the *league's own* role averages at the
                      cutoff rather than toward frozen constants. Unavailable
                      pitchers are zeroed.
    blend_il          the station B fix, ported: an unavailable pitcher is
                      weighed as he was the day he went out, and scaled by the
                      fraction of the horizon he is expected to be back for.
    structural_hazard the served projection times an attrition term: a
                      constant per-club-game hazard, per role, of a currently
                      healthy pitcher losing the rest of his season. One at a
                      short horizon and a haircut at a long one, which is the
                      shape the residual actually has.
    structural_cal    the served projection times one constant per role,
                      chosen on the fitting seasons to minimize MAE. The
                      cheapest possible improvement, and the one a projection
                      built as an expectation invites when it is scored on an
                      absolute error.
    blend_il_share    the pitcher analogue of station B's share
                      normalization: a club's staff must, between them, face
                      the club's opponents. Every staff's projections are
                      scaled to the club's own projected total.

Walk-forward honesty: every function that reads appearances, rosters or
transactions takes a cutoff and uses only rows strictly before it. A game
played *on* the cutoff has not finished when the morning's projection is made.
`tests/test_projections/test_pitcher_workload.py` pins that with a synthetic
season whose post-cutoff rows are absurd.

Everything is a pure function over DataFrames; the fetch layer is
`scripts/build_pitcher_workload.py` and the harness is
`scripts/run_pitcher_workload_backtest.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.projections import pitcher_ros
from src.projections.playing_time import logistic_from_anchors

# --- constants ---------------------------------------------------------
#
# Every number below that is not lineup or calendar arithmetic was chosen on
# the 2022 and 2023 seasons and frozen before 2024-2026 were scored. See
# docs/pitcher-workload.md.

# Trailing window, in days. Thirty is station B's, and a month is about
# twenty-six club games — five or six turns for a starter and ten outings for
# a reliever, which is the shortest window in which an appearance rate is
# estimable at all.
RECENT_DAYS = 30

# A pitcher whose appearances are at least this fraction starts is being used
# as a starter. Read off `gamesStarted`, not off batters faced: an opener
# starts, and the served model calls him a reliever because he faces four
# batters. Whether that is the right call is an empirical question and this is
# the side of it the data can answer.
STARTER_START_SHARE = 0.5
# Role is read from the trailing window when he pitched in it, and from the
# season otherwise. Forty-five days is long enough that a starter who missed a
# fortnight is still a starter.
ROLE_WINDOW_DAYS = 45

# Ballast on the appearance rate, in club games, and on the workload per
# appearance, in appearances. Both are regressions toward the league's own
# role average computed at the cutoff, not toward a frozen constant, so they
# only have to say how much of a short sample to believe.
RATE_BALLAST_GAMES = 12.0
UNIT_BALLAST_APPEARANCES = 4.0

# The horizon blend, in station B's parameterization: the weight on the
# trailing window at 30 and at 90 club games remaining, with a logistic
# through them. Chosen by sweeping a constant weight at every 2022-2023 cutoff
# and fitting the two anchors to the resulting curves.
BLEND_ANCHOR_GAMES = (30.0, 90.0)
BLEND_WEIGHT_SHORT = 0.76
BLEND_WEIGHT_LONG = 0.43
BLEND_MIDPOINT_GAMES, BLEND_SCALE_GAMES = logistic_from_anchors(
    BLEND_WEIGHT_SHORT, BLEND_WEIGHT_LONG)

# A pitcher needs this many appearances before his own appearance rate is
# used to classify him at all; below it he is whatever his starts say and the
# ballast does the rest.
MIN_APPEARANCES_FOR_ROLE = 1

# The multiplier that minimizes MAE on the fitting seasons, per role, applied
# to the served projection. Rest-of-season workload has a long left tail — a
# pitcher who tears something in August faces nobody — so MAE is minimized at
# a conditional *median* and a projection built as an expectation sits above
# it. Chosen on 2022-2023 by `--calibrate --method structural` and frozen
# before 2024-2026 were scored.
STRUCTURAL_CALIBRATION = {"SP": 0.90, "RP": 0.87}

# Per-club-game hazard of a *currently healthy* pitcher losing the rest of his
# season, per role. The served projection assumes he keeps his turn until the
# end of the year; over a three-month horizon a fair number of pitchers do not,
# and none of the model's terms knows it. With a constant hazard the expected
# share of a horizon of `h` club games he is still available for is
# `(1 - exp(-lambda h)) / (lambda h)` — one at a short horizon, a haircut at a
# long one, which is the shape the by-horizon table in docs/pitcher-workload.md
# actually shows. Two parameters, chosen on 2022-2023 by
# `--calibrate-hazard` and frozen before 2024-2026 were scored.
ATTRITION_HAZARD = {"SP": 0.0020, "RP": 0.0030}

BASELINES = ("zero", "last_season", "season_rate", "recent_rate",
             "structural", "structural_nogate")
CANDIDATES = ("blend", "blend_il", "blend_il_share", "structural_cal",
              "structural_hazard")
METHODS = BASELINES + CANDIDATES

PRODUCTION_METHOD = "structural"

PROJECTION_COLUMNS = ["pitcher", "team_id", "role", "projected"]


def _as_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _dates(frame: pd.DataFrame, column: str = "date") -> pd.Series:
    return pd.to_datetime(frame[column]).dt.normalize()


def horizon_weight(games_remaining,
                   midpoint: float = BLEND_MIDPOINT_GAMES,
                   scale: float = BLEND_SCALE_GAMES):
    """Weight on the trailing window at a horizon of `games_remaining`.

    The same decreasing logistic station B uses, and for the same reason: the
    recent window answers "who is in the rotation now" and the season answers
    "who will still be in it in September", and which one is better is a
    function of the horizon rather than a fact about pitching.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    h = np.asarray(games_remaining, dtype=float)
    w = 1.0 / (1.0 + np.exp(np.clip((h - midpoint) / scale, -60.0, 60.0)))
    return w if w.ndim else float(w)


# --- windows over the appearance log -----------------------------------

def window_totals(appearances: pd.DataFrame, cutoff, unit: str = "bf",
                  window_days: int | None = None) -> pd.DataFrame:
    """Per-pitcher totals in `[cutoff - window_days, cutoff)`.

    `appearances`: pitcher, date, bf, outs, gs — one row per appearance.
    `window_days=None` is the season to date. The upper bound is strict.

    Returns pitcher, appearances, starts, `unit`.
    """
    cutoff = _as_date(cutoff)
    cols = ["pitcher", "appearances", "starts", unit]
    if appearances is None or not len(appearances):
        return pd.DataFrame(columns=cols)
    dates = _dates(appearances)
    mask = dates < cutoff
    if window_days is not None:
        mask &= dates >= cutoff - pd.Timedelta(days=int(window_days))
    kept = appearances[mask]
    if not len(kept):
        return pd.DataFrame(columns=cols)
    out = (kept.groupby("pitcher", as_index=False)
           .agg(appearances=("pitcher", "size"),
                starts=("gs", "sum"),
                **{unit: (unit, "sum")}))
    return out.loc[:, cols]


def realized(appearances: pd.DataFrame, start, end, unit: str = "bf") -> pd.DataFrame:
    """What each pitcher actually did in `[start, end]`, both ends inclusive.

    `start` is the cutoff — the first day the projection is responsible for.
    """
    start, end = _as_date(start), _as_date(end)
    cols = ["pitcher", "realized"]
    if appearances is None or not len(appearances):
        return pd.DataFrame(columns=cols)
    dates = _dates(appearances)
    kept = appearances[(dates >= start) & (dates <= end)]
    if not len(kept):
        return pd.DataFrame(columns=cols)
    return (kept.groupby("pitcher", as_index=False)[unit].sum()
            .rename(columns={unit: "realized"}))


def club_games(team_games: pd.DataFrame, lo, hi) -> pd.Series:
    """Club games in `[lo, hi)`, per team_id."""
    lo, hi = _as_date(lo), _as_date(hi)
    if team_games is None or not len(team_games):
        return pd.Series(dtype="float64")
    dates = _dates(team_games)
    kept = team_games[(dates >= lo) & (dates < hi)]
    if not len(kept):
        return pd.Series(dtype="float64")
    return (kept.groupby("team_id")["game_pk"].nunique().astype(float))


def games_remaining(schedule: pd.DataFrame, cutoff, end) -> pd.Series:
    """Regular-season games each club still has to play in `[cutoff, end]`.

    From the schedule rather than from the game logs, which is what a
    projection made on the cutoff morning would have had.
    """
    cutoff, end = _as_date(cutoff), _as_date(end)
    if schedule is None or not len(schedule):
        return pd.Series(dtype="float64")
    dates = _dates(schedule)
    window = schedule[(dates >= cutoff) & (dates <= end)]
    if not len(window):
        return pd.Series(dtype="float64")
    counts = pd.concat([window["home_id"], window["away_id"]]).value_counts()
    counts.index.name = "team_id"
    return counts.astype(float)


# --- role, and the league's own role averages --------------------------

def role_of(starts, appearances, threshold: float = STARTER_START_SHARE) -> np.ndarray:
    """"SP" or "RP" from the share of appearances that were starts."""
    starts = np.asarray(starts, dtype=float)
    appearances = np.asarray(appearances, dtype=float)
    share = np.divide(starts, np.where(appearances > 0, appearances, np.nan),
                      out=np.zeros_like(starts), where=appearances > 0)
    return np.where(share >= threshold, "SP", "RP")


def role_priors(season_totals: pd.DataFrame, roles: np.ndarray,
                games_played: pd.Series, team_of: pd.Series,
                unit: str = "bf") -> dict:
    """League averages per role, computed from data before the cutoff only.

    Two numbers per role: workload per appearance (pooled, so a September
    call-up's three outings do not weigh as much as a full season) and
    appearances per club game (the median over pitchers with a real sample,
    so one pitcher who has been up for a week does not drag it).

    Returning the league's *current* averages rather than frozen constants is
    the point: 2022's rotation is not 2026's, and a projection made in May of
    a season can see May of that season.
    """
    out = {}
    frame = season_totals.assign(role=roles)
    games = frame["pitcher"].map(team_of).map(games_played).astype(float)
    for role in ("SP", "RP"):
        rows = frame[frame["role"] == role]
        g = games[frame["role"] == role]
        apps = float(rows["appearances"].sum())
        per_app = float(rows[unit].sum()) / apps if apps > 0 else np.nan
        established = rows["appearances"] >= 5
        rate = float((rows.loc[established, "appearances"]
                      / g[established].where(g[established] > 0)).median()) \
            if int(established.sum()) else np.nan
        out[role] = {"per_appearance": per_app, "rate": rate}
    # A season with no starters or no relievers before the cutoff cannot
    # happen in practice; fall back to the served model's constants so the
    # function is total.
    for role in ("SP", "RP"):
        if not np.isfinite(out[role]["per_appearance"]):
            out[role]["per_appearance"] = (
                pitcher_ros.ROLE_BF_PER_APPEARANCE[role] if unit == "bf"
                else pitcher_ros.ROLE_BF_PER_APPEARANCE[role] / 4.3 * 3.0)
        if not np.isfinite(out[role]["rate"]):
            out[role]["rate"] = pitcher_ros.ROLE_APPEARANCE_RATE[role]
    return out


# --- the inputs a cutoff needs -----------------------------------------

@dataclass
class CutoffInputs:
    """Everything a projection at one as-of date is allowed to see.

    Constructing this does not filter anything — every method filters on the
    cutoff itself, so a leakage bug shows up as a changed number rather than
    as a missing column. `tests/test_projections/test_pitcher_workload.py`
    exploits that: it hands the whole season in, with the post-cutoff rows made
    absurd, and asserts nothing moves.
    """
    cutoff: pd.Timestamp
    score_end: pd.Timestamp
    appearances: pd.DataFrame
    team_games: pd.DataFrame
    schedule: pd.DataFrame
    roster: pd.DataFrame
    prior_totals: pd.DataFrame = field(default_factory=pd.DataFrame)
    active_fraction: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    spell_start: pd.Series = field(default_factory=lambda: pd.Series(dtype="datetime64[ns]"))

    def __post_init__(self):
        self.cutoff = _as_date(self.cutoff)
        self.score_end = _as_date(self.score_end)


def _staff(inputs: CutoffInputs) -> pd.DataFrame:
    """`pitcher, team_id, status_code` — who is on a 40-man at the cutoff."""
    roster = inputs.roster.loc[:, ["pitcher", "team_id", "status_code"]].copy()
    roster["pitcher"] = roster["pitcher"].astype("int64")
    roster["team_id"] = roster["team_id"].astype("int64")
    return roster.drop_duplicates(subset="pitcher", keep="first").reset_index(drop=True)


def _horizon(inputs: CutoffInputs) -> pd.Series:
    return games_remaining(inputs.schedule, inputs.cutoff, inputs.score_end)


# --- the methods -------------------------------------------------------

def project(inputs: CutoffInputs, method: str, unit: str = "bf",
            **params) -> pd.DataFrame:
    """`pitcher, team_id, role, projected` for one method at one cutoff."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if method == "zero":
        return _project_zero(inputs, unit)
    if method == "last_season":
        return _project_last_season(inputs, unit)
    if method in ("season_rate", "recent_rate"):
        window = None if method == "season_rate" else RECENT_DAYS
        return _project_flat_rate(inputs, unit, window)
    if method in ("structural", "structural_nogate", "structural_cal",
                  "structural_hazard"):
        calibration = (params.get("calibration", STRUCTURAL_CALIBRATION)
                       if method == "structural_cal" else None)
        hazard = (params.get("hazard", ATTRITION_HAZARD)
                  if method == "structural_hazard" else None)
        return _project_structural(inputs, unit,
                                   gate=(method != "structural_nogate"),
                                   calibration=calibration, hazard=hazard)
    return _project_blend(inputs, unit, method=method, **params)


def _frame(staff: pd.DataFrame, projected, roles=None) -> pd.DataFrame:
    out = staff.loc[:, ["pitcher", "team_id"]].copy()
    out["role"] = "RP" if roles is None else roles
    out["projected"] = np.asarray(projected, dtype=float).clip(min=0.0)
    return out.loc[:, PROJECTION_COLUMNS].reset_index(drop=True)


def _project_zero(inputs: CutoffInputs, unit: str) -> pd.DataFrame:
    staff = _staff(inputs)
    return _frame(staff, np.zeros(len(staff)))


def _project_last_season(inputs: CutoffInputs, unit: str) -> pd.DataFrame:
    """Last season's total, prorated to the games left.

    The only baseline with no current-season information in it at all, which
    is what makes it the interesting floor: everything else on the list is
    allowed to know that the pitcher has been in the rotation since April.
    """
    staff = _staff(inputs)
    left = staff["team_id"].map(_horizon(inputs)).astype(float).fillna(0.0)
    prior = inputs.prior_totals
    total = (staff["pitcher"].map(prior.set_index("pitcher")[unit])
             if prior is not None and len(prior) else pd.Series(np.nan, index=staff.index))
    total = total.astype(float).fillna(0.0)
    return _frame(staff, total * left / 162.0)


def _project_flat_rate(inputs: CutoffInputs, unit: str,
                       window_days: int | None) -> pd.DataFrame:
    """Workload per club game in a window, times the club games left.

    No role, no regression, no roster gate — the extrapolation anybody would
    write first, and the one the structural model has to justify itself
    against.
    """
    staff = _staff(inputs)
    totals = window_totals(inputs.appearances, inputs.cutoff, unit, window_days)
    lo = (inputs.cutoff - pd.Timedelta(days=int(window_days))
          if window_days is not None else pd.Timestamp("1900-01-01"))
    played = club_games(inputs.team_games, lo, inputs.cutoff)
    left = staff["team_id"].map(_horizon(inputs)).astype(float).fillna(0.0)
    total = (staff["pitcher"].map(totals.set_index("pitcher")[unit])
             .astype(float).fillna(0.0) if len(totals)
             else pd.Series(0.0, index=staff.index))
    games = staff["team_id"].map(played).astype(float)
    rate = total / games.where(games > 0)
    return _frame(staff, (rate.fillna(0.0) * left))


def attrition_fraction(games_remaining, hazard: float):
    """Expected share of `games_remaining` a healthy pitcher is still around for.

    A constant per-club-game hazard `lambda` of losing the rest of the season,
    averaged over the horizon: `(1 - exp(-lambda h)) / (lambda h)`. One when
    the hazard is zero or the horizon is, and falling smoothly from there —
    the smallest form that is a survival curve rather than a fudge factor.
    """
    h = np.asarray(games_remaining, dtype=float)
    lam = float(hazard)
    if lam <= 0:
        out = np.ones_like(h)
        return out if h.ndim else float(out)
    x = np.clip(lam * h, 0.0, 60.0)
    out = np.where(x > 1e-9, (1.0 - np.exp(-x)) / np.where(x > 0, x, 1.0), 1.0)
    return out if h.ndim else float(out)


def _project_structural(inputs: CutoffInputs, unit: str, gate: bool = True,
                        calibration: dict | None = None,
                        hazard: dict | None = None) -> pd.DataFrame:
    """The served projection, through the served function.

    `pitcher_ros.projected_batters_faced` is called here rather than
    re-implemented, so what the harness scores is what the site publishes. It
    wants a season-to-date aggregate shaped like the PA parquet's, which is
    what `_structural_partial` builds out of the appearance log; the two agree
    because both are sums of the same per-appearance batters faced.

    `gate=False` drops the injured-list fractions, which is how the harness
    measures what station B's return-time distribution is worth *on the model
    that is actually running*.
    """
    staff = _staff(inputs)
    # The served function's role threshold and role priors are in *batters
    # faced*. Running it on outs means putting outs on that scale first —
    # divide by the league's outs per batter faced before the cutoff, project,
    # then multiply back — so a start is still a start and the regression
    # still points at the right average. Anything else would be scoring a
    # different model and calling it the served one.
    scale = _unit_scale(inputs, unit)
    partial = _structural_partial(inputs, inputs.cutoff, unit, None, scale)
    recent = _structural_partial(inputs, inputs.cutoff, unit, RECENT_DAYS, scale)
    played = club_games(inputs.team_games, pd.Timestamp("1900-01-01"), inputs.cutoff)
    recent_played = club_games(
        inputs.team_games, inputs.cutoff - pd.Timedelta(days=RECENT_DAYS),
        inputs.cutoff)
    fraction = inputs.active_fraction if gate else None
    if gate:
        # An unavailable pitcher the transactions cannot date is projected at
        # zero, which is the fallback the live builder uses.
        unavailable = set(staff.loc[staff["status_code"].astype(str) != "A",
                                    "pitcher"].astype("int64"))
        undated = sorted(unavailable - set(pd.Series(fraction).index))
        fraction = pd.concat([pd.Series(fraction, dtype="float64"),
                              pd.Series(0.0, index=undated, dtype="float64")])
    work = pitcher_ros.projected_batters_faced(
        partial, recent, played, recent_played, _horizon(inputs),
        staff.set_index("pitcher")["team_id"], fraction)
    out = staff.loc[:, ["pitcher", "team_id"]].merge(
        work.loc[:, ["pitcher", "role", "bf_ros"]], on="pitcher", how="left")
    out["role"] = out["role"].fillna("RP")
    out["projected"] = out["bf_ros"].astype(float).fillna(0.0) * scale
    if calibration:
        out["projected"] = out["projected"] * out["role"].map(
            calibration).astype(float).fillna(1.0)
    if hazard:
        left = out["team_id"].map(_horizon(inputs)).astype(float).fillna(0.0)
        lam = out["role"].map(hazard).astype(float).fillna(0.0)
        out["projected"] = out["projected"] * [
            attrition_fraction(h, l) for h, l in zip(left, lam)]
    return out.loc[:, PROJECTION_COLUMNS].reset_index(drop=True)


def _unit_scale(inputs: CutoffInputs, unit: str) -> float:
    """League `unit` per batter faced, before the cutoff. 1.0 for batters faced.

    About 0.70 for outs — every plate appearance that is not a hit, a walk or
    a hit batsman is an out somewhere. Taken from the season to date rather
    than from a constant, so it is data the projection could have had.
    """
    if unit == "bf":
        return 1.0
    log = inputs.appearances
    if log is None or not len(log):
        return 3.0 / 4.3
    past = log[_dates(log) < inputs.cutoff]
    bf = float(past["bf"].sum()) if len(past) else 0.0
    if bf <= 0:
        return 3.0 / 4.3
    return float(past[unit].sum()) / bf


def _structural_partial(inputs: CutoffInputs, cutoff, unit: str,
                        window_days: int | None,
                        scale: float = 1.0) -> pd.DataFrame:
    """The `pitcher, bf, games` aggregate the served function expects.

    `bf` carries whichever unit is being projected, divided by `scale` so it
    sits on the batters-faced scale the served constants (`STARTER_MIN_BF`,
    `ROLE_BF_PER_APPEARANCE`) are written in. The caller multiplies the
    projection back.
    """
    totals = window_totals(inputs.appearances, cutoff, unit, window_days)
    if not len(totals):
        return pd.DataFrame(columns=["pitcher", "bf", "games"])
    return pd.DataFrame({"pitcher": totals["pitcher"],
                         "bf": totals[unit].astype(float) / float(scale),
                         "games": totals["appearances"].astype(float)})


def _project_blend(inputs: CutoffInputs, unit: str, method: str,
                   weight_short: float = BLEND_WEIGHT_SHORT,
                   weight_long: float = BLEND_WEIGHT_LONG,
                   rate_ballast: float = RATE_BALLAST_GAMES,
                   unit_ballast: float = UNIT_BALLAST_APPEARANCES,
                   calibration: dict | None = None,
                   blend_weight: float | None = None) -> pd.DataFrame:
    """The candidate: role from starts, a horizon blend, league role priors.

        projected = appearance rate  x  club games left  x  workload/appearance

    with the appearance rate blended between the trailing window and the
    season and both halves regressed toward the league's own role averages at
    the cutoff. `blend_il` weighs an unavailable pitcher as he was the day he
    went out and scales him by the fraction of the horizon he is expected back
    for; `blend_il_share` additionally scales each club's staff so that between
    them they face the club's own projected total.
    """
    staff = _staff(inputs)
    cutoff = inputs.cutoff
    left = staff["team_id"].map(_horizon(inputs)).astype(float).fillna(0.0)

    season = window_totals(inputs.appearances, cutoff, unit, None)
    recent = window_totals(inputs.appearances, cutoff, unit, RECENT_DAYS)
    role_window = window_totals(inputs.appearances, cutoff, unit, ROLE_WINDOW_DAYS)

    played = club_games(inputs.team_games, pd.Timestamp("1900-01-01"), cutoff)
    recent_played = club_games(inputs.team_games,
                               cutoff - pd.Timedelta(days=RECENT_DAYS), cutoff)
    team_of = staff.set_index("pitcher")["team_id"]

    # Role: the trailing window when he pitched in it, the season otherwise.
    roles = _roles_for(staff, role_window, season)
    priors = role_priors(season, _roles_for_frame(season, role_window),
                         played, team_of, unit)
    prior_rate = pd.Series(roles, index=staff.index).map(
        {r: priors[r]["rate"] for r in priors}).astype(float)
    prior_unit = pd.Series(roles, index=staff.index).map(
        {r: priors[r]["per_appearance"] for r in priors}).astype(float)

    # Who is out, how long he has been out, and how much of the horizon he is
    # expected back for. `blend` keeps the hard zero; `blend_il` replaces it.
    status = staff["status_code"].astype(str)
    active = status.eq("A")
    fraction = active.astype(float)
    as_of = pd.Series(cutoff, index=staff.index)
    if method != "blend":
        f = staff["pitcher"].map(pd.Series(inputs.active_fraction, dtype="float64"))
        fraction = fraction.where(active, f.fillna(0.0).clip(0.0, 1.0))
        spells = pd.Series(inputs.spell_start)
        starts = (pd.to_datetime(staff["pitcher"].map(spells)) if len(spells)
                  else pd.Series(pd.NaT, index=staff.index))
        # Clamped at the cutoff: a spell the transactions date on or after
        # the cutoff would otherwise send this pitcher's windows forward in
        # time, which is the one way this method could read the future.
        as_of = as_of.where(active | starts.isna(), starts.clip(upper=cutoff))

    rate, per_app = _rate_and_unit(
        inputs, staff, as_of, unit, prior_rate, prior_unit,
        rate_ballast, unit_ballast, played, recent_played,
        season, recent)

    h = left.to_numpy(float)
    w = (np.full(len(staff), float(blend_weight)) if blend_weight is not None
         else horizon_weight(h, *logistic_from_anchors(weight_short, weight_long)))
    blended = w * rate["recent"].to_numpy(float) + (1.0 - w) * rate["season"].to_numpy(float)
    projected = pd.Series(blended, index=staff.index) * left * per_app * fraction

    if calibration:
        projected = projected * pd.Series(roles, index=staff.index).map(
            calibration).astype(float).fillna(1.0)

    if method == "blend_il_share":
        projected = _normalize_to_club(projected, staff["team_id"], left,
                                       inputs, unit)
    return _frame(staff, projected, roles)


def _roles_for(staff: pd.DataFrame, role_window: pd.DataFrame,
               season: pd.DataFrame) -> np.ndarray:
    """One role per staff row: the trailing window if he pitched in it."""
    def lookup(table, column):
        if not len(table):
            return pd.Series(0.0, index=staff.index)
        return (staff["pitcher"].map(table.set_index("pitcher")[column])
                .astype(float).fillna(0.0))
    apps_w, starts_w = lookup(role_window, "appearances"), lookup(role_window, "starts")
    apps_s, starts_s = lookup(season, "appearances"), lookup(season, "starts")
    apps = apps_w.where(apps_w >= MIN_APPEARANCES_FOR_ROLE, apps_s)
    starts = starts_w.where(apps_w >= MIN_APPEARANCES_FOR_ROLE, starts_s)
    return role_of(starts, apps)


def _roles_for_frame(season: pd.DataFrame, role_window: pd.DataFrame) -> np.ndarray:
    """`_roles_for` on the season-totals frame itself, for the league priors."""
    if not len(season):
        return np.array([], dtype=object)
    staff = season.loc[:, ["pitcher"]].copy()
    return _roles_for(staff, role_window, season)


def _rate_and_unit(inputs: CutoffInputs, staff: pd.DataFrame, as_of: pd.Series,
                   unit: str, prior_rate: pd.Series, prior_unit: pd.Series,
                   rate_ballast: float, unit_ballast: float,
                   played: pd.Series, recent_played: pd.Series,
                   season: pd.DataFrame, recent: pd.DataFrame):
    """Regressed appearance rate (season and recent) and workload per outing.

    `as_of` is per pitcher: the cutoff for everyone who is available, and the
    day he went out for everyone who is not. That is the whole of station B's
    injured-list fix — a starter three weeks into a stint has an empty trailing
    window *because* he is hurt, and reading it at the cutoff says he is a
    replacement-level nobody rather than the pitcher he was.
    """
    team_of = staff.set_index("pitcher")["team_id"]
    n = len(staff)
    rate_season = np.zeros(n)
    rate_recent = np.zeros(n)
    per_app = np.zeros(n)

    # Everyone read at the cutoff shares one pass over the log.
    cutoff = inputs.cutoff
    at_cutoff = (pd.to_datetime(as_of) == cutoff).to_numpy()
    groups = [(cutoff, at_cutoff)]
    for day in sorted(pd.unique(pd.to_datetime(as_of)[~at_cutoff].dropna())):
        groups.append((_as_date(day), (pd.to_datetime(as_of) == day).to_numpy()))

    for day, mask in groups:
        if not mask.any():
            continue
        if day == cutoff:
            s, r = season, recent
            g_played, g_recent = played, recent_played
        else:
            ids = set(staff.loc[mask, "pitcher"])
            log = inputs.appearances[inputs.appearances["pitcher"].isin(ids)]
            s = window_totals(log, day, unit, None)
            r = window_totals(log, day, unit, RECENT_DAYS)
            g_played = club_games(inputs.team_games, pd.Timestamp("1900-01-01"), day)
            g_recent = club_games(inputs.team_games,
                                  day - pd.Timedelta(days=RECENT_DAYS), day)
        rows = staff.loc[mask]
        idx = np.flatnonzero(mask)

        def take(table, column):
            if not len(table):
                return np.zeros(len(rows))
            return (rows["pitcher"].map(table.set_index("pitcher")[column])
                    .astype(float).fillna(0.0).to_numpy())

        apps_s, apps_r = take(s, "appearances"), take(r, "appearances")
        unit_s = take(s, unit)
        gp = rows["team_id"].map(g_played).astype(float).fillna(0.0).to_numpy()
        gr = rows["team_id"].map(g_recent).astype(float).fillna(0.0).to_numpy()
        pr = prior_rate.to_numpy()[idx]
        pu = prior_unit.to_numpy()[idx]
        rate_season[idx] = (apps_s + rate_ballast * pr) / (gp + rate_ballast)
        rate_recent[idx] = (apps_r + rate_ballast * pr) / (gr + rate_ballast)
        per_app[idx] = (unit_s + unit_ballast * pu) / (apps_s + unit_ballast)

    return ({"season": pd.Series(rate_season, index=staff.index),
             "recent": pd.Series(rate_recent, index=staff.index)},
            pd.Series(per_app, index=staff.index))


def _normalize_to_club(projected: pd.Series, team_id: pd.Series,
                       left: pd.Series, inputs: CutoffInputs,
                       unit: str) -> pd.Series:
    """Scale each club's staff to the club's own projected total.

    Station B's normalization, on the other side of the ball: a club's
    hitters must, between them, take the club's plate appearances, and a
    club's pitchers must, between them, face the opposing club's. The club
    total comes from its own season to date — workload per club game, over the
    games it has left — so a club that has been playing long games keeps them.
    """
    per_game = _club_unit_per_game(inputs, unit)
    target = team_id.map(per_game).astype(float) * left
    total = projected.groupby(team_id).transform("sum")
    scale = (target / total.where(total > 0)).fillna(1.0)
    return projected * scale


def _club_unit_per_game(inputs: CutoffInputs, unit: str) -> pd.Series:
    """Workload the club's staff got through per club game, before the cutoff.

    Summed over every pitcher who appeared *for that club*, from the
    appearance log's own `team` column, so a reliever traded in July counts
    toward whichever pen he was actually in.
    """
    log = inputs.appearances
    if log is None or not len(log) or "team" not in log.columns:
        return pd.Series(dtype=float)
    dates = _dates(log)
    past = log[dates < inputs.cutoff]
    if not len(past):
        return pd.Series(dtype=float)
    totals = past.groupby("team")[unit].sum().astype(float)
    played = club_games(inputs.team_games, pd.Timestamp("1900-01-01"), inputs.cutoff)
    per_game = totals / played.reindex(totals.index)
    per_game.index.name = "team_id"
    return per_game.dropna()


# --- scoring -----------------------------------------------------------

def _aligned(projection: pd.DataFrame, actual: pd.DataFrame,
             universe=None) -> pd.DataFrame:
    """pitcher, role, projected, realized on a common pitcher set."""
    proj = (projection.groupby("pitcher", as_index=False)
            .agg(projected=("projected", "sum"), role=("role", "first")))
    real = actual.groupby("pitcher", as_index=False)["realized"].sum()
    df = proj.merge(real, on="pitcher", how="outer")
    if universe is not None:
        keep = pd.Index(pd.unique(pd.Series(list(universe))), name="pitcher")
        df = df.set_index("pitcher").reindex(keep).reset_index()
    df["projected"] = df["projected"].fillna(0.0)
    df["realized"] = df["realized"].fillna(0.0)
    return df


def absolute_errors(projection: pd.DataFrame, actual: pd.DataFrame,
                    universe=None) -> pd.Series:
    """|projected - realized| per pitcher, indexed by `pitcher`.

    Two methods scored on the same universe give two of these on the same
    index, so their difference is paired.
    """
    df = _aligned(projection, actual, universe=universe)
    err = (df["projected"] - df["realized"]).abs()
    return pd.Series(err.to_numpy(float), index=pd.Index(df["pitcher"], name="pitcher"))


def paired_difference(errors_a: pd.Series, errors_b: pd.Series) -> dict:
    """Mean and standard error of `errors_a - errors_b`, pitcher by pitcher.

    Negative `mean` means A is the better method; `t` is `mean / se`.
    """
    a, b = errors_a.align(errors_b, join="inner")
    d = (a - b).to_numpy(float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean = float(d.mean()) if n else float("nan")
    return {"n": n, "mean": mean, "se": se,
            "t": mean / se if se else float("nan")}


def score_projection(projection: pd.DataFrame, actual: pd.DataFrame,
                     universe=None, roles: pd.Series | None = None) -> dict:
    """MAE / RMSE of projected vs realized rest-of-season workload.

    Weighted variants weight by realized workload, the `src/eval/metrics`
    convention that a rotation starter's miss counts for more than a
    September call-up's. `top5_capture` is the share of a club's realized
    workload taken by the five pitchers each method projected highest — "did
    you pick the rotation?" separated from "did you get the counts right?".

    `roles` (pitcher -> "SP"/"RP") splits the same metrics by role when given.
    Starters and relievers are different workload processes and a method can
    win one and lose the other.
    """
    df = _aligned(projection, actual, universe=universe)
    if roles is not None:
        df["role"] = df["pitcher"].map(roles).fillna(df["role"]).fillna("RP")
    out = _metrics(df)
    if roles is not None:
        for role in ("SP", "RP"):
            sub = df[df["role"] == role]
            for key, value in _metrics(sub).items():
                out[f"{role.lower()}_{key}"] = value
    return out


def _metrics(df: pd.DataFrame) -> dict:
    err = df["projected"].to_numpy(float) - df["realized"].to_numpy(float)
    w = df["realized"].to_numpy(float)
    w_sum = float(w.sum())
    return {
        "n": int(len(df)),
        "mae": float(np.abs(err).mean()) if len(df) else float("nan"),
        "rmse": float(np.sqrt((err ** 2).mean())) if len(df) else float("nan"),
        "weighted_mae": float((w * np.abs(err)).sum() / w_sum) if w_sum else float("nan"),
        "bias": float(err.mean()) if len(df) else float("nan"),
    }


def top_n_capture(projection: pd.DataFrame, actual: pd.DataFrame, n: int = 5,
                  universe=None) -> float:
    """Share of realized club workload taken by the n pitchers ranked highest."""
    df = _aligned(projection, actual, universe=universe)
    df = df.merge(projection.loc[:, ["pitcher", "team_id"]].drop_duplicates(),
                  on="pitcher", how="left").dropna(subset=["team_id"])
    total = float(df["realized"].sum())
    if not len(df) or total <= 0:
        return float("nan")
    picked = (df.sort_values("projected", ascending=False)
              .groupby("team_id").head(n))
    return float(picked["realized"].sum() / total)
