"""Station B — projected rest-of-season plate appearances per hitter.

Everything above station A in the rollup multiplies a *rate* by a *playing
time* number: team run environment (C) needs PA-weighted wOBA, WAR needs
playing time to turn a rate edge into runs, contract valuation needs it to
turn runs into dollars. Today there is no playing-time number at all;
`assemble_and_compare.py` hardcodes `pa = 550` for everyone. This module is
the simplest honest replacement.

The model, in one line:

    projected_pa(hitter) = team_pa_per_game x games_remaining x pa_share

`team_pa_per_game` is the team's season-to-date plate appearances per game
(a very stable ~37-39); `games_remaining` comes from the schedule; the only
modelled quantity is `pa_share`, the hitter's slice of his team's PA, taken
from the trailing 30 days.

Everything here is a pure function over DataFrames — roster frame, game-log
frame, games-remaining frame in, projection frame out — so it unit-tests
without a network. The fetch/assemble layer is `scripts/build_playing_time.py`
on top of `src/data/mlb_stats_api.py`.

Walk-forward honesty: every function that looks at game logs takes a
`cutoff` and uses only rows with `date < cutoff`, strictly. A game played
*on* the cutoff date is future information (it happens after the morning the
projection is made) and is excluded.

Baselines (architecture.md section 3, the gate rule — the model does not get
to be the only option):

    uniform       equal share across the active-roster hitters
    season_share  season-to-date PA share, no window and no IL handling
    last_30       the model: trailing-30-day share, IL zeroed, bench default

`scripts/build_playing_time.py --score` scores all three walk-forward at two
cutoffs against realized PA; see docs/playing-time.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Status codes the Stats API puts on a 40-man roster entry. Anything starting
# with "D" is a day-count injured list (D7/D10/D15/D60); the rest of these are
# the other ways a player on the 40-man is unavailable to bat today.
# Everything else on a 40-man is unavailable and projects to zero the same
# way: RM optioned to the minors, PL paternity, BRV bereavement, RL
# restricted, SU suspended, FA free agent.
ACTIVE_STATUS = "A"
IL_STATUS_PREFIX = "D"

# Trailing windows, in days, tried in order. A hitter with no PA in the last
# 30 days falls back to his last-60 rate, then to his season rate; each is
# rescaled to a 30-day-equivalent so the numbers stay commensurate before the
# per-team normalization.
PRIMARY_WINDOW_DAYS = 30
FALLBACK_WINDOW_DAYS = 60

# A hitter on the active roster with no plate appearances at all in the
# trailing windows or the season to date is a call-up or a fresh acquisition.
# He is not a zero — he is a bench bat. A 26-man roster carries ~13 hitters,
# nine of whom start; a fringe hitter who plays maybe 40% of games at ~4 PA is
# ~3% of his team's plate appearances. That is the default, expressed as a
# share of the team's window total so it scales with the window.
DEFAULT_BENCH_SHARE = 0.03

# No hitter can take more than one lineup slot's worth of his team's plate
# appearances. Nine slots, and the top of the order absorbs the extra PA of an
# incomplete final turn, so the leadoff hitter of a team averaging 37.5 PA a
# game gets about 4.7 of them — one eighth. That is lineup arithmetic, not a
# fitted constant, and 2026 agrees: over every 30-day window at every cutoff,
# the largest share any hitter actually took was 0.123-0.125, never more.
#
# The cap binds because the model renormalizes over the *active* roster: a
# club that just lost three regulars to the IL has its survivors' shares
# scaled up, and without the cap the leftover is dumped on the best remaining
# hitter (a projected 0.171 share, 6.5 PA a game, on the 2026-09-02 Cardinals).
# The excess is water-filled onto the hitters still below the cap instead.
MAX_PA_SHARE = 1.0 / 8.0
MAX_SHARE_ITERATIONS = 12

METHODS = ("uniform", "season_share", "last_30")

PROJECTION_COLUMNS = [
    "batter", "team_id", "cutoff_date", "games_remaining",
    "pa_share", "projected_pa_ros",
]


def _as_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _dates(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["date"]).dt.normalize()


def is_active(status_code) -> bool:
    """True when a 40-man roster status means the player can bat today."""
    return str(status_code).upper() == ACTIVE_STATUS


def is_injured(status_code) -> bool:
    """True for any day-count injured-list status (D7/D10/D15/D60)."""
    code = str(status_code).upper()
    return code.startswith(IL_STATUS_PREFIX) and code[1:].isdigit()


# --- the pieces ---

def team_pa_per_game(team_logs: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Season-to-date plate appearances per game, per team, before `cutoff`.

    `team_logs`: team_id, date, pa — one row per team-game (the Stats API's
    team hitting gameLog). Returns team_id, team_pa, team_games, pa_per_game.
    """
    cutoff = _as_date(cutoff)
    cols = ["team_id", "team_pa", "team_games", "pa_per_game"]
    if team_logs.empty:
        return pd.DataFrame(columns=cols)
    past = team_logs[_dates(team_logs) < cutoff]
    if not len(past):
        return pd.DataFrame(columns=cols)
    agg = (past.groupby("team_id", as_index=False)
           .agg(team_pa=("pa", "sum"), team_games=("pa", "size")))
    agg["pa_per_game"] = agg["team_pa"] / agg["team_games"].where(agg["team_games"] > 0)
    return agg


def window_pa(game_logs: pd.DataFrame, cutoff, window_days: int | None = None) -> pd.DataFrame:
    """Per-hitter plate appearances in [cutoff - window_days, cutoff).

    `game_logs`: batter, date, pa (team_id optional and ignored here — a
    traded hitter's PA still count toward the share he earns on the roster he
    is on at the cutoff). `window_days=None` means the whole season to date.

    The upper bound is strict: a game played *on* the cutoff date has not
    happened yet from the projection's point of view.
    """
    cutoff = _as_date(cutoff)
    cols = ["batter", "pa"]
    if game_logs.empty:
        return pd.DataFrame(columns=cols)
    dates = _dates(game_logs)
    mask = dates < cutoff
    if window_days is not None:
        mask &= dates >= cutoff - pd.Timedelta(days=window_days)
    kept = game_logs[mask]
    if not len(kept):
        return pd.DataFrame(columns=cols)
    return kept.groupby("batter", as_index=False)["pa"].sum()


def _mapped(batters: pd.Series, table: pd.DataFrame) -> pd.Series:
    """`batter -> pa` lookup that survives an empty table."""
    if table.empty:
        return pd.Series(0.0, index=batters.index)
    lookup = table.set_index("batter")["pa"]
    return batters.map(lookup).fillna(0.0).astype(float)


def _weights_last_30(roster: pd.DataFrame, game_logs: pd.DataFrame,
                     cutoff) -> pd.Series:
    """The model's raw (un-normalized) weight per roster row.

    Trailing 30 days, falling back to trailing 60 and then the season to date,
    each rescaled to a 30-day equivalent, then to a league bench default.
    """
    cutoff = _as_date(cutoff)
    season_start = _dates(game_logs).min() if len(game_logs) else cutoff
    season_days = max(float((cutoff - season_start).days), 1.0)

    batters = roster["batter"]
    p30 = _mapped(batters, window_pa(game_logs, cutoff, PRIMARY_WINDOW_DAYS))
    # Rescale the fallbacks onto the primary window's time base so a hitter
    # who played only in the first half is not handed a 60-day count that
    # would outweigh a regular's 30-day count.
    p60 = _mapped(batters, window_pa(game_logs, cutoff, FALLBACK_WINDOW_DAYS))
    p60 = p60 * (PRIMARY_WINDOW_DAYS / FALLBACK_WINDOW_DAYS)
    psn = _mapped(batters, window_pa(game_logs, cutoff, None))
    psn = psn * (PRIMARY_WINDOW_DAYS / season_days)

    weight = p30.where(p30 > 0, p60)
    weight = weight.where(weight > 0, psn)

    # League bench default for anyone still at zero: a share of what the
    # team's *known* hitters produced in the window (league mean if this team
    # has none at all, e.g. opening day).
    team_window = p30.groupby(roster["team_id"].to_numpy()).transform("sum")
    n_teams = max(int(roster["team_id"].nunique()), 1)
    league_mean = float(p30.sum()) / n_teams
    fill = DEFAULT_BENCH_SHARE * team_window.where(team_window > 0, league_mean)
    return weight.where(weight > 0, fill).astype(float)


def cap_shares(shares: pd.Series, team_id: pd.Series,
               cap: float = MAX_PA_SHARE) -> pd.Series:
    """Clip every share to `cap` and water-fill the excess onto the rest.

    Shares that summed to 1 within a team still sum to 1 within a team. A club
    with fewer than `1/cap` eligible hitters cannot satisfy the cap at all —
    its plate appearances still have to be taken by someone — so there the
    effective cap relaxes to an even split.
    """
    weight = shares.astype(float).clip(lower=0.0)
    eligible = (weight > 0).groupby(team_id).transform("sum")
    effective = np.maximum(cap, 1.0 / eligible.where(eligible > 0, 1.0))

    # Standard water-filling: grow the set pinned at the cap one pass at a
    # time and rescale everyone else onto the remaining budget. At most
    # 1/cap players per team can be pinned, so this terminates exactly rather
    # than converging.
    capped = pd.Series(False, index=weight.index)
    out = weight
    for _ in range(MAX_SHARE_ITERATIONS):
        budget = 1.0 - effective.where(capped, 0.0).groupby(team_id).transform("sum")
        free = weight.where(~capped, 0.0).groupby(team_id).transform("sum")
        scaled = weight * budget / free.where(free > 0, 1.0)
        out = scaled.where(~capped, effective)
        newly = (~capped) & (weight > 0) & (out > effective + 1e-12)
        if not newly.any():
            break
        capped = capped | newly
    return out.where(weight > 0, 0.0)


def project_playing_time(
    roster: pd.DataFrame,
    game_logs: pd.DataFrame,
    games_remaining: pd.DataFrame,
    cutoff,
    team_logs: pd.DataFrame | None = None,
    method: str = "last_30",
    pa_per_game: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Project rest-of-season plate appearances for every hitter on `roster`.

    Parameters
    ----------
    roster
        batter, team_id, status_code — one row per hitter on the 40-man as of
        `cutoff`. Pitchers should already be filtered out.
    game_logs
        batter, date, pa — per-game hitting lines for the season. Only rows
        strictly before `cutoff` are ever read.
    games_remaining
        team_id, games_remaining.
    team_logs
        team_id, date, pa — team-level per-game lines, used for the team's
        PA/game. Pass `pa_per_game` (team_id, pa_per_game) instead if you
        already have it.
    method
        "last_30" (the model), "season_share" or "uniform" (the baselines).

    Returns one row per input roster row with `pa_share` (sums to 1 within a
    team, or to 0 for a team with nobody eligible) and `projected_pa_ros`.
    Injured and otherwise-unavailable players get exactly zero.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    cutoff = _as_date(cutoff)

    out = roster.loc[:, ["batter", "team_id"]].copy().reset_index(drop=True)
    roster = roster.reset_index(drop=True)
    out["cutoff_date"] = cutoff.date().isoformat()
    status = (roster["status_code"] if "status_code" in roster.columns
              else pd.Series(ACTIVE_STATUS, index=roster.index))
    active = status.map(is_active)

    if method == "uniform":
        weight = active.astype(float)
    elif method == "season_share":
        # Deliberately dumb: season-to-date share with no roster filter at
        # all, so a hitter who tore his ACL in May still gets projected PA.
        # That is the point of the baseline — it isolates what the 30-day
        # window and the IL zeroing are worth.
        weight = _mapped(out["batter"], window_pa(game_logs, cutoff, None))
    else:
        weight = _weights_last_30(roster, game_logs, cutoff).where(active, 0.0)

    weight = pd.Series(np.asarray(weight, dtype=float), index=out.index).clip(lower=0.0)
    totals = weight.groupby(out["team_id"]).transform("sum")
    share = pd.Series(np.where(totals > 0, weight / totals.where(totals > 0, 1.0), 0.0),
                      index=out.index)
    if method == "last_30":
        # The baselines stay deliberately dumb; the lineup-slot ceiling is
        # part of the model.
        share = cap_shares(share, out["team_id"])
    out["pa_share"] = share

    if pa_per_game is None:
        if team_logs is None:
            raise ValueError("pass either team_logs or pa_per_game")
        pa_per_game = team_pa_per_game(team_logs, cutoff)
    ppg = (pa_per_game.set_index("team_id")["pa_per_game"]
           if len(pa_per_game) else pd.Series(dtype=float))
    remaining = games_remaining.set_index("team_id")["games_remaining"]

    out["games_remaining"] = out["team_id"].map(remaining).astype(float).fillna(0.0)
    team_pa_ros = out["games_remaining"] * out["team_id"].map(ppg).astype(float).fillna(0.0)
    out["projected_pa_ros"] = out["pa_share"] * team_pa_ros
    return out.loc[:, PROJECTION_COLUMNS]


# --- scoring ---

def realized_pa(game_logs: pd.DataFrame, start, end) -> pd.DataFrame:
    """Plate appearances actually taken in [start, end], inclusive of both.

    `start` is the projection's cutoff (the first day it is responsible for)
    and `end` the last day scored. Returns batter, realized_pa.
    """
    start, end = _as_date(start), _as_date(end)
    cols = ["batter", "realized_pa"]
    if game_logs.empty:
        return pd.DataFrame(columns=cols)
    dates = _dates(game_logs)
    kept = game_logs[(dates >= start) & (dates <= end)]
    if not len(kept):
        return pd.DataFrame(columns=cols)
    return (kept.groupby("batter", as_index=False)["pa"].sum()
            .rename(columns={"pa": "realized_pa"}))


def score_projection(projection: pd.DataFrame, realized: pd.DataFrame,
                     universe=None) -> dict:
    """MAE / RMSE of projected vs realized rest-of-season PA.

    The comparison set is `universe` (an iterable of batter ids) when given,
    otherwise the union of the projected and the realized hitters — a
    projected hitter who never played counts as realized 0, and a September
    call-up nobody projected counts as projected 0. Pass the same `universe`
    to every method so the methods are scored on identical players.

    Weighted metrics use realized PA as the weight, matching the
    `src/eval/metrics` convention that a regular's miss counts more than a
    bench bat's.

    `top9_capture` is the share of realized team PA taken by the nine hitters
    each method projected highest for that team, summed league-wide — the
    question "did you pick the right lineup?" separated from "did you get the
    counts right?".
    """
    proj = (projection.groupby("batter", as_index=False)
            .agg(projected_pa_ros=("projected_pa_ros", "sum"),
                 team_id=("team_id", "first")))
    real = realized.groupby("batter", as_index=False)["realized_pa"].sum()
    df = proj.merge(real, on="batter", how="outer")
    if universe is not None:
        keep = pd.Index(pd.unique(pd.Series(list(universe))), name="batter")
        df = df.set_index("batter").reindex(keep).reset_index()
    df["projected_pa_ros"] = df["projected_pa_ros"].fillna(0.0)
    df["realized_pa"] = df["realized_pa"].fillna(0.0)

    err = df["projected_pa_ros"].to_numpy(float) - df["realized_pa"].to_numpy(float)
    w = df["realized_pa"].to_numpy(float)
    w_sum = float(w.sum())

    top9 = float("nan")
    ranked = df.dropna(subset=["team_id"])
    if len(ranked) and ranked["realized_pa"].sum() > 0:
        ranked = ranked.sort_values("projected_pa_ros", ascending=False)
        picked = ranked.groupby("team_id").head(9)
        team_total = float(ranked["realized_pa"].sum())
        top9 = float(picked["realized_pa"].sum() / team_total) if team_total else float("nan")

    return {
        "n_hitters": int(len(df)),
        "mae": float(np.abs(err).mean()) if len(df) else float("nan"),
        "rmse": float(np.sqrt((err ** 2).mean())) if len(df) else float("nan"),
        "weighted_mae": float((w * np.abs(err)).sum() / w_sum) if w_sum else float("nan"),
        "weighted_rmse": float(np.sqrt((w * err ** 2).sum() / w_sum)) if w_sum else float("nan"),
        "top9_capture": top9,
    }


def walk_forward_scores(
    roster_by_cutoff: dict,
    game_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    games_remaining_by_cutoff: dict,
    score_end,
    methods=METHODS,
) -> pd.DataFrame:
    """Score every method at every cutoff against realized PA through `score_end`.

    `roster_by_cutoff` and `games_remaining_by_cutoff` are keyed by cutoff
    date string. At each cutoff the projection sees only game-log rows before
    the cutoff (enforced inside `project_playing_time`); it is scored on PA
    from the cutoff through `score_end` inclusive.

    Returns one row per (cutoff, method) with the metrics from
    `score_projection`, on a hitter universe shared by all methods at a cutoff.
    """
    rows = []
    for cutoff in sorted(roster_by_cutoff):
        roster = roster_by_cutoff[cutoff]
        remaining = games_remaining_by_cutoff[cutoff]
        real = realized_pa(game_logs, cutoff, score_end)
        ppg = team_pa_per_game(team_logs, cutoff)
        projections = {
            m: project_playing_time(roster, game_logs, remaining, cutoff,
                                    pa_per_game=ppg, method=m)
            for m in methods
        }
        # One universe for all methods so the comparison is on a common set.
        universe = sorted(
            set(real["batter"]) |
            {b for p in projections.values()
             for b in p.loc[p["projected_pa_ros"] > 0, "batter"]}
        )
        for m, proj in projections.items():
            rows.append({"cutoff": str(cutoff), "method": m,
                         **score_projection(proj, real, universe=universe)})
    return pd.DataFrame(rows)
