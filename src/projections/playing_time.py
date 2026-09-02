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
from a horizon-weighted blend of his trailing-30-day and season-to-date
shares — the recent window is the better predictor over a few weeks and the
worse one over a few months, so the weight decays with the horizon.

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
    last_30       trailing-30-day share, IL zeroed, bench default, capped
    blend         w(h) x last_30 + (1 - w(h)) x season, IL zeroed, capped
    blend_il      the same, with the injured and optioned projected at their
                  pre-injury share times their expected return fraction

`last_30` was the model until the blend replaced it, and `blend` until the
return fractions replaced the hard roster gate; each stays in the table as the
baseline the next one has to beat.

`scripts/build_playing_time.py --score` scores all five walk-forward at two
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

# --- the horizon blend ---
#
# The trailing-30-day share is the sharper answer to "who is playing right
# now" and the noisier answer to "who will be playing in six weeks"; the
# season-to-date share is the reverse. Which one is better is therefore a
# function of the horizon, not a fact about the method, so the model blends
# them with a weight that decays as the horizon grows:
#
#     share(h) = w(h) x share_30 + (1 - w(h)) x share_season
#     w(h)     = 1 / (1 + exp((h - midpoint) / scale))
#
# `h` is the club's games remaining in the horizon being projected. A logistic
# is the smallest form that is monotone, bounded to [0, 1] and has a knee.
#
# Its two parameters are given as the weight at two anchor horizons rather
# than as a midpoint and a scale, because the midpoint and scale the fit
# actually lands on are far outside the range of horizons anybody projects
# (see below) and are meaningless read on their own, whereas "82% weight on
# the recent window one month out, 76% three months out" is the finding.
# Midpoint and scale are derived from the anchors and exported for the few
# places that want the raw logistic.
#
# Both numbers were chosen walk-forward on **2025** cutoffs only and frozen
# before the 2026 table was scored (`--sweep --season 2025`). The honest
# summary of that fit: 2025 wants a weight near 0.8 at *every* horizon from
# two weeks to three and a half months, and the selection surface is nearly
# flat, so the horizon slope the logistic carries is real but small. See
# docs/playing-time.md section 3.
BLEND_ANCHOR_GAMES = (30.0, 90.0)
BLEND_WEIGHT_SHORT = 0.83   # w at 30 games remaining
BLEND_WEIGHT_LONG = 0.75    # w at 90 games remaining


def logistic_from_anchors(w_short: float, w_long: float,
                           anchors=BLEND_ANCHOR_GAMES) -> tuple[float, float]:
    """(midpoint, scale) of the logistic through two (horizon, weight) points."""
    h_short, h_long = anchors
    if not 0.0 < w_long < w_short < 1.0:
        raise ValueError("need 0 < w_long < w_short < 1 for a decreasing logistic")
    l_short = float(np.log(w_short / (1.0 - w_short)))
    l_long = float(np.log(w_long / (1.0 - w_long)))
    scale = (h_long - h_short) / (l_short - l_long)
    return h_short + scale * l_short, scale


BLEND_MIDPOINT_GAMES, BLEND_SCALE_GAMES = logistic_from_anchors(
    BLEND_WEIGHT_SHORT, BLEND_WEIGHT_LONG)

# --- the roster gate, and the option to soften it ---
#
# `blend` zeroes every hitter the 40-man roster says is unavailable at the
# cutoff. `blend_il` replaces that zero with an *expected* share: the share he
# would have taken healthy (computed from the data before he was placed, since
# after it his trailing window is empty by construction) times the fraction of
# the remaining horizon he is expected to be back for, from the injured-list
# return-time distribution in `src/projections/il_returns.py`. Everything else
# — the horizon blend, the per-club normalization, the lineup-slot cap — is
# untouched, and passing no fractions at all makes `blend_il` identical to
# `blend` row for row.
#
# Which of the two production runs is this flag, and it is set by the gate in
# docs/playing-time.md, not by preference.
USE_IL_RETURNS = True
PRODUCTION_METHOD = "blend_il" if USE_IL_RETURNS else "blend"

METHODS = ("uniform", "season_share", "last_30", "blend", "blend_il")

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


def window_pa_by_team(game_logs: pd.DataFrame, cutoff,
                      window_days: int | None = PRIMARY_WINDOW_DAYS) -> pd.DataFrame:
    """`window_pa`, but split by the club the plate appearances were taken for.

    `game_logs`: batter, team_id, date, pa. Same strict `date < cutoff` guard,
    same window semantics (`window_days=None` for the season to date); the only
    difference is the grouping key.

    `window_pa` deliberately ignores `team_id` because station B starts from a
    roster frame that already says who is on which club, so a traded hitter's
    old plate appearances should count toward the share he earns on his new
    team. Station C (`src.sim.run_environment`) has no roster frame — it infers
    membership *from* the appearances — so it needs the split, and a traded
    hitter contributes to whichever pen, lineup and share he actually batted
    for on the day. Returns `team_id, batter, pa`.
    """
    cutoff = _as_date(cutoff)
    cols = ["team_id", "batter", "pa"]
    if game_logs.empty or "team_id" not in game_logs.columns:
        return pd.DataFrame(columns=cols)
    dates = _dates(game_logs)
    mask = dates < cutoff
    if window_days is not None:
        mask &= dates >= cutoff - pd.Timedelta(days=window_days)
    kept = game_logs[mask].dropna(subset=["team_id"])
    if not len(kept):
        return pd.DataFrame(columns=cols)
    out = kept.groupby(["team_id", "batter"], as_index=False)["pa"].sum()
    out["team_id"] = out["team_id"].astype("int64")
    return out.loc[:, cols]


def _mapped(batters: pd.Series, table: pd.DataFrame) -> pd.Series:
    """`batter -> pa` lookup that survives an empty table."""
    if table.empty:
        return pd.Series(0.0, index=batters.index)
    lookup = table.set_index("batter")["pa"]
    return batters.map(lookup).fillna(0.0).astype(float)


def _bench_default(base: pd.Series, team_id: pd.Series) -> pd.Series:
    """League bench default, on the time base of whatever window `base` is.

    A share of what the team's *known* hitters produced in that window (the
    league mean if this club has none at all, e.g. opening day), so the
    default scales with the window rather than being a raw PA count.
    """
    team_window = base.groupby(team_id.to_numpy()).transform("sum")
    n_teams = max(int(team_id.nunique()), 1)
    league_mean = float(base.sum()) / n_teams
    return DEFAULT_BENCH_SHARE * team_window.where(team_window > 0, league_mean)


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
    fill = _bench_default(p30, roster["team_id"])
    return weight.where(weight > 0, fill).astype(float)


def _weights_season(roster: pd.DataFrame, game_logs: pd.DataFrame,
                    cutoff) -> pd.Series:
    """Season-to-date PA per roster row, with the same bench default.

    The long-horizon half of the blend. It is *not* the `season_share`
    baseline: that one is deliberately dumb (no roster filter, no default, no
    cap) so it can isolate what the window and the IL zeroing are worth. This
    is the same quantity carried through the model's own plumbing, so the only
    thing that differs between the two halves of the blend is the window.
    """
    psn = _mapped(roster["batter"], window_pa(game_logs, cutoff, None))
    fill = _bench_default(psn, roster["team_id"])
    return psn.where(psn > 0, fill).astype(float)


def _season_days(game_logs: pd.DataFrame, through) -> float:
    """Days of season played before `through` — the time base the weights use."""
    through = _as_date(through)
    season_start = _dates(game_logs).min() if len(game_logs) else through
    return max(float((through - season_start).days), 1.0)


def preinjury_weights(roster: pd.DataFrame, game_logs: pd.DataFrame, cutoff,
                      spell_start: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Both halves' raw weights, read as of the day each hitter went out.

    A hitter placed on the injured list three weeks ago has no plate
    appearances in the trailing 30 days *because* he is injured, so weighing
    him at the cutoff would say he is a zero-share bench bat rather than the
    regular he was. These are the same two weights as `_weights_last_30` and
    `_weights_season`, computed at his own `spell_start` instead — the share
    he would be taking if he were healthy.

    `spell_start` is a per-row date (NaT for a hitter who is not out, and for
    one whose spell the transactions cannot date). Both series are zero
    wherever `spell_start` is NaT or the hitter had no plate appearances
    before it, which tells the caller to keep the cutoff-dated weight.

    Both are put back on the cutoff's time base so they stay commensurate with
    the healthy hitters they are normalized against: the 30-day window is the
    same 30 days long wherever it sits, and the season-to-date count is scaled
    up by the ratio of season lengths, which is what "at the rate he was going"
    means.
    """
    cutoff = _as_date(cutoff)
    w30 = pd.Series(0.0, index=roster.index)
    wsn = pd.Series(0.0, index=roster.index)
    starts = pd.to_datetime(spell_start)
    season_days_at_cutoff = _season_days(game_logs, cutoff)
    for start in sorted(pd.unique(starts.dropna())):
        start = _as_date(start)
        if start >= cutoff:
            continue
        rows = roster.index[starts == start]
        batters = roster.loc[rows, "batter"]
        # Only this handful of hitters' logs matter, so the repeated windowing
        # stays cheap however many distinct placement dates there are.
        logs = game_logs[game_logs["batter"].isin(set(batters))]
        p30 = _mapped(batters, window_pa(logs, start, PRIMARY_WINDOW_DAYS))
        p60 = _mapped(batters, window_pa(logs, start, FALLBACK_WINDOW_DAYS)) * (
            PRIMARY_WINDOW_DAYS / FALLBACK_WINDOW_DAYS)
        season = _mapped(batters, window_pa(logs, start, None))
        season_days = _season_days(game_logs, start)
        short = p30.where(p30 > 0, p60)
        short = short.where(short > 0, season * (PRIMARY_WINDOW_DAYS / season_days))
        w30.loc[rows] = short.to_numpy(float)
        wsn.loc[rows] = (season * (season_days_at_cutoff / season_days)).to_numpy(float)
    return w30, wsn


def horizon_weight(games_remaining,
                   midpoint: float = BLEND_MIDPOINT_GAMES,
                   scale: float = BLEND_SCALE_GAMES):
    """Weight on the trailing-30-day share at a horizon of `games_remaining`.

    A logistic, monotonically *decreasing* in the horizon: short horizons
    trust the recent window, long ones lean further on the season, and
    `midpoint` is the horizon where the two would be trusted equally. At the
    fitted parameters that midpoint is ~280 games — far past any horizon a
    projection is ever asked for — so over the range that matters, a fortnight
    to a full season, this is a shallow slide from about 0.84 down to about
    0.74 rather than a switch between two windows. Returns the input's shape.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    h = np.asarray(games_remaining, dtype=float)
    w = 1.0 / (1.0 + np.exp(np.clip((h - midpoint) / scale, -60.0, 60.0)))
    return w if w.ndim else float(w)


def _normalize(weight, team_id: pd.Series) -> pd.Series:
    """Non-negative weights -> shares summing to 1 within each club (or to 0)."""
    w = pd.Series(np.asarray(weight, dtype=float), index=team_id.index).clip(lower=0.0)
    totals = w.groupby(team_id).transform("sum")
    return pd.Series(np.where(totals > 0, w / totals.where(totals > 0, 1.0), 0.0),
                     index=w.index)


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
    blend_weight: float | None = None,
    active_fractions: pd.DataFrame | None = None,
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
        "blend_il" or "blend" (the model, with and without expected returns
        from the injured list), "last_30" (the model's short-horizon half),
        "season_share" or "uniform" (the baselines).
    active_fractions
        For `method="blend_il"`: `batter, elapsed_days, active_fraction` for
        the hitters who are unavailable at the cutoff, from
        `il_returns.expected_active_fractions`. Each such hitter is weighed as
        he was the day he went out (`elapsed_days` before the cutoff) and then
        scaled by `active_fraction`, the share of the remaining horizon he is
        expected to be back for. Any unavailable hitter not in the frame keeps
        the hard zero, so passing None makes `blend_il` identical to `blend`.
    blend_weight
        Override for `w(h)` under `method="blend"`: a constant in [0, 1] used
        for every club instead of the horizon logistic. `1.0` reproduces
        `last_30` exactly and `0.0` gives the season share carried through the
        same plumbing. For tests and for sweeping the parameter; leave it
        `None` in production so the weight follows the horizon.

    Returns one row per input roster row with `pa_share` (sums to 1 within a
    team, or to 0 for a team with nobody eligible) and `projected_pa_ros`.
    Injured and otherwise-unavailable players get exactly zero under every
    method but `blend_il`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    cutoff = _as_date(cutoff)

    out = roster.loc[:, ["batter", "team_id"]].copy().reset_index(drop=True)
    roster = roster.reset_index(drop=True)
    out["cutoff_date"] = cutoff.date().isoformat()
    remaining = games_remaining.set_index("team_id")["games_remaining"]
    # The horizon is needed before the shares, because the blend weight is a
    # function of it.
    out["games_remaining"] = out["team_id"].map(remaining).astype(float).fillna(0.0)
    status = (roster["status_code"] if "status_code" in roster.columns
              else pd.Series(ACTIVE_STATUS, index=roster.index))
    active = status.map(is_active)
    team_id = out["team_id"]

    if method == "uniform":
        share = _normalize(active.astype(float), team_id)
    elif method == "season_share":
        # Deliberately dumb: season-to-date share with no roster filter at
        # all, so a hitter who tore his ACL in May still gets projected PA.
        # That is the point of the baseline — it isolates what the 30-day
        # window and the IL zeroing are worth.
        share = _normalize(_mapped(out["batter"], window_pa(game_logs, cutoff, None)),
                           team_id)
    elif method == "last_30":
        # The baselines stay deliberately dumb; the lineup-slot ceiling is
        # part of the model.
        share = _normalize(_weights_last_30(roster, game_logs, cutoff).where(active, 0.0),
                           team_id)
        share = cap_shares(share, team_id)
    else:
        # The blend. Both halves are normalized to club shares *first*, so the
        # weight is a weight on shares rather than on two incommensurate PA
        # counts (a 30-day count and a season count).
        w30 = _weights_last_30(roster, game_logs, cutoff)
        wsn = _weights_season(roster, game_logs, cutoff)
        # The roster gate. `blend` is binary — one for an active hitter, zero
        # for everyone else. `blend_il` keeps the one, and replaces the zero
        # with the fraction of the horizon an unavailable hitter is expected
        # to be back for, weighing him as of the day he went out. Multiplying
        # the weights and normalizing after is the same thing as scaling his
        # club share, since the normalization divides the scale out.
        fraction = active.astype(float)
        if method == "blend_il" and active_fractions is not None and len(active_fractions):
            fractions = (active_fractions.set_index("batter")
                         .loc[:, ["elapsed_days", "active_fraction"]])
            f = out["batter"].map(fractions["active_fraction"]).astype(float)
            fraction = fraction.where(active, f.fillna(0.0))
            elapsed = out["batter"].map(fractions["elapsed_days"]).astype(float)
            spell_start = cutoff - pd.to_timedelta(elapsed.where(~active), unit="D")
            shift_30, shift_season = preinjury_weights(roster, game_logs, cutoff,
                                                       spell_start)
            w30 = w30.where(shift_30 <= 0, shift_30)
            wsn = wsn.where(shift_season <= 0, shift_season)
        share_30 = _normalize(w30 * fraction, team_id)
        share_season = _normalize(wsn * fraction, team_id)
        if blend_weight is None:
            w = pd.Series(horizon_weight(out["games_remaining"]), index=out.index)
        else:
            w = pd.Series(float(blend_weight), index=out.index)
        # Both halves already sum to 1 per club and `w` is constant within a
        # club, so the blend does too; the renormalization only matters for a
        # club where one half is empty.
        share = _normalize(w * share_30 + (1.0 - w) * share_season, team_id)
        share = cap_shares(share, team_id)
    out["pa_share"] = share

    if pa_per_game is None:
        if team_logs is None:
            raise ValueError("pass either team_logs or pa_per_game")
        pa_per_game = team_pa_per_game(team_logs, cutoff)
    ppg = (pa_per_game.set_index("team_id")["pa_per_game"]
           if len(pa_per_game) else pd.Series(dtype=float))

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


def _aligned(projection: pd.DataFrame, realized: pd.DataFrame,
             universe=None) -> pd.DataFrame:
    """batter, team_id, projected_pa_ros, realized_pa on a common hitter set.

    `universe` (an iterable of batter ids) when given, otherwise the union of
    the projected and the realized hitters — a projected hitter who never
    played counts as realized 0, a September call-up nobody projected counts
    as projected 0.
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
    return df


def absolute_errors(projection: pd.DataFrame, realized: pd.DataFrame,
                    universe=None) -> pd.Series:
    """|projected - realized| per hitter, indexed by `batter`.

    The per-hitter term whose mean is the MAE in `score_projection`. Two
    methods scored on the same universe give two of these on the same index,
    so their difference is *paired* — see `paired_difference`.
    """
    df = _aligned(projection, realized, universe=universe)
    err = (df["projected_pa_ros"] - df["realized_pa"]).abs()
    return pd.Series(err.to_numpy(float), index=pd.Index(df["batter"], name="batter"))


def paired_difference(errors_a: pd.Series, errors_b: pd.Series) -> dict:
    """Mean and standard error of `errors_a - errors_b`, hitter by hitter.

    The right uncertainty for "did method A beat method B": the two methods
    saw the same hitters and the same season, so almost all of the variance in
    either MAE is common and cancels. `mean` negative means A is the better
    method; `t` is `mean / se`.
    """
    a, b = errors_a.align(errors_b, join="inner")
    d = (a - b).to_numpy(float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean = float(d.mean()) if n else float("nan")
    return {"n": n, "mean": mean, "se": se,
            "t": mean / se if se else float("nan")}


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
    df = _aligned(projection, realized, universe=universe)
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
    active_fractions_by_cutoff: dict | None = None,
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
    for cutoff, projections, real, universe in walk_forward_projections(
            roster_by_cutoff, game_logs, team_logs, games_remaining_by_cutoff,
            score_end, methods=methods,
            active_fractions_by_cutoff=active_fractions_by_cutoff):
        for m, proj in projections.items():
            rows.append({"cutoff": str(cutoff), "method": m,
                         **score_projection(proj, real, universe=universe)})
    return pd.DataFrame(rows)


def walk_forward_projections(
    roster_by_cutoff: dict,
    game_logs: pd.DataFrame,
    team_logs: pd.DataFrame,
    games_remaining_by_cutoff: dict,
    score_end,
    methods=METHODS,
    blend_weights=None,
    active_fractions_by_cutoff: dict | None = None,
):
    """The projections behind `walk_forward_scores`, one cutoff at a time.

    Yields `(cutoff, {name: projection}, realized, universe)`. The universe is
    shared by every method at a cutoff so nobody gets credit for declining to
    project someone.

    `blend_weights` is an optional iterable of constant `w` values; each adds
    a `blend@w` entry to the projections dict. That is how the parameter sweep
    traces MAE against the blend weight at a fixed horizon without re-reading
    the game logs for every candidate.

    `active_fractions_by_cutoff` is keyed by the same cutoffs and feeds
    `blend_il`; without it `blend_il` is `blend`.
    """
    for cutoff in sorted(roster_by_cutoff):
        roster = roster_by_cutoff[cutoff]
        remaining = games_remaining_by_cutoff[cutoff]
        real = realized_pa(game_logs, cutoff, score_end)
        ppg = team_pa_per_game(team_logs, cutoff)
        fractions = (active_fractions_by_cutoff or {}).get(cutoff)
        projections = {
            m: project_playing_time(roster, game_logs, remaining, cutoff,
                                    pa_per_game=ppg, method=m,
                                    active_fractions=fractions)
            for m in methods
        }
        for w in (blend_weights or ()):
            projections[f"blend@{w:g}"] = project_playing_time(
                roster, game_logs, remaining, cutoff, pa_per_game=ppg,
                method="blend", blend_weight=w)
        universe = sorted(
            set(real["batter"]) |
            {b for p in projections.values()
             for b in p.loc[p["projected_pa_ros"] > 0, "batter"]}
        )
        yield cutoff, projections, real, universe
