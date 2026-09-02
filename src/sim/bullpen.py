"""Bullpen availability for station E (per-game P(win)).

`starters.py` prices the ~5.5 innings the announced starter covers; this module
prices the other ~3.5, which the model it extends charges to the club's
season-to-date runs-allowed rate and nothing else. The term completes the
staff: every inning of the game now has a component-rate pitcher behind it.

    ra9 = team_RA9
          + (5.5/9) · (starter FIP RA9 − league RA9)            ← starters.py
          + (3.5/9) · (available pen FIP RA9 − league pen RA9)  ← this module

Two things are folded into that one delta, and they turned out to be worth very
different amounts (docs/market-benchmark-2026.md):

  * **Who is in the pen and how good he is.** This is nearly all of it. A
    bullpen's Marcel/FIP rate is a much steadier estimate of the relief innings
    than the club's own runs allowed, which is a 60-game-ballast number
    carrying a rotation, a defence and a park along with it.
  * **Who is unavailable tonight.** A reliever who has worked each of the last
    three days is not coming out of that pen, and his innings fall to whoever
    is left. This is the part the ticket was aimed at, and measured it is worth
    almost nothing — the pen shifts by 0.03 runs per nine, an order of
    magnitude less than the quality term's 0.13, because one arm out of eight
    barely moves a workload-weighted average.

The baseline is the **league's relievers**, not the club's own pen — the
opposite of the choice `lineups.py` makes, and chosen the same way, walk-forward
on 2025, where it beat the club-baseline form on both seasons. The club
baseline isolates availability news and so carries only that 0.03; the league
baseline also carries pen quality, which is the part that pays even though team
RA/9 already contains a version of it. That is the same trade `starters.py`
makes with `lg_ra9`.

The chain, all pure functions over DataFrames so it unit-tests without a
network:

    pitching game logs ─► relief_appearances()  relief outings only (gs == 0),
                                                with the club he threw for
                       ─► pen_window()          batters faced per (team,
                                                pitcher) in the trailing
                                                roster window — who is in this
                                                pen, and how much he works
                       ─► unavailable()         pitchers who threw on enough of
                                                the last few calendar days
                       ─► pen_ra9()             workload-weighted RA/9 of a set
                                                of relievers, from the same
                                                Marcel/FIP rates starters.py
                                                builds
                       ─► blend_bullpen_team()  the delta above

Provenance: 3.5 relief innings is 9 minus the 5.5 `starters.py` already uses.
The two free knobs — how many trailing days make a pen (`ROSTER_WINDOW_DAYS`)
and what counts as used up (`REST_DAYS`/`REST_MIN_DAYS`) — were chosen
walk-forward on **2025 only**. Nothing here is fit to the 2026 games this model
is scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sim.starters import GAME_IP, STARTER_IP

# The innings the starter does not cover. Not an independent constant: it is
# 9 minus `starters.STARTER_IP`, so the two terms partition the game exactly.
RELIEF_IP = GAME_IP - STARTER_IP

# How far back to look for "who is in this bullpen". Long enough to catch a
# reliever who has not been needed for a week, short enough that a July trade
# or an option to Triple-A drops him within a couple of turns of the rotation.
ROSTER_WINDOW_DAYS = 21

# Used up: appeared on at least REST_MIN_DAYS of the REST_DAYS calendar days
# before the game. Three straight days is the hard stop clubs actually work to;
# back-to-back is routine and usually leaves a man available, which is why the
# stricter readings of the rule (`min_days=1`, "threw yesterday at all") score
# worse on 2025. Calendar days, not games, so an off day rests everyone.
# Chosen walk-forward on 2025 only.
REST_DAYS = 3
REST_MIN_DAYS = 3

# What the available pen is measured against: "league" is the league's
# relievers (so the term carries pen quality as well as availability),
# "team" is the club's own whole pen (availability news only). "league" won on
# 2025 and on 2026; see the module docstring.
BASELINE = "league"

APPEARANCE_COLS = ["pitcher", "team", "date", "bf", "outs"]


def relief_appearances(logs: pd.DataFrame) -> pd.DataFrame:
    """Relief outings from a pitching game-log frame.

    Keeps rows the pitcher did not start (`gs == 0`) and returns
    `pitcher, team, date, bf, outs`. A starter's own outings are dropped
    because `starters.py` already prices those innings; a swingman contributes
    to the pen only on the days he relieved.
    """
    if logs.empty:
        return pd.DataFrame(columns=APPEARANCE_COLS)
    keep = logs[pd.to_numeric(logs.get("gs", 0), errors="coerce").fillna(0) == 0]
    out = pd.DataFrame({
        "pitcher": pd.to_numeric(keep["pitcher"], errors="coerce").astype("int64"),
        "team": pd.to_numeric(keep["team"], errors="coerce").astype("Int64"),
        "date": keep["date"].astype(str),
        "bf": pd.to_numeric(keep.get("bf", 0), errors="coerce").fillna(0.0).astype(float),
        "outs": pd.to_numeric(keep.get("outs", 0), errors="coerce").fillna(0.0).astype(float),
    })
    return out.dropna(subset=["team"]).reset_index(drop=True)


def _shift(as_of: str, days: int) -> str:
    return str((pd.Timestamp(str(as_of)) - pd.Timedelta(days=days)).date())


def pen_window(relief: pd.DataFrame, as_of: str,
               days: int = ROSTER_WINDOW_DAYS) -> pd.DataFrame:
    """Who is in each pen as of `as_of`, and how much of it he throws.

    Returns `team, pitcher, bf` — batters faced in relief over the `days`
    calendar days strictly before `as_of`. That window is doing two jobs:
    membership (a reliever who has not thrown for this club in three weeks is
    not in this pen) and *expected usage* (the shares are what `pen_ra9`
    weights by, so the arms a manager actually leans on count for more than the
    mop-up man).

    Strictly before, always: an appearance on the game's own date is the game
    being predicted, or an earlier game of a doubleheader that has not been
    played when the line is priced.
    """
    cols = ["team", "pitcher", "bf"]
    if relief.empty:
        return pd.DataFrame(columns=cols)
    lo = _shift(as_of, days)
    win = relief[(relief["date"] < str(as_of)) & (relief["date"] >= lo)]
    if win.empty:
        return pd.DataFrame(columns=cols)
    out = win.groupby(["team", "pitcher"], as_index=False)["bf"].sum()
    return out[out["bf"] > 0].reset_index(drop=True)


def unavailable(relief: pd.DataFrame, as_of: str, days: int = REST_DAYS,
                min_days: int = REST_MIN_DAYS) -> set[int]:
    """Relievers who cannot pitch on `as_of` because they just did.

    A pitcher is out if he appeared on at least `min_days` of the `days`
    calendar days before the game. `days=2, min_days=2` is back-to-back;
    `min_days=1` is the stricter "threw yesterday at all" reading.

    Calendar days rather than games, deliberately: after an off day nobody
    pitched yesterday, so the whole pen is available, which is right.
    """
    if relief.empty or days <= 0:
        return set()
    lo = _shift(as_of, days)
    win = relief[(relief["date"] < str(as_of)) & (relief["date"] >= lo)]
    if win.empty:
        return set()
    used = win.groupby("pitcher")["date"].nunique()
    return {int(p) for p in used[used >= min_days].index}


def pen_ra9(pen: pd.DataFrame, ra9_lookup: dict, lg_ra9: float,
            exclude=()) -> float:
    """Workload-weighted runs allowed per 9 for a set of relievers.

    `pen` is a `pen_window` slice for one club; `ra9_lookup` is
    `starters.starter_ra9_lookup` — the same Marcel-weighted, league-regressed
    FIP the rotation is priced with, because a relief inning and a starting
    inning are the same three outs. Relievers missing from the lookup (a rookie
    with no history at all) are scored at `lg_ra9`.

    Weights are trailing batters faced, so the arms the manager leans on move
    the number most, and when the leaned-on arms are excluded the innings fall
    to whoever is left — which is the whole point of the term.

    An empty pen (everyone excluded, or a club with no relief history yet)
    returns `lg_ra9`, i.e. no adjustment at all.
    """
    if pen is None or len(pen) == 0:
        return float(lg_ra9)
    drop = {int(p) for p in exclude}
    use = pen[~pen["pitcher"].astype("int64").isin(drop)]
    w = use["bf"].to_numpy(dtype=float)
    if w.sum() <= 0:
        return float(lg_ra9)
    r = np.array([float(ra9_lookup.get(int(p), lg_ra9)) for p in use["pitcher"]],
                 dtype=float)
    return float(np.dot(w, r) / w.sum())


def league_pen_ra9(pen: pd.DataFrame, ra9_lookup: dict, lg_ra9: float) -> float:
    """The same number pooled over every club — the default "league" baseline.

    Computed from the same relievers and the same trailing-workload weights, so
    a club with a perfectly average pen gets a delta of exactly zero and the
    term cannot shift the league's run environment.
    """
    return pen_ra9(pen, ra9_lookup, lg_ra9)


def blend_bullpen_team(bp_ra9, team_ra9, baseline_ra9,
                       relief_ip: float = RELIEF_IP, game_ip: float = GAME_IP):
    """Expected runs allowed per 9 with *this* pen behind the staff.

        team_ra9 + (relief_ip/game_ip) · (bp_ra9 − baseline_ra9)

    Stacks additively with `starters.blend_starter_team`: the starter moves the
    5.5 innings he covers, the pen moves the other 3.5, and a league-average
    starter in front of a fully rested pen leaves `team_ra9` untouched. The
    delta form is the same argument `starters.py` and `lineups.py` make — FIP
    is park- and defense-neutral, team RA/9 is not, so an absolute blend would
    quietly re-regress every club's run prevention toward the league mean.

    Scalars or aligned arrays.
    """
    w = float(relief_ip) / float(game_ip)
    return np.asarray(team_ra9, dtype=float) + w * (
        np.asarray(bp_ra9, dtype=float) - np.asarray(baseline_ra9, dtype=float))
