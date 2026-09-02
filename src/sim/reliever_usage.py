"""Reliever workload → who is actually available tonight (station E).

`bullpen.py` prices the 3.5 innings the starter does not cover with a
*workload-weighted average of the whole pen*, and rules a reliever out with a
binary calendar rule: worked each of the last three days, he is gone. Measured,
that rule fires on 4 of 756 club-games and moves a pen by 0.001 runs per nine —
nothing (docs/market-benchmark-2026.md). The reason is that it only ever knows
*whether* a man pitched, never *how much*: a 9-pitch save and a 38-pitch
two-inning outing are the same fact to it, and the arm that threw 38 pitches
last night is the one who is not warming up tonight.

This module replaces the switch with a **dial**, read off the pitch counts the
game logs already carry:

    pitcher game logs ─► appearance_pitches()  one row per appearance with the
                                               pitches thrown (or an estimate
                                               from batters faced)
                      ─► recent_pitches()      pitches on each of the 1, 2 and
                                               3 calendar days before the game,
                                               per pitcher — strictly before
                      ─► availability_weight() 0 for a man who is used up,
                                               otherwise a fraction of a
                                               fresh arm
                      ─► available_pen_ra9()   the pen's FIP RA/9 with each
                                               reliever weighted by trailing
                                               workload *times* his
                                               availability
                      ─► bullpen.blend_bullpen_team()  the same 3.5/9 delta

The availability rule has three inspectable constants and no fitted curve:

    unavailable if   d1 ≥ HARD_1D_PITCHES              (heavy work yesterday)
                or   d1 + d2 ≥ HARD_2D_PITCHES         (heavy work over two)
                or   d1 > 0 and d2 > 0 and d3 > 0      (three days running)
    otherwise        w = 1 − (d1 + 0.5·d2 + 0.25·d3) / HARD_2D_PITCHES,
                     clipped to [0, 1]

The three-days-running clause is `bullpen.unavailable`'s rule kept intact — it
is what clubs actually work to — and the two pitch thresholds are what it was
missing. The taper divides by the same two-day threshold rather than
introducing a fourth constant, and the 1 / 0.5 / 0.25 recency discount is the
standard halving, not a fit.

Provenance: `HARD_1D_PITCHES` and `HARD_2D_PITCHES` were chosen walk-forward
on the **2025** season, scored on 2025 games only, and then applied unchanged
to 2026. Nothing here is fit to the 2026 games the term is scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sim.bullpen import pen_ra9

# A reliever who threw this many pitches yesterday is not available tonight.
# Roughly two innings of work: the outing after which clubs give a man a day.
HARD_1D_PITCHES = 40.0
# ...or this many across yesterday and the day before, which is the same
# ceiling reached over a back-to-back instead of in one night.
HARD_2D_PITCHES = 50.0
# How many calendar days of workload the rule looks at. Three, because the
# third day only ever enters through the three-days-running clause and the
# quarter-weight tail of the taper.
USAGE_DAYS = 3
# Recency discount on the taper: yesterday counts fully, the day before half,
# two days before a quarter. Halving, not a fit.
USAGE_DECAY = 0.5
# What the available pen is measured against: "league" is every club's pen
# scored with the same availability weights (so the term carries pen quality as
# well as availability), "team" is the club's own whole pen ignoring
# availability (availability news only). Chosen walk-forward on 2025; see
# docs/market-benchmark-2026.md.
BASELINE = "league"
# Fallback when a game log carries no pitch count: the league's pitches per
# batter faced. Overridden by `pitches_per_bf()` on real data; this is only the
# value used when there is nothing to measure it from.
DEFAULT_PITCHES_PER_BF = 3.9

APPEARANCE_COLS = ["pitcher", "team", "date", "pitches"]
DAY_COLS = [f"d{i}" for i in range(1, USAGE_DAYS + 1)]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """`df[col]` as floats, or a column of zeros when it is not there."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)


def pitches_per_bf(logs: pd.DataFrame,
                   default: float = DEFAULT_PITCHES_PER_BF) -> float:
    """League pitches per batter faced, measured from the rows that have both.

    Used only to fill in an appearance whose `pitches` column is missing or
    zero. Measured rather than assumed so the fallback lands on the same scale
    as the real counts it stands in for.
    """
    if logs.empty or "pitches" not in logs.columns:
        return float(default)
    p, bf = _num(logs, "pitches"), _num(logs, "bf")
    have = (p > 0) & (bf > 0)
    if not bool(have.any()):
        return float(default)
    return float(p[have].sum() / bf[have].sum())


def appearance_pitches(logs: pd.DataFrame, per_bf: float | None = None) -> pd.DataFrame:
    """Every pitching appearance as `pitcher, team, date, pitches`.

    Starts are kept, unlike `bullpen.relief_appearances`: this frame answers
    "how tired is this arm", and a man who opened or made a spot start two days
    ago is exactly as unavailable as one who threw two innings in relief. Which
    of those appearances count as *bullpen* work is a separate question, and
    `bullpen.pen_window` still answers it.

    `pitches` is the game log's own pitch count where it has one; where it does
    not (a zero or a missing column) it is `bf × per_bf`, and `per_bf` defaults
    to the league rate measured off `logs` itself.
    """
    if logs.empty:
        return pd.DataFrame(columns=APPEARANCE_COLS)
    rate = pitches_per_bf(logs) if per_bf is None else float(per_bf)
    p, bf = _num(logs, "pitches"), _num(logs, "bf")
    out = pd.DataFrame({
        "pitcher": pd.to_numeric(logs["pitcher"], errors="coerce").astype("int64"),
        "team": pd.to_numeric(logs.get("team", np.nan), errors="coerce").astype("Int64"),
        "date": logs["date"].astype(str),
        "pitches": np.where(p > 0, p, bf * rate),
    })
    return out.reset_index(drop=True)


def recent_pitches(appearances: pd.DataFrame, as_of: str,
                   days: int = USAGE_DAYS) -> pd.DataFrame:
    """Pitches thrown on each of the `days` calendar days before `as_of`.

    Returns `pitcher, d1, d2, d3` — `d1` is the day before `as_of`, `d2` the
    one before that — with one row per pitcher who threw at all in the window.
    Two outings on the same day (a doubleheader) are summed into one day.

    Strictly before `as_of`, always. An appearance dated `as_of` is either the
    game being predicted or an earlier game of that day's doubleheader, and
    neither had happened when the line was priced; this is the leakage guard
    and it is unit-tested.
    """
    cols = ["pitcher", *[f"d{i}" for i in range(1, days + 1)]]
    empty = pd.DataFrame(columns=cols).astype({"pitcher": "int64"})
    if appearances is None or len(appearances) == 0 or days <= 0:
        return empty
    day0 = pd.Timestamp(str(as_of))
    date = pd.to_datetime(appearances["date"], errors="coerce")
    back = (day0 - date).dt.days
    win = appearances[(back >= 1) & (back <= days)].assign(back=back)
    if win.empty:
        return empty
    wide = (win.pivot_table(index="pitcher", columns="back", values="pitches",
                            aggfunc="sum")
            .reindex(columns=range(1, days + 1), fill_value=0.0)
            .fillna(0.0))
    wide.columns = [f"d{int(c)}" for c in wide.columns]
    return wide.reset_index().astype({"pitcher": "int64"})


def availability_weight(d1, d2=0.0, d3=0.0, hard_1d: float = HARD_1D_PITCHES,
                        hard_2d: float = HARD_2D_PITCHES,
                        decay: float = USAGE_DECAY):
    """How much of a fresh arm this reliever is tonight, in [0, 1].

    Zero on any of the three hard stops (heavy yesterday, heavy over two days,
    three days running); otherwise one minus the recency-discounted load as a
    share of the two-day threshold. A man who has not pitched in three days is
    exactly 1 and leaves the pen's rate untouched.

    Scalars or aligned arrays.
    """
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    d3 = np.asarray(d3, dtype=float)
    load = d1 + decay * d2 + decay * decay * d3
    w = np.clip(1.0 - load / float(hard_2d), 0.0, 1.0)
    out = np.where((d1 >= float(hard_1d))
                   | (d1 + d2 >= float(hard_2d))
                   | ((d1 > 0) & (d2 > 0) & (d3 > 0)), 0.0, w)
    return float(out) if out.ndim == 0 else out


def availability(appearances: pd.DataFrame, as_of: str,
                 hard_1d: float = HARD_1D_PITCHES,
                 hard_2d: float = HARD_2D_PITCHES,
                 days: int = USAGE_DAYS, decay: float = USAGE_DECAY) -> dict:
    """{pitcher_id: availability weight} for one date, from the past only.

    Only pitchers who worked inside the window appear. Everyone else is fully
    available and is scored at 1.0 by `available_pen_ra9`'s lookup default, so
    the map stays small (a few dozen arms a night rather than 800).
    """
    recent = recent_pitches(appearances, as_of, days=days)
    if recent.empty:
        return {}
    zero = np.zeros(len(recent), dtype=float)
    d = [recent[f"d{i}"].to_numpy(dtype=float) if f"d{i}" in recent.columns else zero
         for i in (1, 2, 3)]
    w = availability_weight(*d, hard_1d=hard_1d, hard_2d=hard_2d, decay=decay)
    return {int(p): float(x) for p, x in zip(recent["pitcher"], np.atleast_1d(w))}


def available_pen_ra9(pen: pd.DataFrame, ra9_lookup: dict, lg_ra9: float,
                      weights: dict, exclude=()) -> float:
    """The pen's runs allowed per 9 with each arm weighted by *how* available he is.

    `pen` is a `bullpen.pen_window` slice for one club (trailing batters faced
    per reliever); `weights` is `availability`'s map, defaulting to 1 for an arm
    that has not pitched lately; `exclude` is tonight's announced starter, who
    is on the roster but is not in the pen behind himself.

    The weight on a reliever is `trailing bf × availability`, so the innings a
    tired arm would have thrown fall to whoever is left — the same
    redistribution `bullpen.pen_ra9` does with a binary exclusion, but by
    degrees.

    When every arm is at zero (or the club has no relief history yet) this
    falls back to the *whole* pen's rate rather than to league average: a club
    whose entire pen worked last night still has to pitch the game with it.
    """
    if pen is None or len(pen) == 0:
        return float(lg_ra9)
    drop = {int(p) for p in exclude}
    use = pen[~pen["pitcher"].astype("int64").isin(drop)]
    if len(use) == 0:
        return float(lg_ra9)
    ids = use["pitcher"].astype("int64").to_numpy()
    w = use["bf"].to_numpy(dtype=float) * np.array(
        [float(weights.get(int(p), 1.0)) for p in ids], dtype=float)
    if w.sum() <= 0:
        return pen_ra9(use, ra9_lookup, lg_ra9)
    r = np.array([float(ra9_lookup.get(int(p), lg_ra9)) for p in ids], dtype=float)
    return float(np.dot(w, r) / w.sum())


def league_available_pen_ra9(pens: pd.DataFrame, ra9_lookup: dict, lg_ra9: float,
                             weights: dict) -> float:
    """The same number pooled over every club — the "league" baseline.

    Every club's pen is tired on some night and rested on another, so pooling
    with the *same* availability weights leaves an average club at a delta of
    exactly zero and the term cannot move the league's run environment.
    """
    return available_pen_ra9(pens, ra9_lookup, lg_ra9, weights)
