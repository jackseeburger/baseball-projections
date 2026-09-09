"""Swing decisions as covariates on a component projection (BAS-75, stage 2).

Contact quality is a measurement of batted balls, so the components it has a
right to speak to are the ones a batted ball is part of. K% and BB% are the
two rates with no measurement of their own — the harness sees the realized
strikeout and walk counts and Marcel regresses them, and that is all. Swing
decisions are the missing measurement: whether the batter offered at a pitch
outside the zone, whether he offered at one inside it, and whether the swing
found the ball.

The question is the same one contact quality had to answer, asked of a
different measurement:

    does a swing-decision aggregate carry information about the rest of the
    season **beyond** what the realized K% and BB% already carry, or does it
    restate them with less noise?

and it has the same two shapes in the numbers — information survives at large
samples and at the late cutoff, variance reduction lives in the small ones and
is gone by August. `scripts/run_swing_backtest.py` runs
docs/contact-quality.md §6's split and reports it whichever way it comes out.

**The six aggregates**, each formed at read time from the summed counts of
`src/data/swing_decisions.py` and each taken **relative to the league over the
same months the player's own window covers**:

    chase      swings at pitches outside the zone, per such pitch
    zswing     swings at pitches inside the zone, per such pitch
    zcontact   swings inside the zone that touched the ball, per such swing
    ocontact   swings outside the zone that touched the ball, per such swing
    whiff      swings that missed, per swing
    cstrike    called strikes on pitches taken inside the zone, per such take

The league-relative step matters more here than it did for contact quality.
The zone-call distribution is not stationary — the strike zone shrank from
2015 to 2019 and the automated-ball-strike era changes it again — so a raw
chase rate is partly a statement about which season it was measured in. Each
bucket's league rate for *its own* (season, month) is carried through the same
weighted sum as the player's own counts, so the reference a player is
differenced against is the league over exactly his own window, and the
shrinkage target is that same reference rather than a pooled constant.

**Leakage.** The features are summed from monthly buckets strictly before the
cutoff. `assert_month_boundary` refuses a cutoff that is not the first of a
month rather than rounding one (rounding forward is leakage), and
`assert_window_clean` re-checks the *filtered* rows rather than trusting the
filter. `tests/test_eval/test_swing.py` drives that guard with a synthetic
season whose post-cutoff months are thousands of pitches of pathological
swing behaviour, and asserts the feature at a May 1 cutoff is identical
bit-for-bit to the feature built from a frame with those rows deleted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.swing_decisions import COUNT_COLUMNS

# The covariates, in a fixed order so a coefficient vector is readable.
FEATURES = ("chase", "zswing", "zcontact", "ocontact", "whiff", "cstrike")

# name -> (numerator columns, denominator columns). Both sides are sums of
# bucket counts, which is what makes a ratio over any window a ratio of two
# sums rather than an average of monthly ratios.
RATIOS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "chase": (("swings_o",), ("pitches_o",)),
    "zswing": (("swings_z",), ("pitches_z",)),
    "zcontact": (("contact_z",), ("swings_z",)),
    "ocontact": (("contact_o",), ("swings_o",)),
    "whiff": (("whiff_z", "whiff_o"), ("swings_z", "swings_o")),
    "cstrike": (("called_z",), ("pitches_z", "-swings_z")),
}
EXPECTED_COLUMNS = [f"exp_{f}" for f in FEATURES]
DEN_COLUMNS = [f"den_{f}" for f in FEATURES]

# Recency over the three seasons the projection window covers, most recent
# first — the current season through the cutoff, then the two before it. The
# same grid contact quality sweeps.
SWING_WEIGHT_GRID = [
    (1.0, 0.0, 0.0),
    (1.0, 0.35, 0.1),
    (1.0, 0.6, 0.35),
    (1.0, 0.8, 0.6),
    (1.0, 1.0, 1.0),
]
# Pitches (or swings, or takes — each ratio's own denominator) of league
# behaviour to regress a player's own toward. The grid runs from almost no
# shrinkage to a third of a season's pitches, since a player-month carries an
# order of magnitude more pitches than batted balls and it was not obvious in
# advance which end that would land on.
SWING_BALLAST_GRID = [2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]

# Chosen by `scripts/run_swing_backtest.py --tune` on the tuning window only
# (cells through 2021), by pooled trials-weighted MAE of the `swing` arm over
# K% and BB% — the two components this measurement is for. Every scored
# season is later than the tuning window.
# It lands at the low end: 10 pitches of the league's own behaviour, i.e.
# almost none. That is the pitch grain paying for itself — a batter has
# hundreds of pre-cutoff pitches where he has dozens of batted balls, so the
# raw rate is already stable and shrinking it only blurs it. The surface is
# shallow around the choice (10 -> .024395, 5 -> .024400, 25 -> .024410,
# 100 -> .024462 pooled MAE), so read it as the flat region and not a sharp
# optimum; the grid extends below the chosen value, so it is interior.
TUNED = {"weights": (1.0, 0.35, 0.1), "ballast": 10.0}
DEFAULT_WINDOW_WEIGHTS = TUNED["weights"]
DEFAULT_BALLAST = TUNED["ballast"]


# --- window sums -------------------------------------------------------------

def assert_month_boundary(cutoff) -> None:
    """Monthly buckets can only answer questions asked on the 1st of a month."""
    if pd.Timestamp(cutoff).day != 1:
        raise ValueError(
            f"swing-decision buckets are monthly; cutoff "
            f"{pd.Timestamp(cutoff).date()} is not the first of a month. "
            "Rounding a cutoff forward would leak post-cutoff pitches into "
            "the features.")


def assert_window_clean(rows: pd.DataFrame, cutoff, predict_year: int) -> None:
    """Leakage guard: no bucket at or after the cutoff month entered the window.

    `rows` is the monthly frame *after* the window filter, so this checks the
    filter rather than trusting it. Raises ValueError naming the offenders.
    """
    cutoff = pd.Timestamp(cutoff)
    if rows.empty:
        return
    if int(rows["season"].max()) > predict_year:
        raise ValueError(
            f"leakage: swing window contains season {int(rows['season'].max())}"
            f" > predict year {predict_year}")
    current = rows[rows["season"] == predict_year]
    if not current.empty and int(current["month"].max()) >= cutoff.month:
        bad = current[current["month"] >= cutoff.month]
        raise ValueError(
            f"leakage: {len(bad)} swing bucket(s) in month "
            f"{sorted(bad['month'].unique())} of {predict_year} are on or "
            f"after the cutoff {cutoff.date()}")


def _sum_cols(df: pd.DataFrame, cols) -> np.ndarray:
    """Sum a ratio's columns; a leading '-' subtracts (takes = pitches − swings)."""
    out = np.zeros(len(df), dtype="float64")
    for c in cols:
        if c.startswith("-"):
            out -= df[c[1:]].to_numpy(dtype="float64")
        else:
            out += df[c].to_numpy(dtype="float64")
    return out


def league_month_rates(monthly: pd.DataFrame) -> pd.DataFrame:
    """The six rates for the whole league, per (season, month).

    One row per calendar month of the league's own pitches — the reference a
    player's window is differenced against, month by month, so an era's zone
    is compared with itself.
    """
    tot = monthly.groupby(["season", "month"], as_index=False)[COUNT_COLUMNS].sum()
    out = tot[["season", "month"]].copy()
    for f, (num, den) in RATIOS.items():
        n, d = _sum_cols(tot, num), _sum_cols(tot, den)
        out[f] = np.where(d > 0, n / np.maximum(d, 1e-9), np.nan)
    return out


def window_counts(
    monthly: pd.DataFrame,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
) -> pd.DataFrame:
    """Recency-weighted swing-decision counts per batter, strictly before the cutoff.

    Returns one row per batter with every column of `COUNT_COLUMNS`, an
    unweighted `pitches_raw` (the real pitches behind the row, which is what a
    sample-size split should be cut on), and for each ratio a `den_*` (its
    weighted denominator) and an `exp_*` (that denominator's league
    expectation, summed over the same buckets at the same weights). The
    league reference for a player is then `exp_f / den_f`, which is the
    league's rate over exactly his own months.
    """
    cutoff = pd.Timestamp(cutoff)
    assert_month_boundary(cutoff)
    # A weight of zero means the season is *not in the window*, so it is
    # dropped rather than multiplied by nothing: its pitches would otherwise
    # still count toward `pitches_raw`, which the standardization weights by
    # and the exposure split is cut on.
    w = {predict_year - i: float(x) for i, x in enumerate(weights)
         if float(x) != 0.0}

    rows = monthly[monthly["season"].isin(w)]
    rows = rows[(rows["season"] < predict_year)
                | (rows["month"] < cutoff.month)].copy()
    assert_window_clean(rows, cutoff, predict_year)
    if rows.empty:
        return pd.DataFrame(columns=["player", *COUNT_COLUMNS, "pitches_raw",
                                     *DEN_COLUMNS, *EXPECTED_COLUMNS])

    lg = league_month_rates(monthly[
        (monthly["season"] < predict_year)
        | (monthly["month"] < cutoff.month)])
    rows = rows.merge(lg, on=["season", "month"], how="left",
                      suffixes=("", "_lg"))
    wt = rows["season"].map(w).to_numpy(dtype="float64")

    out = pd.DataFrame({"player": rows["batter"].to_numpy()})
    for c in COUNT_COLUMNS:
        out[c] = rows[c].to_numpy(dtype="float64") * wt
    out["pitches_raw"] = (rows["pitches_z"].to_numpy(dtype="float64")
                          + rows["pitches_o"].to_numpy(dtype="float64"))
    for f in FEATURES:
        den = _sum_cols(rows, RATIOS[f][1]) * wt
        out[f"den_{f}"] = den
        out[f"exp_{f}"] = den * np.nan_to_num(
            rows[f].to_numpy(dtype="float64"), nan=0.0)
    return out.groupby("player", as_index=False).sum()


# --- metrics -----------------------------------------------------------------

def swing_metrics(counts: pd.DataFrame,
                  ballast: float = DEFAULT_BALLAST) -> pd.DataFrame:
    """Shrunk, league-month-relative swing-decision rates per batter.

    Every metric is a ratio of two summed counts, so shrinkage is one
    operation: add `ballast` of the player's *own* league reference to both
    sides and take the ratio. The feature is then the difference from that
    reference, which is zero for a batter with no pitches at all and for a
    batter who behaved exactly like his league.
    """
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *FEATURES])
    b = float(ballast)
    out = pd.DataFrame({
        "player": counts["player"].to_numpy(),
        "pitches_raw": counts["pitches_raw"].to_numpy(dtype="float64"),
    })
    for f in FEATURES:
        den = counts[f"den_{f}"].to_numpy(dtype="float64")
        num = _sum_cols(counts, RATIOS[f][0])
        ref = np.where(den > 0, counts[f"exp_{f}"].to_numpy(dtype="float64")
                       / np.maximum(den, 1e-9), 0.0)
        out[f] = (num + b * ref) / (den + b) - ref
    return out


def standardize(metrics: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Pitch-weighted z-scores of each metric across the batters present.

    Weighted by raw exposure so the centre is the league's typical pitch
    rather than the typical September call-up, and computed from the cutoff's
    own pre-cutoff window, so nothing outside the training data enters it.
    """
    out = metrics[["player", "pitches_raw"]].copy()
    w = metrics["pitches_raw"].to_numpy(dtype="float64")
    if w.sum() <= 0:
        w = np.ones_like(w)
    for f in features:
        x = metrics[f].to_numpy(dtype="float64")
        mu = float(np.average(x, weights=w))
        sd = float(np.sqrt(np.average((x - mu) ** 2, weights=w)))
        out[f] = (x - mu) / sd if sd > 0 else 0.0
    return out


def features_at_cutoff(
    monthly: pd.DataFrame,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
    ballast: float = DEFAULT_BALLAST,
) -> pd.DataFrame:
    """Standardized swing-decision covariates for every batter with pre-cutoff pitches."""
    counts = window_counts(monthly, cutoff, predict_year, weights)
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *FEATURES])
    return standardize(swing_metrics(counts, ballast))


# --- the vacuity check -------------------------------------------------------

def player_season_rates(monthly: pd.DataFrame, season: int) -> pd.DataFrame:
    """Whole-season rates per batter, with the pitches behind them.

    The input to the pre-registered vacuity check
    (docs/swing-decisions.md): a rate that does not correlate with the same
    player's rate a year later is not measuring a skill, and stage 2 could not
    mean anything.
    """
    rows = monthly[monthly["season"] == season]
    g = rows.groupby("batter", as_index=False)[COUNT_COLUMNS].sum()
    out = pd.DataFrame({
        "batter": g["batter"].to_numpy(),
        "pitches": (g["pitches_z"] + g["pitches_o"]).to_numpy(dtype="float64"),
    })
    for f, (num, den) in RATIOS.items():
        n, d = _sum_cols(g, num), _sum_cols(g, den)
        out[f] = np.where(d > 0, n / np.maximum(d, 1e-9), np.nan)
    return out


def year_over_year(monthly: pd.DataFrame, seasons, min_pitches: int = 500,
                   features=FEATURES) -> pd.DataFrame:
    """Year-over-year correlation of each rate, for batters over `min_pitches`
    in *both* years. One row per consecutive season pair in `seasons`."""
    rows = []
    seasons = sorted(seasons)
    for y0, y1 in zip(seasons, seasons[1:]):
        if y1 != y0 + 1:
            continue
        a = player_season_rates(monthly, y0)
        b = player_season_rates(monthly, y1)
        a = a[a["pitches"] >= min_pitches]
        b = b[b["pitches"] >= min_pitches]
        j = a.merge(b, on="batter", suffixes=("_0", "_1"))
        if len(j) < 3:
            continue
        row = {"year0": y0, "year1": y1, "n": int(len(j))}
        for f in features:
            row[f] = float(np.corrcoef(j[f"{f}_0"], j[f"{f}_1"])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_year_over_year(monthly: pd.DataFrame, seasons,
                          min_pitches: int = 500,
                          features=FEATURES) -> dict:
    """The same correlation pooled over every consecutive pair — the single
    number docs/swing-decisions.md's vacuity threshold is stated against."""
    seasons = sorted(seasons)
    pairs = []
    for y0, y1 in zip(seasons, seasons[1:]):
        if y1 != y0 + 1:
            continue
        a = player_season_rates(monthly, y0)
        b = player_season_rates(monthly, y1)
        j = a[a["pitches"] >= min_pitches].merge(
            b[b["pitches"] >= min_pitches], on="batter", suffixes=("_0", "_1"))
        pairs.append(j)
    if not pairs:
        return {"n": 0}
    j = pd.concat(pairs, ignore_index=True)
    out = {"n": int(len(j))}
    for f in features:
        out[f] = float(np.corrcoef(j[f"{f}_0"], j[f"{f}_1"])[0, 1])
    return out


__all__ = [
    "DEFAULT_BALLAST", "DEFAULT_WINDOW_WEIGHTS", "DEN_COLUMNS",
    "EXPECTED_COLUMNS", "FEATURES", "RATIOS", "SWING_BALLAST_GRID",
    "SWING_WEIGHT_GRID", "TUNED", "assert_month_boundary",
    "assert_window_clean", "features_at_cutoff", "league_month_rates",
    "player_season_rates", "pooled_year_over_year", "standardize",
    "swing_metrics", "window_counts", "year_over_year",
]
