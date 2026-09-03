"""Dated cutoffs for the PA-level models — pymc-free so CI can test it.

The intra-season harness (`src/eval/intraseason.py`) cuts a season at a date
and hands the baselines the prior full seasons plus the current season
*through* the cutoff. Until this module existed there was no way to hand the
Bayesian model the same thing: `src/models/` knew only `cutoff_year`, which
filters *active batters*, not plate appearances. Every published comparison
therefore ran a model that had never seen a 2026 PA against Marcel fed 2026
through the day before.

The rule here is the harness's rule, to the row:

    training PA  ==  game_date <  cutoff
    withheld PA  ==  game_date >= cutoff

`assert_no_post_cutoff` is the leakage guard, the model-side twin of
`intraseason.assert_split_clean`.
"""
from __future__ import annotations

import pandas as pd

DATE_COLUMN = "game_date"


def apply_cutoff(
    pa: pd.DataFrame,
    cutoff_date: str | pd.Timestamp | None,
    date_col: str = DATE_COLUMN,
) -> pd.DataFrame:
    """PA strictly before `cutoff_date`. `None` returns the frame unchanged.

    Matches `intraseason.split_at_cutoff`'s first half exactly: strict `<`, so
    a game played *on* the cutoff date is withheld in both places. The frame
    keeps every column; only rows are dropped.
    """
    if cutoff_date is None:
        return pa
    if date_col not in pa.columns:
        raise KeyError(
            f"cutoff_date needs a {date_col!r} column; got {list(pa.columns)}"
        )
    cutoff = pd.Timestamp(cutoff_date)
    dates = pd.to_datetime(pa[date_col])
    return pa[dates < cutoff].copy()


def assert_no_post_cutoff(
    pa: pd.DataFrame,
    cutoff_date: str | pd.Timestamp | None,
    date_col: str = DATE_COLUMN,
    what: str = "training",
) -> None:
    """Raise if any row is dated on or after the cutoff. The leakage guard.

    Cheap enough to call on every prepared frame, and it is the only thing
    standing between a model fit and a silently leaky comparison.
    """
    if cutoff_date is None:
        return
    if date_col not in pa.columns:
        raise KeyError(
            f"leakage guard needs a {date_col!r} column; got {list(pa.columns)}"
        )
    cutoff = pd.Timestamp(cutoff_date)
    dates = pd.to_datetime(pa[date_col])
    late = dates[dates >= cutoff]
    if len(late):
        raise ValueError(
            f"leakage: {len(late)} {what} PA dated on or after the cutoff "
            f"{cutoff.date()} (latest {late.max().date()})"
        )


def cutoff_exposure(
    pa: pd.DataFrame,
    cutoff_date: str | pd.Timestamp,
    date_col: str = DATE_COLUMN,
    season_col: str = "game_year",
) -> dict:
    """What the partial season actually contributes: season, PA, date bounds.

    Reported so a run's log says how much of the current season the model was
    given, in the same units the baselines are weighted by (plate appearances).
    """
    cutoff = pd.Timestamp(cutoff_date)
    before = apply_cutoff(pa, cutoff, date_col)
    season = int(cutoff.year)
    current = before[before[season_col] == season] if season_col in before else before
    dates = pd.to_datetime(current[date_col]) if len(current) else pd.Series(dtype="datetime64[ns]")
    return {
        "cutoff": str(cutoff.date()),
        "partial_season": season,
        "partial_pa": int(len(current)),
        "partial_batters": int(current["batter"].nunique()) if "batter" in current else 0,
        "first_game": str(dates.min().date()) if len(dates) else None,
        "last_game": str(dates.max().date()) if len(dates) else None,
        "prior_pa": int(len(before) - len(current)),
    }
