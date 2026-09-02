"""Baseline prediction providers for the backtest harness.

A provider maps (train_seasons, component_spec, predict_year) to a frame of
[batter, predicted]. `train_seasons` contains ONLY seasons up to the training
cutoff — the harness enforces that, so a provider cannot leak the future.

Baselines are deliberately dumb. They exist to be beaten; a model change
that does not beat Marcel is a regression no matter how principled it looks.

**Partial seasons.** In intra-season mode the training frame's most recent
row is the current season *through the cutoff*, flagged `partial=True`. The
baselines read that flag rather than assuming every season is complete:

    marcel           treats it as the most recent season; because Marcel
                     weights by trials, a 200-PA partial season naturally
                     counts a third of a full one.
    league_average   league rate through the cutoff (the latest season).
    previous_season  the last *full* season — the partial one is skipped, so
                     this stays the "no in-season information" arm.
    season_to_date   the player's own partial-season rate regressed to
                     league with the component's ballast: "just use this
                     year".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.marcel import age_adjustment

# Marcel constants (Tango): 5/4/3 weights, 200 PA of league-average ballast.
MARCEL_YEAR_WEIGHTS = {0: 5.0, 1: 4.0, 2: 3.0}  # years before predict_year - 1
MARCEL_BALLAST = 200.0

# Trials of league-average ballast for the season-to-date baseline: the
# published stabilization points, where a player's own sample and the league
# prior carry equal weight (Carleton / FanGraphs). BABIP's is huge, which is
# the whole reason "he's hitting .400 on balls in play" means so little.
SEASON_TO_DATE_BALLAST = {
    "k_rate": 60.0,     # PA
    "bb_rate": 120.0,   # PA
    "hr_rate": 170.0,   # PA
    "babip": 820.0,     # BIP
    "iso": 160.0,       # AB
}


def _league_rate(train: pd.DataFrame, spec, year: int | None = None) -> float:
    """Trials-weighted league rate, optionally for a single season."""
    df = train if year is None else train[train["season"] == year]
    return float(df[spec.successes].sum() / df[spec.trials].sum())


def _partial_mask(train: pd.DataFrame) -> pd.Series:
    """Boolean mask of partial (through-the-cutoff) rows; all False without the flag."""
    if "partial" not in train.columns:
        return pd.Series(False, index=train.index)
    return train["partial"].fillna(False).astype(bool)


def full_seasons(train: pd.DataFrame) -> pd.DataFrame:
    """Training rows for complete seasons only (drops a partial current season)."""
    return train[~_partial_mask(train)]


def latest_rows(train: pd.DataFrame) -> pd.DataFrame:
    """The most recent training slice: the partial season if there is one,
    else the last full season."""
    partial = train[_partial_mask(train)]
    if not partial.empty:
        return partial
    return train[train["season"] == int(train["season"].max())]


def league_average(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Everyone projects to the most recent training slice's league rate.

    With a partial current season that is the league rate through the cutoff.
    """
    g = latest_rows(train)
    rate = float(g[spec.successes].sum() / g[spec.trials].sum())
    return pd.DataFrame({"batter": g["batter"].unique(), "predicted": rate})


def previous_season(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Player's own rate in the last *complete* training season, no regression."""
    full = full_seasons(train)
    if full.empty:
        raise ValueError("previous_season needs at least one complete season")
    last = int(full["season"].max())
    g = full[full["season"] == last]
    pred = g[spec.successes] / g[spec.trials]
    return pd.DataFrame({"batter": g["batter"].values, "predicted": pred.values})


def season_to_date(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """The player's own rate so far this season, regressed to league.

    pred = (successes + b·league) / (trials + b), with b the component's
    stabilization point. A player with zero trials gets exactly the league
    rate, which is what makes this a well-behaved arm at an April cutoff.
    """
    g = latest_rows(train).groupby("batter", as_index=False).agg(
        successes=(spec.successes, "sum"), trials=(spec.trials, "sum")
    )
    league = float(g["successes"].sum() / g["trials"].sum())
    b = SEASON_TO_DATE_BALLAST.get(spec.name, MARCEL_BALLAST)
    return pd.DataFrame({
        "batter": g["batter"].values,
        "predicted": (g["successes"] + b * league) / (g["trials"] + b),
    })


def marcel(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Marcel the Monkey: 5/4/3-weighted 3-year rates regressed to league
    mean with 200 trials of ballast, plus the simple age adjustment when an
    `age` column is available.
    """
    last = int(train["season"].max())
    league = _league_rate(train, spec, last)

    recent = train[train["season"] >= last - 2].copy()
    recent["w"] = (last - recent["season"]).map(MARCEL_YEAR_WEIGHTS)
    recent["w_trials"] = recent["w"] * recent[spec.trials]
    recent["w_successes"] = recent["w"] * recent[spec.successes]

    g = recent.groupby("batter").agg(
        w_trials=("w_trials", "sum"),
        w_successes=("w_successes", "sum"),
    )
    # Regress toward league mean with MARCEL_BALLAST weighted-trials of it.
    # Scale ballast by the mean year weight so it matches Marcel's intent of
    # ~200 real PA of league average.
    ballast = MARCEL_BALLAST * np.mean(list(MARCEL_YEAR_WEIGHTS.values()))
    pred = (g["w_successes"] + ballast * league) / (g["w_trials"] + ballast)
    out = pred.rename("predicted").reset_index()

    if "age" in train.columns:
        age_last = (
            train[train["season"] == last]
            .dropna(subset=["age"])
            .set_index("batter")["age"]
        )
        proj_age = out["batter"].map(age_last) + (predict_year - last)
        adj = np.array([
            age_adjustment(int(a), spec.name) if np.isfinite(a) else 1.0
            for a in proj_age
        ])
        out["predicted"] = out["predicted"] * adj
    return out


def marcel_preseason(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Marcel with the partial current season withheld.

    The control arm for intra-season backtests: identical to the Marcel a
    preseason run would have produced, scored on the same rest-of-season
    outcomes. `marcel` minus `marcel_preseason` is the value of in-season
    information, with the model held fixed.
    """
    return marcel(full_seasons(train), spec, predict_year)


BASELINES = {
    "marcel": marcel,
    "previous_season": previous_season,
    "league_average": league_average,
}

# `season_to_date` and `marcel_preseason` only say something interesting when
# the training frame carries a partial current season, so they are opt-in at
# the season level and default in intra-season mode.
INTRASEASON_BASELINES = {
    **BASELINES,
    "season_to_date": season_to_date,
    "marcel_preseason": marcel_preseason,
}
