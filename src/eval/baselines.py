"""Baseline prediction providers for the backtest harness.

A provider maps (train_seasons, component_spec, predict_year) to a frame of
[batter, predicted]. `train_seasons` contains ONLY seasons up to the training
cutoff — the harness enforces that, so a provider cannot leak the future.

Baselines are deliberately dumb. They exist to be beaten; a model change
that does not beat Marcel is a regression no matter how principled it looks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.marcel import age_adjustment

# Marcel constants (Tango): 5/4/3 weights, 200 PA of league-average ballast.
MARCEL_YEAR_WEIGHTS = {0: 5.0, 1: 4.0, 2: 3.0}  # years before predict_year - 1
MARCEL_BALLAST = 200.0


def _league_rate(train: pd.DataFrame, spec, year: int | None = None) -> float:
    """Trials-weighted league rate, optionally for a single season."""
    df = train if year is None else train[train["season"] == year]
    return float(df[spec.successes].sum() / df[spec.trials].sum())


def league_average(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Everyone projects to the most recent training season's league rate."""
    last = int(train["season"].max())
    rate = _league_rate(train, spec, last)
    batters = train.loc[train["season"] == last, "batter"].unique()
    return pd.DataFrame({"batter": batters, "predicted": rate})


def previous_season(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Player's own rate in the last training season, no regression."""
    last = int(train["season"].max())
    g = train[train["season"] == last]
    pred = g[spec.successes] / g[spec.trials]
    return pd.DataFrame({"batter": g["batter"].values, "predicted": pred.values})


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


BASELINES = {
    "marcel": marcel,
    "previous_season": previous_season,
    "league_average": league_average,
}
