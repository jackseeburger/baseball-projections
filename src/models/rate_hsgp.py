"""Shared helpers for the HSGP batter-season rate models (`iso_rate.py`,
`babip_rate.py`) — age attachment and HSGP posterior-age-effect evaluation.

Split out so the two models share one implementation instead of one
importing the other's private functions. Both were extracted from
`modal_functions/app.py` (issue #86); see their module docstrings and
docs/modal-src-divergence.md for provenance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_AGE = 27.0


def attach_calendar_age(
    df: pd.DataFrame,
    batter_col: str = "batter",
    year_col: str = "game_year",
    reference_age: float = REFERENCE_AGE,
    hitter_seasons: pd.DataFrame | None = None,
) -> None:
    """In-place: birth_year / age / age_centered.

    Calendar-year convention (`age = game_year - birth_year`), not the
    June-30 seasonal-age convention `src.data.birthdates.seasonal_age` uses
    for `pa_k_rate.py`. This is Modal's original convention, preserved
    as-is per docs/modal-src-divergence.md — not something this extraction
    changed. Three tiers, same order the inlined code used:

      1. The PA row's own `birth_year` column, when present (100% coverage
         in the current pipeline, so this is the tier that actually fires).
      2. A per-player median of `year - age` from `hitter_seasons`, when
         supplied and column 1 is absent.
      3. `first_year - 24`, where `first_year` is this player's first
         season in `df`.
    """
    if "birth_year" in df.columns:
        df["birth_year"] = df["birth_year"].astype(np.float64)
        df["age"] = (df[year_col] - df["birth_year"]).astype(np.float64)
        df["age_centered"] = (df["age"] - reference_age).astype(np.float64)
        return

    birth_map: dict = {}
    if hitter_seasons is not None:
        hs = hitter_seasons
        age_col = next((c for c in hs.columns if c.lower() == "age"), None)
        yr_col = next((c for c in hs.columns if c.lower() in ("season", "game_year", "year")), None)
        id_col = next((c for c in hs.columns if "mlbam" in c.lower() or c in ("IDfg", "fg_id")), None)
        if age_col and yr_col and id_col:
            hs = hs.copy()
            hs["_birth"] = hs[yr_col] - hs[age_col]
            birth_map = hs.groupby(id_col)["_birth"].median().round().astype(int).to_dict()

    first_year = df.groupby(batter_col)[year_col].min().to_dict()

    def _birth_year(batter_id):
        if batter_id in birth_map:
            return float(birth_map[batter_id])
        return float(first_year.get(batter_id, 2015) - 24)

    df["birth_year"] = df[batter_col].map(_birth_year).astype(np.float64)
    df["age"] = (df[year_col] - df["birth_year"]).astype(np.float64)
    df["age_centered"] = (df["age"] - reference_age).astype(np.float64)


def gp_age_evaluator(trace, obs_age_values: np.ndarray):
    """Closure evaluating the learned HSGP age effect at arbitrary ages.

    The HSGP prior gives an age effect at each *training* age directly in
    the posterior; projecting to a new age interpolates the posterior mean
    age effect across unique training ages (linear, extrapolated at the
    ends). Cheaper and more robust than reconstructing the HSGP spectral
    basis by hand, at the cost of being an interpolation rather than an
    exact GP evaluation off the training grid.

    Returns (eval_at, n_samples) where `eval_at(ages) -> (len(ages), n_samples)`.
    """
    from scipy.interpolate import interp1d

    post = trace.posterior
    age_effect_post = post["age_effect"].values
    nc, nd = age_effect_post.shape[:2]
    ns = nc * nd
    age_effect_flat = age_effect_post.reshape(ns, -1)

    unique_ages = np.sort(np.unique(obs_age_values))
    age_to_idx: dict[float, list[int]] = {}
    for i, a in enumerate(obs_age_values):
        age_to_idx.setdefault(float(a), []).append(i)

    age_effect_by_age = np.zeros((len(unique_ages), ns))
    for j, a in enumerate(unique_ages):
        idx = age_to_idx[float(a)]
        age_effect_by_age[j, :] = age_effect_flat[:, idx].mean(axis=1)

    interp = interp1d(unique_ages, age_effect_by_age, axis=0,
                       kind="linear", fill_value="extrapolate")

    def eval_at(ages_new: np.ndarray) -> np.ndarray:
        return interp(ages_new)

    return eval_at, ns
