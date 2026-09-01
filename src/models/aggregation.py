"""Binomial cell aggregation (roadmap 0.4) — pymc-free so CI can test it.

Collapsing PA rows to counts within cells keyed by every discrete predictor
leaves the likelihood identical (up to the constant binomial coefficient)
while cutting likelihood rows by an order of magnitude. The continuous
features (age, park factor) are constant within a cell because they only
vary by batter-season and team-season, both contained in the key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_binomial_cells(
    df: pd.DataFrame,
    cell_cols: tuple[str, ...] = ("batter_idx", "season_idx",
                                  "team_idx", "stand_idx"),
    outcome: str = "is_k",
    carry_cols: tuple[str, ...] = ("age_centered", "age", "birth_year",
                                   "log_pf_k"),
) -> pd.DataFrame:
    """One row per cell: `k` (successes), `n` (trials), carried features."""
    carry = [c for c in carry_cols if c in df.columns]
    cells = (
        df.groupby(list(cell_cols), as_index=False)
        .agg(
            k=(outcome, "sum"),
            n=(outcome, "size"),
            **{c: (c, "first") for c in carry},
        )
    )
    cells["k"] = cells["k"].astype(np.int64)
    cells["n"] = cells["n"].astype(np.int64)
    return cells
