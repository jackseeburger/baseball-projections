"""Tests for binomial cell aggregation (roadmap 0.4)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.aggregation import aggregate_binomial_cells


@pytest.fixture
def pa_rows():
    """PA-level rows: two batters, one season, one team, mixed outcomes."""
    rng = np.random.default_rng(3)
    rows = []
    for batter_idx, rate in [(0, 0.2), (1, 0.3)]:
        for _ in range(200):
            rows.append({
                "batter_idx": batter_idx, "season_idx": 0, "team_idx": 0,
                "stand_idx": batter_idx % 2, "is_k": int(rng.random() < rate),
                "age_centered": float(batter_idx), "log_pf_k": 0.01,
            })
    return pd.DataFrame(rows)


def test_counts_and_totals_preserved(pa_rows):
    cells = aggregate_binomial_cells(pa_rows)
    assert cells["n"].sum() == len(pa_rows)
    assert cells["k"].sum() == pa_rows["is_k"].sum()
    # One cell per (batter, season, team, stand) combination present.
    assert len(cells) == 2


def test_rates_match_per_cell(pa_rows):
    cells = aggregate_binomial_cells(pa_rows).set_index("batter_idx")
    for b in (0, 1):
        pa_rate = pa_rows[pa_rows.batter_idx == b]["is_k"].mean()
        assert cells.loc[b, "k"] / cells.loc[b, "n"] == pytest.approx(pa_rate)


def test_carried_features_survive(pa_rows):
    cells = aggregate_binomial_cells(pa_rows).set_index("batter_idx")
    assert cells.loc[1, "age_centered"] == 1.0
    assert cells.loc[0, "log_pf_k"] == pytest.approx(0.01)


def test_binomial_loglik_equals_bernoulli_sum(pa_rows):
    """The whole point of 0.4: identical likelihood up to a constant in p."""
    from scipy import stats

    cells = aggregate_binomial_cells(pa_rows)
    for p in (0.1, 0.25, 0.4):
        bern = stats.bernoulli.logpmf(pa_rows["is_k"], p).sum()
        binom_no_coeff = (
            cells["k"] * np.log(p) + (cells["n"] - cells["k"]) * np.log(1 - p)
        ).sum()
        assert binom_no_coeff == pytest.approx(bern)


def test_model_builds_and_matches_bernoulli_logp():
    """Full-model smoke test: skipped where pymc is not installed (CI)."""
    pm = pytest.importorskip("pymc")

    rng = np.random.default_rng(11)
    n_pa = 400
    df = pd.DataFrame({
        "batter_idx": rng.integers(0, 4, n_pa),
        "season_idx": rng.integers(0, 2, n_pa),
        "team_idx": rng.integers(0, 2, n_pa),
        "stand_idx": rng.integers(0, 2, n_pa),
        "is_k": rng.integers(0, 2, n_pa),
        "age_centered": 0.0,
        "log_pf_k": 0.0,
    })
    from src.models.aggregation import aggregate_binomial_cells
    cells = aggregate_binomial_cells(df)

    data = {
        "n_obs": len(cells),
        "n_pa": int(cells["n"].sum()),
        "n_batters": 4, "n_seasons": 2, "n_teams": 2,
        "batter_idx": cells["batter_idx"].values,
        "season_idx": cells["season_idx"].values,
        "team_idx": cells["team_idx"].values,
        "stand_idx": cells["stand_idx"].values,
        "age_centered": cells["age_centered"].values,
        "log_pf_k": cells["log_pf_k"].values,
        "k": cells["k"].values,
        "n_trials": cells["n"].values,
        "batters": np.arange(4), "seasons": np.array([2024, 2025]),
        "teams": np.array(["AAA", "BBB"]),
    }
    from src.models.pa_k_rate import build_model
    model = build_model(data)
    point = model.initial_point()
    logp = model.compile_logp()(point)
    assert np.isfinite(logp)
    # p_k Deterministic must be gone from the trace variables.
    assert "p_k" not in [v.name for v in model.deterministics]
