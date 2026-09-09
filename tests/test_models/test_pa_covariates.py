"""The layer-1 covariate block on the PA rate model (BAS-83).

Three things have to be true and none of them is checkable by reading the
code:

1. **Off is off.** With `covariates=None` the graph carries no new free RV and
   the data dict carries no `cov_x`. The bit-for-bit fixture in
   `tests/test_models/test_pa_rate.py` pins the numbers; this file pins the
   variable set, which is what says *why* the numbers held.
2. **On changes the graph, by name.** The pre-registration names the new RVs
   `beta_cov_<aggregate>` and the dense sweep's `VARIANT_OWN_PARAMS` reads
   them by that name, so a rename has to fail here rather than silently
   produce an empty vacuity column.
3. **A same-month bucket cannot reach the design matrix.** The leakage test
   pattern from `tests/test_eval/test_contact.py`: every bucket from the
   cutoff month on is 1,000 batted balls of 120 mph barrels, so a single
   leaked row moves the covariate enormously. Deleting those rows must change
   `cov_x` by exactly nothing — `assert_array_equal`, not `allclose`.

The graph tests need pymc; the covariate-array tests do not, and
`src/models/pa_covariates.py` imports neither pymc nor arviz for that reason.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.contact_quality import COUNT_COLUMNS, EV_BIN_COLUMNS  # noqa: E402
from src.models.pa_covariates import (  # noqa: E402
    CONTACT_COVARIATES, attach_covariates, contact_covariate_array,
    covariate_names, season_cutoff,
)

HAS_PYMC = all(importlib.util.find_spec(m) is not None
               for m in ("pymc", "arviz"))
needs_pymc = pytest.mark.skipif(
    not HAS_PYMC, reason="MCMC deps (pymc/arviz) are not installed in CI")

SEASONS = (2023, 2024)
BATTERS = (101, 102)


def _bucket(player, season, month, bbe, ev, la, barrel_frac=0.0, ev_bin=8):
    row = {"side": "hitter", "player": player, "season": season, "month": month}
    row.update({c: 0.0 for c in COUNT_COLUMNS})
    row["bbe"] = float(bbe)
    row["sum_ev"] = float(bbe) * ev
    row["sum_ev2"] = float(bbe) * ev * ev
    row["sum_la"] = float(bbe) * la
    row["n_barrel"] = float(bbe) * barrel_frac
    row["n_hardhit"] = float(bbe) * (1.0 if ev >= 95 else 0.0)
    row["n_sweetspot"] = float(bbe) * (1.0 if 8 <= la <= 32 else 0.0)
    row[EV_BIN_COLUMNS[ev_bin]] = float(bbe)
    return row


@pytest.fixture
def monthly():
    """Two hitters with a real April/May and a monstrous June onward.

    The two differ in April so the standardization has something to spread,
    and every bucket from June (the cutoff month) on is 1,000 batted balls of
    120 mph barrels — so any leak is visible without a tolerance.
    """
    rows = []
    for season in SEASONS:
        rows.append(_bucket(101, season, 4, 60, 86.0, 8.0, barrel_frac=0.02, ev_bin=5))
        rows.append(_bucket(102, season, 4, 60, 92.0, 16.0, barrel_frac=0.09, ev_bin=9))
        rows.append(_bucket(101, season, 5, 55, 87.0, 10.0, barrel_frac=0.03, ev_bin=6))
        rows.append(_bucket(102, season, 5, 55, 91.0, 14.0, barrel_frac=0.08, ev_bin=8))
        for month in (6, 7, 8, 9):
            for player in BATTERS:
                rows.append(_bucket(player, season, month, 1000, 120.0, 25.0,
                                    barrel_frac=1.0,
                                    ev_bin=len(EV_BIN_COLUMNS) - 1))
    return pd.DataFrame(rows)


# ─── the month lag ─────────────────────────────────────────────────────────

class TestSeasonCutoff:
    def test_a_prior_season_is_read_whole(self):
        assert season_cutoff(2023, "2024-06-15") == "2023-12-01"

    def test_the_cutoff_season_is_lagged_to_its_month_boundary(self):
        """Exactly `src.projections.ros.contact_cutoff`: a June 15 cutoff
        reads contact through May 31, never through June 14."""
        from src.projections.ros import contact_cutoff

        for date in ("2024-06-01", "2024-06-15", "2024-06-30"):
            assert season_cutoff(2024, date) == "2024-06-01"
            assert season_cutoff(2024, date) == str(contact_cutoff(date).date())


# ─── the leakage guard ─────────────────────────────────────────────────────

def test_a_same_month_bucket_is_excluded_bit_for_bit(monthly):
    """Deleting every bucket at or after the cutoff month changes nothing.

    The `tests/test_eval/test_contact.py` pattern, applied to the design
    matrix the model actually reads rather than to the feature frame.
    """
    before = contact_covariate_array(BATTERS, SEASONS, monthly, "2024-06-15")
    clean = monthly[~((monthly["season"] == 2024) & (monthly["month"] >= 6))]
    after = contact_covariate_array(BATTERS, SEASONS, clean, "2024-06-15")
    np.testing.assert_array_equal(before, after)


def test_a_leaked_month_would_have_moved_it(monthly):
    """The control on the test above: the monstrous buckets are not inert."""
    lagged = contact_covariate_array(BATTERS, SEASONS, monthly, "2024-06-15")
    leaked = contact_covariate_array(BATTERS, SEASONS, monthly, "2024-08-01")
    assert not np.allclose(lagged[:, 1, :], leaked[:, 1, :])


def test_a_prior_season_reads_all_of_its_months(monthly):
    """2023 is complete before any 2024 cutoff, so its row must contain the
    June-onward buckets that 2024's row must not."""
    x = contact_covariate_array(BATTERS, SEASONS, monthly, "2024-06-15")
    clean_prior = monthly[~((monthly["season"] == 2023) & (monthly["month"] >= 6))]
    y = contact_covariate_array(BATTERS, SEASONS, clean_prior, "2024-06-15")
    assert not np.allclose(x[:, 0, :], y[:, 0, :])
    np.testing.assert_array_equal(x[:, 1, :], y[:, 1, :])


# ─── shape, standardization, and the missing set ───────────────────────────

def test_shape_and_column_order(monthly):
    x = contact_covariate_array(BATTERS, SEASONS, monthly, "2024-06-15")
    assert x.shape == (len(BATTERS), len(SEASONS), len(CONTACT_COVARIATES))


def test_a_batter_with_no_tracked_contact_sits_at_the_league_mean(monthly):
    """x = 0 on every aggregate, and no indicator column joins it — the
    choice documented in src/models/pa_covariates.py's module docstring."""
    x = contact_covariate_array([101, 999], SEASONS, monthly, "2024-06-15")
    np.testing.assert_array_equal(x[1], np.zeros((len(SEASONS),
                                                  len(CONTACT_COVARIATES))))
    assert x.shape[2] == len(CONTACT_COVARIATES), "no has_cov column"


def test_each_season_is_standardized_on_its_own(monthly):
    """Two batters, batted-ball weighted z-scores: within a season the two
    sit on opposite sides of zero and the weighted mean is zero.

    Checked on the three aggregates the fixture actually separates the two
    hitters on — both are under the 95 mph hard-hit line and both put their
    launch angles inside the sweet spot, so `hardhit` and `sweetspot` are
    constant across the pair and standardize to 0 by construction (which
    `test_a_constant_metric_standardizes_to_zero_not_to_nan` in
    `tests/test_eval/test_contact.py` already pins).
    """
    varying = [CONTACT_COVARIATES.index(f) for f in ("ev_mean", "barrel", "la_mean")]
    x = contact_covariate_array(BATTERS, SEASONS, monthly, "2024-06-15")
    for s in range(len(SEASONS)):
        # Equal exposure here, so the plain mean is the weighted one.
        np.testing.assert_allclose(x[:, s, :].mean(axis=0), 0.0, atol=1e-12)
        assert np.all(x[0, s, varying] < 0) and np.all(x[1, s, varying] > 0)


def test_covariate_names_resolves_and_refuses():
    assert covariate_names(None) == ()
    assert covariate_names("contact") == CONTACT_COVARIATES
    assert covariate_names(("barrel", "ev90")) == ("barrel", "ev90")
    with pytest.raises(ValueError, match="unknown covariate set"):
        covariate_names("stuff")
    with pytest.raises(ValueError, match="unknown covariate"):
        covariate_names(("spin_rate",))


def test_attach_covariates_needs_a_cutoff(monthly):
    data = {"batters": np.array(BATTERS), "seasons": np.array(SEASONS),
            "n_batters": 2, "n_seasons": 2, "cutoff_date": None}
    with pytest.raises(ValueError, match="no cutoff_date"):
        attach_covariates(data, "contact", monthly)


def test_attach_covariates_is_a_no_op_when_off(monthly):
    data = {"batters": np.array(BATTERS), "seasons": np.array(SEASONS),
            "n_batters": 2, "n_seasons": 2, "cutoff_date": None}
    assert attach_covariates(data, None, monthly) is data
    assert "cov_x" not in data and "cov_names" not in data


# ─── the graph ─────────────────────────────────────────────────────────────

def _fixture_data(monthly, cutoff="2024-06-15"):
    from src.models.pa_rate import prepare_model_data

    rng = np.random.default_rng(5)
    rows = []
    for b in BATTERS:
        for s in SEASONS:
            for _ in range(80):
                rows.append({
                    "batter": b, "game_year": s,
                    "game_date": pd.Timestamp(f"{s}-04-20"),
                    "stand": "R", "is_k": int(rng.random() < 0.24),
                    "home_team": "HOU", "away_team": "SEA",
                    "inning_topbot": "Top",
                })
    pa = pd.DataFrame(rows)
    return prepare_model_data(pa, None, min_pa=1, cutoff_date=cutoff,
                              component="k_rate")


@needs_pymc
class TestTheGraph:
    def test_covariates_off_adds_no_variable_and_no_data_container(self, monthly):
        from src.models.pa_rate import ModelOptions, build_model

        data = _fixture_data(monthly)
        model = build_model(data, ModelOptions())
        names = {v.name for v in model.free_RVs}
        assert not any(n.startswith("beta_cov") for n in names)
        assert "cov_effect" not in {v.name for v in model.deterministics}
        assert "cov_x" not in data

    def test_covariates_on_add_one_named_rv_per_aggregate(self, monthly):
        from src.models.pa_rate import ModelOptions, build_model

        data = _fixture_data(monthly)
        attach_covariates(data, "contact", monthly)
        assert data["cov_x"].shape == (data["n_batters"], data["n_seasons"],
                                       len(CONTACT_COVARIATES))

        off = {v.name for v in build_model(data, ModelOptions()).free_RVs}
        on = {v.name for v in
              build_model(data, ModelOptions(covariates="contact")).free_RVs}
        assert on - off == {f"beta_cov_{f}" for f in CONTACT_COVARIATES}
        assert off - on == set(), "the covariate block removes nothing"

    def test_the_walk_and_the_covariates_compose(self, monthly):
        from src.models.pa_rate import ModelOptions, build_model

        data = _fixture_data(monthly)
        attach_covariates(data, "contact", monthly)
        names = {v.name for v in build_model(
            data, ModelOptions(ability_walk=True, covariates="contact")).free_RVs}
        assert "sigma_step" in names
        assert {f"beta_cov_{f}" for f in CONTACT_COVARIATES} <= names

    def test_asking_for_covariates_without_a_design_matrix_raises(self, monthly):
        from src.models.pa_rate import ModelOptions, build_model

        data = _fixture_data(monthly)
        with pytest.raises(KeyError, match="carries no 'cov_x'"):
            build_model(data, ModelOptions(covariates="contact"))

    def test_a_design_matrix_that_names_other_aggregates_raises(self, monthly):
        from src.models.pa_rate import ModelOptions, build_model

        data = _fixture_data(monthly)
        attach_covariates(data, ("barrel", "ev90"), monthly)
        with pytest.raises(ValueError, match="but the options ask for"):
            build_model(data, ModelOptions(covariates="contact"))
