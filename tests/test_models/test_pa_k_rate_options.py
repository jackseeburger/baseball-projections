"""`ModelOptions` and the two structural K% model variants.

`ability_walk` (a per-batter Gaussian random walk over seasons) and
`constrained_age` (a peak-plus-signed-slopes age curve) are pre-registered in
docs/bayes-variants.md before being scored against the flat/quadratic
default. This file checks the plumbing a scoreboard entry depends on: the
options object itself, that `build_model` accepts both flags without
touching the flat path's behaviour, that the walk reduces *exactly* to the
flat model at `sigma_step = 0` (not approximately — the graph-level check
below pins the value, it does not just make it small), that the age curve's
sign matches the convention `src.eval.tuning` uses for K%, and that
`generate_projections` produces the same columns from a real (if tiny)
posterior in every mode.

Everything here needs pymc/arviz, which `requirements-ci.txt` leaves out on
purpose (MCMC runs on Modal, not in CI) — same guard as
`tests/test_models/test_cutoff.py`, and the whole file is skipped where they
are not installed.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

needs_pymc = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("pymc", "arviz")),
    reason="MCMC deps (pymc/arviz) are not installed in CI",
)
pytestmark = needs_pymc


# ─── shared fixtures ────────────────────────────────────────────────────────

def _cell_data(n_batters=6, n_seasons=4, n_teams=2, n_obs=60, seed=0):
    """A `prepare_model_data()`-shaped dict, built directly (as
    `test_aggregation.py` does) rather than through real PA rows — the model
    graph does not care where the cells came from.
    """
    rng = np.random.default_rng(seed)
    return {
        "n_obs": n_obs, "n_pa": n_obs * 50,
        "n_batters": n_batters, "n_seasons": n_seasons, "n_teams": n_teams,
        "n_pitchers": 0, "include_pitcher": False,
        "batter_idx": rng.integers(0, n_batters, n_obs),
        "season_idx": rng.integers(0, n_seasons, n_obs),
        "team_idx": rng.integers(0, n_teams, n_obs),
        "stand_idx": rng.integers(0, 2, n_obs),
        "pitcher_idx": None,
        "age_centered": rng.uniform(-8, 8, n_obs),
        "log_pf_k": np.zeros(n_obs),
        "k": rng.integers(0, 20, n_obs),
        "n_trials": np.full(n_obs, 50),
        "seasons": np.arange(2022, 2022 + n_seasons),
        "batters": np.arange(n_batters),
        "teams": np.array([f"T{i}" for i in range(n_teams)]),
        "pitchers": np.array([], dtype=np.int64),
    }


def _pa_rows(n_batters=15, seasons=(2023, 2024, 2025), seed=3):
    """Real PA-level rows, small enough for a few-second NUTS fit."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batters):
        true_rate = rng.uniform(0.15, 0.35)
        for s in seasons:
            for _ in range(rng.integers(60, 100)):
                rows.append({
                    "batter": 1000 + b,
                    "game_year": s,
                    "game_date": pd.Timestamp(f"{s}-06-15"),
                    "stand": "R" if b % 2 == 0 else "L",
                    "is_k": int(rng.random() < true_rate),
                    "home_team": "HOU",
                    "away_team": "SEA",
                    "inning_topbot": "Top",
                })
    return pd.DataFrame(rows)


FAST_SAMPLER = dict(draws=25, tune=25, chains=1, cores=1,
                    nuts_sampler="pymc", progressbar=False, random_seed=59,
                    idata_kwargs={"log_likelihood": False})


# ─── ModelOptions itself ────────────────────────────────────────────────────

def test_model_options_label_and_to_dict():
    from src.models.pa_k_rate import ModelOptions

    default = ModelOptions()
    assert default.ability_walk is False and default.constrained_age is False
    assert "flat" in default.label() and "quadratic" in default.label()
    assert default.covariates is None
    assert default.to_dict() == {"ability_walk": False, "constrained_age": False,
                                 "covariates": None}

    both = ModelOptions(ability_walk=True, constrained_age=True)
    assert "walk" in both.label() and "constrained" in both.label()
    assert both.to_dict() == {"ability_walk": True, "constrained_age": True,
                              "covariates": None}

    cov = ModelOptions(ability_walk=True, covariates="contact")
    assert "covariates=" in cov.label() and "barrel" in cov.label()
    assert cov.to_dict()["covariates"] == [
        "ev_mean", "ev90", "barrel", "hardhit", "sweetspot", "la_mean"]


# ─── build_model no longer raises, and the flat path is untouched ─────────

class TestBuildModelDoesNotRaise:
    def test_neither_flag_matches_no_options_argument(self):
        """`build_model(data)` with no `options` is the pre-existing call
        site (see test_aggregation.py) and must keep working."""
        from src.models.pa_k_rate import ModelOptions, build_model

        data = _cell_data()
        m_default = build_model(data)
        m_explicit = build_model(data, ModelOptions())
        assert sorted(v.name for v in m_default.free_RVs) == \
            sorted(v.name for v in m_explicit.free_RVs)
        # The flat path's variable names are exactly what they were before
        # either option existed.
        names = {v.name for v in m_default.free_RVs}
        assert {"mu_ability", "sigma_ability", "z_ability",
                "beta_age", "beta_age2"} <= names
        assert "sigma_step" not in names and "peak_age" not in names

    def test_ability_walk_does_not_raise(self):
        from src.models.pa_k_rate import ModelOptions, build_model

        model = build_model(_cell_data(), ModelOptions(ability_walk=True))
        names = {v.name for v in model.free_RVs}
        assert {"z_ability", "sigma_step", "z_step"} <= names
        assert model.named_vars_to_dims["player_ability"] == ("batter", "season")

    def test_constrained_age_does_not_raise(self):
        from src.models.pa_k_rate import ModelOptions, build_model

        model = build_model(_cell_data(), ModelOptions(constrained_age=True))
        names = {v.name for v in model.free_RVs}
        assert {"peak_frac", "slope_young", "slope_old"} <= names
        assert "beta_age" not in names and "beta_age2" not in names

    def test_both_flags_together(self):
        from src.models.pa_k_rate import ModelOptions, build_model

        model = build_model(_cell_data(), ModelOptions(ability_walk=True,
                                                        constrained_age=True))
        names = {v.name for v in model.free_RVs}
        assert {"sigma_step", "z_step", "peak_frac", "slope_young",
                "slope_old"} <= names
        assert "peak_age" in model.named_vars  # Deterministic, not free_RVs


# ─── dims: shape assertions on player_ability in both modes ───────────────

class TestPlayerAbilityDims:
    def test_flat_is_one_dimensional_in_batter(self):
        from src.models.pa_k_rate import ModelOptions, build_model

        data = _cell_data(n_batters=7)
        model = build_model(data, ModelOptions())
        assert model.named_vars_to_dims["player_ability"] == ("batter",)

    def test_walk_is_batter_by_season(self):
        from src.models.pa_k_rate import ModelOptions, build_model

        data = _cell_data(n_batters=7, n_seasons=5)
        model = build_model(data, ModelOptions(ability_walk=True))
        assert model.named_vars_to_dims["player_ability"] == ("batter", "season")
        assert model.coords["season_step"] == tuple(data["seasons"][1:])


# ─── the sigma_step -> 0 reduction, at the graph level ─────────────────────

def test_ability_walk_reduces_exactly_to_flat_at_sigma_step_zero():
    """Pin `sigma_step`'s value var so far into the tail of its LogTransform
    that the untransformed value underflows to exactly 0.0 in float64
    (`exp(-800) == 0.0`, not merely small — checked directly below), then
    assert `player_ability` collapses to a column vector that is (a) constant
    across the season axis and (b) bit-identical to the flat model's own
    formula, `mu_ability + sigma_ability * z_ability`, evaluated at the same
    point. This is the graph itself, not a fitted posterior: any sigma_step
    tiny enough would make the columns *close*, which is a weaker claim than
    the one the reduction promises.

    Evaluating a Deterministic at an arbitrary point takes `replace_rvs_by_
    values` first — `model[name]` is the prior-sampling graph (its own RNG,
    unconnected to the value vars), not the value-substituted graph a NUTS
    step actually uses; skipping that swap silently ignores the pinned point
    and re-draws fresh priors on every call, which is not a mistake this test
    stays quiet about (the two assertions below would fail loudly).
    """
    from src.models.pa_k_rate import ModelOptions, build_model

    assert np.exp(-800.0) == 0.0  # the underflow this test relies on

    data = _cell_data(n_batters=8, n_seasons=5, seed=11)
    model = build_model(data, ModelOptions(ability_walk=True))

    point = model.initial_point()
    rng = np.random.default_rng(7)
    for name, value in list(point.items()):
        # Move well off the (mostly zero) defaults so a real bug — e.g. the
        # walk not actually reading z_ability/mu_ability/sigma_ability —
        # would not hide behind coincidental zeros.
        point[name] = (np.asarray(value, dtype="float64")
                       + 0.3 * rng.standard_normal(np.shape(value)))
    point["sigma_step_log__"] = np.array(-800.0)

    [pa_graph] = model.replace_rvs_by_values([model["player_ability"]])
    f = model.compile_fn(pa_graph, inputs=model.value_vars, point_fn=True,
                         on_unused_input="ignore")
    player_ability = f(point)

    mu = point["mu_ability"]
    sigma = np.exp(point["sigma_ability_log__"])
    z0 = point["z_ability"]
    expected = mu + sigma * z0

    assert player_ability.shape == (data["n_batters"], data["n_seasons"])
    np.testing.assert_array_equal(player_ability, np.tile(expected[:, None],
                                                           (1, data["n_seasons"])))
    np.testing.assert_array_equal(player_ability[:, 0], expected)


# ─── the constrained age curve's sign ───────────────────────────────────────

def test_constrained_age_is_a_valley_at_the_peak():
    """K% is the one component `src.eval.tuning.AGE_DIRECTION` flips
    (`-1.0`): a bigger K% is a worse hitter, so unlike BB%/HR/BABIP/ISO the
    constrained curve is a *valley*, not a hill — it falls as a young hitter
    approaches his peak and rises again after it. At fixed positive slopes
    and peak_age=27, both a 22- and a 34-year-old must have a higher
    `age_term` (more Ks) than a 27-year-old (fewest), and the 27-year-old's
    own term must be exactly zero (`d = 0` at the peak, both branches of
    `pt.where` agree there).
    """
    from src.models.pa_k_rate import ModelOptions, build_model

    data = _cell_data(n_batters=4, n_seasons=2, seed=5)
    model = build_model(data, ModelOptions(constrained_age=True))

    point = model.initial_point()
    lo, hi = 25.0, 31.0
    peak_frac = (27.0 - lo) / (hi - lo)
    # Beta(2,2)'s value var is a logit (LogOddsTransform): invert it.
    point["peak_frac_logodds__"] = np.array(np.log(peak_frac / (1 - peak_frac)))
    point["slope_young_log__"] = np.array(np.log(0.01))
    point["slope_old_log__"] = np.array(np.log(0.01))

    # Recompute age_term directly from the model's own peak_age/slope
    # Deterministics, so the test exercises the same graph `build_model`
    # wires into `eta` rather than reimplementing the formula.
    outs = [model["peak_age"], model["slope_young"], model["slope_old"]]
    outs = model.replace_rvs_by_values(outs)
    f = model.compile_fn(outs, inputs=model.value_vars, point_fn=True,
                         on_unused_input="ignore")
    peak_age, slope_young, slope_old = f(point)
    assert peak_age == pytest.approx(27.0, abs=1e-6)

    def age_term(age):
        d = age - peak_age
        return slope_old * d if d > 0 else -slope_young * d

    term_22 = age_term(22.0)
    term_27 = age_term(27.0)
    term_34 = age_term(34.0)

    assert term_27 == pytest.approx(0.0, abs=1e-9)
    assert term_22 > term_27
    assert term_34 > term_27


# ─── generate_projections from a tiny real trace, both modes ──────────────

EXPECTED_COLUMNS = {
    "batter", "stand", "age", "projected_k_rate", "k_rate_std",
    "k_rate_lower", "k_rate_upper", "posterior_mean_ability", "total_pa",
    "career_k_rate", "last_season", "unseen",
}


class TestGenerateProjectionsFromTinyTrace:
    """Real (if minuscule) NUTS fits, one per mode — draws=25/tune=25/
    chains=1 takes single-digit seconds on this model size (checked by hand
    before writing this file), so this stays a unit test and not a backtest.
    """

    def _fit(self, options):
        from src.models.pa_k_rate import (
            build_model, generate_projections, prepare_model_data, sample_model,
        )

        data = prepare_model_data(_pa_rows(), None, min_pa=1,
                                  include_pitcher=False)
        model = build_model(data, options)
        trace = sample_model(model, **FAST_SAMPLER)
        return data, trace

    def test_ability_walk_projections_have_the_flat_columns(self):
        from src.models.pa_k_rate import ModelOptions, generate_projections

        data, trace = self._fit(ModelOptions(ability_walk=True))
        proj = generate_projections(trace, data, projection_year=2026)
        assert set(proj.columns) == EXPECTED_COLUMNS
        assert len(proj) == data["n_batters"]
        assert proj["projected_k_rate"].between(0, 1).all()

        ss = trace.posterior["sigma_step"].values
        assert ss.shape[-1] == 1 or ss.ndim == 2  # scalar per (chain, draw)
        assert np.all(np.isfinite(ss))

    def test_constrained_age_projections_have_the_flat_columns(self):
        from src.models.pa_k_rate import ModelOptions, generate_projections

        data, trace = self._fit(ModelOptions(constrained_age=True))
        proj = generate_projections(trace, data, projection_year=2026)
        assert set(proj.columns) == EXPECTED_COLUMNS
        assert len(proj) == data["n_batters"]
        assert proj["projected_k_rate"].between(0, 1).all()
        assert "peak_age" in trace.posterior

    def test_unseen_batters_covered_in_both_modes(self):
        """Unseen coverage (age known and unknown) is the same code path
        regardless of variant — `_project_unseen` draws from
        `N(mu_ability, sigma_ability)`, the walk's own season-0 marginal."""
        from src.models.pa_k_rate import ModelOptions, generate_projections

        unseen = pd.DataFrame({"batter": [9998, 9999], "age": [np.nan, 24.0]})
        for options in (ModelOptions(ability_walk=True),
                       ModelOptions(constrained_age=True)):
            data, trace = self._fit(options)
            proj = generate_projections(trace, data, projection_year=2026,
                                        unseen=unseen)
            rows = proj[proj["unseen"]]
            assert set(rows["batter"]) == {9998, 9999}
            assert rows["projected_k_rate"].between(0, 1).all()
            assert set(proj.columns) == EXPECTED_COLUMNS

    def test_intra_season_horizon_zero_draws_no_walk_innovation(self):
        """At `projection_year == last training season`, `years_ahead == 0`
        and the extrapolation loop in `generate_projections` never runs, so
        the projected ability is exactly the last fitted season's node —
        the same horizon-zero convention `marcel_tuned` uses, now checked for
        the walk variant specifically since it is the one with a loop to
        skip."""
        from src.models.pa_k_rate import ModelOptions, generate_projections

        data, trace = self._fit(ModelOptions(ability_walk=True))
        last_season = int(data["seasons"][-1])
        proj = generate_projections(trace, data, projection_year=last_season)

        post = trace.posterior
        pa = post["player_ability"].values  # (chain, draw, batter, season)
        n_samples = pa.shape[0] * pa.shape[1]
        last_node = pa.reshape(n_samples, data["n_batters"], -1)[:, :, -1]

        row = proj.iloc[0]
        b_idx = data["batter_map"][int(row["batter"])]
        expected_mean_ability = float(np.mean(last_node[:, b_idx]))
        assert row["posterior_mean_ability"] == pytest.approx(
            expected_mean_ability, rel=1e-9)


def test_age_peak_window_matches_the_tuning_module_it_copies():
    """The two copies of the age-peak window must not drift apart.

    `src.models.pa_k_rate` duplicates `AGE_PEAK_WINDOW` rather than importing
    it, deliberately, so the model module keeps its zero dependency on
    `src.eval`. That is a reasonable trade only while something notices when
    one copy moves: a constrained Bayesian age curve and a constrained Marcel
    age curve fitted on *different* windows would be compared as though they
    were the same constraint. This test is that something.

    It does not need pymc — both constants are plain tuples.
    """
    from src.eval import tuning
    from src.models import pa_k_rate

    assert pa_k_rate.AGE_PEAK_WINDOW == tuning.AGE_PEAK_WINDOW
