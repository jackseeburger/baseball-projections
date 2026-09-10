"""The joint multi-component model (BAS-84, docs/bayes-joint.md).

`src/models/pa_joint.py` fits K%, BB% and HR/PA in one PyMC model with a
per-batter ability *vector* under an LKJ(2) correlation prior. Three claims
have to hold for that to be the same model rather than a different one, and
this file checks all three without sampling:

1. **It is the same cells.** `prepare_joint_data` runs
   `pa_rate.prepare_model_data` per component and proves the cell keys, ages
   and trial counts came back identical — so the joint likelihood is the
   three single-component likelihoods on one table, not a re-derivation.
2. **The graph is the single-component graph with a component axis.** Every
   variable a `pa_rate` fit declares is declared here with the same name and
   a `component` dim, which is what lets `component_posterior` hand a
   one-component view to `pa_rate.generate_projections` unchanged.
3. **The log-probability is finite** at the initial point, for both variants,
   on a tiny synthetic dataset.

Building a PyMC graph needs pymc (and `component_posterior` needs xarray via
the trace), so everything here is marked `needs_pymc` the same way
`tests/test_models/test_pa_rate.py` is: `requirements-ci.txt` leaves pymc and
arviz out on purpose, and nothing in this file may run without them. No
sampling anywhere — the graph, the shapes and one `logp` evaluation are what
is being pinned, and MCMC belongs on the grid, not in the test suite.
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

COMPONENTS = ("k_rate", "bb_rate", "hr_rate")


def _pa_rows(n_batters=12, seasons=(2023, 2024, 2025), seed=7):
    """Synthetic PA rows carrying all three numerators, with K% and HR/PA
    deliberately correlated across batters (a shared latent "power" term) so
    the fixture is not one where a correlation could only be zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batters):
        power = rng.normal()
        k_rate = 1 / (1 + np.exp(-(-1.2 + 0.3 * power)))
        hr_rate = 1 / (1 + np.exp(-(-3.4 + 0.4 * power)))
        bb_rate = 1 / (1 + np.exp(-(-2.4 + rng.normal(0, 0.3))))
        for s in seasons:
            for _ in range(int(rng.integers(60, 90))):
                rows.append({
                    "batter": 1000 + b,
                    "game_year": s,
                    "game_date": pd.Timestamp(f"{s}-06-15"),
                    "stand": "R" if b % 2 == 0 else "L",
                    "is_k": int(rng.random() < k_rate),
                    "is_bb": int(rng.random() < bb_rate),
                    "is_hr": int(rng.random() < hr_rate),
                    "home_team": "HOU" if b % 3 else "SEA",
                    "away_team": "SEA" if b % 3 else "HOU",
                    "inning_topbot": "Top",
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def joint_data():
    from src.models.pa_joint import prepare_joint_data

    return prepare_joint_data(_pa_rows(), None, min_pa=1, components=COMPONENTS)


# --- 1. one set of cells ---------------------------------------------------

class TestJointDataIsOneCellTable:
    def test_every_component_lands_on_the_same_cells(self, joint_data):
        first = joint_data.component_data("k_rate")
        for name in COMPONENTS[1:]:
            other = joint_data.component_data(name)
            for key in ("batter_idx", "season_idx", "team_idx", "stand_idx",
                        "age_centered", "n_trials"):
                np.testing.assert_array_equal(first[key], other[key])

    def test_the_stacked_successes_are_each_components_own(self, joint_data):
        for j, name in enumerate(COMPONENTS):
            np.testing.assert_array_equal(
                joint_data.k[:, j], joint_data.component_data(name)["k"])

    def test_successes_never_exceed_the_shared_trials(self, joint_data):
        n = joint_data.shared["n_trials"]
        assert (joint_data.k <= n[:, None]).all()

    def test_each_component_keeps_its_own_league_prior(self, joint_data):
        # K% pins -1.27; the other two are read off the earliest season, and
        # HR/PA's is far below BB%'s because the rate is ~3% against ~8.5%.
        mus = dict(zip(joint_data.components, joint_data.league_init_mu))
        assert mus["k_rate"] == -1.27
        assert mus["hr_rate"] < mus["bb_rate"] < mus["k_rate"]

    def test_the_per_component_dict_is_exactly_what_pa_rate_would_build(self):
        from src.models.pa_joint import prepare_joint_data
        from src.models.pa_rate import prepare_model_data

        pa = _pa_rows()
        joint = prepare_joint_data(pa, None, min_pa=1, components=COMPONENTS)
        alone = prepare_model_data(pa, None, min_pa=1, component="hr_rate")
        via_joint = joint.component_data("hr_rate")
        assert via_joint["component"] == alone["component"] == "hr_rate"
        assert via_joint["league_init_mu"] == alone["league_init_mu"]
        np.testing.assert_array_equal(via_joint["k"], alone["k"])
        np.testing.assert_array_equal(via_joint["n_trials"], alone["n_trials"])

    def test_a_single_component_joint_model_is_refused(self):
        from src.models.pa_joint import prepare_joint_data

        with pytest.raises(ValueError, match="fewer than two"):
            prepare_joint_data(_pa_rows(), None, min_pa=1, components=("k_rate",))

    def test_an_unknown_component_is_refused(self):
        from src.models.pa_joint import prepare_joint_data

        with pytest.raises(ValueError, match="unknown joint component"):
            prepare_joint_data(_pa_rows(), None, min_pa=1,
                               components=("k_rate", "babip"))

    def test_mismatched_cells_raise_rather_than_pairing_the_wrong_batters(self):
        """The assertion that makes claim 1 a check and not a comment."""
        from src.models import pa_joint

        real = pa_joint.prepare_model_data
        calls = {"n": 0}

        def shifted(*args, **kwargs):
            out = real(*args, **kwargs)
            calls["n"] += 1
            if calls["n"] == 2:      # the second component's cells, rolled
                out = dict(out)
                out["batter_idx"] = np.roll(out["batter_idx"], 1)
            return out

        pa_joint.prepare_model_data = shifted
        try:
            with pytest.raises(AssertionError, match="did not produce the same cells"):
                pa_joint.prepare_joint_data(_pa_rows(), None, min_pa=1,
                                            components=COMPONENTS)
        finally:
            pa_joint.prepare_model_data = real


# --- 2. the graph ----------------------------------------------------------

def _named(model):
    return {v.name for v in model.free_RVs} | {d.name for d in model.deterministics}


class TestJointGraph:
    @pytest.mark.parametrize("walk", [False, True])
    def test_shapes_are_the_single_component_shapes_plus_a_component_axis(
            self, joint_data, walk):
        from src.models.pa_joint import build_joint_model
        from src.models.pa_rate import ModelOptions

        model = build_joint_model(joint_data, ModelOptions(ability_walk=walk))
        K, B = len(COMPONENTS), joint_data.shared["n_batters"]
        S = joint_data.shared["n_seasons"]
        # Untransformed variables keep their own name in the initial point,
        # which is all four of the ones pinned here.
        point = model.initial_point()
        assert point["z_ability"].shape == (B, K)
        assert point["league_innovations"].shape == (S, K)
        assert point["beta_hand"].shape == (K,)
        if walk:
            assert point["z_step"].shape == (B, S - 1, K)
        # And the ability the linear predictor reads is (batter, season?, K).
        ability = model["player_ability"].eval().shape
        assert ability == ((B, S, K) if walk else (B, K))

    def test_every_component_gets_its_own_binomial_on_its_own_successes(
            self, joint_data):
        from src.models.pa_joint import build_joint_model
        from src.models.pa_components import get_component

        model = build_joint_model(joint_data)
        observed = {v.name for v in model.observed_RVs}
        assert observed == {get_component(c).obs_name for c in COMPONENTS}
        for j, name in enumerate(COMPONENTS):
            rv = next(v for v in model.observed_RVs
                      if v.name == get_component(name).obs_name)
            np.testing.assert_array_equal(
                model.rvs_to_values[rv].data, joint_data.k[:, j])

    def test_the_correlation_matrix_does_not_repeat_a_dimension_name(self, joint_data):
        """xarray cannot broadcast a variable whose two dims share a name: it
        warns at construction and then fails on the first `.sel`, which is
        how the first smoke run lost a whole cell (`broadcasting cannot
        handle duplicate dimensions`) after a clean 208s fit. The square
        correlation matrix is the only variable here with two component
        axes, so its second axis gets its own coordinate over the same
        labels."""
        from src.models.pa_joint import build_joint_model

        model = build_joint_model(joint_data)
        dims = model.named_vars_to_dims["ability_corr"]
        assert len(set(dims)) == len(dims) == 2
        assert list(model.coords["component_2"]) == list(model.coords["component"])

    def test_the_correlation_prior_and_its_pairwise_readouts_exist(self, joint_data):
        from src.models.pa_joint import build_joint_model, corr_name

        names = _named(build_joint_model(joint_data))
        assert "ability_corr" in names
        assert "sigma_ability" in names       # the LKJ's stds, per component
        for a, b in (("k_rate", "bb_rate"), ("k_rate", "hr_rate"),
                     ("bb_rate", "hr_rate")):
            assert corr_name(a, b) in names

    def test_the_walk_declares_one_step_size_per_component(self, joint_data):
        from src.models.pa_joint import build_joint_model, sigma_step_name
        from src.models.pa_rate import ModelOptions

        names = _named(build_joint_model(joint_data,
                                         ModelOptions(ability_walk=True)))
        assert "sigma_step_component" in names
        for c in COMPONENTS:
            assert sigma_step_name(c) in names
        # Deliberately absent: a scalar `sigma_step` in a joint trace would be
        # read by the single-component tooling as one number for three
        # components. See VARIANT_OWN_PARAMS in the dense sweep.
        assert "sigma_step" not in names

    def test_the_flat_variant_has_no_step_size_at_all(self, joint_data):
        from src.models.pa_joint import build_joint_model

        names = _named(build_joint_model(joint_data))
        assert not any(n.startswith("sigma_step") for n in names)

    def test_the_age_curve_carries_each_components_own_direction(self, joint_data):
        from src.models.pa_joint import build_joint_model
        from src.models.pa_components import AGE_DIRECTION

        model = build_joint_model(joint_data)
        # The quadratic's curvature prior mean is -0.005 * age_direction, per
        # component — a U for K% (bigger is worse), an inverted U for the two
        # counting skills.
        mu = model["beta_age2"].owner.inputs[2].eval()
        expected = [-0.005 * AGE_DIRECTION[c] for c in COMPONENTS]
        np.testing.assert_allclose(np.asarray(mu).ravel(), expected)

    @pytest.mark.parametrize("walk,constrained",
                             [(False, False), (True, False), (True, True)])
    def test_the_log_probability_is_finite_at_the_initial_point(
            self, joint_data, walk, constrained):
        from src.models.pa_joint import build_joint_model
        from src.models.pa_rate import ModelOptions

        model = build_joint_model(
            joint_data, ModelOptions(ability_walk=walk,
                                     constrained_age=constrained))
        logp = model.compile_logp()(model.initial_point())
        assert np.isfinite(float(logp))

    def test_no_pitcher_term_anywhere(self, joint_data):
        from src.models.pa_joint import build_joint_model

        names = _named(build_joint_model(joint_data))
        assert not any("pitcher" in n for n in names)


# --- 3. reading one component back out -------------------------------------

class _FakeTrace:
    """A posterior with the joint model's names and dims, without sampling."""

    def __init__(self, walk=True, n_seasons=3, n_batters=4, chains=2, draws=5):
        import xarray as xr

        comps = list(COMPONENTS)
        rng = np.random.default_rng(0)
        c, d, K = chains, draws, len(comps)

        def da(shape, dims):
            return xr.DataArray(rng.normal(size=(c, d) + shape),
                                dims=("chain", "draw") + dims)

        data = {
            "league_trend": da((n_seasons, K), ("season", "component")),
            "league_innovations": da((n_seasons, K), ("season", "component")),
            "mu_ability": da((K,), ("component",)),
            "sigma_ability": da((K,), ("component",)),
            "beta_hand": da((K,), ("component",)),
            "beta_age": da((K,), ("component",)),
            "beta_age2": da((K,), ("component",)),
        }
        if walk:
            data["player_ability"] = da((n_batters, n_seasons, K),
                                        ("batter", "season", "component"))
            for name in comps:
                data[f"sigma_step_{name}"] = da((), ())
        else:
            data["player_ability"] = da((n_batters, K), ("batter", "component"))
        for a, b in (("k_rate", "bb_rate"), ("k_rate", "hr_rate"),
                     ("bb_rate", "hr_rate")):
            data[f"corr_{a}__{b}"] = da((), ())
        self.posterior = xr.Dataset(data).assign_coords(component=comps)


class TestComponentPosterior:
    @pytest.mark.parametrize("walk", [False, True])
    def test_the_view_drops_the_component_axis_and_keeps_the_names(self, walk):
        from src.models.pa_joint import component_posterior

        view = component_posterior(_FakeTrace(walk=walk), "hr_rate").posterior
        assert "component" not in view["league_trend"].dims
        assert view["mu_ability"].dims == ("chain", "draw")
        expected = ("chain", "draw", "batter", "season") if walk else (
            "chain", "draw", "batter")
        assert view["player_ability"].dims == expected
        assert ("sigma_step" in view) is walk

    def test_the_view_is_that_components_slice_and_not_another(self):
        from src.models.pa_joint import component_posterior

        trace = _FakeTrace()
        for j, name in enumerate(COMPONENTS):
            view = component_posterior(trace, name).posterior
            np.testing.assert_array_equal(
                view["beta_hand"].values,
                trace.posterior["beta_hand"].values[:, :, j])

    def test_the_view_is_what_the_projection_code_reads(self):
        """The names `pa_rate.generate_projections` and `_age_term_function`
        look up, all present in a one-component view — the check that keeps
        the joint arm's projections on the single-component code path."""
        from src.models.pa_joint import component_posterior

        view = component_posterior(_FakeTrace(), "k_rate").posterior
        for name in ("league_trend", "player_ability", "beta_hand",
                     "league_innovations", "sigma_step", "beta_age",
                     "beta_age2", "mu_ability", "sigma_ability"):
            assert name in view


class TestJointParamSummary:
    def test_reports_a_mean_and_a_95_interval_per_correlation_and_step(self):
        from src.models.pa_joint import joint_param_summary

        out = joint_param_summary(_FakeTrace(), COMPONENTS)
        assert set(out) == {
            "corr_k_rate__bb_rate", "corr_k_rate__hr_rate",
            "corr_bb_rate__hr_rate",
            "sigma_step_k_rate", "sigma_step_bb_rate", "sigma_step_hr_rate",
        }
        for stats in out.values():
            assert set(stats) == {"mean", "sd", "q2.5", "q97.5"}
            assert stats["q2.5"] <= stats["mean"] <= stats["q97.5"]

    def test_a_flat_joint_fit_reports_correlations_and_no_step_sizes(self):
        from src.models.pa_joint import joint_param_summary

        out = joint_param_summary(_FakeTrace(walk=False), COMPONENTS)
        assert all(k.startswith("corr_") for k in out)

    def test_no_trace_is_an_empty_summary_not_a_crash(self):
        from src.models.pa_joint import joint_param_summary

        assert joint_param_summary(None) == {}
