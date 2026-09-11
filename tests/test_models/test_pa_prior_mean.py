"""The prior's mean as a function of the profile (BAS-94).

Five things have to be true and none of them is checkable by reading the code:

1. **Off is off.** With `prior_mean_covariates=None` the graph carries no new
   free RV, `player_ability` keeps the shape it always had, and the data dict
   carries no `pm_x`. The bit-for-bit fixture in
   `tests/test_models/test_pa_rate.py` pins the numbers; this file pins the
   variable set, which is what says *why* the numbers held.
2. **On moves the prior's mean, by name.** The pre-registration names the new
   RVs `gamma_<feature>` and the dense sweep's `VARIANT_OWN_PARAMS` reads them
   by that name.
3. **The walk's innovations are unchanged.** Under `ability_walk` the prior
   mean has to enter every season's *level* and leave `sigma_step * z_step`
   alone — otherwise "the shrinkage target moves with the profile" would be
   indistinguishable from "the walk got a second scale". Checked at the graph
   level, at a pinned point, against the arithmetic written down in
   docs/bayes-prior-mean.md.
4. **Arm A never reads the cutoff's own season.** The leakage test pattern
   from `tests/test_models/test_pa_covariates.py`: every bucket from the
   cutoff month on is 1,000 batted balls of 120 mph barrels, so a single
   leaked row moves the feature enormously. For arm A deleting the *whole*
   current season must change `pm_x` by exactly nothing; for arm B deleting
   the months at or after the cutoff must.
5. **A thin current window moves arm B's target little.** That is the ballast
   claim the pre-registration asks to be described and held fixed, so it is
   measured rather than asserted.

The graph tests need pymc; the feature-array tests do not, and
`src/models/pa_prior_mean.py` imports neither pymc nor arviz for that reason.
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
from src.eval.contact import FEATURES  # noqa: E402
from src.models.pa_prior_mean import (  # noqa: E402
    CURRENT_WEIGHTS, GAMMA_SIGMA, PRIOR_MEAN_SETS, PRIOR_WEIGHTS, WHIFF,
    attach_prior_mean, prior_mean_array, prior_mean_features, prior_mean_set,
    prior_mean_summary, whiff_metrics, whiff_window,
)

HAS_PYMC = all(importlib.util.find_spec(m) is not None
               for m in ("pymc", "arviz"))
needs_pymc = pytest.mark.skipif(
    not HAS_PYMC, reason="MCMC deps (pymc/arviz) are not installed in CI")

SEASONS = (2023, 2024)
BATTERS = (101, 102)
ALL_SEASONS = (2022, 2023, 2024)


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
    """Two hitters with ordinary prior seasons and a monstrous current one.

    Every 2024 bucket — April included, so arm A's window is tested and not
    just arm B's month lag — is 1,000 batted balls of 120 mph barrels, which
    makes any leak visible without a tolerance.
    """
    rows = []
    for season in (2022, 2023):
        for month in (4, 5, 6, 7, 8, 9):
            rows.append(_bucket(101, season, month, 60, 86.0, 8.0,
                                barrel_frac=0.02, ev_bin=5))
            rows.append(_bucket(102, season, month, 60, 92.0, 16.0,
                                barrel_frac=0.09, ev_bin=9))
    for month in (4, 5, 6, 7, 8, 9):
        for player in BATTERS:
            rows.append(_bucket(player, 2024, month, 1000, 120.0, 25.0,
                                barrel_frac=1.0,
                                ev_bin=len(EV_BIN_COLUMNS) - 1))
    return pd.DataFrame(rows)


@pytest.fixture
def swing():
    """The swing-decisions artifact's shape, same monstrous-current design."""
    rows = []
    for season in (2022, 2023):
        for month in (4, 5, 6, 7, 8, 9):
            rows.append({"batter": 101, "season": season, "month": month,
                         "pitches_z": 200, "pitches_o": 200, "swings_z": 120,
                         "swings_o": 40, "contact_z": 100, "contact_o": 25,
                         "whiff_z": 20, "whiff_o": 15, "called_z": 40})
            rows.append({"batter": 102, "season": season, "month": month,
                         "pitches_z": 200, "pitches_o": 200, "swings_z": 130,
                         "swings_o": 60, "contact_z": 120, "contact_o": 55,
                         "whiff_z": 10, "whiff_o": 5, "called_z": 30})
    for month in (4, 5, 6, 7, 8, 9):
        for player in BATTERS:
            rows.append({"batter": player, "season": 2024, "month": month,
                         "pitches_z": 2000, "pitches_o": 2000,
                         "swings_z": 1000, "swings_o": 1000,
                         "contact_z": 0, "contact_o": 0,
                         "whiff_z": 1000, "whiff_o": 1000, "called_z": 0})
    return pd.DataFrame(rows)


# ─── the registry ───────────────────────────────────────────────────────────

class TestTheRegistry:
    def test_none_is_off(self):
        assert prior_mean_set(None) is None
        assert prior_mean_features(None) == ()
        assert prior_mean_features(None, "k_rate") == ()

    def test_an_unknown_set_raises_rather_than_fitting_something_else(self):
        with pytest.raises(ValueError, match="unknown prior-mean set"):
            prior_mean_set("barrels_only")

    def test_arm_a_reads_prior_seasons_and_arm_b_the_current_one_too(self):
        assert PRIOR_MEAN_SETS["contact"].weights == PRIOR_WEIGHTS
        assert PRIOR_MEAN_SETS["contact"].weights[0] == 0.0
        assert PRIOR_MEAN_SETS["contact_cur"].weights == CURRENT_WEIGHTS
        assert PRIOR_MEAN_SETS["contact_cur"].weights[0] == 1.0
        assert PRIOR_MEAN_SETS["contact"].arm == "A"
        assert PRIOR_MEAN_SETS["contact_cur"].arm == "B"

    def test_whiff_joins_the_features_for_k_rate_only(self):
        """The pre-registration adds whiff share for K%; a home-run rate gets
        the six batted-ball aggregates and no claim about swings."""
        for name in PRIOR_MEAN_SETS:
            assert prior_mean_features(name, "k_rate") == (*FEATURES, WHIFF)
            assert prior_mean_features(name, "hr_rate") == tuple(FEATURES)
            assert prior_mean_features(name, "bb_rate") == tuple(FEATURES)


# ─── the leakage guards ─────────────────────────────────────────────────────

def test_arm_a_ignores_the_whole_cutoff_season(monthly, swing):
    """Arm A's window is `s-1` and `s-2`; deleting every 2024 bucket — not
    just the ones at or after the cutoff month — must change nothing."""
    names = prior_mean_features("contact", "k_rate")
    before = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                              "2024-08-01", names, "contact")
    clean_c = monthly[monthly["season"] != 2024]
    clean_s = swing[swing["season"] != 2024]
    after = prior_mean_array(BATTERS, ALL_SEASONS, clean_c, clean_s,
                             "2024-08-01", names, "contact")
    np.testing.assert_array_equal(before, after)


def test_arm_a_is_the_same_feature_at_every_cutoff_in_a_season(monthly, swing):
    """The other half of the same claim, and the reason arm A is the primary:
    its shrinkage target does not move between April and August, so a thin
    current window cannot attenuate the coefficient that reads it."""
    names = prior_mean_features("contact", "k_rate")
    april = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                             "2024-04-01", names, "contact")
    august = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                              "2024-08-01", names, "contact")
    np.testing.assert_array_equal(april, august)


def test_arm_b_excludes_the_cutoff_month_bit_for_bit(monthly, swing):
    """Arm B *does* read the current season, up to the last month boundary on
    or before the cutoff and not one bucket further."""
    names = prior_mean_features("contact_cur", "k_rate")
    before = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                              "2024-06-15", names, "contact_cur")
    clean_c = monthly[~((monthly["season"] == 2024) & (monthly["month"] >= 6))]
    clean_s = swing[~((swing["season"] == 2024) & (swing["month"] >= 6))]
    after = prior_mean_array(BATTERS, ALL_SEASONS, clean_c, clean_s,
                             "2024-06-15", names, "contact_cur")
    np.testing.assert_array_equal(before, after)


def test_a_leaked_month_would_have_moved_arm_b(monthly, swing):
    """The control on the test above: the monstrous buckets are not inert."""
    names = prior_mean_features("contact_cur", "hr_rate")
    lagged = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                              "2024-05-01", names, "contact_cur")
    leaked = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                              "2024-09-01", names, "contact_cur")
    assert not np.allclose(lagged[:, 2, :], leaked[:, 2, :])


def test_a_cutoff_that_is_not_a_month_boundary_is_refused_for_whiff(swing):
    """`window_counts` refuses one for the contact half; the whiff window has
    to refuse it too, or the guard has a hole exactly where the new feature
    is."""
    with pytest.raises(ValueError, match="monthly"):
        whiff_window(swing, "2024-06-15", 2024, CURRENT_WEIGHTS)


# ─── the whiff feature itself ───────────────────────────────────────────────

class TestWhiff:
    def test_the_window_sums_zone_and_out_of_zone(self, swing):
        counts = whiff_window(swing, "2024-01-01", 2024, PRIOR_WEIGHTS)
        row = counts[counts["player"] == 101].iloc[0]
        # 2022 and 2023, six months each, 120 + 40 swings and 20 + 15 whiffs.
        assert row["swings"] == pytest.approx(2 * 6 * 160)
        assert row["whiffs"] == pytest.approx(2 * 6 * 35)

    def test_the_shrunk_share_sits_between_the_player_and_the_league(self, swing):
        counts = whiff_window(swing, "2024-01-01", 2024, PRIOR_WEIGHTS)
        m = whiff_metrics(counts).set_index("player")
        league = counts["whiffs"].sum() / counts["swings"].sum()
        raw_101 = (counts.set_index("player").loc[101, "whiffs"]
                   / counts.set_index("player").loc[101, "swings"])
        assert raw_101 > league > m.loc[102, WHIFF]
        assert raw_101 > m.loc[101, WHIFF] > league

    def test_the_whiff_column_lands_in_the_design_matrix(self, monthly, swing):
        names = prior_mean_features("contact", "k_rate")
        x = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                             "2024-08-01", names, "contact")
        j = names.index(WHIFF)
        # 101 whiffs more than 102, so his z-score is the positive one, and
        # the two are on opposite sides of a weighted zero.
        assert x[0, 2, j] > 0 > x[1, 2, j]

    def test_a_whiff_feature_without_the_artifact_raises(self, monthly):
        names = prior_mean_features("contact", "k_rate")
        with pytest.raises(ValueError, match="swing-decisions"):
            prior_mean_array(BATTERS, ALL_SEASONS, monthly, None,
                             "2024-08-01", names, "contact")


# ─── the shape of the matrix, and the rookie ────────────────────────────────

class TestTheMatrix:
    def test_shape_and_column_order(self, monthly, swing):
        names = prior_mean_features("contact", "k_rate")
        x = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                             "2024-08-01", names, "contact")
        assert x.shape == (len(BATTERS), len(ALL_SEASONS), len(names))
        assert names == (*FEATURES, WHIFF)

    def test_a_batter_with_no_profile_sits_at_the_league_mean(self, monthly,
                                                              swing):
        """Zero after standardization *is* the league mean for that season —
        the same answer `src.models.pa_covariates` gives, and the same one the
        old model gives a rookie."""
        names = prior_mean_features("contact", "hr_rate")
        batters = (*BATTERS, 999)
        x = prior_mean_array(batters, ALL_SEASONS, monthly, swing,
                             "2024-08-01", names, "contact")
        np.testing.assert_array_equal(x[2], np.zeros((len(ALL_SEASONS),
                                                      len(names))))

    def test_the_earliest_season_has_no_prior_window_here(self, monthly, swing):
        """2022's own prior seasons (2020, 2021) are not in this fixture, so
        every batter sits at the league mean there — the honest answer, and
        the one that keeps a short model window from inventing a profile."""
        names = prior_mean_features("contact", "hr_rate")
        x = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                             "2024-08-01", names, "contact")
        np.testing.assert_array_equal(x[:, 0, :], 0.0)
        assert np.abs(x[:, 2, :]).sum() > 0


def test_arm_b_is_ballasted_by_the_prior_profile(monthly, swing):
    """The pre-registration asks for the ballast to be described and fixed;
    this measures it. A one-month current window against two full prior
    seasons moves the target a fraction of what a six-month one does, with
    nothing tuned — the weights are (1, 1, 1) and the exposure does the rest.
    """
    names = prior_mean_features("contact_cur", "hr_rate")
    arm_a = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                             "2024-05-01", names, "contact")
    thin = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                            "2024-05-01", names, "contact_cur")
    thick = prior_mean_array(BATTERS, ALL_SEASONS, monthly, swing,
                             "2024-09-01", names, "contact_cur")
    # Both batters' 2024 contact is identical in the fixture, so the feature
    # moves toward a common value as the current window grows; what is
    # measured here is how far, not in which direction.
    moved_thin = float(np.abs(thin[:, 2, :] - arm_a[:, 2, :]).mean())
    moved_thick = float(np.abs(thick[:, 2, :] - arm_a[:, 2, :]).mean())
    assert moved_thick > moved_thin
    assert moved_thin < 0.5 * moved_thick


# ─── attach_prior_mean ──────────────────────────────────────────────────────

class TestAttach:
    def _data(self):
        return {"batters": np.array(BATTERS), "seasons": np.array(ALL_SEASONS),
                "component": "k_rate", "cutoff_date": "2024-08-01"}

    def test_off_attaches_nothing(self, monthly, swing):
        data = self._data()
        attach_prior_mean(data, None, monthly, swing)
        assert "pm_x" not in data and "pm_names" not in data

    def test_on_attaches_x_names_and_the_set(self, monthly, swing):
        data = self._data()
        attach_prior_mean(data, "contact", monthly, swing)
        assert data["pm_set"] == "contact"
        assert data["pm_names"] == (*FEATURES, WHIFF)
        assert data["pm_x"].shape == (2, 3, 7)

    def test_no_cutoff_is_an_error_not_a_full_season_read(self, monthly, swing):
        data = self._data()
        del data["cutoff_date"]
        with pytest.raises(ValueError, match="cutoff"):
            attach_prior_mean(data, "contact", monthly, swing)


# ─── the fit record's vacuity numbers ───────────────────────────────────────

class _FakePosterior(dict):
    pass


class _FakeTrace:
    def __init__(self, posterior):
        self.posterior = posterior


class _Var:
    def __init__(self, values):
        self.values = np.asarray(values, dtype="float64")


def test_prior_mean_summary_reports_gammas_and_the_sd_ratio():
    """Prediction 1 is two claims — a coefficient excludes zero, and the
    prior mean's between-player spread is at least 30% of `sigma_ability` —
    and this is where the second one is computed. Needs no pymc: a plain
    object with a `.posterior` mapping is the whole contract.
    """
    n_draws, n_b, n_s = 40, 6, 3
    rng = np.random.default_rng(0)
    effect = rng.standard_normal((2, n_draws, n_b, n_s)) * 0.1
    effect[..., -1] *= 3.0                      # the last season is the wide one
    posterior = _FakePosterior({
        "gamma_barrel": _Var(rng.standard_normal((2, n_draws)) * 0.01 + 0.3),
        "gamma_ev_mean": _Var(rng.standard_normal((2, n_draws)) * 0.01),
        "prior_mean_effect": _Var(effect),
        "sigma_ability": _Var(np.full((2, n_draws), 0.4)),
    })
    data = {"pm_set": "contact", "pm_names": ("barrel", "ev_mean")}
    out = prior_mean_summary(_FakeTrace(posterior), data)

    assert out["set"] == "contact"
    assert out["gamma"]["gamma_barrel"]["excludes_zero"] is True
    assert out["gamma"]["gamma_ev_mean"]["excludes_zero"] is False
    # Read off the LAST fitted season, which is the one a projection uses.
    expected_sd = float(np.mean(
        effect.reshape(-1, n_b, n_s)[:, :, -1].std(axis=1)))
    assert out["prior_mean_sd_between_players"] == pytest.approx(expected_sd)
    assert out["sigma_ability"] == pytest.approx(0.4)
    assert out["sd_ratio"] == pytest.approx(expected_sd / 0.4)
    assert out["sd_ratio_q05"] <= out["sd_ratio"] <= out["sd_ratio_q95"]


def test_prior_mean_summary_survives_a_trace_without_the_block():
    out = prior_mean_summary(None, {"pm_set": "contact", "pm_names": ("barrel",)})
    assert out["set"] == "contact" and "gamma" not in out


# ─── the graph ──────────────────────────────────────────────────────────────

def _cell_data(n_batters=6, n_seasons=4, n_teams=2, n_obs=60, seed=0,
               pm_names=None):
    rng = np.random.default_rng(seed)
    data = {
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
    if pm_names:
        data["pm_set"] = "contact"
        data["pm_names"] = tuple(pm_names)
        data["pm_x"] = rng.standard_normal(
            (n_batters, n_seasons, len(pm_names)))
    return data


@needs_pymc
class TestTheGraph:
    def test_off_is_the_model_it_always_was(self):
        from src.models.pa_rate import ModelOptions, build_model

        model = build_model(_cell_data(), ModelOptions())
        names = {v.name for v in model.free_RVs}
        assert not any(n.startswith("gamma_") for n in names)
        assert "prior_mean_effect" not in model.named_vars
        assert "pm_x" not in model.named_vars
        assert model.named_vars_to_dims["player_ability"] == ("batter",)

    def test_on_declares_one_gamma_per_feature(self):
        from src.models.pa_rate import ModelOptions, build_model

        pm_names = ("ev_mean", "barrel", WHIFF)
        model = build_model(_cell_data(pm_names=pm_names),
                            ModelOptions(ability_walk=True,
                                         prior_mean_covariates="contact"))
        names = {v.name for v in model.free_RVs}
        assert {"gamma_ev_mean", "gamma_barrel", "gamma_whiff"} <= names
        assert model.named_vars_to_dims["prior_mean_effect"] == ("batter", "season")
        assert model.named_vars_to_dims["player_ability"] == ("batter", "season")

    def test_the_prior_scale_is_the_pre_registered_one(self):
        from src.models.pa_rate import ModelOptions, build_model

        model = build_model(_cell_data(pm_names=("barrel",)),
                            ModelOptions(prior_mean_covariates="contact"))
        gamma = model["gamma_barrel"]
        mu, sd = (float(i.eval()) for i in gamma.owner.inputs[-2:])
        assert mu == pytest.approx(0.0)
        assert sd == pytest.approx(GAMMA_SIGMA)

    def test_the_flat_arm_gains_a_season_axis(self):
        """Without the walk the level is still constant in time, but the
        *target* is not — so `player_ability` has to carry a season axis, and
        `generate_projections` reads its last column unchanged."""
        from src.models.pa_rate import ModelOptions, build_model

        model = build_model(_cell_data(pm_names=("barrel",)),
                            ModelOptions(prior_mean_covariates="contact"))
        assert model.named_vars_to_dims["player_ability"] == ("batter", "season")
        assert "sigma_step" not in {v.name for v in model.free_RVs}

    def test_a_missing_design_matrix_raises(self):
        from src.models.pa_rate import ModelOptions, build_model

        with pytest.raises(KeyError, match="pm_x"):
            build_model(_cell_data(), ModelOptions(prior_mean_covariates="contact"))

    def test_a_design_matrix_from_the_other_arm_raises(self):
        from src.models.pa_rate import ModelOptions, build_model

        data = _cell_data(pm_names=("barrel",))
        data["pm_set"] = "contact_cur"
        with pytest.raises(ValueError, match="different windows"):
            build_model(data, ModelOptions(prior_mean_covariates="contact"))

    def test_a_wrong_shaped_design_matrix_raises(self):
        from src.models.pa_rate import ModelOptions, build_model

        data = _cell_data(pm_names=("barrel", "ev_mean"))
        data["pm_names"] = ("barrel",)
        with pytest.raises(ValueError, match="pm_x has shape"):
            build_model(data, ModelOptions(prior_mean_covariates="contact"))


@needs_pymc
def test_the_prior_mean_enters_every_level_and_leaves_the_walk_alone():
    """The arithmetic docs/bayes-prior-mean.md writes down, at the graph level:

        ability[b, s] = mu + gamma . x[b, s] + sigma * z[b]
                      + cumsum(sigma_step * z_step)[b, s-1]

    Two claims in one evaluation. First, the prior mean is added to *every*
    season's level (not only season 0), which is what makes the shrinkage
    target follow the measured profile year to year. Second, the innovations
    are untouched: the season-to-season *differences* of `player_ability`
    minus the profile's own differences are exactly `sigma_step * z_step`, so
    the walk still means what it meant before the block existed.

    Evaluated at a pinned point via `replace_rvs_by_values`, the same way
    `test_pa_k_rate_options.py` pins the `sigma_step -> 0` reduction — a
    fitted posterior could only make these *close*.
    """
    from src.models.pa_rate import ModelOptions, build_model

    pm_names = ("ev_mean", "barrel")
    data = _cell_data(n_batters=5, n_seasons=4, seed=17, pm_names=pm_names)
    model = build_model(data, ModelOptions(ability_walk=True,
                                           prior_mean_covariates="contact"))

    point = model.initial_point()
    rng = np.random.default_rng(3)
    for name, value in list(point.items()):
        point[name] = (np.asarray(value, dtype="float64")
                       + 0.3 * rng.standard_normal(np.shape(value)))

    [graph] = model.replace_rvs_by_values([model["player_ability"]])
    f = model.compile_fn(graph, inputs=model.value_vars, point_fn=True,
                         on_unused_input="ignore")
    ability = f(point)

    mu = point["mu_ability"]
    sigma = np.exp(point["sigma_ability_log__"])
    sigma_step = np.exp(point["sigma_step_log__"])
    gamma = np.array([point[f"gamma_{n}"] for n in pm_names])
    profile = (data["pm_x"] * gamma).sum(axis=-1)          # (batter, season)
    base = mu + sigma * point["z_ability"]
    steps = np.cumsum(sigma_step * point["z_step"], axis=1)

    expected = base[:, None] + profile
    expected[:, 1:] += steps
    np.testing.assert_allclose(ability, expected, rtol=0, atol=1e-10)

    # The innovations, recovered: differences net of the profile's own.
    innovations = np.diff(ability - profile, axis=1)
    np.testing.assert_allclose(innovations,
                               sigma_step * point["z_step"], rtol=0, atol=1e-10)


@needs_pymc
def test_a_zero_profile_is_exactly_the_model_without_the_block():
    """`x = 0` is the league mean, so a design matrix of zeros has to give
    back the old `player_ability` bit for bit — which is what makes "a rookie
    gets the league mean, as today" a checked claim rather than a hope."""
    from src.models.pa_rate import ModelOptions, build_model

    data = _cell_data(n_batters=5, n_seasons=4, seed=23, pm_names=("barrel",))
    data["pm_x"] = np.zeros_like(data["pm_x"])
    model = build_model(data, ModelOptions(ability_walk=True,
                                           prior_mean_covariates="contact"))
    plain = build_model(data, ModelOptions(ability_walk=True))

    point = model.initial_point()
    rng = np.random.default_rng(5)
    for name, value in list(point.items()):
        point[name] = (np.asarray(value, dtype="float64")
                       + 0.3 * rng.standard_normal(np.shape(value)))

    def _ability(m, pt):
        [graph] = m.replace_rvs_by_values([m["player_ability"]])
        fn = m.compile_fn(graph, inputs=m.value_vars, point_fn=True,
                          on_unused_input="ignore")
        return fn(pt)

    plain_point = {k: v for k, v in point.items()
                   if not k.startswith("gamma_")}
    np.testing.assert_array_equal(_ability(model, point),
                                  _ability(plain, plain_point))
