"""The refit Bayesian arm's provider shape and its coverage bookkeeping.

Nothing here samples. What is worth testing without MCMC is the plumbing that
decides *what gets compared*: the arm has to be importable where pymc is not
installed (CI), it has to refuse components it does not model instead of
serving the K% number under another name, and it has to cover the batters the
baselines cover — because `common_players=True` means an arm with thin
coverage silently shrinks the player set every other arm is scored on, which
changes the comparison rather than making it fair.
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

from src.eval.backtest import COMPONENTS
from src.eval.bayes_arm import (
    BayesArmConfig, bayes_k_rate_provider, unseen_from_train,
)


def test_module_imports_without_mcmc_deps():
    """pymc is imported inside functions, so CI can import this module.

    requirements-ci.txt deliberately has no pymc; a module-scope import here
    would break the whole eval package's test collection.
    """
    import src.eval.bayes_arm as arm

    source = Path(arm.__file__).read_text()
    for line in source.splitlines():
        if line.startswith(("import ", "from ")) and not line.startswith("from __future__"):
            assert "pymc" not in line and "arviz" not in line, line
            assert "pa_k_rate" not in line, line


def test_config_label_states_the_scale():
    label = BayesArmConfig(draws=500, tune=400, chains=2, nuts_sampler="numpyro",
                           include_pitcher=True, max_batters=300).label()
    assert "2x500" in label and "tune 400" in label
    assert "numpyro" in label and "pitcher" in label and "300" in label
    assert "no-pitcher" in BayesArmConfig(include_pitcher=False).label()


def test_provider_refuses_components_this_model_cannot_serve():
    """BABIP and ISO are not per-PA binomials — BABIP's denominator is balls
    in play and ISO's numerator is not a count of trials at all — so they are
    separate models, and serving another component's number under their name
    would put a wrong number on the scoreboard. BB% and HR/PA are no longer
    in this list: since BAS-73 they are the same model with a different
    numerator."""
    provider = bayes_k_rate_provider("2026-07-01", 2026)
    train = pd.DataFrame({"batter": [1], "season": [2026], "pa": [100], "k": [20]})
    for name in ("iso", "babip"):
        with pytest.raises(ValueError, match="per-PA"):
            provider(train, COMPONENTS[name], 2026)


def test_provider_refuses_a_component_its_config_was_not_built_for():
    """One provider per component. A BB% spec reaching a K%-configured
    provider is a wiring bug, and it is invisible downstream — the frame it
    would return is well-formed and wrong."""
    provider = bayes_k_rate_provider("2026-07-01", 2026)
    train = pd.DataFrame({"batter": [1], "season": [2026], "pa": [100], "k": [20]})
    with pytest.raises(ValueError, match="built to fit 'k_rate'"):
        provider(train, COMPONENTS["bb_rate"], 2026)


def test_a_bb_rate_config_serves_bb_rate_and_names_its_column():
    """The arm's config decides the component, and the column it reads off
    the projection frame follows it — `projected_bb_rate`, never
    `projected_k_rate` with BB% numbers in it."""
    config = BayesArmConfig(component="bb_rate")
    assert config.rate_component().numerator == "is_bb"
    assert config.projected_col() == "projected_bb_rate"
    assert "bb_rate" in config.label()


def test_a_config_for_a_component_the_model_cannot_fit_raises_early():
    with pytest.raises(ValueError, match="unknown component"):
        BayesArmConfig(component="babip").rate_component()


def test_unseen_lists_batters_the_fit_missed_with_forward_aged_ages():
    train = pd.DataFrame({
        "batter": [1, 1, 2, 3, 3],
        "season": [2024, 2025, 2025, 2024, 2026],
        "age": [27.0, 28.0, 22.0, 31.0, 33.0],
    })
    unseen = unseen_from_train(train, fitted_batters=[1], predict_year=2026)
    assert list(unseen["batter"]) == [2, 3]
    # Batter 2's last training row is 2025 at 22, so 23 in 2026; batter 3 has a
    # 2026 row already, so his age is used as it stands.
    assert unseen.set_index("batter")["age"].to_dict() == {2: 23.0, 3: 33.0}


def test_unseen_is_empty_when_the_fit_covers_everyone():
    train = pd.DataFrame({"batter": [1, 2], "season": [2026, 2026],
                          "age": [27.0, 28.0]})
    assert unseen_from_train(train, [1, 2], 2026).empty


def test_unseen_survives_a_training_frame_with_no_ages():
    """Age is optional in the season frame; a missing age must not raise —
    the projection falls back to the population at the reference age."""
    train = pd.DataFrame({"batter": [1, 2], "season": [2026, 2026]})
    unseen = unseen_from_train(train, [1], 2026)
    assert list(unseen["batter"]) == [2]
    assert np.isnan(unseen["age"]).all()


# --- the joint arm (BAS-84) -------------------------------------------------

class TestJointConfig:
    """`joint=True` is a structure, not a component: one MCMC fit over K%,
    BB% and HR/PA, and `component` still says which of the three this arm
    serves. None of this needs pymc — `src.models.pa_joint` is imported
    inside the fit functions, like every other model import here."""

    def test_the_joint_flag_names_itself_in_the_variant_slug(self):
        assert BayesArmConfig(joint=True).variant() == "joint"
        assert BayesArmConfig(joint=True, ability_walk=True).variant() == (
            "joint+ability_walk")

    def test_the_default_arm_is_unchanged_by_the_new_flag(self):
        assert BayesArmConfig().joint is False
        assert BayesArmConfig().variant() == "flat"
        assert BayesArmConfig(ability_walk=True).variant() == "ability_walk"

    def test_the_label_says_a_joint_fit_is_joint(self):
        assert "(joint)" in BayesArmConfig(joint=True, component="hr_rate").label()
        assert "(joint)" not in BayesArmConfig(component="hr_rate").label()

    def test_the_joint_memo_key_is_blind_to_the_component_and_nothing_else(self):
        """Three components share one fit, so the key that decides "same fit"
        must differ in `component` alone — and must still separate two arms
        that differ in anything else."""
        k = BayesArmConfig(joint=True, component="k_rate")
        hr = BayesArmConfig(joint=True, component="hr_rate")
        assert k.joint_key() == hr.joint_key()
        assert k.joint_key() != BayesArmConfig(
            joint=True, component="k_rate", draws=17).joint_key()
        assert k.joint_key() != BayesArmConfig(
            joint=True, component="k_rate", ability_walk=True).joint_key()
        assert k.joint_key() != BayesArmConfig(
            joint=True, component="k_rate", seasons=(2024,)).joint_key()

    def test_a_joint_config_still_validates_its_component(self):
        with pytest.raises(ValueError):
            BayesArmConfig(joint=True, component="babip").rate_component()


class TestMeasurementConfig:
    """`measurement=True` is BAS-84's structure with the observation channels
    added (BAS-85, docs/bayes-measurement.md). It implies `joint` — the
    channels read a latent the joint graph writes — but over the two
    components the pre-registration names, not three. None of this needs
    pymc: `src.models.pa_measurement`'s graph builder is imported inside the
    fit functions like every other model import here.
    """

    def test_measurement_replaces_joint_in_the_variant_slug(self):
        """One structure, not two. "measurement+joint" would read as a joint
        arm that also has channels, and there is no such arm — the channels
        have nothing to load on without the joint graph."""
        cfg = BayesArmConfig(measurement=True, joint=True)
        assert cfg.variant() == "measurement"
        assert BayesArmConfig(measurement=True, joint=True,
                              ability_walk=True).variant() == (
            "measurement+ability_walk")

    def test_the_default_arm_is_unchanged_by_the_new_flag(self):
        assert BayesArmConfig().measurement is False
        assert BayesArmConfig().extra_quantiles == ()
        assert BayesArmConfig().variant() == "flat"
        assert BayesArmConfig(joint=True).variant() == "joint"

    def test_the_label_says_a_measurement_fit_is_one(self):
        cfg = BayesArmConfig(measurement=True, joint=True, component="hr_rate")
        assert "(measurement)" in cfg.label()
        assert "(joint)" not in cfg.label()

    def test_it_fits_the_two_components_the_pre_registration_names(self):
        """K% and HR/PA. BB% is not in scope — there is no whiff or barrel
        channel that loads on a walk-rate latent, and a fit that carried BB%
        anyway would pay for a third likelihood the ticket cannot read.

        No pymc: a measurement arm's component list must be readable in CI,
        which is why `joint_components` defers the `pa_joint` import to the
        joint branch.
        """
        from src.eval.bayes_arm import joint_components

        assert joint_components(BayesArmConfig(measurement=True, joint=True)) == (
            "k_rate", "hr_rate")

    @needs_pymc
    def test_a_plain_joint_arm_still_fits_three_components(self):
        """The joint list lives in `src.models.pa_joint`, which needs pymc."""
        from src.eval.bayes_arm import joint_components

        assert joint_components(BayesArmConfig(joint=True)) == (
            "k_rate", "bb_rate", "hr_rate")

    def test_the_memo_key_separates_a_measurement_fit_from_a_joint_one(self):
        """`joint_key` blanks `component` and nothing else, so a flag added
        to the config cannot be forgotten there — this is that check for
        `measurement`, whose fit covers different components entirely."""
        joint = BayesArmConfig(joint=True, component="k_rate")
        meas = BayesArmConfig(joint=True, measurement=True, component="k_rate")
        assert joint.joint_key() != meas.joint_key()
        assert meas.joint_key() == BayesArmConfig(
            joint=True, measurement=True, component="hr_rate").joint_key()

    def test_the_quantiles_a_coverage_check_needs_ride_on_the_config(self):
        """docs/bayes-measurement.md's prediction 4 scores the 80% posterior
        interval, which no arm kept before: `extra_quantiles` is what puts
        the 10th and 90th percentiles in the projection frame, and it is
        empty by default so no other arm's frame gains a column."""
        cfg = BayesArmConfig(measurement=True, joint=True,
                             extra_quantiles=(10.0, 90.0))
        assert cfg.extra_quantiles == (10.0, 90.0)
        assert BayesArmConfig(joint=True).extra_quantiles == ()

    def test_the_provider_renames_the_posterior_columns_it_carries(self):
        """`<component>_q10` -> `pred_q10`, and the posterior sd along with
        them, because the harness carries `pred_*` columns and nothing else.

        The sd is there because the quantiles are an interval on the *rate*
        and the thing scored against them is a realised rate over a finite
        number of trials; reconstructing a predictive interval needs a scale,
        and two quantiles are not one.
        """
        import numpy as np

        from src.eval.bayes_arm import bayes_k_rate_provider
        import src.eval.bayes_arm as arm

        config = BayesArmConfig(component="hr_rate", measurement=True,
                                joint=True, extra_quantiles=(10.0, 90.0))
        projections = pd.DataFrame({
            "batter": [1, 2], "projected_hr_rate": [0.03, 0.04],
            "hr_rate_std": [0.004, 0.005],
            "hr_rate_q10": [0.02, 0.03], "hr_rate_q90": [0.04, 0.05],
        })
        fit = type("F", (), {"projections": projections})()
        # One fit, handed straight back: the rename is what is under test.
        arm_module_fit = lambda *a, **k: fit  # noqa: E731
        old = arm.fit_bayes_k_rate
        arm.fit_bayes_k_rate = arm_module_fit
        try:
            provider = bayes_k_rate_provider("2024-06-01", 2024, config)
            out = provider(pd.DataFrame({"batter": [1, 2], "season": [2024, 2024]}),
                           COMPONENTS["hr_rate"], 2024)
        finally:
            arm.fit_bayes_k_rate = old
        assert list(out.columns) == ["batter", "predicted", "pred_q10",
                                     "pred_q90", "pred_sd"]
        assert np.allclose(out["pred_sd"], [0.004, 0.005])
