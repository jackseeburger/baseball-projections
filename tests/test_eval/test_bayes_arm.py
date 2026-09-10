"""The refit Bayesian arm's provider shape and its coverage bookkeeping.

Nothing here samples. What is worth testing without MCMC is the plumbing that
decides *what gets compared*: the arm has to be importable where pymc is not
installed (CI), it has to refuse components it does not model instead of
serving the K% number under another name, and it has to cover the batters the
baselines cover — because `common_players=True` means an arm with thin
coverage silently shrinks the player set every other arm is scored on, which
changes the comparison rather than making it fair.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
