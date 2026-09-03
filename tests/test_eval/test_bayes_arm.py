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


def test_provider_refuses_components_it_does_not_model():
    """Four of the five components are separate models; serving K% under their
    name would put a wrong number on the scoreboard."""
    provider = bayes_k_rate_provider("2026-07-01", 2026)
    train = pd.DataFrame({"batter": [1], "season": [2026], "pa": [100], "k": [20]})
    for name in ("bb_rate", "hr_rate", "iso", "babip"):
        with pytest.raises(ValueError, match="k_rate only"):
            provider(train, COMPONENTS[name], 2026)


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
