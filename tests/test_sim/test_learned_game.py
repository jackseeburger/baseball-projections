"""The learned model's plumbing: fit, calibrate, save, load, blend.

Small and synthetic on purpose. The real training run reads twelve seasons and
belongs in `scripts/train_game_learned.py`; what has to hold in CI is that a
saved artifact serves exactly the probabilities the fitted object did, that the
calibrator is a monotone map that actually fixes a miscalibrated input, and
that the blend collapses to the chain when the learned model has nothing to
add — which is the outcome the scoreboard has to be able to express.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sim import learned_game as lg   # noqa: E402

FEATURES = ["a", "b", "c"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 800
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(0.9 * X["a"] - 0.5 * X["b"])))
    y = (rng.random(n) < p).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    model = lg.fit_booster(X, y, {"n_estimators": 60, "min_child_samples": 40})
    return lg.LearnedModel.from_fitted(model, FEATURES, meta={"n": len(X)})


class TestTheArtifact:
    def test_a_saved_model_serves_the_same_numbers(self, fitted, data, tmp_path):
        X, _ = data
        before = fitted.predict(X)
        path = fitted.save(tmp_path / "learned.json")
        reloaded = lg.LearnedModel.load(path)
        np.testing.assert_allclose(reloaded.predict(X), before, atol=1e-12)
        assert reloaded.features == fitted.features
        assert reloaded.meta["n"] == len(X)

    def test_the_feature_order_is_the_artifact_not_the_caller(self, fitted, data):
        """Columns handed in a different order still score the same game."""
        X, _ = data
        shuffled = X.loc[:, ["c", "a", "b"]]
        np.testing.assert_allclose(fitted.predict(shuffled), fitted.predict(X),
                                   atol=1e-12)

    def test_a_missing_feature_is_an_error_not_a_guess(self, fitted, data):
        X, _ = data
        with pytest.raises(KeyError):
            fitted.predict(X.drop(columns=["b"]))

    def test_importances_name_the_features(self, fitted):
        imp = fitted.importances()
        assert set(imp.index) == set(FEATURES)
        assert imp.iloc[0] >= imp.iloc[-1]


class TestTheCalibrator:
    def test_isotonic_fixes_an_overconfident_probability(self):
        rng = np.random.default_rng(3)
        n = 4000
        truth = rng.uniform(0.35, 0.65, size=n)
        y = (rng.random(n) < truth).astype(int)
        # Push the probabilities away from 0.5: a classic overconfident model.
        z = np.log(truth / (1 - truth)) * 3.0
        raw = 1 / (1 + np.exp(-z))
        cal = lg.Calibrator.fit(raw, y, kind="isotonic")
        fixed = cal(raw)
        before = np.mean((raw - y) ** 2)
        after = np.mean((fixed - y) ** 2)
        assert after < before
        # Monotone: calibration reorders nothing.
        order = np.argsort(raw)
        assert np.all(np.diff(fixed[order]) >= -1e-9)

    def test_platt_is_a_logistic_on_the_log_odds(self):
        rng = np.random.default_rng(4)
        raw = rng.uniform(0.2, 0.8, size=500)
        y = (rng.random(500) < raw).astype(int)
        cal = lg.Calibrator.fit(raw, y, kind="platt")
        assert cal.kind == "platt"
        out = cal(raw)
        assert out.min() > 0 and out.max() < 1
        round_trip = lg.Calibrator.from_dict(cal.to_dict())
        np.testing.assert_allclose(round_trip(raw), out, atol=1e-12)

    def test_identity_leaves_the_probability_alone(self):
        cal = lg.Calibrator()
        p = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(cal(p), p)


class TestTheBlend:
    def test_a_learned_model_that_adds_nothing_gets_no_weight(self):
        """When the learned probability is noise, the stack keeps the chain."""
        rng = np.random.default_rng(5)
        n = 6000
        chain = np.clip(rng.normal(0.54, 0.06, size=n), 0.05, 0.95)
        y = (rng.random(n) < chain).astype(int)
        noise = np.clip(rng.uniform(0.4, 0.6, size=n), 0.05, 0.95)
        w = lg.blend_weights(chain, noise, y)
        assert abs(w["learned"]) < abs(w["chain"])
        blended = lg.blend_predict(w, chain, noise)
        assert np.mean((blended - y) ** 2) <= np.mean((chain - y) ** 2) + 1e-3

    def test_two_identical_inputs_reproduce_them(self):
        rng = np.random.default_rng(6)
        n = 4000
        chain = np.clip(rng.normal(0.54, 0.08, size=n), 0.05, 0.95)
        y = (rng.random(n) < chain).astype(int)
        w = lg.blend_weights(chain, chain, y)
        blended = lg.blend_predict(w, chain, chain)
        # The stack recovers a calibrated version of the one signal it has.
        assert np.corrcoef(blended, chain)[0, 1] > 0.99
