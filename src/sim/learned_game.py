"""The learned station-E challenger: fit it, calibrate it, save it, serve it.

The hand-built chain is a sequence of decisions a person made — log5 on
Pythagenpat, the starter as a delta over the innings he covers, the pen over
the rest, each term worth one to four ten-thousandths. This module is the
control on that: give a gradient-boosted model the *same* pre-game inputs
(`src/sim/game_features.py`) and let it choose the functional form.

Three pieces, deliberately small:

  * `fit_booster` — LightGBM on a feature matrix, with the regularisation this
    problem needs written down as the default. A baseball game is close to a
    coin flip: the whole distance from "know nothing" (Brier .2489) to the
    exchanges (.2416) is 0.007, so a model with room to memorise 20,000 rows
    will spend that room on noise. Shallow trees, a high minimum leaf count
    and heavy subsampling are not tuning taste, they are the shape of the
    problem.
  * `Calibrator` — isotonic or Platt, fitted on held-out training-season
    predictions and applied to everything after. A ranking that is right and a
    probability that is right are different things, and Brier and log loss
    both price the second.
  * `LearnedModel` — booster plus calibrator plus the feature order, saved as
    one JSON so the nightly and the backtest load the same artifact and cannot
    drift on column order.

Nothing here fetches, and nothing here chooses a hyperparameter by looking at
a scored season; `scripts/train_game_learned.py` owns the walk-forward
protocol that keeps that true.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# The artifact the nightly serves, when one has been fitted and gated.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "game_learned.json"

# Defaults chosen for the shape of the problem, not for a score: shallow trees
# (the signal is a smooth function of a few run rates), a leaf that must hold
# 1-2% of the training rows, and both row and column subsampling. The learning
# rate and tree count are the two the inner split is allowed to move.
DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 300,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 20.0,
    "n_estimators": 400,
    "verbose": -1,
    # A table this small (tens of thousands of rows, forty-odd columns) is
    # dominated by OpenMP synchronisation, not by arithmetic: the same fit that
    # takes 0.1 s on two threads takes 20 s on all of them. Pinned low so a
    # test that trains a few hundred rows stays a test.
    "n_jobs": 2,
}


def _lgb():
    """Import LightGBM at call time so importing this module stays cheap."""
    import lightgbm as lgb
    return lgb


def fit_booster(X: pd.DataFrame, y, params: dict | None = None,
                seed: int = 20260903):
    """A fitted LightGBM classifier on `X`/`y`, all parameters explicit."""
    lgb = _lgb()
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_estimators = int(p.pop("n_estimators"))
    model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=seed, **p)
    model.fit(X, np.asarray(y).astype(int))
    return model


@dataclass
class Calibrator:
    """A monotone map from raw probability to calibrated probability.

    `kind` is "isotonic" (a step function through the held-out data, fitted by
    `sklearn.isotonic`) , "platt" (a one-variable logistic on the log-odds) or
    "identity". Stored as its own knots / coefficients rather than as a pickled
    estimator so the artifact is readable and version-independent.
    """
    kind: str = "identity"
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, p, y, kind: str = "isotonic") -> "Calibrator":
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=float)
        if kind == "identity":
            return cls(kind="identity")
        if kind == "platt":
            from sklearn.linear_model import LogisticRegression
            z = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y.astype(int))
            return cls(kind="platt", a=float(lr.coef_[0][0]),
                       b=float(lr.intercept_[0]))
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(p, y)
        # `X_thresholds_`/`y_thresholds_` are the fitted step function; keeping
        # them makes the artifact a plain interpolation table.
        return cls(kind="isotonic", x=[float(v) for v in iso.X_thresholds_],
                   y=[float(v) for v in iso.y_thresholds_])

    def __call__(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.kind == "identity" or (self.kind == "isotonic" and len(self.x) < 2):
            return p
        if self.kind == "platt":
            z = self.a * np.log(p / (1 - p)) + self.b
            return 1.0 / (1.0 + np.exp(-z))
        return np.clip(np.interp(p, self.x, self.y), 1e-6, 1 - 1e-6)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x": self.x, "y": self.y,
                "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(kind=d.get("kind", "identity"), x=list(d.get("x", [])),
                   y=list(d.get("y", [])), a=float(d.get("a", 1.0)),
                   b=float(d.get("b", 0.0)))


@dataclass
class LearnedModel:
    """A fitted booster, its calibrator and the feature order it was fitted on."""
    booster_text: str
    features: list
    calibrator: Calibrator = field(default_factory=Calibrator)
    meta: dict = field(default_factory=dict)
    _booster: object = None

    @classmethod
    def from_fitted(cls, model, features, calibrator: Calibrator | None = None,
                    meta: dict | None = None) -> "LearnedModel":
        booster = model.booster_ if hasattr(model, "booster_") else model
        return cls(booster_text=booster.model_to_string(),
                   features=[str(c) for c in features],
                   calibrator=calibrator or Calibrator(),
                   meta=dict(meta or {}))

    @property
    def booster(self):
        if self._booster is None:
            self._booster = _lgb().Booster(model_str=self.booster_text)
        return self._booster

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(home) for a frame carrying `self.features`."""
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise KeyError(f"learned model is missing features {missing}")
        mat = X.loc[:, self.features].astype("float32").to_numpy()
        return np.asarray(self.booster.predict(mat), dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(home) — the number that goes on a scoreboard."""
        return np.clip(self.calibrator(self.raw(X)), 1e-6, 1 - 1e-6)

    def importances(self, kind: str = "gain") -> pd.Series:
        imp = self.booster.feature_importance(importance_type=kind)
        return pd.Series(imp, index=self.features, dtype=float).sort_values(
            ascending=False)

    def to_dict(self) -> dict:
        return {"features": list(self.features), "booster": self.booster_text,
                "calibration": self.calibrator.to_dict(), "meta": dict(self.meta)}

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedModel":
        return cls(booster_text=d["booster"], features=list(d["features"]),
                   calibrator=Calibrator.from_dict(d.get("calibration", {})),
                   meta=dict(d.get("meta", {})))

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH) -> "LearnedModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def blend_weights(chain_p, learned_p, y) -> dict:
    """Logistic stack of the two probabilities, fitted on log-odds.

        logit(p) = w0 + w_chain · logit(chain) + w_learned · logit(learned)

    Fitted on training-season games only. `w_learned` near zero is the finding
    that the learned model adds nothing the chain does not already have.
    """
    from sklearn.linear_model import LogisticRegression
    z = np.column_stack([_logit(chain_p), _logit(learned_p)])
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(z, np.asarray(y).astype(int))
    return {"intercept": float(lr.intercept_[0]),
            "chain": float(lr.coef_[0][0]), "learned": float(lr.coef_[0][1])}


def blend_predict(weights: dict, chain_p, learned_p) -> np.ndarray:
    z = (weights["intercept"] + weights["chain"] * _logit(chain_p)
         + weights["learned"] * _logit(learned_p))
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


__all__ = ["Calibrator", "DEFAULT_MODEL_PATH", "DEFAULT_PARAMS", "LearnedModel",
           "blend_predict", "blend_weights", "fit_booster"]
