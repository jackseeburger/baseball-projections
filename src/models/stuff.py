"""Stage 1 of BAS-71: the measurement model that scores a pitch from its physics.

A gradient-boosted classifier from pitch characteristics to the pitch's own
outcome, fitted **walk-forward by season** — the model that scores season Y
sees pitches from seasons <= Y-1 and nothing else. Two targets (whiff given a
swing, CSW given a pitch) and four feature sets, three of which exist only to
stop the fourth from claiming credit it has not earned:

    stuff        the pre-registered feature set: velocity, movement, release,
                 spin, arm angle, the velocity/acceleration vector, handedness
                 and everything relative to the pitcher's own fastball.
                 No location.
    pitch_type   what pitch was it, and nothing else.
    fb_velo      how hard does this pitcher throw his fastball, and nothing
                 else.
    pitching     stuff plus `plate_x`/`plate_z`. Labelled "pitching" rather
                 than "stuff" because location is command, and it is never
                 merged into the stuff score — it is here to size how much of
                 a pitch's outcome the physics can reach at all.

`pitch_type` and `fb_velo` are the pre-registration's controls: a stuff model
that cannot beat "what pitch was it and how hard does he throw" out of sample
has not measured anything, and the whole downstream test would be measuring
pitch mix.

**Why LightGBM.** Already a dependency (`src/sim/learned_game.py`), routes NaN
natively — which matters here because `spin_axis` does not exist before 2017
and `arm_angle` before 2021, so a walk-forward model genuinely has missing
columns rather than missing values — and is the only fit in this file that is
not instant. `n_jobs` is pinned at 2 for the same reason `learned_game` pins
it, and because another job may be running on the same box.

**Subsampling.** A fold that scores 2026 has 7.7 million prior pitches behind
it. Every fit is capped at `MAX_TRAIN_ROWS` rows drawn without replacement at a
fixed seed, so the fits are reproducible and none of them takes more than a
couple of minutes. The cap binds from the 2020 fold onward on the CSW target
and from 2023 onward on whiffs; below it the full sample is used.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.pitching_stuff import FEATURE_SETS

logger = logging.getLogger(__name__)

# Deliberately close to `src/sim/learned_game.DEFAULT_PARAMS`: the same shape
# of problem (a binary outcome off a few dozen physical columns), tuned no
# further than "does not overfit a million rows". Depth is larger here because
# the table is two orders of magnitude bigger than that one.
DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.06,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "n_estimators": 350,
    "verbose": -1,
    "n_jobs": 2,
}
MAX_TRAIN_ROWS = 1_500_000
SEED = 20260909

TARGETS = {
    # (label column, row filter) — a whiff rate is per swing, so the whiff
    # model is fitted and scored on swings only. CSW is per pitch.
    "whiff": ("is_whiff", "is_swing"),
    "csw": ("is_csw", None),
}


def target_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """The rows a target is defined on: swings for whiff, everything for CSW."""
    _, mask = TARGETS[target]
    return df if mask is None else df[df[mask].to_numpy()]


@dataclass(frozen=True)
class StuffModel:
    """A fitted classifier and the seasons it was allowed to see."""
    target: str
    feature_set: str
    features: tuple[str, ...]
    train_seasons: tuple[int, ...]
    n_train: int
    booster: object

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[list(self.features)]
        return self.booster.predict_proba(X)[:, 1]


def assert_no_leak(train: pd.DataFrame, score_year: int) -> None:
    """Leakage guard: a pitch from season Y cannot enter the model scoring Y.

    Re-checks the training frame itself rather than trusting the caller's
    filter, exactly as `src.eval.contact.assert_window_clean` does for the
    monthly buckets.
    """
    if train.empty:
        raise ValueError(f"no training pitches for score year {score_year}")
    latest = int(train["game_year"].max())
    if latest >= score_year:
        bad = int((train["game_year"] >= score_year).sum())
        raise ValueError(
            f"leakage: {bad} training pitch(es) from season {latest} >= the "
            f"scored season {score_year}")


def fit_stuff_model(train: pd.DataFrame, target: str, feature_set: str,
                    score_year: int, params: dict | None = None,
                    max_rows: int = MAX_TRAIN_ROWS, seed: int = SEED
                    ) -> StuffModel:
    """Fit one arm on `train`, which must contain no pitch from `score_year`."""
    import lightgbm as lgb

    assert_no_leak(train, score_year)
    label, _ = TARGETS[target]
    features = FEATURE_SETS[feature_set]
    rows = target_rows(train, target)
    if len(rows) > max_rows:
        rows = rows.sample(n=max_rows, random_state=seed)
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_estimators = int(p.pop("n_estimators"))
    model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=seed, **p)
    model.fit(rows[list(features)], rows[label].to_numpy().astype(int))
    logger.info("fit %s/%s for %d on %d rows (seasons %s)", target, feature_set,
                score_year, len(rows), sorted(train["game_year"].unique()))
    return StuffModel(target=target, feature_set=feature_set,
                      features=tuple(features),
                      train_seasons=tuple(sorted(
                          int(s) for s in train["game_year"].unique())),
                      n_train=len(rows), booster=model)


def evaluate(model: StuffModel, held: pd.DataFrame) -> dict:
    """Out-of-sample log-loss, AUC and base rate on one season."""
    from sklearn.metrics import log_loss, roc_auc_score

    rows = target_rows(held, model.target)
    y = rows[TARGETS[model.target][0]].to_numpy().astype(int)
    p = model.predict(rows)
    return {
        "target": model.target, "arm": model.feature_set,
        "n": int(len(rows)), "base_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)),
        "train_seasons": list(model.train_seasons), "n_train": model.n_train,
    }


def baseline_log_loss(y: np.ndarray) -> float:
    """Log-loss of the constant base rate — the floor every arm has to beat."""
    from sklearn.metrics import log_loss

    y = np.asarray(y).astype(int)
    return float(log_loss(y, np.full(len(y), y.mean()), labels=[0, 1]))


def walk_forward(seasons: dict[int, pd.DataFrame], score_years,
                 arms=("stuff", "pitch_type", "fb_velo", "pitching"),
                 targets=("whiff", "csw"), min_train_seasons: int = 2,
                 params: dict | None = None, **kwargs
                 ) -> tuple[pd.DataFrame, dict]:
    """Fit and score every (target, arm) at every scored season.

    Returns the per-season metric table and, for the `stuff` arm only, a dict
    from season to that season's per-pitch predictions — the input to the
    monthly reduction. Only `stuff` predictions are kept: the comparisons exist
    to be scored, not to be aggregated, and the pre-registration is explicit
    that the location arm never reaches the stuff score.
    """
    rows, scored = [], {}
    for year in score_years:
        train_years = [y for y in sorted(seasons) if y < year]
        if len(train_years) < min_train_seasons or year not in seasons:
            continue
        train = pd.concat([seasons[y] for y in train_years], ignore_index=True)
        held = seasons[year]
        for target in targets:
            for arm in arms:
                model = fit_stuff_model(train, target, arm, year,
                                        params=params, **kwargs)
                m = evaluate(model, held)
                m["season"] = year
                m["base_log_loss"] = baseline_log_loss(
                    target_rows(held, target)[TARGETS[target][0]].to_numpy())
                rows.append(m)
                logger.info("%d %s/%s: logloss %.5f auc %.4f", year, target,
                            arm, m["log_loss"], m["auc"])
                if arm == "stuff":
                    scored.setdefault(year, {})[target] = (
                        model.predict(held), model)
        del train
    return pd.DataFrame(rows), scored


__all__ = [
    "DEFAULT_PARAMS", "MAX_TRAIN_ROWS", "SEED", "TARGETS", "StuffModel",
    "assert_no_leak", "baseline_log_loss", "evaluate", "fit_stuff_model",
    "target_rows", "walk_forward",
]
