"""Stage 1 of BAS-76: what a pitch's *location* is worth on top of its physics.

The same object as `src/models/stuff.py`, one step further in. A gradient-
boosted classifier from pitch characteristics to the pitch's own outcome,
fitted **walk-forward by season** — the model that scores season Y sees pitches
from seasons <= Y-1 and nothing else — with four feature sets:

    pitching     stuff *and* location: where the pitch crossed relative to this
                 batter's own zone, the count, the handedness matchup and the
                 pitch group, on top of every physics column the stuff model
                 reads. The pre-registered arm.
    stuff        BAS-71's feature set, refitted here on the same rows. Both a
                 control and the thing the command residual is measured
                 against.
    location     the location block with no physics at all.
    pitch_type   what pitch was it, and nothing else.

and two targets:

    csw            called strike or whiff, given the pitch
    called_taken   called strike, given the batter did not swing

`called_taken` is the target a stuff model has the least business predicting:
the swing decision is held fixed and all that is left is whether the pitch was
in a place the umpire calls. If location does not beat physics there, it does
not beat it anywhere.

**What comes out.** Unlike `stuff.walk_forward`, which keeps one arm's
predictions because the stuff score is one model's output, this keeps *two*:
`pitching` and `stuff`, on both targets, for every scored pitch. The reduction
downstream is their difference — the location contribution — and a difference
of two models fitted on different rows would not be one. Everything else about
the fit (LightGBM, the 1.5M-row cap, the seed, the leakage re-check inside
`fit_stuff_model`) is BAS-71's and is imported rather than re-specified.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.pitching_command import FEATURE_SETS, RESIDUAL_ARM, RESIDUAL_BASE_ARM
from src.models.stuff import (
    DEFAULT_PARAMS,
    MAX_TRAIN_ROWS,
    SEED,
    StuffModel,
    assert_no_leak,
    baseline_log_loss,
)

logger = logging.getLogger(__name__)

# (label column, row-filter column). CSW is per pitch; a called strike given a
# take is per taken pitch, so the model is fitted and scored on takes only.
TARGETS = {
    "csw": ("is_csw", None),
    "called_taken": ("is_called", "is_taken"),
}
ARMS = ("pitching", "stuff", "location", "pitch_type")
# Which arms' per-pitch predictions the reduction needs.
KEEP_ARMS = (RESIDUAL_ARM, RESIDUAL_BASE_ARM)


def target_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """The rows a target is defined on: takes for `called_taken`, all for CSW."""
    _, mask = TARGETS[target]
    return df if mask is None else df[df[mask].to_numpy()]


def training_columns(arms=ARMS, targets=("csw", "called_taken")) -> list[str]:
    """Every column a fit on these arms and targets reads, and nothing else.

    `game_year` is in because `assert_no_leak` re-derives the training seasons
    from the frame rather than trusting the caller.
    """
    cols = {"game_year"}
    for a in arms:
        cols.update(FEATURE_SETS[a])
    for t in targets:
        label, mask = TARGETS[t]
        cols.add(label)
        if mask:
            cols.add(mask)
    return sorted(cols)


def fit_command_model(train: pd.DataFrame, target: str, feature_set: str,
                      score_year: int, params: dict | None = None,
                      max_rows: int = MAX_TRAIN_ROWS, seed: int = SEED
                      ) -> StuffModel:
    """Fit one arm on `train`, which must contain no pitch from `score_year`.

    `src.models.stuff.assert_no_leak` re-checks the training frame itself
    rather than trusting the caller's filter, exactly as BAS-71 does.
    """
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


def walk_forward(seasons: dict[int, pd.DataFrame], score_years,
                 arms=ARMS, targets=("csw", "called_taken"),
                 min_train_seasons: int = 2, params: dict | None = None,
                 **kwargs) -> tuple[pd.DataFrame, dict]:
    """Fit and score every (target, arm) at every scored season.

    Returns the per-season metric table and, for the two arms the residual is
    built from, per-pitch predictions indexed as `scored[year][target][arm]`.

    A `called_taken` model is fitted on takes only, so its predictions are
    defined on takes only; they come back as a full-length array with NaN on
    the pitches the batter swung at, which is what
    `src.data.pitching_command.monthly_buckets` masks out anyway.
    """
    rows, scored = [], {}
    keep = training_columns(arms, targets)
    for year in score_years:
        train_years = [y for y in sorted(seasons) if y < year]
        if len(train_years) < min_train_seasons or year not in seasons:
            continue
        # Only the columns a fit reads. Eleven seasons is 7.7 million pitches
        # and the frame carries two dozen columns the training half never
        # touches (identifiers, the pitch-name strings, the region flags the
        # *reduction* wants); concatenating them would double the peak.
        train = pd.concat([seasons[y][keep] for y in train_years],
                          ignore_index=True)
        held = seasons[year]
        for target in targets:
            held_rows = target_rows(held, target)
            for arm in arms:
                model = fit_command_model(train, target, arm, year,
                                          params=params, **kwargs)
                m = evaluate(model, held)
                m["season"] = year
                m["base_log_loss"] = baseline_log_loss(
                    held_rows[TARGETS[target][0]].to_numpy())
                rows.append(m)
                logger.info("%d %s/%s: logloss %.5f auc %.4f", year, target,
                            arm, m["log_loss"], m["auc"])
                if arm in KEEP_ARMS:
                    scored.setdefault(year, {}).setdefault(target, {})[arm] = (
                        _full_length(model, held, held_rows))
        del train
    return pd.DataFrame(rows), scored


def _full_length(model: StuffModel, held: pd.DataFrame,
                 held_rows: pd.DataFrame) -> np.ndarray:
    """Predictions aligned to every row of `held`, NaN off the target's rows."""
    if len(held_rows) == len(held):
        return model.predict(held)
    out = np.full(len(held), np.nan)
    out[held.index.get_indexer(held_rows.index)] = model.predict(held_rows)
    return out


def attach_scores(held: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """Name the four prediction columns the monthly reduction expects."""
    return held.assign(
        p_csw_pitching=scores["csw"][RESIDUAL_ARM],
        p_csw_stuff=scores["csw"][RESIDUAL_BASE_ARM],
        p_cs_pitching=scores["called_taken"][RESIDUAL_ARM],
        p_cs_stuff=scores["called_taken"][RESIDUAL_BASE_ARM],
    )


__all__ = [
    "ARMS", "KEEP_ARMS", "TARGETS", "attach_scores", "evaluate",
    "training_columns",
    "fit_command_model", "target_rows", "walk_forward",
]
