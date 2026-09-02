"""Baseline prediction providers for the backtest harness.

A provider maps (train_seasons, component_spec, predict_year) to a frame of
[batter, predicted]. `train_seasons` contains ONLY seasons up to the training
cutoff — the harness enforces that, so a provider cannot leak the future.

Baselines are deliberately dumb. They exist to be beaten; a model change
that does not beat Marcel is a regression no matter how principled it looks.

**Partial seasons.** In intra-season mode the training frame's most recent
row is the current season *through the cutoff*, flagged `partial=True`. The
baselines read that flag rather than assuming every season is complete:

    marcel           treats it as the most recent season; because Marcel
                     weights by trials, a 200-PA partial season naturally
                     counts a third of a full one.
    league_average   league rate through the cutoff (the latest season).
    previous_season  the last *full* season — the partial one is skipped, so
                     this stays the "no in-season information" arm.
    season_to_date   the player's own partial-season rate regressed to
                     league with the component's ballast: "just use this
                     year".
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.marcel import age_adjustment

# Marcel constants (Tango): 5/4/3 weights, 200 PA of league-average ballast.
MARCEL_YEAR_WEIGHTS = {0: 5.0, 1: 4.0, 2: 3.0}  # years before predict_year - 1
MARCEL_BALLAST = 200.0

# Trials of league-average ballast for the season-to-date baseline: the
# published stabilization points, where a player's own sample and the league
# prior carry equal weight (Carleton / FanGraphs). BABIP's is huge, which is
# the whole reason "he's hitting .400 on balls in play" means so little.
SEASON_TO_DATE_BALLAST = {
    "k_rate": 60.0,     # PA
    "bb_rate": 120.0,   # PA
    "hr_rate": 170.0,   # PA
    "babip": 820.0,     # BIP
    "iso": 160.0,       # AB
}


def _league_rate(train: pd.DataFrame, spec, year: int | None = None) -> float:
    """Trials-weighted league rate, optionally for a single season."""
    df = train if year is None else train[train["season"] == year]
    return float(df[spec.successes].sum() / df[spec.trials].sum())


def _partial_mask(train: pd.DataFrame) -> pd.Series:
    """Boolean mask of partial (through-the-cutoff) rows; all False without the flag."""
    if "partial" not in train.columns:
        return pd.Series(False, index=train.index)
    return train["partial"].fillna(False).astype(bool)


def full_seasons(train: pd.DataFrame) -> pd.DataFrame:
    """Training rows for complete seasons only (drops a partial current season)."""
    return train[~_partial_mask(train)]


def latest_rows(train: pd.DataFrame) -> pd.DataFrame:
    """The most recent training slice: the partial season if there is one,
    else the last full season."""
    partial = train[_partial_mask(train)]
    if not partial.empty:
        return partial
    return train[train["season"] == int(train["season"].max())]


def league_average(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Everyone projects to the most recent training slice's league rate.

    With a partial current season that is the league rate through the cutoff.
    """
    g = latest_rows(train)
    rate = float(g[spec.successes].sum() / g[spec.trials].sum())
    return pd.DataFrame({"batter": g["batter"].unique(), "predicted": rate})


def previous_season(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Player's own rate in the last *complete* training season, no regression."""
    full = full_seasons(train)
    if full.empty:
        raise ValueError("previous_season needs at least one complete season")
    last = int(full["season"].max())
    g = full[full["season"] == last]
    pred = g[spec.successes] / g[spec.trials]
    return pd.DataFrame({"batter": g["batter"].values, "predicted": pred.values})


def season_to_date(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """The player's own rate so far this season, regressed to league.

    pred = (successes + b·league) / (trials + b), with b the component's
    stabilization point. A player with zero trials gets exactly the league
    rate, which is what makes this a well-behaved arm at an April cutoff.
    """
    g = latest_rows(train).groupby("batter", as_index=False).agg(
        successes=(spec.successes, "sum"), trials=(spec.trials, "sum")
    )
    league = float(g["successes"].sum() / g["trials"].sum())
    b = SEASON_TO_DATE_BALLAST.get(spec.name, MARCEL_BALLAST)
    return pd.DataFrame({
        "batter": g["batter"].values,
        "predicted": (g["successes"] + b * league) / (g["trials"] + b),
    })


def marcel(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Marcel the Monkey: 5/4/3-weighted 3-year rates regressed to league
    mean with 200 trials of ballast, plus the simple age adjustment when an
    `age` column is available.
    """
    last = int(train["season"].max())
    league = _league_rate(train, spec, last)

    recent = train[train["season"] >= last - 2].copy()
    recent["w"] = (last - recent["season"]).map(MARCEL_YEAR_WEIGHTS)
    recent["w_trials"] = recent["w"] * recent[spec.trials]
    recent["w_successes"] = recent["w"] * recent[spec.successes]

    g = recent.groupby("batter").agg(
        w_trials=("w_trials", "sum"),
        w_successes=("w_successes", "sum"),
    )
    # Regress toward league mean with MARCEL_BALLAST weighted-trials of it.
    # Scale ballast by the mean year weight so it matches Marcel's intent of
    # ~200 real PA of league average.
    ballast = MARCEL_BALLAST * np.mean(list(MARCEL_YEAR_WEIGHTS.values()))
    pred = (g["w_successes"] + ballast * league) / (g["w_trials"] + ballast)
    out = pred.rename("predicted").reset_index()

    if "age" in train.columns:
        age_last = (
            train[train["season"] == last]
            .dropna(subset=["age"])
            .set_index("batter")["age"]
        )
        proj_age = out["batter"].map(age_last) + (predict_year - last)
        adj = np.array([
            age_adjustment(int(a), spec.name) if np.isfinite(a) else 1.0
            for a in proj_age
        ])
        out["predicted"] = out["predicted"] * adj
    return out


def marcel_preseason(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
    """Marcel with the partial current season withheld.

    The control arm for intra-season backtests: identical to the Marcel a
    preseason run would have produced, scored on the same rest-of-season
    outcomes. `marcel` minus `marcel_preseason` is the value of in-season
    information, with the model held fixed.
    """
    return marcel(full_seasons(train), spec, predict_year)


BASELINES = {
    "marcel": marcel,
    "previous_season": previous_season,
    "league_average": league_average,
}

# `season_to_date` and `marcel_preseason` only say something interesting when
# the training frame carries a partial current season, so they are opt-in at
# the season level and default in intra-season mode.
INTRASEASON_BASELINES = {
    **BASELINES,
    "season_to_date": season_to_date,
    "marcel_preseason": marcel_preseason,
}


# ---------------------------------------------------------------------------
# Tuned Marcel
# ---------------------------------------------------------------------------
# Stock Marcel's constants are Tango's defaults, chosen once and never fit to
# anything: 5/4/3 recency, one 200-trial ballast for every component, and one
# age curve whose slopes do not know whether the component stabilizes in 60
# PA or 820 BIP. The pitcher work found that per-component ballasts near the
# stabilization points matter a lot. `marcel_tuned` is the same estimator
# with those constants made per-component parameters, so they can be *fit* by
# walk-forward (`scripts/tune_marcel.py`) instead of assumed.
#
# The estimator, for a component with successes s and trials t:
#
#     pred = (sum_i w_i*s_i + B*league) / (sum_i w_i*t_i + B),  B = ballast * mean(w)
#
# over the three seasons ending with the most recent training season, times
# the age multiplier
#
#     adj = 1 + (floor(age) - peak_age) * slope
#     slope = age_slope_old if age > peak_age else age_slope_young
#
# Scaling B by mean(w) is what stock Marcel does, and it makes the ballast
# scale-free in the weights: multiply every w_i by c and the prediction is
# unchanged. Read `ballast` as trials of league average *at the average year
# weight* — with the weights normalized to mean 1 it is plain trials.
# (Consequence: the two knobs are not independent. Shrinking the tail weights
# toward zero raises the normalized weight on the recent season and so lowers
# the effective ballast against it. The search moves both, so it explores the
# product space regardless.)


@dataclass(frozen=True)
class MarcelParams:
    """Per-component Marcel constants. Six numbers, three of them the age curve.

    ballast:          trials of league average to regress to, in the units of
                      the component's denominator (PA / AB / BIP), at the
                      average year weight.
    weights:          (w1, w2, w3) recency weights, most recent first. Only
                      the ratios matter (see the module note above).
    peak_age:         age at which the multiplier is exactly 1.0.
    age_slope_young:  per-year multiplier slope *below* the peak.
    age_slope_old:    per-year multiplier slope *above* the peak.

    Both slopes are signed the same way — the multiplier is always
    1 + (age - peak)*slope — so a component that decays with age has negative
    slopes and K%, which rises with age, has positive ones.
    """
    ballast: float = MARCEL_BALLAST
    weights: tuple[float, float, float] = (5.0, 4.0, 3.0)
    peak_age: float = 27.0
    age_slope_young: float = 0.0
    age_slope_old: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ballast": float(self.ballast),
            "weights": [float(w) for w in self.weights],
            "peak_age": float(self.peak_age),
            "age_slope_young": float(self.age_slope_young),
            "age_slope_old": float(self.age_slope_old),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MarcelParams":
        return cls(
            ballast=float(d["ballast"]),
            weights=tuple(float(w) for w in d["weights"]),  # type: ignore[arg-type]
            peak_age=float(d["peak_age"]),
            age_slope_young=float(d["age_slope_young"]),
            age_slope_old=float(d["age_slope_old"]),
        )

    def replace(self, **kwargs) -> "MarcelParams":
        return dataclasses.replace(self, **kwargs)


# The stock-Marcel parameterisation, component by component. These reproduce
# `marcel` exactly (see tests/test_eval/test_marcel_tuned.py): the age slopes
# are `src.models.marcel.age_adjustment` rewritten in the signed form above —
# K% rises with age, the "positive" rate stats decay, and HR/PA is in neither
# of that function's sets, so stock Marcel ages HR not at all.
_STOCK_POSITIVE = (-0.001, -0.003)   # (young, old) for bb_rate / iso / babip
_STOCK_NEGATIVE = (0.002, 0.005)     # (young, old) for k_rate
_STOCK_FLAT = (0.0, 0.0)             # (young, old) for hr_rate

STOCK_PARAMS: dict[str, MarcelParams] = {
    name: MarcelParams(
        ballast=MARCEL_BALLAST,
        weights=(MARCEL_YEAR_WEIGHTS[0], MARCEL_YEAR_WEIGHTS[1],
                 MARCEL_YEAR_WEIGHTS[2]),
        peak_age=27.0,
        age_slope_young=slopes[0],
        age_slope_old=slopes[1],
    )
    for name, slopes in {
        "k_rate": _STOCK_NEGATIVE,
        "bb_rate": _STOCK_POSITIVE,
        "hr_rate": _STOCK_FLAT,
        "iso": _STOCK_POSITIVE,
        "babip": _STOCK_POSITIVE,
    }.items()
}

MARCEL_PARAMS_PATH = Path(__file__).with_name("marcel_params.json")


def load_marcel_params(
    path: str | Path | None = None, strict: bool = False
) -> dict[str, MarcelParams]:
    """Fitted per-component params from `src/eval/marcel_params.json`.

    Falls back to `STOCK_PARAMS` for any component the file does not carry
    (and entirely, if the file is absent and `strict` is False), so
    `marcel_tuned` is always callable — an unfit component then simply *is*
    stock Marcel.
    """
    path = Path(path) if path is not None else MARCEL_PARAMS_PATH
    if not path.exists():
        if strict:
            raise FileNotFoundError(f"{path} not found — run scripts/tune_marcel.py")
        return dict(STOCK_PARAMS)
    blob = json.loads(path.read_text())
    out = dict(STOCK_PARAMS)
    for name, d in blob.get("components", {}).items():
        out[name] = MarcelParams.from_dict(d)
    return out


def save_marcel_params(
    params: dict[str, MarcelParams],
    path: str | Path | None = None,
    **extra,
) -> Path:
    """Write the params file `load_marcel_params` reads. `extra` is metadata."""
    path = Path(path) if path is not None else MARCEL_PARAMS_PATH
    blob = {**extra, "components": {k: v.to_dict() for k, v in params.items()}}
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return path


def tuned_age_adjustment(age, params: MarcelParams) -> np.ndarray:
    """Multiplicative age factor, vectorised. 1.0 wherever age is missing.

    Ages are floored before use, exactly as `marcel` does (`int(a)`), so a
    fractional Chadwick age and the integer Stats API age give the same
    answer and a tuned-vs-stock comparison isolates the parameters.
    """
    a = np.floor(np.asarray(age, dtype="float64"))
    d = a - params.peak_age
    adj = np.where(
        d > 0,
        1.0 + d * params.age_slope_old,
        1.0 + d * params.age_slope_young,
    )
    return np.where(np.isfinite(a), adj, 1.0)


def _resolve_params(params, component: str) -> MarcelParams:
    """Accept a MarcelParams, a {component: MarcelParams} map, or None."""
    if params is None:
        return load_marcel_params().get(component, MarcelParams())
    if isinstance(params, MarcelParams):
        return params
    if isinstance(params, dict):
        if component not in params:
            raise KeyError(f"no tuned params for component {component!r}")
        p = params[component]
        return p if isinstance(p, MarcelParams) else MarcelParams.from_dict(p)
    raise TypeError(f"unsupported params: {type(params).__name__}")


def marcel_tuned(
    train: pd.DataFrame,
    spec,
    predict_year: int,
    *,
    params: "MarcelParams | dict | None" = None,
) -> pd.DataFrame:
    """Marcel with fitted per-component ballast, recency weights and age curve.

    Same provider signature as `marcel`, and the same treatment of a partial
    current season: the partial row is simply the most recent season, and
    because the estimator weights by trials it scales itself down. With
    `params=STOCK_PARAMS[spec.name]` this *is* `marcel`, to the bit.

    `params` may be a single MarcelParams, a {component: MarcelParams} map,
    or None to read `src/eval/marcel_params.json`.
    """
    p = _resolve_params(params, spec.name)
    if sum(p.weights) <= 0:
        raise ValueError("marcel_tuned needs at least one positive year weight")

    last = int(train["season"].max())
    league = _league_rate(train, spec, last)

    weights = {i: float(w) for i, w in enumerate(p.weights)}
    recent = train[train["season"] >= last - 2].copy()
    recent["w"] = (last - recent["season"]).map(weights)
    recent["w_trials"] = recent["w"] * recent[spec.trials]
    recent["w_successes"] = recent["w"] * recent[spec.successes]

    g = recent.groupby("batter").agg(
        w_trials=("w_trials", "sum"),
        w_successes=("w_successes", "sum"),
    )
    ballast = p.ballast * np.mean(list(p.weights))
    pred = (g["w_successes"] + ballast * league) / (g["w_trials"] + ballast)
    out = pred.rename("predicted").reset_index()

    if "age" in train.columns:
        age_last = (
            train[train["season"] == last]
            .dropna(subset=["age"])
            .set_index("batter")["age"]
        )
        proj_age = out["batter"].map(age_last) + (predict_year - last)
        out["predicted"] = out["predicted"] * tuned_age_adjustment(proj_age, p)
    return out


def marcel_tuned_provider(params: "MarcelParams | dict | None" = None):
    """Bind params to `marcel_tuned` to get a plain (train, spec, year) provider."""

    def provider(train: pd.DataFrame, spec, predict_year: int) -> pd.DataFrame:
        return marcel_tuned(train, spec, predict_year, params=params)

    return provider


def marcel_tuned_preseason(
    train: pd.DataFrame, spec, predict_year: int, *,
    params: "MarcelParams | dict | None" = None,
) -> pd.DataFrame:
    """`marcel_tuned` with the partial current season withheld (the control)."""
    return marcel_tuned(full_seasons(train), spec, predict_year, params=params)
