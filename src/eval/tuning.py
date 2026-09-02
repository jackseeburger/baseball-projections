"""Walk-forward parameter search for `marcel_tuned`, and paired scoring.

Stock Marcel's constants were never fit to anything. This module fits them,
under the harness's own rules:

* **Walk-forward only.** For each predict year Y in the tuning window, train
  on seasons <= Y-1 and score Y. The objective is the mean trials-weighted
  MAE across those years; log loss is carried alongside for the binomial
  components but is not what the search minimises (MAE is the number the
  scoreboard reports).
* **One tuning window, one holdout.** The search never sees the out-of-sample
  years. `scripts/tune_marcel.py` tunes on 2020-2024 and scores 2025/2026.
* **Coordinate search, not a joint grid.** Five axes (ballast, the two weight
  ratios jointly, peak age, the two age slopes) swept in turn for a few
  passes. A joint grid over all of them is ~50k points per component per
  year; the coordinate sweep is ~80 and finds the same shallow optimum,
  because the axes are nearly separable at this resolution. A tie, or an
  improvement below `TOL`, keeps the incumbent value — so a component where
  tuning does nothing comes back holding its stock constants rather than
  some equally-good arbitrary point.

The scoring path here is the harness's, reduced to one provider: same
min_trials filter, same trials-weighted metrics, same realized rates. It is
checked against `backtest()`/`score()` in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval import metrics
from src.eval.backtest import COMPONENTS, ComponentSpec
from src.eval.baselines import STOCK_PARAMS, MarcelParams, marcel_tuned

# --- the grid ---------------------------------------------------------------
# Deliberately coarse: the objective is flat enough that a finer grid buys
# fractions of a percent of MAE and buys them in-sample, which is exactly the
# kind of gain that does not survive a holdout.

BALLAST_GRID = [25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0,
                600.0, 800.0, 1200.0, 1800.0, 2600.0]
# (w2/w1, w3/w1). w1 is fixed at 1 because the estimator is scale-free in the
# weights. Stock Marcel's 5/4/3 is exactly (0.8, 0.6) here.
WEIGHT_RATIO_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PEAK_AGE_GRID = [23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
AGE_SLOPE_GRID = [-0.012, -0.008, -0.006, -0.004, -0.003, -0.002, -0.001, 0.0,
                  0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012]

# Relative improvement a candidate must beat to displace the incumbent.
TOL = 1e-9


def grid_summary() -> dict:
    """The searched grid, for the params file's provenance block."""
    return {
        "ballast": BALLAST_GRID,
        "weight_ratios": WEIGHT_RATIO_GRID,
        "peak_age": PEAK_AGE_GRID,
        "age_slope": AGE_SLOPE_GRID,
        "n_points_per_pass": (
            len(BALLAST_GRID) + len(WEIGHT_RATIO_GRID) ** 2
            + len(PEAK_AGE_GRID) + 2 * len(AGE_SLOPE_GRID)
        ),
    }


# --- ages -------------------------------------------------------------------

def chadwick_ages(
    seasons: pd.DataFrame, birthdates: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Replace `age` with the Chadwick register's age as of June 30.

    The season table's own `age` is the Stats API's; the register is the
    project's age of record (roadmap 0.1, real birthdates rather than the
    `debut - 23` proxy the early fits used). Ids missing from the register
    keep their existing age if there is one, else NaN — and `marcel_tuned`
    applies no age adjustment where age is missing.
    """
    from src.data.birthdates import load_birthdates, seasonal_age

    if birthdates is None:
        birthdates = load_birthdates()
    out = seasons.copy()
    ages = seasonal_age(birthdates, out["batter"].to_numpy(),
                        out["season"].to_numpy())
    ages = pd.Series(ages, index=out.index)
    out["age"] = ages.fillna(out["age"]) if "age" in out.columns else ages
    return out


def age_source_report(
    seasons: pd.DataFrame, birthdates: pd.DataFrame | None = None
) -> dict:
    """Coverage of the register on this season frame, and agreement with the API age."""
    from src.data.birthdates import load_birthdates, seasonal_age

    if birthdates is None:
        birthdates = load_birthdates()
    chad = pd.Series(
        seasonal_age(birthdates, seasons["batter"].to_numpy(),
                     seasons["season"].to_numpy()),
        index=seasons.index,
    )
    report = {
        "rows": int(len(seasons)),
        "batters": int(seasons["batter"].nunique()),
        "register_coverage": float(chad.notna().mean()),
    }
    if "age" in seasons.columns:
        both = chad.notna() & seasons["age"].notna()
        diff = np.floor(chad[both]) - seasons.loc[both, "age"]
        report["floor_matches_api_age"] = float((diff == 0).mean())
        report["mean_abs_year_diff"] = float(diff.abs().mean())
    return report


# --- splits and evaluation ---------------------------------------------------

@dataclass(frozen=True)
class Split:
    """One walk-forward year: what a provider may see, and what it is scored on."""
    predict_year: int
    train: pd.DataFrame
    realized: pd.DataFrame     # batter, realized_successes, realized_rate, trials


def make_split(
    seasons: pd.DataFrame,
    spec: ComponentSpec,
    predict_year: int,
    min_trials: int = 100,
) -> Split:
    """Season-level split: train on everything before `predict_year`, score it."""
    train = seasons[seasons["season"] < predict_year].copy()
    if train.empty:
        raise ValueError(f"no training seasons before {predict_year}")
    realized = seasons[seasons["season"] == predict_year]
    realized = realized[realized[spec.trials] >= min_trials]
    if realized.empty:
        raise ValueError(f"no realized rows with >= {min_trials} {spec.trials} "
                         f"in {predict_year}")
    realized = realized.assign(
        realized_rate=realized[spec.successes] / realized[spec.trials]
    )[["batter", spec.successes, spec.trials, "realized_rate"]].rename(
        columns={spec.successes: "realized_successes", spec.trials: "trials"}
    )
    return Split(predict_year, train, realized.reset_index(drop=True))


def make_splits(
    seasons: pd.DataFrame,
    component: str,
    predict_years: list[int],
    min_trials: int = 100,
) -> list[Split]:
    spec = COMPONENTS[component]
    return [make_split(seasons, spec, y, min_trials) for y in predict_years]


def predict_split(split: Split, spec: ComponentSpec, params: MarcelParams) -> pd.DataFrame:
    """Scored frame for one split: realized joined to `marcel_tuned`'s prediction."""
    pred = marcel_tuned(split.train, spec, split.predict_year, params=params)
    pred = pred.dropna(subset=["predicted"])[["batter", "predicted"]]
    return split.realized.merge(pred, on="batter", how="inner")


def score_split(split: Split, spec: ComponentSpec, params: MarcelParams) -> dict:
    j = predict_split(split, spec, params)
    out = {
        "predict_year": split.predict_year,
        "n_players": len(j),
        "mae": metrics.weighted_mae(j["predicted"], j["realized_rate"], j["trials"]),
        "rmse": metrics.weighted_rmse(j["predicted"], j["realized_rate"], j["trials"]),
    }
    out["log_loss"] = (
        metrics.binomial_log_loss(j["predicted"], j["realized_successes"], j["trials"])
        if spec.binomial else float("nan")
    )
    return out


def evaluate(splits: list[Split], spec: ComponentSpec, params: MarcelParams) -> dict:
    """Mean walk-forward scores across the splits. `mae` is the objective."""
    rows = [score_split(s, spec, params) for s in splits]
    return {
        "mae": float(np.mean([r["mae"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "log_loss": float(np.mean([r["log_loss"] for r in rows])),
        "by_year": rows,
    }


# --- the search --------------------------------------------------------------

def _candidates(axis: str, params: MarcelParams) -> list[MarcelParams]:
    if axis == "ballast":
        return [params.replace(ballast=b) for b in BALLAST_GRID]
    if axis == "weights":
        return [params.replace(weights=(1.0, r2, r3))
                for r2 in WEIGHT_RATIO_GRID for r3 in WEIGHT_RATIO_GRID]
    if axis == "peak_age":
        return [params.replace(peak_age=a) for a in PEAK_AGE_GRID]
    if axis == "age_slope_old":
        return [params.replace(age_slope_old=s) for s in AGE_SLOPE_GRID]
    if axis == "age_slope_young":
        return [params.replace(age_slope_young=s) for s in AGE_SLOPE_GRID]
    raise ValueError(f"unknown axis {axis!r}")


AXES = ["ballast", "weights", "peak_age", "age_slope_old", "age_slope_young"]


def coordinate_search(
    splits: list[Split],
    spec: ComponentSpec,
    start: MarcelParams | None = None,
    passes: int = 3,
    axes: list[str] | None = None,
    verbose: bool = False,
) -> tuple[MarcelParams, list[dict]]:
    """Sweep each axis in turn, keeping the best, for `passes` rounds.

    Returns the chosen params and a trace (one row per axis sweep) so the
    doc can say what each knob was worth.
    """
    axes = axes or AXES
    best = start or STOCK_PARAMS.get(spec.name, MarcelParams())
    best_mae = evaluate(splits, spec, best)["mae"]
    trace = [{"pass": 0, "axis": "start", "mae": best_mae,
              "params": best.to_dict()}]
    for p in range(1, passes + 1):
        improved = False
        for axis in axes:
            for cand in _candidates(axis, best):
                mae = evaluate(splits, spec, cand)["mae"]
                if mae < best_mae * (1 - TOL):
                    best, best_mae, improved = cand, mae, True
            trace.append({"pass": p, "axis": axis, "mae": best_mae,
                          "params": best.to_dict()})
            if verbose:
                print(f"    pass {p} {axis:16s} mae={best_mae:.6f}")
        if not improved:
            break
    return best, trace


# --- paired comparison -------------------------------------------------------

def paired_abs_error_diff(
    joined_a: pd.DataFrame, joined_b: pd.DataFrame, weight_col: str = "trials"
) -> dict:
    """Trials-weighted paired difference in absolute error, A minus B.

    Both frames are scored frames (batter, predicted, realized_rate, trials)
    for the *same* split; only batters in both are used, so the difference is
    a within-player comparison and its SE is not inflated by the spread of
    player skill. Negative means A is the better model.
    """
    a = joined_a.set_index("batter")
    b = joined_b.set_index("batter")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    d = ((a["predicted"] - a["realized_rate"]).abs()
         - (b["predicted"] - b["realized_rate"]).abs()).to_numpy()
    w = a[weight_col].to_numpy(dtype="float64")
    mean = float(np.sum(w * d) / np.sum(w))
    # SE of a weighted mean of independent observations.
    se = float(np.sqrt(np.sum(w ** 2 * (d - mean) ** 2)) / np.sum(w))
    return {
        "n": int(len(common)),
        "diff": mean,
        "se": se,
        "t": mean / se if se > 0 else float("nan"),
        "win_rate": float(np.sum(w * (d < 0)) / np.sum(w)),
    }


def paired_from_results(
    results: pd.DataFrame, model_a: str, model_b: str
) -> dict:
    """`paired_abs_error_diff` on a long frame from `backtest()`."""
    cols = ["batter", "predicted", "realized_rate", "trials"]
    a = results[results["model"] == model_a][cols]
    b = results[results["model"] == model_b][cols]
    if a.empty or b.empty:
        raise ValueError(f"missing model in results: {model_a!r} / {model_b!r}")
    return paired_abs_error_diff(a, b)
