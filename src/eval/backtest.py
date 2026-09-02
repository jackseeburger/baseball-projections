"""Backtest harness: refit-on-the-past, score-on-the-realized-future.

Interface (roadmap 0.2):

    backtest(component, train_through_year, predict_year) -> DataFrame

and, for rest-of-season projections, the same call with a date cutoff:

    backtest(component, cutoff_date="2026-07-01", pa_frame=..., seasons=...)

which trains on everything through July 1 and scores the rest of the season
(`src/eval/intraseason.py`).

The harness slices a season-level frame at the training cutoff, hands ONLY
past seasons to each prediction provider, joins realized outcomes from the
predict year, and scores component-wise log loss, MAE, RMSE, and calibration
against the Marcel / previous-season / league-average baselines.

Season frame schema (one row per batter-season):
    batter, season, and the component's numerator/denominator columns
    (see COMPONENTS), plus optional `age` for Marcel's age adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.eval import metrics
from src.eval.baselines import BASELINES


@dataclass(frozen=True)
class ComponentSpec:
    """A projectable rate: numerator / denominator at the season level."""
    name: str
    successes: str   # numerator column
    trials: str      # denominator column (also the scoring weight)
    binomial: bool   # True → log loss applies (numerator is a count)


COMPONENTS = {
    "k_rate": ComponentSpec("k_rate", "k", "pa", binomial=True),
    "bb_rate": ComponentSpec("bb_rate", "bb", "pa", binomial=True),
    "hr_rate": ComponentSpec("hr_rate", "hr", "pa", binomial=True),
    "babip": ComponentSpec("babip", "hits_in_play", "bip", binomial=True),
    # ISO numerator is extra-base points (2B + 2·3B + 3·HR), not a count of
    # independent trials, so log loss is not meaningful for it.
    "iso": ComponentSpec("iso", "xb_points", "ab", binomial=False),
}

Provider = Callable[[pd.DataFrame, ComponentSpec, int], pd.DataFrame]


def backtest(
    component: str,
    train_through_year: int | None = None,
    predict_year: int | None = None,
    *,
    seasons: pd.DataFrame,
    providers: dict[str, Provider] | None = None,
    min_trials: int = 100,
    common_players: bool = True,
    cutoff_date: str | None = None,
    pa_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run one train/predict split for one component.

    Args:
        component: key in COMPONENTS.
        train_through_year: last season providers are allowed to see.
            Not used (and not required) when `cutoff_date` is given.
        predict_year: season to score against (default: the next one, or the
            cutoff's own year in intra-season mode).
        seasons: season-level frame (see module docstring).
        providers: model name → provider. Defaults to the three baselines
            (plus `season_to_date` in intra-season mode).
            Add your model with e.g. {"bayes": parquet_provider(path), **BASELINES}.
        min_trials: drop realized seasons below this many trials — tiny
            samples measure noise, not skill.
        common_players: score every model on the same batters (the
            intersection of their coverage). Without this a model that
            covers rookies is scored on a harder population than one that
            doesn't, and the comparison is meaningless.
        cutoff_date: ISO date. Switches the harness to intra-season mode:
            training is the prior seasons plus the current season *through*
            the cutoff, and the realized side is the rest of that season.
            Requires `pa_frame` (PA-level outcomes for the predict year) —
            season totals have no dates to cut on. See
            `src/eval/intraseason.py`.
        pa_frame: PA-level outcomes, only used with `cutoff_date`.

    Returns:
        Long frame: [component, model, batter, predicted, realized_successes,
        realized_rate, trials]. Feed to score() / calibration().
    """
    if cutoff_date is not None:
        from src.eval.intraseason import backtest_intraseason

        return backtest_intraseason(
            component, cutoff_date, predict_year, seasons=seasons,
            pa_frame=pa_frame, providers=providers, min_trials=min_trials,
            common_players=common_players,
        )
    if pa_frame is not None:
        raise ValueError("pa_frame is only meaningful with cutoff_date")
    if train_through_year is None:
        raise ValueError("train_through_year is required without a cutoff_date")

    spec = COMPONENTS[component]
    predict_year = predict_year or train_through_year + 1
    if predict_year <= train_through_year:
        raise ValueError("predict_year must be after train_through_year")

    # The leakage guard: providers never see the predict year.
    train = seasons[seasons["season"] <= train_through_year].copy()
    if train.empty:
        raise ValueError(f"no training seasons at or before {train_through_year}")

    realized = seasons[seasons["season"] == predict_year].copy()
    return _run_split(
        component, spec, train, realized, predict_year,
        providers=providers, min_trials=min_trials,
        common_players=common_players,
    )


def _run_split(
    component: str,
    spec: ComponentSpec,
    train: pd.DataFrame,
    realized: pd.DataFrame,
    predict_year: int,
    *,
    providers: dict[str, Provider] | None,
    min_trials: int,
    common_players: bool,
) -> pd.DataFrame:
    """Score providers trained on `train` against `realized`.

    The shared tail of every split — season-level or intra-season. `train`
    and `realized` are already sliced; nothing here re-checks the cutoff,
    that is the caller's leakage guard.
    """
    providers = providers or dict(BASELINES)
    realized = realized[realized[spec.trials] >= min_trials]
    if realized.empty:
        raise ValueError(f"no realized rows with >= {min_trials} "
                         f"{spec.trials} in {predict_year}")
    realized = realized.assign(
        realized_rate=realized[spec.successes] / realized[spec.trials]
    )[["batter", spec.successes, spec.trials, "realized_rate"]].rename(
        columns={spec.successes: "realized_successes", spec.trials: "trials"}
    )

    preds = {}
    for name, provider in providers.items():
        pred = provider(train, spec, predict_year)[["batter", "predicted"]]
        pred = pred.dropna(subset=["predicted"])
        if pred["batter"].duplicated().any():
            raise ValueError(f"provider {name!r} returned duplicate batters")
        preds[name] = pred
    if common_players:
        keep = set(realized["batter"])
        for pred in preds.values():
            keep &= set(pred["batter"])
        realized = realized[realized["batter"].isin(keep)]

    frames = []
    for name, pred in preds.items():
        joined = realized.merge(pred, on="batter", how="inner")
        joined["model"] = name
        joined["component"] = component
        frames.append(joined)
    out = pd.concat(frames, ignore_index=True)
    return out[["component", "model", "batter", "predicted",
                "realized_successes", "realized_rate", "trials"]]


def score(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model scores from a backtest() frame (lower is better everywhere)."""
    rows = []
    for (component, model), g in results.groupby(["component", "model"]):
        binomial = COMPONENTS[component].binomial
        rows.append({
            "component": component,
            "model": model,
            "n_players": len(g),
            "total_trials": int(g["trials"].sum()),
            "log_loss": metrics.binomial_log_loss(
                g["predicted"], g["realized_successes"], g["trials"]
            ) if binomial else float("nan"),
            "mae": metrics.weighted_mae(g["predicted"], g["realized_rate"], g["trials"]),
            "rmse": metrics.weighted_rmse(g["predicted"], g["realized_rate"], g["trials"]),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["component", "mae"])
        .reset_index(drop=True)
    )


def calibration(results: pd.DataFrame, model: str, n_bins: int = 10) -> pd.DataFrame:
    """Decile calibration table for one model in a backtest() frame."""
    g = results[results["model"] == model]
    if g.empty:
        raise ValueError(f"model {model!r} not in results")
    return metrics.calibration_table(
        g["predicted"], g["realized_rate"], g["trials"], n_bins=n_bins
    )


def parquet_provider(
    path: str | Path,
    batter_col: str = "batter",
    pred_col: str = "predicted",
) -> Provider:
    """Wrap a projections parquet (e.g. a Bayesian run's output) as a provider.

    The file must correspond to a model trained only through the backtest's
    cutoff — the harness cannot verify that for precomputed files, so keep
    train-year discipline in the filename (e.g. k_rate_train2023.parquet).
    """
    path = Path(path)

    def provider(train: pd.DataFrame, spec: ComponentSpec, predict_year: int) -> pd.DataFrame:
        return frame_provider(pd.read_parquet(path), batter_col, pred_col)(
            train, spec, predict_year)

    return provider


def frame_provider(
    df: pd.DataFrame,
    batter_col: str = "batter",
    pred_col: str = "predicted",
    year_col: str = "projection_year",
) -> Provider:
    """Wrap an in-memory projections frame as a provider.

    If `year_col` is present (multi-year projection files), only rows for
    the backtest's predict year are used.
    """

    def provider(train: pd.DataFrame, spec: ComponentSpec, predict_year: int) -> pd.DataFrame:
        out = df
        if year_col in out.columns:
            out = out[out[year_col] == predict_year]
        return out.rename(
            columns={batter_col: "batter", pred_col: "predicted"}
        )[["batter", "predicted"]]

    return provider
