"""The Bayesian K% arm as a backtest provider — refit at the cutoff (BAS-59).

Until this module existed the only Bayesian arm anywhere in the eval was
`bayes_preseason`: a fixed file fit through 2025 and scored unchanged at every
2026 cutoff. Every published comparison therefore ran a model that had never
seen a 2026 plate appearance against Marcel fed 2026 through the day before,
and the harness itself measures in-season information at 5-6% of K% MAE —
the same order as the entire reported deficit. This is the arm that makes the
fight fair: the same estimator, refit on exactly the PA the baselines see.

Provider shape is the harness's: `(train, spec, predict_year) -> [batter,
predicted]`. The training frame is *not* the model's input — the model reads
PA rows, which is the whole reason it needs a cutoff of its own — but it is
read for two things: the batter set to cover, and the ages of batters the fit
never saw.

pymc is imported inside the functions, not at module scope, so importing this
module is free in CI (which installs `requirements-ci.txt`, no pymc, no
arviz). Nothing here runs during the test suite.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PA_DIR = Path("data/parquet/pa_outcomes")


@dataclass(frozen=True)
class BayesArmConfig:
    """Everything that decides what the arm actually is, in one object.

    The sampler fields exist because the honest local answer is a *reduced*
    fit: this sandbox has no JAX, so `nuts_sampler="pymc"` and a few hundred
    draws is what is reachable. The full Modal refit overrides them (see
    `src.models.pa_k_rate.SAMPLER_KWARGS`). Whatever a run used is echoed back
    on every fit so a table can be labelled with its own scale instead of
    being mistaken for the full thing.
    """
    pa_dir: Path = DEFAULT_PA_DIR
    seasons: tuple[int, ...] | None = None   # None = every parquet in pa_dir
    min_pa: int = 50
    include_pitcher: bool = True
    max_batters: int | None = None           # reduced-scale subsampling
    # Sampler
    draws: int = 500
    tune: int = 500
    chains: int = 2
    cores: int = 2
    target_accept: float = 0.9
    nuts_sampler: str = "pymc"
    random_seed: int = 59
    cache_dir: Path | None = None

    def sampler_kwargs(self) -> dict:
        return dict(
            draws=self.draws, tune=self.tune, chains=self.chains,
            cores=self.cores, target_accept=self.target_accept,
            nuts_sampler=self.nuts_sampler, random_seed=self.random_seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": False},
        )

    def label(self) -> str:
        pitch = "pitcher" if self.include_pitcher else "no-pitcher"
        return (f"{self.chains}x{self.draws} draws (tune {self.tune}), "
                f"{self.nuts_sampler}, {pitch}"
                + (f", <={self.max_batters} batters" if self.max_batters else ""))


@dataclass
class BayesFit:
    """One fitted model plus everything needed to score and to label it."""
    cutoff_date: str
    predict_year: int
    projections: pd.DataFrame
    diagnostics: dict
    config: BayesArmConfig
    data_summary: dict = field(default_factory=dict)
    trace: object = None
    model_data: dict | None = None

    def project(self, unseen: pd.DataFrame | None = None) -> pd.DataFrame:
        """Re-project from the same posterior, optionally covering unseen batters.

        Kept separate from the fit so adding population-level projections for
        batters the model never saw costs a numpy pass, not a second MCMC run.
        """
        from src.models.pa_k_rate import generate_projections

        if self.trace is None or self.model_data is None:
            raise RuntimeError("this fit did not keep its posterior")
        self.projections = generate_projections(
            self.trace, self.model_data,
            projection_year=self.predict_year, unseen=unseen,
        )
        return self.projections


def _load_cut_pa(config: BayesArmConfig, cutoff_date: str) -> pd.DataFrame:
    from src.models.pa_k_rate import load_pa_data

    pa = load_pa_data(config.pa_dir, cutoff_date=cutoff_date,
                      include_pitcher=config.include_pitcher)
    if config.seasons is not None:
        pa = pa[pa["game_year"].isin(config.seasons)].copy()
    if config.max_batters:
        # Reduced scale: keep the busiest batters, which is the population the
        # harness scores anyway (it needs 100 realized trials after the
        # cutoff). Deterministic, so a rerun is the same fit.
        counts = pa.groupby("batter").size().sort_values(ascending=False)
        keep = set(counts.index[: config.max_batters])
        pa = pa[pa["batter"].isin(keep)].copy()
    return pa


def fit_bayes_k_rate(
    cutoff_date: str,
    predict_year: int | None = None,
    config: BayesArmConfig | None = None,
    unseen: pd.DataFrame | None = None,
) -> BayesFit:
    """Fit the PA-level K% model on everything strictly before `cutoff_date`.

    The leakage guard runs twice on the way in (`load_pa_data` and
    `prepare_model_data` both call `assert_no_post_cutoff`), so a post-cutoff
    PA cannot reach the likelihood.
    """
    from src.models.pa_k_rate import (
        build_model, generate_projections, load_park_factors, model_diagnostics,
        prepare_model_data, sample_model,
    )
    from src.models.cutoff import cutoff_exposure

    config = config or BayesArmConfig()
    predict_year = predict_year or pd.Timestamp(cutoff_date).year

    pa = _load_cut_pa(config, cutoff_date)
    exposure = cutoff_exposure(pa, cutoff_date)
    logger.info("bayes arm @ %s: %s", cutoff_date, exposure)

    data = prepare_model_data(
        pa, load_park_factors(), min_pa=config.min_pa,
        cutoff_date=cutoff_date, include_pitcher=config.include_pitcher,
    )
    model = build_model(data)
    trace = sample_model(model, **config.sampler_kwargs())
    diagnostics = model_diagnostics(trace)

    projections = generate_projections(
        trace, data, projection_year=predict_year, unseen=unseen,
    )
    return BayesFit(
        cutoff_date=str(cutoff_date),
        predict_year=int(predict_year),
        projections=projections,
        diagnostics=diagnostics,
        config=config,
        trace=trace,
        model_data=data,
        data_summary={
            **exposure,
            "n_cells": int(data["n_obs"]),
            "n_pa": int(data["n_pa"]),
            "n_batters": int(data["n_batters"]),
            "n_pitchers": int(data["n_pitchers"]),
            "n_seasons": int(data["n_seasons"]),
            "seasons": [int(s) for s in data["seasons"]],
        },
    )


def unseen_from_train(
    train: pd.DataFrame, fitted_batters, predict_year: int
) -> pd.DataFrame:
    """Batters the harness will score that the fit never saw, with their ages.

    A September call-up at an April cutoff, and anyone the `min_pa` floor or a
    reduced-scale subsample dropped. Without this they simply vanish from the
    arm's coverage, and `common_players=True` then quietly shrinks the set
    *every* arm is scored on — which changes the comparison rather than making
    it fair. Ages come from the training frame's most recent row, aged forward
    to the predict year, the same way `marcel` gets them.
    """
    fitted = set(int(b) for b in fitted_batters)
    missing = sorted(set(int(b) for b in train["batter"].unique()) - fitted)
    if not missing:
        return pd.DataFrame(columns=["batter", "age"])

    age = pd.Series(dtype="float64")
    if "age" in train.columns:
        last = (train.dropna(subset=["age"]).sort_values("season")
                .groupby("batter").agg(age=("age", "last"),
                                       season=("season", "last")))
        age = last["age"] + (predict_year - last["season"])
    out = pd.DataFrame({"batter": missing})
    out["age"] = out["batter"].map(age)
    return out


def bayes_k_rate_provider(
    cutoff_date: str,
    predict_year: int | None = None,
    config: BayesArmConfig | None = None,
    on_fit=None,
):
    """A `(train, spec, predict_year)` provider that refits at the cutoff.

    The fit is memoized across calls, so scoring several components at one
    cutoff costs one fit — though only `k_rate` is served: this module wraps
    `src.models.pa_k_rate`, and the other four components' models live in
    `modal_functions/app.py` as separate functions with their own
    denominators. Asking for another component raises rather than silently
    returning the K% number under a different name.

    `on_fit(fit)` is called once with the `BayesFit`, so a caller can record
    the diagnostics and the scale the run actually used.
    """
    config = config or BayesArmConfig()
    cache: dict = {}

    def provider(train: pd.DataFrame, spec, year: int) -> pd.DataFrame:
        if spec.name != "k_rate":
            raise ValueError(
                f"the bayes arm serves k_rate only, not {spec.name!r} — the "
                f"other components are separate models (modal_functions/app.py)"
            )
        target_year = predict_year or year
        key = (str(cutoff_date), int(target_year))
        if key not in cache:
            # One fit, two projection passes: the first says which batters the
            # model covers, the second adds the rest from the fitted
            # population. Only the fit is expensive.
            fit = fit_bayes_k_rate(cutoff_date, target_year, config)
            unseen = unseen_from_train(
                train, fit.projections["batter"], target_year)
            if len(unseen):
                logger.info("bayes arm: %d batters projected from the "
                            "fitted population", len(unseen))
                fit.project(unseen)
            cache[key] = fit
            if on_fit is not None:
                on_fit(fit)
        fit = cache[key]
        out = fit.projections[["batter", "projected_k_rate"]].rename(
            columns={"projected_k_rate": "predicted"})
        return out[np.isfinite(out["predicted"])]

    return provider
