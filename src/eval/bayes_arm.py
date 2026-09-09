"""The Bayesian rate arm as a backtest provider — refit at the cutoff (BAS-59).

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

**Components (BAS-73).** The arm served `k_rate` only until BB/PA and HR/PA
turned out to be the same hierarchical binomial with a different numerator
(`src.models.pa_rate`). `BayesArmConfig.component` picks which, and it is
part of the memoization key: one fit per (component, cutoff, predict year),
never a K% fit handed back under a BB% label. BABIP and ISO still raise —
they are not per-PA binomials and need their own denominators.

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
    `src.models.pa_rate.SAMPLER_KWARGS`). Whatever a run used is echoed back
    on every fit so a table can be labelled with its own scale instead of
    being mistaken for the full thing.
    """
    pa_dir: Path = DEFAULT_PA_DIR
    # Which per-PA rate this arm fits (`src.models.pa_rate.RATE_COMPONENTS`).
    # Defaults to k_rate so every pre-BAS-73 construction of this config
    # means exactly what it did.
    component: str = "k_rate"
    seasons: tuple[int, ...] | None = None   # None = every parquet in pa_dir
    min_pa: int = 50
    include_pitcher: bool = True
    max_batters: int | None = None           # reduced-scale subsampling
    # Structural variants (docs/bayes-variants.md), passed straight through to
    # `src.models.pa_k_rate.ModelOptions`.
    ability_walk: bool = False
    constrained_age: bool = False
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

    def model_options(self):
        """The structural variant this config asks for."""
        from src.models.pa_rate import ModelOptions

        return ModelOptions(ability_walk=self.ability_walk,
                            constrained_age=self.constrained_age)

    def rate_component(self):
        """The `RateComponent` this arm fits, validated. Raises for a
        component this model cannot serve (BABIP, ISO) rather than at the
        first missing column three function calls later."""
        from src.models.pa_components import get_component

        return get_component(self.component)

    def projected_col(self) -> str:
        """Column `generate_projections` writes the point estimate to."""
        return self.rate_component().projected_col

    def variant(self) -> str:
        """Short, stable name for the structure — "flat" for the arm on the board.

        A results table keyed on this is how a sweep keeps four arms apart, so
        it is a slug, not prose: joined flag names in declaration order, and
        the empty set is "flat" rather than "" so no row is ever unlabelled.
        """
        on = [n for n in ("ability_walk", "constrained_age") if getattr(self, n)]
        return "+".join(on) if on else "flat"

    def label(self) -> str:
        pitch = "pitcher" if self.include_pitcher else "no-pitcher"
        ability = "ability=walk" if self.ability_walk else "ability=flat"
        age = "age=constrained" if self.constrained_age else "age=quadratic"
        return (f"{self.component}, "
                f"{self.chains}x{self.draws} draws (tune {self.tune}), "
                f"{self.nuts_sampler}, {pitch}, {ability}, {age}"
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

    @property
    def component(self) -> str:
        """Which rate this fit is of, read off its own config so a fit and
        its label cannot disagree."""
        return self.config.component

    def project(self, unseen: pd.DataFrame | None = None) -> pd.DataFrame:
        """Re-project from the same posterior, optionally covering unseen batters.

        Kept separate from the fit so adding population-level projections for
        batters the model never saw costs a numpy pass, not a second MCMC run.
        """
        from src.models.pa_rate import generate_projections

        if self.trace is None or self.model_data is None:
            raise RuntimeError("this fit did not keep its posterior")
        self.projections = generate_projections(
            self.trace, self.model_data,
            projection_year=self.predict_year, unseen=unseen,
        )
        return self.projections


def _load_cut_pa(config: BayesArmConfig, cutoff_date: str) -> pd.DataFrame:
    from src.models.pa_rate import load_pa_data

    pa = load_pa_data(config.pa_dir, cutoff_date=cutoff_date,
                      include_pitcher=config.include_pitcher,
                      component=config.rate_component())
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
    """Fit the PA-level rate model on everything strictly before `cutoff_date`.

    Which rate comes from `config.component` (default `k_rate`); the model is
    one object across components (`src.models.pa_rate`).

    The leakage guard runs twice on the way in (`load_pa_data` and
    `prepare_model_data` both call `assert_no_post_cutoff`), so a post-cutoff
    PA cannot reach the likelihood.
    """
    from src.models.pa_rate import (
        build_model, generate_projections, load_park_factors,
        model_diagnostics, prepare_model_data, sample_model,
    )
    from src.models.cutoff import cutoff_exposure

    config = config or BayesArmConfig()
    comp = config.rate_component()
    predict_year = predict_year or pd.Timestamp(cutoff_date).year

    pa = _load_cut_pa(config, cutoff_date)
    exposure = cutoff_exposure(pa, cutoff_date)
    logger.info("bayes arm [%s] @ %s: %s", comp.name, cutoff_date, exposure)

    data = prepare_model_data(
        pa, load_park_factors(), min_pa=config.min_pa,
        cutoff_date=cutoff_date, include_pitcher=config.include_pitcher,
        component=comp,
    )
    model = build_model(data, config.model_options())
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
            "component": comp.name,
            "league_init_mu": float(data["league_init_mu"]),
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

    The fit is memoized across calls, so scoring one component at one cutoff
    several times costs one fit. The key is `(component, cutoff_date,
    predict_year)` — the component is in it because the same provider object
    can legitimately be asked for more than one, and a cache that keyed only
    on the date would hand back a K% frame for a BB% request with the right
    column name and the wrong numbers. That is the failure mode
    `make_variant_providers` in `scripts/run_intraseason_backtest_dense.py`
    guards against one level up for *variants*; this is the same guard for
    components, at the level where the cache lives.

    A `spec` naming a component this arm cannot serve raises rather than
    silently returning some other component's number under its name. Since
    BAS-73 that is BABIP and ISO only: both need a denominator that is not
    the plate appearance (balls in play, at bats) and neither is a count of
    independent binomial trials in the ISO case, so they are separate models
    rather than another entry in `RATE_COMPONENTS`.

    A `spec` that disagrees with `config.component` also raises: the harness
    decides which component it is scoring, the config decides which one gets
    fit, and if those two ever disagree the scoring is wrong in a way no
    number downstream would reveal.

    `on_fit(fit)` is called once with the `BayesFit`, so a caller can record
    the diagnostics and the scale the run actually used.
    """
    config = config or BayesArmConfig()
    comp = config.rate_component()
    cache: dict = {}

    def provider(train: pd.DataFrame, spec, year: int) -> pd.DataFrame:
        from src.models.pa_components import RATE_COMPONENTS

        if spec.name not in RATE_COMPONENTS:
            raise ValueError(
                f"the bayes arm serves {sorted(RATE_COMPONENTS)} — per-PA "
                f"binomials — not {spec.name!r}; BABIP is per ball in play "
                f"and ISO is not a count of trials at all, so both need "
                f"their own model"
            )
        if spec.name != comp.name:
            raise ValueError(
                f"this provider was built to fit {comp.name!r} but the "
                f"harness asked it to score {spec.name!r} — one provider per "
                f"component, or the fit and the label come apart"
            )
        target_year = predict_year or year
        key = (comp.name, str(cutoff_date), int(target_year))
        if key not in cache:
            # One fit, two projection passes: the first says which batters the
            # model covers, the second adds the rest from the fitted
            # population. Only the fit is expensive.
            fit = fit_bayes_k_rate(cutoff_date, target_year, config)
            unseen = unseen_from_train(
                train, fit.projections["batter"], target_year)
            if len(unseen):
                logger.info("bayes arm [%s]: %d batters projected from the "
                            "fitted population", comp.name, len(unseen))
                fit.project(unseen)
            cache[key] = fit
            if on_fit is not None:
                on_fit(fit)
        fit = cache[key]
        out = fit.projections[["batter", comp.projected_col]].rename(
            columns={comp.projected_col: "predicted"})
        return out[np.isfinite(out["predicted"])]

    return provider


# ─── component-neutral names ───────────────────────────────────────────────
# The `_k_rate` names are what every existing call site uses, and what the
# dense sweep's tests monkeypatch, so they stay the real definitions rather
# than becoming aliases of a renamed pair — a `monkeypatch.setattr` doing
# exactly the right thing should not stop working because of a rename. These
# two forward through the module globals rather than binding the function
# objects, so a patched `fit_bayes_k_rate` is what they call too.

def fit_bayes_rate(*args, **kwargs) -> BayesFit:
    """`fit_bayes_k_rate` under a name that does not claim a component."""
    return fit_bayes_k_rate(*args, **kwargs)


def bayes_rate_provider(*args, **kwargs):
    """`bayes_k_rate_provider` under a name that does not claim a component."""
    return bayes_k_rate_provider(*args, **kwargs)
