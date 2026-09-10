"""One hierarchical model over K%, BB% and HR/PA at once (BAS-84).

`src/models/pa_rate.py` fits each per-PA rate alone: a hitter's power tells
the walk model nothing and his strikeouts tell the home-run model nothing.
This module fits the same cells with a per-batter *vector* of abilities drawn
from a multivariate normal whose correlation matrix carries an LKJ(2) prior
and whose scales are per component, so the three components borrow strength
from each other through that correlation. Everything else is the model
`pa_rate.build_model` already builds, once per component: a league random
walk, the season-to-season ability walk (`ability_walk`), the age curve
(quadratic or constrained), handedness, park effect and the park-factor
offset — each with its own parameters per component, and each component
keeping its own Binomial likelihood on its own successes.

**The cells are literally the same cells.** All three components are per-PA
binomials over the same plate appearances, so the cell key
(batter, season, team, stand) and the trial count `n` are identical across
them; only the successes `k`, the park-factor column and the league prior
differ. `prepare_joint_data` therefore runs `pa_rate.prepare_model_data` once
per component on one shared PA frame and *asserts* the index arrays and trial
counts match, rather than building a second, parallel data pipeline that
could drift from the single-component one. That also means every
per-component dict this module hands out is exactly what `pa_rate` would have
produced on its own, which is what lets `generate_joint_projections` reuse
`pa_rate.generate_projections` unchanged.

pymc/arviz are imported at module scope, exactly as `pa_rate` does, so this
module is not importable in CI (`requirements-ci.txt` omits both on purpose).
Nothing that has to run without them may import it — `src/eval/bayes_arm.py`
imports it inside functions, and `src/models/pa_components.py` remains the
pymc-free home of everything a caller needs in order to *name* a component.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from src.models.pa_components import RATE_COMPONENTS, get_component
from src.models.pa_rate import (
    ABILITY_STEP_SIGMA_PRIOR, AGE_PEAK_WINDOW, MIN_PA_THRESHOLD, REFERENCE_AGE,
    ModelOptions, base_pa_columns, generate_projections, prepare_model_data,
)
from src.models.cutoff import apply_cutoff, assert_no_post_cutoff

logger = logging.getLogger(__name__)

# The components this pass fits jointly, in the order their ability vector is
# indexed. Fixed here rather than taken from `RATE_COMPONENTS`'s dict order so
# a correlation named `corr_k_rate__hr_rate` in a fit record always refers to
# the same two entries of the same matrix.
JOINT_COMPONENTS = ("k_rate", "bb_rate", "hr_rate")

# LKJ shape parameter on the ability correlation matrix. eta=2 puts a mild
# prior mass away from the identity's neighbours *and* away from perfect
# correlation — the pre-registration (docs/bayes-joint.md) names it, and its
# prediction 5 is partly a check that no posterior correlation ends up pinned
# at +-1, which this prior already makes a strange place to land.
LKJ_ETA = 2.0
# Prior scale on each component's marginal ability sd, the `sd_dist` the LKJ
# Cholesky prior multiplies. 0.4 is `pa_rate.build_model`'s `sigma_ability`
# HalfNormal scale, unchanged: the joint model's marginal for one component
# is meant to be the single-component model's prior, so that a correlation of
# zero reduces this to three independent fits rather than to three
# differently-shrunk ones.
ABILITY_SD_PRIOR = 0.4


def joint_pa_columns(components=JOINT_COMPONENTS) -> list[str]:
    """The PA columns one load needs to serve every joint component.

    `pa_rate.base_pa_columns` swaps the component's numerator into `is_k`'s
    slot; the joint load needs all of them at once, K%'s slot first so the
    shared prefix reads the same as the single-component list.
    """
    cols = list(base_pa_columns(get_component(components[0])))
    for name in components[1:]:
        num = get_component(name).numerator
        if num not in cols:
            cols.append(num)
    return cols


def load_joint_pa_data(pa_dir, cutoff_date=None, include_pitcher: bool = False,
                       components=JOINT_COMPONENTS) -> pd.DataFrame:
    """`pa_rate.load_pa_data` with every joint component's numerator read.

    One pass over the parquets instead of three. The cutoff is applied as the
    files are read, so post-cutoff PA never reach memory — the same guarantee,
    with the same helper, as the single-component loader.
    """
    from pathlib import Path

    pa_dir = Path(pa_dir)
    files = sorted(pa_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pa_dir}")
    keep = joint_pa_columns(components)
    if cutoff_date is not None:
        keep = keep + ["game_date"]
    if include_pitcher:
        keep = keep + ["pitcher"]
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=keep)
        if cutoff_date is not None:
            df = apply_cutoff(df, cutoff_date)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if cutoff_date is not None:
        assert_no_post_cutoff(data, cutoff_date)
    logger.info("joint load: %d PAs, components %s", len(data), list(components))
    return data


@dataclass
class JointData:
    """The three components' cells, proven to be one set of cells.

    `per_component[name]` is exactly the dict `pa_rate.prepare_model_data`
    returns for that component — same keys, same arrays — so anything on the
    single-component path (projections, diagnostics, the eval harness's data
    summary) works on it unchanged. The joint-only fields are the ones the
    shared graph needs: the successes and park offsets stacked
    (cell, component), and the per-component league priors.
    """
    components: tuple[str, ...]
    per_component: dict
    k: np.ndarray            # (n_obs, n_components) successes
    log_pf: np.ndarray       # (n_obs, n_components) log park factor
    league_init_mu: np.ndarray   # (n_components,)
    age_direction: np.ndarray    # (n_components,)
    shared: dict = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def component_data(self, component: str) -> dict:
        if component not in self.per_component:
            raise KeyError(
                f"this joint fit covers {list(self.components)}, not "
                f"{component!r}")
        return self.per_component[component]

    def summary(self) -> dict:
        d = self.shared
        return {
            "n_cells": int(d["n_obs"]), "n_pa": int(d["n_pa"]),
            "n_batters": int(d["n_batters"]), "n_seasons": int(d["n_seasons"]),
            "n_pitchers": int(d["n_pitchers"]),
            "seasons": [int(s) for s in d["seasons"]],
            "joint_components": list(self.components),
            "league_init_mu": {c: float(m) for c, m
                               in zip(self.components, self.league_init_mu)},
        }


# Index arrays that must agree across components for the joint model to be
# fitting one set of cells rather than three that happen to be the same
# length. `n_trials` is in the list because the whole construction rests on
# every component sharing the plate appearance as its denominator.
_SHARED_ARRAYS = ("batter_idx", "season_idx", "team_idx", "stand_idx",
                  "age_centered", "n_trials")


def prepare_joint_data(
    pa: pd.DataFrame,
    park_factors: pd.DataFrame | None = None,
    min_pa: int = MIN_PA_THRESHOLD,
    birthdates: pd.DataFrame | None = None,
    cutoff_date=None,
    components=JOINT_COMPONENTS,
) -> JointData:
    """Prepare one cell table serving every joint component.

    Runs `pa_rate.prepare_model_data` once per component on the same PA frame
    and checks that the cell keys, the ages and the trial counts came back
    identical. They must: the components differ only in which PA column
    counts as a success, and the cell key does not contain the outcome. If a
    future component ever broke that (a different denominator, a different
    filter) the assertion here is what would say so, instead of a joint model
    silently pairing batter 12's K cells with batter 40's HR cells.
    """
    components = tuple(components)
    unknown = [c for c in components if c not in RATE_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown joint component(s) {unknown}; the per-PA "
                         f"binomial serves {sorted(RATE_COMPONENTS)}")
    if len(components) < 2:
        raise ValueError("a joint model over fewer than two components is "
                         "the single-component model — use src.models.pa_rate")

    per: dict = {}
    for name in components:
        per[name] = prepare_model_data(
            pa, park_factors, min_pa=min_pa, birthdates=birthdates,
            cutoff_date=cutoff_date, include_pitcher=False, component=name,
        )
    first = per[components[0]]
    for name in components[1:]:
        other = per[name]
        for key in _SHARED_ARRAYS:
            if not np.array_equal(np.asarray(first[key]), np.asarray(other[key])):
                raise AssertionError(
                    f"joint components {components[0]!r} and {name!r} did not "
                    f"produce the same cells ({key} differs) — every per-PA "
                    f"binomial shares the plate appearance as its denominator "
                    f"and the cell key carries no outcome, so this can only "
                    f"mean the two runs saw different PA")
        if not np.array_equal(other["batters"], first["batters"]):
            raise AssertionError("joint components disagree about the batter set")

    k = np.column_stack([per[c]["k"] for c in components]).astype(np.int64)
    log_pf = np.column_stack([per[c]["log_pf_k"] for c in components]).astype(np.float64)
    league_init_mu = np.array([per[c]["league_init_mu"] for c in components],
                              dtype=np.float64)
    age_direction = np.array([get_component(c).age_direction for c in components],
                             dtype=np.float64)

    logger.info(
        "joint data ready: %d cells (%d PAs), %d batters, %d seasons, "
        "components %s, league_init %s",
        first["n_obs"], first["n_pa"], first["n_batters"], first["n_seasons"],
        list(components),
        ", ".join(f"{c}={m:.2f}" for c, m in zip(components, league_init_mu)),
    )
    return JointData(components=components, per_component=per, k=k,
                     log_pf=log_pf, league_init_mu=league_init_mu,
                     age_direction=age_direction, shared=first)


def corr_pairs(components) -> list[tuple[str, str]]:
    """Every unordered pair, in the order the correlation Deterministics are
    named. `itertools.combinations` over the component order, so the name
    `corr_k_rate__hr_rate` always means `corr[i, j]` with i < j."""
    return list(itertools.combinations(tuple(components), 2))


def corr_name(a: str, b: str) -> str:
    return f"corr_{a}__{b}"


def sigma_step_name(component: str) -> str:
    return f"sigma_step_{component}"


def build_joint_model(data: JointData,
                      options: ModelOptions | None = None) -> pm.Model:
    """The joint model, on the logit scale, per component j:

        eta[cell, j] = league_trend[season, j]
                     + player_ability[batter, (season,) j]
                     + beta_hand[j] * stand
                     + park_effect[j, team]
                     + age_term[cell, j]
                     + log_pf[cell, j]
        k[:, j] ~ Binomial(n, logistic(eta[:, j]))

    Every term above is `pa_rate.build_model`'s term with a component axis
    added, and the one structural difference is the ability prior:

        ability0[b, :] ~ MvNormal(mu_ability, chol chol')
        chol from LKJCholeskyCov(eta=2, sd_dist=HalfNormal(0.4))

    so the per-component marginal sd keeps the single-component model's
    prior and the correlation between components is what the joint model
    adds. At correlation zero this is three independent `pa_rate` fits sharing
    a sampler; the pre-registration's prediction 1 is that the posterior is
    not there.

    `options.ability_walk` gives each component its own `sigma_step` and each
    batter a walk per component, the same non-centered cumsum construction
    (and the same exact reduction to the flat model at `sigma_step = 0`).
    `options.constrained_age` gives each component its own peak and slopes,
    with the component's own `age_direction` deciding whether the curve is a
    hill or a valley — the same `curve_sign = -age_direction` convention
    `pa_rate.build_model` uses, applied per column.

    No pitcher effect: the joint grid is pre-registered without one, and the
    cell key that a pitcher term needs would multiply the cell count by ~20
    for each of three likelihoods.
    """
    options = options or ModelOptions()
    shared = data.shared
    comps = list(data.components)
    K = len(comps)
    n_seasons = shared["n_seasons"]
    coords = {
        "component": comps,
        # The correlation matrix is square over components, and xarray refuses
        # to work with a variable whose two dims share a name (it warns and
        # then fails silently on most operations), so the second axis gets its
        # own coordinate over the same labels.
        "component_2": comps,
        "batter": shared["batters"],
        "season": shared["seasons"],
        "team": shared["teams"],
        "cell": np.arange(shared["n_obs"]),
    }
    if options.ability_walk:
        coords["season_step"] = shared["seasons"][1:]

    with pm.Model(coords=coords) as model:
        batter_idx = pm.Data("batter_idx", shared["batter_idx"], dims="cell")
        season_idx = pm.Data("season_idx", shared["season_idx"], dims="cell")
        team_idx = pm.Data("team_idx", shared["team_idx"], dims="cell")
        stand_idx = pm.Data("stand_idx", shared["stand_idx"], dims="cell")
        age_c = pm.Data("age_centered", shared["age_centered"], dims="cell")
        log_pf = pm.Data("log_pf_k", data.log_pf, dims=("cell", "component"))
        n_trials = pm.Data("n_trials", shared["n_trials"], dims="cell")

        # ─── League trend per component ──────────────────────────────────
        league_init = pm.Normal("league_init", mu=data.league_init_mu,
                                sigma=0.3, dims="component")
        league_innovations = pm.Normal("league_innovations", mu=0, sigma=0.05,
                                       dims=("season", "component"))
        league_trend = pm.Deterministic(
            "league_trend",
            league_init[None, :] + pt.cumsum(league_innovations, axis=0),
            dims=("season", "component"),
        )

        # ─── Ability vector: MvNormal with an LKJ(2) correlation ─────────
        mu_ability = pm.Normal("mu_ability", mu=0.0, sigma=0.3, dims="component")
        sd_dist = pm.HalfNormal.dist(ABILITY_SD_PRIOR, shape=K)
        chol, corr, stds = pm.LKJCholeskyCov(
            "ability_cov", n=K, eta=LKJ_ETA, sd_dist=sd_dist, compute_corr=True,
        )
        # Named so the single-component projection path finds it: this *is*
        # `pa_rate`'s `sigma_ability`, per component — the marginal sd of the
        # season-0 ability, which is what `_project_unseen` draws an unseen
        # batter's ability from.
        sigma_ability = pm.Deterministic("sigma_ability", stds, dims="component")
        pm.Deterministic("ability_corr", corr,
                         dims=("component", "component_2"))
        for i, j in corr_pairs(comps):
            pm.Deterministic(corr_name(i, j),
                             corr[comps.index(i), comps.index(j)])

        z_ability = pm.Normal("z_ability", mu=0, sigma=1,
                              dims=("batter", "component"))
        # (batter, K): each row is mu + L z, i.e. a draw from the MvNormal
        # with covariance L L'. Non-centered, exactly as the single-component
        # model is, so the batter dimension stays a standard normal.
        ability0 = mu_ability[None, :] + pt.dot(z_ability, chol.T)

        if options.ability_walk:
            sigma_step = pm.HalfNormal("sigma_step_component",
                                       sigma=ABILITY_STEP_SIGMA_PRIOR,
                                       dims="component")
            # Scalars so a fit record (and the sweep's variant summary) can
            # read one component's step size without averaging three of them
            # together; the vector above is what the graph uses.
            for idx, name in enumerate(comps):
                pm.Deterministic(sigma_step_name(name), sigma_step[idx])
            z_step = pm.Normal("z_step", mu=0, sigma=1,
                               dims=("batter", "season_step", "component"))
            steps = pt.cumsum(sigma_step[None, None, :] * z_step, axis=1)
            walk = pt.concatenate(
                [ability0[:, None, :], ability0[:, None, :] + steps], axis=1
            )
            player_ability = pm.Deterministic(
                "player_ability", walk, dims=("batter", "season", "component"))
            ability_term = player_ability[batter_idx, season_idx, :]
        else:
            player_ability = pm.Deterministic(
                "player_ability", ability0, dims=("batter", "component"))
            ability_term = player_ability[batter_idx, :]

        # ─── Handedness, park, age: one per component ────────────────────
        beta_hand = pm.Normal("beta_hand", mu=0.0, sigma=0.2, dims="component")
        # Zero-sum over teams (the last axis), not over components.
        park_effect = pm.ZeroSumNormal("park_effect", sigma=0.05,
                                       dims=("component", "team"))

        if options.constrained_age:
            peak_frac = pm.Beta("peak_frac", alpha=2.0, beta=2.0, dims="component")
            lo, hi = AGE_PEAK_WINDOW
            peak_age = pm.Deterministic("peak_age", lo + (hi - lo) * peak_frac,
                                        dims="component")
            slope_young = pm.HalfNormal("slope_young", sigma=0.02, dims="component")
            slope_old = pm.HalfNormal("slope_old", sigma=0.02, dims="component")
            age = age_c + REFERENCE_AGE
            d = age[:, None] - peak_age[None, :]
            curve_sign = -data.age_direction
            age_term = curve_sign[None, :] * pt.where(
                d > 0, slope_old[None, :] * d, -slope_young[None, :] * d)
        else:
            beta_age = pm.Normal("beta_age", mu=0.0, sigma=0.02, dims="component")
            beta_age2 = pm.Normal("beta_age2", mu=-0.005 * data.age_direction,
                                  sigma=0.01, dims="component")
            age_term = (beta_age[None, :] * age_c[:, None]
                        + beta_age2[None, :] * (age_c[:, None] ** 2))

        eta = (
            league_trend[season_idx, :]
            + ability_term
            + beta_hand[None, :] * stand_idx[:, None]
            + park_effect.T[team_idx, :]
            + age_term
            + log_pf
        )
        p = pm.math.invlogit(eta)

        # ─── One Binomial per component, on its own successes ────────────
        for idx, name in enumerate(comps):
            pm.Binomial(
                get_component(name).obs_name,
                n=n_trials, p=p[:, idx],
                observed=data.k[:, idx], dims="cell",
            )

    n_params = (
        K                                        # league_init
        + K * n_seasons                          # league_innovations
        + K + K * (K + 1) // 2                   # mu_ability, chol
        + shared["n_batters"] * K                # z_ability
        + (shared["n_batters"] * max(n_seasons - 1, 0) * K + K
           if options.ability_walk else 0)
        + K                                      # beta_hand
        + K * (shared["n_teams"] - 1)            # park_effect
        + K * (3 if options.constrained_age else 2)
    )
    logger.info(
        "Joint model built [%s]: ~%d free parameters, %d cells (%d PAs) x %d "
        "likelihoods%s%s", ",".join(comps), n_params, shared["n_obs"],
        shared["n_pa"], K,
        ", ability_walk" if options.ability_walk else "",
        ", constrained_age" if options.constrained_age else "",
    )
    return model


# ─── reading one component back out ────────────────────────────────────────

# Variables `pa_rate.generate_projections` and `_age_term_function` read off a
# trace. Anything with a `component` dim is sliced to the requested component;
# anything without is passed through. Names absent from a given fit (the age
# block the other variant would have written, `sigma_step` under the flat
# model) are simply skipped, which is what the reader downstream already
# expects — it detects the age curve from the names present.
_VIEW_NAMES = (
    "league_init", "league_innovations", "league_trend",
    "mu_ability", "sigma_ability", "player_ability", "beta_hand",
    "park_effect", "beta_age", "beta_age2",
    "peak_age", "slope_young", "slope_old",
)


def component_posterior(trace, component: str):
    """A one-component view of a joint trace, shaped like a single-component one.

    Returns an object with a `.posterior` xarray Dataset whose variables have
    the names and dimensions `pa_rate.generate_projections` expects — which is
    the whole reason the joint model names its per-component parameters after
    the single-component ones. Slicing (not copying) an xarray Dataset costs
    nothing, so three components' projections cost one trace.
    """
    post = trace.posterior
    out: dict = {}
    for name in _VIEW_NAMES:
        if name in post:
            var = post[name]
            out[name] = (var.sel(component=component, drop=True)
                         if "component" in var.dims else var)
    step = sigma_step_name(component)
    if step in post:
        out["sigma_step"] = post[step]
    return SimpleNamespace(posterior=xr.Dataset(out))


def generate_joint_projections(trace, data: JointData, component: str,
                               projection_year: int, recent_seasons: int = 3,
                               unseen: pd.DataFrame | None = None) -> pd.DataFrame:
    """One component's projections out of the joint posterior.

    `pa_rate.generate_projections`, unchanged, on the component's own view of
    the trace and its own `prepare_model_data` dict — so a joint HR/PA
    projection is computed by exactly the code a single-component HR/PA
    projection is, and any difference between the two arms is the posterior,
    not the projection arithmetic.
    """
    return generate_projections(
        component_posterior(trace, component), data.component_data(component),
        projection_year=projection_year, recent_seasons=recent_seasons,
        unseen=unseen,
    )


def _summarise(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype="float64").ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    return {
        "mean": float(vals.mean()), "sd": float(vals.std()),
        "q2.5": float(np.percentile(vals, 2.5)),
        "q97.5": float(np.percentile(vals, 97.5)),
    }


def joint_param_summary(trace, components=JOINT_COMPONENTS) -> dict:
    """Posterior mean and 95% interval for every pairwise ability correlation
    and every `sigma_step`, which is what docs/bayes-joint.md's predictions 1
    and 5 are read off.

    An interval, not a mean and an sd: "the interval excludes zero" is the
    pre-registered test, and a correlation posterior near the edge of [-1, 1]
    is exactly the case where a normal approximation from a mean and an sd
    would answer it wrongly. Reads straight off `.posterior[name].values`, so
    a plain object with a `.posterior` mapping is enough to test it.
    """
    post = getattr(trace, "posterior", None) if trace is not None else None
    out: dict = {}
    if post is None:
        return out
    for a, b in corr_pairs(components):
        name = corr_name(a, b)
        if name in post:
            s = _summarise(post[name].values)
            if s:
                out[name] = s
    for name in components:
        step = sigma_step_name(name)
        if step in post:
            s = _summarise(post[step].values)
            if s:
                out[step] = s
    return out
