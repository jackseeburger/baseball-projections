"""One latent talent per hitter, read by several observation channels (BAS-85).

`src/models/pa_covariates.py` (BAS-83) put contact quality into the
hierarchical model as a **regressor**: a pooled coefficient times a
standardized aggregate, added to the batter's logit rate. That failed, and
`docs/bayes-covariates.md` diagnosed why — the coefficient is pooled across
players while the aggregate is computed in each cell's own window, so at a
May cutoff the covariate is a forty-batted-ball average multiplied by a
number fitted mostly off full seasons, and the arm imports the thin window's
noise at full weight.

This module is the other way round. Contact quality is not a regressor on the
rate; it is a **second observation of the same latent state**, with its own
counts:

    theta_hr[batter, season]    latent power    (the ability the HR logit uses)
    theta_k [batter, season]    latent contact  (the ability the K logit uses)

    HR      ~ Binomial(PA,     logistic(... + theta_hr ...))   outcome channel
    barrels ~ Binomial(BBE,    logistic(a_barrel + l_barrel * z_hr))
    EV_mean ~ Normal(a_ev + l_ev * z_hr, sigma_ev / sqrt(BBE))
    whiffs  ~ Binomial(swings, logistic(a_whiff + l_whiff * z_k))

Because each channel is an **observed node carrying its own trial count**, a
batter with forty batted balls contributes a likelihood worth forty batted
balls — by construction, with no coefficient estimated elsewhere and then
applied to him at full strength. That is the whole structural difference from
BAS-83, and it is what prediction 2 in `docs/bayes-measurement.md`
(early-season is where it wins) is a test of.

**Identification.** The outcome loading is fixed at 1: theta enters the HR
and K logits with coefficient 1, exactly as `player_ability` already does in
`src/models/pa_joint.py`. The extra channels load on the *centred* latent
`theta - mu_ability`, and the loading carries the scale.

An earlier version divided by `sigma_ability` so that a loading read directly
in "channel units per standard deviation of ability". That is the natural
unit to report, and it is still what `measurement_param_summary` reports -
but it is a catastrophic thing to put in the *graph*. A parameter in a
denominator is the classic funnel, and on this model it made the posterior
unsamplable by NumPyro: two chains at 2 x 300 came back with R-hat 2.23,
ESS 3 and the two power loadings collapsed onto zero with prior-wide
intervals (`lambda_barrel` +0.005 [-0.613, +0.611] against PyMC's +0.583
[+0.544, +0.623]). Raising `target_accept` to 0.99 removed every divergence
and left R-hat at 2.24: the chains were not taking bad steps, they were in
different places and could not mix. Since NumPyro is the sweep's default
sampler, that would have been a *spurious* failure of prediction 1 - the
channels reported as not loading because two chains disagreed - and it would
have looked exactly like a real result.

So the division is gone and the loading absorbs the scale. The prior is
`Normal(0, LAMBDA_SIGMA)` with `LAMBDA_SIGMA = 2.5`, which is
`1 / ABILITY_SD_PRIOR`: at the prior's own ability scale of 0.4 that is the
same prior predictive the per-sd `Normal(0, 1)` gave, so nothing about how
much the prior is allowed to say has changed. Per-sd loadings are recovered
after the fact by multiplying each draw by that draw's `sigma_ability`, which
is what the summary reports - so the numbers remain directly comparable to
the per-sd parameterisation, and every pre-registered test on them (excludes
zero; pairwise |corr| < 0.95) is invariant to the rescaling anyway.

What the channels must *not* read is `z_ability`, the joint model's own
non-centred standard normal. It is tempting - it is right there, and it is
already standard normal - but it is a different quantity twice over. The LKJ
Cholesky mixes the components (`ability0 = mu + z @ L'`), so `z[:, 0]` is the
standardised K ability but `z[:, 1]` is the part of HR ability *orthogonal*
to K ability: at this model's own fitted values its correlation with the
standardised HR ability is 0.886, not 1. And `z_ability` is a season-0
quantity, while the channels are per (batter, season) and must read the
season's state, which under `ability_walk` is `ability0` plus the walk.

**Everything else is the joint model.** `build_measurement_model` builds
`pa_joint.build_joint_model` over the two components in scope and then
re-enters that model to add the channels, reading the `player_ability`
Deterministic the joint model already writes. The latent states, the league
random walk, the ability walk, the age curve, handedness and park are
therefore the BAS-84 graph unchanged, and the only thing this ticket adds is
the observation channels and their loadings. A run with every channel absent
is bit-for-bit the joint model.

**Walk-forward.** The channels are summed from the committed monthly
artifacts under the same month-lagged rule as `pa_covariates.season_cutoff`
and `src.projections.ros.contact_cutoff`: a season before the cutoff's year
is read whole, the cutoff's own season is read up to the last month boundary
on or before the cutoff. A July 15 cutoff sees batted balls through June 30,
never through July 14, because the buckets are monthly and rounding a cutoff
forward is leakage.

pymc/arviz are imported inside the graph builder rather than at module scope,
unlike `pa_rate` and `pa_joint`: everything above that line — the channel
preparation, which is the part with a walk-forward rule worth testing — uses
only numpy and pandas, and is therefore testable in CI, the same split
`src/models/pa_covariates.py` uses.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The two components `docs/bayes-measurement.md` puts in scope. BB% is not
# here: the pre-registration names K% and HR/PA, and a whiff channel loading
# on a walk-rate latent is not a claim this ticket made.
MEASUREMENT_COMPONENTS = ("k_rate", "hr_rate")

# Which latent each channel reads. The name on the left is the channel, the
# name on the right is the component whose ability *is* that latent state —
# `theta_hr` is the HR/PA ability and `theta_k` is the K% ability, because
# the outcome loading is fixed at 1 and the ability the outcome logit uses is
# therefore the latent by definition.
CHANNEL_COMPONENT = {"barrel": "hr_rate", "ev": "hr_rate", "whiff": "k_rate"}
CHANNELS = ("barrel", "ev", "whiff")

# Prior scale on every channel loading. The pre-registration names
# Normal(0, 1) on a loading read *per standard deviation of ability*; the
# graph carries the loading per *logit* of ability instead (see the module
# docstring - the division by sigma_ability is a funnel), so the prior scale
# is divided by the ability scale to leave the prior predictive where the
# pre-registration put it. 1 / 0.4 = 2.5.
LAMBDA_SIGMA_PER_SD = 1.0
LAMBDA_SIGMA = LAMBDA_SIGMA_PER_SD / 0.4
# Prior scale on each channel intercept. Wide: the intercept absorbs the
# league's own barrel rate and whiff rate, which the latent must not have to.
ALPHA_SIGMA = 2.0
# Prior scale on the exit-velocity residual sd, in per-batted-ball sd units,
# where the truth is by construction near 1 (see `channel_arrays`).
SIGMA_EV_PRIOR = 1.0
# A cell with fewer tracked batted balls than this contributes no barrel and
# no EV observation, and one with fewer swings contributes no whiff. Not a
# quality filter — the binomial and the 1/sqrt(BBE) scale already weight a
# thin cell down correctly — but a guard on the EV channel's normal
# approximation to the mean of a handful of numbers.
MIN_BBE = 5
MIN_SWINGS = 20

# Stamped on every fit record. The channels' latent has been parameterised two
# ways and only one of them samples (see the module docstring); a fit record
# that does not say which one produced it cannot be compared with one from the
# other, and the numbers differ in *units* as well as in quality.
PARAMETERISATION = "centred-latent/loading-carries-scale"

# Columns each artifact must carry, checked on the way in so a renamed
# artifact column is an error here and not three silent zeros later.
CQ_COLUMNS = ("side", "player", "season", "month", "bbe", "sum_ev",
              "sum_ev2", "n_barrel")
SWING_COLUMNS = ("batter", "season", "month", "swings_z", "swings_o",
                 "whiff_z", "whiff_o")


def _require(df: pd.DataFrame, columns, what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; got "
                         f"{sorted(df.columns)[:12]}...")


def _season_window(monthly: pd.DataFrame, season: int, cutoff_date,
                   season_col: str = "season") -> pd.DataFrame:
    """The rows of a monthly artifact inside one season's pre-cutoff window.

    A season before the cutoff's year is whole; the cutoff's own season stops
    at the last month boundary on or before the cutoff (`month <
    cutoff.month`), which is `pa_covariates.season_cutoff`'s rule expressed
    as a row filter rather than as a date. A season *after* the cutoff's year
    has no window at all — it cannot happen on the walk-forward path, and
    returning its rows would be leakage, so it returns nothing.
    """
    cutoff = pd.Timestamp(cutoff_date)
    season = int(season)
    rows = monthly[monthly[season_col] == season]
    if season > cutoff.year:
        return rows.iloc[:0]
    if season == cutoff.year:
        rows = rows[rows["month"] < cutoff.month]
    return rows


def _grid(values: pd.Series, index: dict) -> np.ndarray:
    """Row positions of a player column in the model's batter order, -1 for
    a player the model does not carry."""
    return np.array([index.get(int(p), -1) for p in values], dtype="int64")


def channel_arrays(batters, seasons, contact_monthly: pd.DataFrame,
                   swing_monthly: pd.DataFrame, cutoff_date) -> dict:
    """Per (batter, season) channel counts in the model's own window.

    Returns a dict of (n_batters, n_seasons) arrays:

        bbe, n_barrel, sum_ev, sum_ev2     from the contact-quality artifact
        swings, whiffs                     from the swing-decisions artifact

    summed over the months this season is allowed to see. Zero everywhere the
    batter has no tracked contact (or no swings) in his window, which the
    caller turns into "no observation for that cell" rather than into a zero
    rate — an unobserved channel must contribute nothing to the likelihood,
    not evidence of a zero barrel rate.
    """
    _require(contact_monthly, CQ_COLUMNS, "the contact-quality artifact")
    _require(swing_monthly, SWING_COLUMNS, "the swing-decisions artifact")

    batters = np.asarray(batters)
    seasons = np.asarray(seasons)
    index = {int(b): i for i, b in enumerate(batters)}
    shape = (len(batters), len(seasons))
    out = {k: np.zeros(shape, dtype="float64")
           for k in ("bbe", "n_barrel", "sum_ev", "sum_ev2", "swings", "whiffs")}

    hitters = contact_monthly[contact_monthly["side"] == "hitter"]
    for s_idx, season in enumerate(seasons):
        cq = _season_window(hitters, season, cutoff_date)
        if len(cq):
            g = cq.groupby("player", as_index=False)[
                ["bbe", "sum_ev", "sum_ev2", "n_barrel"]].sum()
            rows = _grid(g["player"], index)
            keep = rows >= 0
            for col in ("bbe", "sum_ev", "sum_ev2", "n_barrel"):
                out[col][rows[keep], s_idx] = g.loc[keep, col].to_numpy("float64")

        sw = _season_window(swing_monthly, season, cutoff_date)
        if len(sw):
            g = sw.assign(
                swings=sw["swings_z"] + sw["swings_o"],
                whiffs=sw["whiff_z"] + sw["whiff_o"],
            ).groupby("batter", as_index=False)[["swings", "whiffs"]].sum()
            rows = _grid(g["batter"], index)
            keep = rows >= 0
            for col in ("swings", "whiffs"):
                out[col][rows[keep], s_idx] = g.loc[keep, col].to_numpy("float64")

    logger.info(
        "channels @ %s: %.1f%% of (batter, season) cells have >=%d batted "
        "balls, %.1f%% have >=%d swings",
        cutoff_date, 100.0 * float((out["bbe"] >= MIN_BBE).mean()), MIN_BBE,
        100.0 * float((out["swings"] >= MIN_SWINGS).mean()), MIN_SWINGS,
    )
    return out


@dataclass
class MeasurementChannels:
    """The three channels, flattened to observed vectors with their indices.

    One entry per *observed* (batter, season) cell per channel, not a dense
    grid with a mask: pymc has no missing-data story that keeps a Binomial's
    `n` aligned, and an unobserved channel must contribute exactly nothing.
    `batter_idx` / `season_idx` index the joint model's `player_ability`.

    `ev_z` is the cell's mean exit velocity standardized by the **per batted
    ball** mean and sd pooled over every observed cell, so that
    `sigma_ev / sqrt(bbe)` is the standard error of a mean of `bbe` draws and
    a fitted `sigma_ev` near 1 says the channel's residual noise is exactly
    sampling noise. `ev_mu` and `ev_sd` are kept so a posterior loading can be
    read back in miles per hour.
    """
    barrel_batter: np.ndarray
    barrel_season: np.ndarray
    barrel_k: np.ndarray
    barrel_n: np.ndarray
    ev_batter: np.ndarray
    ev_season: np.ndarray
    ev_z: np.ndarray
    ev_n: np.ndarray
    whiff_batter: np.ndarray
    whiff_season: np.ndarray
    whiff_k: np.ndarray
    whiff_n: np.ndarray
    ev_mu: float = 0.0
    ev_sd: float = 1.0
    n_batters: int = 0
    n_seasons: int = 0

    def counts(self) -> dict:
        return {"barrel": int(self.barrel_k.size),
                "ev": int(self.ev_z.size),
                "whiff": int(self.whiff_k.size)}

    def summary(self) -> dict:
        """What the fit record keeps about the channels' exposure."""
        return {
            "n_channel_cells": self.counts(),
            "n_bbe": int(self.barrel_n.sum()),
            "n_swings": int(self.whiff_n.sum()),
            "ev_mu": float(self.ev_mu), "ev_sd": float(self.ev_sd),
        }


def prepare_channels(batters, seasons, contact_monthly: pd.DataFrame,
                     swing_monthly: pd.DataFrame,
                     cutoff_date) -> MeasurementChannels:
    """`channel_arrays` flattened into the observed vectors the graph needs."""
    arr = channel_arrays(batters, seasons, contact_monthly, swing_monthly,
                         cutoff_date)
    n_b, n_s = arr["bbe"].shape

    bb = arr["bbe"] >= MIN_BBE
    b_i, b_s = np.nonzero(bb)
    bbe = arr["bbe"][b_i, b_s]
    barrels = arr["n_barrel"][b_i, b_s]

    # Per-batted-ball mean and sd, pooled over the observed cells: the
    # sufficient statistics on the artifact are exactly what that needs, and
    # standardizing by the *per-ball* sd is what makes sigma_ev/sqrt(bbe) the
    # standard error of the cell mean rather than an arbitrary rescaling.
    total_bbe = float(bbe.sum())
    if total_bbe > 1:
        s1 = float(arr["sum_ev"][b_i, b_s].sum())
        s2 = float(arr["sum_ev2"][b_i, b_s].sum())
        ev_mu = s1 / total_bbe
        var = max(s2 / total_bbe - ev_mu ** 2, 1e-9)
        ev_sd = float(np.sqrt(var))
    else:
        ev_mu, ev_sd = 0.0, 1.0
    ev_mean = np.divide(arr["sum_ev"][b_i, b_s], np.maximum(bbe, 1.0))
    ev_z = (ev_mean - ev_mu) / ev_sd if ev_sd > 0 else np.zeros_like(ev_mean)

    sw = arr["swings"] >= MIN_SWINGS
    w_i, w_s = np.nonzero(sw)

    ch = MeasurementChannels(
        barrel_batter=b_i.astype("int64"), barrel_season=b_s.astype("int64"),
        barrel_k=barrels.astype("int64"), barrel_n=bbe.astype("int64"),
        ev_batter=b_i.astype("int64"), ev_season=b_s.astype("int64"),
        ev_z=ev_z.astype("float64"), ev_n=bbe.astype("float64"),
        whiff_batter=w_i.astype("int64"), whiff_season=w_s.astype("int64"),
        whiff_k=arr["whiffs"][w_i, w_s].astype("int64"),
        whiff_n=arr["swings"][w_i, w_s].astype("int64"),
        ev_mu=ev_mu, ev_sd=ev_sd, n_batters=int(n_b), n_seasons=int(n_s),
    )
    if (ch.barrel_k > ch.barrel_n).any() or (ch.whiff_k > ch.whiff_n).any():
        raise ValueError("a channel has more successes than trials — the "
                         "artifacts' counts are not what this code thinks "
                         "they are")
    logger.info("channels prepared: %s observed cells, ev ~ N(%.1f, %.1f) mph "
                "per batted ball", ch.counts(), ev_mu, ev_sd)
    return ch


# ─── the graph ─────────────────────────────────────────────────────────────

def loading_name(channel: str) -> str:
    return f"lambda_{channel}"


def intercept_name(channel: str) -> str:
    return f"alpha_{channel}"


def build_measurement_model(data, channels: MeasurementChannels,
                            options=None, components=None):
    """`pa_joint.build_joint_model` with the observation channels added.

    The joint model is built first and then *re-entered* (`with model:`) so
    that every term of it is the BAS-84 graph, untouched, and the channels are
    an addition rather than a fork. What the channels read is the
    `player_ability` Deterministic that model already writes, standardized by
    `mu_ability` and `sigma_ability` — the same node the outcome likelihood
    reads with a coefficient of 1, which is the identification.

    Under `options.ability_walk` `player_ability` is (batter, season,
    component) and a channel cell reads its own season's state; under the flat
    model it is (batter, component) and every season of a batter reads the one
    state, which is correct — a flat model says his talent did not move.
    """
    import pymc as pm
    import pytensor.tensor as pt

    from src.models.pa_joint import build_joint_model

    comps = list(data.components)
    for channel, comp in CHANNEL_COMPONENT.items():
        if comp not in comps:
            raise ValueError(
                f"the {channel!r} channel loads on {comp!r}, which this fit "
                f"does not carry (components {comps}); the measurement model "
                f"is pre-registered over {list(MEASUREMENT_COMPONENTS)}")
    if channels.n_batters and channels.n_batters != data.shared["n_batters"]:
        raise ValueError(
            f"the channels were prepared for {channels.n_batters} batters and "
            f"the cells carry {data.shared['n_batters']}; they must be "
            f"prepared from the same `batters`/`seasons` arrays")

    model = build_joint_model(data, options)
    # `named_vars_to_dims`, not the variable's own `.dims`: a PyMC model
    # keeps its coords mapping on the model, and a plain TensorVariable has
    # no dims attribute at all.
    walk = "season" in (model.named_vars_to_dims.get("player_ability") or ())

    with model:
        ability = model["player_ability"]
        mu_ability = model["mu_ability"]

        def latent(component: str, batter_idx, season_idx):
            """The centred latent for one channel's observed cells.

            Centred, not standardized: dividing by `sigma_ability` here is
            what made this posterior unsamplable (module docstring). The
            loading carries the scale instead, and the summary converts back
            to per-sd units from the posterior draws.
            """
            j = comps.index(component)
            theta = (ability[batter_idx, season_idx, j] if walk
                     else ability[batter_idx, j])
            return theta - mu_ability[j]

        alpha, lam = {}, {}
        for channel in CHANNELS:
            alpha[channel] = pm.Normal(intercept_name(channel), mu=0.0,
                                       sigma=ALPHA_SIGMA)
            lam[channel] = pm.Normal(loading_name(channel), mu=0.0,
                                     sigma=LAMBDA_SIGMA)

        # ─── barrels per batted ball ─────────────────────────────────────
        if channels.barrel_k.size:
            z = latent(CHANNEL_COMPONENT["barrel"],
                       pm.Data("barrel_batter_idx", channels.barrel_batter),
                       pm.Data("barrel_season_idx", channels.barrel_season))
            pm.Binomial(
                "barrel_obs",
                n=pm.Data("barrel_n", channels.barrel_n),
                p=pm.math.invlogit(alpha["barrel"] + lam["barrel"] * z),
                observed=channels.barrel_k,
            )

        # ─── mean exit velocity per batted ball ──────────────────────────
        if channels.ev_z.size:
            sigma_ev = pm.HalfNormal("sigma_ev", sigma=SIGMA_EV_PRIOR)
            z = latent(CHANNEL_COMPONENT["ev"],
                       pm.Data("ev_batter_idx", channels.ev_batter),
                       pm.Data("ev_season_idx", channels.ev_season))
            # sigma_ev / sqrt(BBE): the cell's observation is a mean of BBE
            # batted balls, so its standard error falls with the count. This
            # is the line that makes a thin window contribute little — the
            # thing BAS-83's pooled coefficient could not do.
            pm.Normal(
                "ev_obs",
                mu=alpha["ev"] + lam["ev"] * z,
                sigma=sigma_ev / pt.sqrt(pm.Data("ev_n", channels.ev_n)),
                observed=channels.ev_z,
            )

        # ─── whiffs per swing ────────────────────────────────────────────
        if channels.whiff_k.size:
            z = latent(CHANNEL_COMPONENT["whiff"],
                       pm.Data("whiff_batter_idx", channels.whiff_batter),
                       pm.Data("whiff_season_idx", channels.whiff_season))
            pm.Binomial(
                "whiff_obs",
                n=pm.Data("whiff_n", channels.whiff_n),
                p=pm.math.invlogit(alpha["whiff"] + lam["whiff"] * z),
                observed=channels.whiff_k,
            )

    logger.info("measurement model built: %s channel cells on latents %s",
                channels.counts(), sorted(set(CHANNEL_COMPONENT.values())))
    return model


# ─── reading the predictions off ───────────────────────────────────────────

def _summarise(values: np.ndarray) -> dict:
    vals = np.asarray(values, dtype="float64").ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    return {
        "mean": float(vals.mean()), "sd": float(vals.std()),
        "q2.5": float(np.percentile(vals, 2.5)),
        "q97.5": float(np.percentile(vals, 97.5)),
        "p_gt_0": float((vals > 0).mean()),
    }


def measurement_param_summary(trace, channels: MeasurementChannels | None = None,
                              components=MEASUREMENT_COMPONENTS) -> dict:
    """Everything `docs/bayes-measurement.md`'s predictions 1 and 5 are read off.

    * every loading's posterior with a 95% interval, plus `p_gt_0` — an
      interval that excludes zero is prediction 1, and the tail mass is what
      says by how much;
    * `ev_mph`, the EV loading converted back to miles per hour per sd of
      power, because "0.62" in standardized units is not a number anyone can
      sanity-check against public work and "1.9 mph" is;
    * `loading_corr`, the posterior correlation between each pair of loadings
      — prediction 5's degeneracy check, computed across draws rather than
      from a normal approximation;
    * `sigma_ability` per component, prediction 5's latent-state scale.

    Reads straight off `.posterior[name].values`, so a plain object with a
    `.posterior` mapping is enough to test it.
    """
    post = getattr(trace, "posterior", None) if trace is not None else None
    out: dict = {}
    if post is None:
        return out

    # sigma_ability per component, draw by draw, so a per-logit loading can be
    # turned back into the per-sd number the pre-registration talks about
    # *inside* the posterior rather than by scaling a summary by a mean. The
    # two differ whenever sigma_ability is not tight, and the interval is what
    # prediction 1 is read off.
    sigma_draws = {}
    if "sigma_ability" in post:
        var = post["sigma_ability"]
        if "component" in var.dims:
            for c in var.coords["component"].values:
                sigma_draws[str(c)] = np.asarray(
                    var.sel(component=c).values, dtype="float64").ravel()

    draws = {}
    for channel in CHANNELS:
        name = loading_name(channel)
        if name in post:
            raw = np.asarray(post[name].values, dtype="float64").ravel()
            draws[name] = raw
            # Per sd of ability: the loading times that draw's ability scale.
            sigma = sigma_draws.get(CHANNEL_COMPONENT[channel])
            per_sd = raw * sigma if (sigma is not None
                                     and sigma.size == raw.size) else raw
            s = _summarise(per_sd)
            if s:
                # The headline numbers are per sd, which is the unit
                # docs/bayes-measurement.md states the loadings in; the
                # per-logit parameter the graph actually carries is kept
                # beside it so a fit record says what was sampled as well as
                # what was reported.
                s["per_logit"] = _summarise(raw)
                if channel == "ev" and channels is not None:
                    s["mph_per_sd"] = s["mean"] * float(channels.ev_sd)
                out[name] = s
        alpha = intercept_name(channel)
        if alpha in post:
            s = _summarise(post[alpha].values)
            if s:
                out[alpha] = s
    if "sigma_ev" in post:
        s = _summarise(post["sigma_ev"].values)
        if s:
            out["sigma_ev"] = s

    corr = {}
    for a, b in itertools.combinations(sorted(draws), 2):
        x, y = draws[a], draws[b]
        if x.size == y.size and x.size > 1 and x.std() > 0 and y.std() > 0:
            corr[f"{a}__{b}"] = float(np.corrcoef(x, y)[0, 1])
    if corr:
        out["loading_corr"] = corr
        out["max_abs_loading_corr"] = float(max(abs(v) for v in corr.values()))

    if "sigma_ability" in post:
        var = post["sigma_ability"]
        if "component" in var.dims:
            out["sigma_ability"] = {
                str(c): float(np.mean(var.sel(component=c).values))
                for c in var.coords["component"].values}
        else:
            out["sigma_ability"] = {"": float(np.mean(var.values))}
    return out


__all__ = [
    "ALPHA_SIGMA", "CHANNELS", "CHANNEL_COMPONENT", "LAMBDA_SIGMA",
    "MEASUREMENT_COMPONENTS", "MIN_BBE", "MIN_SWINGS", "MeasurementChannels",
    "SIGMA_EV_PRIOR", "build_measurement_model", "channel_arrays",
    "intercept_name", "loading_name", "measurement_param_summary",
    "prepare_channels",
]
