"""The prior's *mean* as a function of the player's profile (BAS-94).

Every hierarchical arm on this board shrinks a batter toward one league mean:
`ability_i ~ Normal(mu_ability, sigma_ability)`. That is why the
single-component model draws with tuned Marcel — it *is* Marcel with a
learned ballast — and it is the structure neither BAS-83 (the aggregates as
regressors in the likelihood) nor BAS-85 (the aggregates as observation
channels of a latent) changed. This module builds the design matrix for the
one that does:

    ability_i,s ~ Normal(mu_ability + gamma . x[i, s], sigma_ability)

with `x` the player's **contact profile measured on prior seasons**, so the
shrinkage target is "what players who hit the ball like this usually do"
rather than "the league".

**Why the window is the thing.** BAS-83 put the *same* aggregates into the
*same* graph and lost by 19% of HR/PA MAE, and `docs/bayes-covariates.md`
diagnosed it precisely: one coefficient was fitted mostly on full prior
seasons (~400 batted balls behind each aggregate) and then applied to the
current season's partial window (a few dozen), which is errors in variables
imported at the full window's coefficient. This module's arm A never reads
the current season at all. The profile is stale by up to a season — recorded
in the pre-registration as the design, not as a defect — and in exchange it
is measured on ~700-900 batted balls at every cutoff in the season, April
included, so the coefficient and the covariate are on the same footing.

**The two arms** (`PRIOR_MEAN_SETS`):

    contact      arm A, primary. The previous two full seasons only
                 (`PRIOR_WEIGHTS`). Nothing from the cutoff's own season
                 enters, at any cutoff.
    contact_cur  arm B, secondary. The previous two full seasons *plus* the
                 cutoff season through the last month boundary on or before
                 the cutoff (`CURRENT_WEIGHTS`), summed as counts.

**Arm B's ballast, stated once and fixed.** The current season enters as raw
counts added to the prior seasons' raw counts, so the two are combined in
proportion to their own exposure and the "ballast" on the thin current
window is the prior profile's own batted balls: about 700-900 of them
against roughly 30 in mid-April, 150 by June and 300 by August. A hitter's
April therefore moves his target by a few percent and his August by a
quarter, with nothing to tune — the weights are (1, 1, 1), the shrinkage
ballast toward the league is `src.eval.contact.DEFAULT_BALLAST`, the value
the served `contact_additive` engine already uses, and neither was chosen by
looking at a scored season.

**The features.** `src.eval.contact.FEATURES` (mean exit velocity, EV90,
barrel rate, hard-hit rate, sweet-spot rate, mean launch angle) for every
component, plus **whiff share per swing** for K% — the pre-registration names
it, and a K% prior mean built out of batted-ball quality alone would have no
feature that measures the thing K% is. Whiff comes from the swing-decisions
artifact (`src.data.swing_decisions`), summed over the same window and shrunk
toward the league by `WHIFF_BALLAST` swings.

**Standardised per season on the training population**, exposure-weighted, by
the same `src.eval.contact.standardize` the served engine uses — so `x = 0`
is "league average for that season" and a batter with no prior-season profile
(a rookie, a September call-up) sits exactly at the league mean, which is the
old model's answer for him. Per season and not pooled for the reason
`src.models.pa_covariates` gives: the league's own contact profile drifts,
and a covariate whose zero moves with the league is the one that means
"average for his year" at every node of the league random walk.

**Leakage.** Every window goes through `src.eval.contact.window_counts`,
which refuses a cutoff that is not a month boundary and re-checks the filter
with `assert_window_clean`; the whiff window applies the same two guards
directly. For arm A the question is moot by construction (season `s`'s
features come from `s-1` and `s-2`, both of which ended before any cutoff in
season `s`), and the test suite proves it anyway on both arms.

Imports neither pymc nor arviz, so it is readable in CI; the graph that
consumes `pm_x` lives in `src.models.pa_rate.build_model`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Recency weights over (current season, previous, the one before that) — the
# shape `src.eval.contact.window_counts` takes. A zero drops the season from
# the window entirely rather than multiplying its counts by nothing, which is
# what makes `PRIOR_WEIGHTS` mean "the current season is not in this feature".
PRIOR_WEIGHTS = (0.0, 1.0, 1.0)
CURRENT_WEIGHTS = (1.0, 1.0, 1.0)

# The whiff feature's name in the design matrix and in the model's own
# `gamma_<feature>` RVs.
WHIFF = "whiff"

# Swings of league-average whiff rate to regress a player's own toward,
# matched to `src.eval.contact.DEFAULT_BALLAST` (5 batted balls) through the
# exposure ratio between the two channels: a hitter takes about three swings
# per batted ball (2024: median 470 swings against 168 batted balls), so 15
# swings is the same fraction of a season's swing sample that 5 batted balls
# is of its contact sample. Fixed here, never swept — the pre-registration
# freezes the feature set and this is part of it.
WHIFF_BALLAST = 15.0

# Prior scale on each prior-mean coefficient, on the logit scale
# (pre-registered in docs/bayes-prior-mean.md). The features are z-scores, so
# 0.5 lets a one-sd hitter's profile move his shrinkage target by up to ~1
# logit at two prior sds — far wider than anything the served additive arm
# fits, which is the point: the prior must not be what produces the answer.
GAMMA_SIGMA = 0.5


class PriorMeanSpec:
    """One named prior-mean feature set: which window, and whether whiff is in.

    A tiny object rather than a bare tuple so the arm a fit ran under is
    stamped on the fit record by name ("contact", "contact_cur") and a table
    can never show arm A's number under arm B's label.
    """

    __slots__ = ("name", "weights", "arm", "description")

    def __init__(self, name: str, weights: tuple[float, float, float],
                 arm: str, description: str):
        self.name = name
        self.weights = weights
        self.arm = arm
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PriorMeanSpec({self.name!r}, arm={self.arm!r})"


PRIOR_MEAN_SETS: dict[str, PriorMeanSpec] = {
    "contact": PriorMeanSpec(
        "contact", PRIOR_WEIGHTS, "A",
        "the previous two full seasons only; the cutoff's own season never "
        "enters the shrinkage target"),
    "contact_cur": PriorMeanSpec(
        "contact_cur", CURRENT_WEIGHTS, "B",
        "the previous two full seasons plus the cutoff season through the "
        "last month boundary, summed as counts so a thin current window is "
        "ballasted by the prior profile's own exposure"),
}


def prior_mean_set(spec) -> str | None:
    """Resolve and validate a prior-mean set name. `None` is off."""
    if spec is None:
        return None
    if isinstance(spec, PriorMeanSpec):
        spec = spec.name
    if not isinstance(spec, str):
        raise TypeError(
            f"prior_mean_covariates takes a set name or None, not {spec!r}")
    if spec not in PRIOR_MEAN_SETS:
        raise ValueError(f"unknown prior-mean set {spec!r}; known: "
                         f"{sorted(PRIOR_MEAN_SETS)}")
    return spec


def prior_mean_features(spec, component: str | None = None) -> tuple[str, ...]:
    """The feature names a set selects for a component, in design-matrix order.

    `src.eval.contact.FEATURES` for every component, plus `whiff` for K%. The
    component decides because the pre-registration says so: whiff share is
    the K% latent's measurement and there is no claim here that it belongs on
    a home-run rate.
    """
    from src.eval.contact import FEATURES

    name = prior_mean_set(spec)
    if name is None:
        return ()
    comp = getattr(component, "name", component)
    return tuple(FEATURES) + ((WHIFF,) if comp == "k_rate" else ())


# --- the whiff window -------------------------------------------------------

def whiff_window(swing_monthly: pd.DataFrame, cutoff, predict_year: int,
                 weights: tuple[float, float, float]) -> pd.DataFrame:
    """Per-batter swings and whiffs in one window, strictly before the cutoff.

    `src.eval.contact.window_counts` for the swing-decisions artifact: the
    same recency weights, the same month-boundary refusal, and the same
    post-filter re-check (`assert_window_clean`) so the filter is proved
    rather than trusted. Returns `player` / `swings` / `whiffs`, with an
    unweighted `swings_raw` — the real exposure behind the row, which is what
    the standardization weights by.
    """
    from src.eval.contact import assert_month_boundary, assert_window_clean

    cutoff = pd.Timestamp(cutoff)
    assert_month_boundary(cutoff)
    w = {predict_year - i: float(x) for i, x in enumerate(weights)
         if float(x) != 0.0}

    rows = swing_monthly[swing_monthly["season"].isin(w)]
    rows = rows[(rows["season"] < predict_year)
                | (rows["month"] < cutoff.month)]
    assert_window_clean(rows, cutoff, predict_year)
    if rows.empty:
        return pd.DataFrame(columns=["player", "swings", "whiffs", "swings_raw"])

    weight = rows["season"].map(w).astype("float64").to_numpy()
    swings = (rows["swings_z"].to_numpy("float64")
              + rows["swings_o"].to_numpy("float64"))
    whiffs = (rows["whiff_z"].to_numpy("float64")
              + rows["whiff_o"].to_numpy("float64"))
    out = pd.DataFrame({
        "player": rows["batter"].to_numpy(),
        "swings": swings * weight,
        "whiffs": whiffs * weight,
        "swings_raw": swings,
    })
    return out.groupby("player", as_index=False).sum()


def whiff_metrics(counts: pd.DataFrame,
                  ballast: float = WHIFF_BALLAST) -> pd.DataFrame:
    """Whiff share per swing, shrunk toward the window's own league rate.

    Shaped like `src.eval.contact.contact_metrics` output — `player`,
    `bbe_raw` (here the raw swings, the exposure the standardization weights
    by) and one feature column — so `src.eval.contact.standardize` consumes
    it unchanged and the whiff z-score is built by exactly the same code as
    the six batted-ball z-scores.
    """
    if counts.empty:
        return pd.DataFrame(columns=["player", "bbe_raw", WHIFF])
    swings = counts["swings"].to_numpy("float64")
    whiffs = counts["whiffs"].to_numpy("float64")
    total = float(swings.sum())
    league = float(whiffs.sum()) / total if total > 0 else 0.0
    b = float(ballast)
    return pd.DataFrame({
        "player": counts["player"].to_numpy(),
        "bbe_raw": counts["swings_raw"].to_numpy("float64"),
        WHIFF: (whiffs + b * league) / (swings + b),
    })


# --- the design matrix ------------------------------------------------------

def prior_mean_array(
    batters,
    seasons,
    monthly: pd.DataFrame,
    swing_monthly: pd.DataFrame | None,
    cutoff_date,
    names: tuple[str, ...],
    spec="contact",
    side: str = "hitter",
) -> np.ndarray:
    """Standardised prior-mean features, shaped (n_batters, n_seasons, n_feat).

    One independent standardization per season on the players present in that
    season's own window, exposure-weighted; every (batter, season) with no
    profile in its window is left at 0.0, which after standardization *is*
    that season's league mean. Column order is `names`.
    """
    from src.eval.contact import (
        DEFAULT_BALLAST, contact_metrics, standardize, window_counts,
    )
    from src.models.pa_covariates import season_cutoff

    set_name = prior_mean_set(spec)
    if set_name is None:
        raise ValueError("prior_mean_array needs a prior-mean set, not None")
    weights = PRIOR_MEAN_SETS[set_name].weights
    names = tuple(names)
    contact_names = tuple(n for n in names if n != WHIFF)
    want_whiff = WHIFF in names

    batters = np.asarray(batters)
    seasons = np.asarray(seasons)
    index = {int(b): i for i, b in enumerate(batters)}
    x = np.zeros((len(batters), len(seasons), len(names)), dtype="float64")
    col = {n: j for j, n in enumerate(names)}

    if want_whiff and swing_monthly is None:
        raise ValueError(
            "this prior-mean set carries a whiff feature but no "
            "swing-decisions artifact was handed in")

    for s_idx, season in enumerate(seasons):
        season = int(season)
        stop = season_cutoff(season, cutoff_date)
        counts = window_counts(monthly, side, stop, season, weights=weights)
        if counts.empty:
            logger.info("prior mean [%s]: %d has no window contact at all — "
                        "every batter sits at the league mean", set_name, season)
        else:
            z = standardize(contact_metrics(counts, DEFAULT_BALLAST),
                            features=contact_names)
            rows = np.array([index.get(int(p), -1) for p in z["player"]])
            keep = rows >= 0
            if keep.any():
                cols = [col[n] for n in contact_names]
                x[np.ix_(rows[keep], [s_idx], cols)] = \
                    z.loc[keep, list(contact_names)].to_numpy("float64")[:, None, :]

        if want_whiff:
            wc = whiff_window(swing_monthly, stop, season, weights)
            if wc.empty:
                continue
            zw = standardize(whiff_metrics(wc), features=(WHIFF,))
            rows = np.array([index.get(int(p), -1) for p in zw["player"]])
            keep = rows >= 0
            if keep.any():
                x[rows[keep], s_idx, col[WHIFF]] = \
                    zw.loc[keep, WHIFF].to_numpy("float64")

    covered = float((np.abs(x).sum(axis=2) > 0).mean())
    logger.info("prior mean [%s, arm %s] %s: %d batters x %d seasons, %.1f%% "
                "of (batter, season) pairs carry a profile", set_name,
                PRIOR_MEAN_SETS[set_name].arm, list(names), len(batters),
                len(seasons), 100.0 * covered)
    return x


def attach_prior_mean(
    data: dict,
    spec,
    monthly: pd.DataFrame | None = None,
    swing_monthly: pd.DataFrame | None = None,
    cutoff_date=None,
    component: str | None = None,
) -> dict:
    """Put `pm_x` / `pm_names` / `pm_set` on a prepared model-data dict, in place.

    Kept out of `prepare_model_data` for the same reason
    `src.models.pa_covariates.attach_covariates` is: with the prior mean off
    nothing in this file runs and the model's data pipeline is byte-for-byte
    what it was, which is what keeps the frozen K% fixture green.
    """
    set_name = prior_mean_set(spec)
    if set_name is None:
        return data
    component = component or data.get("component")
    names = prior_mean_features(set_name, component)
    if not names:
        raise ValueError(f"prior-mean set {set_name!r} selected no features")
    if monthly is None:
        from src.data.contact_quality import load_monthly

        monthly = load_monthly()
    if swing_monthly is None and WHIFF in names:
        from src.data.swing_decisions import load_monthly as load_swing

        swing_monthly = load_swing()
    cutoff_date = cutoff_date or data.get("cutoff_date")
    if cutoff_date is None:
        raise ValueError(
            "prior-mean features are summed strictly before a cutoff; this "
            "model data carries no cutoff_date to sum up to")
    data["pm_set"] = set_name
    data["pm_names"] = tuple(names)
    data["pm_x"] = prior_mean_array(
        data["batters"], data["seasons"], monthly, swing_monthly, cutoff_date,
        names, set_name)
    return data


# --- what the fit record keeps ---------------------------------------------

def prior_mean_summary(trace, data: dict) -> dict:
    """The vacuity numbers docs/bayes-prior-mean.md's prediction 1 is read off.

    Two things, per fit:

    * every `gamma_<feature>`'s posterior mean and 90% interval — "the prior
      moves" is a statement about whether a coefficient excludes zero, which
      a mean alone cannot answer;
    * the **between-player sd of the prior mean** at the last fitted season,
      against `sigma_ability`. That ratio is the real vacuity test: a
      `gamma` that excludes zero but moves the target by 1% of the population
      spread is a relabelled league mean, and the pre-registration puts the
      floor at 30%.

    Reads `.posterior[name].values` directly rather than going through arviz,
    so it needs neither pymc nor arviz and a plain object with a `.posterior`
    mapping is enough to test it.
    """
    out: dict = {"set": data.get("pm_set"),
                 "features": list(data.get("pm_names", ()))}
    posterior = getattr(trace, "posterior", None) if trace is not None else None
    if posterior is None:
        return out

    gammas: dict = {}
    for name in out["features"]:
        key = f"gamma_{name}"
        if key not in posterior:
            continue
        v = np.asarray(posterior[key].values, dtype="float64").ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        gammas[key] = {"mean": float(v.mean()), "sd": float(v.std()),
                       "q05": float(np.percentile(v, 5)),
                       "q95": float(np.percentile(v, 95)),
                       "excludes_zero": bool(np.percentile(v, 5) > 0
                                             or np.percentile(v, 95) < 0)}
    out["gamma"] = gammas

    if "prior_mean_effect" in posterior and "sigma_ability" in posterior:
        eff = np.asarray(posterior["prior_mean_effect"].values, dtype="float64")
        # (chain, draw, batter, season) -> the last fitted season, which at an
        # intra-season cutoff is the one the projection reads.
        eff = eff.reshape(-1, eff.shape[-2], eff.shape[-1])[:, :, -1]
        sd_by_draw = eff.std(axis=1)
        sig = np.asarray(posterior["sigma_ability"].values,
                         dtype="float64").ravel()
        sd_mean = float(np.mean(sd_by_draw))
        sig_mean = float(np.mean(sig))
        out["prior_mean_sd_between_players"] = sd_mean
        out["sigma_ability"] = sig_mean
        out["sd_ratio"] = (sd_mean / sig_mean) if sig_mean > 0 else float("nan")
        # Per draw, so the ratio carries an interval rather than a point that
        # divides two independently-averaged numbers.
        ratio = sd_by_draw / np.where(sig > 0, sig, np.nan)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size:
            out["sd_ratio_q05"] = float(np.percentile(ratio, 5))
            out["sd_ratio_q95"] = float(np.percentile(ratio, 95))
    return out


__all__ = [
    "CURRENT_WEIGHTS", "GAMMA_SIGMA", "PRIOR_MEAN_SETS", "PRIOR_WEIGHTS",
    "PriorMeanSpec", "WHIFF", "WHIFF_BALLAST", "attach_prior_mean",
    "prior_mean_array", "prior_mean_features", "prior_mean_set",
    "prior_mean_summary", "whiff_metrics", "whiff_window",
]
