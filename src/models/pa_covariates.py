"""Layer-1 covariates for the PA-level rate model (BAS-83).

`docs/bayes-variants.md` closed with the hierarchical arm *drawing* with tuned
Marcel, and `docs/contact-quality.md` showed that the same six exit-velocity /
launch-angle aggregates beat tuned Marcel by 1.6-4.8% of component MAE when
bolted onto it as a linear correction. Those two results were never run
against each other on the same information, so "the hierarchical model is as
good as Marcel" and "contact quality is worth 2% of K% MAE" have never been
statements about the same fight. This module is the missing half: it turns the
contact aggregates into a **per (batter, season)** design matrix the
hierarchical model can read, so both arms see the same layer-1 measurement and
the comparison is about the estimator rather than about who got the data.

**What a row is.** One vector of standardized aggregates per (batter, season)
in the model's own window. For a season that finished before the cutoff, that
is the whole season. For the season the cutoff falls in, it is the months that
ended before the cutoff's month boundary — the same month lag
`src.projections.ros.contact_cutoff` serves live, because the committed
artifact is monthly and rounding a cutoff *forward* is leakage
(docs/contact-quality.md §2). A July 15 cutoff therefore reads contact through
June 30, exactly as a July 15 live build would.

**Standardized per season**, batted-ball weighted, by the same
`src.eval.contact.standardize` the served `contact_additive` engine uses. Per
season and not pooled because the league's own contact profile drifts (the
ball, the shift ban, the pitch clock), and a covariate whose zero moves with
the league is the one that means "average for his year" at every node of the
league random walk.

**A batter with no tracked contact before the cutoff gets x = 0**, which after
standardization *is* the league mean for that season, and **no `has_cov`
indicator is added.** That is a deliberate choice and it is worth stating why,
because the alternative is the more usual one. An indicator would let the
model fit a level offset for "we have no Statcast on this man", and every
batter in that set is there for one of two reasons: he is a September call-up
with no prior season, or he is a hitter whose only pre-cutoff plate
appearances all ended without a batted ball. Both are already expressed, and
expressed better, elsewhere in the model — the first by partial pooling on a
batter with few trials, the second by the strikeout and walk outcomes
themselves, which are the very thing being fit. An indicator would compete
with those for the same signal, and its coefficient would be identified almost
entirely off April cells, where the missing set is largest and least like the
rest of the season. Setting the covariate to the league mean says "we have no
contact information about him", which is true; an indicator would say "having
no contact information is itself informative about his rate", which is a
claim this ticket did not pre-register and cannot test on the cells it scores.

Nothing here is imported by `prepare_model_data`: the covariate block is
attached to an already-prepared `data` dict by `attach_covariates`, so the
model's data pipeline — and therefore the frozen K% fixture in
`tests/test_models/test_pa_rate.py` — is untouched when covariates are off.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The six aggregates, in `src.eval.contact.FEATURES` order so a coefficient
# vector printed here and one printed by the comparator read the same way.
CONTACT_COVARIATES = ("ev_mean", "ev90", "barrel", "hardhit", "sweetspot",
                      "la_mean")

# Named covariate sets `ModelOptions.covariates` and `--bayes-covariates`
# accept. One entry today; the registry exists so the next layer-1
# measurement (swing decisions, stuff) is a dict entry rather than a new flag.
COVARIATE_SETS: dict[str, tuple[str, ...]] = {"contact": CONTACT_COVARIATES}

# Prior scale on each aggregate's coefficient, on the logit scale
# (pre-registered in docs/bayes-covariates.md). The covariates are z-scores,
# so 0.5 puts a one-sd hitter's contact worth up to ~1 logit at two prior sds
# — far wider than the ~0.1 logit per sd the served additive arm fits, which
# is the point: the prior must not be what produces the answer.
BETA_COV_SIGMA = 0.5


def covariate_names(covariates) -> tuple[str, ...]:
    """Resolve a covariate spec to the aggregate names it selects.

    Accepts `None` (off), a registry name like `"contact"`, or an explicit
    tuple of aggregate names. Raises on anything else rather than silently
    fitting a different design matrix than the caller asked for.
    """
    if covariates is None:
        return ()
    if isinstance(covariates, str):
        if covariates not in COVARIATE_SETS:
            raise ValueError(
                f"unknown covariate set {covariates!r}; known: "
                f"{sorted(COVARIATE_SETS)}")
        return COVARIATE_SETS[covariates]
    names = tuple(covariates)
    unknown = [n for n in names if n not in CONTACT_COVARIATES]
    if unknown:
        raise ValueError(
            f"unknown covariate(s) {unknown}; the contact-quality artifact "
            f"carries {list(CONTACT_COVARIATES)}")
    return names


def season_cutoff(season: int, cutoff_date) -> str:
    """The month boundary this season's contact window is summed up to.

    A season strictly before the cutoff's year is read whole (December 1 in
    `window_counts`'s convention admits every month a baseball season has).
    The cutoff's own season is read up to the last month boundary on or
    before the cutoff, which is exactly `src.projections.ros.contact_cutoff`
    — a July 15 cutoff reads through June 30, never through July 14, because
    the buckets are monthly and rounding forward would leak.
    """
    cutoff = pd.Timestamp(cutoff_date)
    if int(season) < cutoff.year:
        return f"{int(season)}-12-01"
    return f"{cutoff.year}-{cutoff.month:02d}-01"


def contact_covariate_array(
    batters,
    seasons,
    monthly: pd.DataFrame,
    cutoff_date,
    names: tuple[str, ...] = CONTACT_COVARIATES,
    ballast: float | None = None,
    side: str = "hitter",
) -> np.ndarray:
    """Standardized contact aggregates, shaped (n_batters, n_seasons, n_cov).

    One independent standardization per season (see the module docstring), so
    column `j` of season `s` is a z-score against the batters who put a ball
    in play in season `s` before that season's cutoff, batted-ball weighted.

    Every (batter, season) with no tracked contact in its window is left at
    0.0 — the league mean after standardization, and the whole of what this
    module does about missingness.
    """
    from src.eval.contact import (
        DEFAULT_BALLAST, contact_metrics, standardize, window_counts,
    )

    ballast = DEFAULT_BALLAST if ballast is None else float(ballast)
    batters = np.asarray(batters)
    seasons = np.asarray(seasons)
    index = {int(b): i for i, b in enumerate(batters)}
    x = np.zeros((len(batters), len(seasons), len(names)), dtype="float64")

    for s_idx, season in enumerate(seasons):
        season = int(season)
        counts = window_counts(monthly, side, season_cutoff(season, cutoff_date),
                               season, weights=(1.0, 0.0, 0.0))
        if counts.empty:
            logger.info("covariates: %d has no pre-cutoff contact at all — "
                        "every batter sits at the league mean", season)
            continue
        z = standardize(contact_metrics(counts, ballast), features=names)
        rows = np.array([index.get(int(p), -1) for p in z["player"]])
        keep = rows >= 0
        if not keep.any():
            continue
        x[rows[keep], s_idx, :] = z.loc[keep, list(names)].to_numpy("float64")

    covered = float((np.abs(x).sum(axis=2) > 0).mean())
    logger.info("covariates %s: %d batters x %d seasons, %.1f%% of "
                "(batter, season) pairs carry pre-cutoff contact",
                list(names), len(batters), len(seasons), 100.0 * covered)
    return x


def attach_covariates(
    data: dict,
    covariates,
    monthly: pd.DataFrame | None = None,
    cutoff_date=None,
    ballast: float | None = None,
) -> dict:
    """Put `cov_x` / `cov_names` on a prepared model-data dict, in place.

    Kept out of `prepare_model_data` on purpose: with covariates off nothing
    in this file runs and the model's data pipeline is byte-for-byte what it
    was, which is what keeps the frozen K% fixture green.
    """
    names = covariate_names(covariates)
    if not names:
        return data
    if monthly is None:
        from src.data.contact_quality import load_monthly

        monthly = load_monthly()
    cutoff_date = cutoff_date or data.get("cutoff_date")
    if cutoff_date is None:
        raise ValueError(
            "contact covariates are summed strictly before a cutoff; this "
            "model data carries no cutoff_date to sum up to")
    data["cov_names"] = tuple(names)
    data["cov_x"] = contact_covariate_array(
        data["batters"], data["seasons"], monthly, cutoff_date, names, ballast)
    return data


__all__ = [
    "BETA_COV_SIGMA", "CONTACT_COVARIATES", "COVARIATE_SETS",
    "attach_covariates", "contact_covariate_array", "covariate_names",
    "season_cutoff",
]
