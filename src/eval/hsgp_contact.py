"""Stage 2: a Hilbert-space GP over (exit velocity, launch angle).

This is the tutorial's technique, recorded in methods.md §3 and §6: learn the
hard nonlinear measurement with a flexible model, then feed it as a covariate
into a model that does the pooling. The flexible model here is `pm.gp.HSGP` —
a Hilbert-space approximation to a Gaussian process, which replaces the N x N
covariance with a fixed set of basis functions and so scales past the handful
of points an exact GP allows — over the two-dimensional (EV, LA) plane, with
the value of a batted ball as the response.

**What the surface is.** For every batted ball the archive records an exit
velocity, a launch angle and the wOBA the plate appearance was worth. Pool
those into (EV, LA) cells, model the cell mean with a GP, and the posterior
mean of that GP *is* a contact-quality surface: what a ball hit that hard at
that angle is worth, on average, to anybody. It is the thing the six hand-
chosen aggregates of stage 1 are crude summaries of — barrel rate is a hand-
drawn contour of this surface, hard-hit rate a hand-drawn half-plane.

**How a player gets a number.** Average the surface over the batted balls he
actually hit before the cutoff, shrink toward the league by the same ballast
stage 1 uses, standardize. One covariate replaces six.

**What is fitted where.** The surface for a scored season is fitted on batted
balls from strictly earlier seasons; the regression coefficient on top of it is
fitted on strictly earlier cells. Neither ever sees the season it scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Grid the plane before fitting. A GP over 1.4 million individual batted balls
# is neither affordable nor necessary: the response is a smooth function of two
# coordinates measured to about a mile an hour and a degree, so binning to
# 2.5 mph by 5 degrees throws away nothing the surface could represent and
# turns the fit into a few hundred cell means with known precisions.
EV_EDGES = np.arange(40.0, 122.51, 2.5)
LA_EDGES = np.arange(-70.0, 72.51, 5.0)
# Cells thinner than this are noise and are dropped from the likelihood; the
# fitted surface still covers them, because the GP is placed over the whole
# grid and interpolates into them.
MIN_CELL_BBE = 25

BATTED_BALL_COLUMNS = ["batter", "pitcher", "game_date", "game_year", "event",
                       "launch_speed", "launch_angle", "woba_value",
                       "woba_denom"]
# Same batted-ball definition as `src.data.contact_quality`: the archive fills
# missing values with 0, and bunts are not batted-ball events.
EV_MISSING_BELOW = 5.0
BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}


def batted_balls_from_pa(pa: pd.DataFrame) -> pd.DataFrame:
    """Tracked batted balls with their wOBA value, from a PA-outcomes frame."""
    keep = ((pa["launch_speed"] > EV_MISSING_BELOW)
            & pa["launch_speed"].notna()
            & pa["launch_angle"].notna()
            & ~pa["event"].isin(BUNT_EVENTS)
            & (pa["woba_denom"] > 0))
    out = pa[keep].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["month"] = out["game_date"].dt.month
    out["value"] = out["woba_value"].astype("float64")
    return out[["batter", "pitcher", "game_year", "month", "launch_speed",
                "launch_angle", "value"]]


def cell_index(ev, la) -> tuple[np.ndarray, np.ndarray]:
    """(EV bin, LA bin) per batted ball, clipped into the grid."""
    i = np.clip(np.digitize(np.asarray(ev, dtype="float64"), EV_EDGES) - 1,
                0, len(EV_EDGES) - 2)
    j = np.clip(np.digitize(np.asarray(la, dtype="float64"), LA_EDGES) - 1,
                0, len(LA_EDGES) - 2)
    return i, j


def cell_centres() -> tuple[np.ndarray, np.ndarray]:
    return ((EV_EDGES[:-1] + EV_EDGES[1:]) / 2,
            (LA_EDGES[:-1] + LA_EDGES[1:]) / 2)


def grid_cells(bb: pd.DataFrame, min_bbe: int = MIN_CELL_BBE) -> pd.DataFrame:
    """Cell means of batted-ball value: one row per populated (EV, LA) cell."""
    i, j = cell_index(bb["launch_speed"], bb["launch_angle"])
    df = pd.DataFrame({"i": i, "j": j, "value": bb["value"].to_numpy()})
    g = df.groupby(["i", "j"], as_index=False).agg(n=("value", "size"),
                                                   mean=("value", "mean"),
                                                   sd=("value", "std"))
    g = g[g["n"] >= min_bbe].reset_index(drop=True)
    ev_c, la_c = cell_centres()
    g["ev"] = ev_c[g["i"].to_numpy()]
    g["la"] = la_c[g["j"].to_numpy()]
    return g


@dataclass
class Surface:
    """A fitted contact-quality surface on the (EV, LA) grid.

    `values` is indexed [i, j] over the grid cells; every cell has a value,
    including the ones the likelihood never saw, because the GP covers the
    whole plane.
    """
    values: np.ndarray
    diagnostics: dict
    seasons: tuple[int, ...] = ()

    def lookup(self, ev, la) -> np.ndarray:
        i, j = cell_index(ev, la)
        return self.values[i, j]


def fit_surface(cells: pd.DataFrame, *, m=(12, 12), c: float = 1.5,
                draws: int = 500, tune: int = 500, chains: int = 2,
                seed: int = 0, sampler: str = "numpyro",
                seasons: tuple[int, ...] = ()) -> Surface:
    """Fit `pm.gp.HSGP` over (EV, LA) to the cell means and return the surface.

    The GP is placed over the *whole* grid and the likelihood indexes into it
    at the cells with enough batted balls to say anything. That is what makes a
    sparsely populated corner of the plane — 118 mph at 60 degrees, which
    nobody hits — come back as the GP's own interpolation rather than as a
    hole, and it means the fitted coefficients and the surface read off them
    live on one basis rather than two.

    The likelihood is Normal on the cell mean with a *known* precision — the
    cell's own standard deviation over its own batted balls, divided by the
    root of its count — so a cell built from thirty balls constrains the
    surface thirty times less than one built from nine hundred, without
    anything being hand-weighted.

    Priors: the length scales get an InverseGamma placing its mass inside the
    span of the standardized inputs (a GP with a length scale far shorter than
    the grid spacing fits noise; one far longer than the plane is a constant),
    and the amplitude a HalfNormal on the scale of wOBA itself.
    """
    import arviz as az
    import pymc as pm

    ev_c, la_c = cell_centres()
    gi, gj = np.meshgrid(np.arange(len(ev_c)), np.arange(len(la_c)),
                         indexing="ij")
    grid = np.column_stack([ev_c[gi.ravel()], la_c[gj.ravel()]])
    mu_x, sd_x = grid.mean(axis=0), grid.std(axis=0)
    grid_s = (grid - mu_x) / sd_x

    # Row of the flattened grid each observed cell sits in.
    flat = cells["i"].to_numpy() * len(la_c) + cells["j"].to_numpy()
    y = cells["mean"].to_numpy(dtype="float64")
    sd = np.nan_to_num(cells["sd"].to_numpy(dtype="float64"), nan=0.5)
    n = cells["n"].to_numpy(dtype="float64")
    se = np.maximum(sd / np.sqrt(n), 1e-3)

    with pm.Model():
        ls = pm.InverseGamma("ls", alpha=3.0, beta=1.5, shape=2)
        eta = pm.HalfNormal("eta", sigma=0.5)
        cov = eta ** 2 * pm.gp.cov.ExpQuad(2, ls=ls)
        gp = pm.gp.HSGP(m=list(m), c=c, cov_func=cov)
        f = gp.prior("f", X=grid_s)
        mean = pm.Normal("mean", mu=0.4, sigma=0.5)
        # A little extra spread beyond each cell's own measurement error: the
        # surface is not exactly smooth at the grid's resolution.
        extra = pm.HalfNormal("extra", sigma=0.05)
        pm.Normal("obs", mu=mean + f[flat],
                  sigma=pm.math.sqrt(se ** 2 + extra ** 2), observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          cores=min(chains, 4), random_seed=seed,
                          nuts_sampler=sampler, target_accept=0.9,
                          progressbar=False)

    summary = az.summary(idata, var_names=["ls", "eta", "mean", "extra"])
    post = idata.posterior
    values = (float(post["mean"].mean())
              + post["f"].mean(dim=("chain", "draw")).to_numpy())
    diagnostics = {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "n_cells": int(len(cells)),
        "n_bbe": int(n.sum()),
        "ls": [float(v) for v in summary.loc[["ls[0]", "ls[1]"], "mean"]],
        "eta": float(summary.loc["eta", "mean"]),
    }
    return Surface(values=values.reshape(len(ev_c), len(la_c)),
                   diagnostics=diagnostics, seasons=seasons)


def window_batted_balls(bb: pd.DataFrame, cutoff, predict_year: int,
                        weights: tuple[float, float, float]) -> pd.DataFrame:
    """Batted balls in the projection window, strictly before the cutoff.

    The same window `src.eval.contact.window_counts` sums monthly buckets
    over, applied to the individual batted balls stage 2 needs — and guarded
    the same way: the cutoff must be the first of a month, and the guard
    re-checks the result rather than trusting the filter.
    """
    from src.eval.contact import assert_month_boundary

    cutoff = pd.Timestamp(cutoff)
    assert_month_boundary(cutoff)
    # A weight of zero means the season is *not in the window*, so it is
    # dropped rather than multiplied by nothing: it would otherwise still
    # count toward the exposure the covariate is shrunk and weighted by.
    w = {predict_year - i: float(x) for i, x in enumerate(weights)
         if float(x) != 0.0}
    rows = bb[bb["game_year"].isin(w)]
    rows = rows[(rows["game_year"] < predict_year)
                | (rows["month"] < cutoff.month)]
    late = rows[(rows["game_year"] == predict_year)
                & (rows["month"] >= cutoff.month)]
    if len(late):
        raise ValueError(
            f"leakage: {len(late)} batted ball(s) on or after the cutoff "
            f"{cutoff.date()} entered the window")
    if int(rows["game_year"].max() if len(rows) else predict_year) > predict_year:
        raise ValueError("leakage: contact window contains a future season")
    return rows


def player_values(bb: pd.DataFrame, surface: Surface, id_col: str,
                  weights: dict[int, float]) -> pd.DataFrame:
    """Recency-weighted mean surface value per player, and the batted balls
    behind it.

    `weights` maps season -> weight; a season absent from it is not in the
    window and contributes nothing.
    """
    w = bb["game_year"].map(weights)
    keep = w.notna()
    v = surface.lookup(bb.loc[keep, "launch_speed"], bb.loc[keep, "launch_angle"])
    df = pd.DataFrame({
        "player": bb.loc[keep, id_col].to_numpy(),
        "w": w[keep].to_numpy(dtype="float64"),
        "wv": w[keep].to_numpy(dtype="float64") * v,
        "bbe_raw": 1.0,
    })
    return df.groupby("player", as_index=False).sum()


def shrink_and_standardize(vals: pd.DataFrame, ballast: float) -> pd.DataFrame:
    """The stage-1 treatment, applied to the one HSGP covariate.

    Shrunk toward the league's own weighted mean by `ballast` batted balls,
    then standardized batted-ball-weighted across the players present.
    """
    league = float(vals["wv"].sum() / vals["w"].sum())
    x = ((vals["wv"].to_numpy() + ballast * league)
         / (vals["w"].to_numpy() + ballast))
    wt = vals["bbe_raw"].to_numpy(dtype="float64")
    mu = float(np.average(x, weights=wt))
    sd = float(np.sqrt(np.average((x - mu) ** 2, weights=wt)))
    return pd.DataFrame({"player": vals["player"].to_numpy(),
                         "bbe_raw": wt,
                         "xcon": (x - mu) / sd if sd > 0 else 0.0})


__all__ = ["EV_EDGES", "LA_EDGES", "MIN_CELL_BBE", "Surface",
           "window_batted_balls",
           "batted_balls_from_pa", "cell_centres", "cell_index", "fit_surface",
           "grid_cells", "player_values", "shrink_and_standardize"]
