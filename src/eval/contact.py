"""Contact quality as covariates on a component projection (BAS-58, stage 1).

The question this module exists to answer is narrower than "is Statcast
useful". The projection harness already sees the *outcomes* contact produced —
a hitter's home runs, his hits on balls in play, his extra bases — and Marcel
regresses those to the league with a fitted ballast. Exit velocity and launch
angle are a *measurement of the same events*. So the only thing worth asking
is:

    does a contact-quality aggregate carry information about the rest of the
    season **beyond** what the realized outcome rate already carries, or does
    it merely restate it with less noise?

Both answers are interesting and they look different in the numbers. Extra
information shows up as a gain that does not shrink with sample size. Pure
variance reduction shows up as a gain concentrated in players with little
pre-cutoff exposure and absent in the ones with a lot — so
`scripts/run_contact_backtest.py` splits the paired difference by pre-cutoff
trials, and the write-up reports that split whichever way it comes out.

**The estimator.** Three arms, all reading the same training frame:

    marcel_tuned     the baseline, untouched: the live model.
    contact_recal    a + b * marcel_tuned, with (a, b) fitted walk-forward on
                     *earlier seasons only*. This is the control that absorbs
                     any pure recalibration gain, so the covariate is not
                     credited with one.
    contact          a + b * marcel_tuned + sum_k g_k * z_k, the same fit with
                     standardized contact-quality covariates added.

`contact` minus `marcel_tuned` is the gate. `contact` minus `contact_recal` is
the covariate's own contribution. Every constant — the recency weights over
seasons, the shrinkage ballast on the covariates, and the coefficients
themselves — is fitted on cells strictly before the season being scored.

**The covariates**, each shrunk toward the league by `ballast` batted balls
and then standardized (batted-ball weighted) across the players present at
that cutoff:

    ev_mean     mean exit velocity
    ev90        90th-percentile exit velocity, off the committed histogram
    barrel      Statcast barrels per batted ball
    hardhit     batted balls at 95 mph or more, per batted ball
    sweetspot   launch angle in [8, 32], per batted ball
    la_mean     mean launch angle

**Leakage.** Contact features are summed from monthly buckets strictly before
the cutoff — `window_counts` refuses a cutoff that is not the first of a
month rather than rounding one (rounding forward is leakage), and
`assert_window_clean` re-checks that no bucket at or after the cutoff month
entered the sum. `tests/test_eval/test_contact.py` drives that guard with a
synthetic season whose post-cutoff months are 120 mph barrels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.contact_quality import (
    COUNT_COLUMNS,
    EV_BIN_COLUMNS,
    EV_BIN_EDGES,
)
from src.eval.backtest import COMPONENTS, ComponentSpec

# The covariates, in a fixed order so a coefficient vector is readable.
FEATURES = ("ev_mean", "ev90", "barrel", "hardhit", "sweetspot", "la_mean")

# Recency over the three seasons the projection window covers, most recent
# first — the current season through the cutoff, then the two before it.
CONTACT_WEIGHT_GRID = [
    (1.0, 0.0, 0.0),
    (1.0, 0.35, 0.1),
    (1.0, 0.6, 0.35),
    (1.0, 0.8, 0.6),
    (1.0, 1.0, 1.0),
]

# Batted balls of league-average contact to regress a player's own toward.
# Barrel rate is the noisiest of these per batted ball, so the ballast is set
# for the set rather than per feature; the grid is swept walk-forward.
CONTACT_BALLAST_GRID = [1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]

# The two constants, per side of the ball, as
# `scripts/run_contact_backtest.py --tune` chose them: the pooled
# trials-weighted MAE of the contact arm over the contact-dependent components
# on the *tuning* seasons (2019 and 2021), with every later season untouched.
# Both surfaces are shallow — the whole grid spans 0.4% of MAE on the hitter
# side — so read these as "the flat region", not as sharp optima.
TUNED = {
    "hitter": {"weights": (1.0, 0.35, 0.1), "ballast": 5.0},
    "pitcher": {"weights": (1.0, 0.6, 0.35), "ballast": 20.0},
}
DEFAULT_WINDOW_WEIGHTS = TUNED["hitter"]["weights"]
DEFAULT_BALLAST = TUNED["hitter"]["ballast"]

EV90_Q = 0.90


# --- window sums -------------------------------------------------------------

def assert_month_boundary(cutoff: pd.Timestamp) -> None:
    """Monthly buckets can only answer questions asked on the 1st of a month."""
    if pd.Timestamp(cutoff).day != 1:
        raise ValueError(
            f"contact-quality buckets are monthly; cutoff {pd.Timestamp(cutoff).date()} "
            "is not the first of a month. Rounding a cutoff forward would leak "
            "post-cutoff batted balls into the features."
        )


def assert_window_clean(rows: pd.DataFrame, cutoff, predict_year: int) -> None:
    """Leakage guard: no bucket at or after the cutoff month entered the window.

    Raises ValueError naming the offending buckets. `rows` is the monthly
    frame *after* the window filter, so this checks the filter rather than
    trusting it.
    """
    cutoff = pd.Timestamp(cutoff)
    if rows.empty:
        return
    if int(rows["season"].max()) > predict_year:
        raise ValueError(
            f"leakage: contact window contains season {int(rows['season'].max())} "
            f"> predict year {predict_year}")
    current = rows[rows["season"] == predict_year]
    if not current.empty and int(current["month"].max()) >= cutoff.month:
        bad = current[current["month"] >= cutoff.month]
        raise ValueError(
            f"leakage: {len(bad)} contact bucket(s) in month "
            f"{sorted(bad['month'].unique())} of {predict_year} are on or after "
            f"the cutoff {cutoff.date()}")


def window_counts(
    monthly: pd.DataFrame,
    side: str,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
) -> pd.DataFrame:
    """Recency-weighted contact counts per player, strictly before the cutoff.

    Sums the monthly buckets for the predict year (months before the cutoff
    month), the season before it and the one before that, at `weights`.
    Returns one row per player with every column of `COUNT_COLUMNS` plus an
    unweighted `bbe_raw` — the real batted balls behind the row, which is what
    a sample-size split should be cut on.
    """
    cutoff = pd.Timestamp(cutoff)
    assert_month_boundary(cutoff)
    # A weight of zero means the season is *not in the window*, so it is
    # dropped rather than multiplied by nothing: its batted balls would
    # otherwise still count toward `bbe_raw`, which is the exposure the
    # standardization weights by and the sample-size split is cut on.
    w = {predict_year - i: float(x) for i, x in enumerate(weights)
         if float(x) != 0.0}

    rows = monthly[(monthly["side"] == side)
                   & monthly["season"].isin(w)].copy()
    rows = rows[(rows["season"] < predict_year)
                | (rows["month"] < cutoff.month)]
    assert_window_clean(rows, cutoff, predict_year)
    if rows.empty:
        return pd.DataFrame(columns=["player", *COUNT_COLUMNS, "bbe_raw"])

    rows["_w"] = rows["season"].map(w).astype("float64")
    out = pd.DataFrame({"player": rows["player"].to_numpy()})
    for c in COUNT_COLUMNS:
        out[c] = rows[c].to_numpy() * rows["_w"].to_numpy()
    out["bbe_raw"] = rows["bbe"].to_numpy()
    return out.groupby("player", as_index=False).sum()


# --- metrics -----------------------------------------------------------------

def _quantile_from_hist(counts: np.ndarray, q: float) -> np.ndarray:
    """Interpolated quantile of an EV histogram, one row per player.

    `counts` is (n_players, n_bins) over the bins of
    `src.data.contact_quality`. The under- and over-flow bins are given a
    nominal 20 mph width so a quantile that lands in one is still finite and
    monotone; in practice EV90 never lands there for a player with any
    exposure at all.
    """
    lo = np.concatenate([[EV_BIN_EDGES[0] - 20.0], EV_BIN_EDGES])
    hi = np.concatenate([EV_BIN_EDGES, [EV_BIN_EDGES[-1] + 20.0]])
    total = counts.sum(axis=1)
    cum = np.cumsum(counts, axis=1)
    target = q * total
    idx = (cum < target[:, None]).sum(axis=1)
    idx = np.clip(idx, 0, counts.shape[1] - 1)
    rows = np.arange(counts.shape[0])
    below = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    inbin = counts[rows, idx]
    frac = np.where(inbin > 0, (target - below) / np.maximum(inbin, 1e-9), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    return lo[idx] + frac * (hi[idx] - lo[idx])


def league_profile(counts: pd.DataFrame) -> dict:
    """Pooled league contact profile from a window — the shrinkage target."""
    tot = counts[COUNT_COLUMNS].sum()
    bbe = float(tot["bbe"])
    if bbe <= 0:
        raise ValueError("empty contact window: no batted balls before the cutoff")
    prof = {c: float(tot[c]) / bbe for c in COUNT_COLUMNS if c != "bbe"}
    prof["bbe"] = bbe
    return prof


def contact_metrics(
    counts: pd.DataFrame, ballast: float = DEFAULT_BALLAST,
    league: dict | None = None,
) -> pd.DataFrame:
    """Shrunk contact-quality metrics per player.

    Every metric is a ratio whose numerator and denominator are both counts,
    so shrinkage is one operation: add `ballast` batted balls of the league's
    own profile to the player's counts, then take the ratio. The EV histogram
    is shrunk the same way, bin by bin, before the quantile — which is what
    keeps a 12-batted-ball EV90 from being a single lucky ball.
    """
    if counts.empty:
        return pd.DataFrame(columns=["player", "bbe_raw", *FEATURES])
    league = league or league_profile(counts)
    b = float(ballast)

    bbe = counts["bbe"].to_numpy(dtype="float64") + b

    def shrunk(col: str) -> np.ndarray:
        return (counts[col].to_numpy(dtype="float64") + b * league[col]) / bbe

    hist = (counts[EV_BIN_COLUMNS].to_numpy(dtype="float64")
            + b * np.array([league[c] for c in EV_BIN_COLUMNS])[None, :])

    return pd.DataFrame({
        "player": counts["player"].to_numpy(),
        "bbe_raw": counts["bbe_raw"].to_numpy(dtype="float64"),
        "ev_mean": shrunk("sum_ev"),
        "ev90": _quantile_from_hist(hist, EV90_Q),
        "barrel": shrunk("n_barrel"),
        "hardhit": shrunk("n_hardhit"),
        "sweetspot": shrunk("n_sweetspot"),
        "la_mean": shrunk("sum_la"),
    })


def standardize(metrics: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Batted-ball-weighted z-scores of each metric across the players present.

    Weighted by raw exposure so the centre is the league's typical batted ball
    rather than the typical September call-up, and computed from the cutoff's
    own pre-cutoff window, so nothing outside the training data enters it.
    """
    out = metrics[["player", "bbe_raw"]].copy()
    w = metrics["bbe_raw"].to_numpy(dtype="float64")
    if w.sum() <= 0:
        w = np.ones_like(w)
    for f in features:
        x = metrics[f].to_numpy(dtype="float64")
        mu = float(np.average(x, weights=w))
        sd = float(np.sqrt(np.average((x - mu) ** 2, weights=w)))
        out[f] = (x - mu) / sd if sd > 0 else 0.0
    return out


def features_at_cutoff(
    monthly: pd.DataFrame,
    side: str,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
    ballast: float = DEFAULT_BALLAST,
) -> pd.DataFrame:
    """Standardized contact covariates for every player with pre-cutoff contact."""
    counts = window_counts(monthly, side, cutoff, predict_year, weights)
    if counts.empty:
        return pd.DataFrame(columns=["player", "bbe_raw", *FEATURES])
    return standardize(contact_metrics(counts, ballast))


# --- the estimator -----------------------------------------------------------

@dataclass(frozen=True)
class ContactFit:
    """Coefficients of one arm, and what they were fitted on."""
    component: str
    features: tuple[str, ...]
    coef: dict[str, float]
    n_cells: int
    n_rows: int
    seasons: tuple[int, ...] = ()

    def predict(self, base: np.ndarray, z: pd.DataFrame | None) -> np.ndarray:
        out = self.coef["intercept"] + self.coef["base"] * np.asarray(base)
        for f in self.features:
            out = out + self.coef[f] * z[f].to_numpy(dtype="float64")
        return out


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * sw, y * np.sqrt(w), rcond=None)
    return beta


def fit_contact(
    cells: pd.DataFrame, component: str, features=FEATURES,
    fixed_base: bool = False,
) -> ContactFit:
    """Weighted least squares of the realized rest-of-season rate on the
    baseline projection and the contact covariates.

    Rows are (player, cutoff-cell) pairs from *earlier seasons only*; the
    weight is the realized trials, which is the harness's own scoring weight.
    Pass `features=()` for the recalibration control.

    `fixed_base` pins the coefficient on the baseline at exactly 1 and
    regresses the *residual* on the covariates instead. That is the deployable
    shape of this idea — the served projection is left alone and contact
    quality is a correction added to it — and separating it from the free fit
    says how much of the gain needs the baseline rescaled and how much does
    not.
    """
    g = cells[cells["component"] == component]
    if g.empty:
        raise ValueError(f"no training cells for {component!r}")
    y = g["realized_rate"].to_numpy(dtype="float64")
    w = g["trials"].to_numpy(dtype="float64")
    base = g["base"].to_numpy(dtype="float64")
    feat = [g[f].to_numpy(dtype="float64") for f in features]

    if fixed_base:
        X = np.column_stack([np.ones(len(g))] + feat) if feat else \
            np.ones((len(g), 1))
        beta = _wls(X, y - base, w)
        coef = {"intercept": float(beta[0]), "base": 1.0}
        coef.update({f: float(b) for f, b in zip(features, beta[1:])})
    else:
        X = np.column_stack([np.ones(len(g)), base] + feat)
        beta = _wls(X, y, w)
        coef = {"intercept": float(beta[0]), "base": float(beta[1])}
        coef.update({f: float(b) for f, b in zip(features, beta[2:])})
    return ContactFit(component=component, features=tuple(features), coef=coef,
                      n_cells=int(g[["season", "cutoff"]].drop_duplicates().shape[0]),
                      n_rows=int(len(g)),
                      seasons=tuple(sorted(g["season"].unique().tolist())))


# --- the provider ------------------------------------------------------------

@dataclass
class ContactProviderConfig:
    """Everything a contact arm needs that the provider signature cannot carry."""
    monthly: pd.DataFrame
    cutoff: str
    predict_year: int
    fit: ContactFit
    base_provider: object
    side: str = "hitter"
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS
    ballast: float = DEFAULT_BALLAST
    clip: tuple[float, float] = (1e-4, 0.999)
    _cache: dict = field(default_factory=dict)


def contact_provider(config: ContactProviderConfig):
    """A harness provider: baseline projection plus fitted contact covariates.

    Covers exactly the players the baseline covers — a player with no tracked
    contact before the cutoff gets z = 0 on every covariate and therefore the
    recalibrated baseline, rather than being dropped. That matters for the
    gate: the common player set is the baseline's, so the paired comparison is
    not quietly run on a different population.
    """

    def provider(train: pd.DataFrame, spec: ComponentSpec, predict_year: int):
        base = config.base_provider(train, spec, predict_year)
        z = features_at_cutoff(config.monthly, config.side, config.cutoff,
                               config.predict_year, config.weights,
                               config.ballast)
        zi = z.set_index("player").reindex(base[spec.id_col].to_numpy())
        for f in config.fit.features:
            zi[f] = zi[f].fillna(0.0)
        pred = config.fit.predict(base["predicted"].to_numpy(dtype="float64"), zi)
        out = base[[spec.id_col]].copy()
        out["predicted"] = np.clip(pred, *config.clip)
        return out

    return provider


def spec_for(component: str) -> ComponentSpec:
    return COMPONENTS[component]


__all__ = [
    "CONTACT_BALLAST_GRID", "CONTACT_WEIGHT_GRID", "DEFAULT_BALLAST",
    "DEFAULT_WINDOW_WEIGHTS", "FEATURES", "ContactFit",
    "ContactProviderConfig", "assert_month_boundary", "assert_window_clean",
    "contact_metrics", "contact_provider", "features_at_cutoff",
    "fit_contact", "league_profile", "standardize", "window_counts",
]
