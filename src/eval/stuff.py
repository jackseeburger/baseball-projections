"""Stuff aggregates as covariates on a pitcher component projection (BAS-71, stage 2).

The pitcher mirror of `src/eval/contact.py`, and deliberately the same object
so the gate, the controls and the honesty checks are reused rather than
reinvented. The question has the same shape:

    does a measurement of the *pitch* carry information about the rest of the
    season beyond what the pitcher's realized strikeout, walk and home-run
    rates already carry, or does it merely restate them with less noise?

**The arms.**

    marcel_pitcher_tuned   the live baseline, untouched
    stuff_recal            a + b * baseline, with (a, b) fitted walk-forward
                           on earlier seasons only — the control that absorbs
                           a pure recalibration gain
    stuff                  the same fit plus standardized stuff covariates

`stuff` minus `marcel_pitcher_tuned` is the gate. `stuff` minus `stuff_recal`
is what the covariate itself is worth. Contact quality's pitcher walk rates are
the reason the control is not optional: they showed a t of -5.6 against the
baseline of which *none* was contact quality, all of it a fitted rescaling of
Marcel.

**The covariates**, each a ratio of two additive monthly sums, shrunk toward
the league by `ballast` pitches and then standardized (pitch-weighted) across
the pitchers present at that cutoff:

    xwhiff      predicted whiffs per swing, over all pitches
    xcsw        predicted called-strikes-plus-whiffs per pitch
    xwhiff_fb   the same on four-seamers and sinkers only
    xwhiff_nfb  the same on everything else
    velo        mean release speed
    fb_share    fastballs as a fraction of pitches

`velo` and `fb_share` are in because the pre-registration's second control is
fastball velocity: if the aggregate's whole content were "he throws hard", the
whiff covariates would carry no coefficient once velocity is in the same fit.

**Leakage.** Identical to contact quality's: features are summed from monthly
buckets strictly before the cutoff, a cutoff that is not the first of a month
is refused rather than rounded, and the guard re-checks the filtered rows.
`src.eval.contact.assert_month_boundary` and `assert_window_clean` are imported
rather than copied — the check is the same check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.pitching_stuff import COUNT_COLUMNS
from src.eval.backtest import COMPONENTS, ComponentSpec
from src.eval.contact import assert_month_boundary, assert_window_clean

FEATURES = ("xwhiff", "xcsw", "xwhiff_fb", "xwhiff_nfb", "velo", "fb_share")

# Recency over the three seasons the projection window covers, most recent
# first — the current season through the cutoff, then the two before it. Same
# grid contact quality sweeps, so the two results are comparable.
STUFF_WEIGHT_GRID = [
    (1.0, 0.0, 0.0),
    (1.0, 0.35, 0.1),
    (1.0, 0.6, 0.35),
    (1.0, 0.8, 0.6),
    (1.0, 1.0, 1.0),
]
# Pitches of league-average stuff to regress a pitcher's own toward. A pitch is
# a much cheaper observation than a batted ball — a starter throws 3,000 in a
# season against 400 batted balls allowed — so the grid runs an order of
# magnitude higher than contact quality's.
STUFF_BALLAST_GRID = [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]

# What `scripts/run_stuff_backtest.py --tune` chose on the tuning seasons
# (2019 and 2021), by pooled trials-weighted MAE of the stuff arm over K/BF and
# HR/BF, with every later season untouched. Both land on the **corner** of the
# grid — the current season alone, and the smallest ballast offered — and that
# is worth reading rather than papering over.
#
# The ballast corner is not a failure to bracket the optimum, it is the
# quantity being different from contact quality's. A pitcher's covariate is
# already an average over 1,500-3,000 model predictions, each of which is
# itself a shrunk estimate; there is very little sampling noise left for a
# ballast to remove, so the grid slides to the bottom and the whole ballast
# sweep at fixed weights moves MAE by 1.1% (.020896 to .021126 at 1,000).
# The recency corner is a larger effect and a real claim: over the grid, MAE
# runs .020896 for the current season alone to .021629 for a flat three-season
# window, 3.5%. Stuff moves — a pitcher who added two miles an hour is a
# different pitcher — and averaging in two-year-old pitches makes the
# measurement worse. That is the same statement prediction 3 makes.
DEFAULT_WINDOW_WEIGHTS = (1.0, 0.0, 0.0)
DEFAULT_BALLAST = 10.0


# --- window sums -------------------------------------------------------------

def window_counts(
    monthly: pd.DataFrame,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
) -> pd.DataFrame:
    """Recency-weighted stuff counts per pitcher, strictly before the cutoff.

    Sums the monthly buckets for the predict year (months before the cutoff
    month), the season before it and the one before that, at `weights`.
    Returns one row per pitcher with every column of `COUNT_COLUMNS` plus an
    unweighted `pitches_raw` — the real pitches behind the row, which is what
    the exposure split is cut on and what the standardization weights by.
    """
    cutoff = pd.Timestamp(cutoff)
    assert_month_boundary(cutoff)
    # A weight of zero means the season is not in the window at all, so it is
    # dropped rather than multiplied by nothing — otherwise its pitches would
    # still count toward `pitches_raw`.
    w = {predict_year - i: float(x) for i, x in enumerate(weights)
         if float(x) != 0.0}

    rows = monthly[monthly["season"].isin(w)].copy()
    rows = rows[(rows["season"] < predict_year)
                | (rows["month"] < cutoff.month)]
    assert_window_clean(rows, cutoff, predict_year)
    if rows.empty:
        return pd.DataFrame(columns=["player", *COUNT_COLUMNS, "pitches_raw"])

    rows["_w"] = rows["season"].map(w).astype("float64")
    out = pd.DataFrame({"player": rows["pitcher"].to_numpy()})
    for c in COUNT_COLUMNS:
        out[c] = rows[c].to_numpy() * rows["_w"].to_numpy()
    out["pitches_raw"] = rows["pitches"].to_numpy()
    return out.groupby("player", as_index=False).sum()


# --- metrics -----------------------------------------------------------------

def league_profile(counts: pd.DataFrame) -> dict:
    """Pooled league stuff profile from a window — the shrinkage target.

    Every count is expressed per pitch, including the swing count, so the
    shrinkage of a per-swing rate adds league-average swings along with the
    league-average predicted whiffs on them.
    """
    tot = counts[COUNT_COLUMNS].sum()
    pitches = float(tot["pitches"])
    if pitches <= 0:
        raise ValueError("empty stuff window: no pitches before the cutoff")
    prof = {c: float(tot[c]) / pitches for c in COUNT_COLUMNS if c != "pitches"}
    prof["pitches"] = pitches
    return prof


def stuff_metrics(counts: pd.DataFrame, ballast: float = DEFAULT_BALLAST,
                  league: dict | None = None) -> pd.DataFrame:
    """Shrunk stuff metrics per pitcher.

    Every metric is a ratio of two additive counts, so shrinkage is one
    operation: add `ballast` pitches of the league's own profile to the
    pitcher's counts, then take the ratio. Both numerator and denominator are
    shrunk, which is what keeps a reliever with forty fastballs from having a
    fastball whiff rate built out of four swings.
    """
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *FEATURES])
    league = league or league_profile(counts)
    b = float(ballast)

    def sh(col: str) -> np.ndarray:
        return counts[col].to_numpy(dtype="float64") + b * league[col]

    pitches = counts["pitches"].to_numpy(dtype="float64") + b

    def ratio(num: str, den: str) -> np.ndarray:
        d = sh(den)
        return np.divide(sh(num), d, out=np.zeros(len(counts)), where=d > 0)

    return pd.DataFrame({
        "player": counts["player"].to_numpy(),
        "pitches_raw": counts["pitches_raw"].to_numpy(dtype="float64"),
        "xwhiff": ratio("p_whiff_sum", "swings"),
        "xcsw": sh("p_csw_sum") / pitches,
        "xwhiff_fb": ratio("fb_p_whiff_sum", "fb_swings"),
        "xwhiff_nfb": ratio("nfb_p_whiff_sum", "nfb_swings"),
        "velo": sh("sum_velo") / pitches,
        "fb_share": sh("fb_pitches") / pitches,
    })


def standardize(metrics: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Pitch-weighted z-scores of each metric across the pitchers present.

    Weighted by raw exposure so the centre is the league's typical *pitch*
    rather than the typical September call-up, and computed from the cutoff's
    own pre-cutoff window, so nothing outside the training data enters it.
    """
    out = metrics[["player", "pitches_raw"]].copy()
    w = metrics["pitches_raw"].to_numpy(dtype="float64")
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
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
    ballast: float = DEFAULT_BALLAST,
) -> pd.DataFrame:
    """Standardized stuff covariates for every pitcher with pre-cutoff pitches."""
    counts = window_counts(monthly, cutoff, predict_year, weights)
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *FEATURES])
    return standardize(stuff_metrics(counts, ballast))


# --- the estimator -----------------------------------------------------------

@dataclass(frozen=True)
class StuffFit:
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


def fit_stuff(cells: pd.DataFrame, component: str, features=FEATURES,
              fixed_base: bool = False) -> StuffFit:
    """Weighted least squares of the realized rest-of-season rate on the
    baseline projection and the stuff covariates.

    Rows are (pitcher, cutoff-cell) pairs from *earlier seasons only*; the
    weight is the realized batters faced, which is the harness's own scoring
    weight. Pass `features=()` for the recalibration control.

    `fixed_base` pins the coefficient on the baseline at exactly 1 and
    regresses the residual on the covariates instead — the deployable shape,
    which leaves the served projection alone and adds stuff as a correction.
    """
    g = cells[cells["component"] == component]
    if g.empty:
        raise ValueError(f"no training cells for {component!r}")
    y = g["realized_rate"].to_numpy(dtype="float64")
    w = g["trials"].to_numpy(dtype="float64")
    base = g["base"].to_numpy(dtype="float64")
    feat = [g[f].to_numpy(dtype="float64") for f in features]

    if fixed_base:
        X = (np.column_stack([np.ones(len(g))] + feat) if feat
             else np.ones((len(g), 1)))
        beta = _wls(X, y - base, w)
        coef = {"intercept": float(beta[0]), "base": 1.0}
        coef.update({f: float(b) for f, b in zip(features, beta[1:])})
    else:
        X = np.column_stack([np.ones(len(g)), base] + feat)
        beta = _wls(X, y, w)
        coef = {"intercept": float(beta[0]), "base": float(beta[1])}
        coef.update({f: float(b) for f, b in zip(features, beta[2:])})
    return StuffFit(component=component, features=tuple(features), coef=coef,
                    n_cells=int(g[["season", "cutoff"]].drop_duplicates().shape[0]),
                    n_rows=int(len(g)),
                    seasons=tuple(sorted(g["season"].unique().tolist())))


# --- the provider ------------------------------------------------------------

@dataclass
class StuffProviderConfig:
    """Everything a stuff arm needs that the provider signature cannot carry."""
    monthly: pd.DataFrame
    cutoff: str
    predict_year: int
    fit: StuffFit
    base_provider: object
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS
    ballast: float = DEFAULT_BALLAST
    clip: tuple[float, float] = (1e-4, 0.999)
    _cache: dict = field(default_factory=dict)


def stuff_provider(config: StuffProviderConfig):
    """A harness provider: baseline projection plus fitted stuff covariates.

    Covers exactly the pitchers the baseline covers — a pitcher with no tracked
    pitches before the cutoff gets z = 0 on every covariate and therefore the
    recalibrated baseline, rather than being dropped. The common pitcher set is
    the baseline's, so the paired comparison is not quietly run on a different
    population.
    """

    def provider(train: pd.DataFrame, spec: ComponentSpec, predict_year: int):
        base = config.base_provider(train, spec, predict_year)
        z = features_at_cutoff(config.monthly, config.cutoff,
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


# --- walk-forward fit for live serving ---------------------------------------

# The cells `scripts/run_stuff_backtest.py` builds, named here so the serving
# path fits on exactly the same rows the gate was scored on rather than on a
# second, subtly different definition. 2020 is out everywhere in this repo: a
# 60-game season that started July 23 has no May 1 cutoff.
LIVE_CELL_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
LIVE_CUTOFF_MONTHS = ("05-01", "07-01", "08-01")
LIVE_MIN_TRIALS = 100


def build_pitcher_cells(
    seasons_table: pd.DataFrame,
    pa_dir,
    components,
    seasons=LIVE_CELL_SEASONS,
    cutoff_months=LIVE_CUTOFF_MONTHS,
    min_trials: int = LIVE_MIN_TRIALS,
) -> pd.DataFrame:
    """One row per (component, season, cutoff, pitcher): the baseline
    projection, the realized rest-of-season rate, trials and pre-cutoff
    exposure.

    The pitcher mirror of `contact.build_hitter_cells`, and the same frame
    `scripts/run_stuff_backtest.py`'s `build_cells` produces — the split is the
    harness's own (`partial_and_realized` either side of the date,
    `assert_split_clean` on both, the same `min_trials` filter and the same
    intersection with the baseline's coverage). It lives here rather than in
    the script so `fit_live_stuff` can call it at serve time and fit on
    exactly what the gate scored.
    """
    from src.eval import pitchers as pitcher_eval
    from src.eval.intraseason import assert_split_clean, build_training_frame

    rows = []
    for season in seasons:
        pa = pd.read_parquet(
            f"{pa_dir}/pa_outcomes_{season}.parquet",
            columns=["batter", "pitcher", "game_pk", "game_date", "game_year",
                     "event", "is_k", "is_bb", "is_hbp", "is_hit", "is_hr",
                     "is_single", "is_double", "is_triple"])
        pa["game_date"] = pd.to_datetime(pa["game_date"])
        for md in cutoff_months:
            cutoff = f"{season}-{md}"
            partial, realized = pitcher_eval.partial_and_realized(pa, cutoff,
                                                                  season)
            train = build_training_frame(seasons_table, partial, season,
                                         "pitcher")
            assert_split_clean(train, realized, cutoff, season)
            pre = partial.set_index("pitcher")
            for component in components:
                spec = COMPONENTS[component]
                real = realized[realized[spec.trials] >= min_trials]
                if real.empty:
                    continue
                base = pitcher_eval.marcel_pitcher_tuned(
                    train, spec, season)[["pitcher", "predicted"]]
                base = base.dropna(subset=["predicted"])
                j = real[["pitcher", spec.successes, spec.trials]].merge(
                    base, on="pitcher", how="inner")
                if j.empty:
                    continue
                rows.append(pd.DataFrame({
                    "component": component, "side": "pitcher",
                    "season": season, "cutoff": cutoff,
                    "player": j["pitcher"].to_numpy(),
                    "base": j["predicted"].to_numpy(dtype="float64"),
                    "realized_successes": j[spec.successes].to_numpy(
                        dtype="float64"),
                    "realized_rate": (j[spec.successes] / j[spec.trials]
                                      ).to_numpy(dtype="float64"),
                    "trials": j[spec.trials].to_numpy(dtype="float64"),
                    "pre_trials": pre[spec.trials].reindex(
                        j["pitcher"].to_numpy()).fillna(0.0
                                                        ).to_numpy(dtype="float64"),
                }))
    if not rows:
        return pd.DataFrame(columns=["component", "side", "season", "cutoff",
                                     "player", "base", "realized_successes",
                                     "realized_rate", "trials", "pre_trials",
                                     *FEATURES])
    return pd.concat(rows, ignore_index=True)


def attach_live_features(
    cells: pd.DataFrame, monthly: pd.DataFrame,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
    ballast: float = DEFAULT_BALLAST,
) -> pd.DataFrame:
    """Merge the standardized stuff covariates onto `build_pitcher_cells`'s
    output, one cutoff-cell at a time (the covariates do not depend on the
    component). A pitcher with no tracked pitches before the cutoff gets z = 0.
    """
    out = []
    for (season, cutoff), g in cells.groupby(["season", "cutoff"]):
        z = features_at_cutoff(monthly, cutoff, season, weights, ballast)
        zi = z.set_index("player").reindex(g["player"].to_numpy())
        g = g.copy()
        for f in FEATURES:
            g[f] = zi[f].fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def fit_live_stuff(
    component: str,
    seasons_table: pd.DataFrame,
    monthly: pd.DataFrame,
    pa_dir,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_WINDOW_WEIGHTS,
    ballast: float = DEFAULT_BALLAST,
    fixed_base: bool = True,
) -> StuffFit:
    """The served arm's coefficients for `predict_year`, fitted exactly as the
    harness fits them walk-forward: on cell seasons strictly before the one
    being served, never on `predict_year` itself.

    `fixed_base=True` (the default) is `stuff_additive` — the shape
    docs/pitching-stuff.md's "Serving" section pre-registered before this was
    wired: the baseline's coefficient is pinned at exactly 1 and the stuff
    aggregate is a pure correction added to `marcel_pitcher_tuned`, rather
    than a fit that also rescales the baseline. The free `stuff` arm's extra
    gain on the two walk rates is almost entirely that rescaling (the control
    `stuff_recal` gets −2.49% of the free fit's −3.26% on BB/BF by itself),
    which is a claim about the pitcher Marcel's ballasts and belongs in a
    ticket about the pitcher Marcel.
    """
    train_seasons = tuple(s for s in LIVE_CELL_SEASONS if s < predict_year)
    if not train_seasons:
        raise ValueError(
            f"no stuff training seasons strictly before {predict_year}")
    cells = build_pitcher_cells(seasons_table, pa_dir, [component],
                                seasons=train_seasons)
    if cells.empty:
        raise ValueError(f"no stuff training cells for {component!r} "
                         f"before {predict_year}")
    cells = attach_live_features(cells, monthly, weights, ballast)
    return fit_stuff(cells, component, features=FEATURES,
                     fixed_base=fixed_base)


__all__ = [
    "DEFAULT_BALLAST", "DEFAULT_WINDOW_WEIGHTS", "FEATURES",
    "LIVE_CELL_SEASONS", "LIVE_CUTOFF_MONTHS", "LIVE_MIN_TRIALS",
    "STUFF_BALLAST_GRID", "STUFF_WEIGHT_GRID", "StuffFit",
    "StuffProviderConfig", "attach_live_features", "build_pitcher_cells",
    "features_at_cutoff", "fit_live_stuff", "fit_stuff", "league_profile",
    "standardize", "stuff_metrics", "stuff_provider", "window_counts",
]
