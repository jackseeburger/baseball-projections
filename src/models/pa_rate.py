"""
PA-level Bayesian Rate Model — one hierarchical binomial, several components

Hierarchical Bayesian model for projecting a batter's per-plate-appearance
rate, collapsed to Binomial counts per (batter, season, team, stand [,
pitcher]) cell with a logit-linear predictor:

    logit(p) = league_trend[season] + player_ability[batter]
             + pitcher_ability[pitcher]
             + handedness[stand] + park_effect[team]
             + age_curve(age)

    successes ~ Binomial(n, logistic(eta))   per cell

**One model, three numerators (BAS-73).** This module started life as
`pa_k_rate.py`, the K% model. BB/PA and HR/PA are the same object: the same
cells, the same likelihood, the same random effects — a different numerator
column in the PA parquet (`is_bb`, `is_hr` against `is_k`), a different
league-level prior mean, and a different direction for the age curve. So the
component is a parameter (`RATE_COMPONENTS`, `RateComponent`) rather than a
copy of the file. `src.models.pa_k_rate` is a thin wrapper that pins
`component="k_rate"`, and `tests/test_models/test_pa_rate.py` pins the K%
numbers bit-for-bit against a fixture generated from the pre-split module, so
"the same object" is a checked claim and not a hopeful one.

The three components and where their differences live:

    component   numerator   league rate (2019-2026)   age curve
    k_rate      is_k        .221-.232, logit ~ -1.23  valley (K% rises after peak)
    bb_rate     is_bb       .082-.089, logit ~ -2.38  hill  (BB% falls after peak)
    hr_rate     is_hr       .029-.036, logit ~ -3.44  hill  (HR/PA falls after peak)

`bb_rate`'s numerator is `is_bb` alone, *not* `is_bb + is_hbp` — matched to
what `src.eval.backtest.COMPONENTS["bb_rate"]` scores, whose `bb` column
`src.eval.intraseason.aggregate_pa` builds from `is_bb` only and keeps `hbp`
in a column of its own. Getting this wrong would score the model against a
denominator it was never fit on, and the error (HBP is ~1% of PA against
BB's ~8.5%) is big enough to matter and small enough to miss.

Components of the linear predictor:
    - League trend: random walk on logit scale across seasons
    - Player ability: partial pooling with non-centered parameterization
    - Opposing pitcher: partial pooling, non-centered, mean fixed at zero
      (BAS-59). Without it a hitter who drew tough arms looks worse than he
      is; the term reads a batter's rate net of who he faced. Projections are
      made at a neutral pitcher, which is the point of having it.
    - Handedness: batter stand (L/R) effect
    - Park effects: ZeroSumNormal across teams
    - Age curve: quadratic on centered age (peak ~ 27)

Likelihood: Binomial(n, logistic(eta)) per cell — identical to per-PA
Bernoulli up to a constant, with ~10x fewer likelihood rows (roadmap 0.4).
The pitcher term is the one thing that varies *within* an old cell, so
`include_pitcher=True` puts `pitcher_idx` in the cell key: exact, but the
compression drops to roughly 2 PA per cell (2026: 796 cells → 81,468). That
is the price of the term and the full refit has to budget for it.

**Dated cutoffs (BAS-59).** `cutoff_date` gives the model the same partial
current season the intra-season baselines get: PA strictly before the cutoff
train the model, PA on or after it are withheld, and the current season is
its own cell in the season random walk carrying its actual partial exposure.
See `prepare_model_data` for exactly where the semantics match
`src.eval.baselines.marcel` and where they cannot.

**Structural variants (`ModelOptions`, docs/bayes-variants.md).** Two flags,
pre-registered before being scored: `ability_walk` turns the fixed player
ability into a per-batter random walk over seasons, and `constrained_age`
replaces the quadratic age curve with a peak plus two signed slopes. Both
default off; `build_model(data)` with no options is the model above,
unchanged. Both work for every component. See `build_model`'s docstring.

Designed for Modal deployment (8GB RAM, 4 CPU, NumpyRo backend).

Usage:
    from src.models.pa_rate import run_model
    run_model(component="hr_rate")
"""

from __future__ import annotations

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.cutoff import apply_cutoff, assert_no_post_cutoff, cutoff_exposure
from src.models.pa_components import (  # noqa: F401  (re-exported surface)
    AGE_DIRECTION, AGE_PEAK_WINDOW, DEFAULT_COMPONENT, RATE_COMPONENTS,
    RateComponent, get_component,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_DIR = DATA_DIR / "parquet"
FEATURES_DIR = DATA_DIR / "features"

# Model hyperparameters
REFERENCE_AGE = 27.0       # center of age curve (typical peak)
MIN_PA_THRESHOLD = 50      # minimum PAs to include a batter
PROJECTION_YEAR = 2026
# Prior scale on the opposing-pitcher random effect, on the logit scale.
# `pm.find_constrained_prior(pm.HalfNormal, lower=0, upper=0.45, mass=0.95)`
# returns sigma ≈ 0.23; 0.45 on the logit scale is roughly the gap between a
# league-average arm and an elite one at league K% (.22 → .31), so the prior
# puts 95% of its mass on pitcher spreads no wider than the ones we can see.
PITCHER_SIGMA_PRIOR = 0.23
# Prior scale on the season-to-season random-walk step of a batter's ability,
# on the logit scale (`ability_walk`, see docs/bayes-variants.md). 0.15 is
# roughly a 3-point year-over-year swing in K% at league level (d(logit)/dp
# at p=.22 is 1/(p(1-p)) ≈ 5.8, so a 0.15 logit step ≈ 0.026 of rate) — a
# large but observed one-season change for a real hitter, not a hypothetical
# one. The HalfNormal puts most of its mass well below that: at sigma=0.15,
# P(|step| > 0.15) ≈ 0.32, so a full-league-sized swing is already in the
# tail, and a Marcel-sized one (a point or two of K%) is unremarkable.
ABILITY_STEP_SIGMA_PRIOR = 0.15
SAMPLER_KWARGS = dict(
    draws=2000,
    tune=1500,
    chains=4,
    cores=1,                # Modal: 4 CPU but NumpyRo handles parallelism internally
    target_accept=0.9,
    nuts_sampler="numpyro",
    random_seed=42,
    idata_kwargs={"log_likelihood": False},  # save memory
)


def league_logit_earliest_season(cells: pd.DataFrame) -> float:
    """logit of the pooled success rate in the training data's earliest season.

    The league random walk starts at `league_init` and every later season is
    reached by an innovation, so the prior on `league_init` is a prior on the
    *first* season in the window, not on the average of them. Computed from
    the aggregated cells (`k` successes over `n` trials), so it costs two
    sums and needs no second pass over the PA rows.
    """
    first = cells[cells["season_idx"] == cells["season_idx"].min()]
    n = float(first["n"].sum())
    k = float(first["k"].sum())
    # Guard the degenerate ends so a tiny fixture cannot produce +-inf: one
    # half-success of Laplace smoothing, which moves nothing at real scale
    # (a season is ~180,000 PA).
    p = min(max((k + 0.5) / (n + 1.0), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1.0 - p)))


@dataclass(frozen=True)
class ModelOptions:
    """Structural variants of the K% model, pre-registered before being scored
    (docs/bayes-variants.md) so a positive or negative result is not chosen
    after seeing which one looks better.

    ability_walk:    per-batter ability is a Gaussian random walk over seasons
        instead of one fixed level (`build_model`'s "Player ability" block).
        Answers whether a hitter's true skill drifts within a career faster
        than the fixed-effect model can express, at the cost of one more
        hyperparameter (`sigma_step`) to estimate from the same data.
    constrained_age: the quadratic age curve is replaced by a peak-plus-two-
        signed-slopes curve, the same shape `src.eval.tuning`'s constrained
        Marcel age term uses, so a fitted peak has to actually be a peak
        (turn over) rather than a parabola that can silently act as a level
        correction on the whole age range.
    covariates:      per-(batter, season) layer-1 covariates entering the
        batter's logit rate on top of the ability term, as
        `sum_j beta_cov_j * x[batter, season, j]` with each
        `beta_cov_<aggregate> ~ Normal(0, BETA_COV_SIGMA)`. `None` is off and
        is bit-for-bit the model above; `"contact"` selects the six
        contact-quality aggregates (`src.models.pa_covariates`), and an
        explicit tuple of aggregate names selects a subset. The design matrix
        is *not* built here — `pa_covariates.attach_covariates` puts it on the
        data dict — so with covariates off nothing about the data pipeline
        changes either. See docs/bayes-covariates.md.

    Both flags default to False and `covariates` to None, which is the model
    exactly as it existed before any variant — `build_model(data)` with no
    options is byte-for-byte the old behaviour.
    """
    ability_walk: bool = False
    constrained_age: bool = False
    covariates: tuple[str, ...] | str | None = None

    def covariate_names(self) -> tuple[str, ...]:
        """The aggregates `covariates` selects, resolved and validated."""
        from src.models.pa_covariates import covariate_names

        return covariate_names(self.covariates)

    def label(self) -> str:
        cov = self.covariate_names()
        return (f"ability={'walk' if self.ability_walk else 'flat'}, "
                f"age={'constrained' if self.constrained_age else 'quadratic'}"
                + (f", covariates={'+'.join(cov)}" if cov else ""))

    def to_dict(self) -> dict:
        return {"ability_walk": self.ability_walk,
                "constrained_age": self.constrained_age,
                "covariates": list(self.covariate_names()) or None}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

BASE_PA_COLUMNS = [
    "batter", "game_year", "stand", "is_k",
    "home_team", "away_team", "inning_topbot",
]


def base_pa_columns(component: RateComponent) -> list[str]:
    """`BASE_PA_COLUMNS` with this component's numerator in place of `is_k`.

    Kept in the `is_k` slot rather than appended so K% reads exactly the
    column list, in exactly the order, that it always did — the parquet
    reader does not care, but a diff of the two frames would.
    """
    return [component.numerator if c == "is_k" else c for c in BASE_PA_COLUMNS]


def load_pa_data(
    pa_dir: Path | str | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
    include_pitcher: bool = False,
    component: str | RateComponent | None = None,
) -> pd.DataFrame:
    """Load and concatenate PA outcome parquet files.

    Args:
        pa_dir: Directory containing pa_outcomes_YYYY.parquet files.
                Defaults to DATA_DIR / 'parquet' / 'pa_outcomes'.
        cutoff_date: ISO date. Rows dated on or after it are dropped as they
            are read, so post-cutoff PA never reach memory, let alone the
            likelihood. Requires a `game_date` column.
        include_pitcher: also read `pitcher`, for the opposing-pitcher term.
        component: which rate to read the numerator for (default K%).

    Returns:
        DataFrame with one row per PA, columns include:
        batter, game_year, stand, <numerator>, home_team, away_team,
        inning_topbot (plus game_date and pitcher when asked for).
    """
    comp = get_component(component)
    if pa_dir is None:
        pa_dir = PARQUET_DIR / "pa_outcomes"
    pa_dir = Path(pa_dir)

    parquet_files = sorted(pa_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pa_dir}")

    logger.info(f"Loading {len(parquet_files)} PA parquet files from {pa_dir}")

    # Only read the columns we need to save memory
    keep_cols = base_pa_columns(comp)
    if cutoff_date is not None:
        keep_cols.append("game_date")
    if include_pitcher:
        keep_cols.append("pitcher")

    frames = []
    for f in parquet_files:
        df = pd.read_parquet(f, columns=keep_cols)
        if cutoff_date is not None:
            df = apply_cutoff(df, cutoff_date)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if cutoff_date is not None:
        assert_no_post_cutoff(data, cutoff_date)
        logger.info("cutoff %s: %s", cutoff_date,
                    cutoff_exposure(data, cutoff_date))
    logger.info(f"Loaded {len(data):,} PAs across years "
                f"{data['game_year'].min()}-{data['game_year'].max()}")
    return data


# Where the park factors live, in the order they are looked for. The committed
# artifact (BAS-86, `scripts/build_park_factors.py`, docs/park-factors.md) is
# the walk-forward one: home/away paired, a three-season window ending strictly
# before the stamped season, regressed toward 1 and centred on the league, with
# a column per component. The gitignored `data/parquet` file is what
# `scripts/generate_pa_parquet.py` writes — the same season's home rate over
# the league rate, which is neither paired nor walk-forward — and is kept only
# as a fallback so a checkout that has that file and not the artifact behaves
# as it did before.
PARK_FACTOR_PATHS = (FEATURES_DIR / "park_factors.parquet",
                     PARQUET_DIR / "park_factors.parquet")


def load_park_factors(pf_path: Path | str | None = None) -> pd.DataFrame:
    """Load park-factor parquet.

    Args:
        pf_path: Path to park_factors.parquet. `None` tries
            `PARK_FACTOR_PATHS` in order and takes the first that exists.

    Returns:
        DataFrame with columns `team`, `game_year` and one factor column per
        component (`k_park_factor`, `bb_park_factor`, `hr_park_factor`,
        `babip_park_factor`, `iso_park_factor` — the names
        `RateComponent.park_factor_col` carries). A component whose column is
        absent runs at a neutral offset; see `prepare_model_data`.

        `None` when no file is found, which is what leaves every offset at
        exactly zero — the path the K% bit-for-bit reference test fits under.
    """
    candidates = ([Path(pf_path)] if pf_path is not None
                  else list(PARK_FACTOR_PATHS))
    for path in candidates:
        if path.exists():
            pf = pd.read_parquet(path)
            logger.info(f"Loaded park factors from {path}: {len(pf)} rows, "
                        f"{pf['team'].nunique()} teams, components "
                        f"{[c for c in pf.columns if c.endswith('_park_factor')]}")
            return pf
    logger.warning("Park factors not found at %s; using neutral (1.0)",
                   ", ".join(str(p) for p in candidates))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_model_data(
    pa: pd.DataFrame,
    park_factors: pd.DataFrame | None = None,
    min_pa: int = MIN_PA_THRESHOLD,
    birthdates: pd.DataFrame | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
    include_pitcher: bool = False,
    component: str | RateComponent | None = None,
) -> dict:
    """Prepare PA data for the PyMC model.

    Steps:
        0. Apply the dated cutoff, if any, and assert nothing survives it.
        1. Filter pitchers (batters with < min_pa total PAs are dropped).
        2. Determine batting team from inning_topbot.
        3. Create integer indices for batters, seasons, teams (and pitchers).
        4. Compute age as of June 30 from Chadwick register birthdates.
        5. Merge park factors (default to 1.0 if unavailable).

    Args:
        pa: Raw PA DataFrame from load_pa_data().
        park_factors: Park factor DataFrame (optional).
        min_pa: Minimum career PAs to include a batter.
        birthdates: Chadwick register frame from src.data.birthdates.
            Loaded from the parquet cache when None; falls back to the
            legacy first_year - 23 estimate only if no register data is
            available at all.
        cutoff_date: ISO date. Only PA strictly before it are kept, so the
            current season enters as a partial one. Safe to pass even when
            `load_pa_data` already cut — the filter is idempotent and the
            guard runs either way.
        include_pitcher: put `pitcher_idx` in the cell key so the model can
            carry an opposing-pitcher random effect.
        component: which rate to count as the Binomial's successes. The cell
            structure is identical across components — only the numerator
            column and the derived `league_init_mu` change.

    **Partial-season semantics, against `src.eval.baselines.marcel`.**
    The two arms are handed the same plate appearances: the harness's
    training frame is the prior full seasons plus the current season through
    the cutoff (`intraseason.build_training_frame`), and `apply_cutoff` keeps
    exactly `game_date < cutoff` — the same strict inequality
    `intraseason.split_at_cutoff` uses, so a game played *on* the cutoff date
    is withheld from both. The partial season is not reweighted or annualized
    here any more than it is there: it is one more season in the random walk
    whose cells carry the PA actually played, which is the model's analogue of
    Marcel weighting by trials.

    Four differences remain, all deliberate:

    1. **Window and recency.** Marcel reads three seasons at fixed 5/4/3
       weights; this model reads every season in `pa_dir` with no recency
       weight on the player term. Recency enters only through the league
       random walk. Feeding the model the same three seasons Marcel sees is
       a matter of which parquets are in `pa_dir`.
    2. **Universe.** Marcel's prior seasons come from the Stats API season
       table and the current partial season from PA data; this model reads PA
       data throughout. The Statcast universe runs ~0.7% more PA per player,
       so the *prior* seasons differ slightly between the arms. The
       current-season slice is bit-identical.
    3. **Coverage.** `min_pa` drops batters below a career-PA floor, which
       Marcel does not do. The harness scores on the common player set, so
       this costs coverage rather than fairness; `generate_projections` can
       cover the remainder from the fitted population instead.
    4. **League level.** Marcel regresses toward a projected league rate; at
       an intra-season cutoff its horizon is zero, so that is the current
       season's own partial rate. The random walk's last node is the same
       partial season, and `generate_projections` extrapolates zero steps when
       the projection year is the cutoff year. Same convention, reached from
       opposite directions.

    Returns:
        Dictionary with all arrays/indices needed by the model.
    """
    comp = get_component(component)
    if comp.numerator not in pa.columns:
        raise KeyError(
            f"component {comp.name!r} needs a {comp.numerator!r} column; got "
            f"{sorted(pa.columns)} — load_pa_data(component={comp.name!r})"
        )
    if cutoff_date is not None:
        pa = apply_cutoff(pa, cutoff_date)
        assert_no_post_cutoff(pa, cutoff_date)
    df = pa.copy()

    # --- Batting team: if Top of inning, batter is away; Bottom → home ---
    df["bat_team"] = np.where(
        df["inning_topbot"] == "Top",
        df["away_team"],
        df["home_team"],
    )

    # --- Filter low-PA batters (likely pitchers or cup-of-coffee) ---
    pa_counts = df.groupby("batter").size()
    qualified = pa_counts[pa_counts >= min_pa].index
    n_before = df["batter"].nunique()
    df = df[df["batter"].isin(qualified)].copy()
    logger.info(f"Filtered batters: {n_before} → {df['batter'].nunique()} "
                f"(>= {min_pa} career PAs)")

    # --- Create integer indices ---
    # Seasons
    seasons = np.sort(df["game_year"].unique())
    season_map = {yr: i for i, yr in enumerate(seasons)}
    df["season_idx"] = df["game_year"].map(season_map).astype(np.int64)

    # Batters
    batters = np.sort(df["batter"].unique())
    batter_map = {b: i for i, b in enumerate(batters)}
    df["batter_idx"] = df["batter"].map(batter_map).astype(np.int64)

    # Teams (for park effects)
    teams = np.sort(df["bat_team"].unique())
    team_map = {t: i for i, t in enumerate(teams)}
    df["team_idx"] = df["bat_team"].map(team_map).astype(np.int64)

    # Handedness: L=0, R=1
    df["stand_idx"] = (df["stand"] == "R").astype(np.int64)

    # Opposing pitchers (for the pitcher random effect). Every pitcher who
    # threw a PA in the window gets a level; partial pooling shrinks the ones
    # with a handful of batters faced back to zero on its own, so there is no
    # minimum-batters-faced filter to tune.
    pitchers = np.array([], dtype=np.int64)
    pitcher_map: dict = {}
    if include_pitcher:
        if "pitcher" not in df.columns:
            raise KeyError("include_pitcher=True needs a 'pitcher' column; "
                           "load_pa_data(include_pitcher=True)")
        pitchers = np.sort(df["pitcher"].unique())
        pitcher_map = {p: i for i, p in enumerate(pitchers)}
        df["pitcher_idx"] = df["pitcher"].map(pitcher_map).astype(np.int64)

    # --- Age from real birthdates (Chadwick register), June 30 convention ---
    from src.data.birthdates import (
        BIRTHDATES_PARQUET, birth_year_map, load_birthdates, seasonal_age,
    )

    first_year = df.groupby("batter")["game_year"].min()
    if birthdates is None and BIRTHDATES_PARQUET.exists():
        birthdates = load_birthdates()

    if birthdates is not None:
        by = birth_year_map(birthdates, fallback_first_year=first_year)
        df["birth_year"] = df["batter"].map(by).astype(np.float64)
        df["age"] = seasonal_age(birthdates, df["batter"], df["game_year"])
        # Register misses fall back to the year-based estimate.
        est_age = (df["game_year"] - df["birth_year"]).astype(np.float64)
        df["age"] = np.where(np.isnan(df["age"]), est_age, df["age"])
    else:
        logger.warning(
            "No birthdates parquet found — falling back to first_year - 23 "
            "estimate. Run scripts/build_birthdates.py; ages are unreliable "
            "until you do."
        )
        birth_year = (first_year - 23).to_dict()
        df["birth_year"] = df["batter"].map(birth_year).astype(np.float64)
        df["age"] = (df["game_year"] - df["birth_year"]).astype(np.float64)
    df["age_centered"] = (df["age"] - REFERENCE_AGE).astype(np.float64)

    # --- Park factor lookup ---
    # Build a (team_idx, season_idx) → log(pf_k) array
    n_teams = len(teams)
    n_seasons = len(seasons)
    log_pf = np.zeros((n_teams, n_seasons), dtype=np.float64)

    # A park-factor table that has no column for this component leaves the
    # offset neutral rather than borrowing K%'s: a K% park factor applied to
    # HR/PA would be worse than none at all.
    if park_factors is not None and comp.park_factor_col in park_factors.columns:
        for _, row in park_factors.iterrows():
            t = team_map.get(row["team"])
            s = season_map.get(int(row["game_year"]))
            if t is not None and s is not None:
                log_pf[t, s] = np.log(float(row[comp.park_factor_col]))
    elif park_factors is not None:
        logger.warning("park factors have no %r column — %s runs at a "
                       "neutral park offset", comp.park_factor_col, comp.name)

    # Per-PA log park factor
    df["log_pf_k"] = log_pf[
        df["team_idx"].values, df["season_idx"].values
    ].astype(np.float64)

    # --- Batter-level metadata for projections ---
    batter_meta = (
        df.groupby("batter")
        .agg(
            stand=("stand", "first"),
            birth_year=("birth_year", "first"),
            last_season=("game_year", "max"),
            total_pa=(comp.numerator, "size"),
            **{comp.career_col: (comp.numerator, "mean")},
        )
        .reset_index()
    )

    # --- Binomial aggregation (roadmap 0.4) ---
    from src.models.aggregation import aggregate_binomial_cells

    cell_cols = ("batter_idx", "season_idx", "team_idx", "stand_idx")
    if include_pitcher:
        # The pitcher effect varies within the old cell, so it has to join the
        # key for the Binomial to stay an exact rewrite of the per-PA
        # Bernoulli. Compression drops to ~2 PA/cell — that is the cost of the
        # term, not a bug.
        cell_cols = cell_cols + ("pitcher_idx",)
    cells = aggregate_binomial_cells(df, cell_cols=cell_cols,
                                     outcome=comp.numerator)
    logger.info(f"Binomial aggregation: {len(df):,} PAs → {len(cells):,} cells "
                f"({len(df) / max(len(cells), 1):.1f}x compression)")

    league_init_mu = (comp.league_init_mu if comp.league_init_mu is not None
                      else league_logit_earliest_season(cells))

    model_data = {
        # Which rate this is, and the league prior it implies. Carried on the
        # data (not passed to `build_model` separately) so a trace, a
        # projection frame and the cells it came from can never disagree
        # about which component was fit.
        "component": comp.name,
        "league_init_mu": float(league_init_mu),
        # Dimensions
        "n_obs": len(cells),
        "n_pa": int(cells["n"].sum()),
        "n_batters": len(batters),
        "n_seasons": len(seasons),
        "n_teams": n_teams,
        "n_pitchers": len(pitchers),
        # Index arrays (int64)
        "batter_idx": cells["batter_idx"].values,
        "season_idx": cells["season_idx"].values,
        "team_idx": cells["team_idx"].values,
        "stand_idx": cells["stand_idx"].values,
        "pitcher_idx": (cells["pitcher_idx"].values if include_pitcher else None),
        # Continuous features (float64)
        "age_centered": cells["age_centered"].values,
        "log_pf_k": cells["log_pf_k"].values,
        # Outcome: successes and trials per cell
        "k": cells["k"].values,
        "n_trials": cells["n"].values,
        # Lookup tables
        "seasons": seasons,
        "batters": batters,
        "teams": teams,
        "pitchers": pitchers,
        "season_map": season_map,
        "batter_map": batter_map,
        "team_map": team_map,
        "pitcher_map": pitcher_map,
        # Metadata
        "batter_meta": batter_meta,
        "log_pf_matrix": log_pf,
        "df": cells,
        "include_pitcher": bool(include_pitcher),
        "cutoff_date": None if cutoff_date is None else str(pd.Timestamp(cutoff_date).date()),
        # Trials-weighted share of right-handed PA, used to marginalize the
        # handedness term for a batter whose stand we never saw.
        "stand_share_r": float(df["stand_idx"].mean()),
    }

    logger.info(
        f"Model data ready: {model_data['n_obs']:,} cells "
        f"({model_data['n_pa']:,} PAs), "
        f"{model_data['n_batters']:,} batters, "
        f"{model_data['n_seasons']} seasons, "
        f"{model_data['n_teams']} teams, "
        f"{model_data['n_pitchers']:,} pitchers"
    )
    logger.info("component %s: league_init prior mean %.3f (%s), "
                "pooled rate %.4f",
                comp.name, league_init_mu,
                "pinned" if comp.league_init_mu is not None
                else f"logit of the {seasons[0]} league rate",
                float(cells['k'].sum()) / max(float(cells['n'].sum()), 1.0))
    return model_data


# ═══════════════════════════════════════════════════════════════════════════════
# PyMC Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(data: dict, options: ModelOptions | None = None) -> pm.Model:
    """Build the hierarchical Bayesian K-rate model.

    Structure (all on logit scale):
        eta = league_trend[season]
            + player_ability[batter]                       (or [batter, season], ability_walk)
            + pitcher_ability[pitcher]        (when data carries pitchers)
            + handedness * stand_idx
            + park_effect[team]
            + age_term(age)                                (quadratic, or peak+slopes, constrained_age)
            + log_pf_k  (park factor offset)

        k ~ Binomial(n, logistic(eta))   per cell

    Non-centered parameterization is used for player abilities and for the
    pitcher effect, to improve sampling geometry.

    **The pitcher term.** `pitcher_ability = sigma_pitcher * z_pitcher` with
    `z ~ N(0,1)` and *no* free mean: a mean would be exactly confounded with
    `league_init`, which is the one thing that turns this from a random effect
    into a funnel with an unidentified ridge. `sigma_pitcher`'s HalfNormal
    scale comes from `PITCHER_SIGMA_PRIOR` (see the constant for how it was
    chosen). Everything about a batter is now read net of the arms he faced,
    and projections are made at `pitcher_ability = 0`.

    **`options.ability_walk`.** Replaces the single per-batter level with a
    non-centered Gaussian random walk over the model's seasons:
    `ability[b, 0] = mu_ability + sigma_ability * z_ability[b]`,
    `ability[b, s] = ability[b, s-1] + sigma_step * z_step[b, s]` for `s >= 1`.
    `sigma_step -> 0` reduces this exactly to the flat model (see
    `ABILITY_STEP_SIGMA_PRIOR` and `tests/test_models/test_pa_k_rate_options.py`
    for the graph-level check) — the walk is a strict generalization, not a
    different model, so a fit that finds no season-to-season drift collapses
    back to the flat answer rather than to something else.

    **`options.constrained_age`.** Replaces the quadratic
    `beta_age * age_c + beta_age2 * age_c^2` — which can fit a peak *or* a
    level shift and a backtest cannot always tell the two apart — with a peak
    age plus two signed slopes, the same family `src.eval.tuning` constrains
    tuned Marcel's age curve to: `slope_young`/`slope_old` are HalfNormal (a
    sign is asserted, not fit), so the curve can only turn over at
    `peak_age`, never run monotonically across the age range as a level
    correction. `peak_age` is a `pm.Deterministic` on `peak_frac ~ Beta(2,2)`
    scaled into `AGE_PEAK_WINDOW`, which keeps its prior on a window instead
    of an unconstrained real line while still letting the data place it
    anywhere in that window with a distribution that discourages the edges.

    **`options.covariates` (BAS-83, docs/bayes-covariates.md).** Adds
    `sum_j beta_cov_j * x[batter, season, j]` to the batter's own term, where
    `x` is `data["cov_x"]` — standardized layer-1 aggregates per (batter,
    season), built by `src.models.pa_covariates.attach_covariates` and summed
    strictly before the cutoff with the same month lag the served engine uses.
    Each coefficient is its own scalar RV named `beta_cov_<aggregate>` with a
    `Normal(0, 0.5)` prior. With `covariates=None` — the default — not one
    line of this block runs, no `cov_x` container is created, and the graph is
    the model above unchanged; `tests/test_models/test_pa_rate.py` pins that
    bit-for-bit.

    Args:
        data: Dictionary from prepare_model_data().
        options: Structural variants (default: neither — the original model).

    Returns:
        PyMC Model object (not yet sampled).
    """
    options = options or ModelOptions()
    from src.models.pa_covariates import BETA_COV_SIGMA

    cov_names = options.covariate_names()
    if cov_names:
        if data.get("cov_x") is None:
            raise KeyError(
                "options.covariates asks for "
                f"{list(cov_names)} but this model data carries no 'cov_x' — "
                "call src.models.pa_covariates.attach_covariates(data, ...) "
                "after prepare_model_data")
        if tuple(data.get("cov_names", ())) != tuple(cov_names):
            raise ValueError(
                f"model data carries covariates {list(data.get('cov_names', ()))} "
                f"but the options ask for {list(cov_names)} — the design "
                "matrix and the coefficients would name different things")
        expected = (data["n_batters"], data["n_seasons"], len(cov_names))
        if np.shape(data["cov_x"]) != expected:
            raise ValueError(
                f"cov_x has shape {np.shape(data['cov_x'])}, expected {expected} "
                "(batter, season, covariate)")
    comp = get_component(data.get("component"))
    league_init_mu = float(data.get("league_init_mu", comp.league_init_mu
                                    if comp.league_init_mu is not None else -1.27))
    include_pitcher = bool(data.get("include_pitcher")) and data.get("n_pitchers")
    n_seasons = data["n_seasons"]
    coords = {
        "batter": data["batters"],
        "season": data["seasons"],
        "team": data["teams"],
        "cell": np.arange(data["n_obs"]),
    }
    if include_pitcher:
        coords["pitcher"] = data["pitchers"]
    if cov_names:
        coords["covariate"] = list(cov_names)
    if options.ability_walk:
        # n_seasons - 1 step innovations: one per transition between
        # consecutive seasons, not one per season. Coordinate is the season
        # each step *arrives at*, so it lines up with `data["seasons"][1:]`.
        coords["season_step"] = data["seasons"][1:]

    with pm.Model(coords=coords) as model:
        # ─── Mutable data containers (for posterior predictive) ───────────
        batter_idx = pm.Data("batter_idx", data["batter_idx"], dims="cell")
        season_idx = pm.Data("season_idx", data["season_idx"], dims="cell")
        team_idx = pm.Data("team_idx", data["team_idx"], dims="cell")
        stand_idx = pm.Data("stand_idx", data["stand_idx"], dims="cell")
        age_c = pm.Data("age_centered", data["age_centered"], dims="cell")
        log_pf = pm.Data("log_pf_k", data["log_pf_k"], dims="cell")
        n_trials = pm.Data("n_trials", data["n_trials"], dims="cell")

        # ─── League trend: random walk on logit scale ─────────────────────
        # Initial intercept ~ the earliest training season's league rate on
        # the logit scale: -1.27 for K% (pinned, ~22%), about -2.38 for BB%
        # (~8.5%) and -3.44 for HR/PA (~3.1%), read off the data by
        # `prepare_model_data`. See `RateComponent.league_init_mu`.
        league_init = pm.Normal("league_init", mu=league_init_mu, sigma=0.3)
        league_innovations = pm.Normal(
            "league_innovations",
            mu=0,
            sigma=0.05,
            dims="season",
        )
        # Cumulative sum to build the random walk
        league_trend = pm.Deterministic(
            "league_trend",
            league_init + pt.cumsum(league_innovations),
            dims="season",
        )

        # ─── Player ability: partial pooling, non-centered ────────────────
        # Hyperpriors for the batter population
        mu_ability = pm.Normal("mu_ability", mu=0.0, sigma=0.3)
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=0.4)

        # Non-centered parameterization: z ~ N(0,1), ability = mu + sigma * z
        z_ability = pm.Normal("z_ability", mu=0, sigma=1, dims="batter")

        if options.ability_walk:
            # Season-0 level is exactly the flat model's ability. Every later
            # season adds a non-centered innovation; `pt.cumsum` builds the
            # whole walk in one vectorized op instead of a scan, which is
            # what makes this affordable at ~800 batters x 1500+2000 draws.
            ability0 = mu_ability + sigma_ability * z_ability            # (batter,)
            sigma_step = pm.HalfNormal("sigma_step", sigma=ABILITY_STEP_SIGMA_PRIOR)
            z_step = pm.Normal("z_step", mu=0, sigma=1,
                               dims=("batter", "season_step"))
            steps = pt.cumsum(sigma_step * z_step, axis=1)               # (batter, n_seasons-1)
            walk = pt.concatenate(
                [ability0[:, None], ability0[:, None] + steps], axis=1
            )                                                            # (batter, n_seasons)
            # At sigma_step == 0, `steps` is identically zero and `walk` is
            # `ability0` broadcast across every season column — the flat
            # model, exactly, not approximately (tests/test_models/
            # test_pa_k_rate_options.py checks this at the graph level).
            player_ability = pm.Deterministic(
                "player_ability", walk, dims=("batter", "season")
            )
        else:
            player_ability = pm.Deterministic(
                "player_ability",
                mu_ability + sigma_ability * z_ability,
                dims="batter",
            )

        # ─── Opposing pitcher: partial pooling, non-centered, zero mean ───
        if include_pitcher:
            pitcher_idx = pm.Data("pitcher_idx", data["pitcher_idx"], dims="cell")
            sigma_pitcher = pm.HalfNormal("sigma_pitcher", sigma=PITCHER_SIGMA_PRIOR)
            z_pitcher = pm.Normal("z_pitcher", mu=0, sigma=1, dims="pitcher")
            pitcher_ability = pm.Deterministic(
                "pitcher_ability",
                sigma_pitcher * z_pitcher,
                dims="pitcher",
            )

        # ─── Handedness effect ────────────────────────────────────────────
        # R vs L batter (R=1, L=0); positive = R batters strike out more
        beta_hand = pm.Normal("beta_hand", mu=0.0, sigma=0.2)

        # ─── Park effects: zero-sum constraint ───────────────────────────
        park_effect = pm.ZeroSumNormal(
            "park_effect",
            sigma=0.05,
            dims="team",
        )

        # ─── Age curve ──────────────────────────────────────────────────
        if options.constrained_age:
            # Peak-plus-signed-slopes: the multiplier (here, additive term on
            # the logit scale) can only turn over at `peak_age`, never run
            # monotonically across the age range, because the slopes are
            # HalfNormal (sign asserted, magnitude fit).
            #
            # `curve_sign` is what makes this work for more than one
            # component, and it is `-age_direction` rather than
            # `+age_direction` because the bracket it multiplies is written
            # as a *valley*: `-slope_young * d` rises as age falls below the
            # peak, `slope_old * d` rises as age climbs past it. K% has
            # `age_direction = -1` (a bigger K% is a worse hitter), so its
            # sign is +1 and the curve stays the valley the K% sweep fit —
            # falling toward the peak, rising after. BB% and HR/PA have
            # `age_direction = +1` and flip it to a hill: they climb toward
            # the peak and decline after, which is what an aging curve for a
            # skill you *want* more of looks like. Same family and the same
            # sign convention `src.eval.tuning.constrained_slope_grid`
            # enforces for tuned Marcel, on the logit scale.
            peak_frac = pm.Beta("peak_frac", alpha=2.0, beta=2.0)
            lo, hi = AGE_PEAK_WINDOW
            peak_age = pm.Deterministic("peak_age", lo + (hi - lo) * peak_frac)
            slope_young = pm.HalfNormal("slope_young", sigma=0.02)
            slope_old = pm.HalfNormal("slope_old", sigma=0.02)
            age = age_c + REFERENCE_AGE
            d = age - peak_age
            curve_sign = -comp.age_direction
            age_term = curve_sign * pt.where(d > 0, slope_old * d,
                                             -slope_young * d)
        else:
            # Quadratic on centered age. Only the curvature's prior *mean*
            # carries the component's direction — the sign is still fit, not
            # asserted, which is the whole difference between this branch and
            # the constrained one. +0.005 for K% (a U: strikeouts rise away
            # from the peak in both directions), -0.005 for BB%/HR/PA (an
            # inverted U). The prior sd of 0.01 is twice the mean either way,
            # so the data can and do overrule it.
            beta_age = pm.Normal("beta_age", mu=0.0, sigma=0.02)
            beta_age2 = pm.Normal("beta_age2",
                                  mu=-0.005 * comp.age_direction, sigma=0.01)
            age_term = beta_age * age_c + beta_age2 * (age_c ** 2)

        # ─── Layer-1 covariates (BAS-83) ──────────────────────────────────
        # One scalar coefficient per aggregate, named `beta_cov_<aggregate>`
        # rather than a single vector RV: the vacuity check in
        # docs/bayes-covariates.md is about *one* aggregate's posterior at a
        # time (barrel rate on HR/PA, average EV on K%), and a named scalar
        # is what `variant_param_summary` in the dense sweep can pick up
        # without knowing the vector's coordinate order.
        cov_term = None
        if cov_names:
            cov_x = pm.Data("cov_x", np.asarray(data["cov_x"], dtype="float64"),
                            dims=("batter", "season", "covariate"))
            betas = [
                pm.Normal(f"beta_cov_{name}", mu=0.0, sigma=BETA_COV_SIGMA)
                for name in cov_names
            ]
            beta_vec = pt.stack(betas)                       # (covariate,)
            # (batter, season): the covariate contribution to every batter's
            # logit rate in every season, indexed by the cell's own
            # (batter, season) exactly as the ability walk is.
            cov_effect = pm.Deterministic(
                "cov_effect", (cov_x * beta_vec).sum(axis=-1),
                dims=("batter", "season"),
            )
            cov_term = cov_effect[batter_idx, season_idx]

        # ─── Linear predictor ────────────────────────────────────────────
        ability_term = (
            player_ability[batter_idx, season_idx] if options.ability_walk
            else player_ability[batter_idx]
        )
        if cov_term is not None:
            ability_term = ability_term + cov_term
        eta = (
            league_trend[season_idx]
            + ability_term
            + beta_hand * stand_idx
            + park_effect[team_idx]
            + age_term
            + log_pf
        )
        if include_pitcher:
            eta = eta + pitcher_ability[pitcher_idx]

        # ─── Likelihood ──────────────────────────────────────────────────
        # Binomial over cells; no p_k Deterministic — writing a float per
        # cell per draw into the trace was most of the old memory bill.
        pm.Binomial(
            comp.obs_name,
            n=n_trials,
            p=pm.math.invlogit(eta),
            observed=data["k"],
            dims="cell",
        )

    if options.ability_walk:
        ability_params = (
            data["n_batters"]                    # z_ability (season 0)
            + data["n_batters"] * max(n_seasons - 1, 0)  # z_step
            + 1                                   # sigma_step
        )
    else:
        ability_params = data["n_batters"]        # z_ability

    age_params = 3 if options.constrained_age else 2  # peak_frac/slopes, or beta_age/beta_age2

    n_params = (
        1                        # league_init
        + data["n_seasons"]      # league_innovations
        + 1 + 1                  # mu_ability, sigma_ability
        + ability_params
        + 1                      # beta_hand
        + data["n_teams"] - 1    # park_effect (zero-sum = n-1 free)
        + age_params
        + len(cov_names)         # beta_cov_<aggregate>, one scalar each
        + (data["n_pitchers"] + 1 if include_pitcher else 0)  # z_pitcher, sigma
    )
    logger.info(f"Model built [{comp.name}]: ~{n_params:,} free parameters, "
                f"{data['n_obs']:,} cells ({data.get('n_pa', 0):,} PAs)"
                f"{', + pitcher effect' if include_pitcher else ''}"
                f"{', ability_walk' if options.ability_walk else ''}"
                f"{', constrained_age' if options.constrained_age else ''}"
                f"{', covariates=' + '+'.join(cov_names) if cov_names else ''}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_model(
    model: pm.Model,
    **sampler_overrides,
) -> az.InferenceData:
    """Sample the model using NumpyRo NUTS.

    Args:
        model: PyMC Model from build_model().
        **sampler_overrides: Override any default SAMPLER_KWARGS.

    Returns:
        ArviZ InferenceData with posterior samples.
    """
    kwargs = {**SAMPLER_KWARGS, **sampler_overrides}
    logger.info(f"Starting sampling: {kwargs['chains']} chains × "
                f"{kwargs['draws']} draws (tune={kwargs['tune']})")

    t0 = time.time()
    with model:
        trace = pm.sample(**kwargs)
    elapsed = time.time() - t0

    logger.info(f"Sampling complete in {elapsed:.0f}s")

    diag = model_diagnostics(trace)
    logger.info(
        "Max R-hat: %.4f (%s), min ESS bulk: %.0f (%s), divergences: %d, "
        "BFMI: %s",
        diag["max_rhat"], diag["max_rhat_var"], diag["min_ess_bulk"],
        diag["min_ess_var"], diag["divergences"],
        ", ".join(f"{b:.2f}" for b in diag["bfmi"]),
    )

    return trace


def model_diagnostics(trace: az.InferenceData) -> dict:
    """R-hat, ESS, divergences and BFMI in one dict. No sampling, no plots.

    Split out so a validation run and a CI test can assert on the same
    numbers, and so the energy diagnostic (BFMI) is reported alongside R-hat
    rather than only when someone remembers to look — a new group-level scale
    like `sigma_pitcher` is exactly the thing that shows up in the energy
    plot before it shows up in R-hat.
    """
    rhat = az.rhat(trace)
    ess = az.ess(trace)

    def _extreme(ds, worst_is_max: bool):
        """(value, variable) for the worst finite entry across every var."""
        best, name = None, None
        for v in ds.data_vars:
            vals = np.asarray(ds[v].values, dtype="float64").ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            cand = float(vals.max() if worst_is_max else vals.min())
            if best is None or (cand > best if worst_is_max else cand < best):
                best, name = cand, str(v)
        return best, name

    max_rhat, max_rhat_var = _extreme(rhat, worst_is_max=True)
    min_ess, min_ess_var = _extreme(ess, worst_is_max=False)

    divergences = 0
    bfmi: list[float] = []
    stats = getattr(trace, "sample_stats", None)
    if stats is not None:
        div = stats.get("diverging")
        if div is not None:
            divergences = int(np.asarray(div.values).sum())
        if "energy" in stats:
            bfmi = [float(b) for b in np.atleast_1d(az.bfmi(trace))]

    return {
        "max_rhat": float(max_rhat) if max_rhat is not None else float("nan"),
        "max_rhat_var": max_rhat_var,
        "min_ess_bulk": float(min_ess) if min_ess is not None else float("nan"),
        "min_ess_var": min_ess_var,
        "divergences": divergences,
        "bfmi": bfmi,
        # BFMI below ~0.3 is the usual "this posterior has a funnel" alarm.
        "bfmi_ok": bool(bfmi) and min(bfmi) >= 0.3,
        "healthy": (
            (max_rhat is not None and float(max_rhat) < 1.01)
            and divergences == 0
            and (not bfmi or min(bfmi) >= 0.3)
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Projections
# ═══════════════════════════════════════════════════════════════════════════════

def _age_term_function(post, n_samples: int, age_direction: float = -1.0):
    """The posterior age curve as (raw age -> array of shape (n_samples,)).

    Detected from the trace's own variable names rather than a parameter a
    caller has to thread through: `generate_projections` and
    `bayes_arm.BayesFit.project` are handed a trace and nothing else, so the
    curve the fit actually used has to be self-describing. `"peak_age"` only
    exists in the trace when `build_model` was called with
    `options.constrained_age=True`.

    `age_direction` is the one thing the trace cannot describe: the slopes
    are stored as the positive HalfNormals they are, and the sign that turns
    the valley into a hill lives in `build_model`'s `curve_sign`. It has to
    be reapplied here identically or a BB% projection would age like a K%
    one, so the caller reads it off the component (`generate_projections`
    does, from `data["component"]`) rather than guessing. The default is K%'s
    `-1.0`, which keeps every pre-BAS-73 call site meaning what it did.
    """
    curve_sign = -float(age_direction)
    if "peak_age" in post:
        peak_age = post["peak_age"].values.reshape(n_samples)
        slope_young = post["slope_young"].values.reshape(n_samples)
        slope_old = post["slope_old"].values.reshape(n_samples)

        def age_term(age: float) -> np.ndarray:
            d = age - peak_age
            return curve_sign * np.where(d > 0, slope_old * d,
                                         -slope_young * d)

        return age_term

    beta_age = post["beta_age"].values.reshape(n_samples)
    beta_age2 = post["beta_age2"].values.reshape(n_samples)

    def age_term(age: float) -> np.ndarray:
        age_c = age - REFERENCE_AGE
        return beta_age * age_c + beta_age2 * (age_c ** 2)

    return age_term


def _project_unseen(
    unseen: pd.DataFrame | None,
    data: dict,
    post,
    projected_trend: np.ndarray,
    beta_hand_flat: np.ndarray,
    age_term_fn,
    projection_year: int,
    n_samples: int,
    already: set[int],
    component: RateComponent | None = None,
) -> list[dict]:
    """Population-level projections for batters the fit never saw.

    A hierarchical model already says what to do about a player with no
    history: his ability is a draw from the fitted population,
    `N(mu_ability, sigma_ability)`. That is the same construction as drawing a
    fresh random effect for an unseen group, done here in numpy on the
    posterior rather than by extending the model, because nothing else about
    the prediction needs the graph. It is also, unchanged, the right answer
    under `ability_walk`: `N(mu_ability, sigma_ability)` is that walk's own
    season-0 marginal (`ability[b, 0] = mu_ability + sigma_ability * z[b]`),
    so an unseen batter gets a draw from the same distribution a seen batter's
    season-0 level came from — there is no later season to walk forward from,
    since there was never a first draw for this batter at all.

    Under `options.covariates` an unseen batter carries no covariate term at
    all, which is the same thing a seen batter with no pre-cutoff batted ball
    carries: `x = 0`, the league mean after standardization
    (`src.models.pa_covariates`, and the module docstring there for why no
    `has_cov` indicator joins it).

    `unseen` is [batter, age] with an optional `stand`. Without a stand the
    handedness term is marginalized at the training set's right-handed PA
    share — the honest answer when we have not seen the player bat, and the
    only place a projection here is not conditioned on a known covariate.
    """
    if unseen is None or len(unseen) == 0:
        return []
    if "batter" not in unseen.columns:
        raise KeyError("unseen frame needs a 'batter' column")

    comp = get_component(component)
    cols = comp.out_columns()
    mu_ability = post["mu_ability"].values.reshape(n_samples)
    sigma_ability = post["sigma_ability"].values.reshape(n_samples)
    rng = np.random.default_rng(2026)
    share_r = float(data.get("stand_share_r", 0.5))

    rows = []
    for _, row in unseen.iterrows():
        batter_id = int(row["batter"])
        if batter_id in already:
            continue
        already.add(batter_id)

        age = row.get("age", np.nan)
        age = float(age) if age is not None and np.isfinite(float(age)) else np.nan
        # Unknown age: no adjustment away from the population baseline,
        # regardless of which age curve the fit used.
        age_term = np.zeros(n_samples) if np.isnan(age) else age_term_fn(age)

        stand = row.get("stand") if "stand" in unseen.columns else None
        s_term = share_r if stand not in ("L", "R") else (1.0 if stand == "R" else 0.0)

        ability = mu_ability + sigma_ability * rng.standard_normal(n_samples)
        eta = (
            projected_trend
            + ability
            + beta_hand_flat * s_term
            + age_term
        )
        p = 1.0 / (1.0 + np.exp(-eta))
        rows.append({
            "batter": batter_id,
            "stand": stand if stand in ("L", "R") else None,
            "age": age,
            cols["mean"]: float(np.mean(p)),
            cols["std"]: float(np.std(p)),
            cols["lower"]: float(np.percentile(p, 5)),
            cols["upper"]: float(np.percentile(p, 95)),
            "posterior_mean_ability": float(np.mean(ability)),
            "total_pa": 0,
            comp.career_col: float("nan"),
            "last_season": int(projection_year),
            "unseen": True,
        })
    logger.info("Projected %d batters from the fitted population (unseen)", len(rows))
    return rows


def generate_projections(
    trace: az.InferenceData,
    data: dict,
    projection_year: int = PROJECTION_YEAR,
    recent_seasons: int = 3,
    unseen: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate rate projections from posterior samples.

    Column names follow the component the fit was prepared for: `k_rate`
    yields `projected_k_rate`/`k_rate_std`/`k_rate_lower`/`k_rate_upper`/
    `career_k_rate`, `bb_rate` the same five with `bb_rate` in place of
    `k_rate`, and so on. The component is read off `data["component"]`, so a
    caller that hands over the wrong `data` gets wrong *names*, loudly, rather
    than a BB% number labelled as a K% one.


    For each batter who appeared in at least one of the last `recent_seasons`,
    compute the posterior predictive K-rate at the projection year by:
        - Extrapolating the league trend (last value + one innovation draw)
        - Using the player's posterior ability
        - Applying the age curve at their projected age
        - Using neutral park/handedness (or their actual stand), and a neutral
          (average) opposing pitcher when the fit carries a pitcher effect

    At an intra-season cutoff the projection year *is* the last training
    season, so `years_ahead` is zero and no innovation is drawn: the level is
    the partial season's own fitted league node. That matches
    `marcel_tuned`'s "drift collapses to last at horizon zero".

    Args:
        trace: Posterior trace from sample_model().
        data: Model data dictionary.
        projection_year: Year to project (default 2026).
        recent_seasons: Include batters active within this many years.
        unseen: optional [batter, age, (stand)] frame of batters with no
            training PA — a September call-up at an April cutoff, or anyone
            below `min_pa`. Their ability is drawn from the fitted population,
            `N(mu_ability, sigma_ability)` per posterior sample, which is the
            hierarchical model's own answer for a player it has never seen
            rather than a hard-coded league rate. Rows carry `unseen=True`.

    Returns:
        DataFrame with columns: batter, stand, age, projected_k_rate,
        k_rate_lower, k_rate_upper, posterior_mean_ability, total_pa, unseen.
    """
    post = trace.posterior
    comp = get_component(data.get("component"))
    out_cols = comp.out_columns()

    # Extract posterior arrays (chains × draws × ...)
    league_trend = post["league_trend"].values         # (chains, draws, n_seasons)
    player_ability = post["player_ability"].values     # (chains, draws, n_batters[, n_seasons])
    beta_hand = post["beta_hand"].values                # (chains, draws)
    league_innovations = post["league_innovations"].values  # (chains, draws, n_seasons)

    # Flatten chains × draws → samples
    n_chains, n_draws = league_trend.shape[:2]
    n_samples = n_chains * n_draws
    league_trend_flat = league_trend.reshape(n_samples, -1)
    beta_hand_flat = beta_hand.reshape(n_samples)
    innovations_flat = league_innovations.reshape(n_samples, -1)
    age_term_fn = _age_term_function(post, n_samples,
                                     age_direction=comp.age_direction)

    # Extrapolate league trend: last season value + draw from innovation dist
    # Use the empirical std of innovations for the extrapolation step
    last_trend = league_trend_flat[:, -1]                # (n_samples,)
    innov_std = innovations_flat.std(axis=1)             # per-sample innovation scale
    rng = np.random.default_rng(42)
    # Number of years to extrapolate
    years_ahead = projection_year - int(data["seasons"][-1])
    projected_trend = last_trend.copy()
    for _ in range(years_ahead):
        projected_trend += rng.normal(0, innov_std)

    # `player_ability` is (batter,) under the flat model and (batter, season)
    # under `ability_walk`. Reduce both to one (n_samples, n_batters) array
    # up front so every batter-indexed line below reads identically either
    # way: the flat case is just a reshape, and the walk case reads its last
    # fitted season and extrapolates forward exactly like the league trend
    # above — a fresh `sigma_step` innovation per posterior sample, per year
    # ahead. At an intra-season cutoff `years_ahead` is 0 and this is a
    # no-op, so the loop below never runs and the level is the last node's,
    # matching the league trend's own horizon-zero convention.
    if "season" in post["player_ability"].dims:
        ability_walk = player_ability.reshape(n_samples, data["n_batters"], -1)
        player_ability_flat = ability_walk[:, :, -1].copy()   # last fitted season
        sigma_step_flat = post["sigma_step"].values.reshape(n_samples)
        step_rng = np.random.default_rng(43)
        for _ in range(years_ahead):
            player_ability_flat += step_rng.normal(
                0.0, sigma_step_flat[:, None],
                size=(n_samples, data["n_batters"]),
            )
    else:
        player_ability_flat = player_ability.reshape(n_samples, -1)

    # ─── Layer-1 covariates (BAS-83) ─────────────────────────────────────
    # `cov_effect` is (batter, season) per draw; a projection reads the LAST
    # fitted season, which at an intra-season cutoff is the partial season
    # the covariates were summed up to — the same horizon-zero convention the
    # league trend and the ability walk use above. There is no forward model
    # for a batter's future contact quality, so a projection more than zero
    # years ahead carries his last observed season's covariates unchanged;
    # every cell this ticket scores is horizon zero, where that is exact.
    cov_effect_flat = None
    if "cov_effect" in post:
        cov_effect = post["cov_effect"].values
        cov_effect_flat = cov_effect.reshape(n_samples, data["n_batters"], -1)[:, :, -1]

    # Filter to recently active batters
    meta = data["batter_meta"]
    cutoff_year = int(data["seasons"][-1]) - recent_seasons + 1
    active = meta[meta["last_season"] >= cutoff_year].copy()
    logger.info(f"Projecting {len(active)} batters active since {cutoff_year}")

    results = []
    for _, row in active.iterrows():
        batter_id = int(row["batter"])
        b_idx = data["batter_map"][batter_id]

        # Projected age
        proj_age = projection_year - float(row["birth_year"])

        # Stand index
        s_idx = 1 if row["stand"] == "R" else 0

        # Compute eta for each posterior sample (neutral park)
        eta = (
            projected_trend
            + player_ability_flat[:, b_idx]
            + beta_hand_flat * s_idx
            + age_term_fn(proj_age)
            # No park effect (neutral venue) and no log_pf, and a neutral
            # opposing pitcher: pitcher_ability is zero-mean by construction,
            # so leaving it out *is* the average-arm projection.
        )
        if cov_effect_flat is not None:
            eta = eta + cov_effect_flat[:, b_idx]

        # Convert to probability
        p = 1.0 / (1.0 + np.exp(-eta))

        results.append({
            "batter": batter_id,
            "stand": row["stand"],
            "age": proj_age,
            out_cols["mean"]: float(np.mean(p)),
            out_cols["std"]: float(np.std(p)),
            out_cols["lower"]: float(np.percentile(p, 5)),
            out_cols["upper"]: float(np.percentile(p, 95)),
            "posterior_mean_ability": float(np.mean(player_ability_flat[:, b_idx])),
            "total_pa": int(row["total_pa"]),
            comp.career_col: float(row[comp.career_col]),
            "last_season": int(row["last_season"]),
            "unseen": False,
        })

    results.extend(_project_unseen(
        unseen, data, post, projected_trend,
        beta_hand_flat, age_term_fn,
        projection_year, n_samples,
        already={int(r["batter"]) for r in results},
        component=comp,
    ))

    proj_df = pd.DataFrame(results)
    mean_col = out_cols["mean"]
    proj_df = proj_df.sort_values(mean_col, ascending=True).reset_index(drop=True)

    logger.info(
        f"Projections generated [{comp.name}]: median = "
        f"{proj_df[mean_col].median():.3f}, "
        f"range [{proj_df[mean_col].min():.3f}, "
        f"{proj_df[mean_col].max():.3f}]"
    )
    return proj_df


# ═══════════════════════════════════════════════════════════════════════════════
# wandb Logging
# ═══════════════════════════════════════════════════════════════════════════════

def log_to_wandb(
    trace: az.InferenceData,
    projections: pd.DataFrame,
    data: dict,
    model_config: dict,
    offline: bool = False,
    component: str | RateComponent | None = None,
) -> None:
    """Log model diagnostics, projections, and artifacts to wandb.

    Uses the WandbTracker pattern from src.tracking.wandb_tracker.

    Args:
        trace: Posterior InferenceData.
        projections: Projection DataFrame.
        data: Model data dictionary.
        model_config: Config dict for the run.
        offline: If True, log locally for later sync.
    """
    from src.tracking.wandb_tracker import WandbTracker

    comp = get_component(component or data.get("component"))
    slug = comp.name.replace("_", "-")
    tracker = WandbTracker(
        run_name=f"{slug}-bayesian-{PROJECTION_YEAR}",
        model_type="hitter",
        config=model_config,
        tags=[comp.name, "bayesian", "pa-level"],
        notes=f"Binomial-cell {comp.name} model projecting to {PROJECTION_YEAR}",
        offline=offline,
    )

    try:
        # MCMC diagnostics
        diagnostics = tracker.log_mcmc_diagnostics(trace)
        logger.info(f"wandb: logged diagnostics (healthy={diagnostics['healthy']})")

        # Dataset stats
        tracker.log_dataset_stats(
            data["df"][["k", "n", "age", "stand_idx"]],
            name="training_data",
        )

        # Posterior plots for key scalar parameters
        scalar_params = [
            "league_init", "mu_ability", "sigma_ability",
            "beta_hand", "beta_age", "beta_age2",
        ]
        tracker.log_posterior_plots(trace, params=scalar_params, prefix="posterior")

        # Save projections artifact
        tracker.save_projections_artifact(
            projections,
            artifact_name=f"{slug}-projections-{PROJECTION_YEAR}",
            metadata={
                "projection_year": PROJECTION_YEAR,
                "n_batters": len(projections),
                f"median_{comp.name}": float(
                    projections[comp.projected_col].median()),
            },
        )

        # Save model trace artifact
        tracker.save_model_artifact(
            trace,
            metadata={
                "projection_year": PROJECTION_YEAR,
                "n_obs": data["n_obs"],
                "n_batters": data["n_batters"],
                "n_seasons": data["n_seasons"],
                **diagnostics,
            },
            artifact_name=f"{slug}-trace-{PROJECTION_YEAR}",
            aliases=["latest"],
        )

        logger.info(f"wandb run URL: {tracker.url}")
    finally:
        tracker.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_model(
    pa_dir: Path | str | None = None,
    pf_path: Path | str | None = None,
    min_pa: int = MIN_PA_THRESHOLD,
    log_wandb: bool = True,
    wandb_offline: bool = False,
    projection_year: int = PROJECTION_YEAR,
    cutoff_date: str | pd.Timestamp | None = None,
    include_pitcher: bool = False,
    unseen: pd.DataFrame | None = None,
    component: str | RateComponent | None = None,
    options: ModelOptions | None = None,
    **sampler_overrides,
) -> tuple[az.InferenceData, pd.DataFrame, dict]:
    """End-to-end model pipeline: load → prep → build → sample → project → log.

    Args:
        pa_dir: PA outcomes parquet directory.
        pf_path: Park factors parquet path.
        min_pa: Minimum PAs to include a batter.
        log_wandb: Whether to log to wandb.
        wandb_offline: If True, wandb logs locally.
        projection_year: Year to project.
        cutoff_date: ISO date; train on PA strictly before it (see
            `prepare_model_data` for the partial-season semantics).
        include_pitcher: fit the opposing-pitcher random effect.
        unseen: [batter, age, (stand)] frame of batters to project from the
            fitted population because they have no training PA.
        component: which rate to fit — "k_rate" (default), "bb_rate" or
            "hr_rate".
        options: structural variants (`ability_walk`, `constrained_age`).
        **sampler_overrides: Override default sampler kwargs.

    Returns:
        Tuple of (trace, projections_df, model_data_dict).
    """
    comp = get_component(component)
    logger.info("=" * 60)
    logger.info("PA-level Bayesian rate model: %s", comp.name)
    logger.info("=" * 60)

    # 1. Load data
    pa = load_pa_data(pa_dir, cutoff_date=cutoff_date,
                      include_pitcher=include_pitcher, component=comp)
    park_factors = load_park_factors(pf_path)

    # 2. Prepare model data
    model_data = prepare_model_data(
        pa, park_factors, min_pa=min_pa,
        cutoff_date=cutoff_date, include_pitcher=include_pitcher,
        component=comp,
    )
    del pa  # free raw data
    gc.collect()

    # 3. Build model
    model = build_model(model_data, options)

    # 4. Sample
    trace = sample_model(model, **sampler_overrides)

    # 5. Generate projections
    projections = generate_projections(
        trace, model_data, projection_year=projection_year, unseen=unseen,
    )

    # 6. Save projections to parquet
    output_dir = DATA_DIR / "projections"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{comp.name}_projections_{projection_year}.parquet"
    projections.to_parquet(output_path, index=False)
    logger.info(f"Projections saved to {output_path}")

    # 7. wandb logging
    model_config = {
        "model": f"pa_rate_binomial:{comp.name}",
        "component": comp.name,
        "league_init_mu": model_data["league_init_mu"],
        "min_pa": min_pa,
        "projection_year": projection_year,
        "n_obs": model_data["n_obs"],
        "n_batters": model_data["n_batters"],
        "n_seasons": model_data["n_seasons"],
        "n_teams": model_data["n_teams"],
        "n_pitchers": model_data["n_pitchers"],
        "include_pitcher": include_pitcher,
        "cutoff_date": model_data["cutoff_date"],
        "reference_age": REFERENCE_AGE,
        **{k: v for k, v in SAMPLER_KWARGS.items() if k != "idata_kwargs"},
        **sampler_overrides,
    }

    if log_wandb:
        try:
            log_to_wandb(
                trace, projections, model_data, model_config,
                offline=wandb_offline,
            )
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return trace, projections, model_data


# ═══════════════════════════════════════════════════════════════════════════════
# Local Testing
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """Run the model locally with reduced sampling for testing.

    Set environment variables to override defaults:
        PA_DIR:    Path to PA outcomes parquet directory
        PF_PATH:   Path to park_factors.parquet
        MIN_PA:    Minimum PA threshold (default 50)
        COMPONENT: k_rate (default), bb_rate or hr_rate
        FAST:      If '1', use minimal sampling for quick test
        NO_WANDB:  If '1', skip wandb logging
    """
    fast_mode = os.environ.get("FAST", "0") == "1"
    no_wandb = os.environ.get("NO_WANDB", "0") == "1"
    comp = get_component(os.environ.get("COMPONENT", DEFAULT_COMPONENT))

    overrides = {}
    if fast_mode:
        logger.info("FAST MODE: minimal sampling for testing")
        overrides = dict(
            draws=100,
            tune=100,
            chains=2,
            cores=1,
        )

    trace, projections, model_data = run_model(
        pa_dir=os.environ.get("PA_DIR"),
        pf_path=os.environ.get("PF_PATH"),
        min_pa=int(os.environ.get("MIN_PA", MIN_PA_THRESHOLD)),
        log_wandb=not no_wandb,
        wandb_offline=True,  # default to offline for local runs
        component=comp,
        **overrides,
    )

    col = comp.projected_col
    print("\n" + "=" * 60)
    print(f"{comp.name} projections for {PROJECTION_YEAR}")
    print("=" * 60)
    print(f"\nLowest projected {comp.name}:")
    print(projections.head(15).to_string(index=False))
    print(f"\nHighest projected {comp.name}:")
    print(projections.tail(15).to_string(index=False))
    print(f"\nMedian: {projections[col].median():.1%}")
    print(f"Mean:   {projections[col].mean():.1%}")
