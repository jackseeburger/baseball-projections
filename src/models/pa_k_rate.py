"""
PA-level Bayesian Strikeout Rate Model

Hierarchical Bayesian model for projecting batter strikeout rates using
plate-appearance-level data, collapsed to Binomial counts per
(batter, season, team, stand [, pitcher]) cell with a logit-linear predictor:

    logit(p_K) = league_trend[season] + player_ability[batter]
               + pitcher_ability[pitcher]
               + handedness[stand] + park_effect[team]
               + age_curve(age)

Components:
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

Designed for Modal deployment (8GB RAM, 4 CPU, NumpyRo backend).

Usage:
    python -m src.models.pa_k_rate              # local test
    from src.models.pa_k_rate import run_model  # programmatic
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.cutoff import apply_cutoff, assert_no_post_cutoff, cutoff_exposure

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


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

BASE_PA_COLUMNS = [
    "batter", "game_year", "stand", "is_k",
    "home_team", "away_team", "inning_topbot",
]


def load_pa_data(
    pa_dir: Path | str | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
    include_pitcher: bool = False,
) -> pd.DataFrame:
    """Load and concatenate PA outcome parquet files.

    Args:
        pa_dir: Directory containing pa_outcomes_YYYY.parquet files.
                Defaults to DATA_DIR / 'parquet' / 'pa_outcomes'.
        cutoff_date: ISO date. Rows dated on or after it are dropped as they
            are read, so post-cutoff PA never reach memory, let alone the
            likelihood. Requires a `game_date` column.
        include_pitcher: also read `pitcher`, for the opposing-pitcher term.

    Returns:
        DataFrame with one row per PA, columns include:
        batter, game_year, stand, is_k, home_team, away_team, inning_topbot
        (plus game_date and pitcher when asked for).
    """
    if pa_dir is None:
        pa_dir = PARQUET_DIR / "pa_outcomes"
    pa_dir = Path(pa_dir)

    parquet_files = sorted(pa_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pa_dir}")

    logger.info(f"Loading {len(parquet_files)} PA parquet files from {pa_dir}")

    # Only read the columns we need to save memory
    keep_cols = list(BASE_PA_COLUMNS)
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


def load_park_factors(pf_path: Path | str | None = None) -> pd.DataFrame:
    """Load park-factor parquet.

    Args:
        pf_path: Path to park_factors.parquet.

    Returns:
        DataFrame with columns: team, game_year, k_park_factor.
    """
    if pf_path is None:
        pf_path = PARQUET_DIR / "park_factors.parquet"
    pf_path = Path(pf_path)

    if not pf_path.exists():
        logger.warning(f"Park factors not found at {pf_path}; using neutral (1.0)")
        return None

    pf = pd.read_parquet(pf_path)
    logger.info(f"Loaded park factors: {len(pf)} rows, "
                f"{pf['team'].nunique()} teams")
    return pf


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

    if park_factors is not None:
        for _, row in park_factors.iterrows():
            t = team_map.get(row["team"])
            s = season_map.get(int(row["game_year"]))
            if t is not None and s is not None:
                log_pf[t, s] = np.log(float(row["k_park_factor"]))

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
            total_pa=("is_k", "size"),
            career_k_rate=("is_k", "mean"),
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
    cells = aggregate_binomial_cells(df, cell_cols=cell_cols)
    logger.info(f"Binomial aggregation: {len(df):,} PAs → {len(cells):,} cells "
                f"({len(df) / max(len(cells), 1):.1f}x compression)")

    model_data = {
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
    return model_data


# ═══════════════════════════════════════════════════════════════════════════════
# PyMC Model
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(data: dict) -> pm.Model:
    """Build the hierarchical Bayesian K-rate model.

    Structure (all on logit scale):
        eta = league_trend[season]
            + player_ability[batter]
            + pitcher_ability[pitcher]        (when data carries pitchers)
            + handedness * stand_idx
            + park_effect[team]
            + beta_age * age_centered
            + beta_age2 * age_centered^2
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

    Args:
        data: Dictionary from prepare_model_data().

    Returns:
        PyMC Model object (not yet sampled).
    """
    include_pitcher = bool(data.get("include_pitcher")) and data.get("n_pitchers")
    coords = {
        "batter": data["batters"],
        "season": data["seasons"],
        "team": data["teams"],
        "cell": np.arange(data["n_obs"]),
    }
    if include_pitcher:
        coords["pitcher"] = data["pitchers"]

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
        # Initial intercept ~ league-average K rate (~22% → logit ≈ -1.27)
        league_init = pm.Normal("league_init", mu=-1.27, sigma=0.3)
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

        # ─── Age curve: quadratic on centered age ─────────────────────────
        # Linear and quadratic coefficients
        beta_age = pm.Normal("beta_age", mu=0.0, sigma=0.02)
        beta_age2 = pm.Normal("beta_age2", mu=0.005, sigma=0.01)
        # Positive beta_age2 → K rate increases away from peak age (U-shape)

        # ─── Linear predictor ────────────────────────────────────────────
        eta = (
            league_trend[season_idx]
            + player_ability[batter_idx]
            + beta_hand * stand_idx
            + park_effect[team_idx]
            + beta_age * age_c
            + beta_age2 * (age_c ** 2)
            + log_pf
        )
        if include_pitcher:
            eta = eta + pitcher_ability[pitcher_idx]

        # ─── Likelihood ──────────────────────────────────────────────────
        # Binomial over cells; no p_k Deterministic — writing a float per
        # cell per draw into the trace was most of the old memory bill.
        pm.Binomial(
            "obs_k",
            n=n_trials,
            p=pm.math.invlogit(eta),
            observed=data["k"],
            dims="cell",
        )

    n_params = (
        1                        # league_init
        + data["n_seasons"]      # league_innovations
        + 1 + 1                  # mu_ability, sigma_ability
        + data["n_batters"]      # z_ability
        + 1                      # beta_hand
        + data["n_teams"] - 1    # park_effect (zero-sum = n-1 free)
        + 2                      # beta_age, beta_age2
        + (data["n_pitchers"] + 1 if include_pitcher else 0)  # z_pitcher, sigma
    )
    logger.info(f"Model built: ~{n_params:,} free parameters, "
                f"{data['n_obs']:,} cells ({data.get('n_pa', 0):,} PAs)"
                f"{', + pitcher effect' if include_pitcher else ''}")
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

def _project_unseen(
    unseen: pd.DataFrame | None,
    data: dict,
    post,
    projected_trend: np.ndarray,
    beta_hand_flat: np.ndarray,
    beta_age_flat: np.ndarray,
    beta_age2_flat: np.ndarray,
    projection_year: int,
    n_samples: int,
    already: set[int],
) -> list[dict]:
    """Population-level projections for batters the fit never saw.

    A hierarchical model already says what to do about a player with no
    history: his ability is a draw from the fitted population,
    `N(mu_ability, sigma_ability)`. That is the same construction as drawing a
    fresh random effect for an unseen group, done here in numpy on the
    posterior rather than by extending the model, because nothing else about
    the prediction needs the graph.

    `unseen` is [batter, age] with an optional `stand`. Without a stand the
    handedness term is marginalized at the training set's right-handed PA
    share — the honest answer when we have not seen the player bat, and the
    only place a projection here is not conditioned on a known covariate.
    """
    if unseen is None or len(unseen) == 0:
        return []
    if "batter" not in unseen.columns:
        raise KeyError("unseen frame needs a 'batter' column")

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
        age_c = 0.0 if np.isnan(age) else age - REFERENCE_AGE

        stand = row.get("stand") if "stand" in unseen.columns else None
        s_term = share_r if stand not in ("L", "R") else (1.0 if stand == "R" else 0.0)

        ability = mu_ability + sigma_ability * rng.standard_normal(n_samples)
        eta = (
            projected_trend
            + ability
            + beta_hand_flat * s_term
            + beta_age_flat * age_c
            + beta_age2_flat * (age_c ** 2)
        )
        p_k = 1.0 / (1.0 + np.exp(-eta))
        rows.append({
            "batter": batter_id,
            "stand": stand if stand in ("L", "R") else None,
            "age": age,
            "projected_k_rate": float(np.mean(p_k)),
            "k_rate_std": float(np.std(p_k)),
            "k_rate_lower": float(np.percentile(p_k, 5)),
            "k_rate_upper": float(np.percentile(p_k, 95)),
            "posterior_mean_ability": float(np.mean(ability)),
            "total_pa": 0,
            "career_k_rate": float("nan"),
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
    """Generate K-rate projections from posterior samples.

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

    # Extract posterior arrays (chains × draws × ...)
    league_trend = post["league_trend"].values         # (chains, draws, n_seasons)
    player_ability = post["player_ability"].values     # (chains, draws, n_batters)
    beta_hand = post["beta_hand"].values               # (chains, draws)
    beta_age = post["beta_age"].values                 # (chains, draws)
    beta_age2 = post["beta_age2"].values               # (chains, draws)
    league_innovations = post["league_innovations"].values  # (chains, draws, n_seasons)

    # Flatten chains × draws → samples
    n_chains, n_draws = league_trend.shape[:2]
    n_samples = n_chains * n_draws
    league_trend_flat = league_trend.reshape(n_samples, -1)
    player_ability_flat = player_ability.reshape(n_samples, -1)
    beta_hand_flat = beta_hand.reshape(n_samples)
    beta_age_flat = beta_age.reshape(n_samples)
    beta_age2_flat = beta_age2.reshape(n_samples)
    innovations_flat = league_innovations.reshape(n_samples, -1)

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
        age_c = proj_age - REFERENCE_AGE

        # Stand index
        s_idx = 1 if row["stand"] == "R" else 0

        # Compute eta for each posterior sample (neutral park)
        eta = (
            projected_trend
            + player_ability_flat[:, b_idx]
            + beta_hand_flat * s_idx
            + beta_age_flat * age_c
            + beta_age2_flat * (age_c ** 2)
            # No park effect (neutral venue) and no log_pf, and a neutral
            # opposing pitcher: pitcher_ability is zero-mean by construction,
            # so leaving it out *is* the average-arm projection.
        )

        # Convert to probability
        p_k = 1.0 / (1.0 + np.exp(-eta))

        results.append({
            "batter": batter_id,
            "stand": row["stand"],
            "age": proj_age,
            "projected_k_rate": float(np.mean(p_k)),
            "k_rate_std": float(np.std(p_k)),
            "k_rate_lower": float(np.percentile(p_k, 5)),
            "k_rate_upper": float(np.percentile(p_k, 95)),
            "posterior_mean_ability": float(np.mean(player_ability_flat[:, b_idx])),
            "total_pa": int(row["total_pa"]),
            "career_k_rate": float(row["career_k_rate"]),
            "last_season": int(row["last_season"]),
            "unseen": False,
        })

    results.extend(_project_unseen(
        unseen, data, post, projected_trend,
        beta_hand_flat, beta_age_flat, beta_age2_flat,
        projection_year, n_samples,
        already={int(r["batter"]) for r in results},
    ))

    proj_df = pd.DataFrame(results)
    proj_df = proj_df.sort_values("projected_k_rate", ascending=True).reset_index(drop=True)

    logger.info(
        f"Projections generated: median K% = {proj_df['projected_k_rate'].median():.3f}, "
        f"range [{proj_df['projected_k_rate'].min():.3f}, "
        f"{proj_df['projected_k_rate'].max():.3f}]"
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

    tracker = WandbTracker(
        run_name=f"k-rate-bayesian-{PROJECTION_YEAR}",
        model_type="hitter",
        config=model_config,
        tags=["k-rate", "bayesian", "pa-level"],
        notes=f"Binomial-cell K% model projecting to {PROJECTION_YEAR}",
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
            artifact_name=f"k-rate-projections-{PROJECTION_YEAR}",
            metadata={
                "projection_year": PROJECTION_YEAR,
                "n_batters": len(projections),
                "median_k_rate": float(projections["projected_k_rate"].median()),
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
            artifact_name=f"k-rate-trace-{PROJECTION_YEAR}",
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
        **sampler_overrides: Override default sampler kwargs.

    Returns:
        Tuple of (trace, projections_df, model_data_dict).
    """
    logger.info("=" * 60)
    logger.info("PA-level Bayesian K-Rate Model")
    logger.info("=" * 60)

    # 1. Load data
    pa = load_pa_data(pa_dir, cutoff_date=cutoff_date,
                      include_pitcher=include_pitcher)
    park_factors = load_park_factors(pf_path)

    # 2. Prepare model data
    model_data = prepare_model_data(
        pa, park_factors, min_pa=min_pa,
        cutoff_date=cutoff_date, include_pitcher=include_pitcher,
    )
    del pa  # free raw data
    gc.collect()

    # 3. Build model
    model = build_model(model_data)

    # 4. Sample
    trace = sample_model(model, **sampler_overrides)

    # 5. Generate projections
    projections = generate_projections(
        trace, model_data, projection_year=projection_year, unseen=unseen,
    )

    # 6. Save projections to parquet
    output_dir = DATA_DIR / "projections"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"k_rate_projections_{projection_year}.parquet"
    projections.to_parquet(output_path, index=False)
    logger.info(f"Projections saved to {output_path}")

    # 7. wandb logging
    model_config = {
        "model": "pa_k_rate_bernoulli",
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
        FAST:      If '1', use minimal sampling for quick test
        NO_WANDB:  If '1', skip wandb logging
    """
    fast_mode = os.environ.get("FAST", "0") == "1"
    no_wandb = os.environ.get("NO_WANDB", "0") == "1"

    overrides = {}
    if fast_mode:
        logger.info("⚡ FAST MODE: minimal sampling for testing")
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
        **overrides,
    )

    # Print top/bottom projections
    print("\n" + "=" * 60)
    print(f"K-Rate Projections for {PROJECTION_YEAR}")
    print("=" * 60)
    print("\n🔝 Lowest projected K% (best contact):")
    print(projections.head(15).to_string(index=False))
    print("\n⬇️  Highest projected K% (most Ks):")
    print(projections.tail(15).to_string(index=False))
    print(f"\n📊 Median projected K%: {projections['projected_k_rate'].median():.1%}")
    print(f"📊 Mean projected K%:   {projections['projected_k_rate'].mean():.1%}")
