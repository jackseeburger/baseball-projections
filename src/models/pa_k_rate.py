"""
PA-level Bayesian Strikeout Rate Model

Hierarchical Bayesian model for projecting batter strikeout rates using
plate-appearance-level data. Each PA is modeled as a Bernoulli outcome
(K or not-K) with a logit-linear predictor combining:

    logit(p_K) = league_trend[season] + player_ability[batter]
               + handedness[stand] + park_effect[team]
               + age_curve(age)

Components:
    - League trend: random walk on logit scale across seasons
    - Player ability: partial pooling with non-centered parameterization
    - Handedness: batter stand (L/R) effect
    - Park effects: ZeroSumNormal across teams
    - Age curve: quadratic on centered age (peak ~ 27)

Likelihood: Bernoulli(logistic(eta)) per PA

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

def load_pa_data(pa_dir: Path | str | None = None) -> pd.DataFrame:
    """Load and concatenate PA outcome parquet files.

    Args:
        pa_dir: Directory containing pa_outcomes_YYYY.parquet files.
                Defaults to DATA_DIR / 'parquet' / 'pa_outcomes'.

    Returns:
        DataFrame with one row per PA, columns include:
        batter, game_year, stand, is_k, home_team, away_team, inning_topbot.
    """
    if pa_dir is None:
        pa_dir = PARQUET_DIR / "pa_outcomes"
    pa_dir = Path(pa_dir)

    parquet_files = sorted(pa_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pa_dir}")

    logger.info(f"Loading {len(parquet_files)} PA parquet files from {pa_dir}")

    # Only read the columns we need to save memory
    keep_cols = [
        "batter", "game_year", "stand", "is_k",
        "home_team", "away_team", "inning_topbot",
    ]
    frames = []
    for f in parquet_files:
        df = pd.read_parquet(f, columns=keep_cols)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
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
) -> dict:
    """Prepare PA data for the PyMC model.

    Steps:
        1. Filter pitchers (batters with < min_pa total PAs are dropped).
        2. Determine batting team from inning_topbot.
        3. Create integer indices for batters, seasons, teams.
        4. Estimate birth year and compute centered age.
        5. Merge park factors (default to 1.0 if unavailable).

    Args:
        pa: Raw PA DataFrame from load_pa_data().
        park_factors: Park factor DataFrame (optional).
        min_pa: Minimum career PAs to include a batter.

    Returns:
        Dictionary with all arrays/indices needed by the model.
    """
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

    # --- Age estimation ---
    # Approximate birth year as first appearance year minus 23
    first_year = df.groupby("batter")["game_year"].min()
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

    # Outcome
    is_k = df["is_k"].values.astype(np.int64)

    model_data = {
        # Dimensions
        "n_obs": len(df),
        "n_batters": len(batters),
        "n_seasons": len(seasons),
        "n_teams": n_teams,
        # Index arrays (int64)
        "batter_idx": df["batter_idx"].values,
        "season_idx": df["season_idx"].values,
        "team_idx": df["team_idx"].values,
        "stand_idx": df["stand_idx"].values,
        # Continuous features (float64)
        "age_centered": df["age_centered"].values,
        "log_pf_k": df["log_pf_k"].values,
        # Outcome
        "is_k": is_k,
        # Lookup tables
        "seasons": seasons,
        "batters": batters,
        "teams": teams,
        "season_map": season_map,
        "batter_map": batter_map,
        "team_map": team_map,
        # Metadata
        "batter_meta": batter_meta,
        "log_pf_matrix": log_pf,
        "df": df,
    }

    logger.info(
        f"Model data ready: {model_data['n_obs']:,} obs, "
        f"{model_data['n_batters']:,} batters, "
        f"{model_data['n_seasons']} seasons, "
        f"{model_data['n_teams']} teams"
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
            + handedness * stand_idx
            + park_effect[team]
            + beta_age * age_centered
            + beta_age2 * age_centered^2
            + log_pf_k  (park factor offset)

        is_k ~ Bernoulli(logistic(eta))

    Non-centered parameterization is used for player abilities to
    improve sampling geometry.

    Args:
        data: Dictionary from prepare_model_data().

    Returns:
        PyMC Model object (not yet sampled).
    """
    coords = {
        "batter": data["batters"],
        "season": data["seasons"],
        "team": data["teams"],
        "obs_id": np.arange(data["n_obs"]),
    }

    with pm.Model(coords=coords) as model:
        # ─── Mutable data containers (for posterior predictive) ───────────
        batter_idx = pm.Data("batter_idx", data["batter_idx"], dims="obs_id")
        season_idx = pm.Data("season_idx", data["season_idx"], dims="obs_id")
        team_idx = pm.Data("team_idx", data["team_idx"], dims="obs_id")
        stand_idx = pm.Data("stand_idx", data["stand_idx"], dims="obs_id")
        age_c = pm.Data("age_centered", data["age_centered"], dims="obs_id")
        log_pf = pm.Data("log_pf_k", data["log_pf_k"], dims="obs_id")

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

        # ─── Likelihood ──────────────────────────────────────────────────
        p_k = pm.Deterministic("p_k", pm.math.invlogit(eta), dims="obs_id")
        pm.Bernoulli(
            "obs_k",
            p=p_k,
            observed=data["is_k"],
            dims="obs_id",
        )

    n_params = (
        1                        # league_init
        + data["n_seasons"]      # league_innovations
        + 1 + 1                  # mu_ability, sigma_ability
        + data["n_batters"]      # z_ability
        + 1                      # beta_hand
        + data["n_teams"] - 1    # park_effect (zero-sum = n-1 free)
        + 2                      # beta_age, beta_age2
    )
    logger.info(f"Model built: ~{n_params:,} free parameters, "
                f"{data['n_obs']:,} observations")
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

    # Quick diagnostics
    rhat = az.rhat(trace)
    max_rhat = max(
        float(rhat[v].values.max()) if rhat[v].values.ndim > 0
        else float(rhat[v].values)
        for v in rhat.data_vars
    )
    divergences = 0
    if hasattr(trace, "sample_stats"):
        div = trace.sample_stats.get("diverging")
        if div is not None:
            divergences = int(div.values.sum())
    logger.info(f"Max R-hat: {max_rhat:.4f}, Divergences: {divergences}")

    return trace


# ═══════════════════════════════════════════════════════════════════════════════
# Projections
# ═══════════════════════════════════════════════════════════════════════════════

def generate_projections(
    trace: az.InferenceData,
    data: dict,
    projection_year: int = PROJECTION_YEAR,
    recent_seasons: int = 3,
) -> pd.DataFrame:
    """Generate 2026 K-rate projections from posterior samples.

    For each batter who appeared in at least one of the last `recent_seasons`,
    compute the posterior predictive K-rate at the projection year by:
        - Extrapolating the league trend (last value + one innovation draw)
        - Using the player's posterior ability
        - Applying the age curve at their projected age
        - Using neutral park/handedness (or their actual stand)

    Args:
        trace: Posterior trace from sample_model().
        data: Model data dictionary.
        projection_year: Year to project (default 2026).
        recent_seasons: Include batters active within this many years.

    Returns:
        DataFrame with columns: batter, stand, age, projected_k_rate,
        k_rate_lower, k_rate_upper, posterior_mean_ability, total_pa.
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
            # No park effect (neutral venue) and no log_pf
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
        })

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
        notes=f"PA-level Bernoulli K% model projecting to {PROJECTION_YEAR}",
        offline=offline,
    )

    try:
        # MCMC diagnostics
        diagnostics = tracker.log_mcmc_diagnostics(trace)
        logger.info(f"wandb: logged diagnostics (healthy={diagnostics['healthy']})")

        # Dataset stats
        tracker.log_dataset_stats(
            data["df"][["is_k", "age", "stand_idx"]],
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
        **sampler_overrides: Override default sampler kwargs.

    Returns:
        Tuple of (trace, projections_df, model_data_dict).
    """
    logger.info("=" * 60)
    logger.info("PA-level Bayesian K-Rate Model")
    logger.info("=" * 60)

    # 1. Load data
    pa = load_pa_data(pa_dir)
    park_factors = load_park_factors(pf_path)

    # 2. Prepare model data
    model_data = prepare_model_data(pa, park_factors, min_pa=min_pa)
    del pa  # free raw data
    gc.collect()

    # 3. Build model
    model = build_model(model_data)

    # 4. Sample
    trace = sample_model(model, **sampler_overrides)

    # 5. Generate projections
    projections = generate_projections(
        trace, model_data, projection_year=projection_year,
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
