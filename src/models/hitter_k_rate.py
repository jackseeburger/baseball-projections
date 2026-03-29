"""Hierarchical Bayesian model for hitter strikeout rate (K%).

Model structure:
    K ~ Binomial(PA, p)
    logit(p) = mu_league[year] + player_offset[i] + age_effect(age)

Hierarchy:
    - Population: league-average K% per year (captures rising K% trend)
    - Player ability: partial pooling across players
    - Age curve: quadratic on logit scale (K% improves early, worsens ~30+)

Stabilization: K% stabilizes in ~150 PA — one of the fastest.

BAS-30
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_training_data(
    conn,
    min_pa: int = 50,
    seasons: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """Load and prepare K% training data from Turso.

    Args:
        conn: libsql connection.
        min_pa: Minimum plate appearances to include.
        seasons: (start_year, end_year) or None for all.

    Returns:
        DataFrame with player_id, season, pa, so, k_pct, age, name, position.
    """
    where = f"WHERE b.pa >= {min_pa}"
    if seasons:
        where += f" AND b.season BETWEEN {seasons[0]} AND {seasons[1]}"

    rows = conn.execute(f"""
        SELECT b.player_id, b.season, b.pa, b.so, b.k_pct, b.team,
               p.name, p.primary_position, p.birth_date
        FROM batting_stats b
        JOIN player_metadata p ON b.player_id = p.player_id
        {where}
        ORDER BY b.season, b.player_id
    """).fetchall()

    cols = ['player_id', 'season', 'pa', 'so', 'k_pct', 'team',
            'name', 'position', 'birth_date']
    df = pd.DataFrame(rows, columns=cols)

    # Calculate age (approximate: season - birth year)
    df['birth_year'] = pd.to_datetime(df['birth_date'], errors='coerce').dt.year
    df['age'] = df['season'] - df['birth_year']

    # Clean: drop rows with missing age or impossible values
    df = df.dropna(subset=['age'])
    df = df[(df['age'] >= 18) & (df['age'] <= 45)]

    # Compute strikeouts from k_pct if so is missing/zero
    df['so'] = df['so'].fillna((df['k_pct'] * df['pa']).round().astype(int))
    df['so'] = df['so'].astype(int)

    # Create integer indices for PyMC
    player_ids = df['player_id'].unique()
    player_map = {pid: i for i, pid in enumerate(player_ids)}
    df['player_idx'] = df['player_id'].map(player_map)

    season_vals = sorted(df['season'].unique())
    season_map = {s: i for i, s in enumerate(season_vals)}
    df['season_idx'] = df['season'].map(season_map)

    return df


def load_marcel_projections(conn, season: int = 2026) -> pd.DataFrame:
    """Load Marcel K% projections for comparison."""
    rows = conn.execute("""
        SELECT player_id, projected_value
        FROM marcel_projections
        WHERE stat_name = 'k_pct' AND season = ?
    """, (season,)).fetchall()
    return pd.DataFrame(rows, columns=['player_id', 'marcel_k_pct'])


def build_model(df: pd.DataFrame, projection_year: int = 2026):
    """Build the hierarchical K% model.

    Args:
        df: Training data from load_training_data().
        projection_year: Year to project for (used in age calculations).

    Returns:
        (model, shared_vars) — PyMC model and dict of shared variables.
    """
    import pymc as pm
    import pytensor.tensor as pt

    n_players = df['player_idx'].nunique()
    n_seasons = df['season_idx'].nunique()

    # Center age at 28 (typical peak) for numerical stability
    AGE_CENTER = 28.0
    age_centered = (df['age'].values - AGE_CENTER) / 5.0  # scale by 5 years

    with pm.Model() as model:
        # === Data ===
        player_idx = pm.Data("player_idx", df['player_idx'].values)
        season_idx = pm.Data("season_idx", df['season_idx'].values)
        pa = pm.Data("pa", df['pa'].values)
        age_z = pm.Data("age_z", age_centered)

        # === League-level K% per season ===
        # League K% has been rising (~16% in 2000 → ~23% in 2024)
        # Use a random walk to capture this drift
        mu_league_init = pm.Normal("mu_league_init", mu=-1.2, sigma=0.3)  # ~23% on logit
        mu_league_drift = pm.Normal("mu_league_drift", mu=0, sigma=0.1,
                                     shape=n_seasons - 1)
        mu_league = pm.Deterministic(
            "mu_league",
            pt.concatenate([
                pt.stack([mu_league_init]),
                mu_league_init + pt.cumsum(mu_league_drift)
            ])
        )

        # === Player ability (partial pooling) ===
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.5)
        player_offset_raw = pm.Normal("player_offset_raw", mu=0, sigma=1,
                                       shape=n_players)
        player_offset = pm.Deterministic(
            "player_offset", player_offset_raw * sigma_player
        )

        # === Age curve (quadratic on logit scale) ===
        # K% tends to decrease (improve) in early career, increase after ~30
        age_linear = pm.Normal("age_linear", mu=0, sigma=0.2)
        age_quad = pm.Normal("age_quad", mu=0.05, sigma=0.1)  # slight prior toward U-shape

        age_effect = age_linear * age_z + age_quad * age_z**2

        # === Logit K% ===
        logit_k = mu_league[season_idx] + player_offset[player_idx] + age_effect

        # === Likelihood ===
        p_k = pm.Deterministic("p_k", pm.math.invlogit(logit_k))
        pm.Binomial("obs_k", n=pa, p=p_k, observed=df['so'].values)

    return model


def sample_model(
    model,
    n_samples: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    cores: int = 1,
):
    """Sample the model with NUTS.

    Args:
        model: PyMC model from build_model().
        n_samples: Draws per chain.
        n_chains: Number of chains.
        target_accept: NUTS target acceptance rate.
        cores: Parallel cores (1 on Modal).

    Returns:
        ArviZ InferenceData trace.
    """
    import pymc as pm

    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=1000,
            chains=n_chains,
            cores=cores,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True,
            nuts_sampler="numpyro",  # faster JAX backend
        )
    return trace


def project_players(
    trace,
    df: pd.DataFrame,
    projection_year: int = 2026,
    n_posterior_samples: int = 500,
) -> pd.DataFrame:
    """Generate K% projections from the posterior.

    Uses the last season's league mean + player offset + projected age.

    Returns:
        DataFrame with player_id, name, position, projected_k_pct (mean),
        k_pct_std, k_pct_5, k_pct_25, k_pct_50, k_pct_75, k_pct_95.
    """
    from scipy.special import expit

    # Get posterior samples
    mu_league = trace.posterior["mu_league"].values  # (chains, draws, seasons)
    player_offset = trace.posterior["player_offset"].values  # (chains, draws, players)
    age_linear = trace.posterior["age_linear"].values  # (chains, draws)
    age_quad = trace.posterior["age_quad"].values

    # Flatten chains
    n_chains, n_draws = mu_league.shape[:2]
    mu_league = mu_league.reshape(-1, mu_league.shape[-1])
    player_offset = player_offset.reshape(-1, player_offset.shape[-1])
    age_linear = age_linear.flatten()
    age_quad = age_quad.flatten()

    # Use last season's league mean as baseline for projection year
    league_baseline = mu_league[:, -1]  # last season

    # Get most recent data per player
    latest = df.sort_values('season').groupby('player_id').last().reset_index()

    results = []
    for _, row in latest.iterrows():
        pid = row['player_idx']
        proj_age = projection_year - row['birth_year']
        age_z = (proj_age - 28.0) / 5.0

        # Sample from posterior
        idx = np.random.choice(len(league_baseline), size=n_posterior_samples, replace=False)
        logit_k = (league_baseline[idx] + player_offset[idx, pid] +
                   age_linear[idx] * age_z + age_quad[idx] * age_z**2)
        k_pct_samples = expit(logit_k)

        results.append({
            'player_id': row['player_id'],
            'name': row['name'],
            'position': row['position'],
            'age': int(proj_age),
            'last_season': int(row['season']),
            'last_k_pct': round(float(row['k_pct']), 3),
            'last_pa': int(row['pa']),
            'projected_k_pct': round(float(np.mean(k_pct_samples)), 4),
            'k_pct_std': round(float(np.std(k_pct_samples)), 4),
            'k_pct_5': round(float(np.percentile(k_pct_samples, 5)), 4),
            'k_pct_25': round(float(np.percentile(k_pct_samples, 25)), 4),
            'k_pct_50': round(float(np.percentile(k_pct_samples, 50)), 4),
            'k_pct_75': round(float(np.percentile(k_pct_samples, 75)), 4),
            'k_pct_95': round(float(np.percentile(k_pct_samples, 95)), 4),
        })

    proj_df = pd.DataFrame(results)
    proj_df = proj_df.sort_values('projected_k_pct')
    return proj_df
