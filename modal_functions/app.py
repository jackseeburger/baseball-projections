"""Baseball Projections — Modal compute infrastructure.

Single-file Modal app with all entrypoints. Keeps it simple and avoids
cross-module import issues in Modal's container environment.

Usage:
    modal run modal_functions/app.py                    # smoke test
    modal run modal_functions/app.py::upload_data       # upload parquet data
    modal run modal_functions/app.py::run_training      # run training (placeholder)
    modal run modal_functions/app.py::run_simulation    # run simulation (placeholder)
    modal run modal_functions/app.py::run_wandb_test    # test wandb integration
"""
import json
import os
from pathlib import Path

import modal

# ═══════════════════════════════════════════════════════════════════════════
# App, Volumes, Secrets, Image
# ═══════════════════════════════════════════════════════════════════════════

app = modal.App("baseball-projections")

# Persistent volumes
data_volume = modal.Volume.from_name("baseball-data", create_if_missing=True)
models_volume = modal.Volume.from_name("baseball-models", create_if_missing=True)

VOLUME_MOUNTS = {
    "/data": data_volume,
    "/models": models_volume,
}

# Secrets — Turso DB + Weights & Biases
_turso = modal.Secret.from_name("turso-baseball")
_wandb = modal.Secret.from_name("wandb-baseball")
ALL_SECRETS = [_turso, _wandb]

# Container image — PyMC + data stack + Turso client + wandb
pymc_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML
        "pymc>=5.21",
        "arviz>=0.20",
        "jax[cpu]",
        # Data
        "pandas>=2.2",
        "pyarrow>=18",
        "numpy>=1.26",
        # Turso / libSQL
        "libsql-experimental>=0.0.50",
        # Experiment tracking
        "wandb>=0.19",
        # JAX MCMC backend
        "numpyro>=0.15",
        # Visualization
        "matplotlib>=3.9",
        # Utilities
        "scipy>=1.14",
        "scikit-learn>=1.5",
        "tqdm",
    )
)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke Test — verifies entire stack
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=600,
    memory=4096,
)
def smoke_test():
    """Verify: image builds, volumes mount, Turso connects, PyMC samples."""
    import pymc as pm
    import arviz as az
    import pandas as pd
    import numpy as np

    results = {}

    # 1. Package versions
    results["pymc"] = pm.__version__
    results["arviz"] = az.__version__
    results["pandas"] = pd.__version__

    # 2. Data volume
    parquet_dir = Path("/data/parquet")
    if parquet_dir.exists():
        parquet_files = list(parquet_dir.rglob("*.parquet"))
        results["parquet_files"] = len(parquet_files)
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            results["sample_file"] = parquet_files[0].name
            results["sample_rows"] = len(df)
    else:
        results["parquet_files"] = 0
        results["note"] = "Run upload_data first"

    # 3. Turso connection
    try:
        import libsql_experimental as libsql
        conn = libsql.connect(
            "baseball.db",
            sync_url=os.environ["TURSO_DATABASE_URL"],
            auth_token=os.environ["TURSO_AUTH_TOKEN"],
        )
        conn.sync()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        results["turso_tables"] = tables
        results["turso"] = "connected"
    except Exception as e:
        results["turso"] = f"FAILED: {e}"

    # 4. PyMC sampling (tiny model — proves MCMC works)
    with pm.Model():
        mu = pm.Normal("mu", mu=0.260, sigma=0.05)
        pm.Normal("obs", mu=mu, sigma=0.03,
                  observed=np.array([0.250, 0.270, 0.265]))
        trace = pm.sample(200, cores=1, chains=1,
                         progressbar=False, return_inferencedata=True)
    results["pymc_sampling"] = "OK"
    results["mu_posterior_mean"] = round(float(trace.posterior["mu"].mean()), 4)

    # 5. Models volume
    results["models_volume"] = Path("/models").exists()

    # 6. wandb connectivity
    try:
        import wandb
        results["wandb_version"] = wandb.__version__
        results["wandb_api_key"] = "configured" if os.environ.get("WANDB_API_KEY") else "MISSING"
    except Exception as e:
        results["wandb"] = f"FAILED: {e}"

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Data Upload — push local parquets to Modal Volume
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def run_smoke_test():
    """Run full smoke test — verifies image, volumes, Turso, PyMC."""
    print("🧪 Running Modal smoke test...")
    results = smoke_test.remote()
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if results.get("turso") == "connected" and results.get("pymc_sampling") == "OK":
        print("\n✅ All systems go! Modal infrastructure is ready.")
    else:
        print("\n⚠️  Some checks failed — review above.")


@app.local_entrypoint()
def upload_data():
    """Upload parquet files from local data/parquet/ to Modal volume."""
    local_dir = Path(__file__).parent.parent / "data" / "parquet"
    if not local_dir.exists():
        print(f"❌ Not found: {local_dir}")
        return

    files = []
    for root, _, fnames in os.walk(local_dir):
        for f in fnames:
            if f.endswith(".parquet"):
                lp = Path(root) / f
                rp = f"/data/parquet/{lp.relative_to(local_dir)}"
                files.append((str(lp), rp))

    print(f"📦 Uploading {len(files)} parquet files...")
    with data_volume.batch_upload() as batch:
        for local_path, remote_path in files:
            batch.put_file(local_path, remote_path)
            print(f"  ↑ {remote_path}")
    print(f"✅ Done — {len(files)} files on 'baseball-data' volume")


# ═══════════════════════════════════════════════════════════════════════════
# Training Entrypoint (Phase 2 placeholder)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=3600,
    memory=8192,
    cpu=4.0,
)
def train_hitter_model(
    projection_year: int = 2026,
    n_samples: int = 2000,
    n_chains: int = 4,
    run_name: str = "",
):
    """Train hierarchical Bayesian hitter projection model.

    Currently a placeholder that validates the data pipeline end-to-end.
    Full model implementation comes in Phase 2 (SIG-232+).
    """
    import pandas as pd
    from datetime import datetime

    if not run_name:
        run_name = f"hitter_{projection_year}_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"🏗️  Training run: {run_name}")
    print(f"   Year: {projection_year} | {n_samples} samples × {n_chains} chains")

    # Load data
    parquet_dir = Path("/data/parquet")
    hitter_seasons = pd.read_parquet(parquet_dir / "hitter_seasons.parquet")
    marcel = pd.read_parquet(parquet_dir / f"marcel_hitters_{projection_year}.parquet")
    print(f"   {len(hitter_seasons)} hitter-seasons, {len(marcel)} Marcel projections")

    # Placeholder — Phase 2 replaces this with PyMC hierarchical model
    print("⚠️  Placeholder model — full implementation in Phase 2")

    results = {
        "run_name": run_name,
        "projection_year": projection_year,
        "model_type": "placeholder",
        "n_hitters": len(marcel),
        "status": "complete",
    }

    # Save metadata to models volume
    run_dir = Path(f"/models/runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    models_volume.commit()

    print(f"✅ {run_name} complete")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Entrypoint (Phase 3 placeholder)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=3600,
    memory=8192,
    cpu=4.0,
)
def simulate_season(
    projection_year: int = 2026,
    n_seasons: int = 10_000,
    run_name: str = "",
):
    """Monte Carlo season simulation. Stub — Phase 3 (SIG-237+)."""
    from datetime import datetime

    if not run_name:
        run_name = f"sim_{projection_year}_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"🎲 Simulation: {run_name} ({n_seasons:,} seasons)")
    print("⚠️  Stub — implementation in Phase 3")

    results = {
        "run_name": run_name,
        "projection_year": projection_year,
        "n_seasons": n_seasons,
        "status": "stub",
    }

    run_dir = Path(f"/models/sims/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(results, f, indent=2)
    models_volume.commit()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# W&B Integration Test
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=600,
    memory=4096,
)
def wandb_integration_test():
    """End-to-end test of wandb tracking with a tiny PyMC model."""
    import wandb
    import pymc as pm
    import arviz as az
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")

    results = {}

    # 1. Init wandb run
    run = wandb.init(
        project="baseball-projections",
        entity="jseeburger",
        name="integration-test",
        config={
            "model_type": "test",
            "n_samples": 500,
            "n_chains": 2,
            "test": True,
        },
        tags=["test", "integration"],
        notes="Automated integration test — validates wandb logging from Modal",
        reinit=True,
    )
    results["wandb_run_url"] = run.url

    # 2. Run tiny PyMC model
    observed_ba = np.array([0.250, 0.270, 0.265, 0.280, 0.245])
    with pm.Model() as model:
        mu = pm.Normal("mu_ba", mu=0.260, sigma=0.05)
        sigma = pm.HalfNormal("sigma_ba", sigma=0.03)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=observed_ba)
        trace = pm.sample(500, cores=1, chains=2,
                         progressbar=False, return_inferencedata=True)

    # 3. Log MCMC diagnostics
    rhat = az.rhat(trace)
    ess = az.ess(trace, method="bulk")
    summary = az.summary(trace)

    wandb.log({
        "diagnostics/rhat/mu_ba": float(rhat["mu_ba"].values),
        "diagnostics/rhat/sigma_ba": float(rhat["sigma_ba"].values),
        "diagnostics/ess/mu_ba_bulk": float(ess["mu_ba"].values),
        "diagnostics/ess/sigma_ba_bulk": float(ess["sigma_ba"].values),
    })

    # Check for divergences
    div = trace.sample_stats.get("diverging")
    n_div = int(div.values.sum()) if div is not None else 0
    wandb.log({"diagnostics/divergences": n_div})

    # 4. Log summary table
    table = wandb.Table(dataframe=summary.reset_index())
    wandb.log({"diagnostics/summary": table})

    # 5. Log posterior plots
    import matplotlib.pyplot as plt

    ax = az.plot_posterior(trace)
    fig = ax.ravel()[0].figure
    wandb.log({"posterior/plot": wandb.Image(fig)})
    plt.close(fig)

    ax = az.plot_trace(trace, compact=True)
    fig = ax.ravel()[0].figure
    wandb.log({"posterior/trace_plot": wandb.Image(fig)})
    plt.close(fig)

    # 6. Log a sample projections table
    import pandas as pd
    sample_proj = pd.DataFrame({
        "player": ["Test Player A", "Test Player B", "Test Player C"],
        "projected_ba": [0.265, 0.280, 0.250],
        "projected_obp": [0.340, 0.360, 0.320],
        "projected_slg": [0.420, 0.480, 0.400],
    })
    proj_table = wandb.Table(dataframe=sample_proj)
    wandb.log({"projections_preview": proj_table})

    # 7. Save model artifact
    import tempfile
    artifact = wandb.Artifact("integration-test-model", type="model",
                               metadata={"test": True})
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        trace.to_netcdf(f.name)
        artifact.add_file(f.name, name="trace.nc")
    wandb.log_artifact(artifact, aliases=["test"])

    results["mu_ba_mean"] = round(float(trace.posterior["mu_ba"].mean()), 4)
    results["rhat_mu"] = round(float(rhat["mu_ba"].values), 4)
    results["divergences"] = n_div
    results["status"] = "success"

    wandb.finish()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Hitter K% Model (BAS-30)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=3600,
    memory=8192,
    cpu=4.0,
)
def train_k_rate_model(
    projection_year: int = 2026,
    n_samples: int = 2000,
    n_chains: int = 4,
    target_accept: float = 0.9,
    min_pa: int = 50,
    train_seasons: tuple = (2000, 2025),
    run_name: str = "",
):
    """Train hierarchical Bayesian K% model with full wandb tracking."""
    import numpy as np
    import pandas as pd
    import pymc as pm
    import arviz as az
    import wandb
    import libsql_experimental as libsql
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pytensor.tensor as pt
    from datetime import datetime
    from scipy.special import expit

    if not run_name:
        run_name = f"k-rate-v1-{projection_year}-{datetime.now():%Y%m%d-%H%M%S}"

    print(f"🎯 K% Model Training: {run_name}")
    print(f"   Projection year: {projection_year}")
    print(f"   Samples: {n_samples} × {n_chains} chains")
    print(f"   Training data: {train_seasons[0]}-{train_seasons[1]}, min PA={min_pa}")

    # ── Connect to Turso ──
    conn = libsql.connect("baseball.db",
        sync_url=os.environ["TURSO_DATABASE_URL"],
        auth_token=os.environ["TURSO_AUTH_TOKEN"])
    conn.sync()

    # ── Load data ──
    rows = conn.execute(f"""
        SELECT b.player_id, b.season, b.pa, b.so, b.k_pct, b.team,
               p.name, p.primary_position, p.birth_date
        FROM batting_stats b
        JOIN player_metadata p ON b.player_id = p.player_id
        WHERE b.pa >= {min_pa}
          AND b.season BETWEEN {train_seasons[0]} AND {train_seasons[1]}
        ORDER BY b.season, b.player_id
    """).fetchall()
    cols = ['player_id', 'season', 'pa', 'so', 'k_pct', 'team',
            'name', 'position', 'birth_date']
    df = pd.DataFrame(rows, columns=cols)

    # Calculate age — birth_date may be NULL, estimate from debut
    df['birth_year'] = pd.to_datetime(df['birth_date'], errors='coerce').dt.year
    # If birth_date is missing, estimate: debut_year - 23 (average debut age)
    if df['birth_year'].isna().all():
        debut_years = df.groupby('player_id')['season'].min().rename('debut_year')
        df = df.merge(debut_years, on='player_id')
        df['birth_year'] = df['debut_year'] - 23
        df = df.drop(columns=['debut_year'])
    else:
        df['birth_year'] = df['birth_year'].fillna(
            df.groupby('player_id')['season'].transform('min') - 23)
    df['age'] = df['season'] - df['birth_year']
    df = df[(df['age'] >= 18) & (df['age'] <= 45)]
    df['so'] = df['so'].fillna((df['k_pct'] * df['pa']).round().astype(int)).astype(int)

    # Create indices
    player_ids = df['player_id'].unique()
    player_map = {pid: i for i, pid in enumerate(player_ids)}
    df['player_idx'] = df['player_id'].map(player_map)
    season_vals = sorted(df['season'].unique())
    season_map = {s: i for i, s in enumerate(season_vals)}
    df['season_idx'] = df['season'].map(season_map)

    # Marcel projections
    marcel_rows = conn.execute(
        "SELECT player_id, projected_value FROM marcel_projections WHERE stat_name = 'k_pct' AND season = ?",
        (projection_year,)).fetchall()
    marcel = pd.DataFrame(marcel_rows, columns=['player_id', 'marcel_k_pct'])

    print(f"   Loaded {len(df)} player-seasons, {df['player_idx'].nunique()} players")
    print(f"   Marcel projections: {len(marcel)}")

    # ── Init wandb ──
    run = wandb.init(
        project="baseball-projections",
        entity="jseeburger",
        name=run_name,
        config={
            "model": "hitter_k_rate",
            "version": "v1",
            "projection_year": projection_year,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "target_accept": target_accept,
            "min_pa": min_pa,
            "train_seasons": list(train_seasons),
            "n_player_seasons": len(df),
            "n_players": int(df['player_idx'].nunique()),
            "n_seasons": int(df['season_idx'].nunique()),
        },
        tags=["hitter", "k-rate", "v1"],
        group="hitter-k-rate",
        job_type="train",
        reinit=True,
    )

    # Log data summary
    wandb.log({
        "data/n_player_seasons": len(df),
        "data/n_players": int(df['player_idx'].nunique()),
        "data/n_seasons": int(df['season_idx'].nunique()),
        "data/mean_pa": float(df['pa'].mean()),
        "data/mean_k_pct": float(df['k_pct'].mean()),
        "data/k_pct_std": float(df['k_pct'].std()),
    })

    # ── Build & sample model ──
    print("🔨 Building model...")
    n_players = df['player_idx'].nunique()
    n_seasons_model = df['season_idx'].nunique()
    AGE_CENTER = 28.0
    age_centered = (df['age'].values - AGE_CENTER) / 5.0

    with pm.Model() as model:
        player_idx_data = pm.Data("player_idx", df['player_idx'].values.astype(np.int64))
        season_idx_data = pm.Data("season_idx", df['season_idx'].values.astype(np.int64))
        pa_data = pm.Data("pa", df['pa'].values.astype(np.int64))
        age_z = pm.Data("age_z", age_centered.astype(np.float64))

        # League K% per season (random walk — captures rising trend)
        mu_league_init = pm.Normal("mu_league_init", mu=-1.2, sigma=0.3)
        mu_league_drift = pm.Normal("mu_league_drift", mu=0, sigma=0.1,
                                     shape=n_seasons_model - 1)
        mu_league = pm.Deterministic("mu_league",
            pt.concatenate([pt.stack([mu_league_init]),
                           mu_league_init + pt.cumsum(mu_league_drift)]))

        # Player ability (partial pooling)
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.5)
        player_offset_raw = pm.Normal("player_offset_raw", mu=0, sigma=1, shape=n_players)
        player_offset = pm.Deterministic("player_offset", player_offset_raw * sigma_player)

        # Age curve (quadratic on logit scale)
        age_linear = pm.Normal("age_linear", mu=0, sigma=0.2)
        age_quad = pm.Normal("age_quad", mu=0.05, sigma=0.1)
        age_effect = age_linear * age_z + age_quad * age_z**2

        # Likelihood
        logit_k = mu_league[season_idx_data] + player_offset[player_idx_data] + age_effect
        p_k = pm.Deterministic("p_k", pm.math.invlogit(logit_k))
        pm.Binomial("obs_k", n=pa_data, p=p_k, observed=df['so'].values.astype(np.int64))

    print(f"   Model variables: {[v.name for v in model.free_RVs]}")

    print(f"⚡ Sampling ({n_samples} draws × {n_chains} chains)...")
    with model:
        trace = pm.sample(draws=n_samples, tune=1000, chains=n_chains, cores=1,
                         target_accept=target_accept, return_inferencedata=True,
                         progressbar=True, nuts_sampler="numpyro")
    print("   Sampling complete!")

    # ── Log MCMC diagnostics ──
    print("📊 Logging diagnostics...")
    rhat = az.rhat(trace)
    ess_bulk = az.ess(trace, method="bulk")
    summary = az.summary(trace, var_names=["mu_league_init", "sigma_player",
                                            "age_linear", "age_quad"])

    # R-hat
    for var in ["mu_league_init", "sigma_player", "age_linear", "age_quad"]:
        if var in rhat:
            val = float(rhat[var].values) if rhat[var].values.ndim == 0 else float(rhat[var].values.max())
            wandb.log({f"diagnostics/rhat/{var}": val})

    # ESS
    for var in ["mu_league_init", "sigma_player", "age_linear", "age_quad"]:
        if var in ess_bulk:
            val = float(ess_bulk[var].values) if ess_bulk[var].values.ndim == 0 else float(ess_bulk[var].values.min())
            wandb.log({f"diagnostics/ess_bulk/{var}": val})

    # Divergences
    div = trace.sample_stats.get("diverging")
    n_div = int(div.values.sum()) if div is not None else 0
    wandb.log({"diagnostics/divergences": n_div})

    # Summary table
    wandb.log({"diagnostics/summary": wandb.Table(dataframe=summary.reset_index())})

    # ── Posterior plots ──
    print("📈 Generating plots...")

    # Trace plot
    ax = az.plot_trace(trace, var_names=["mu_league_init", "sigma_player",
                                          "age_linear", "age_quad"], compact=True)
    fig = ax.ravel()[0].figure
    wandb.log({"posterior/trace_plot": wandb.Image(fig)})
    plt.close(fig)

    # League K% trend
    mu_league = trace.posterior["mu_league"].values
    league_k = expit(mu_league.reshape(-1, mu_league.shape[-1]))
    seasons = sorted(df['season'].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(seasons, np.percentile(league_k, 5, axis=0),
                    np.percentile(league_k, 95, axis=0), alpha=0.2, color='steelblue')
    ax.fill_between(seasons, np.percentile(league_k, 25, axis=0),
                    np.percentile(league_k, 75, axis=0), alpha=0.4, color='steelblue')
    ax.plot(seasons, np.median(league_k, axis=0), 'b-', linewidth=2, label='Model median')
    # Actual league averages
    actual_league = df.groupby('season')['k_pct'].mean()
    ax.plot(actual_league.index, actual_league.values, 'ro', markersize=4, label='Actual')
    ax.set_xlabel("Season")
    ax.set_ylabel("K%")
    ax.set_title("League K% Trend (posterior)")
    ax.legend()
    wandb.log({"posterior/league_k_trend": wandb.Image(fig)})
    plt.close(fig)

    # Age curve
    ages = np.linspace(20, 40, 50)
    age_z = (ages - 28.0) / 5.0
    al = trace.posterior["age_linear"].values.flatten()
    aq = trace.posterior["age_quad"].values.flatten()
    age_effects = np.array([al * z + aq * z**2 for z in age_z])  # (50, n_samples)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(ages, np.percentile(age_effects, 5, axis=1),
                    np.percentile(age_effects, 95, axis=1), alpha=0.2, color='coral')
    ax.plot(ages, np.median(age_effects, axis=1), 'r-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(28, color='gray', linestyle=':', alpha=0.5, label='Age 28 (center)')
    ax.set_xlabel("Age")
    ax.set_ylabel("Effect on logit(K%)")
    ax.set_title("Age Effect on K% (logit scale)")
    ax.legend()
    wandb.log({"posterior/age_curve": wandb.Image(fig)})
    plt.close(fig)

    # ── Generate projections ──
    print("🔮 Generating projections...")
    mu_league_post = trace.posterior["mu_league"].values
    player_offset_post = trace.posterior["player_offset"].values
    al_post = trace.posterior["age_linear"].values
    aq_post = trace.posterior["age_quad"].values

    # Flatten chains
    mu_league_flat = mu_league_post.reshape(-1, mu_league_post.shape[-1])
    player_offset_flat = player_offset_post.reshape(-1, player_offset_post.shape[-1])
    al_flat = al_post.flatten()
    aq_flat = aq_post.flatten()
    league_baseline = mu_league_flat[:, -1]  # last season as projection baseline

    latest = df.sort_values('season').groupby('player_id').last().reset_index()
    n_posterior_samples = min(500, len(league_baseline))
    proj_results = []
    for _, row in latest.iterrows():
        pid = row['player_idx']
        proj_age = projection_year - row['birth_year']
        az_val = (proj_age - 28.0) / 5.0
        idx = np.random.choice(len(league_baseline), size=n_posterior_samples, replace=False)
        logit_k_samples = (league_baseline[idx] + player_offset_flat[idx, pid] +
                           al_flat[idx] * az_val + aq_flat[idx] * az_val**2)
        k_samples = expit(logit_k_samples)
        proj_results.append({
            'player_id': row['player_id'], 'name': row['name'],
            'position': row['position'], 'age': int(proj_age),
            'last_season': int(row['season']), 'last_k_pct': round(float(row['k_pct']), 3),
            'last_pa': int(row['pa']),
            'projected_k_pct': round(float(np.mean(k_samples)), 4),
            'k_pct_std': round(float(np.std(k_samples)), 4),
            'k_pct_5': round(float(np.percentile(k_samples, 5)), 4),
            'k_pct_50': round(float(np.percentile(k_samples, 50)), 4),
            'k_pct_95': round(float(np.percentile(k_samples, 95)), 4),
        })
    proj = pd.DataFrame(proj_results).sort_values('projected_k_pct')
    print(f"   Projected {len(proj)} players")

    # Log projections table
    wandb.log({"projections/k_rate": wandb.Table(dataframe=proj.head(200))})

    # Top/bottom K% projections
    print("\n🏆 Lowest projected K% (best contact):")
    for _, r in proj.head(10).iterrows():
        print(f"   {r['name']:25s} {r['projected_k_pct']:.1%} ({r['k_pct_5']:.1%}-{r['k_pct_95']:.1%})")

    print("\n💨 Highest projected K% (most Ks):")
    for _, r in proj.tail(10).iterrows():
        print(f"   {r['name']:25s} {r['projected_k_pct']:.1%} ({r['k_pct_5']:.1%}-{r['k_pct_95']:.1%})")

    # ── Compare to Marcel ──
    if len(marcel) > 0:
        comparison = proj.merge(marcel, on='player_id', how='inner')
        if len(comparison) > 0:
            diff = comparison['projected_k_pct'] - comparison['marcel_k_pct']
            wandb.log({
                "vs_marcel/n_players_compared": len(comparison),
                "vs_marcel/mean_abs_diff": float(diff.abs().mean()),
                "vs_marcel/correlation": float(comparison['projected_k_pct'].corr(comparison['marcel_k_pct'])),
            })
            print(f"\n📊 vs Marcel ({len(comparison)} players):")
            print(f"   Correlation: {comparison['projected_k_pct'].corr(comparison['marcel_k_pct']):.4f}")
            print(f"   Mean absolute difference: {diff.abs().mean():.4f}")

    # ── Save artifacts ──
    print("💾 Saving artifacts...")

    # Save trace
    import tempfile
    artifact = wandb.Artifact(f"k-rate-model-{projection_year}", type="model",
                               metadata={"projection_year": projection_year,
                                         "n_samples": n_samples, "n_chains": n_chains})
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        trace.to_netcdf(f.name)
        artifact.add_file(f.name, name="trace.nc")
    wandb.log_artifact(artifact, aliases=["latest"])

    # Save projections to Modal volume
    proj_path = Path(f"/models/projections/k_rate_{projection_year}.parquet")
    proj_path.parent.mkdir(parents=True, exist_ok=True)
    proj.to_parquet(str(proj_path), index=False)
    models_volume.commit()

    results = {
        "run_name": run_name,
        "wandb_url": run.url,
        "projection_year": projection_year,
        "n_players_projected": len(proj),
        "divergences": n_div,
        "status": "success",
    }

    wandb.finish()
    print(f"\n✅ Done! wandb run: {results['wandb_url']}")
    return results


@app.local_entrypoint()
def run_k_rate_model(
    year: int = 2026,
    samples: int = 2000,
    chains: int = 4,
):
    """Train the K% model from the VPS."""
    print(f"🎯 Launching K% model training for {year}...")
    result = train_k_rate_model.remote(
        projection_year=year,
        n_samples=samples,
        n_chains=chains,
    )
    print("\n" + "=" * 60)
    print("K% MODEL RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# VPS Trigger Helpers
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def run_training(
    year: int = 2026,
    samples: int = 2000,
    chains: int = 4,
):
    """Trigger a training run from the VPS."""
    print(f"🚀 Triggering hitter model training for {year}...")
    result = train_hitter_model.remote(
        projection_year=year,
        n_samples=samples,
        n_chains=chains,
    )
    print("\nResults:")
    for k, v in result.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def run_simulation(year: int = 2026, seasons: int = 10_000):
    """Trigger a season simulation from the VPS."""
    print(f"🎲 Triggering season simulation for {year}...")
    result = simulate_season.remote(
        projection_year=year,
        n_seasons=seasons,
    )
    print("\nResults:")
    for k, v in result.items():
        print(f"  {k}: {v}")


@app.local_entrypoint()
def run_wandb_test():
    """Test wandb integration end-to-end on Modal."""
    print("🧪 Testing wandb integration on Modal...")
    results = wandb_integration_test.remote()
    print("\n" + "=" * 60)
    print("WANDB INTEGRATION TEST RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if results.get("status") == "success":
        print("\n✅ wandb integration working! Check your dashboard:")
        print(f"   {results.get('wandb_run_url', 'https://wandb.ai/jseeburger/baseball-projections')}")
    else:
        print("\n⚠️  Something went wrong — review above.")
