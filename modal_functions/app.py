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
