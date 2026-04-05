"""Baseball Projections — Modal compute infrastructure.

Single-file Modal app with all entrypoints. Data lives in Cloudflare R2
(accessed via CloudBucketMount), models stored on Modal Volume.

Usage:
    modal run modal_functions/app.py                    # smoke test
    modal run modal_functions/app.py::run_training      # run training (placeholder)
    modal run modal_functions/app.py::run_simulation    # run simulation (placeholder)
    modal run modal_functions/app.py::run_wandb_test    # test wandb integration
"""
import json
import os
from pathlib import Path

import modal

# ═══════════════════════════════════════════════════════════════════════════
# App, Storage, Secrets, Image
# ═══════════════════════════════════════════════════════════════════════════

app = modal.App("baseball-projections")

# R2 cloud bucket for data (replaces Modal Volume for data)
r2_bucket = modal.CloudBucketMount(
    bucket_name="baseball-data",
    bucket_endpoint_url="https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com",
    secret=modal.Secret.from_name("r2-baseball"),
    read_only=True,  # Training reads data, doesn't write back to R2
)

# Modal Volume for model artifacts (traces, checkpoints — small, mutable)
models_volume = modal.Volume.from_name("baseball-models", create_if_missing=True)

# Secrets
_r2 = modal.Secret.from_name("r2-baseball")
_wandb = modal.Secret.from_name("wandb-baseball")
ALL_SECRETS = [_r2, _wandb]

# Container image — PyMC + data stack + wandb
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
        "duckdb>=1.2",
        # Experiment tracking
        "wandb>=0.19",
        # Utilities
        "scipy>=1.14",
        "scikit-learn>=1.5",
        "tqdm",
        "boto3",
    )
)

# Volume + CloudBucketMount mapping
MOUNTS = {
    "/data": r2_bucket,        # R2: read-only Parquet data
    "/models": models_volume,   # Modal Volume: model artifacts
}


# ═══════════════════════════════════════════════════════════════════════════
# Smoke Test — verifies entire stack
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes={"/models": models_volume},
    network_file_systems={"/data": r2_bucket} if False else {},
    secrets=ALL_SECRETS,
    timeout=600,
    memory=4096,
)
def smoke_test():
    """Verify: image builds, R2 data accessible, PyMC samples."""
    import pymc as pm
    import arviz as az
    import pandas as pd
    import numpy as np
    import duckdb

    results = {}

    # 1. Package versions
    results["pymc"] = pm.__version__
    results["arviz"] = az.__version__
    results["pandas"] = pd.__version__
    results["duckdb"] = duckdb.__version__

    # 2. R2 data access via boto3
    import boto3
    from botocore.config import Config
    s3 = boto3.client('s3',
        endpoint_url=os.environ['R2_ENDPOINT_URL'],
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )
    response = s3.list_objects_v2(Bucket='baseball-data', Prefix='statcast/')
    files = response.get('Contents', [])
    results["r2_parquet_files"] = len(files)
    total_size = sum(f['Size'] for f in files)
    results["r2_total_size_gb"] = round(total_size / 1e9, 2)

    # 3. Download and read a sample Parquet to verify data integrity
    import tempfile
    if files:
        sample_key = files[0]['Key']
        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
            s3.download_file('baseball-data', sample_key, tmp.name)
            df = pd.read_parquet(tmp.name)
            results["sample_file"] = sample_key
            results["sample_rows"] = len(df)
            results["sample_columns"] = len(df.columns)

    # 4. DuckDB can query Parquet from R2
    try:
        conn = duckdb.connect()
        conn.execute(f"""
            INSTALL httpfs; LOAD httpfs;
            SET s3_endpoint='{os.environ["R2_ENDPOINT_URL"].replace("https://", "")}';
            SET s3_access_key_id='{os.environ["AWS_ACCESS_KEY_ID"]}';
            SET s3_secret_access_key='{os.environ["AWS_SECRET_ACCESS_KEY"]}';
            SET s3_region='auto';
            SET s3_url_style='path';
        """)
        row_count = conn.execute(
            "SELECT COUNT(*) FROM read_parquet('s3://baseball-data/statcast/statcast_2024.parquet')"
        ).fetchone()[0]
        results["duckdb_r2_query"] = f"OK ({row_count:,} rows from 2024)"
    except Exception as e:
        results["duckdb_r2_query"] = f"FAILED: {e}"

    # 5. PyMC sampling (tiny model — proves MCMC works)
    with pm.Model():
        mu = pm.Normal("mu", mu=0.260, sigma=0.05)
        pm.Normal("obs", mu=mu, sigma=0.03,
                  observed=np.array([0.250, 0.270, 0.265]))
        trace = pm.sample(200, cores=1, chains=1,
                         progressbar=False, return_inferencedata=True)
    results["pymc_sampling"] = "OK"
    results["mu_posterior_mean"] = round(float(trace.posterior["mu"].mean()), 4)

    # 6. Models volume
    results["models_volume"] = Path("/models").exists()

    # 7. wandb connectivity
    try:
        import wandb
        results["wandb_version"] = wandb.__version__
        results["wandb_api_key"] = "configured" if os.environ.get("WANDB_API_KEY") else "MISSING"
    except Exception as e:
        results["wandb"] = f"FAILED: {e}"

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Data Helpers — read Parquet from R2
# ═══════════════════════════════════════════════════════════════════════════

def get_s3_client():
    """Get boto3 S3 client configured for R2."""
    import boto3
    from botocore.config import Config
    return boto3.client('s3',
        endpoint_url=os.environ['R2_ENDPOINT_URL'],
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )


def load_statcast_years(years: list[int] | None = None) -> "pd.DataFrame":
    """Load statcast data from R2 for specified years (or all)."""
    import pandas as pd
    import tempfile

    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket='baseball-data', Prefix='statcast/')

    dfs = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        # Extract year from statcast/statcast_2024.parquet
        file_year = int(key.split('_')[-1].replace('.parquet', ''))
        if years and file_year not in years:
            continue

        with tempfile.NamedTemporaryFile(suffix='.parquet') as tmp:
            s3.download_file('baseball-data', key, tmp.name)
            df = pd.read_parquet(tmp.name)
            dfs.append(df)
            print(f"  Loaded {key}: {len(df):,} rows")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def get_duckdb_conn():
    """Get a DuckDB connection configured for R2 queries."""
    import duckdb
    conn = duckdb.connect()
    conn.execute(f"""
        INSTALL httpfs; LOAD httpfs;
        SET s3_endpoint='{os.environ["R2_ENDPOINT_URL"].replace("https://", "")}';
        SET s3_access_key_id='{os.environ["AWS_ACCESS_KEY_ID"]}';
        SET s3_secret_access_key='{os.environ["AWS_SECRET_ACCESS_KEY"]}';
        SET s3_region='auto';
        SET s3_url_style='path';
    """)
    return conn


# ═══════════════════════════════════════════════════════════════════════════
# Local Entrypoints
# ═══════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def run_smoke_test():
    """Run full smoke test — verifies image, R2 data, PyMC, DuckDB."""
    print("🧪 Running Modal smoke test...")
    results = smoke_test.remote()
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if results.get("pymc_sampling") == "OK" and results.get("r2_parquet_files", 0) > 0:
        print("\n✅ All systems go! Modal + R2 infrastructure is ready.")
    else:
        print("\n⚠️  Some checks failed — review above.")


# ═══════════════════════════════════════════════════════════════════════════
# Training Entrypoint (Phase 2 placeholder)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes={"/models": models_volume},
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

    Reads statcast data from R2, trains PyMC model, saves artifacts
    to Modal Volume and logs to W&B.
    """
    import pandas as pd
    from datetime import datetime

    if not run_name:
        run_name = f"hitter_{projection_year}_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"🏗️  Training run: {run_name}")
    print(f"   Year: {projection_year} | {n_samples} samples × {n_chains} chains")

    # Load data from R2
    print("   Loading statcast data from R2...")
    df = load_statcast_years()
    print(f"   Loaded {len(df):,} total pitches")

    # Placeholder — Phase 2 replaces this with PyMC hierarchical model
    print("⚠️  Placeholder model — full implementation in Phase 2")

    results = {
        "run_name": run_name,
        "projection_year": projection_year,
        "model_type": "placeholder",
        "total_pitches": len(df),
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


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Entrypoint (Phase 3 placeholder)
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes={"/models": models_volume},
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
    """Monte Carlo season simulation. Stub — Phase 3."""
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


# ═══════════════════════════════════════════════════════════════════════════
# W&B Integration Test
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes={"/models": models_volume},
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

    observed_ba = np.array([0.250, 0.270, 0.265, 0.280, 0.245])
    with pm.Model() as model:
        mu = pm.Normal("mu_ba", mu=0.260, sigma=0.05)
        sigma = pm.HalfNormal("sigma_ba", sigma=0.03)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=observed_ba)
        trace = pm.sample(500, cores=1, chains=2,
                         progressbar=False, return_inferencedata=True)

    rhat = az.rhat(trace)
    ess = az.ess(trace, method="bulk")
    summary = az.summary(trace)

    wandb.log({
        "diagnostics/rhat/mu_ba": float(rhat["mu_ba"].values),
        "diagnostics/rhat/sigma_ba": float(rhat["sigma_ba"].values),
        "diagnostics/ess/mu_ba_bulk": float(ess["mu_ba"].values),
        "diagnostics/ess/sigma_ba_bulk": float(ess["sigma_ba"].values),
    })

    div = trace.sample_stats.get("diverging")
    n_div = int(div.values.sum()) if div is not None else 0
    wandb.log({"diagnostics/divergences": n_div})

    table = wandb.Table(dataframe=summary.reset_index())
    wandb.log({"diagnostics/summary": table})

    import matplotlib.pyplot as plt
    ax = az.plot_posterior(trace)
    fig = ax.ravel()[0].figure
    wandb.log({"posterior/plot": wandb.Image(fig)})
    plt.close(fig)

    ax = az.plot_trace(trace, compact=True)
    fig = ax.ravel()[0].figure
    wandb.log({"posterior/trace_plot": wandb.Image(fig)})
    plt.close(fig)

    import pandas as pd
    sample_proj = pd.DataFrame({
        "player": ["Test Player A", "Test Player B", "Test Player C"],
        "projected_ba": [0.265, 0.280, 0.250],
        "projected_obp": [0.340, 0.360, 0.320],
        "projected_slg": [0.420, 0.480, 0.400],
    })
    proj_table = wandb.Table(dataframe=sample_proj)
    wandb.log({"projections_preview": proj_table})

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
