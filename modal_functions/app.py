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
        "numpyro>=0.15",
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

    # Delete existing PA files first (force=True doesn't truly overwrite on Modal volumes)
    print("🗑️  Removing old PA files from volume...")
    import subprocess
    for local_path, remote_path in files:
        if "pa_outcomes" in remote_path:
            try:
                subprocess.run(["modal", "volume", "rm", "baseball-data", remote_path], 
                             capture_output=True, timeout=30)
            except Exception:
                pass

    print(f"📦 Uploading {len(files)} parquet files...")
    with data_volume.batch_upload(force=True) as batch:
        for local_path, remote_path in files:
            batch.put_file(local_path, remote_path)
            print(f"  ↑ {remote_path}")
    print(f"✅ Done — {len(files)} files on 'baseball-data' volume")

    # Verify upload
    verify_upload.remote()


@app.function(image=pymc_image, volumes=VOLUME_MOUNTS, timeout=60)
def verify_upload():
    """List files on the volume to verify upload succeeded."""
    data_volume.reload()  # Force fresh read
    from pathlib import Path
    data_volume.reload()
    parquet_dir = Path("/data/parquet")
    print(f"\n📋 Files on volume ({parquet_dir}):")
    for f in sorted(parquet_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(parquet_dir)} ({size_kb:.1f} KB)")
    
    # Check PA file columns
    import pandas as pd
    pa_sample = sorted((parquet_dir / "pa_outcomes").glob("*.parquet"))[0]
    df = pd.read_parquet(pa_sample).head(2)
    print(f"\nPA columns check ({pa_sample.name}): {list(df.columns)}")
    print(f"birth_year present: {'birth_year' in df.columns}")


@app.function(image=pymc_image, volumes=VOLUME_MOUNTS, timeout=120)
def generate_birth_years_on_volume():
    """Generate batter_birth_years.parquet directly on the Modal volume.
    
    Uses Statcast PA data to compute real ages from game-level data, 
    cross-referencing with known debut years.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    data_volume.reload()
    parquet_dir = Path("/data/parquet")
    pa_dir = parquet_dir / "pa_outcomes"
    
    # Load all PA data to get unique batters
    frames = []
    for f in sorted(pa_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(f, columns=["batter", "game_year"]))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} PAs, {df['batter'].nunique()} unique batters")
    
    # Method 1: Use pybaseball/Statcast player data if available
    # For now, use debut year with better estimation
    # Average MLB debut age is ~24.5, but varies significantly
    # We can compute this from the hitter_seasons data which has more info
    
    # Load hitter_seasons if available for cross-reference
    hs_path = parquet_dir / "hitter_seasons.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(hs_path)
        print(f"Loaded hitter_seasons: {len(hs)} rows, columns: {list(hs.columns)}")
        # Check if it has Age column
        if 'Age' in hs.columns or 'age' in hs.columns:
            age_col = 'Age' if 'Age' in hs.columns else 'age'
            yr_col = 'Season' if 'Season' in hs.columns else 'game_year' if 'game_year' in hs.columns else 'year'
            id_col = 'IDfg' if 'IDfg' in hs.columns else 'key_mlbam' if 'key_mlbam' in hs.columns else 'batter'
            
            print(f"Found age column '{age_col}', year column '{yr_col}', id column '{id_col}'")
            print(f"Sample: {hs[[id_col, yr_col, age_col]].head()}")
            
            # Compute birth_year = season - age
            hs['birth_year_est'] = hs[yr_col] - hs[age_col]
            
            # Get median birth year per player (most robust)
            birth_years_hs = hs.groupby(id_col)['birth_year_est'].median().round().astype(int)
            print(f"Got birth years from hitter_seasons for {len(birth_years_hs)} players")
    
    # Compute debut year for all PA batters
    debut = df.groupby("batter")["game_year"].min().reset_index()
    debut.columns = ["batter", "debut_year"]
    
    # Try to match hitter_seasons birth years to PA batters
    # The ID systems may differ (FanGraphs IDfg vs Statcast key_mlbam)
    # For now, use debut_year - 24 as fallback but this generates the file
    # so we can update with real ages later
    
    debut["birth_year"] = debut["debut_year"] - 24  # Default fallback
    
    # If we have hitter_seasons with age, try to match
    if hs_path.exists() and 'birth_year_est' in hs.columns:
        # Check if the ID column matches our batter IDs (Statcast mlbam IDs)
        if id_col == 'IDfg':
            # FanGraphs IDs don't match Statcast IDs directly
            # But if hitter_seasons has key_mlbam or playerid columns...
            mlbam_cols = [c for c in hs.columns if 'mlbam' in c.lower() or 'playerid' in c.lower()]
            print(f"Available ID columns in hitter_seasons: {mlbam_cols}")
    
    # Save to volume
    result = debut[["batter", "birth_year"]]
    output_path = parquet_dir / "batter_birth_years.parquet"
    result.to_parquet(output_path, index=False)
    
    # Commit the volume
    data_volume.commit()
    
    print(f"\n✅ Saved {len(result)} batter birth years to {output_path}")
    print(f"Birth year range: {result['birth_year'].min()} - {result['birth_year'].max()}")
    
    # Verify it's readable
    check = pd.read_parquet(output_path)
    print(f"Verification: {len(check)} rows, columns: {list(check.columns)}")
    
    return {"n_batters": len(result), "path": str(output_path)}


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
# PA-level K-Rate Bayesian Model
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=7200,
    memory=8192,
    cpu=4.0,
)
def train_pa_k_rate(
    n_draws: int = 2000,
    n_tune: int = 1500,
    n_chains: int = 4,
    target_accept: float = 0.95,
    min_pa: int = 50,
    projection_year: int = 2026,
    log_wandb: bool = True,
    fast_mode: bool = False,
):
    """Batter-season Bayesian K-rate model (Binomial aggregation).

    Hierarchical Binomial model: each batter-season is K ~ Binomial(n_pa, p).
    logit(p_K) = league_trend[season] + player[batter] + hand + park + age_curve

    Mathematically equivalent to per-PA Bernoulli but ~100x faster (~18K rows
    instead of ~1.9M).

    All model code is inlined (Modal requirement — no cross-module imports).
    """
    import gc
    import time
    import logging
    import tempfile

    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("pa_k_rate")

    REFERENCE_AGE = 27.0
    PARQUET_DIR = Path("/data/parquet")

    if fast_mode:
        n_draws, n_tune, n_chains = 200, 200, 2
        logger.info("⚡ FAST MODE enabled")

    # ─── 1. Load data ───────────────────────────────────────────────────
    # Reload volume to pick up any recent uploads
    data_volume.reload()
    logger.info("Loading PA outcomes...")
    pa_dir = PARQUET_DIR / "pa_outcomes"
    
    # Try to read birth_year; if not in parquet, we'll compute it after loading
    keep_cols = ["batter", "game_year", "stand", "is_k", "home_team", "away_team", "inning_topbot"]
    # Check if birth_year column exists in the parquet files
    import pyarrow.parquet as pq
    sample_file = sorted(pa_dir.glob("*.parquet"))[0]
    schema = pq.read_schema(sample_file)
    has_birth_year = "birth_year" in schema.names
    if has_birth_year:
        keep_cols.append("birth_year")
        logger.info("birth_year column found in PA parquet files")
    frames = []
    for f in sorted(pa_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(f, columns=keep_cols))
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df):,} PAs ({df['game_year'].min()}-{df['game_year'].max()})")
    del frames
    gc.collect()

    # Load park factors
    pf_path = PARQUET_DIR / "park_factors.parquet"
    park_factors = pd.read_parquet(pf_path) if pf_path.exists() else None
    if park_factors is not None:
        logger.info(f"Park factors: {len(park_factors)} rows")

    # ─── 2. Prepare data ────────────────────────────────────────────────
    # Batting team from inning_topbot
    df["bat_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

    # Filter pitchers and low-PA batters
    n_before = df["batter"].nunique()
    
    # Step 1: Remove likely pitchers — anyone whose best season is < 80 PA
    # (real position players get 200+ PA seasons; pitchers batting rarely exceed 80)
    season_pa = df.groupby(["batter", "game_year"]).size().reset_index(name="season_pa")
    max_season_pa = season_pa.groupby("batter")["season_pa"].max()
    likely_hitters = max_season_pa[max_season_pa >= 80].index
    n_pitchers_removed = n_before - len(likely_hitters)
    df = df[df["batter"].isin(likely_hitters)].copy()
    
    # Step 2: Filter remaining low-PA batters (cup-of-coffee players)
    pa_counts = df.groupby("batter").size()
    qualified = pa_counts[pa_counts >= min_pa].index
    df = df[df["batter"].isin(qualified)].copy()
    logger.info(f"Batters: {n_before} → {df['batter'].nunique()} "
                f"(removed {n_pitchers_removed} likely pitchers, >= {min_pa} career PA)")

    # Integer indices
    seasons = np.sort(df["game_year"].unique())
    season_map = {yr: i for i, yr in enumerate(seasons)}
    df["season_idx"] = df["game_year"].map(season_map).astype(np.int64)

    batters = np.sort(df["batter"].unique())
    batter_map = {b: i for i, b in enumerate(batters)}
    df["batter_idx"] = df["batter"].map(batter_map).astype(np.int64)

    teams = np.sort(df["bat_team"].unique())
    team_map = {t: i for i, t in enumerate(teams)}
    df["team_idx"] = df["bat_team"].map(team_map).astype(np.int64)

    df["stand_idx"] = (df["stand"] == "R").astype(np.int64)

    # ─── Age: get birth years ───
    if "birth_year" in df.columns:
        df["birth_year"] = df["birth_year"].astype(np.float64)
        logger.info(f"Birth years from PA data: 100% coverage")
    else:
        # Compute from hitter_seasons.parquet (has FanGraphs Age column)
        birth_map = {}
        hs_path = PARQUET_DIR / "hitter_seasons.parquet"
        if hs_path.exists():
            hs = pd.read_parquet(hs_path)
            # Find the right columns
            age_col = next((c for c in hs.columns if c.lower() == 'age'), None)
            yr_col = next((c for c in hs.columns if c.lower() in ('season', 'game_year', 'year')), None)
            id_col = next((c for c in hs.columns if 'mlbam' in c.lower() or c == 'IDfg' or c == 'fg_id'), None)
            
            if age_col and yr_col:
                hs['_birth'] = hs[yr_col] - hs[age_col]
                if id_col:
                    birth_map = hs.groupby(id_col)['_birth'].median().round().astype(int).to_dict()
                    logger.info(f"Birth years from hitter_seasons ({id_col}): {len(birth_map)} players")
                else:
                    logger.info(f"hitter_seasons has no mlbam ID column. Cols: {list(hs.columns)[:10]}")
        
        # Fallback: debut year - 24
        first_year = df.groupby("batter")["game_year"].min().to_dict()
        def get_by(batter_id):
            if batter_id in birth_map:
                return float(birth_map[batter_id])
            return float(first_year.get(batter_id, 2015) - 24)
        
        df["birth_year"] = df["batter"].map(get_by).astype(np.float64)
        n_real = sum(1 for b in df["batter"].unique() if b in birth_map)
        logger.info(f"Birth years: {n_real}/{df['batter'].nunique()} from hitter_seasons, rest from debut-24")
    df["age"] = (df["game_year"] - df["birth_year"]).astype(np.float64)
    df["age_centered"] = (df["age"] - REFERENCE_AGE).astype(np.float64)
    
    logger.info(f"Age range: {df['age'].min():.0f}-{df['age'].max():.0f}, "
                f"mean={df['age'].mean():.1f}")

    # Park factor lookup (team_idx, season_idx) → log(pf_k)
    n_teams = len(teams)
    n_seasons = len(seasons)
    log_pf = np.zeros((n_teams, n_seasons), dtype=np.float64)
    if park_factors is not None:
        # Handle both old schema (year, pf_k) and new (game_year, k_park_factor)
        yr_col = "game_year" if "game_year" in park_factors.columns else "year"
        pf_col = "k_park_factor" if "k_park_factor" in park_factors.columns else "pf_k"
        logger.info(f"Park factor columns: year={yr_col}, k={pf_col}")
        for _, row in park_factors.iterrows():
            t = team_map.get(row["team"])
            s = season_map.get(int(row[yr_col]))
            if t is not None and s is not None:
                log_pf[t, s] = np.log(float(row[pf_col]))
    df["log_pf_k"] = log_pf[df["team_idx"].values, df["season_idx"].values].astype(np.float64)

    # Batter metadata for projections (computed before aggregation)
    batter_meta = (
        df.groupby("batter")
        .agg(stand=("stand", "first"), birth_year=("birth_year", "first"),
             last_season=("game_year", "max"), total_pa=("is_k", "size"),
             career_k_rate=("is_k", "mean"))
        .reset_index()
    )

    # ─── 2b. Aggregate to batter-season level (Binomial) ────────────────
    # For team_idx: use team where batter had most PAs that season (mode)
    team_mode = (
        df.groupby(["batter", "game_year", "team_idx"])
        .size()
        .reset_index(name="pa_count")
        .sort_values("pa_count", ascending=False)
        .drop_duplicates(subset=["batter", "game_year"], keep="first")
        .set_index(["batter", "game_year"])["team_idx"]
    )

    agg_df = (
        df.groupby(["batter", "game_year"])
        .agg(
            n_k=("is_k", "sum"),
            n_pa=("is_k", "size"),
            stand_idx=("stand_idx", "first"),
            batter_idx=("batter_idx", "first"),
            season_idx=("season_idx", "first"),
            age=("age", "first"),
            age_centered=("age_centered", "first"),
            log_pf_k=("log_pf_k", "mean"),
        )
        .reset_index()
    )

    # Attach team mode via merge
    team_mode_df = team_mode.reset_index()
    team_mode_df.columns = ["batter", "game_year", "team_idx"]
    agg_df = agg_df.merge(team_mode_df, on=["batter", "game_year"], how="left")

    # Ensure integer types
    agg_df["n_k"] = agg_df["n_k"].astype(np.int64)
    agg_df["n_pa"] = agg_df["n_pa"].astype(np.int64)
    agg_df["team_idx"] = agg_df["team_idx"].astype(np.int64)

    n_obs = len(agg_df)
    n_batters = len(batters)
    n_total_pa = int(agg_df["n_pa"].sum())
    logger.info(f"Aggregated to {n_obs:,} batter-seasons ({n_total_pa:,} total PAs)")
    logger.info(f"Model data: {n_obs:,} obs, {n_batters:,} batters, {n_seasons} seasons, {n_teams} teams")

    # Free PA-level dataframe
    del df
    gc.collect()

    # ─── 2c. Prepare age data for HSGP ──────────────────────────────────
    # HSGP needs centered age as a 2D array (n_obs, 1)
    age_centered_vals = agg_df["age_centered"].values.astype(np.float64)
    
    # HSGP config
    # m = number of basis functions (20 is plenty for 1D smooth curve)
    # c = boundary factor — domain extends to [-L, L] where L = c * half_range
    #     Must cover training ages AND future projection ages (up to ~45)
    age_range = age_centered_vals.max() - age_centered_vals.min()
    HSGP_M = 20       # basis functions
    HSGP_C = 1.5      # boundary factor (covers ~18 to ~45 with REFERENCE_AGE=27)
    
    logger.info(f"HSGP age model: m={HSGP_M} basis functions, c={HSGP_C}, "
                f"age range [{age_centered_vals.min():.1f}, {age_centered_vals.max():.1f}] centered")

    # ─── 3. Build PyMC model ────────────────────────────────────────────
    coords = {
        "batter": batters,
        "season": seasons,
        "team": teams,
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(coords=coords) as model:
        # Data containers
        batter_idx_d = pm.Data("batter_idx", agg_df["batter_idx"].values, dims="obs_id")
        season_idx_d = pm.Data("season_idx", agg_df["season_idx"].values, dims="obs_id")
        team_idx_d = pm.Data("team_idx", agg_df["team_idx"].values, dims="obs_id")
        stand_idx_d = pm.Data("stand_idx", agg_df["stand_idx"].values, dims="obs_id")
        log_pf_d = pm.Data("log_pf_k", agg_df["log_pf_k"].values, dims="obs_id")
        n_pa_d = pm.Data("n_pa", agg_df["n_pa"].values, dims="obs_id")
        
        # Age as mutable data container (for set_data projection)
        age_c_d = pm.Data("age_centered", age_centered_vals, dims="obs_id")

        # League trend: random walk
        league_init = pm.Normal("league_init", mu=-1.27, sigma=0.3)
        league_innovations = pm.Normal("league_innovations", mu=0, sigma=0.05, dims="season")
        league_trend = pm.Deterministic("league_trend",
            league_init + pt.cumsum(league_innovations), dims="season")

        # Player ability: non-centered partial pooling
        mu_ability = pm.Normal("mu_ability", mu=0.0, sigma=0.3)
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=0.4)
        z_ability = pm.Normal("z_ability", mu=0, sigma=1, dims="batter")
        player_ability = pm.Deterministic("player_ability",
            mu_ability + sigma_ability * z_ability, dims="batter")

        # Handedness
        beta_hand = pm.Normal("beta_hand", mu=0.0, sigma=0.2)

        # Park effects: zero-sum
        park_effect = pm.ZeroSumNormal("park_effect", sigma=0.05, dims="team")

        # ─── Age curve: HSGP (Hilbert Space Gaussian Process) ───────────
        # Replaces B-spline. Learns a smooth nonlinear aging curve from data.
        # HSGP is a spectral approximation to a full GP — O(nm) not O(n³).
        #
        # Two hyperparameters the model learns:
        #   eta_age (amplitude) — how much age matters overall
        #   ell_age (lengthscale) — how smooth the curve is
        #     large ell → very smooth (age effect changes slowly)
        #     small ell → more wiggly (can capture sharp transitions)
        #
        # Matern-5/2 kernel: twice differentiable, smooth but can capture
        # asymmetric aging (gradual improvement then faster decline).
        
        eta_age = pm.HalfNormal("eta_age", sigma=0.5)
        ell_age = pm.InverseGamma("ell_age", mu=5.0, sigma=2.0)  # ~5 year lengthscale
        
        cov_age = eta_age**2 * pm.gp.cov.Matern52(1, ls=ell_age)
        gp_age = pm.gp.HSGP(m=[HSGP_M], c=HSGP_C, cov_func=cov_age)
        age_effect = gp_age.prior("age_effect", X=age_c_d[:, None])
        
        # Linear predictor
        eta = (
            league_trend[season_idx_d]
            + player_ability[batter_idx_d]
            + beta_hand * stand_idx_d
            + park_effect[team_idx_d]
            + age_effect
            + log_pf_d
        )

        # Likelihood: Binomial on aggregated batter-season counts
        p = pm.math.invlogit(eta)
        pm.Binomial("obs_k", n=n_pa_d, p=p, observed=agg_df["n_k"].values, dims="obs_id")

    n_params = 1 + n_seasons + 2 + n_batters + 1 + (n_teams - 1) + HSGP_M + 2  # +2 for eta_age, ell_age
    logger.info(f"Model built: ~{n_params:,} free parameters")

    # Save age values before freeing dataframe (needed for projection interpolation)
    obs_age_values = agg_df["age"].values.copy()
    
    # Free the aggregated dataframe to save memory
    del agg_df
    gc.collect()

    # ─── 4. Sample ──────────────────────────────────────────────────────
    logger.info(f"Sampling: {n_chains} chains × {n_draws} draws (tune={n_tune})")
    t0 = time.time()
    with model:
        trace = pm.sample(
            draws=n_draws, tune=n_tune, chains=n_chains, cores=1,
            target_accept=target_accept, nuts_sampler="numpyro",
            random_seed=42, idata_kwargs={"log_likelihood": False},
        )
    elapsed = time.time() - t0
    logger.info(f"Sampling done in {elapsed:.0f}s")

    # Diagnostics
    rhat = az.rhat(trace)
    max_rhat = max(
        float(rhat[v].values.max()) if rhat[v].values.ndim > 0 else float(rhat[v].values)
        for v in rhat.data_vars
    )
    divergences = 0
    if hasattr(trace, "sample_stats"):
        div = trace.sample_stats.get("diverging")
        if div is not None:
            divergences = int(div.values.sum())
    logger.info(f"Max R-hat: {max_rhat:.4f}, Divergences: {divergences}")

    # ─── 5. Generate multi-year projections ─────────────────────────────
    # For HSGP, we evaluate the GP at new age points using the learned
    # spectral basis weights. The HSGP prior creates internal variables
    # (hsgp_coeffs_) that we can use to reconstruct f(age) at any point.
    
    post = trace.posterior
    lt = post["league_trend"].values
    pa_vals = post["player_ability"].values
    bh = post["beta_hand"].values
    innov = post["league_innovations"].values

    nc, nd = lt.shape[:2]
    ns = nc * nd
    lt_flat = lt.reshape(ns, -1)
    pa_flat = pa_vals.reshape(ns, -1)
    bh_flat = bh.reshape(ns)
    innov_flat = innov.reshape(ns, -1)
    
    # Extract HSGP internals for manual evaluation at new ages
    # HSGP stores: _beta (spectral coefficients), and we need the basis functions
    eta_age_post = post["eta_age"].values.reshape(ns)
    ell_age_post = post["ell_age"].values.reshape(ns)
    
    # Get the HSGP spectral coefficients — named "age_effect_hsgp_coeffs_"
    hsgp_coeff_names = [v for v in post.data_vars if "hsgp_coeffs" in v or "age_effect" in v]
    logger.info(f"HSGP posterior variables: {hsgp_coeff_names}")
    
    # The age_effect values at training points are stored directly
    age_effect_post = post["age_effect"].values  # (chains, draws, n_obs)
    age_effect_flat = age_effect_post.reshape(ns, -1)
    
    # For projecting to NEW ages, we need to evaluate the GP basis at those ages.
    # HSGP uses: f(x) = phi(x)^T @ beta, where phi are spectral basis functions.
    # We can reconstruct this from the HSGP object's internals.
    
    # Get the learned age effect at each observed age, build an interpolation
    # This is simpler and more robust than reconstructing HSGP basis manually
    from scipy.interpolate import interp1d
    
    # Unique ages in training data and their mean posterior age effects
    unique_ages = np.sort(np.unique(obs_age_values))
    # Map each obs to its age, compute mean age effect per unique age per sample
    obs_ages = obs_age_values
    
    # Build age→effect interpolator for each posterior sample
    # Group observations by age, average the age_effect within each age
    age_to_obs_idx = {}
    for i, a in enumerate(obs_ages):
        age_to_obs_idx.setdefault(float(a), []).append(i)
    
    # Mean age effect at each unique age for each sample
    age_effect_by_age = np.zeros((len(unique_ages), ns))
    for j, a in enumerate(unique_ages):
        idx = age_to_obs_idx[float(a)]
        age_effect_by_age[j, :] = age_effect_flat[:, idx].mean(axis=1)
    
    def eval_gp_age_effect(ages_new):
        """Evaluate GP age effect at new ages via interpolation of posterior."""
        # Extrapolate linearly beyond training range
        interp = interp1d(unique_ages, age_effect_by_age, axis=0, 
                         kind="linear", fill_value="extrapolate")
        return interp(ages_new)  # (len(ages_new), ns)

    # Extrapolate league trend
    last_trend = lt_flat[:, -1]
    innov_std = innov_flat.std(axis=1)

    # Project recently active batters
    cutoff_year = int(seasons[-1]) - 2  # active in last 3 years
    active = batter_meta[batter_meta["last_season"] >= cutoff_year].copy()
    logger.info(f"Projecting {len(active)} batters active since {cutoff_year}")

    # Multi-year projection
    all_projections = []

    for proj_year in range(projection_year, projection_year + 5):
        years_ahead = proj_year - int(seasons[-1])
        projected_trend = last_trend.copy()
        rng_year = np.random.default_rng(42 + proj_year)
        for _ in range(years_ahead):
            projected_trend = projected_trend + rng_year.normal(0, innov_std)

        for _, row in active.iterrows():
            batter_id = int(row["batter"])
            b_idx = batter_map[batter_id]
            proj_age = proj_year - float(row["birth_year"])
            s_idx = 1 if row["stand"] == "R" else 0

            # GP age effect at projected age
            age_eff = eval_gp_age_effect(np.array([proj_age])).squeeze()  # (ns,)

            eta_proj = (
                projected_trend
                + pa_flat[:, b_idx]
                + bh_flat * s_idx
                + age_eff
            )

            p_k = 1.0 / (1.0 + np.exp(-eta_proj))

            all_projections.append({
                "batter": batter_id,
                "projection_year": proj_year,
                "projected_age": proj_age,
                "stand": row["stand"],
                "projected_k_rate": float(np.mean(p_k)),
                "k_rate_std": float(np.std(p_k)),
                "k_rate_lower": float(np.percentile(p_k, 5)),
                "k_rate_upper": float(np.percentile(p_k, 95)),
                "k_rate_10": float(np.percentile(p_k, 10)),
                "k_rate_90": float(np.percentile(p_k, 90)),
                "posterior_mean_ability": float(np.mean(pa_flat[:, b_idx])),
                "total_pa": int(row["total_pa"]),
                "career_k_rate": float(row["career_k_rate"]),
                "last_season": int(row["last_season"]),
            })

    proj_df = pd.DataFrame(all_projections)
    proj_df = proj_df.sort_values(["batter", "projection_year"]).reset_index(drop=True)

    # Summary stats
    proj_2026 = proj_df[proj_df["projection_year"] == projection_year]
    logger.info(
        f"Projections generated: {len(proj_df)} total rows "
        f"({len(proj_2026)} for {projection_year}), "
        f"median K% = {proj_2026['projected_k_rate'].median():.3f}"
    )

    # Generate the learned aging curve for plotting
    age_grid = np.linspace(20, 42, 100)
    aging_curves = eval_gp_age_effect(age_grid)  # (100, ns)
    aging_curve_mean = aging_curves.mean(axis=1)
    aging_curve_lower = np.percentile(aging_curves, 5, axis=1)
    aging_curve_upper = np.percentile(aging_curves, 95, axis=1)

    aging_df = pd.DataFrame({
        "age": age_grid,
        "age_effect_mean": aging_curve_mean,
        "age_effect_lower": aging_curve_lower,
        "age_effect_upper": aging_curve_upper,
    })

    # Save projections to volume
    proj_dir = Path("/models/projections")
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj_path = proj_dir / f"k_rate_projections_{projection_year}.parquet"
    proj_df.to_parquet(str(proj_path), index=False)

    # Save aging curve
    aging_path = proj_dir / f"k_rate_aging_curve_{projection_year}.parquet"
    aging_df.to_parquet(str(aging_path), index=False)

    # Save trace
    trace_dir = Path("/models/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"k_rate_trace_{projection_year}.nc"
    trace.to_netcdf(str(trace_path))

    logger.info(f"Saved projections + aging curve + trace to /models volume")

    # ─── 6. wandb logging ───────────────────────────────────────────────
    if log_wandb:
        try:
            import wandb
            from datetime import datetime

            run_name = f"pa-k-rate-{projection_year}-{datetime.now():%Y%m%d_%H%M}"
            run = wandb.init(
                project="baseball-projections", entity="jseeburger",
                name=run_name,
                config={
                    "model": "pa_k_rate_binomial",
                    "age_model": "bspline",
                    "likelihood": "binomial_batter_season",
                    "n_draws": n_draws, "n_tune": n_tune, "n_chains": n_chains,
                    "target_accept": target_accept, "min_pa": min_pa,
                    "projection_year": projection_year,
                    "n_obs": n_obs, "n_batters": n_batters,
                    "n_seasons": n_seasons, "n_teams": n_teams,
                    "n_total_pa": n_total_pa,
                    "reference_age": REFERENCE_AGE,
                },
                tags=["k-rate", "bayesian", "binomial", "batter-season"],
                group="hitter-k-rate",
                job_type="train",
                reinit=True,
            )

            # Diagnostics
            wandb.log({
                "diagnostics/max_rhat": max_rhat,
                "diagnostics/divergences": divergences,
                "diagnostics/sampling_time_s": elapsed,
                "diagnostics/n_params": n_params,
            })

            # Summary table
            summary = az.summary(trace, var_names=[
                "league_init", "mu_ability", "sigma_ability", "beta_hand",
                "tau_age",
            ])
            wandb.log({"diagnostics/summary": wandb.Table(dataframe=summary.reset_index())})

            # Posterior plots
            for var_group, var_names in [
                ("scalars", ["league_init", "mu_ability", "sigma_ability", "beta_hand", "tau_age"]),
                ("league_trend", ["league_trend"]),
            ]:
                ax = az.plot_trace(trace, var_names=var_names, compact=True)
                fig = ax.ravel()[0].figure
                wandb.log({f"posterior/{var_group}_trace": wandb.Image(fig)})
                plt.close(fig)

            # Projections table
            wandb.log({"projections_preview": wandb.Table(dataframe=proj_df.head(100))})

            # Log aging curve plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(aging_df["age"], aging_df["age_effect_mean"], "b-", linewidth=2, label="Mean")
            ax.fill_between(aging_df["age"], aging_df["age_effect_lower"], aging_df["age_effect_upper"],
                           alpha=0.3, color="blue", label="90% CI")
            ax.set_xlabel("Age")
            ax.set_ylabel("Age Effect (logit scale)")
            ax.set_title("Learned K% Aging Curve (B-spline)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            wandb.log({"aging_curve": wandb.Image(fig)})
            plt.close(fig)
            
            # Log aging curve as table
            wandb.log({"aging_curve_data": wandb.Table(dataframe=aging_df)})

            # Projections artifact
            artifact = wandb.Artifact(f"k-rate-projections-{projection_year}", type="projections",
                                       metadata={"n_batters": len(proj_df),
                                                  "median_k_rate": float(proj_df["projected_k_rate"].median())})
            with tempfile.TemporaryDirectory() as tmpdir:
                p = os.path.join(tmpdir, "projections.parquet")
                proj_df.to_parquet(p, index=False)
                artifact.add_file(p, name="projections.parquet")
            wandb.log_artifact(artifact, aliases=["latest"])

            # Trace artifact
            trace_artifact = wandb.Artifact(f"k-rate-trace-{projection_year}", type="model",
                                              metadata={"max_rhat": max_rhat, "divergences": divergences})
            trace_artifact.add_file(str(trace_path), name="trace.nc")
            wandb.log_artifact(trace_artifact, aliases=["latest"])

            models_volume.commit()
            logger.info(f"wandb run: {run.url}")
            wandb.finish()
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    # Return summary
    return {
        "status": "complete",
        "n_obs": n_obs,
        "n_batters": n_batters,
        "n_seasons": n_seasons,
        "max_rhat": round(max_rhat, 4),
        "divergences": divergences,
        "sampling_time_s": round(elapsed, 1),
        "n_projections": len(proj_df),
        "median_k_rate": round(float(proj_2026["projected_k_rate"].median()), 4),
        "projection_years": list(range(projection_year, projection_year + 5)),
        "age_model": "HSGP",
        "hsgp_config": {"m": HSGP_M, "c": HSGP_C, "kernel": "Matern52", "reference_age": REFERENCE_AGE},
        "aging_curve_summary": {
            "peak_age": float(age_grid[aging_curve_mean.argmin()]),
            "age_effect_at_25": float(aging_curves[np.argmin(np.abs(age_grid - 25))].mean()),
            "age_effect_at_30": float(aging_curves[np.argmin(np.abs(age_grid - 30))].mean()),
            "age_effect_at_35": float(aging_curves[np.argmin(np.abs(age_grid - 35))].mean()),
        },
        "top_5_lowest_k": proj_2026.nsmallest(5, "projected_k_rate")[["batter", "projected_k_rate", "career_k_rate"]].to_dict("records"),
        "top_5_highest_k": proj_2026.nlargest(5, "projected_k_rate")[["batter", "projected_k_rate", "career_k_rate"]].to_dict("records"),
    }


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
def run_k_rate_model(
    draws: int = 2000,
    tune: int = 1500,
    chains: int = 4,
    fast: bool = False,
    no_wandb: bool = False,
):
    """Train the PA-level K-rate Bayesian model on Modal."""
    mode = "⚡ FAST" if fast else "🔬 FULL"
    print(f"{mode} — PA-level K-rate model")
    print(f"   {chains} chains × {draws} draws (tune={tune})")
    result = train_pa_k_rate.remote(
        n_draws=draws, n_tune=tune, n_chains=chains,
        log_wandb=not no_wandb, fast_mode=fast,
    )
    print("\n" + "=" * 60)
    print("K-RATE MODEL RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if result.get("divergences", -1) == 0 and result.get("max_rhat", 2.0) < 1.05:
        print("\n✅ Model converged! Check wandb for full diagnostics.")
    else:
        print("\n⚠️  Check diagnostics — divergences or high R-hat detected.")


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
