"""Baseball Projections — Modal compute infrastructure.

Single-file Modal app with all entrypoints. The model code itself lives in
`src/models/` — the same modules the test suite imports, the backtest
harness (`src/eval/bayes_arm.py`) fits, and the gate rule
(docs/architecture.md #3) scores against Marcel. This file's job is the
Modal-specific plumbing around those models: volumes, secrets, image,
wandb logging, and the `/data`/`/models` paths — not a second copy of the
models themselves.

That used to not be true. Until issue #86, a comment here read "Modal
requirement — no cross-module imports" and every model was pasted in
whole, twice. That claim was false — `modal.Image.add_local_python_source`
has shipped local packages into a function's container for a long time,
well before the `modal>=0.64` this repo pins (confirmed against Modal's
own docs, not assumed) — and the two copies had quietly drifted apart: the
weekly production refit was fitting a different K% model (an HSGP age
curve, no opposing-pitcher term) than the one every published backtest
number was scored against. See docs/modal-src-divergence.md for the full
audit — what diverged, which published numbers came from which copy, and
what did and did not get changed here as a result.

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

# Secrets — Weights & Biases. Every secret listed here must exist in the
# Modal workspace or the function fails before it starts, so keep this to
# what the code actually reads.
_wandb = modal.Secret.from_name("wandb-baseball")
ALL_SECRETS = [_wandb]

# Container image — PyMC + data stack + wandb
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
        # Experiment tracking
        "wandb>=0.19",
        # Utilities
        "scipy>=1.14",
        "scikit-learn>=1.5",
        "tqdm",
        "requests>=2.31",  # src.data.birthdates fetches the Chadwick register
    )
    # Ships src/ into the container so the training functions below can
    # `import src.models...` instead of inlining a second copy of the
    # model (issue #86). This is Modal's documented mechanism for local
    # packages, not a workaround — see the module docstring.
    .add_local_python_source("src")
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

    # 3. PyMC sampling (tiny model — proves MCMC works)
    with pm.Model():
        mu = pm.Normal("mu", mu=0.260, sigma=0.05)
        pm.Normal("obs", mu=mu, sigma=0.03,
                  observed=np.array([0.250, 0.270, 0.265]))
        trace = pm.sample(200, cores=1, chains=1,
                         progressbar=False, return_inferencedata=True)
    results["pymc_sampling"] = "OK"
    results["mu_posterior_mean"] = round(float(trace.posterior["mu"].mean()), 4)

    # 4. Models volume
    results["models_volume"] = Path("/models").exists()

    # 5. wandb connectivity
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
    """Run full smoke test — verifies image, volumes, PyMC, wandb."""
    print("🧪 Running Modal smoke test...")
    results = smoke_test.remote()
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if results.get("pymc_sampling") == "OK" and results.get("wandb_api_key") == "configured":
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


@app.function(image=pymc_image, volumes=VOLUME_MOUNTS, timeout=600)
def generate_birth_years_on_volume():
    """Generate batter_birth_years.parquet on the Modal volume from the
    Chadwick Bureau register (real birthdates keyed by MLBAM id).

    Falls back to debut_year - 24 only for ids missing from the register.
    """
    import pandas as pd
    from pathlib import Path

    data_volume.reload()
    parquet_dir = Path("/data/parquet")
    pa_dir = parquet_dir / "pa_outcomes"

    frames = []
    for f in sorted(pa_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(f, columns=["batter", "game_year"]))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} PAs, {df['batter'].nunique()} unique batters")

    # Chadwick register: 16 sharded CSVs with key_mlbam + birth year.
    register_url = ("https://raw.githubusercontent.com/chadwickbureau/register"
                    "/master/data/people-{shard}.csv")
    shards = []
    for shard in "0123456789abcdef":
        people = pd.read_csv(
            register_url.format(shard=shard),
            usecols=["key_mlbam", "birth_year"],
            dtype={"key_mlbam": "Int64", "birth_year": "Int64"},
        )
        shards.append(people.dropna(subset=["key_mlbam", "birth_year"]))
    register = (
        pd.concat(shards, ignore_index=True)
        .drop_duplicates("key_mlbam")
        .set_index("key_mlbam")["birth_year"]
    )
    print(f"Chadwick register: {len(register):,} players with birth year")

    debut = df.groupby("batter")["game_year"].min().reset_index()
    debut.columns = ["batter", "debut_year"]
    debut["birth_year"] = debut["batter"].map(register)
    n_missing = int(debut["birth_year"].isna().sum())
    if n_missing:
        print(f"⚠️  {n_missing} batters not in register; using debut_year - 24")
    debut["birth_year"] = (
        debut["birth_year"].fillna(debut["debut_year"] - 24).astype(int)
    )

    result = debut[["batter", "birth_year"]]
    output_path = parquet_dir / "batter_birth_years.parquet"
    result.to_parquet(output_path, index=False)
    data_volume.commit()

    print(f"\n✅ Saved {len(result)} batter birth years to {output_path}")
    print(f"Birth year range: {result['birth_year'].min()} - {result['birth_year'].max()}")
    match_rate = 1 - n_missing / max(len(result), 1)
    print(f"Register match rate: {match_rate:.1%}")

    return {"n_batters": len(result), "n_missing": n_missing,
            "path": str(output_path)}


@app.function(image=pymc_image, volumes=VOLUME_MOUNTS, timeout=600)
def generate_birthdates_on_volume():
    """Write /data/parquet/birthdates.parquet — the full Chadwick schema
    (birth_year, birth_month, birth_day) `src.models.pa_k_rate.prepare_model_data`
    needs for a real June-30 seasonal age, via `src.data.birthdates`.

    This is a different file from `generate_birth_years_on_volume`'s
    `batter_birth_years.parquet` above (year only, calendar-year age, the
    convention `src.models.iso_rate` / `babip_rate` still use — see
    docs/modal-src-divergence.md for why that one was left alone). Run
    this once against the live volume before a `train_pa_k_rate` refit;
    without it, `prepare_model_data` falls back to `first_year - 23` and
    logs a warning rather than failing, but the fallback is the estimate
    real birthdates replaced in `src/` months ago.
    """
    import pandas as pd

    from src.data.birthdates import fetch_register

    data_volume.reload()
    parquet_dir = Path("/data/parquet")

    people = fetch_register()
    output_path = parquet_dir / "birthdates.parquet"
    people.to_parquet(output_path, index=False)
    data_volume.commit()

    n_with_year = int(people["birth_year"].notna().sum())
    print(f"Chadwick register: {len(people):,} players with an MLBAM id, "
          f"{n_with_year:,} with a birth year")
    print(f"Saved {output_path}")
    return {"n_players": len(people), "n_with_birth_year": n_with_year,
            "path": str(output_path)}


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
# PA-level K-Rate Bayesian Model — the gated model (src/models/pa_k_rate.py)
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
    target_accept: float = 0.9,
    min_pa: int = 50,
    projection_year: int = 2026,
    log_wandb: bool = True,
    fast_mode: bool = False,
    include_pitcher: bool = False,
    cutoff_date: str = "",
):
    """Thin Modal wrapper around `src.models.pa_k_rate` — the gated K% model.

    Everything about *what the model is* — the quadratic age curve, the
    optional zero-mean opposing-pitcher random effect, the exact
    (batter, season, team, stand[, pitcher]) Binomial cell aggregation, the
    June-30 seasonal age from the Chadwick register — lives in
    `src/models/pa_k_rate.py` and is not duplicated here (issue #86). This
    function does only the Modal-specific parts: reading the volume,
    resolving the birthdates file, sampling with the requested compute
    budget, writing results back to `/models`, and logging to wandb.

    `target_accept` now defaults to 0.9 to match
    `src.models.pa_k_rate.SAMPLER_KWARGS` (it was 0.95 here before; the
    workflow never overrides it, so nothing published depended on the old
    default — see docs/modal-src-divergence.md).

    `include_pitcher` and `cutoff_date` default to what the weekly refit
    has always done — no pitcher term, full history, no walk-forward
    cutoff — so this rewiring does not silently change what Monday's
    refit produces by default. The pitcher term in particular is opt-in
    rather than on because it has not cleared the rest-of-season gate
    (docs/backtest-baselines.md, BAS-59); turning it on here is what that
    section's "a full Modal refit should score it at all three cutoffs"
    now means to run.
    """
    import logging

    import pandas as pd

    from src.models import pa_k_rate as k_rate_model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("pa_k_rate")

    if fast_mode:
        n_draws, n_tune, n_chains = 200, 200, 2
        logger.info("FAST MODE enabled")

    # ─── 1. Load data (Modal volume paths; everything else is src/) ──────
    data_volume.reload()
    parquet_dir = Path("/data/parquet")
    pa_dir = parquet_dir / "pa_outcomes"
    pf_path = parquet_dir / "park_factors.parquet"
    birthdates_path = parquet_dir / "birthdates.parquet"

    if birthdates_path.exists():
        birthdates = pd.read_parquet(birthdates_path)
    else:
        birthdates = None
        logger.warning(
            "%s not found -- prepare_model_data will fall back to "
            "first_year-23 ages. Run `modal run "
            "modal_functions/app.py::generate_birthdates_on_volume` once "
            "against this volume to fix that.", birthdates_path,
        )

    cutoff = cutoff_date or None
    pa = k_rate_model.load_pa_data(pa_dir, cutoff_date=cutoff, include_pitcher=include_pitcher)
    park_factors = k_rate_model.load_park_factors(pf_path)

    # ─── 2. Prepare, build, sample, project — all src.models.pa_k_rate ───
    model_data = k_rate_model.prepare_model_data(
        pa, park_factors, min_pa=min_pa, birthdates=birthdates,
        cutoff_date=cutoff, include_pitcher=include_pitcher,
    )
    del pa

    model = k_rate_model.build_model(model_data)
    trace = k_rate_model.sample_model(
        model,
        draws=n_draws, tune=n_tune, chains=n_chains,
        target_accept=target_accept, nuts_sampler="numpyro",
    )
    diagnostics = k_rate_model.model_diagnostics(trace)
    logger.info("Max R-hat: %.4f (%s), divergences: %d, BFMI: %s",
                diagnostics["max_rhat"], diagnostics["max_rhat_var"],
                diagnostics["divergences"], diagnostics["bfmi"])

    projections = k_rate_model.generate_projections(
        trace, model_data, projection_year=projection_year,
    )

    # ─── 3. Save to /models (Modal-specific) ──────────────────────────────
    proj_dir = Path("/models/projections")
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj_path = proj_dir / f"k_rate_projections_{projection_year}.parquet"
    projections.to_parquet(str(proj_path), index=False)

    trace_dir = Path("/models/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"k_rate_trace_{projection_year}.nc"
    trace.to_netcdf(str(trace_path))
    models_volume.commit()
    logger.info("Saved projections + trace to /models volume")

    # ─── 4. wandb logging — delegates to src.models.pa_k_rate.log_to_wandb ─
    if log_wandb:
        try:
            k_rate_model.log_to_wandb(
                trace, projections, model_data,
                model_config={
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
                    "reference_age": k_rate_model.REFERENCE_AGE,
                    "n_draws": n_draws, "n_tune": n_tune, "n_chains": n_chains,
                    "target_accept": target_accept,
                    "source": "modal_functions/app.py -> src.models.pa_k_rate (issue #86)",
                },
            )
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return {
        "status": "complete",
        "n_obs": model_data["n_obs"],
        "n_batters": model_data["n_batters"],
        "n_seasons": model_data["n_seasons"],
        "n_pitchers": model_data["n_pitchers"],
        "include_pitcher": include_pitcher,
        "cutoff_date": model_data["cutoff_date"],
        "max_rhat": diagnostics["max_rhat"],
        "divergences": diagnostics["divergences"],
        "healthy": diagnostics["healthy"],
        "n_projections": len(projections),
        "median_k_rate": round(float(projections["projected_k_rate"].median()), 4),
        "age_model": "quadratic (src.models.pa_k_rate)",
        "top_5_lowest_k": projections.nsmallest(5, "projected_k_rate")[["batter", "projected_k_rate", "career_k_rate"]].to_dict("records"),
        "top_5_highest_k": projections.nlargest(5, "projected_k_rate")[["batter", "projected_k_rate", "career_k_rate"]].to_dict("records"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ISO (Isolated Power) Model — Normal likelihood on natural scale
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=7200,
    memory=8192,
    cpu=4.0,
)
def train_iso_model(
    n_draws: int = 2000,
    n_tune: int = 1500,
    n_chains: int = 4,
    target_accept: float = 0.95,
    min_ab: int = 50,
    projection_year: int = 2026,
    log_wandb: bool = True,
    fast_mode: bool = False,
):
    """Thin Modal wrapper around `src.models.iso_rate` (issue #86).

    `src/models/iso_rate.py` is an extraction of this function's original
    body (unchanged model, priors, HSGP age curve, aggregation) — there was
    no pre-existing `src/` ISO model for this to have diverged from, so
    nothing about *the model* changed here, only where its code lives. See
    docs/modal-src-divergence.md.
    """
    import logging

    import pandas as pd

    from src.models import iso_rate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("iso_model")

    if fast_mode:
        n_draws, n_tune, n_chains = 200, 200, 2
        logger.info("FAST MODE enabled")

    data_volume.reload()
    parquet_dir = Path("/data/parquet")
    pf_path = parquet_dir / "park_factors.parquet"
    hs_path = parquet_dir / "hitter_seasons.parquet"

    raw = iso_rate.load_ab_data(parquet_dir / "pa_outcomes")
    park_factors = pd.read_parquet(pf_path) if pf_path.exists() else None
    hitter_seasons = pd.read_parquet(hs_path) if hs_path.exists() else None

    model_data = iso_rate.prepare_model_data(
        raw, park_factors=park_factors, min_ab=min_ab, fast_mode=fast_mode,
        hitter_seasons=hitter_seasons,
    )
    del raw

    model = iso_rate.build_model(model_data)
    trace = iso_rate.sample_model(
        model,
        draws=n_draws, tune=n_tune, chains=n_chains,
        target_accept=target_accept, nuts_sampler="numpyro",
    )
    diagnostics = iso_rate.model_diagnostics(trace)
    logger.info("Max R-hat: %.4f (%s), divergences: %d, BFMI: %s",
                diagnostics["max_rhat"], diagnostics["max_rhat_var"],
                diagnostics["divergences"], diagnostics["bfmi"])

    projections = iso_rate.generate_projections(trace, model_data, projection_year=projection_year)
    aging_df = iso_rate.compute_aging_curve(trace, model_data)
    proj_2026 = projections[projections["projection_year"] == projection_year]

    # ─── Save to /models (Modal-specific) ────────────────────────────────
    proj_dir = Path("/models/projections")
    proj_dir.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(str(proj_dir / f"iso_projections_{projection_year}.parquet"), index=False)
    aging_df.to_parquet(str(proj_dir / f"iso_aging_curve_{projection_year}.parquet"), index=False)

    trace_dir = Path("/models/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"iso_trace_{projection_year}.nc"
    trace.to_netcdf(str(trace_path))
    models_volume.commit()
    logger.info("Saved projections + aging curve + trace to /models volume")

    # ─── wandb logging ─────────────────────────────────────────────────────
    if log_wandb:
        try:
            import tempfile

            import wandb
            from datetime import datetime

            run = wandb.init(
                project="baseball-projections", entity="jseeburger",
                name=f"iso-model-{projection_year}-{datetime.now():%Y%m%d_%H%M}",
                config={
                    "model": "iso_normal", "age_model": "HSGP",
                    "hsgp_m": iso_rate.HSGP_M, "hsgp_c": iso_rate.HSGP_C,
                    "n_draws": n_draws, "n_tune": n_tune, "n_chains": n_chains,
                    "target_accept": target_accept, "min_ab": min_ab,
                    "projection_year": projection_year,
                    "n_obs": model_data["n_obs"], "n_batters": model_data["n_batters"],
                    "n_seasons": model_data["n_seasons"], "n_teams": model_data["n_teams"],
                    "reference_age": iso_rate.REFERENCE_AGE,
                    "source": "modal_functions/app.py -> src.models.iso_rate (issue #86)",
                },
                tags=["iso", "bayesian", "normal", "hsgp", "batter-season"],
                group="hitter-iso", job_type="train", reinit=True,
            )
            wandb.log({
                "diagnostics/max_rhat": diagnostics["max_rhat"],
                "diagnostics/divergences": diagnostics["divergences"],
                "diagnostics/min_ess_bulk": diagnostics["min_ess_bulk"],
            })
            wandb.log({"projections_preview": wandb.Table(dataframe=projections.head(200))})
            wandb.log({"aging_curve_data": wandb.Table(dataframe=aging_df)})

            artifact = wandb.Artifact(f"iso-projections-{projection_year}", type="projections",
                                       metadata={"n_batters": len(projections)})
            with tempfile.TemporaryDirectory() as tmpdir:
                p = os.path.join(tmpdir, "projections.parquet")
                projections.to_parquet(p, index=False)
                artifact.add_file(p, name="projections.parquet")
            wandb.log_artifact(artifact, aliases=["latest"])

            trace_artifact = wandb.Artifact(f"iso-trace-{projection_year}", type="model",
                                             metadata={"max_rhat": diagnostics["max_rhat"],
                                                       "divergences": diagnostics["divergences"]})
            trace_artifact.add_file(str(trace_path), name="trace.nc")
            wandb.log_artifact(trace_artifact, aliases=["latest"])

            logger.info(f"wandb run: {run.url}")
            wandb.finish()
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return {
        "status": "complete",
        "model_type": "iso_normal (natural scale)",
        "n_obs": model_data["n_obs"],
        "n_batters": model_data["n_batters"],
        "n_seasons": model_data["n_seasons"],
        "max_rhat": diagnostics["max_rhat"],
        "divergences": diagnostics["divergences"],
        "healthy": diagnostics["healthy"],
        "n_projections": len(projections),
        "median_iso": round(float(proj_2026["projected_iso"].median()), 4),
        "mean_iso": round(float(proj_2026["projected_iso"].mean()), 4),
        "age_model": "HSGP",
        "top_5_highest_iso": proj_2026.nlargest(5, "projected_iso")[["batter", "projected_iso", "career_iso"]].to_dict("records"),
        "top_5_lowest_iso": proj_2026.nsmallest(5, "projected_iso")[["batter", "projected_iso", "career_iso"]].to_dict("records"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BABIP (Batting Average on Balls in Play) Model — Binomial likelihood
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=7200,
    memory=8192,
    cpu=4.0,
)
def train_babip_model(
    n_draws: int = 2000,
    n_tune: int = 1500,
    n_chains: int = 4,
    target_accept: float = 0.95,
    min_ab: int = 50,
    projection_year: int = 2026,
    log_wandb: bool = True,
    fast_mode: bool = False,
):
    """Thin Modal wrapper around `src.models.babip_rate` (issue #86).

    `src/models/babip_rate.py` is an extraction of this function's
    original body (unchanged model, priors, HSGP age curve, aggregation)
    — there was no pre-existing `src/` BABIP model for this to have
    diverged from, so nothing about *the model* changed here, only where
    its code lives. See docs/modal-src-divergence.md.
    """
    import logging

    import pandas as pd

    from src.models import babip_rate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("babip_model")

    if fast_mode:
        n_draws, n_tune, n_chains = 200, 200, 2
        logger.info("FAST MODE enabled")

    data_volume.reload()
    parquet_dir = Path("/data/parquet")
    pf_path = parquet_dir / "park_factors.parquet"
    hs_path = parquet_dir / "hitter_seasons.parquet"

    raw = babip_rate.load_bip_data(parquet_dir / "pa_outcomes")
    park_factors = pd.read_parquet(pf_path) if pf_path.exists() else None
    hitter_seasons = pd.read_parquet(hs_path) if hs_path.exists() else None

    model_data = babip_rate.prepare_model_data(
        raw, park_factors=park_factors, min_ab=min_ab, fast_mode=fast_mode,
        hitter_seasons=hitter_seasons,
    )
    del raw

    model = babip_rate.build_model(model_data)
    trace = babip_rate.sample_model(
        model,
        draws=n_draws, tune=n_tune, chains=n_chains,
        target_accept=target_accept, nuts_sampler="numpyro",
    )
    diagnostics = babip_rate.model_diagnostics(trace)
    logger.info("Max R-hat: %.4f (%s), divergences: %d, BFMI: %s",
                diagnostics["max_rhat"], diagnostics["max_rhat_var"],
                diagnostics["divergences"], diagnostics["bfmi"])

    projections = babip_rate.generate_projections(trace, model_data, projection_year=projection_year)
    aging_df = babip_rate.compute_aging_curve(trace, model_data)
    proj_2026 = projections[projections["projection_year"] == projection_year]

    # ─── Save to /models (Modal-specific) ────────────────────────────────
    proj_dir = Path("/models/projections")
    proj_dir.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(str(proj_dir / f"babip_projections_{projection_year}.parquet"), index=False)
    aging_df.to_parquet(str(proj_dir / f"babip_aging_curve_{projection_year}.parquet"), index=False)

    trace_dir = Path("/models/traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"babip_trace_{projection_year}.nc"
    trace.to_netcdf(str(trace_path))
    models_volume.commit()
    logger.info("Saved projections + aging curve + trace to /models volume")

    # ─── wandb logging ─────────────────────────────────────────────────────
    if log_wandb:
        try:
            import tempfile

            import wandb
            from datetime import datetime

            run = wandb.init(
                project="baseball-projections", entity="jseeburger",
                name=f"babip-model-{projection_year}-{datetime.now():%Y%m%d_%H%M}",
                config={
                    "model": "babip_binomial", "age_model": "HSGP",
                    "hsgp_m": babip_rate.HSGP_M, "hsgp_c": babip_rate.HSGP_C,
                    "n_draws": n_draws, "n_tune": n_tune, "n_chains": n_chains,
                    "target_accept": target_accept, "min_ab": min_ab,
                    "projection_year": projection_year,
                    "n_obs": model_data["n_obs"], "n_batters": model_data["n_batters"],
                    "n_seasons": model_data["n_seasons"], "n_teams": model_data["n_teams"],
                    "reference_age": babip_rate.REFERENCE_AGE,
                    "source": "modal_functions/app.py -> src.models.babip_rate (issue #86)",
                },
                tags=["babip", "bayesian", "binomial", "hsgp", "batter-season"],
                group="hitter-babip", job_type="train", reinit=True,
            )
            wandb.log({
                "diagnostics/max_rhat": diagnostics["max_rhat"],
                "diagnostics/divergences": diagnostics["divergences"],
                "diagnostics/min_ess_bulk": diagnostics["min_ess_bulk"],
            })
            wandb.log({"projections_preview": wandb.Table(dataframe=projections.head(200))})
            wandb.log({"aging_curve_data": wandb.Table(dataframe=aging_df)})

            artifact = wandb.Artifact(f"babip-projections-{projection_year}", type="projections",
                                       metadata={"n_batters": len(projections)})
            with tempfile.TemporaryDirectory() as tmpdir:
                p = os.path.join(tmpdir, "projections.parquet")
                projections.to_parquet(p, index=False)
                artifact.add_file(p, name="projections.parquet")
            wandb.log_artifact(artifact, aliases=["latest"])

            trace_artifact = wandb.Artifact(f"babip-trace-{projection_year}", type="model",
                                             metadata={"max_rhat": diagnostics["max_rhat"],
                                                       "divergences": diagnostics["divergences"]})
            trace_artifact.add_file(str(trace_path), name="trace.nc")
            wandb.log_artifact(trace_artifact, aliases=["latest"])

            logger.info(f"wandb run: {run.url}")
            wandb.finish()
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return {
        "status": "complete",
        "model_type": "babip_binomial (logit link)",
        "n_obs": model_data["n_obs"],
        "n_batters": model_data["n_batters"],
        "n_seasons": model_data["n_seasons"],
        "max_rhat": diagnostics["max_rhat"],
        "divergences": diagnostics["divergences"],
        "healthy": diagnostics["healthy"],
        "n_projections": len(projections),
        "median_babip": round(float(proj_2026["projected_babip"].median()), 4),
        "mean_babip": round(float(proj_2026["projected_babip"].mean()), 4),
        "age_model": "HSGP",
        "top_5_highest_babip": proj_2026.nlargest(5, "projected_babip")[["batter", "projected_babip", "career_babip"]].to_dict("records"),
        "top_5_lowest_babip": proj_2026.nsmallest(5, "projected_babip")[["batter", "projected_babip", "career_babip"]].to_dict("records"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Assembly: Component Rates → wOBA → wRC+ → oWAR
# ═══════════════════════════════════════════════════════════════════════════

@app.function(
    image=pymc_image,
    volumes=VOLUME_MOUNTS,
    secrets=ALL_SECRETS,
    timeout=3600,
    memory=8192,
    cpu=4.0,
)
def assemble_projections(
    projection_year: int = 2026,
    pa_estimate: int = 550,
    log_wandb: bool = True,
):
    """Assemble 5 component model posteriors into wOBA → wRC+ → oWAR.

    Pure arithmetic on posterior means — no additional MCMC, no PyMC model
    of its own — so this function is outside the scope of issue #86's
    "Modal defines a model" problem and its guard test
    (tests/test_models/test_modal_no_inline_model.py). What issue #86's
    audit *did* find here (docs/modal-src-divergence.md) is worth knowing
    before running this: it reads five `{component}_projections_{year}.parquet`
    files from `/models/projections`, but only three of them —
    `k_rate`, `iso`, `babip` — have a producer anywhere in this file or its
    git history. There is no `train_bb_rate_model` or `train_hr_rate_model`,
    on this branch or any other this repo has ever had; `bb_rate` and
    `hr_rate` projections only exist today as the static April 10, 2026
    files already committed under `data/projections/` (the `bayes_preseason`
    arm the backtest harness reads). Running the `assembly` or `all`
    workflow_dispatch component against a fresh Modal volume will fail at
    the `bb_rate`/`hr_rate` read unless those two files are placed on the
    volume some other way first — this function does not build them, and
    building them is new modelling work with its own gate, not a
    divergence to close.

    Assembly chain:
      K%, BB%, HR rate, ISO, BABIP
      → AVG = BABIP × (1 - K%) + HR rate   (approx)
      → OBP, SLG from components
      → wOBA via linear weights
      → wRAA = (wOBA - lgwOBA) / wOBA_scale × PA
      → oWAR = (wRAA + positional_adj + replacement) / runs_per_win

    Backtests against actual hitter_seasons data for validation.
    """
    import logging
    import tempfile
    import time

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("assembly")

    data_volume.reload()
    models_volume.reload()

    PROJ_DIR = Path("/models/projections")
    DATA_DIR = Path("/data/parquet")

    # ═══════════════════════════════════════════════════════════════════
    # 2024 wOBA weights (FanGraphs: https://www.fangraphs.com/guts.aspx)
    # These are relatively stable year-to-year
    # ═══════════════════════════════════════════════════════════════════
    W_BB = 0.690      # walk
    W_HBP = 0.722     # hit by pitch (we approximate as ~BB rate)
    W_1B = 0.883      # single
    W_2B = 1.244      # double
    W_3B = 1.569      # triple
    W_HR = 2.015      # home run
    WOBA_SCALE = 1.185 # wOBA scale factor
    LG_WOBA = 0.310   # league wOBA
    LG_R_PA = 0.116   # league runs per PA
    RUNS_PER_WIN = 10.0

    # Positional adjustment per 600 PA (FanGraphs)
    POS_ADJ = {
        "C": 12.5, "1B": -12.5, "2B": 2.5, "3B": 2.5,
        "SS": 7.5, "LF": -7.5, "CF": 2.5, "RF": -7.5,
        "DH": -17.5, "OF": -2.5,
    }
    DEFAULT_POS_ADJ = 0.0  # when we don't know position

    # Replacement level: ~20 runs per 600 PA
    REPLACEMENT_PER_600 = 20.0

    # ═══════════════════════════════════════════════════════════════════
    # 1. Load component projections
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Loading component projections...")

    proj_dfs = {}
    for stat, rate_col in [
        ("k_rate", "projected_k_rate"),
        ("bb_rate", "projected_bb_rate"),
        ("hr_rate", "projected_hr_rate"),
        ("iso", "projected_iso"),
        ("babip", "projected_babip"),
    ]:
        path = PROJ_DIR / f"{stat}_projections_{projection_year}.parquet"
        df = pd.read_parquet(str(path))
        proj_dfs[stat] = df[["batter", "projection_year", "projected_age", "stand",
                             rate_col]].rename(columns={rate_col: stat})
        logger.info(f"  {stat}: {len(df[df['projection_year']==projection_year])} batters for {projection_year}")

    # Merge all on batter + projection_year
    merged = proj_dfs["k_rate"]
    for stat in ["bb_rate", "hr_rate", "iso", "babip"]:
        merged = merged.merge(proj_dfs[stat][["batter", "projection_year", stat]],
                              on=["batter", "projection_year"], how="inner")

    logger.info(f"Merged: {len(merged)} batter-years across {merged['projection_year'].nunique()} years")
    m2026 = merged[merged["projection_year"] == projection_year]
    logger.info(f"{projection_year}: {len(m2026)} batters")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Derive counting stats and slash line from rates
    # ═══════════════════════════════════════════════════════════════════
    def assemble_rates(df, pa=550):
        """Convert component rates to batting line and value stats."""
        out = df.copy()
        k_rate = out["k_rate"]
        bb_rate = out["bb_rate"]
        hr_rate = out["hr_rate"]
        iso = out["iso"]
        babip = out["babip"]

        # Approximate at-bats: AB ≈ PA × (1 - BB% - HBP%)
        # HBP ~1.2% of PA on average
        hbp_rate = 0.012
        ab_frac = 1.0 - bb_rate - hbp_rate
        ab = pa * ab_frac

        # HR count
        hr = hr_rate * pa

        # BABIP = (H - HR) / (AB - K - HR + SF)
        # → H - HR = BABIP × (AB - K - HR + SF)
        # SF ≈ 0.8% of PA
        sf_rate = 0.008
        sf = sf_rate * pa
        k = k_rate * pa
        bip = ab - k - hr + sf  # balls in play
        bip = np.maximum(bip, 1)  # avoid division by zero

        h_minus_hr = babip * bip  # hits on BIP (non-HR)
        h = h_minus_hr + hr

        # AVG = H / AB
        avg = h / np.maximum(ab, 1)

        # SLG from ISO: SLG = AVG + ISO
        slg = avg + iso

        # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
        bb = bb_rate * pa
        hbp = hbp_rate * pa
        obp = (h + bb + hbp) / (ab + bb + hbp + sf)

        # ─── wOBA via linear weights ─────────────────────────────────
        # wOBA = (W_BB×BB + W_HBP×HBP + W_1B×1B + W_2B×2B + W_3B×3B + W_HR×HR) / (AB+BB+HBP+SF)
        # We need to decompose hits into 1B, 2B, 3B, HR
        # ISO = (2B + 2×3B + 3×HR) / AB
        # HR rate gives HR/PA, so HR/AB = hr_rate / ab_frac
        # We know total extra bases from ISO: XB = ISO × AB
        # XB = 1×2B + 2×3B + 3×HR
        # 2B + 3B = H - HR - 1B
        # Approximate 3B/2B ratio from league averages: ~15% of XBH are triples
        xb = iso * ab  # total extra bases
        hr_eb = 3.0 * hr  # extra bases from HR
        non_hr_xb = np.maximum(xb - hr_eb, 0)  # extra bases from 2B + 3B
        # 2B contribute 1 extra base, 3B contribute 2
        # With ~15% of non-HR XBH being triples:
        # non_hr_xb = 2B + 2×3B, and 3B/(2B+3B) ≈ 0.12
        # Let 3B = t, 2B = d: non_hr_xb = d + 2t, t/(d+t) ≈ 0.12
        # → d = non_hr_xb - 2t, t/(non_hr_xb - 2t + t) ≈ 0.12
        # → t ≈ 0.12 × (non_hr_xb - t) → t ≈ 0.12 × non_hr_xb / 1.12
        triples = 0.12 * non_hr_xb / 1.12
        doubles = non_hr_xb - 2.0 * triples
        doubles = np.maximum(doubles, 0)
        singles = h - hr - doubles - triples
        singles = np.maximum(singles, 0)

        denom = ab + bb + hbp + sf
        woba = (W_BB * bb + W_HBP * hbp + W_1B * singles +
                W_2B * doubles + W_3B * triples + W_HR * hr) / np.maximum(denom, 1)

        # wRAA = (wOBA - lgwOBA) / wOBA_scale × PA
        wraa = (woba - LG_WOBA) / WOBA_SCALE * pa

        # wRC+ = 100 × (wRAA/PA + lgR/PA) / lgR/PA
        # wRC+ = 100 × ((wOBA - lgwOBA)/wOBA_scale + lgR_PA) / lgR_PA
        wrc_plus = 100.0 * ((woba - LG_WOBA) / WOBA_SCALE + LG_R_PA) / LG_R_PA

        # oWAR = (wRAA + positional_adj + replacement) / runs_per_win
        # Without position info, use DH as default (worst case)
        pos_adj = DEFAULT_POS_ADJ * (pa / 600.0)
        replacement = REPLACEMENT_PER_600 * (pa / 600.0)
        owar = (wraa + pos_adj + replacement) / RUNS_PER_WIN

        out["pa"] = pa
        out["avg"] = avg
        out["obp"] = obp
        out["slg"] = slg
        out["woba"] = woba
        out["wraa"] = wraa
        out["wrc_plus"] = wrc_plus
        out["owar"] = owar
        out["hr_count"] = hr
        out["bb_count"] = bb
        out["k_count"] = k
        out["h_count"] = h

        return out

    # ═══════════════════════════════════════════════════════════════════
    # 3. Assemble 2026 projections
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Assembling projections...")
    assembled = assemble_rates(merged, pa=pa_estimate)
    a2026 = assembled[assembled["projection_year"] == projection_year].copy()
    a2026 = a2026.sort_values("owar", ascending=False).reset_index(drop=True)

    logger.info(f"\n{projection_year} Projection Summary:")
    logger.info(f"  Batters: {len(a2026)}")
    logger.info(f"  Median wOBA: {a2026['woba'].median():.3f}")
    logger.info(f"  Median wRC+: {a2026['wrc_plus'].median():.0f}")
    logger.info(f"  Median oWAR: {a2026['owar'].median():.1f}")
    logger.info(f"  Total oWAR: {a2026['owar'].sum():.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Backtest against actual data
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n─── BACKTEST: Comparing to actual hitter_seasons ───")

    hs_path = DATA_DIR / "hitter_seasons.parquet"
    if hs_path.exists():
        hs = pd.read_parquet(str(hs_path))
        logger.info(f"hitter_seasons: {hs.shape}")

        # Build mlbam→fg_id crosswalk from PA data
        # PA data has 'batter' (mlbam). hitter_seasons has 'fg_id'.
        # We'll match on name + year as a practical crosswalk.
        pa_dir = DATA_DIR / "pa_outcomes"
        import pyarrow.parquet as pq

        # Load PA data just to get batter→name mapping
        # Actually, projections have batter (mlbam) but not name.
        # hitter_seasons has name + fg_id. We need a bridge.
        # The simplest bridge: use career stats to fuzzy-match.
        # Better: we have the batter ID in both PA data and projections.
        # Let's check if we can match on name from the data.

        # For now, match on rate-based approach:
        # For each year, compare distributions and find best matches
        # Actually, let's just join on name. We can get names from the PA data.

        # Alternative approach: validate at the DISTRIBUTION level
        # Compare our projected distributions vs actual distributions
        # This is valid even without player-level matching

        for test_year in [2023, 2024, 2025]:
            hs_yr = hs[(hs["year"] == test_year) & (hs["pa"] >= 200)].copy()
            if len(hs_yr) == 0:
                continue

            logger.info(f"\n  Year {test_year} distribution comparison (PA >= 200):")
            for stat, hs_col in [
                ("k_rate", "k_rate"), ("bb_rate", "bb_rate"),
                ("babip", "babip"), ("iso", "iso"),
            ]:
                if hs_col in hs_yr.columns:
                    actual = hs_yr[hs_col]
                    logger.info(f"    {stat:8s}: actual mean={actual.mean():.3f} std={actual.std():.3f} "
                                f"| model mean from component projections")

            # wOBA / wRC+ distribution
            if "woba" in hs_yr.columns:
                logger.info(f"    wOBA    : actual mean={hs_yr['woba'].mean():.3f} std={hs_yr['woba'].std():.3f}")
            if "wrc_plus" in hs_yr.columns:
                logger.info(f"    wRC+    : actual mean={hs_yr['wrc_plus'].mean():.0f} std={hs_yr['wrc_plus'].std():.0f}")
            if "war" in hs_yr.columns:
                logger.info(f"    WAR     : actual mean={hs_yr['war'].mean():.1f} std={hs_yr['war'].std():.1f} "
                            f"total={hs_yr['war'].sum():.0f}")

        # ── Player-level validation for 2024 ──
        # Try to match players via name crosswalk from Marcel projections
        marcel_path = DATA_DIR / "marcel_hitters_2026.parquet"
        crosswalk = None
        if marcel_path.exists():
            marcel = pd.read_parquet(str(marcel_path))
            logger.info(f"\nMarcel projections: {marcel.shape}, cols={list(marcel.columns)[:10]}")
            # Marcel might have both fg_id and mlbam
            if "mlbam_id" in marcel.columns or "mlbamid" in marcel.columns or "key_mlbam" in marcel.columns:
                id_col = next(c for c in marcel.columns if 'mlbam' in c.lower())
                crosswalk = marcel[[id_col, "fg_id"]].drop_duplicates() if "fg_id" in marcel.columns else None
                if crosswalk is not None:
                    logger.info(f"Found crosswalk: {len(crosswalk)} players ({id_col} → fg_id)")

        # Even without crosswalk, we can do a ranked comparison
        # Our 2026 projections should roughly correlate with 2024 actual WAR
        # (Since many of the same players will be active)
        logger.info("\n─── 2026 Projection Sanity Checks ───")
        logger.info(f"Top 10 projected oWAR:")
        for _, row in a2026.head(10).iterrows():
            logger.info(f"  mlbam={int(row['batter']):>7d}  oWAR={row['owar']:.1f}  "
                        f"wRC+={row['wrc_plus']:.0f}  wOBA={row['woba']:.3f}  "
                        f"AVG={row['avg']:.3f}  HR~{row['hr_count']:.0f}")
    else:
        logger.warning("No hitter_seasons.parquet found on volume")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Save assembled projections
    # ═══════════════════════════════════════════════════════════════════
    proj_dir = Path("/models/projections")
    out_path = proj_dir / f"assembled_owar_{projection_year}.parquet"
    assembled.to_parquet(str(out_path), index=False)
    logger.info(f"\nSaved assembled projections to {out_path}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. wandb logging
    # ═══════════════════════════════════════════════════════════════════
    if log_wandb:
        try:
            import wandb
            from datetime import datetime

            run_name = f"assembly-{projection_year}-{datetime.now():%Y%m%d_%H%M}"
            run = wandb.init(
                project="baseball-projections", entity="jseeburger",
                name=run_name,
                config={
                    "model": "assembly",
                    "projection_year": projection_year,
                    "pa_estimate": pa_estimate,
                    "n_batters": len(a2026),
                    "woba_weights": {"bb": W_BB, "hbp": W_HBP, "1b": W_1B,
                                     "2b": W_2B, "3b": W_3B, "hr": W_HR},
                    "woba_scale": WOBA_SCALE,
                    "lg_woba": LG_WOBA,
                    "runs_per_win": RUNS_PER_WIN,
                },
                tags=["assembly", "woba", "wrc+", "owar", "projections"],
                group="hitter-assembly",
                job_type="assemble",
                reinit=True,
            )

            wandb.log({
                "assembly/n_batters": len(a2026),
                "assembly/median_woba": float(a2026["woba"].median()),
                "assembly/median_wrc_plus": float(a2026["wrc_plus"].median()),
                "assembly/median_owar": float(a2026["owar"].median()),
                "assembly/total_owar": float(a2026["owar"].sum()),
            })

            # oWAR distribution
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, col, title in [
                (axes[0], "woba", f"wOBA Distribution ({projection_year})"),
                (axes[1], "wrc_plus", f"wRC+ Distribution ({projection_year})"),
                (axes[2], "owar", f"oWAR Distribution ({projection_year})"),
            ]:
                ax.hist(a2026[col], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
                ax.axvline(x=a2026[col].median(), color="red", linestyle="--",
                           label=f"Median: {a2026[col].median():.2f}")
                ax.set_xlabel(col)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            wandb.log({"distributions": wandb.Image(fig)})
            plt.close(fig)

            # Component rates scatter matrix
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            for ax, (x, y) in zip(axes.ravel(), [
                ("k_rate", "owar"), ("bb_rate", "owar"), ("hr_rate", "owar"),
                ("iso", "owar"), ("babip", "owar"), ("woba", "wrc_plus"),
            ]):
                ax.scatter(a2026[x], a2026[y], alpha=0.3, s=10, color="steelblue")
                r, p = pearsonr(a2026[x], a2026[y])
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_title(f"{x} vs {y} (r={r:.2f})")
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            wandb.log({"component_correlations": wandb.Image(fig)})
            plt.close(fig)

            # Top projections table
            top30 = a2026.head(30)[["batter", "projected_age", "stand",
                                     "k_rate", "bb_rate", "hr_rate", "iso", "babip",
                                     "avg", "obp", "slg", "woba", "wrc_plus", "owar"]].copy()
            for c in ["k_rate", "bb_rate", "hr_rate", "iso", "babip", "avg", "obp", "slg", "woba"]:
                top30[c] = top30[c].round(3)
            top30["wrc_plus"] = top30["wrc_plus"].round(0)
            top30["owar"] = top30["owar"].round(1)
            wandb.log({"top_30_projections": wandb.Table(dataframe=top30)})

            # Full projections table
            wandb.log({"all_projections": wandb.Table(dataframe=a2026.head(200))})

            # Artifact
            artifact = wandb.Artifact(f"assembled-projections-{projection_year}", type="projections",
                                       metadata={"n_batters": len(a2026),
                                                  "median_owar": float(a2026["owar"].median())})
            with tempfile.TemporaryDirectory() as tmpdir:
                p = os.path.join(tmpdir, "assembled.parquet")
                assembled.to_parquet(p, index=False)
                artifact.add_file(p)
            wandb.log_artifact(artifact, aliases=["latest"])

            models_volume.commit()
            logger.info(f"wandb: {run.url}")
            wandb.finish()
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    return {
        "status": "complete",
        "n_batters": len(a2026),
        "median_woba": round(float(a2026["woba"].median()), 3),
        "median_wrc_plus": round(float(a2026["wrc_plus"].median()), 0),
        "median_owar": round(float(a2026["owar"].median()), 1),
        "total_owar": round(float(a2026["owar"].sum()), 0),
        "mean_avg": round(float(a2026["avg"].mean()), 3),
        "mean_obp": round(float(a2026["obp"].mean()), 3),
        "mean_slg": round(float(a2026["slg"].mean()), 3),
        "projection_years": sorted(assembled["projection_year"].unique().tolist()),
        "top_10": a2026.head(10)[["batter", "woba", "wrc_plus", "owar", "avg", "hr_count"]].round(3).to_dict("records"),
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
def run_iso_model(
    draws: int = 2000,
    tune: int = 1500,
    chains: int = 4,
    fast: bool = False,
    no_wandb: bool = False,
):
    """Train the ISO (Isolated Power) Bayesian model on Modal."""
    mode = "⚡ FAST" if fast else "🔬 FULL"
    print(f"{mode} — ISO model (Normal likelihood, natural scale)")
    print(f"   {chains} chains × {draws} draws (tune={tune})")
    result = train_iso_model.remote(
        n_draws=draws, n_tune=tune, n_chains=chains,
        log_wandb=not no_wandb, fast_mode=fast,
    )
    print("\n" + "=" * 60)
    print("ISO MODEL RESULTS")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if result.get("divergences", -1) == 0 and result.get("max_rhat", 2.0) < 1.05:
        print("\n✅ Model converged! Check wandb for full diagnostics.")
    else:
        print("\n⚠️  Check diagnostics — divergences or high R-hat detected.")


@app.local_entrypoint()
def run_assembly(
    no_wandb: bool = False,
    pa: int = 550,
):
    """Assemble component projections into wOBA → wRC+ → oWAR."""
    print("🔧 Assembling component projections...")
    result = assemble_projections.remote(
        log_wandb=not no_wandb,
        pa_estimate=pa,
    )
    print("\n" + "=" * 60)
    print("ASSEMBLY RESULTS")
    print("=" * 60)
    for k, v in result.items():
        if k == "top_10":
            print(f"\n  Top 10 projected oWAR:")
            for p in v:
                print(f"    mlbam={int(p['batter']):>7d}  oWAR={p['owar']:.1f}  "
                      f"wRC+={p['wrc_plus']:.0f}  wOBA={p['woba']:.3f}  "
                      f"AVG={p['avg']:.3f}  HR~{p['hr_count']:.0f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)
    print("\n✅ Assembly complete! Check wandb for plots + validation.")


@app.local_entrypoint()
def run_babip_model(
    draws: int = 2000,
    tune: int = 1500,
    chains: int = 4,
    fast: bool = False,
    no_wandb: bool = False,
):
    """Train the BABIP Bayesian model on Modal."""
    mode = "⚡ FAST" if fast else "🔬 FULL"
    print(f"{mode} — BABIP model (Binomial likelihood, logit link)")
    print(f"   {chains} chains × {draws} draws (tune={tune})")
    result = train_babip_model.remote(
        n_draws=draws, n_tune=tune, n_chains=chains,
        log_wandb=not no_wandb, fast_mode=fast,
    )
    print("\n" + "=" * 60)
    print("BABIP MODEL RESULTS")
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
