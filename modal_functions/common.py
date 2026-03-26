"""Shared Modal configuration — app, image, volumes, secrets."""
import modal

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("baseball-projections")

# ---------------------------------------------------------------------------
# Volumes
#   - data_volume: parquet files uploaded from VPS
#   - models_volume: serialized PyMC traces / model artifacts
# ---------------------------------------------------------------------------
data_volume = modal.Volume.from_name("baseball-data", create_if_missing=True)
models_volume = modal.Volume.from_name("baseball-models", create_if_missing=True)

VOLUME_MOUNTS = {
    "/data": data_volume,
    "/models": models_volume,
}

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
turso_secret = modal.Secret.from_name("turso-baseball")

# ---------------------------------------------------------------------------
# Container image
#   - CPU-only for now (PyMC uses JAX/NumPy on CPU for MCMC)
#   - GPU can be added later if we move to numpyro GPU backend
# ---------------------------------------------------------------------------
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
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
        "libsql-experimental>=0.0.68",
        # Experiment tracking
        "wandb>=0.19",
        # Utilities
        "scipy>=1.14",
        "scikit-learn>=1.5",
        "tqdm",
    )
)
