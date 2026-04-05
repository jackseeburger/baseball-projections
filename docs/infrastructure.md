# Baseball Projection System — Infrastructure & Architecture

## Overview

This document defines the infrastructure stack for the Bayesian Baseball Projection System.
**Training runs on Modal** (serverless compute), **data lives in Cloudflare R2** (S3-compatible object store),
**DVC tracks data versions** in Git, **DuckDB queries Parquet files** locally and on Modal, and
**dashboards run on Streamlit Cloud**.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    YOU (Browser)                             │
│                                                             │
│   Streamlit Cloud            DuckDB CLI         wandb.ai    │
│   ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
│   │ Projections  │    │ Ad-hoc SQL   │   │ Training run  │  │
│   │ Sim results  │    │ over Parquet │   │ comparison,   │  │
│   │ Comparisons  │    │ files in R2  │   │ MCMC diags    │  │
│   └──────┬───────┘    └──────┬───────┘   └──────┬───────┘  │
└──────────┼───────────────────┼──────────────────┼───────────┘
           │                   │                  │
           ▼                   ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│                   Cloudflare R2 (Object Store)                │
│                                                              │
│   s3://baseball-data/                                        │
│   ├── statcast/           # Pitch-level data (2015-2025)     │
│   │   ├── statcast_2015.parquet                              │
│   │   ├── ...                                                │
│   │   └── statcast_2025.parquet                              │
│   ├── features/           # Engineered features (future)     │
│   ├── models/             # Serialized PyMC traces           │
│   └── projections/        # Model output projections         │
│                                                              │
│   Versioned via DVC (data/.dvc files in Git)                 │
│   Zero egress fees — free to pull from anywhere              │
└──────────────────┬──────────────────┬────────────────────────┘
                   │                  │
        ┌──────────┘                  └──────────┐
        ▼                                        ▼
┌───────────────────┐                 ┌────────────────────────┐
│   Modal (Compute) │                 │   VPS / Hari Seldon    │
│                   │                 │                        │
│ • PyMC training   │                 │ • Data pipelines       │
│ • MCMC sampling   │                 │   (pybaseball → R2)    │
│ • Monte Carlo     │                 │ • Cron jobs            │
│   season sims     │                 │ • DVC push/pull        │
│ • CloudBucketMount│                 │ • Light scripting      │
│   streams from R2 │                 │ • DuckDB queries       │
│                   │                 │                        │
│ Reads data from   │                 │ Writes raw data to R2  │
│ R2, writes results│                 │ triggers Modal runs    │
│ back to R2 + wandb│                 │                        │
└───────────────────┘                 └────────────────────────┘
```

---

## Stack Components

### 1. Cloudflare R2 (Object Store)
**What:** S3-compatible object storage — the single source of truth for all data.

**Why R2 over alternatives:**
- **Zero egress fees** — free to pull from Modal, Streamlit, anywhere
- S3-compatible API — works with boto3, DVC, DuckDB, Modal CloudBucketMount
- 10GB free tier (perpetual), then $0.015/GB/month
- No server process — no RAM impact on VPS
- Modal has native `CloudBucketMount` for streaming reads

**Bucket:** `s3://baseball-data`
**Endpoint:** `https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com`

**Data layout:**
```
statcast/statcast_{year}.parquet  — raw pitch data (119 cols, ~1.3GB total)
features/{feature_set}.parquet   — engineered features for models
models/{model_name}/{version}/   — serialized PyMC traces
projections/{year}/              — model output projections
```

### 2. DVC (Data Version Control)
**What:** Git-integrated data versioning — tracks large files without storing them in Git.

**Why DVC:**
- `.dvc` files in Git track which data version goes with which code
- `dvc push/pull` syncs with R2 like `git push/pull`
- Free, open source, pip-installable
- Perfect for solo workflows — no server needed

**Remote configured:** `r2` → `s3://baseball-data`

### 3. DuckDB (Analytics)
**What:** In-process SQL engine that queries Parquet files directly.

**Why DuckDB:**
- Queries Parquet files in R2 or local — no import step
- Handles 7.8M rows without breaking a sweat
- Python-native (`import duckdb`)
- Replaces SQLite for analytics without the 5GB disk footprint

**Example usage:**
```python
import duckdb
# Query local parquet
duckdb.sql("SELECT * FROM 'data/statcast/statcast_2024.parquet' LIMIT 10")
# Query R2 directly (with httpfs extension)
duckdb.sql("SELECT * FROM read_parquet('s3://baseball-data/statcast/*.parquet')")
```

### 4. Modal (Compute)
**What:** Serverless cloud compute for training runs.

**Why Modal:**
- PyMC MCMC sampling is CPU/memory intensive — don't want to bog down the VPS
- Pay-per-second — only costs money when training
- `CloudBucketMount` streams data from R2 — no download step
- Python-native — define functions, Modal handles infra

**What runs on Modal:**
- Bayesian model training (PyMC sampling)
- Monte Carlo season simulations (10K+ seasons)
- Heavy batch computation
- Model comparison / cross-validation runs

**CloudBucketMount example:**
```python
import modal
bucket = modal.CloudBucketMount(
    "baseball-data",
    secret=modal.Secret.from_name("r2-credentials"),
    endpoint_url="https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com"
)

@app.function(volumes={"/data": bucket})
def train():
    import pyarrow.parquet as pq
    df = pq.read_table("/data/statcast/statcast_2024.parquet")
```

### 5. Weights & Biases (Experiment Tracking)
**What:** Cloud experiment tracking — logs every training run automatically.

**What gets logged:**
- MCMC diagnostics (R-hat, ESS, divergences)
- Model hyperparameters
- Accuracy metrics (vs Marcel baseline, vs actuals)
- Posterior distributions
- Training time, convergence info

### 6. Streamlit Cloud (Dashboard)
**What:** Interactive web dashboard for projections and analysis.

---

## Data in R2

### Current Data (as of April 2026)

| File | Rows | Size | Description |
|------|------|------|-------------|
| statcast_2015.parquet | 712,844 | 102.5 MB | |
| statcast_2016.parquet | 726,275 | 105.4 MB | |
| statcast_2017.parquet | 735,954 | 125.2 MB | |
| statcast_2018.parquet | 734,567 | 122.5 MB | |
| statcast_2019.parquet | 763,198 | 126.9 MB | |
| statcast_2020.parquet | 280,398 | 47.0 MB | COVID shortened |
| statcast_2021.parquet | 765,733 | 127.2 MB | |
| statcast_2022.parquet | 775,330 | 127.5 MB | |
| statcast_2023.parquet | 774,038 | 135.7 MB | |
| statcast_2024.parquet | 760,248 | 145.7 MB | |
| statcast_2025.parquet | 749,091 | 146.2 MB | |
| **Total** | **7,777,676** | **1,312 MB** | 119 columns |

---

## Accounts & Services

| Service | URL | Free Tier | What For |
|---------|-----|-----------|----------|
| Cloudflare R2 | dash.cloudflare.com | 10GB storage, 1M reads/mo | Object store |
| Modal | modal.com | $30/mo free credits | Training compute |
| Weights & Biases | wandb.ai | Unlimited personal | Experiment tracking |
| Streamlit Cloud | streamlit.io | 1 free app | Dashboard |
| GitHub | github.com | ✅ | Code + DVC metadata |

---

## Migration Notes

**Previous stack:** SQLite (5.2GB on VPS disk) → Turso Cloud
**Current stack:** Parquet in R2 + DVC + DuckDB
**Reason:** The 5.2GB SQLite DB was consuming 97% of VPS disk. Parquet+R2 gives us:
- No local disk footprint (data lives in R2)
- Better compression (5.2GB SQLite → 1.3GB Parquet)
- Year-partitioned for efficient queries
- Modal can stream directly via CloudBucketMount
