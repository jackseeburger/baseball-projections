# Baseball Projection System — Infrastructure & Architecture

## Overview

Bayesian Baseball Projection System built for a solo practitioner.
**Training runs on Modal**, **data lives in Cloudflare R2** (Parquet files),
**queries via DuckDB**, **experiment tracking in W&B**, **versioning with DVC+Git**.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       YOU (Browser)                          │
│                                                              │
│     wandb.ai              Streamlit Cloud (future)           │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ Training run  │      │ Projections  │                     │
│  │ comparison,   │      │ Sim results  │                     │
│  │ MCMC diags,   │      │ Comparisons  │                     │
│  │ artifacts     │      │ Diagnostics  │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
└─────────┼─────────────────────┼──────────────────────────────┘
          │                     │
          ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│                  Cloudflare R2 (Object Store)                 │
│                                                              │
│  s3://baseball-data/                                         │
│  ├── statcast/             # 7.7M pitches, 11 Parquet files  │
│  │   ├── statcast_2015.parquet  ...  statcast_2025.parquet   │
│  ├── features/             # Aggregated feature tables        │
│  ├── models/               # Serialized model artifacts       │
│  └── projections/          # Output projections               │
│                                                              │
│  Zero egress fees · S3-compatible · DVC-versioned            │
└──────────────┬───────────────────────────┬───────────────────┘
               │                           │
    ┌──────────┘                           └──────────┐
    ▼                                                  ▼
┌───────────────────┐                    ┌────────────────────────┐
│   Modal (Compute) │                    │   VPS / Hari Seldon    │
│                   │                    │                        │
│ • PyMC training   │                    │ • Data pipelines       │
│ • MCMC sampling   │                    │   (pybaseball → R2)    │
│ • Monte Carlo     │                    │ • DVC versioning       │
│   season sims     │                    │ • Cron jobs            │
│ • DuckDB queries  │                    │ • Orchestration        │
│                   │                    │ • DuckDB local queries │
│ Reads from R2 via │                    │                        │
│ boto3/DuckDB httpfs│                   │ Reads/writes R2 via    │
│ Logs to wandb     │                    │ boto3/DVC              │
└───────────────────┘                    └────────────────────────┘
```

---

## Stack Components

### 1. Cloudflare R2 (Primary Storage)
**What:** S3-compatible object store — zero egress fees.

**Bucket:** `baseball-data`
**Endpoint:** `https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com`

**Current data:**
- `statcast/` — 11 year-partitioned Parquet files (2015-2025), 1.31 GB total
- 7,777,676 pitch-level rows, 119 columns each

**Why R2 over Turso/SQLite:**
- Zero egress = free reads from Modal, VPS, anywhere
- Parquet format = columnar, compressed, fast analytical queries
- Scales to any size without VPS disk pressure
- DuckDB can query Parquet files directly via httpfs

### 2. DVC (Data Version Control)
**What:** Git-native data versioning. Tracks what's in R2 alongside code.

**Remote:** `r2` → `s3://baseball-data`
**Config:** `.dvc/config` (access key in `.dvc/config.local`, not committed)

### 3. DuckDB (Analytics Engine)
**What:** In-process SQL engine that queries Parquet files directly.

**Usage patterns:**
- On Modal: query R2 Parquets via httpfs extension
- On VPS: same, or download Parquet locally for repeated queries
- Replaces need for a persistent database server

### 4. Modal (Compute)
**What:** Serverless cloud compute for training runs.

**App:** `baseball-projections`
**Secrets:** `r2-baseball` (R2 creds), `wandb-baseball` (W&B key)
**Volume:** `baseball-models` (model artifacts only)

**Functions:**
- `smoke_test` — verifies R2 + DuckDB + PyMC + wandb
- `train_hitter_model` — hierarchical Bayesian training
- `simulate_season` — Monte Carlo season simulation
- `wandb_integration_test` — end-to-end W&B test

### 5. Weights & Biases (Experiment Tracking)
**What:** Cloud experiment tracking — logs every training run.

**Project:** `baseball-projections`
**Entity:** `jseeburger`

---

## Data Flow

### Data Ingestion (VPS cron)
```
1. pybaseball fetches new stats
2. Process into Parquet format
3. Upload to R2 (s3://baseball-data/statcast/)
4. DVC tracks the version
```

### Training Run (Modal)
```
1. VPS triggers Modal function
2. Modal reads Parquet data from R2 via boto3/DuckDB
3. PyMC samples posterior → logs diagnostics to wandb
4. Modal saves model artifacts to baseball-models volume
5. Results written back to R2
```

---

## Accounts & Services

| Service | Free Tier | What For |
|---------|-----------|----------|
| Cloudflare R2 | 10GB storage, 10M reads/mo | Data storage |
| Modal | $30/mo free credits | Training compute |
| Weights & Biases | Unlimited personal | Experiment tracking |
| GitHub | ✅ | Code + DVC metadata |
| DVC | ✅ (open source) | Data versioning |

---

## Modal Secrets

| Secret | Keys | Purpose |
|--------|------|---------|
| `r2-baseball` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_ENDPOINT_URL` | R2 access |
| `wandb-baseball` | `WANDB_API_KEY` | W&B logging |
| `turso-baseball` | (legacy, can remove) | Old Turso DB |
