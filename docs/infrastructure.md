     1|# Baseball Projection System — Infrastructure & Architecture
     2|
     3|## Overview
     4|
     5|This document defines the infrastructure stack for the Bayesian Baseball Projection System.
     6|The system is designed so that **training runs on Modal** (GPU/cloud compute), **data lives in Turso Cloud**
     7|(accessible from anywhere), **experiment tracking lives in W&B**, and **dashboards run on Streamlit Cloud** —
     8|all independent of the VPS agent.
     9|
    10|---
    11|
    12|## Architecture Diagram
    13|
    14|```
    15|┌─────────────────────────────────────────────────────────────┐
    16|│                    YOU (Browser)                             │
    17|│                                                             │
    18|│   Streamlit Cloud          Datasette          wandb.ai      │
    19|│   ┌──────────────┐    ┌──────────────┐   ┌──────────────┐  │
    20|│   │ Projections  │    │ Browse raw   │   │ Training run  │  │
    21|│   │ Sim results  │    │ tables, SQL  │   │ comparison,   │  │
    22|│   │ Comparisons  │    │ queries,     │   │ MCMC diags,   │  │
    23|│   │ Diagnostics  │    │ CSV export   │   │ artifacts     │  │
    24|│   └──────┬───────┘    └──────┬───────┘   └──────┬───────┘  │
    25|└──────────┼───────────────────┼──────────────────┼───────────┘
    26|           │                   │                  │
    27|           ▼                   ▼                  ▼
    28|┌──────────────────────────────────────────────────────────────┐
    29|│                     Turso Cloud (SQLite)                      │
    30|│                                                              │
    31|│   Primary database — accessible from everywhere              │
    32|│   ┌────────────┐ ┌────────────┐ ┌────────────┐              │
    33|│   │ player_    │ │ projec-    │ │ season_    │              │
    34|│   │ stats      │ │ tions      │ │ sims       │  + more      │
    35|│   └────────────┘ └────────────┘ └────────────┘              │
    36|│                                                              │
    37|│   Features used:                                             │
    38|│   • Branching — snapshot DB before each training run         │
    39|│   • Concurrent writes — dashboard reads while training writes│
    40|│   • Vector search — player similarity (future)               │
    41|│   • Web console — browse data anytime                        │
    42|└──────────────────┬──────────────────┬────────────────────────┘
    43|                   │                  │
    44|        ┌──────────┘                  └──────────┐
    45|        ▼                                        ▼
    46|┌───────────────────┐                 ┌────────────────────────┐
    47|│   Modal (Compute) │                 │   VPS / Hari Seldon    │
    48|│                   │                 │                        │
    49|│ • PyMC training   │                 │ • Data pipelines       │
    50|│ • MCMC sampling   │                 │   (pybaseball → Turso) │
    51|│ • Monte Carlo     │                 │ • Cron jobs            │
    52|│   season sims     │                 │ • Orchestration        │
    53|│ • Heavy compute   │                 │ • Light scripting      │
    54|│                   │                 │                        │
    55|│ Writes results    │                 │ Writes raw data        │
    56|│ back to Turso +   │                 │ to Turso, triggers     │
    57|│ logs to wandb     │                 │ Modal training runs    │
    58|└───────────────────┘                 └────────────────────────┘
    59|```
    60|
    61|---
    62|
    63|## Stack Components
    64|
    65|### 1. Turso Cloud (Database)
    66|**What:** Cloud-hosted SQLite (libSQL) — the single source of truth for all project data.
    67|
    68|**Why Turso over Postgres:**
    69|- Lightweight — no server process eating VPS RAM
    70|- SQLite-compatible — Python `sqlite3` interface, zero learning curve
    71|- Database branching — snapshot before each experiment run, rollback if needed
    72|- Concurrent writes — dashboard can read while training writes (solves SQLite's main limitation)
    73|- Built-in vector search — player similarity embeddings later
    74|- Web console — browse tables from your browser anytime
    75|- Free tier: 500 databases, 9GB storage (way more than we need)
    76|
    77|**Access pattern:**
    78|- Modal workers → write training results via Turso SDK
    79|- Streamlit dashboard → read projections/sims via Turso SDK
    80|- VPS (Hari) → write raw data from pipelines via Turso SDK
    81|- Your browser → Turso web console for ad-hoc queries
    82|
    83|### 2. Modal (Compute)
    84|**What:** Serverless cloud compute for training runs.
    85|
    86|**Why Modal:**
    87|- PyMC MCMC sampling is CPU/memory intensive — don't want to bog down the VPS
    88|- Pay-per-second — only costs money when training
    89|- GPU available if we add neural net components later
    90|- Python-native — define functions, Modal handles infra
    91|- Easy to trigger from VPS scripts
    92|
    93|**What runs on Modal:**
    94|- Bayesian model training (PyMC sampling)
    95|- Monte Carlo season simulations (10K+ seasons)
    96|- Any heavy batch computation
    97|- Model comparison / cross-validation runs
    98|
    99|### 3. Weights & Biases (Experiment Tracking)
   100|**What:** Cloud experiment tracking — logs every training run automatically.
   101|
   102|**Why wandb over MLflow:**
   103|- Free tier is generous for personal/small team
   104|- Better visualization for Bayesian workflows (trace plots, posteriors)
   105|- Zero infrastructure — no server to host
   106|- Artifacts system for model versioning (store PyMC traces)
   107|- Easy comparison: "run #5 vs run #3" in the web UI
   108|
   109|**What gets logged:**
   110|- MCMC diagnostics (R-hat, ESS, divergences)
   111|- Model hyperparameters
   112|- Accuracy metrics (vs Marcel baseline, vs actuals)
   113|- Posterior distributions
   114|- Training time, convergence info
   115|- Model artifacts (serialized traces)
   116|
   117|### 4. Streamlit Cloud (Dashboard)
   118|**What:** Interactive web dashboard — the main thing you open day-to-day.
   119|
   120|**Dashboard pages (planned):**
   121|- **Player Projections** — search any player, see projection card with confidence intervals
   122|- **Model Comparison** — Marcel vs Bayesian side-by-side, accuracy metrics
   123|- **Season Simulator** — playoff odds, win distributions, standings projections
   124|- **MCMC Diagnostics** — convergence plots, trace plots, posterior distributions
   125|- **Leaderboards** — projected WAR leaders, breakout candidates, regression risks
   126|
   127|**Why Streamlit Cloud:**
   128|- Free hosting
   129|- Python-native — reads directly from Turso
   130|- Interactive widgets (player search, stat filters, year selectors)
   131|- Deploys from a GitHub repo automatically
   132|
   133|### 5. Datasette (Data Explorer) — Optional
   134|**What:** Instant web UI for exploring SQLite databases — browse tables, run SQL, export CSV.
   135|
   136|**Why:**
   137|- When you want to just *look at the data* without building a dashboard page for it
   138|- Great for ad-hoc SQL queries from your browser
   139|- Can connect to Turso or a synced local SQLite file
   140|
   141|**Hosting options:**
   142|- Run on VPS (lightest)
   143|- HuggingFace Spaces (free, public)
   144|- Datasette Cloud
   145|
   146|---
   147|
   148|## Data Flow
   149|
   150|### Daily Pipeline (automated via cron on VPS)
   151|```
   152|1. pybaseball fetches new stats → writes to Turso (raw tables)
   153|2. Park factor calculations update → writes to Turso
   154|3. Marcel baseline recalculates → writes to Turso
   155|```
   156|
   157|### Training Run (triggered manually or on schedule)
   158|```
   159|1. VPS triggers Modal function
   160|2. Modal worker reads training data from Turso
   161|3. PyMC samples posterior → logs diagnostics to wandb
   162|4. Modal writes projection results back to Turso
   163|5. Modal uploads model artifacts to wandb
   164|6. Streamlit dashboard automatically reflects new projections
   165|```
   166|
   167|### Season Simulation (triggered after projection updates)
   168|```
   169|1. Modal reads latest projections from Turso
   170|2. Runs 10K+ Monte Carlo seasons
   171|3. Writes standings/playoff odds/WS odds to Turso
   172|4. Logs run metadata to wandb
   173|5. Streamlit sim page updates automatically
   174|```
   175|
   176|---
   177|
   178|## Database Schema (High-Level)
   179|
   180|Detailed schema will be defined during implementation, but the main table groups:
   181|
   182|```sql
   183|-- Raw data (populated by VPS pipelines)
   184|batting_stats          -- Historical batting stats by player-season
   185|pitching_stats         -- Historical pitching stats by player-season
   186|statcast_data          -- Pitch-level Statcast data (2015+)
   187|park_factors           -- Park factor adjustments by year
   188|player_metadata        -- Player IDs, names, positions, birth dates
   189|
   190|-- Projections (populated by Modal training runs)
   191|marcel_projections     -- Marcel baseline projections
   192|bayesian_projections   -- Hierarchical model projections
   193|projection_runs        -- Metadata about each training run
   194|
   195|-- Simulations (populated by Modal sim runs)
   196|season_simulations     -- Simulated season results
   197|playoff_odds           -- Team playoff/WS probabilities
   198|sim_runs               -- Metadata about each sim run
   199|
   200|-- Model registry
   201|model_versions         -- Serialized model info, hyperparameters
   202|model_comparisons      -- Accuracy metrics across model versions
   203|```
   204|
   205|---
   206|
   207|## Accounts & Services Needed
   208|
   209|| Service | URL | Free Tier | What For |
   210||---------|-----|-----------|----------|
   211|| Turso Cloud | turso.tech | 500 DBs, 9GB | Database |
   212|| Modal | modal.com | $30/mo free credits | Training compute |
   213|| Weights & Biases | wandb.ai | Unlimited personal | Experiment tracking |
   214|| Streamlit Cloud | streamlit.io | 1 free app | Dashboard |
   215|| GitHub (repo) | github.com | ✅ | Code + Streamlit deploy |
   216|
   217|---
   218|
   219|## Repository Structure (Planned)
   220|
   221|```
   222|baseball-projections/
   223|├── README.md
   224|├── pyproject.toml              # Dependencies (pymc, wandb, turso SDK, etc.)
   225|├── data/
   226|│   ├── pipelines/              # Data ingestion scripts
   227|│   │   ├── batting.py
   228|│   │   ├── pitching.py
   229|│   │   ├── statcast.py
   230|│   │   └── park_factors.py
   231|│   └── db.py                   # Turso connection helpers
   232|├── models/
   233|│   ├── marcel/                 # Marcel baseline
   234|│   ├── hitting/                # Hierarchical hitting models
   235|│   ├── pitching/               # Hierarchical pitching models
   236|│   └── supporting/             # Aging, injuries, defense, MiLB
   237|├── simulation/
   238|│   ├── game_engine.py          # Markov chain PA simulator
   239|│   └── season_sim.py           # Monte Carlo season simulator
   240|├── modal_functions/
   241|│   ├── train.py                # Modal training entrypoints
   242|│   └── simulate.py             # Modal simulation entrypoints
   243|├── dashboard/
   244|│   ├── app.py                  # Streamlit main app
   245|│   ├── pages/
   246|│   │   ├── projections.py
   247|│   │   ├── model_comparison.py
   248|│   │   ├── season_sim.py
   249|│   │   └── diagnostics.py
   250|│   └── components/             # Reusable Streamlit components
   251|└── tests/
   252|```
   253|
   254|---
   255|
   256|## Relationship to Existing Plans
   257|
   258|- **baseball-projection-system-plan.md** — The 6-layer model architecture and implementation roadmap (what we're building)
   259|- **baseball-projection-systems-research.md** — Research on existing systems (context and inspiration)
   260|- **This document** — Infrastructure and tooling (how and where it runs)
   261|- **Linear project (SIG-227 to SIG-251)** — Task tracking for implementation
   262|