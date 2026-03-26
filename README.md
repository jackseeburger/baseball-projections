# ⚾ Baseball Projections

A Bayesian baseball projection system using PyMC — hierarchical models for hitter and pitcher projections. Built to produce competitive full-season projections grounded in modern Statcast data and classical reliability-weighted baselines.

## Architecture — 6 Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **1. Data Pipelines + Marcel Baseline** | Historical stats (FanGraphs), Statcast data (Baseball Savant), park factors, and Marcel projection engine | ✅ Complete |
| **2. Bayesian Hierarchical Models** | PyMC models for hitter and pitcher true-talent estimation | 🔨 In Progress |
| **3. Projection Synthesis** | Blend Marcel baseline with Bayesian posteriors, aging curves, platoon splits | Planned |
| **4. Team-Level Aggregation** | Depth charts, lineup optimization, team wins projections | Planned |
| **5. In-Season Updates** | Live Bayesian updating as the season progresses | Planned |
| **6. Evaluation & Dashboard** | Accuracy tracking, comparison to other systems, web dashboard | Planned |

## Current Status

**Phase 1 is complete.** The project includes:

- **Historical data pipeline** — Fetches seasonal hitter/pitcher stats from FanGraphs (2000–present)
- **Statcast pipeline** — Downloads pitch-level data from Baseball Savant (2015–present)
- **Park factors** — Computes multi-year regressed park factors by team
- **Marcel projections** — Tom Tango's Marcel the Monkey system: reliability-weighted 3-year averages with age adjustment and regression to league mean

## Setup

```bash
# Clone the repo
git clone https://github.com/jackseeburger/baseball-projections.git
cd baseball-projections

# Create virtual environment and install dependencies
make setup

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Pipeline

```bash
# Download all data and run Marcel projections
make data

# Or with more control:
python run_pipeline.py                  # Full pipeline
python run_pipeline.py --skip-statcast  # Skip slow Statcast download
python run_pipeline.py --marcel-only    # Just run Marcel on existing data
python run_pipeline.py --force          # Force re-download everything
```

**Note:** The initial Statcast download fetches ~10 years of pitch-level data and may take 30+ minutes.

## Project Structure

```
baseball-projections/
├── run_pipeline.py           # Master pipeline orchestrator
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
├── Makefile                  # Common tasks (setup, data, test, clean)
│
├── src/
│   ├── config.py             # Paths, constants, Marcel parameters
│   ├── data/
│   │   ├── historical_pipeline.py   # FanGraphs seasonal stats
│   │   ├── statcast_pipeline.py     # Baseball Savant pitch data
│   │   ├── park_factors.py          # Park factor computation
│   │   └── pa_level_pipeline.py     # Plate-appearance level data
│   ├── models/
│   │   └── marcel.py               # Marcel projection engine
│   └── utils/
│       └── helpers.py               # Logging, common utilities
│
├── tests/
│   └── test_data/
│       └── test_marcel.py           # Marcel projection tests
│
└── data/                     # Generated data (git-ignored)
    ├── raw/                  # Raw Statcast parquet files
    └── parquet/              # Processed season-level data + projections
```

## Testing

```bash
make test
# or
pytest -v
```

## License

MIT
