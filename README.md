# ⚾ Baseball Projections

A baseball projection and simulation system — hierarchical Bayesian player models, a season Monte Carlo with MLB tiebreakers, and a backtest harness that scores everything against dumb baselines, public systems, and (next) the betting market. Also the template for a general ML factory: data → object store, honest eval harness, serverless training, scheduled agent sessions, public scoreboard.

## Where things stand

**North star:** beat the betting market's closing line, out of sample, and
prove it on a public scoreboard. Read **[docs/architecture.md](docs/architecture.md)**
first — it is the one document every session works against: the model
rollup as stations, each station's baseline and current score, the gate rule
for what gets wired into production, and the edge thesis.

| Doc | What it is |
|---|---|
| [docs/architecture.md](docs/architecture.md) | North star: stations, gate rule, edge thesis, sequencing |
| [docs/roadmap.md](docs/roadmap.md) | Dated v1 plan (ship playoff odds by Sept 28, 2026) |
| [docs/automation.md](docs/automation.md) | How work runs in the background (Modal cron + scheduled Claude sessions) |
| [docs/accuracy-2026.md](docs/accuracy-2026.md) | Honest scoreboard: our models vs Steamer/ZiPS/Depth Charts, per-game Brier |
| [docs/backtest-baselines.md](docs/backtest-baselines.md) | Marcel / naive baselines 2019–2025 that every component must beat |
| [docs/playoff-odds-validation.md](docs/playoff-odds-validation.md) | Simulator vs FanGraphs |

**Live today:** playoff odds for all 30 teams (`public/`), regenerated nightly
with dated snapshots. **Not yet earned:** the Bayesian player components,
which currently tie Marcel and trail Depth Charts, stay out of the rollup
until they beat their baseline in the harness.

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
