"""Project configuration and constants."""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_DIR = DATA_DIR / "parquet"

# Data range
STATCAST_START_YEAR = 2015
HISTORICAL_START_YEAR = 2000
CURRENT_YEAR = 2025  # Latest complete season

# Marcel constants (Tom Tango's formula)
MARCEL_WEIGHTS = {1: 5, 2: 4, 3: 3}  # year_ago: weight
MARCEL_PA_WEIGHTS = {1: 0.5, 2: 0.1}  # regression PA weights
MARCEL_LEAGUE_PA = 200  # regression PA for hitters
MARCEL_LEAGUE_IP = 50   # regression IP for pitchers

# Turso Cloud database
TURSO_DATABASE_URL = "libsql://baseball-projections-jseeburger4.aws-us-east-1.turso.io"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, PARQUET_DIR]:
    d.mkdir(parents=True, exist_ok=True)
