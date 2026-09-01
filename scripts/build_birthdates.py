"""Build the birthdates parquet from the Chadwick Bureau register.

Downloads the register, caches data/parquet/birthdates.parquet, and — when a
PA-level parquet is available to enumerate batter ids — also writes
data/parquet/batter_birth_years.parquet in the schema the Modal training
functions already read from the volume ([batter, birth_year]), so refits pick
up real ages with no code change.

Usage:
    python scripts/build_birthdates.py                 # refresh register cache
    python scripts/build_birthdates.py --pa-parquet data/parquet/pa_data.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import PARQUET_DIR
from src.data.birthdates import build_batter_birth_years, load_birthdates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pa-parquet", type=Path, default=None,
                        help="PA-level parquet with a 'batter' and 'game_year' "
                             "column; enables batter_birth_years.parquet output")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download the register even if cached")
    args = parser.parse_args()

    birthdates = load_birthdates(refresh=args.refresh)
    n_year = int(birthdates["birth_year"].notna().sum())
    n_full = int(birthdates["birth_day"].notna().sum())
    print(f"register: {len(birthdates)} MLBAM ids, "
          f"{n_year} with birth year, {n_full} with full birthdate")

    if args.pa_parquet is not None:
        pa = pd.read_parquet(args.pa_parquet, columns=["batter", "game_year"])
        first_year = pa.groupby("batter")["game_year"].min()
        table = build_batter_birth_years(first_year, birthdates)
        out = PARQUET_DIR / "batter_birth_years.parquet"
        table.to_parquet(out, index=False)
        matched = table["batter"].isin(
            birthdates.loc[birthdates["birth_year"].notna(), "batter"]
        ).mean()
        print(f"wrote {out}: {len(table)} batters, "
              f"{matched:.1%} matched to real register birth years")


if __name__ == "__main__":
    main()
