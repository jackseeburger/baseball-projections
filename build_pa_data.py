#!/usr/bin/env python3
"""Build PA-level dataset from raw Statcast data."""
import sys
sys.path.insert(0, '.')
from src.data.pa_level_pipeline import build_pa_level_data
pa_df = build_pa_level_data(force_rebuild=True)
print(f"\nDone! {len(pa_df):,} plate appearances")
