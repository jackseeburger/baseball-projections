#!/usr/bin/env python3
"""Download Statcast data in background."""
import sys
sys.path.insert(0, '.')
from src.data.statcast_pipeline import build_statcast_pipeline

h, p = build_statcast_pipeline(start_year=2015, end_year=2025)
print(f'Done! Hitters: {len(h)}, Pitchers: {len(p)}')
