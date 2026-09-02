"""Backtest harness (roadmap 0.2).

The factory's objective function: every model change is judged by
`backtest()` numbers against dumb baselines, never by eyeball.
"""
from src.eval.backtest import (
    backtest, score, calibration, COMPONENTS, parquet_provider, frame_provider,
)

__all__ = ["backtest", "score", "calibration", "COMPONENTS",
           "parquet_provider", "frame_provider"]
