"""Backtest harness (roadmap 0.2).

The factory's objective function: every model change is judged by
`backtest()` numbers against dumb baselines, never by eyeball. With a
`cutoff_date` the same call answers the rest-of-season question — given
everything through July 1, how good is the rest of the year?
"""
from src.eval.backtest import (
    backtest, score, calibration, COMPONENTS, parquet_provider, frame_provider,
)
from src.eval.intraseason import (
    aggregate_pa, assert_split_clean, backtest_intraseason,
    partial_and_realized, split_at_cutoff,
)

__all__ = ["backtest", "score", "calibration", "COMPONENTS",
           "parquet_provider", "frame_provider",
           "backtest_intraseason", "aggregate_pa", "partial_and_realized",
           "split_at_cutoff", "assert_split_clean"]
