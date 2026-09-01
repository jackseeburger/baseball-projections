"""Scoring metrics for projection backtests.

All metrics are weighted by trials (PA, AB, BIP — whatever the component's
denominator is), so a 600-PA season counts more than a September cup of
coffee. Inputs are aligned 1-D arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-10


def _as_arrays(*cols) -> list[np.ndarray]:
    return [np.asarray(c, dtype=np.float64) for c in cols]


def binomial_log_loss(pred_rate, successes, trials) -> float:
    """Mean negative log likelihood per trial under Binomial(trials, pred).

    Equivalent (up to the constant binomial coefficient) to per-PA Bernoulli
    log loss, so it is comparable across aggregation levels.
    """
    p, k, n = _as_arrays(pred_rate, successes, trials)
    p = np.clip(p, EPS, 1 - EPS)
    ll = k * np.log(p) + (n - k) * np.log(1 - p)
    return float(-ll.sum() / n.sum())


def weighted_mae(pred_rate, realized_rate, weights) -> float:
    p, r, w = _as_arrays(pred_rate, realized_rate, weights)
    return float(np.sum(w * np.abs(p - r)) / np.sum(w))


def weighted_rmse(pred_rate, realized_rate, weights) -> float:
    p, r, w = _as_arrays(pred_rate, realized_rate, weights)
    return float(np.sqrt(np.sum(w * (p - r) ** 2) / np.sum(w)))


def calibration_table(pred_rate, realized_rate, weights, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs realized in quantile buckets of the prediction.

    Returns one row per bucket: n_players, total_weight, mean_predicted,
    mean_realized (weighted). A calibrated model tracks the diagonal.
    """
    p, r, w = _as_arrays(pred_rate, realized_rate, weights)
    df = pd.DataFrame({"pred": p, "realized": r, "w": w})
    df["bucket"] = pd.qcut(df["pred"].rank(method="first"), n_bins, labels=False)
    rows = []
    for b, g in df.groupby("bucket"):
        rows.append({
            "bucket": int(b),
            "n_players": len(g),
            "total_weight": float(g["w"].sum()),
            "mean_predicted": float(np.average(g["pred"], weights=g["w"])),
            "mean_realized": float(np.average(g["realized"], weights=g["w"])),
        })
    out = pd.DataFrame(rows)
    out["gap"] = out["mean_realized"] - out["mean_predicted"]
    return out
