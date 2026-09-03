"""Scoring the team-season walk-forward: what the projections were worth.

`src/eval/team_season.py` produces the projections — one row per club per
as-of date per arm. This module turns those rows into the numbers the claim
rests on:

  * **projected final wins** — MAE and RMSE against the club's actual final
    wins. Note that projecting *final* wins and projecting *rest-of-season*
    wins are the same problem measured the same way: the banked record is
    common to both the projection and the outcome, so it cancels out of the
    error exactly. What does not cancel is the shrinking schedule, which drags
    every arm's MAE toward zero as September arrives whether or not it knows
    anything. `rest_wpct_mae` — the same error expressed as a win *rate* over
    the games still to be played — is the version of the metric that does not
    shrink for free, and it is the one the through-season curve is read off.
  * **made the playoffs** — Brier and log loss against the binary outcome.
  * **division, pennant, World Series** — Brier, with the small-n caveat
    stated rather than implied: a season produces six division winners, two
    pennants and one champion, so those three columns are 30 clubs' worth of
    probability against one, two or six ones.
  * **calibration** — deciles of projected playoff probability against the
    realized frequency, pooled across seasons, plus the reliability and
    resolution terms of the Brier decomposition.
  * **paired differences** — every baseline against the chain on the same
    club, same date, same season, with standard errors clustered by season.
    Clubs within a season share a schedule and a pennant race, and the same
    club appears at 25 cutoffs; one cluster per season is what keeps those
    from being counted as independent evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.team_season import OUTCOME_OF, PROB_COLUMNS

EPS = 1e-12
# Buckets for the through-season curve: the fraction of its 162 games a club
# has played by the as-of date. Fifths of a season are wide enough that a
# season-clustered standard error means something and narrow enough to show
# where the curves cross.
SEASON_BUCKETS = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60),
                  (0.60, 0.75), (0.75, 0.90), (0.90, 1.01)]
BUCKET_LABELS = ["0-15%", "15-30%", "30-45%", "45-60%", "60-75%", "75-90%",
                 "90-100%"]
# A finer grid for the plotted curve. Too narrow to carry a season-clustered
# standard error, wide enough that a twenty-point line is not a scatter.
FINE_EDGES = [round(0.05 * i, 2) for i in range(21)]
FINE_LABELS = [f"{int(FINE_EDGES[i] * 100)}-{int(FINE_EDGES[i + 1] * 100)}%"
               for i in range(20)]


def attach_outcomes(projections: pd.DataFrame,
                    outcomes: pd.DataFrame) -> pd.DataFrame:
    """Join each projection to what actually happened, and add its errors.

    Rows whose season has no outcomes yet (2026, mid-flight) are dropped: an
    unfinished season cannot be scored, and quietly scoring it against a
    partial record is exactly the kind of number this document exists to stop.
    """
    df = projections.merge(outcomes, on=["season", "team_id"], how="inner")
    df["err_final_wins"] = df["proj_final_wins"] - df["final_wins"]
    df["rest_wins"] = df["final_wins"] - df["wins_to_date"]
    # Identical to err_final_wins by construction — the banked record cancels —
    # and kept under its own name because the *rate* below is built off it and
    # the identity is worth being able to assert in a test.
    df["err_rest_wins"] = df["proj_rest_wins"] - df["rest_wins"]
    df["err_rest_wpct"] = (df["err_rest_wins"]
                           / df["club_games_remaining"].clip(lower=1))
    df["season_fraction"] = (df["games_played"]
                             / (df["games_played"] + df["games_remaining"]))
    df["bucket"] = pd.cut(
        df["season_fraction"],
        bins=[b[0] for b in SEASON_BUCKETS] + [SEASON_BUCKETS[-1][1]],
        labels=BUCKET_LABELS, right=False, include_lowest=True)
    df["fine_bucket"] = pd.cut(df["season_fraction"].clip(upper=0.999),
                               bins=FINE_EDGES, labels=FINE_LABELS,
                               right=False, include_lowest=True)
    # Log loss needs a floor, and the honest floor is the Monte Carlo's own
    # resolution: nought out of 4,000 simulations means p < 1/4,000, not p = 0.
    # Clipping at machine epsilon instead would make a single surprise worth 27
    # nats and turn the statistic into a count of surprises times a constant
    # chosen by the float format.
    floor = (1.0 / (2.0 * df["n_sims"].clip(lower=1))
             if "n_sims" in df.columns else pd.Series(EPS, index=df.index))
    df["prob_floor"] = floor.clip(lower=EPS, upper=0.5)
    for prob, truth in OUTCOME_OF.items():
        df[f"brier_{prob}"] = (df[prob] - df[truth]) ** 2
        p = df[prob].clip(lower=df["prob_floor"]).clip(
            upper=1 - df["prob_floor"])
        df[f"logloss_{prob}"] = -(df[truth] * np.log(p)
                                  + (1 - df[truth]) * np.log(1 - p))
    return df


def _mae(x) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=float))))


def _rmse(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def score(scored: pd.DataFrame, by=("arm",)) -> pd.DataFrame:
    """The headline table: one row per arm (or per arm and bucket)."""
    by = list(by)
    rows = []
    for key, g in scored.groupby(by, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = int(len(g))
        row["n_seasons"] = int(g["season"].nunique())
        row["wins_mae"] = _mae(g["err_final_wins"])
        row["wins_rmse"] = _rmse(g["err_final_wins"])
        row["wins_bias"] = float(np.mean(g["err_final_wins"]))
        row["rest_wpct_mae"] = _mae(g["err_rest_wpct"])
        row["rest_wpct_rmse"] = _rmse(g["err_rest_wpct"])
        for prob in PROB_COLUMNS:
            row[f"brier_{prob[2:]}"] = float(g[f"brier_{prob}"].mean())
        row["logloss_playoffs"] = float(g["logloss_p_playoffs"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(by).reset_index(drop=True)


# ─── paired differences, clustered by season ───

def clustered_mean(diffs: pd.Series, clusters: pd.Series) -> tuple[float, float, int]:
    """Mean of a paired difference with a cluster-robust standard error.

    One cluster per season. The estimator is the usual sandwich for a sample
    mean — the variance of the *cluster* sums rather than of the observations —
    with the finite-cluster correction G/(G−1). With ten seasons that
    correction is 11%, and it is the difference between an honest interval and
    a flattering one.

    Returns `(mean, se, n_clusters)`.
    """
    d = pd.Series(np.asarray(diffs, dtype=float))
    g = pd.Series(np.asarray(clusters))
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(d.mean())
    groups = d.groupby(g.to_numpy())
    G = int(groups.ngroups)
    if G < 2:
        return mean, float("nan"), G
    centered = groups.apply(lambda s: float((s - mean).sum()))
    var = float((centered ** 2).sum()) * (G / (G - 1)) / (n ** 2)
    return mean, float(np.sqrt(max(var, 0.0))), G


PAIRED_METRICS = {
    "wins_abs_err": ("err_final_wins", "abs"),
    "rest_wpct_abs_err": ("err_rest_wpct", "abs"),
    "brier_playoffs": ("brier_p_playoffs", "raw"),
    "brier_division": ("brier_p_division", "raw"),
    "brier_pennant": ("brier_p_pennant", "raw"),
    "brier_ws": ("brier_p_ws", "raw"),
    "logloss_playoffs": ("logloss_p_playoffs", "raw"),
}

KEY = ["season", "as_of", "team_id"]


def paired(scored: pd.DataFrame, arm: str, against: str,
           metrics=None, extra_by=()) -> pd.DataFrame:
    """`arm` minus `against` on the same club, date and season.

    Negative is better for every metric here (they are all losses). `extra_by`
    splits the comparison — pass `("bucket",)` for the through-season curve.
    """
    metrics = metrics or PAIRED_METRICS
    left = scored[scored["arm"] == arm]
    right = scored[scored["arm"] == against]
    cols = KEY + list(extra_by) + [c for c, _ in metrics.values()]
    merged = left[cols].merge(right[cols], on=KEY + list(extra_by),
                              suffixes=("_a", "_b"))
    rows = []
    group_cols = list(extra_by)
    groups = ([(k, g) for k, g in merged.groupby(group_cols, observed=True)]
              if group_cols else [((), merged)])
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key))
        for name, (col, kind) in metrics.items():
            a, b = g[f"{col}_a"], g[f"{col}_b"]
            if kind == "abs":
                a, b = a.abs(), b.abs()
            mean, se, G = clustered_mean(a - b, g["season"])
            rows.append({**base, "arm": arm, "against": against,
                         "metric": name, "n": int(len(g)), "n_seasons": G,
                         "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                         "diff": mean, "se": se,
                         "t": (mean / se if se and np.isfinite(se) and se > 0
                               else float("nan"))})
    return pd.DataFrame(rows)


# ─── calibration ───

def calibration(scored: pd.DataFrame, arm: str, prob: str = "p_playoffs",
                n_bins: int = 10) -> pd.DataFrame:
    """Deciles of the projected probability against the realized frequency.

    Bins are equal-count deciles of the prediction, which is what makes the
    late-season pile-up at 0 and 1 readable: with fixed-width bins nine of the
    ten rows would be empty. Ties are broken by rank so a decile boundary
    inside a run of identical 0.000 predictions does not collapse the table.
    """
    truth = OUTCOME_OF[prob]
    g = scored[scored["arm"] == arm]
    if g.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"pred": g[prob].astype(float).to_numpy(),
                       "truth": g[truth].astype(float).to_numpy()})
    df["bin"] = pd.qcut(df["pred"].rank(method="first"), n_bins, labels=False)
    rows = []
    for b, sub in df.groupby("bin"):
        rows.append({"decile": int(b) + 1, "n": int(len(sub)),
                     "pred_lo": float(sub["pred"].min()),
                     "pred_hi": float(sub["pred"].max()),
                     "mean_pred": float(sub["pred"].mean()),
                     "realized": float(sub["truth"].mean())})
    out = pd.DataFrame(rows)
    out["gap"] = out["realized"] - out["mean_pred"]
    return out


def reliability(scored: pd.DataFrame, arm: str, prob: str = "p_playoffs",
                n_bins: int = 10) -> dict:
    """Murphy's decomposition of the Brier score on the deciles above.

    `brier ≈ reliability − resolution + uncertainty`, where reliability is the
    bin-weighted squared gap between what was predicted and what happened
    (lower is better, zero is perfect calibration), resolution is how far the
    bins' realized rates sit from the base rate (higher is better), and
    uncertainty is the base rate's own variance, which no model can touch.

    The `≈` is honest and not a rounding: the identity is exact only when the
    prediction is constant inside a bin, and a decile of a continuous forecast
    is not. `residual` is what the three terms miss — the within-bin spread of
    the forecast, less twice its within-bin covariance with the outcome — and
    it is reported rather than swept up so the three headline terms can be read
    for what they are.
    """
    table = calibration(scored, arm, prob, n_bins)
    g = scored[scored["arm"] == arm]
    if table.empty or g.empty:
        return {}
    base = float(g[OUTCOME_OF[prob]].mean())
    n = float(table["n"].sum())
    w = table["n"] / n
    rel = float((w * (table["mean_pred"] - table["realized"]) ** 2).sum())
    res = float((w * (table["realized"] - base) ** 2).sum())
    brier = float(g[f"brier_{prob}"].mean())
    unc = base * (1 - base)
    return {"arm": arm, "prob": prob, "n": int(n), "base_rate": base,
            "brier": brier, "reliability": rel, "resolution": res,
            "uncertainty": unc, "residual": brier - (rel - res + unc),
            "skill_score": 1.0 - brier / max(unc, EPS)}


__all__ = ["attach_outcomes", "score", "paired", "clustered_mean",
           "calibration", "reliability", "PAIRED_METRICS", "SEASON_BUCKETS",
           "BUCKET_LABELS"]
