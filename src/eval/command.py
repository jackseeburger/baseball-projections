"""Command aggregates as covariates on a pitcher component projection (BAS-76, stage 2).

The same object as `src/eval/stuff.py`, asked of the other half of the pitch,
and deliberately reusing its estimator (`fit_stuff`, `StuffFit`), its cell
builder (`build_pitcher_cells`) and its leakage guards rather than growing a
second, subtly different copy of any of them. What is new here is the
covariates and one extra arm.

**The question.** BAS-71 cleared the gate on pitcher K/BF and HR/BF and could
not deliver a covariate-only gain on either walk rate: of `stuff_additive`'s
−3.05% on BB/BF, −2.20% was a fitted intercept on the pitcher Marcel and only
−0.87% was stuff. That is not surprising — a pitch's movement says more about
a whiff than about a walk — and it names what walks lack. A walk is a location
outcome. Command is the measurement of location.

**The covariates**, each a ratio of two additive monthly sums, shrunk toward
the league by `ballast` pitches and then standardized (pitch-weighted) across
the pitchers present at that cutoff:

    cmd_resid      the location contribution to CSW, per pitch: the model that
                   sees location minus the model that sees only stuff
    cs_resid       the same difference on called-strike-given-taken, per taken
                   pitch
    zone_share     pitches in the rulebook strike zone
    shadow_share   pitches within a ball's width of an edge
    chase_share    pitches outside the shadow but still reachable
    waste_share    pitches further out than anyone chases

The four region shares are fractions of the pitches whose region is known, and
`heart_share` is deliberately absent: the four regions partition, so all four
shares plus an intercept would be collinear and `heart` is the reference.
`zone_share` crosses heart and shadow and is the thing a walk is adjudicated
on, so it is in on its own.

The residuals are in as *residuals* and not as levels for the reason the whole
ticket exists: the served engine already carries a stuff score, and a level
would hand this arm that score back. `cmd_resid` is what location adds on top
of it, which is the only thing BAS-76 can honestly claim.

**Leakage.** Identical to stuff's and to contact quality's: features are summed
from monthly buckets strictly before the cutoff, a cutoff that is not the first
of a month is refused rather than rounded, and the guard re-checks the filtered
rows. `assert_month_boundary` and `assert_window_clean` are imported rather
than copied — the check is the same check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.pitching_command import COUNT_COLUMNS, REGIONS
from src.eval.contact import assert_month_boundary, assert_window_clean
from src.eval.stuff import (
    DEFAULT_BALLAST,
    DEFAULT_WINDOW_WEIGHTS,
    STUFF_BALLAST_GRID,
    STUFF_WEIGHT_GRID,
    standardize,
)

FEATURES = ("cmd_resid", "cs_resid", "zone_share", "shadow_share",
            "chase_share", "waste_share")

# The same grids stuff sweeps, so the two results are comparable and the
# hyperparameter choice is not quietly a different search.
COMMAND_WEIGHT_GRID = STUFF_WEIGHT_GRID
COMMAND_BALLAST_GRID = STUFF_BALLAST_GRID

# What `scripts/run_command_backtest.py --tune` chose on the tuning seasons
# (2019 and 2021), by pooled trials-weighted MAE of the command arm over the
# two walk rates — the components this pre-registration expects to move, the
# way stuff tuned on the two it expected to move. Overwritten below from the
# tuning run's own output; both defaults start at stuff's so a run with no
# `--tune` is at least on a documented setting.
DEFAULT_COMMAND_WEIGHTS = DEFAULT_WINDOW_WEIGHTS
DEFAULT_COMMAND_BALLAST = DEFAULT_BALLAST

# The four region counts sum to the pitches whose attack region is known,
# which is the region shares' own denominator: a pitch with no tracked strike
# zone has no region, and dividing it into `heart` by default would make a
# missing measurement look like a pitch down the middle.
_REGION_COUNTS = [f"n_{r}" for r in REGIONS]


# --- window sums -------------------------------------------------------------

def window_counts(
    monthly: pd.DataFrame,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_COMMAND_WEIGHTS,
) -> pd.DataFrame:
    """Recency-weighted command counts per pitcher, strictly before the cutoff.

    Sums the monthly buckets for the predict year (months before the cutoff
    month), the season before it and the one before that, at `weights`.
    Returns one row per pitcher with every column of `COUNT_COLUMNS` plus an
    unweighted `pitches_raw` — the real pitches behind the row, which is what
    the exposure split is cut on and what the standardization weights by.
    """
    cutoff = pd.Timestamp(cutoff)
    assert_month_boundary(cutoff)
    # A weight of zero means the season is not in the window at all, so it is
    # dropped rather than multiplied by nothing — otherwise its pitches would
    # still count toward `pitches_raw`.
    w = {predict_year - i: float(x) for i, x in enumerate(weights)
         if float(x) != 0.0}

    rows = monthly[monthly["season"].isin(w)].copy()
    rows = rows[(rows["season"] < predict_year)
                | (rows["month"] < cutoff.month)]
    assert_window_clean(rows, cutoff, predict_year)
    if rows.empty:
        return pd.DataFrame(columns=["player", *COUNT_COLUMNS, "pitches_raw"])

    rows["_w"] = rows["season"].map(w).astype("float64")
    out = pd.DataFrame({"player": rows["pitcher"].to_numpy()})
    for c in COUNT_COLUMNS:
        out[c] = rows[c].to_numpy() * rows["_w"].to_numpy()
    out["pitches_raw"] = rows["pitches"].to_numpy()
    return out.groupby("player", as_index=False).sum()


# --- metrics -----------------------------------------------------------------

def league_profile(counts: pd.DataFrame) -> dict:
    """Pooled league command profile from a window — the shrinkage target.

    Every count is expressed per pitch, including the take count and the
    region counts, so the shrinkage of a per-take or per-region rate adds
    league-average takes and regions along with the numerators on them.
    """
    tot = counts[COUNT_COLUMNS].sum()
    pitches = float(tot["pitches"])
    if pitches <= 0:
        raise ValueError("empty command window: no pitches before the cutoff")
    prof = {c: float(tot[c]) / pitches for c in COUNT_COLUMNS if c != "pitches"}
    prof["pitches"] = pitches
    return prof


def command_metrics(counts: pd.DataFrame, ballast: float = DEFAULT_COMMAND_BALLAST,
                    league: dict | None = None) -> pd.DataFrame:
    """Shrunk command metrics per pitcher.

    Every metric is a ratio of two additive counts, so shrinkage is one
    operation: add `ballast` pitches of the league's own profile to the
    pitcher's counts, then take the ratio. Both numerator and denominator are
    shrunk, which keeps a September call-up with forty takes from having a
    called-strike residual built out of four of them.
    """
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *FEATURES])
    league = league or league_profile(counts)
    b = float(ballast)

    def sh(col: str) -> np.ndarray:
        return counts[col].to_numpy(dtype="float64") + b * league[col]

    pitches = counts["pitches"].to_numpy(dtype="float64") + b
    region_known = sum(sh(c) for c in _REGION_COUNTS)

    def over(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.zeros(len(counts)), where=den > 0)

    return pd.DataFrame({
        "player": counts["player"].to_numpy(),
        "pitches_raw": counts["pitches_raw"].to_numpy(dtype="float64"),
        "cmd_resid": sh("cmd_resid_sum") / pitches,
        "cs_resid": over(sh("cs_resid_sum"), sh("takens")),
        "zone_share": sh("in_zone") / pitches,
        "shadow_share": over(sh("n_shadow"), region_known),
        "chase_share": over(sh("n_chase"), region_known),
        "waste_share": over(sh("n_waste"), region_known),
    })


def features_at_cutoff(
    monthly: pd.DataFrame,
    cutoff,
    predict_year: int,
    weights: tuple[float, float, float] = DEFAULT_COMMAND_WEIGHTS,
    ballast: float = DEFAULT_COMMAND_BALLAST,
) -> pd.DataFrame:
    """Standardized command covariates for every pitcher with pre-cutoff pitches."""
    counts = window_counts(monthly, cutoff, predict_year, weights)
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *FEATURES])
    return standardize(command_metrics(counts, ballast), features=FEATURES)


def attach_command_features(
    cells: pd.DataFrame, monthly: pd.DataFrame,
    weights: tuple[float, float, float] = DEFAULT_COMMAND_WEIGHTS,
    ballast: float = DEFAULT_COMMAND_BALLAST,
    with_exposure: bool = True,
) -> pd.DataFrame:
    """Merge the standardized command covariates onto `build_pitcher_cells`'s
    output, one cutoff-cell at a time (the covariates do not depend on the
    component). A pitcher with no tracked pitches before the cutoff gets z = 0
    on every covariate, which makes the command arm identical to the
    recalibration arm for him — the common pitcher set stays the baseline's.
    """
    out = []
    for (season, cutoff), g in cells.groupby(["season", "cutoff"]):
        z = features_at_cutoff(monthly, cutoff, season, weights, ballast)
        zi = z.set_index("player").reindex(g["player"].to_numpy())
        g = g.copy()
        for f in FEATURES:
            g[f] = zi[f].fillna(0.0).to_numpy()
        if with_exposure:
            g["cmd_pitches_raw"] = zi["pitches_raw"].fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


__all__ = [
    "COMMAND_BALLAST_GRID", "COMMAND_WEIGHT_GRID", "DEFAULT_COMMAND_BALLAST",
    "DEFAULT_COMMAND_WEIGHTS", "FEATURES", "attach_command_features",
    "command_metrics", "features_at_cutoff", "league_profile", "window_counts",
]
