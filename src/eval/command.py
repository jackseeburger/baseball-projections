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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.pitching_command import COUNT_COLUMNS, REGIONS
from src.eval import stuff as stuff_eval
from src.eval.contact import assert_month_boundary, assert_window_clean
from src.eval.stuff import (
    DEFAULT_BALLAST,
    DEFAULT_WINDOW_WEIGHTS,
    LIVE_CELL_SEASONS,
    STUFF_BALLAST_GRID,
    STUFF_WEIGHT_GRID,
    StuffFit,
    build_pitcher_cells,
    fit_stuff,
    standardize,
)
from src.eval.stuff import FEATURES as STUFF_FEATURES

FEATURES = ("cmd_resid", "cs_resid", "zone_share", "shadow_share",
            "chase_share", "waste_share")

# BAS-87's block: the **level** aggregates, not the stuff-differenced residual.
# `cmd_resid` failed BAS-76's own vacuity floor (pooled year-over-year r 0.397
# against 0.45) while the levels carried fine (`cmd_csw` 0.757, `zone_share`
# 0.561, `waste_share` 0.650), so the follow-up asks the level question with
# the served engine's stuff aggregates entered as explicit controls in the same
# fit — which is what stops a level from being the stuff score under a new
# name. The artifact has no `edge_share` column; `shadow_share` (within a
# ball's width of a zone edge) is that quantity under Statcast's own name, and
# is carried as a labelled sensitivity rather than inside the block.
LEVEL_FEATURES = ("cmd_csw", "zone_share", "waste_share")
LEVEL_FEATURES_WITH_EDGE = LEVEL_FEATURES + ("shadow_share",)

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
        # The level: the CSW probability the location-aware `pitching` model
        # assigns this pitcher's pitches, per pitch. `cmd_resid` below is the
        # same quantity minus the stuff model's — the difference BAS-76 scored.
        "cmd_csw": sh("cmd_csw_sum") / pitches,
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
    features: tuple[str, ...] = FEATURES,
) -> pd.DataFrame:
    """Standardized command covariates for every pitcher with pre-cutoff pitches.

    `features` selects the block: `FEATURES` is BAS-76's residual block,
    `LEVEL_FEATURES` BAS-87's level block. Either way the z-score is computed
    on the cutoff's own pre-cutoff window, so nothing outside the training
    data enters it.
    """
    counts = window_counts(monthly, cutoff, predict_year, weights)
    if counts.empty:
        return pd.DataFrame(columns=["player", "pitches_raw", *features])
    return standardize(command_metrics(counts, ballast), features=features)


def attach_command_features(
    cells: pd.DataFrame, monthly: pd.DataFrame,
    weights: tuple[float, float, float] = DEFAULT_COMMAND_WEIGHTS,
    ballast: float = DEFAULT_COMMAND_BALLAST,
    with_exposure: bool = True,
    features: tuple[str, ...] = FEATURES,
) -> pd.DataFrame:
    """Merge the standardized command covariates onto `build_pitcher_cells`'s
    output, one cutoff-cell at a time (the covariates do not depend on the
    component). A pitcher with no tracked pitches before the cutoff gets z = 0
    on every covariate, which makes the command arm identical to the
    recalibration arm for him — the common pitcher set stays the baseline's.
    """
    out = []
    for (season, cutoff), g in cells.groupby(["season", "cutoff"]):
        z = features_at_cutoff(monthly, cutoff, season, weights, ballast,
                               features)
        zi = z.set_index("player").reindex(g["player"].to_numpy())
        g = g.copy()
        for f in features:
            g[f] = zi[f].fillna(0.0).to_numpy()
        if with_exposure:
            g["cmd_pitches_raw"] = zi["pitches_raw"].fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --- the served arm: the joint additive fit (BAS-88) --------------------------

# The hyperparameters `scripts/run_command_level_backtest.py --tune` chose on
# the tuning window (2019 + 2021, the two walk rates, holdout untouched) and
# the ones the arm that cleared BAS-87's gate was scored at — pinned here
# because the served engine has to be the arm that was measured, not a
# re-tuned one. Both are interior points of the grid, unlike stuff's corner: a
# slower-moving skill wants a longer window and more shrinkage.
SERVED_WEIGHTS = (1.0, 0.35, 0.1)
SERVED_BALLAST = 50.0
# The command block moves with the setting above; the six stuff controls stay
# at *their* own pinned setting, because the control has to be the engine that
# is actually served (`stuff_additive`) rather than a re-tuned version of it.
# This is exactly what `run_command_level_backtest.attach_z` does.
SERVED_STUFF_WEIGHTS = DEFAULT_WINDOW_WEIGHTS
SERVED_STUFF_BALLAST = DEFAULT_BALLAST


def served_features(level_features: tuple[str, ...] = LEVEL_FEATURES
                    ) -> tuple[str, ...]:
    """The one covariate block of the served fit: stuff controls, then command.

    Order matters only in that the fit and the provider must agree on it, and
    both take it from here rather than each writing the concatenation out.
    """
    return tuple(STUFF_FEATURES) + tuple(level_features)


def attach_both_blocks(
    cells: pd.DataFrame,
    command_monthly: pd.DataFrame,
    stuff_monthly: pd.DataFrame,
    weights: tuple[float, float, float] = SERVED_WEIGHTS,
    ballast: float = SERVED_BALLAST,
    level_features: tuple[str, ...] = LEVEL_FEATURES,
) -> pd.DataFrame:
    """Both covariate blocks on `build_pitcher_cells`'s output.

    The same two calls, in the same order, at the same hyperparameters as
    `scripts/run_command_level_backtest.py`'s `attach_z`, so the coefficients
    fitted here are the coefficients the gate was scored on.
    """
    out = attach_command_features(cells, command_monthly, weights, ballast,
                                  features=level_features)
    return stuff_eval.attach_live_features(out, stuff_monthly,
                                           SERVED_STUFF_WEIGHTS,
                                           SERVED_STUFF_BALLAST)


def fit_live_command(
    component: str,
    seasons_table: pd.DataFrame,
    command_monthly: pd.DataFrame,
    stuff_monthly: pd.DataFrame,
    pa_dir,
    predict_year: int,
    weights: tuple[float, float, float] = SERVED_WEIGHTS,
    ballast: float = SERVED_BALLAST,
    level_features: tuple[str, ...] = LEVEL_FEATURES,
    fixed_base: bool = True,
) -> StuffFit:
    """`command_level_additive`'s coefficients for `predict_year`, fitted the
    way the harness fits them: on cell seasons strictly before the one being
    served, never on `predict_year` itself.

    The mirror of `src.eval.stuff.fit_live_stuff`, one weighted least squares
    with both blocks in it. `fixed_base=True` pins the baseline coefficient at
    exactly 1, so the fit is a pure correction added to
    `marcel_pitcher_tuned` and the six stuff controls alone *are* the served
    `stuff_additive` engine — which is what makes the command coefficients
    conditional on stuff and makes the increment over the served engine and
    the covariate-only share the same number (docs/pitching-command-level.md).
    """
    train_seasons = tuple(s for s in LIVE_CELL_SEASONS if s < predict_year)
    if not train_seasons:
        raise ValueError(
            f"no command training seasons strictly before {predict_year}")
    cells = build_pitcher_cells(seasons_table, pa_dir, [component],
                                seasons=train_seasons)
    if cells.empty:
        raise ValueError(f"no command training cells for {component!r} "
                         f"before {predict_year}")
    cells = attach_both_blocks(cells, command_monthly, stuff_monthly, weights,
                               ballast, level_features)
    return fit_stuff(cells, component, features=served_features(level_features),
                     fixed_base=fixed_base)


@dataclass
class CommandProviderConfig:
    """Everything the served command arm needs that a provider signature cannot
    carry. The mirror of `src.eval.stuff.StuffProviderConfig`, with the second
    artifact the joint fit reads."""
    command_monthly: pd.DataFrame
    stuff_monthly: pd.DataFrame
    cutoff: str
    predict_year: int
    fit: StuffFit
    base_provider: object
    weights: tuple[float, float, float] = SERVED_WEIGHTS
    ballast: float = SERVED_BALLAST
    stuff_weights: tuple[float, float, float] = SERVED_STUFF_WEIGHTS
    stuff_ballast: float = SERVED_STUFF_BALLAST
    level_features: tuple[str, ...] = LEVEL_FEATURES
    clip: tuple[float, float] = (1e-4, 0.999)


def command_provider(config: CommandProviderConfig):
    """A harness provider: the baseline plus the fitted stuff *and* command
    covariates, in one correction.

    Covers exactly the pitchers the baseline covers — a pitcher with no tracked
    pitches before the cutoff gets z = 0 on every covariate of either block and
    therefore the baseline itself, rather than being dropped. The common
    pitcher set stays the baseline's, so nothing is quietly scored on a
    different population.
    """

    def provider(train: pd.DataFrame, spec, predict_year: int):
        base = config.base_provider(train, spec, predict_year)
        ids = base[spec.id_col].to_numpy()
        zc = features_at_cutoff(config.command_monthly, config.cutoff,
                                config.predict_year, config.weights,
                                config.ballast, config.level_features
                                ).set_index("player").reindex(ids)
        zs = stuff_eval.features_at_cutoff(
            config.stuff_monthly, config.cutoff, config.predict_year,
            config.stuff_weights, config.stuff_ballast
        ).set_index("player").reindex(ids)
        z = pd.DataFrame(index=pd.RangeIndex(len(ids)))
        for f in STUFF_FEATURES:
            z[f] = zs[f].fillna(0.0).to_numpy()
        for f in config.level_features:
            z[f] = zc[f].fillna(0.0).to_numpy()
        pred = config.fit.predict(base["predicted"].to_numpy(dtype="float64"), z)
        out = base[[spec.id_col]].copy()
        out["predicted"] = np.clip(pred, *config.clip)
        return out

    return provider


__all__ = [
    "COMMAND_BALLAST_GRID", "COMMAND_WEIGHT_GRID", "CommandProviderConfig",
    "DEFAULT_COMMAND_BALLAST", "DEFAULT_COMMAND_WEIGHTS", "FEATURES",
    "LEVEL_FEATURES", "LEVEL_FEATURES_WITH_EDGE", "SERVED_BALLAST",
    "SERVED_STUFF_BALLAST", "SERVED_STUFF_WEIGHTS", "SERVED_WEIGHTS",
    "attach_both_blocks", "attach_command_features", "command_metrics",
    "command_provider", "features_at_cutoff", "fit_live_command",
    "league_profile", "served_features", "window_counts",
]
