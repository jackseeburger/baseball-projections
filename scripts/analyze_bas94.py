"""BAS-94's scoring pass: the prior's mean as a function of the profile
(docs/bayes-prior-mean.md).

`scripts/run_intraseason_backtest_dense.py --stage analyze` renders every
variant against a fixed list of bases. This script asks the five questions the
BAS-94 pre-registration asks, and only those, so what it writes maps
one-to-one onto predictions 1-5:

    1. vacuity — `gamma` on barrel or EV excludes zero for HR/PA and on whiff
       for K% at >= 90% of cutoffs, AND the between-player sd of the prior
       mean is >= 30% of `sigma_ability`. Both halves, because a coefficient
       that excludes zero and moves the target by a percent of the population
       spread is a relabelled league mean.
    2. the May cutoffs — arm A beats `bayes_walk` on HR/PA by >= 3% of MAE
       (clustered |t| > 2.5) and on K% by >= 2%.
    3. pooled over every cutoff — arm A vs `bayes_walk` is <= -1.5% on HR/PA
       (t < -2) and <= -1% on K%; arm A vs `contact_additive` is within
       +-1.5% on HR/PA, a draw written down in advance so a win or a loss is
       read as one.
    4. the mechanism — the August gap against `bayes_walk` is at most half
       the May gap: shrinkage matters less as outcomes accumulate.
    5. calibration — the 80% posterior interval covers the realised rate
       75-85% of the time, and arm A's interval is narrower than
       `bayes_walk`'s at May.

Arm B (`contact_cur`, the ballasted current-season feature) is reported beside
arm A everywhere and **never substitutes for it**: the pre-registration says
so in as many words, so every prediction above is scored on arm A and arm B's
row sits next to it, unscored.

**Where the comparators come from.** `bayes_walk`, `contact_additive` and
`marcel_tuned` were already fit on exactly these cells by BAS-85
(`data/eval/bas85/cells_bayes.parquet`), at the same draws, the same seasons
and the same twelve biweekly cutoffs. Refitting them would have cost 27 more
fit-hours and produced a second set of numbers for the same arms, so they are
read in and pinned rather than re-run; `marcel_tuned` is scored from *this*
run's own rows and the two sources are checked against each other on the way
in (`comparator_audit`), which is what says the two grids really are the same
cells. Pairing is an inner join per (season, cutoff, batter), so an arm
missing a batter simply drops that pair rather than silently scoring a
different population.

Every paired difference reuses the dense sweep's own `variant_comparison`, so
the clustering (by player, primary; by cell, reported; unclustered, labelled
wrong) is the same math as every other table on this scoreboard.

Reads the checkpoints and the fit records the sweep wrote; runs no MCMC, and
needs neither pymc nor arviz.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _dense():
    spec = importlib.util.spec_from_file_location(
        "run_intraseason_backtest_dense",
        ROOT / "scripts/run_intraseason_backtest_dense.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dense = _dense()

ARM_A = "bayes_walk+prior_contact"
ARM_B = "bayes_walk+prior_contact_cur"
WALK_ARM = "bayes_walk"
CONTACT_ARM = "contact_additive"
MARCEL_ARM = "marcel_tuned"

# Arms copied in from the BAS-85 grid rather than refit. `marcel_tuned` is not
# here: this run computes it itself (it is one of the sweep's own cheap
# baselines) and the copy is used only to audit that the two grids agree.
COMPARATOR_ARMS = (WALK_ARM, CONTACT_ARM)

# arm, base — in the order the predictions read them.
COMPARISONS = [
    (ARM_A, WALK_ARM),         # predictions 2, 3, 4
    (ARM_A, CONTACT_ARM),      # prediction 3's second half
    (ARM_A, MARCEL_ARM),
    (ARM_B, WALK_ARM),         # reported beside A, never in place of it
    (ARM_B, CONTACT_ARM),
    (ARM_B, ARM_A),
    (WALK_ARM, CONTACT_ARM),
    (CONTACT_ARM, MARCEL_ARM),
]

# The two regimes predictions 2 and 4 name, by the cutoff's month.
REGIMES = {"may": (5,), "august": (8,)}

# Prediction 1.
GAMMA_CUTOFF_SHARE = 0.90
SD_RATIO_FLOOR = 0.30
# Which coefficient carries the claim, per component. The pre-registration
# says "`gamma` on barrel or EV ... for HR/PA and on whiff for K%", so HR/PA
# passes at a cutoff when ANY of the named coefficients excludes zero there.
# "EV" is both exit-velocity features — mean EV and EV90 — because the
# pre-registration names the quantity, not one of the two columns that
# measure it, and picking one after seeing which moved would be choosing the
# test after the answer. Each coefficient's own count is reported beside the
# combined one, so the reading is visible rather than buried.
VACUITY_GAMMAS = {"hr_rate": ("gamma_barrel", "gamma_ev_mean", "gamma_ev90"),
                  "k_rate": ("gamma_whiff",)}

# Prediction 2, as percentages of the base arm's own MAE. `variant_comparison`
# signs a gain negative, so "beats by >= 3%" is pct <= -3.0.
MAY_GAIN_PCT = {"hr_rate": -3.0, "k_rate": -2.0}
MAY_T = 2.5
# Prediction 3.
POOLED_GAIN_PCT = {"hr_rate": -1.5, "k_rate": -1.0}
POOLED_T = -2.0
CONTACT_DRAW_BAND_PCT = 1.5
# Prediction 4: the August gap is at most half the May gap.
AUGUST_SHARE_OF_MAY = 0.5
# Prediction 5.
COVERAGE_TARGET = (0.75, 0.85)

# A fit is named and excluded in a sensitivity, never silently dropped.
RHAT_CEILING = 1.2

# The 80% interval's half-width in sds for the predictive reading, exactly as
# `scripts/analyze_bas85.py` uses it.
Z80 = 1.2815515655446004


# ─── loading, and the comparator splice ────────────────────────────────────

def cutoff_month(cutoff: str) -> int:
    return int(pd.Timestamp(cutoff).month)


def _cell_keys(cells: pd.DataFrame) -> set:
    return set(map(tuple, cells[["component", "season", "cutoff"]]
                   .drop_duplicates().to_numpy().tolist()))


def load_cells(in_dir: Path, comparator_dir: Path) -> tuple[pd.DataFrame, dict]:
    """This ticket's cells with the BAS-85 comparator rows spliced in.

    Only rows for (component, season, cutoff) cells this run actually scored
    are copied: a comparator row at a cutoff where no prior-mean arm was fit
    would inflate a pooled table with cells the arm under test never saw.
    """
    own = pd.read_parquet(in_dir / "cells_bayes.parquet")
    audit: dict = {"own_rows": int(len(own)),
                   "own_arms": sorted(own["model"].unique().tolist())}
    path = comparator_dir / "cells_bayes.parquet"
    if not path.exists():
        audit["comparators"] = f"{path} absent — comparator arms not spliced"
        return own, audit

    other = pd.read_parquet(path)
    keys = _cell_keys(own)
    key_col = list(zip(other["component"], other["season"], other["cutoff"]))
    other = other[[k in keys for k in key_col]]
    audit["comparator_source"] = str(path)
    audit["comparator_cells_matched"] = int(len(_cell_keys(other) & keys))
    audit["comparator_cells_wanted"] = int(len(keys))

    take = other[other["model"].isin(COMPARATOR_ARMS)]
    audit["comparator_rows"] = {m: int((take["model"] == m).sum())
                                for m in COMPARATOR_ARMS}
    audit["marcel_audit"] = comparator_audit(own, other)
    merged = pd.concat([own, take], ignore_index=True)
    return merged, audit


def comparator_audit(own: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Does this run's `marcel_tuned` agree with the grid we borrowed from?

    The one arm both grids computed independently. If these two disagree the
    cells are not the same cells and no borrowed comparator means anything,
    so it is checked rather than assumed — and reported as a number, not as a
    pass/fail, because the honest answer to "how well do they agree" is a
    maximum absolute difference.
    """
    def keyed(df):
        g = df[df["model"] == MARCEL_ARM]
        key = (g["component"].astype(str) + "|" + g["season"].astype(str) + "|"
               + g["cutoff"].astype(str) + "|" + g["batter"].astype(str))
        return g.assign(_key=key).set_index("_key")["predicted"]

    a, b = keyed(own), keyed(other)
    common = a.index.intersection(b.index)
    if common.empty:
        return {"n_common": 0}
    d = (a.loc[common] - b.loc[common]).abs()
    return {"n_common": int(len(common)), "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "n_only_here": int(len(a.index.difference(b.index))),
            "n_only_there": int(len(b.index.difference(a.index)))}


# ─── the paired tables ─────────────────────────────────────────────────────

def arm_mae(cells: pd.DataFrame, model: str, component: str) -> float:
    g = cells[(cells["component"] == component) & (cells["model"] == model)]
    if g.empty:
        return float("nan")
    err = (g["predicted"] - g["realized_rate"]).abs().to_numpy("float64")
    w = g["trials"].to_numpy("float64")
    return float(np.average(err, weights=w))


def _compare(cells: pd.DataFrame, arm: str, base: str, component: str) -> dict:
    r = dense.variant_comparison(cells, arm, base, component)
    if not r:
        return {}
    # MAE levels over the rows the two arms SHARE — `arm_mae` above is the
    # level over each arm's own rows, which is the same thing only when the
    # coverage is identical, and a percentage has to be of the base's error
    # on the pairs that produced the difference.
    r.update(dense.paired_mae(cells, arm, base, component))
    base_mae = r.get("base_mae") or float("nan")
    r["pct_of_base_mae"] = (100.0 * r["diff"] / base_mae
                            if base_mae and np.isfinite(base_mae)
                            else float("nan"))
    return r


def comparisons(cells: pd.DataFrame, component: str) -> list[dict]:
    present = set(cells.loc[cells["component"] == component, "model"].unique())
    rows = []
    for arm, base in COMPARISONS:
        if arm not in present or base not in present:
            continue
        r = _compare(cells, arm, base, component)
        if r:
            rows.append(r)
    return rows


def regime_split(cells: pd.DataFrame, arm: str, base: str,
                 component: str) -> dict:
    """The same paired comparison restricted to each regime's cutoffs — a
    subset of cells fed to the same function, not a second statistic."""
    months = cells["cutoff"].map(cutoff_month)
    out = {}
    for name, want in REGIMES.items():
        sub = cells[months.isin(want)]
        if sub.empty:
            continue
        r = _compare(sub, arm, base, component)
        if r:
            r["cutoffs"] = sorted(sub["cutoff"].unique().tolist())
            out[name] = r
    return out


def by_season(cells: pd.DataFrame, component: str) -> dict:
    """The headline comparisons per season. A result that exists in one season
    is a different claim from one that repeats across four, and a pooled table
    cannot tell them apart."""
    out = {}
    for season, g in cells.groupby("season"):
        rows = []
        for arm, base in ((ARM_A, WALK_ARM), (ARM_A, CONTACT_ARM),
                          (ARM_A, MARCEL_ARM), (ARM_B, WALK_ARM),
                          (ARM_B, CONTACT_ARM), (ARM_B, MARCEL_ARM),
                          (ARM_B, ARM_A)):
            r = _compare(g, arm, base, component)
            if r:
                rows.append(r)
        if rows:
            out[int(season)] = rows
    return out


def wins_by_cutoff(cells: pd.DataFrame, arm: str, base: str,
                   component: str) -> list[dict]:
    """W-L per cutoff — the pre-registration asks for the win/loss count by
    cutoff, and a pooled `arm_wins_cells` hides which part of the season the
    wins are in."""
    per = dense.paired_by_cell(cells[cells["component"] == component], arm, base)
    if per.empty:
        return []
    rows = []
    for md, g in per.groupby(per["cutoff"].str[5:]):
        rows.append({"cutoff_mmdd": md, "n_cells": int(len(g)),
                     "wins": int((g["diff"] < 0).sum()),
                     "losses": int((g["diff"] > 0).sum()),
                     "mean_diff": float(g["diff"].mean())})
    return sorted(rows, key=lambda r: r["cutoff_mmdd"])


# ─── prediction 1: vacuity ─────────────────────────────────────────────────

def gamma_by_cutoff(fits: list[dict], arm: str) -> dict:
    """`{component: {gamma: {cutoff: summary}}}` for one arm's fits."""
    out: dict = {}
    for f in fits:
        if f.get("arm") != arm:
            continue
        params = (f.get("prior_mean_params") or {}).get("gamma") or {}
        comp = f.get("component")
        for name, s in params.items():
            out.setdefault(comp, {}).setdefault(name, {})[f.get("cutoff")] = s
    return out


def sd_ratio_by_cutoff(fits: list[dict], arm: str) -> dict:
    out: dict = {}
    for f in fits:
        if f.get("arm") != arm:
            continue
        p = f.get("prior_mean_params") or {}
        if "sd_ratio" not in p:
            continue
        out.setdefault(f.get("component"), {})[f.get("cutoff")] = {
            "sd_ratio": p["sd_ratio"],
            "prior_mean_sd": p.get("prior_mean_sd_between_players"),
            "sigma_ability": p.get("sigma_ability"),
            "q05": p.get("sd_ratio_q05"), "q95": p.get("sd_ratio_q95"),
        }
    return out


def score_prediction_1(fits: list[dict], arm: str = ARM_A) -> dict:
    """"The prior moves": a named coefficient excludes zero at >= 90% of
    cutoffs, AND the prior mean's between-player sd is >= 30% of
    `sigma_ability`. Both halves have to hold — the pre-registration joins
    them with "and", and each alone is satisfiable by a model that does
    nothing.
    """
    gammas = gamma_by_cutoff(fits, arm)
    ratios = sd_ratio_by_cutoff(fits, arm)
    out: dict = {"arm": arm,
                 "thresholds": {"share_of_cutoffs": GAMMA_CUTOFF_SHARE,
                                "sd_ratio_floor": SD_RATIO_FLOOR},
                 "gamma": {}, "sd_ratio": {}}
    holds = []
    for component, wanted in VACUITY_GAMMAS.items():
        per = gammas.get(component) or {}
        if not per:
            continue
        cutoffs = sorted({c for v in per.values() for c in v})
        # "barrel OR EV" for HR/PA: a cutoff counts when ANY of the named
        # coefficients excludes zero there.
        excl = 0
        for cut in cutoffs:
            if any((per.get(name, {}).get(cut) or {}).get("excludes_zero")
                   for name in wanted):
                excl += 1
        detail = {}
        for name in wanted:
            p = per.get(name) or {}
            means = np.asarray([s["mean"] for s in p.values()], dtype="float64")
            detail[name] = {
                "n_cutoffs": len(p),
                "n_excluding_zero": sum(1 for s in p.values()
                                        if s.get("excludes_zero")),
                "mean_of_means": float(means.mean()) if means.size else float("nan"),
                "min": float(means.min()) if means.size else float("nan"),
                "max": float(means.max()) if means.size else float("nan"),
            }
        share = excl / len(cutoffs) if cutoffs else 0.0
        gamma_ok = bool(cutoffs) and share >= GAMMA_CUTOFF_SHARE
        out["gamma"][component] = {
            "coefficients": list(wanted), "n_cutoffs": len(cutoffs),
            "n_excluding_zero": excl, "share": share, "holds": gamma_ok,
            "per_coefficient": detail,
        }

        per_ratio = ratios.get(component) or {}
        vals = np.asarray([s["sd_ratio"] for s in per_ratio.values()],
                          dtype="float64")
        vals = vals[np.isfinite(vals)]
        ratio_ok = bool(vals.size) and float(vals.mean()) >= SD_RATIO_FLOOR
        out["sd_ratio"][component] = {
            "n_cutoffs": int(vals.size),
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "min": float(vals.min()) if vals.size else float("nan"),
            "max": float(vals.max()) if vals.size else float("nan"),
            "n_at_or_above_floor": int((vals >= SD_RATIO_FLOOR).sum()),
            "holds": ratio_ok,
        }
        holds.append(gamma_ok and ratio_ok)
    out["holds"] = bool(holds) and all(holds)
    return out


# ─── predictions 2, 3, 4: the paired thresholds ────────────────────────────

def score_prediction_2(splits: dict) -> dict:
    """May: arm A beats `bayes_walk` by >= 3% of MAE on HR/PA with |t| > 2.5,
    and by >= 2% on K%.

    The |t| condition is written inside the HR/PA clause of the
    pre-registration and is applied there; K%'s t is reported beside its
    percentage but does not gate it. That reading is recorded here rather
    than decided after seeing the numbers.
    """
    out: dict = {"thresholds": {"may_gain_pct": MAY_GAIN_PCT, "may_t": MAY_T,
                                "t_gates": ["hr_rate"]}}
    holds = []
    for component, threshold in MAY_GAIN_PCT.items():
        may = (splits.get(component) or {}).get("may")
        if not may:
            continue
        t = may.get("clustered_by_player_t")
        pct = may["pct_of_base_mae"]
        ok = bool(np.isfinite(pct) and pct <= threshold)
        if component == "hr_rate":
            ok = ok and t is not None and np.isfinite(t) and abs(t) > MAY_T
        out[component] = {"pct_of_base_mae": pct, "clustered_by_player_t": t,
                          "diff": may["diff"], "n": may["n"],
                          "threshold_pct": threshold, "holds": ok}
        holds.append(ok)
    out["holds"] = bool(holds) and all(holds)
    return out


def score_prediction_3(pooled_walk: dict, pooled_contact: dict) -> dict:
    """Pooled: <= -1.5% on HR/PA (t < -2) and <= -1% on K% against
    `bayes_walk`; within +-1.5% of `contact_additive` on HR/PA."""
    out: dict = {"thresholds": {"pooled_gain_pct": POOLED_GAIN_PCT,
                                "pooled_t": POOLED_T,
                                "contact_draw_band_pct": CONTACT_DRAW_BAND_PCT}}
    holds = []
    for component, threshold in POOLED_GAIN_PCT.items():
        r = pooled_walk.get(component)
        if not r:
            continue
        t = r.get("clustered_by_player_t")
        pct = r["pct_of_base_mae"]
        ok = bool(np.isfinite(pct) and pct <= threshold)
        if component == "hr_rate":
            ok = ok and t is not None and np.isfinite(t) and t < POOLED_T
        out[f"vs_bayes_walk_{component}"] = {
            "pct_of_base_mae": pct, "diff": r["diff"],
            "clustered_by_player_t": t, "threshold_pct": threshold,
            "holds": ok}
        holds.append(ok)
    r = pooled_contact.get("hr_rate")
    if r:
        pct = r["pct_of_base_mae"]
        ok = bool(np.isfinite(pct) and abs(pct) <= CONTACT_DRAW_BAND_PCT)
        out["vs_contact_additive_hr_rate"] = {
            "pct_of_base_mae": pct, "diff": r["diff"],
            "clustered_by_player_t": r.get("clustered_by_player_t"),
            "holds": ok}
        holds.append(ok)
    out["holds"] = bool(holds) and all(holds)
    return out


def score_prediction_4(splits: dict) -> dict:
    """The gain over `bayes_walk` shrinks from May to August: the August gap
    is at most half the May gap.

    Only meaningful where May is actually a gain — "the gain shrinks" is not
    a claim about an arm that lost in May, and reporting `|August| <= 0.5 *
    |May|` for a losing May would score a bigger August loss as a pass. So a
    May that is not a gain is recorded as such and the prediction fails
    rather than being read the flattering way.
    """
    out: dict = {"thresholds": {"august_share_of_may": AUGUST_SHARE_OF_MAY}}
    holds = []
    for component, split in splits.items():
        may, aug = split.get("may"), split.get("august")
        if not may or not aug:
            continue
        may_pct, aug_pct = may["pct_of_base_mae"], aug["pct_of_base_mae"]
        may_is_gain = bool(np.isfinite(may_pct) and may_pct < 0)
        ok = bool(may_is_gain and np.isfinite(aug_pct)
                  and aug_pct >= AUGUST_SHARE_OF_MAY * may_pct)
        out[component] = {
            "may_pct": may_pct, "august_pct": aug_pct,
            "may_is_a_gain": may_is_gain,
            "august_over_may": (aug_pct / may_pct
                                if may_pct else float("nan")),
            "holds": ok}
        holds.append(ok)
    out["holds"] = bool(holds) and all(holds)
    return out


# ─── prediction 5: coverage and width ──────────────────────────────────────

def _band(g: pd.DataFrame, predictive: bool):
    """`(lo, hi)` for one arm's cells, either interval.

    `pred_q10`/`pred_q90` is an interval on the **rate**; what it is scored
    against is a realised rate over a finite number of trials, which carries
    binomial noise the rate's posterior does not. Both readings are computed;
    the verdict is taken on the predictive one, which is the reading under
    which "covers 75-85%" is something a correct model can satisfy. Same
    construction as `scripts/analyze_bas85.py`, so the two tickets' coverage
    numbers are comparable.
    """
    if not predictive:
        return g["pred_q10"], g["pred_q90"]
    p = g["predicted"].clip(1e-6, 1 - 1e-6)
    sd = np.sqrt(g["pred_sd"] ** 2 + p * (1 - p) / g["trials"])
    return g["predicted"] - Z80 * sd, g["predicted"] + Z80 * sd


def coverage(cells: pd.DataFrame, model: str, component: str,
             months=None, predictive: bool = False) -> dict:
    """How often the 80% interval covers the realised rate. Unweighted over
    cells: every batter's interval is one claim."""
    g = cells[(cells["component"] == component) & (cells["model"] == model)]
    if months is not None:
        g = g[g["cutoff"].map(cutoff_month).isin(months)]
    need = ["pred_sd"] if predictive else ["pred_q10", "pred_q90"]
    if g.empty or not set(need) <= set(g.columns):
        return {"n": 0, "covered": None}
    g = g.dropna(subset=need)
    if g.empty:
        return {"n": 0, "covered": None}
    lo, hi = _band(g, predictive)
    inside = (g["realized_rate"] >= lo) & (g["realized_rate"] <= hi)
    below = float((g["realized_rate"] < lo).mean())
    return {"n": int(len(g)), "covered": float(inside.mean()),
            "miss_low": below, "miss_high": float(1.0 - inside.mean() - below),
            "mean_width": float((hi - lo).mean())}


def score_prediction_5(cells: pd.DataFrame, component: str) -> dict:
    """Coverage in 75-85%, and arm A's May interval narrower than
    `bayes_walk`'s.

    The width comparison is on the RATE interval, not the predictive one: the
    predictive band adds the same binomial term to both arms at the same
    trial counts, so comparing predictive widths measures mostly the shared
    noise. Both are reported; the verdict reads the rate interval, which is
    the posterior the pre-registration is talking about.
    """
    lo, hi = COVERAGE_TARGET
    out: dict = {"target": list(COVERAGE_TARGET), "scored_on": "predictive",
                 "width_scored_on": "rate_only"}
    for reading in ("predictive", "rate_only"):
        pred = reading == "predictive"
        out[reading] = {
            ARM_A: coverage(cells, ARM_A, component, predictive=pred),
            f"{ARM_A}_may": coverage(cells, ARM_A, component, REGIMES["may"], pred),
            ARM_B: coverage(cells, ARM_B, component, predictive=pred),
            f"{ARM_B}_may": coverage(cells, ARM_B, component, REGIMES["may"], pred),
            WALK_ARM: coverage(cells, WALK_ARM, component, predictive=pred),
            f"{WALK_ARM}_may": coverage(cells, WALK_ARM, component,
                                        REGIMES["may"], pred),
            CONTACT_ARM: coverage(cells, CONTACT_ARM, component, predictive=pred),
        }
    a = out["predictive"][ARM_A]
    a_may = out["rate_only"][f"{ARM_A}_may"]
    w_may = out["rate_only"][f"{WALK_ARM}_may"]
    covered_ok = bool(a["covered"] is not None and lo <= a["covered"] <= hi)
    narrower = bool(a_may.get("mean_width") is not None
                    and w_may.get("mean_width") is not None
                    and a_may["covered"] is not None
                    and w_may["covered"] is not None
                    and a_may["mean_width"] < w_may["mean_width"])
    out["coverage_holds"] = covered_ok
    out["narrower_at_may"] = narrower
    out["may_width_arm_a"] = a_may.get("mean_width")
    out["may_width_bayes_walk"] = w_may.get("mean_width")
    out["holds"] = bool(covered_ok and narrower)
    return out


# ─── convergence ───────────────────────────────────────────────────────────

def nonconverged_fits(fits: list[dict], arms=(ARM_A, ARM_B)) -> dict:
    """`{(arm, component, cutoff): why}` for prior-mean fits above the R-hat
    ceiling, flattened to a string key so it survives JSON."""
    out: dict = {}
    for f in fits:
        if f.get("arm") not in arms:
            continue
        d = f.get("diagnostics") or {}
        rhat = d.get("max_rhat")
        div = d.get("divergences") or 0
        if rhat is not None and np.isfinite(rhat) and rhat > RHAT_CEILING:
            out[f"{f.get('arm')}|{f.get('component')}|{f.get('cutoff')}"] = (
                f"R-hat {rhat:.3f} > {RHAT_CEILING} on {d.get('max_rhat_var')}"
                f"; {int(div)} divergences")
    return out


def drop_nonconverged(cells: pd.DataFrame, bad: dict) -> pd.DataFrame:
    """Every arm's rows for a (component, cutoff) a prior-mean fit failed at.

    The whole cell, not just the broken arm's rows: the tables are paired, so
    removing one arm from a cell would leave the others scored on a
    population the comparison no longer covers.
    """
    if not bad:
        return cells
    drop = {(k.split("|")[1], k.split("|")[2]) for k in bad}
    keep = [(c, cu) not in drop
            for c, cu in zip(cells["component"], cells["cutoff"])]
    return cells[keep]


def fit_diagnostics_summary(fits: list[dict], arms=(ARM_A, ARM_B)) -> dict:
    out: dict = {}
    for arm in arms:
        rows = [f for f in fits if f.get("arm") == arm]
        if not rows:
            continue
        rhat = np.array([(f.get("diagnostics") or {}).get("max_rhat", np.nan)
                         for f in rows], dtype="float64")
        ess = np.array([(f.get("diagnostics") or {}).get("min_ess_bulk", np.nan)
                        for f in rows], dtype="float64")
        div = np.array([(f.get("diagnostics") or {}).get("divergences", 0)
                        for f in rows], dtype="float64")
        el = np.array([f.get("elapsed_s", np.nan) for f in rows],
                      dtype="float64")
        el = el[np.isfinite(el)]
        out[arm] = {
            "n_fits": len(rows),
            "max_rhat": {"median": float(np.nanmedian(rhat)),
                         "max": float(np.nanmax(rhat)),
                         "n_above_ceiling": int(np.nansum(rhat > RHAT_CEILING))},
            "min_ess_bulk": {"median": float(np.nanmedian(ess)),
                             "min": float(np.nanmin(ess))},
            "divergences": {"total": int(np.nansum(div)),
                            "max": int(np.nanmax(div)) if div.size else 0,
                            "n_fits_with_any": int(np.nansum(div > 0))},
            "elapsed_s": ({"median": float(np.median(el)),
                           "min": float(el.min()), "max": float(el.max()),
                           "total_hours": float(el.sum() / 3600.0)}
                          if el.size else {}),
            "samplers": sorted({str(f.get("sampler")) for f in rows}),
        }
    return out


# ─── rendering ─────────────────────────────────────────────────────────────

def _num(v, w, p=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return f"{'-':>{w}}"
    return f"{'-':>{w}}" if not np.isfinite(v) else f"{v:>{w}.{p}f}"


def render(rows: list[dict]) -> str:
    if not rows:
        return "(nothing scored)"
    aw = max(len(r["arm"]) for r in rows) + 2
    bw = max(len(r["base"]) for r in rows) + 2
    head = (f"{'arm':<{aw}}{'base':<{bw}}{'n':>6}  {'diff':>10}  {'% base':>7}  "
            f"{'t(player)':>10}  {'t(cell)':>8}  {'W-L':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['arm']:<{aw}}{r['base']:<{bw}}{r['n']:>6}  {r['diff']:>+10.5f}  "
            f"{_num(r['pct_of_base_mae'], 7, 2)}  "
            f"{_num(r['clustered_by_player_t'], 10)}  "
            f"{_num(r['clustered_by_cell_t'], 8)}  "
            f"{r['arm_wins_cells']:>3d}-{r['arm_loses_cells']:<3d}")
    return "\n".join(lines)


def render_wl(rows: list[dict]) -> str:
    if not rows:
        return "(nothing scored)"
    lines = [f"{'mmdd':<8}{'W-L':>8}{'mean diff':>12}"]
    for r in rows:
        lines.append(f"{r['cutoff_mmdd']:<8}"
                     f"{r['wins']:>4d}-{r['losses']:<3d}"
                     f"{r['mean_diff']:>+12.5f}")
    return "\n".join(lines)


def render_gammas(fits: list[dict], arm: str) -> str:
    by = gamma_by_cutoff(fits, arm)
    if not by:
        return "(no prior-mean fits recorded)"
    lines = []
    for component, per in sorted(by.items()):
        lines.append(f"-- {component}")
        for name, cuts in sorted(per.items()):
            means = np.asarray([s["mean"] for s in cuts.values()],
                               dtype="float64")
            excl = sum(1 for s in cuts.values() if s.get("excludes_zero"))
            lines.append(f"   {name:<18} mean {means.mean():+.4f}  "
                         f"[{means.min():+.4f}, {means.max():+.4f}]  "
                         f"excludes zero {excl}/{len(cuts)}")
    return "\n".join(lines)


def render_sd_ratio(fits: list[dict], arm: str) -> str:
    by = sd_ratio_by_cutoff(fits, arm)
    if not by:
        return "(no prior-mean fits recorded)"
    lines = []
    for component, per in sorted(by.items()):
        v = np.asarray([s["sd_ratio"] for s in per.values()], dtype="float64")
        sd = np.asarray([s["prior_mean_sd"] for s in per.values()],
                        dtype="float64")
        sig = np.asarray([s["sigma_ability"] for s in per.values()],
                         dtype="float64")
        lines.append(f"   {component:<10} sd(prior mean) {sd.mean():.4f}  "
                     f"sigma_ability {sig.mean():.4f}  ratio {v.mean():.3f} "
                     f"[{v.min():.3f}, {v.max():.3f}]  "
                     f"{int((v >= SD_RATIO_FLOOR).sum())}/{v.size} at or above "
                     f"{SD_RATIO_FLOOR:.0%}")
    return "\n".join(lines)


def render_coverage(cov: dict) -> str:
    lo, hi = COVERAGE_TARGET
    lines = []
    for reading in ("predictive", "rate_only"):
        table = cov.get(reading) or {}
        tag = []
        if reading == cov.get("scored_on"):
            tag.append("coverage scored")
        if reading == cov.get("width_scored_on"):
            tag.append("width scored")
        lines.append(f"{reading}{' (' + ', '.join(tag) + ')' if tag else ''}: "
                     f"target {lo:.0%}-{hi:.0%}")
        for name, s in table.items():
            if not isinstance(s, dict) or "covered" not in s:
                continue
            if s["covered"] is None:
                lines.append(f"  {name:<36} (no interval)")
                continue
            lines.append(f"  {name:<36} {s['covered']:6.1%}  n={s['n']:<6d}  "
                         f"low {s['miss_low']:5.1%}  high {s['miss_high']:5.1%}  "
                         f"width {s['mean_width']:.5f}")
    return "\n".join(lines)


# ─── the verdict ───────────────────────────────────────────────────────────

def verdict(payload: dict) -> dict:
    """The pre-registration's own failure conditions, applied as written.

    * prediction 1 fails -> the profile carries no information the league mean
      does not, and the structural track is closed on public data at these
      sample sizes.
    * prediction 2 fails with 1 holding -> shrinking toward similar players
      does not help where it should; nothing ships.
    * what would ship: arm A beating `contact_additive` pooled at t < -2 on
      HR/PA or K% opens a serving ticket. Nothing else does, and arm B never
      substitutes for arm A.
    """
    p1 = payload["prediction_1_vacuity"]["holds"]
    p2 = payload["prediction_2_may"]["holds"]
    pooled = payload.get("pooled_vs_contact_additive") or {}
    beats_contact = {}
    for component, r in pooled.items():
        t = r.get("clustered_by_player_t")
        beats_contact[component] = bool(
            r.get("diff", 0) < 0 and t is not None and np.isfinite(t)
            and t < POOLED_T)
    ships = any(beats_contact.values())
    if not p1:
        headline = ("prediction 1 fails: the profile carries no information "
                    "the league mean does not. The structural track is closed "
                    "on public data at these sample sizes.")
    elif not p2:
        headline = ("prediction 1 holds and prediction 2 fails: the prior "
                    "moves, and shrinking toward similar players does not "
                    "help where it should. Nothing ships.")
    elif ships:
        headline = ("arm A beats contact_additive pooled at t < -2; a serving "
                    "ticket follows under docs/serving-rules.md.")
    else:
        headline = ("predictions 1 and 2 hold but arm A does not beat "
                    "contact_additive pooled; the arm stays behind the flag.")
    return {
        "predictions": {k: bool(payload[k]["holds"]) for k in (
            "prediction_1_vacuity", "prediction_2_may", "prediction_3_pooled",
            "prediction_4_mechanism")},
        "prediction_5_calibration": {
            c: bool(v["holds"])
            for c, v in payload["prediction_5_calibration"].items()},
        "arm_a_beats_contact_additive_pooled": beats_contact,
        "ships": bool(ships),
        "headline": headline,
    }


# ─── RESULT.md ─────────────────────────────────────────────────────────────

def _md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(nothing scored)_\n"
    out = ["| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(
            f"| `{r['arm']}` | `{r['base']}` | {r['n']} | {r['diff']:+.5f} | "
            f"{r['pct_of_base_mae']:+.2f}% | "
            f"{_num(r['clustered_by_player_t'], 1)} | "
            f"{_num(r['clustered_by_cell_t'], 1)} | "
            f"{r['arm_wins_cells']}–{r['arm_loses_cells']} |")
    return "\n".join(out) + "\n"


def render_result_md(payload: dict) -> str:
    v = payload["verdict"]
    lines = [
        "# BAS-94 — the prior's mean as a function of the profile",
        "",
        f"Scope: seasons {payload['scope']['seasons']}, "
        f"{len(payload['scope']['cutoffs'])} cutoffs, components "
        f"{payload['scope']['components']}, arms under test "
        f"`{ARM_A}` (A) and `{ARM_B}` (B); comparators `{WALK_ARM}`, "
        f"`{CONTACT_ARM}`, `{MARCEL_ARM}`.",
        "",
        f"**Verdict: {v['headline']}**",
        "",
        "## Predictions",
        "",
        "| # | pre-registered | verdict |",
        "| --- | --- | --- |",
    ]
    titles = {
        "prediction_1_vacuity": "the prior moves (gamma excludes zero at ≥90% "
                                "of cutoffs; sd(prior mean) ≥ 30% of sigma_ability)",
        "prediction_2_may": "May: arm A beats bayes_walk by ≥3% on HR/PA "
                            "(|t|>2.5) and ≥2% on K%",
        "prediction_3_pooled": "pooled: ≤−1.5% HR/PA (t<−2), ≤−1% K% vs "
                               "bayes_walk; within ±1.5% of contact_additive on HR/PA",
        "prediction_4_mechanism": "August gap ≤ half the May gap",
    }
    for i, (key, title) in enumerate(titles.items(), start=1):
        lines.append(f"| {i} | {title} | "
                     f"{'PASS' if payload[key]['holds'] else 'FAIL'} |")
    cal = payload["prediction_5_calibration"]
    cal_txt = ", ".join(f"{c}: {'PASS' if s['holds'] else 'FAIL'}"
                        for c, s in cal.items())
    lines += [f"| 5 | 80% interval covers 75–85%; arm A narrower than "
              f"bayes_walk at May | {cal_txt} |", ""]

    for component in payload["scope"]["components"]:
        lines += [f"## {component} — pooled over every cutoff", "",
                  _md_table(payload["comparisons"][component]), ""]
        conv = (payload.get("converged_only") or {}).get("comparisons", {})
        if component in conv:
            lines += [f"### {component} — converged fits only", "",
                      _md_table(conv[component]), ""]
        per = payload["by_season"].get(component) or {}
        if len(per) > 1:
            lines += [f"### {component} — by season", ""]
            for season, rows in sorted(per.items()):
                lines += [f"**{season}**", "", _md_table(rows), ""]
    lines += ["## Fit diagnostics", "",
              "```", json.dumps(payload["fit_diagnostics"], indent=1), "```", ""]
    if payload.get("nonconverged_fits"):
        lines += ["### Fits above the R-hat ceiling (scored in the headline, "
                  "excluded in the sensitivity)", ""]
        for k, why in sorted(payload["nonconverged_fits"].items()):
            lines.append(f"- `{k}` — {why}")
        lines.append("")
    else:
        lines += ["No fit exceeded the R-hat ceiling of "
                  f"{RHAT_CEILING}.", ""]
    return "\n".join(lines)


# ─── main ──────────────────────────────────────────────────────────────────

def build_payload(cells: pd.DataFrame, fits: list[dict],
                  audit: dict) -> dict:
    components = sorted(cells["component"].unique().tolist())
    payload: dict = {
        "scope": {
            "seasons": sorted(int(s) for s in cells["season"].unique()),
            "cutoffs": sorted(cells["cutoff"].unique().tolist()),
            "components": components,
            "arms": sorted(cells["model"].unique().tolist()),
            "arm_a": ARM_A, "arm_b": ARM_B,
        },
        "provenance": audit,
        "fit_diagnostics": fit_diagnostics_summary(fits),
        "comparisons": {}, "by_season": {}, "wins_by_cutoff": {},
        "regime_split": {}, "pooled_vs_bayes_walk": {},
        "pooled_vs_contact_additive": {},
        "gamma_by_cutoff": {ARM_A: gamma_by_cutoff(fits, ARM_A),
                            ARM_B: gamma_by_cutoff(fits, ARM_B)},
        "sd_ratio_by_cutoff": {ARM_A: sd_ratio_by_cutoff(fits, ARM_A),
                               ARM_B: sd_ratio_by_cutoff(fits, ARM_B)},
    }
    for component in components:
        rows = comparisons(cells, component)
        payload["comparisons"][component] = rows
        for r in rows:
            if r["arm"] == ARM_A and r["base"] == WALK_ARM:
                payload["pooled_vs_bayes_walk"][component] = r
            if r["arm"] == ARM_A and r["base"] == CONTACT_ARM:
                payload["pooled_vs_contact_additive"][component] = r
        payload["by_season"][component] = {
            str(k): v for k, v in by_season(cells, component).items()}
        payload["wins_by_cutoff"][component] = {
            f"{ARM_A}_vs_{WALK_ARM}": wins_by_cutoff(cells, ARM_A, WALK_ARM,
                                                     component),
            f"{ARM_A}_vs_{CONTACT_ARM}": wins_by_cutoff(cells, ARM_A,
                                                        CONTACT_ARM, component),
            f"{ARM_B}_vs_{WALK_ARM}": wins_by_cutoff(cells, ARM_B, WALK_ARM,
                                                     component),
        }
        payload["regime_split"][component] = {
            f"{ARM_A}_vs_{WALK_ARM}": regime_split(cells, ARM_A, WALK_ARM,
                                                   component),
            f"{ARM_B}_vs_{WALK_ARM}": regime_split(cells, ARM_B, WALK_ARM,
                                                   component),
        }

    splits_a = {c: payload["regime_split"][c][f"{ARM_A}_vs_{WALK_ARM}"]
                for c in components}
    payload["prediction_1_vacuity"] = score_prediction_1(fits, ARM_A)
    payload["prediction_1_vacuity_arm_b"] = score_prediction_1(fits, ARM_B)
    payload["prediction_2_may"] = score_prediction_2(splits_a)
    payload["prediction_3_pooled"] = score_prediction_3(
        payload["pooled_vs_bayes_walk"], payload["pooled_vs_contact_additive"])
    payload["prediction_4_mechanism"] = score_prediction_4(splits_a)
    payload["prediction_5_calibration"] = {
        c: score_prediction_5(cells, c) for c in components}
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "data/eval/bas94")
    ap.add_argument("--comparator-dir", type=Path,
                    default=ROOT / "data/eval/bas85",
                    help="grid whose bayes_walk / contact_additive rows are "
                         "spliced in rather than refit (BAS-85 ran them on "
                         "exactly these cells)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--result-md", type=Path, default=None)
    args = ap.parse_args()

    cells, audit = load_cells(args.in_dir, args.comparator_dir)
    fits_path = args.in_dir / "bayes_fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else []
    merged_path = args.in_dir / "cells_scored.parquet"
    cells.to_parquet(merged_path, index=False)

    payload = build_payload(cells, fits, audit)

    bad = nonconverged_fits(fits)
    payload["nonconverged_fits"] = bad
    if bad:
        clean = drop_nonconverged(cells, bad)
        clean_fits = [
            f for f in fits
            if f"{f.get('arm')}|{f.get('component')}|{f.get('cutoff')}" not in bad
            and (f.get("component"), f.get("cutoff")) not in
            {(k.split("|")[1], k.split("|")[2]) for k in bad}]
        payload["converged_only"] = {
            "excluded": sorted(bad),
            "n_cutoff_cells_total": int(
                cells[["component", "cutoff"]].drop_duplicates().shape[0]),
            "n_cutoff_cells_kept": int(
                clean[["component", "cutoff"]].drop_duplicates().shape[0]),
            "comparisons": {c: comparisons(clean, c)
                            for c in payload["scope"]["components"]},
            "prediction_1_vacuity": score_prediction_1(clean_fits, ARM_A),
            "prediction_2_may": score_prediction_2({
                c: regime_split(clean, ARM_A, WALK_ARM, c)
                for c in payload["scope"]["components"]}),
            "prediction_5_calibration": {
                c: score_prediction_5(clean, c)
                for c in payload["scope"]["components"]},
        }

    payload["verdict"] = verdict(payload)

    # ── console ──
    print(f"scope: seasons {payload['scope']['seasons']}, "
          f"{len(payload['scope']['cutoffs'])} cutoffs, "
          f"arms {payload['scope']['arms']}")
    print(f"provenance: {json.dumps(audit)}")
    for component in payload["scope"]["components"]:
        print(f"\n=== {component} (pooled, all cutoffs) ===")
        print(render(payload["comparisons"][component]))
        if payload.get("converged_only"):
            print(f"\n=== {component} (converged fits only) ===")
            print(render(payload["converged_only"]["comparisons"][component]))
    for component in payload["scope"]["components"]:
        per = payload["by_season"][component]
        if len(per) > 1:
            print(f"\n=== {component} by season ===")
            for season, rows in sorted(per.items()):
                print(f"-- {season}")
                print(render(rows))
    for component in payload["scope"]["components"]:
        print(f"\n=== {component}: arm A vs bayes_walk, May / August ===")
        split = payload["regime_split"][component][f"{ARM_A}_vs_{WALK_ARM}"]
        print(render(list(split.values())))
        for name, s in split.items():
            print(f"  {name}: cutoffs {s['cutoffs']}")
        print(f"\n-- W-L by cutoff, arm A vs bayes_walk ({component})")
        print(render_wl(payload["wins_by_cutoff"][component]
                        [f"{ARM_A}_vs_{WALK_ARM}"]))

    print("\n=== gamma posteriors (arm A) ===")
    print(render_gammas(fits, ARM_A))
    print("\n=== gamma posteriors (arm B) ===")
    print(render_gammas(fits, ARM_B))
    print("\n=== vacuity: between-player sd of the prior mean (arm A) ===")
    print(render_sd_ratio(fits, ARM_A))

    for component, cov in payload["prediction_5_calibration"].items():
        print(f"\n=== coverage ({component}) ===")
        print(render_coverage(cov))

    print("\n=== fit diagnostics ===")
    print(json.dumps(payload["fit_diagnostics"], indent=1))
    if bad:
        print(f"\n!! {len(bad)} prior-mean fit(s) above R-hat "
              f"{RHAT_CEILING}; scored in the headline, excluded in the "
              f"sensitivity:")
        for k, why in sorted(bad.items()):
            print(f"   {k}  {why}")
    else:
        print(f"\nno fit exceeded R-hat {RHAT_CEILING}")

    print("\n=== pre-registered predictions ===")
    for key in ("prediction_1_vacuity", "prediction_2_may",
                "prediction_3_pooled", "prediction_4_mechanism"):
        print(f"  {key:<30} {'HOLDS' if payload[key]['holds'] else 'FAILS'}")
    for c, s in payload["prediction_5_calibration"].items():
        print(f"  prediction_5_calibration[{c}]".ljust(32)
              + ("HOLDS" if s["holds"] else "FAILS"))
    print(f"\nVERDICT: {payload['verdict']['headline']}")

    out = args.out or (args.in_dir / "analysis_bas94.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    md = args.result_md or (args.in_dir / "RESULT.md")
    md.write_text(render_result_md(payload))
    print(f"\n-> {out}\n-> {md}\n-> {merged_path}")


if __name__ == "__main__":
    main()
