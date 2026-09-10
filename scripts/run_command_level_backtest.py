"""Walk-forward test of the command **level** aggregates on the pitcher rates (BAS-87, stage 2).

BAS-76 built the command artifact, ran stage 1, and stopped at its own vacuity
gate: the stuff-differenced residual `cmd_resid` carries year to year at
pooled r 0.397 against a pre-registered 0.45. Its own diagnostic table showed
the *levels* carry fine (`cmd_csw` 0.757, `zone_share` 0.561, `waste_share`
0.650), and `docs/pitching-command.md` names the follow-up in as many words:
"a level aggregate with the stuff covariates as explicit controls in the
stage-2 fit is a different ticket and needs its own pre-registration". This is
that ticket's runner. Stage 1 is not rebuilt; the committed monthly artifact
is read as it stands.

**Why the controls are not optional.** A level is what the location-aware
`pitching` model thinks of a pitcher's pitches, and that model sees the stuff
features too — `docs/pitching-stuff.md` records that the "location-free" stuff
arm reaches AUC .966 on called-strike-given-taken, so the two blocks overlap
heavily in both directions. Entering the six aggregates the served
`stuff_additive` engine uses as explicit covariates in the same weighted least
squares is what makes the command coefficients conditional on stuff, and it
makes the recalibration control mean the right thing: with the baseline pinned
at 1, the control *is* the served engine.

Arms at every (component, season, cutoff) cell, scored on the harness's own
common pitcher set:

    marcel_pitcher_tuned            the live baseline, untouched
    command_level_recal             free fit, stuff controls only, no command
    command_level                   the same free fit plus the command levels
    stuff_additive                  baseline pinned at 1, stuff controls only
                                    — fitted by the same call BAS-79 serves,
                                    so it is the served engine and doubles as
                                    the additive recalibration control
    command_level_additive          baseline pinned at 1, stuff controls plus
                                    the command levels. `stuff_additive` +
                                    `command_level_additive`, in one fit, so
                                    the command coefficients are conditional
                                    on stuff
    command_level_shuffled          the command columns permuted across
                                    pitchers within a cell, stuff left alone
                                    (methods.md §5.4); free and additive
    stuff_then_command_level        the two-stage reading of the same
                                    question: the served engine's *prediction*
                                    as the base, command levels fitted on top
                                    with its coefficients frozen

`command_level_additive` minus `stuff_additive` is simultaneously the
covariate-only share the serving rule asks for and the increment over the
served engine, because with the baseline pinned the recalibration control and
the served engine are the same fit. Both readings are printed under their own
names rather than one being quietly reused for the other.

    python scripts/run_command_level_backtest.py --tune --shuffle-control

The §6 split of contact-quality.md is printed unconditionally: the paired gain
by pitcher-exposure tercile and by cutoff, as a percent of the baseline's own
MAE on that slice, so an error scale that shrinks with the season cannot
masquerade as a fading effect.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.pitching_command import load_monthly as load_command_monthly
from src.data.pitching_command import year_over_year
from src.data.pitching_stuff import load_monthly as load_stuff_monthly
from src.eval import command as command_eval
from src.eval import pitchers as pitcher_eval
from src.eval import stuff as stuff_eval
from src.eval.backtest import score
from src.eval.command import (
    COMMAND_BALLAST_GRID,
    COMMAND_WEIGHT_GRID,
    DEFAULT_COMMAND_BALLAST,
    DEFAULT_COMMAND_WEIGHTS,
    LEVEL_FEATURES,
    LEVEL_FEATURES_WITH_EDGE,
)
from src.eval.stuff import FEATURES as STUFF_FEATURES
from src.eval.stuff import fit_stuff
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("command-level-backtest")
ROOT = Path(__file__).resolve().parent.parent

CUTOFF_MONTHS = ("05-01", "07-01", "08-01")
# 2017 is the first season with two prior Statcast seasons and two prior
# seasons in the API season table. 2020 is excluded everywhere in this repo: a
# 60-game season that started July 23 has no May 1 cutoff.
CELL_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
TUNE_THROUGH = 2021

DEFAULT_COMPONENTS = ("p_k_rate", "p_bb_rate", "p_bbhbp_rate", "p_hr_rate")
BASE_ARM = "marcel_pitcher_tuned"
SERVED_ARM = "stuff_additive"
ADDITIVE_ARM = "command_level_additive"
FREE_ARM = "command_level"
FREE_RECAL_ARM = "command_level_recal"
TWO_STAGE_ARM = "stuff_then_command_level"

ARMS = ((FREE_ARM, BASE_ARM), (FREE_RECAL_ARM, BASE_ARM),
        (FREE_ARM, FREE_RECAL_ARM),
        (ADDITIVE_ARM, BASE_ARM), (SERVED_ARM, BASE_ARM),
        (ADDITIVE_ARM, SERVED_ARM),
        (TWO_STAGE_ARM, BASE_ARM), (TWO_STAGE_ARM, SERVED_ARM))
SHUFFLE_ARMS = (("command_level_shuffled", BASE_ARM),
                ("command_level_shuffled", FREE_RECAL_ARM),
                ("command_level_additive_shuffled", BASE_ARM),
                ("command_level_additive_shuffled", SERVED_ARM))

# Labelled robustness runs, all of them declared here rather than chosen after
# a table was read: the hyperparameters pinned at stuff's own defaults instead
# of the tuning window's pick, and the block with the artifact's edge region
# (`shadow_share`) added, which is the nearest thing it has to the
# `edge_share` the pre-registration allows for.
SENSITIVITIES = (
    ("stuff-pinned hyperparameters", (1.0, 0.0, 0.0), 10.0, None),
    ("edge share added", None, None, LEVEL_FEATURES_WITH_EDGE),
)

# architecture.md §3 as revised by BAS-82: a layer-1 measurement is served as a
# covariate only if the arm clears the gate *and* its covariate-only share is
# worth at least this much of the served baseline's MAE at this |t|.
SERVE_MIN_PCT = 1.0
SERVE_MIN_T = 2.0

# The pre-registration's own bands, written down before any number was read.
PREREG = {
    "bb_total_pct": (-1.5, -3.0),      # command_level_additive vs baseline
    "bb_total_min_t": 2.5,
    "bb_cov_min_pct": 1.0,             # share vs the stuff-only control
    "bb_cov_min_t": 2.0,
    "incremental_min_pct": 1.0,        # over stuff_additive, BB/BF
    "incremental_min_t": 2.0,
    "k_incremental_max_pct": 0.75,     # over stuff_additive, K/BF
    "k_incremental_max_t": 2.5,
    "hr_max_pct": 1.0,                 # HR/BF movement
    "vacuity_min_r": 0.45,
}


# --- cells -------------------------------------------------------------------

def build_cells(components, seasons_table: pd.DataFrame, pa_dir: Path,
                min_trials: int) -> pd.DataFrame:
    """One row per (component, season, cutoff, pitcher).

    `src.eval.stuff.build_pitcher_cells` builds it — the same rows the stuff
    gate was scored on and the same rows the served arm is fitted on, so the
    incremental comparison below is not quietly run on a different population.
    """
    return stuff_eval.build_pitcher_cells(
        seasons_table, pa_dir, components, seasons=CELL_SEASONS,
        cutoff_months=CUTOFF_MONTHS, min_trials=min_trials)


def attach_z(cells: pd.DataFrame, command_monthly: pd.DataFrame,
             stuff_monthly: pd.DataFrame, weights, ballast,
             level_features=LEVEL_FEATURES) -> pd.DataFrame:
    """Merge both covariate blocks onto the cells.

    The stuff block is attached at *its own* pinned hyperparameters
    (`src.eval.stuff.DEFAULT_WINDOW_WEIGHTS` / `DEFAULT_BALLAST`), not at the
    command sweep's, because the control has to be the engine that is actually
    served rather than a re-tuned version of it. The command block moves with
    the sweep.
    """
    out = command_eval.attach_command_features(
        cells, command_monthly, weights, ballast, features=level_features)
    return stuff_eval.attach_live_features(out, stuff_monthly)


def shuffle_z(cells: pd.DataFrame, level_features, seed: int = 0) -> pd.DataFrame:
    """Permute the **command** covariates across pitchers within each cell.

    The stuff block is left attached to its own pitcher, so the permuted arm
    is the served engine plus a command vector that belongs to somebody else.
    Every pitcher keeps a real command vector — same marginal, same shrinkage,
    same standardization — so an arm fitted and scored on this must land on
    the stuff-only control. If it beats it, the pipeline is fitting the split
    rather than the covariate.

    The pre-registration says "within season"; the permutation here is within
    (season, cutoff), which is strictly tighter — it preserves each cutoff's
    own marginal distribution exactly, where a season-wide shuffle would let a
    May command vector land on an August row.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _, g in cells.groupby(["season", "cutoff"], sort=False):
        g = g.copy()
        perm = rng.permutation(len(g))
        for f in level_features:
            g[f] = g[f].to_numpy()[perm]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --- the arms ----------------------------------------------------------------

def walk_forward(cells: pd.DataFrame, components, score_seasons,
                 level_features=LEVEL_FEATURES,
                 shuffled: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predictions for every arm, refitting coefficients on prior seasons only."""
    both = tuple(STUFF_FEATURES) + tuple(level_features)
    frames = []
    for component in components:
        g = cells[cells["component"] == component]
        for season in score_seasons:
            past = g[g["season"] < season]
            here = g[g["season"] == season]
            if past.empty or here.empty:
                continue
            fits = {
                FREE_RECAL_ARM: fit_stuff(past, component,
                                          features=STUFF_FEATURES),
                FREE_ARM: fit_stuff(past, component, features=both),
                # Pinned baseline, stuff only. This is byte-for-byte the call
                # `src.eval.stuff.fit_live_stuff` makes for the served engine,
                # so this arm *is* `stuff_additive` and is simultaneously the
                # additive recalibration control.
                SERVED_ARM: fit_stuff(past, component, features=STUFF_FEATURES,
                                      fixed_base=True),
                ADDITIVE_ARM: fit_stuff(past, component, features=both,
                                        fixed_base=True),
            }
            base = here["base"].to_numpy(dtype="float64")
            preds = {BASE_ARM: base}
            for name, f in fits.items():
                preds[name] = np.clip(f.predict(base, here), 1e-4, 0.999)
            rows = {}

            # The two-stage reading: the served engine's prediction as the
            # base, command levels fitted on top with the stuff coefficients
            # frozen. `stuff_additive`'s prediction on the training rows is
            # in-sample for its own fit, which is why the joint fit above is
            # the pre-registered arm and this one is the check.
            sa = fits[SERVED_ARM]
            past2 = past.copy()
            past2["base"] = np.clip(
                sa.predict(past["base"].to_numpy(dtype="float64"), past),
                1e-4, 0.999)
            two = fit_stuff(past2, component, features=level_features,
                            fixed_base=True)
            here2 = here.copy()
            here2["base"] = preds[SERVED_ARM]
            preds["stuff_then_command_level"] = np.clip(
                two.predict(here2["base"].to_numpy(dtype="float64"), here2),
                1e-4, 0.999)
            fits["stuff_then_command_level"] = two

            if shuffled is not None:
                sg = shuffled[shuffled["component"] == component]
                sh = sg[sg["season"] == season]
                sbase = sh["base"].to_numpy(dtype="float64")
                for name, kw in (("command_level_shuffled", {}),
                                 ("command_level_additive_shuffled",
                                  {"fixed_base": True})):
                    sfit = fit_stuff(sg[sg["season"] < season], component,
                                     features=both, **kw)
                    preds[name] = np.clip(sfit.predict(sbase, sh), 1e-4, 0.999)
                    rows[name] = sh
                    fits[name] = sfit

            for name, p in preds.items():
                f = rows.get(name, here).copy()
                f["model"] = name
                f["predicted"] = p
                f["coef"] = json.dumps(fits[name].coef if name in fits else {})
                frames.append(f)
    return pd.concat(frames, ignore_index=True)


# --- scoring -----------------------------------------------------------------

def paired(results: pd.DataFrame, arm: str, base: str, mask=None) -> dict:
    """Paired per-pitcher absolute-error difference, arm minus base.

    Pairing is on (season, cutoff, pitcher). The standard error is reported
    twice: clustered on the pitcher (one pitcher appears at three cutoffs of
    five seasons and those rows are not fifteen independent observations) and
    clustered on the cell (fifteen season-cutoff panels, which is the unit a
    league-wide shock moves). `pct` is the difference as a percent of the base
    arm's own MAE *on the same slice*.
    """
    empty = {"n": 0, "n_clusters": 0, "diff": float("nan"), "se": float("nan"),
             "t": float("nan"), "win_rate": float("nan"),
             "base_mae": float("nan"), "pct": float("nan"),
             "t_cell": float("nan"), "n_cells": 0}
    r = results if mask is None else results[mask]
    if r.empty:
        return empty
    r = r.assign(
        _key=r["season"].astype(str) + "|" + r["cutoff"] + "|"
        + r["player"].astype(str),
        _cell=r["season"].astype(str) + "|" + r["cutoff"])
    cols = ["_key", "player", "_cell", "predicted", "realized_rate", "trials"]
    a = r[r["model"] == arm][cols]
    b = r[r["model"] == base][cols]
    if a.empty or b.empty:
        return empty
    out = paired_abs_error_diff(a, b, id_col="_key", cluster_col="player")
    cell = paired_abs_error_diff(a, b, id_col="_key", cluster_col="_cell")
    out["t_cell"] = cell["t"]
    out["n_cells"] = cell["n_clusters"]
    out["base_mae"] = float(np.average(
        np.abs(b["predicted"] - b["realized_rate"]), weights=b["trials"]))
    out["pct"] = 100.0 * out["diff"] / out["base_mae"]
    return out


def score_table(results: pd.DataFrame) -> pd.DataFrame:
    """Harness `score()` over the pooled holdout, per component and arm."""
    return score(results[["component", "model", "player", "predicted",
                          "realized_successes", "realized_rate", "trials"]])


def tune(cells: pd.DataFrame, command_monthly: pd.DataFrame,
         stuff_monthly: pd.DataFrame, tune_seasons, level_features
         ) -> tuple[tuple, float, pd.DataFrame]:
    """Choose the recency weights and the shrinkage ballast on the tuning
    window only, by pooled trials-weighted MAE of the `command_level` arm over
    the two walk rates — the components this pre-registration expects to move,
    by the rule `scripts/run_command_backtest.py` already carried. Every
    scored season is untouched by this."""
    targets = ["p_bb_rate", "p_bbhbp_rate"]
    sub = cells[cells["component"].isin(targets)]
    rows = []
    for weights in COMMAND_WEIGHT_GRID:
        for ballast in COMMAND_BALLAST_GRID:
            z = attach_z(sub, command_monthly, stuff_monthly, weights, ballast,
                         level_features)
            res = walk_forward(z, targets, tune_seasons, level_features)
            g = res[res["model"] == FREE_ARM]
            mae = float(np.average(np.abs(g["predicted"] - g["realized_rate"]),
                                   weights=g["trials"]))
            rows.append({"weights": weights, "ballast": ballast, "mae": mae,
                         "n": len(g)})
            logger.info("tune %s b=%.0f -> MAE %.6f", weights, ballast, mae)
    grid = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = grid.iloc[0]
    return tuple(best["weights"]), float(best["ballast"]), grid


def paired_rows(results: pd.DataFrame, components, arms) -> list[dict]:
    """One paired comparison per (component, arm, base)."""
    rows = []
    for component in components:
        m = results["component"] == component
        for arm, base in arms:
            rows.append({"component": component, "arm": arm, "base": base,
                         "scope": "all", **paired(results, arm, base, m)})
    return rows


def stage2_table(prows: list[dict], components) -> pd.DataFrame:
    """The deployable arm against the baseline, and the covariate's own share.

    With the baseline pinned at 1 the stuff-only control *is* the served
    engine, so one column answers both the serving rule's covariate-share
    question and the increment over what is already served.
    """
    rows = []
    for component in components:
        def get(arm, base):
            return next(p for p in prows if p["component"] == component
                        and p["arm"] == arm and p["base"] == base)
        r, c = get(ADDITIVE_ARM, BASE_ARM), get(ADDITIVE_ARM, SERVED_ARM)
        free = get(FREE_ARM, BASE_ARM)
        rows.append({
            "component": component, "diff": r["diff"], "pct": r["pct"],
            "t": r["t"], "t_cell": r["t_cell"], "win_rate": r["win_rate"],
            "covariate_pct": c["pct"], "covariate_t": c["t"],
            "covariate_t_cell": c["t_cell"],
            "free_fit_pct": free["pct"], "free_fit_t": free["t"],
            # architecture.md §3, BAS-82's floor.
            "serves": bool(r["diff"] < 0 and c["diff"] < 0
                           and -c["pct"] >= SERVE_MIN_PCT
                           and abs(c["t"]) > SERVE_MIN_T)})
    return pd.DataFrame(rows)


def incremental_table(prows: list[dict], components) -> pd.DataFrame:
    """What the command block is worth on top of the served stuff engine.

    `pct`/`t` are the pre-registered joint fit — both blocks in one weighted
    least squares with the baseline pinned — and `two_stage_*` is the literal
    reading of "the served engine's prediction as the base", with the stuff
    coefficients frozen at the stuff-only fit.
    """
    rows = []
    for c in components:
        joint = next(p for p in prows if p["component"] == c
                     and p["arm"] == ADDITIVE_ARM and p["base"] == SERVED_ARM)
        two = next(p for p in prows if p["component"] == c
                   and p["arm"] == "stuff_then_command_level"
                   and p["base"] == SERVED_ARM)
        rows.append({"component": c,
                     **{k: joint[k] for k in ("diff", "pct", "t", "t_cell",
                                              "win_rate", "n", "n_clusters",
                                              "base_mae")},
                     "two_stage_pct": two["pct"], "two_stage_t": two["t"]})
    return pd.DataFrame(rows)


def split_table(results: pd.DataFrame, components) -> pd.DataFrame:
    """contact-quality.md §6: the paired gain by exposure tercile and by
    cutoff, against the baseline and against the served engine, as a percent
    of that slice's own base MAE — an error scale that shrinks with the season
    must not be readable as a fading effect."""
    rows = []
    for component in components:
        m0 = results["component"] == component
        slices = {}
        for axis in ("pre_trials", "cmd_pitches_raw"):
            slices.update(tercile_masks(results, m0, axis))
        for md in CUTOFF_MONTHS:
            slices[f"cutoff {md}"] = m0 & results["cutoff"].str.endswith(md)
        for label, m in slices.items():
            for base in (BASE_ARM, SERVED_ARM):
                rows.append({"component": component, "scope": label,
                             "vs": base,
                             **paired(results, ADDITIVE_ARM, base, m)})
    return pd.DataFrame(rows)


def tercile_masks(results: pd.DataFrame, mask, axis: str) -> dict:
    """Low/mid/high terciles of one exposure axis, within the given slice."""
    x = results.loc[mask, axis]
    lo, hi = float(x.quantile(1 / 3)), float(x.quantile(2 / 3))
    return {f"{axis} T1 (low)": mask & (results[axis] <= lo),
            f"{axis} T2": mask & (results[axis] > lo) & (results[axis] <= hi),
            f"{axis} T3 (high)": mask & (results[axis] > hi)}


# --- the pre-registered scoring ----------------------------------------------

def score_predictions(stage2: pd.DataFrame, inc: pd.DataFrame,
                      vacuity: dict) -> list[dict]:
    """Predictions 1-5 of `docs/pitching-command-level.md`, mechanically."""
    s2 = stage2.set_index("component")
    ic = inc.set_index("component")
    out = []

    def band(component):
        if component not in s2.index:
            return {"component": component, "holds": False,
                    "detail": "not scored in this run"}
        r = s2.loc[component]
        lo, hi = PREREG["bb_total_pct"]
        in_band = hi <= r["pct"] <= lo
        sig = abs(r["t"]) > PREREG["bb_total_min_t"]
        cov = (-r["covariate_pct"] >= PREREG["bb_cov_min_pct"]
               and r["covariate_pct"] < 0
               and abs(r["covariate_t"]) > PREREG["bb_cov_min_t"])
        return {"component": component, "pct": r["pct"], "t": r["t"],
                "covariate_pct": r["covariate_pct"],
                "covariate_t": r["covariate_t"],
                "in_band": bool(in_band), "significant": bool(sig),
                "covariate_share_clears": bool(cov),
                "holds": bool(in_band and sig and cov)}

    p1 = [band("p_bb_rate"), band("p_bbhbp_rate")]
    out.append({"prediction": 1, "holds": all(r["holds"] for r in p1),
                "detail": p1})
    if "p_bb_rate" in ic.index:
        out.append({"prediction": 3,
                    "holds": bool(-ic.loc["p_bb_rate", "pct"]
                                  >= PREREG["incremental_min_pct"]
                                  and ic.loc["p_bb_rate", "pct"] < 0
                                  and abs(ic.loc["p_bb_rate", "t"])
                                  > PREREG["incremental_min_t"]),
                    "holds_on_two_stage_reading": bool(
                        -ic.loc["p_bb_rate", "two_stage_pct"]
                        >= PREREG["incremental_min_pct"]
                        and abs(ic.loc["p_bb_rate", "two_stage_t"])
                        > PREREG["incremental_min_t"]),
                    "detail": {"pct": float(ic.loc["p_bb_rate", "pct"]),
                               "t": float(ic.loc["p_bb_rate", "t"]),
                               "two_stage_pct": float(
                                   ic.loc["p_bb_rate", "two_stage_pct"]),
                               "two_stage_t": float(
                                   ic.loc["p_bb_rate", "two_stage_t"])}})
    if "p_k_rate" in ic.index and "p_hr_rate" in s2.index:
        k = ic.loc["p_k_rate"]
        hr_inc, hr_tot = ic.loc["p_hr_rate"], s2.loc["p_hr_rate"]
        k_holds = (abs(k["pct"]) < PREREG["k_incremental_max_pct"]
                   and abs(k["t"]) < PREREG["k_incremental_max_t"])
        # "HR/BF moves < 1%" does not say which comparison, and the two
        # readings disagree, so both are recorded rather than one being
        # quietly chosen. The sentence sits next to "incremental over
        # `stuff_additive`", so the incremental reading is the primary one and
        # the total against the raw Marcel is reported beside it.
        out.append({"prediction": 4,
                    "holds": bool(k_holds
                                  and abs(hr_inc["pct"])
                                  < PREREG["hr_max_pct"]),
                    "holds_on_total_hr_reading": bool(
                        k_holds and abs(hr_tot["pct"])
                        < PREREG["hr_max_pct"]),
                    "detail": {"k_incremental_pct": float(k["pct"]),
                               "k_incremental_t": float(k["t"]),
                               "hr_incremental_pct": float(hr_inc["pct"]),
                               "hr_incremental_t": float(hr_inc["t"]),
                               "hr_total_pct": float(hr_tot["pct"]),
                               "hr_total_t": float(hr_tot["t"])}})
    served = sorted(stage2[stage2["serves"]]["component"].tolist())
    # `p_bbhbp_rate` is station E's rate and not a served pitcher column
    # (docs/pitching-stuff.md), so prediction 5 is scored on the three that
    # are, and the (BB+HBP)/BF verdict is reported beside it.
    out.append({"prediction": 5,
                "holds": bool("p_bb_rate" in served
                              and "p_k_rate" not in served
                              and "p_hr_rate" not in served),
                "detail": {"would_serve": served,
                           "served_columns_only": [c for c in served
                                                   if c != "p_bbhbp_rate"]}})
    out.append({"prediction": "vacuity",
                "holds": bool(vacuity["r"] >= PREREG["vacuity_min_r"]),
                "detail": vacuity})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    ap.add_argument("--pa-dir", type=Path,
                    default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--command-monthly", type=Path,
                    default=ROOT / "data/features/pitching_command_monthly.parquet")
    ap.add_argument("--stuff-monthly", type=Path,
                    default=ROOT / "data/features/pitching_stuff_monthly.parquet")
    ap.add_argument("--min-trials", type=int, default=100)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--weights", nargs=3, type=float, default=None)
    ap.add_argument("--ballast", type=float, default=None)
    ap.add_argument("--with-edge-share", action="store_true",
                    help="add `shadow_share` (the artifact's edge region) to "
                         "the covariate block — a labelled sensitivity, not "
                         "the pre-registered block")
    ap.add_argument("--cells-out", type=Path, default=None)
    ap.add_argument("--cells-in", type=Path, default=None)
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the labelled robustness runs in SENSITIVITIES")
    ap.add_argument("--shuffle-control", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path,
                    default=ROOT / "data/eval/pitching_command_stage2.json")
    args = ap.parse_args()

    components = tuple(args.components)
    level_features = (LEVEL_FEATURES_WITH_EDGE if args.with_edge_share
                      else LEVEL_FEATURES)
    command_monthly = load_command_monthly(args.command_monthly)
    stuff_monthly = load_stuff_monthly(args.stuff_monthly)

    # The vacuity gate, on the level this ticket registers rather than on
    # BAS-76's residual. It runs first and its verdict is printed whatever
    # stage 2 says.
    vac = {c: year_over_year(command_monthly, c)
           for c in ("cmd_csw", "zone_share", "waste_share", "shadow_share",
                     "cmd_resid", "cs_resid")}
    print("\n=== vacuity: year-over-year r of the pitcher-season aggregate "
          "(>= 1,000 pitches both years) ===")
    print(pd.DataFrame([
        {"aggregate": c, "n_pairs": v["n_pairs"], "r": v["r"],
         "r_season_demeaned": v["r_season_demeaned_diagnostic"],
         "floor": PREREG["vacuity_min_r"],
         "verdict": "PASS" if v["r"] >= PREREG["vacuity_min_r"] else "FAIL"}
        for c, v in vac.items()]).round(4).to_string(index=False))

    if args.cells_in and args.cells_in.exists():
        cells = pd.read_parquet(args.cells_in)
        cells = cells[cells["component"].isin(components)]
        logger.info("loaded %d cell rows from %s", len(cells), args.cells_in)
    else:
        seasons_table = pitcher_eval.normalize_pitcher_seasons(
            pd.read_parquet(ROOT / "data/parquet/pitcher_seasons_api.parquet"))
        cells = build_cells(components, seasons_table, args.pa_dir,
                            args.min_trials)
        if args.cells_out:
            args.cells_out.parent.mkdir(parents=True, exist_ok=True)
            cells.to_parquet(args.cells_out, index=False)

    tune_seasons = [s for s in CELL_SEASONS if s <= TUNE_THROUGH][2:]
    score_seasons = [s for s in CELL_SEASONS if s > TUNE_THROUGH]

    weights, ballast, grid = (DEFAULT_COMMAND_WEIGHTS,
                              DEFAULT_COMMAND_BALLAST, None)
    if args.tune:
        weights, ballast, grid = tune(cells, command_monthly, stuff_monthly,
                                      tune_seasons, level_features)
        logger.info("tuned: weights=%s ballast=%.0f", weights, ballast)
    if args.weights:
        weights = tuple(args.weights)
    if args.ballast:
        ballast = args.ballast

    z = attach_z(cells, command_monthly, stuff_monthly, weights, ballast,
                 level_features)
    shuffled = (shuffle_z(z, level_features, args.shuffle_seed)
                if args.shuffle_control else None)
    results = walk_forward(z, components, score_seasons, level_features,
                           shuffled=shuffled)

    print(f"\n=== command level: holdout seasons {score_seasons}, "
          f"cutoffs {CUTOFF_MONTHS} ===")
    print(f"command block {level_features}, stuff controls {STUFF_FEATURES}")
    print(f"weights={weights} ballast={ballast:.0f} "
          f"(chosen on {tune_seasons}, holdout untouched)")
    print(f"{results['player'].nunique()} pitchers, "
          f"{results[results['model'] == BASE_ARM].shape[0]} cells")
    # How much of the command block the stuff controls already contain. The
    # `pitching` model that produced `cmd_csw` reads the stuff features too,
    # so the two blocks are not independent measurements and the fitted
    # coefficients are not separately interpretable; the R^2 of each command
    # covariate on the six stuff controls says how much is left over.
    zh = z[z["season"].isin(score_seasons)
           & (z["component"] == components[0])]
    S = np.column_stack([np.ones(len(zh))]
                        + [zh[f].to_numpy(dtype="float64")
                           for f in STUFF_FEATURES])
    overlap = []
    for f in level_features:
        y = zh[f].to_numpy(dtype="float64")
        resid = y - S @ np.linalg.lstsq(S, y, rcond=None)[0]
        r2 = 1.0 - float(np.var(resid) / np.var(y)) if np.var(y) > 0 else 0.0
        overlap.append({"command covariate": f, "R2 on stuff controls": r2,
                        "max |r| with a stuff control": max(
                            abs(float(np.corrcoef(y, zh[s].to_numpy(
                                dtype="float64"))[0, 1]))
                            for s in STUFF_FEATURES)})
    print("\n--- how much of each command covariate the stuff controls "
          "already carry (holdout cells) ---")
    print(pd.DataFrame(overlap).round(4).to_string(index=False))

    print("\n--- pooled scores (harness score(), trials-weighted) ---")
    print(score_table(results).round(6).to_string(index=False))

    print("\n--- every arm, paired per-pitcher absolute error, negative = the "
          "arm is better ---")
    arms = list(ARMS)
    if args.shuffle_control:
        arms += list(SHUFFLE_ARMS)
    prows = paired_rows(results, components, arms)
    print(pd.DataFrame(prows).round(6).to_string(index=False))

    print(f"\n=== stage 2: {ADDITIVE_ARM} vs the baseline, and the "
          f"covariate-only share vs {SERVED_ARM} (the stuff-only control) ===")
    s2 = stage2_table(prows, components)
    print(s2.round(6).to_string(index=False))
    served = s2[s2["serves"]]["component"].tolist()
    print(f"\nUnder architecture.md §3 (covariate-only share >= "
          f"{SERVE_MIN_PCT}% of the served baseline's MAE at |t| > "
          f"{SERVE_MIN_T}), the components that would be served on "
          f"{SERVED_ARM} + {ADDITIVE_ARM} are: "
          f"{', '.join(served) if served else '(none)'}")
    print("Nothing is wired in this pass either way — this is a measurement "
          "ticket.")

    print(f"\n=== incremental over the served stuff engine "
          f"(vs {SERVED_ARM}) ===")
    inc = incremental_table(prows, components)
    print(inc.round(6).to_string(index=False))

    print("\n--- information or denoising: by exposure tercile and by cutoff "
          f"({ADDITIVE_ARM} vs baseline, and vs {SERVED_ARM}) ---")
    split = split_table(results, components)
    print(split.round(6).to_string(index=False))

    print(f"\n--- per-season paired ({ADDITIVE_ARM}) ---")
    yrows = []
    for component in components:
        for season in score_seasons:
            m = ((results["component"] == component)
                 & (results["season"] == season))
            for base in (BASE_ARM, SERVED_ARM):
                yrows.append({"component": component, "season": season,
                              "vs": base,
                              **paired(results, ADDITIVE_ARM, base, m)})
    print(pd.DataFrame(yrows).round(6).to_string(index=False))

    print("\n--- fitted coefficients on the last holdout season ---")
    for arm in (ADDITIVE_ARM, SERVED_ARM, "stuff_then_command_level"):
        last = results[(results["model"] == arm)
                       & (results["season"] == score_seasons[-1])]
        for component in components:
            c = last[last["component"] == component]
            if not c.empty:
                print(f"  {arm} / {component}: {c['coef'].iloc[0]}")

    sens = []
    if not args.no_sensitivity:
        for label, sw, sb, sf in SENSITIVITIES:
            w2 = weights if sw is None else sw
            b2 = ballast if sb is None else sb
            f2 = level_features if sf is None else sf
            z2 = attach_z(cells, command_monthly, stuff_monthly, w2, b2, f2)
            r2 = walk_forward(z2, components, score_seasons, f2)
            p2 = paired_rows(r2, components, ARMS)
            t2 = stage2_table(p2, components)
            i2 = incremental_table(p2, components)
            s2b = split_table(r2, components)
            print(f"\n--- sensitivity: {label} "
                  f"(weights={w2} ballast={b2:.0f} block={f2}) ---")
            print(t2.round(6).to_string(index=False))
            print(i2.round(6).to_string(index=False))
            bb = s2b[(s2b["component"] == "p_bb_rate")
                     & (s2b["vs"] == SERVED_ARM)]
            print(bb.round(6).to_string(index=False))
            sens.append({
                "label": label, "weights": list(w2), "ballast": b2,
                "level_features": list(f2),
                "stage2": json.loads(t2.to_json(orient="records")),
                "incremental": json.loads(i2.to_json(orient="records")),
                "split": json.loads(s2b.to_json(orient="records")),
            })

    print("\n=== scoring the pre-registered predictions ===")
    scored = score_predictions(s2, inc, vac["cmd_csw"])
    for row in scored:
        print(f"  prediction {row['prediction']}: "
              f"{'HOLDS' if row['holds'] else 'FAILS'}  {row['detail']}")
    print("  prediction 2 is read off the split table above (August cutoff "
          "and the high-exposure tercile on BB/BF).")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticket": "BAS-87",
            "components": list(components),
            "level_features": list(level_features),
            "stuff_controls": list(STUFF_FEATURES),
            "weights": list(weights), "ballast": ballast,
            "tune_seasons": tune_seasons, "score_seasons": score_seasons,
            "cutoffs": list(CUTOFF_MONTHS), "min_trials": args.min_trials,
            "n_pitchers": int(results["player"].nunique()),
            "n_cells": int(results[results["model"] == BASE_ARM].shape[0]),
            "serve_min_pct": SERVE_MIN_PCT, "serve_min_t": SERVE_MIN_T,
            "prereg": PREREG,
            "vacuity": vac,
            "covariate_overlap": overlap,
            "sensitivities": sens,
            "scores": json.loads(score_table(results).to_json(orient="records")),
            "paired": json.loads(pd.DataFrame(prows).to_json(orient="records")),
            "stage2": json.loads(s2.to_json(orient="records")),
            "incremental": json.loads(inc.to_json(orient="records")),
            "split": json.loads(split.to_json(orient="records")),
            "per_season": json.loads(
                pd.DataFrame(yrows).to_json(orient="records")),
            "predictions": scored,
            "grid": (json.loads(grid.assign(weights=grid["weights"].astype(str))
                                .to_json(orient="records"))
                     if grid is not None else []),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
