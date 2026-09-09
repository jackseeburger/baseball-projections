"""Walk-forward test of pitching command as covariates on the pitcher rates (BAS-76, stage 2).

The stuff backtest's shape, with one question added. Arms at every
(component, season, cutoff) cell, scored with the harness's own metrics on the
harness's own common pitcher set:

    marcel_pitcher_tuned    the live baseline, untouched
    command_recal           a + b * baseline, coefficients fitted on earlier
                            seasons only — the control that absorbs a pure
                            recalibration gain
    command                 the same free fit plus standardized command
                            covariates
    command_additive        the baseline's coefficient pinned at 1 and the
                            command covariates added as a correction — the
                            deployable shape
    command_additive_recal  the same shape with no covariate at all: baseline
                            plus a fitted intercept. What is left of a
                            `command_additive` gain after this control is the
                            covariate's own, and since 2026-09-09
                            (architecture.md §3, BAS-80) that share is what
                            decides whether anything is served
    command_shuffled        the covariates permuted across pitchers within a
                            cell (methods.md §5.4)

and the incremental pair, which is the question this ticket exists to ask:

    stuff_additive          the served engine, exactly as BAS-79 fits it
    stuff_command_additive  the same shape with the command covariates added
                            alongside the stuff ones

`stuff_command_additive` minus `stuff_additive` is what command is worth *on
top of what is already served*, and it is the number that decides whether
BAS-76 is a new layer or a second view of the old one.

    python scripts/build_pitching_command.py --download    # the artifact
    python scripts/run_command_backtest.py --tune

The §6 split of contact-quality.md is printed unconditionally: the paired gain
by pitcher-exposure tercile and by cutoff, as a percent of the baseline's own
MAE on that slice, so an error scale that shrinks with the season cannot
masquerade as a fading effect. The pre-registration asks for it on BB/BF; it
is printed for every component because withholding the other three would be
choosing which slice to look at after seeing one.
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
)
from src.eval.command import FEATURES as COMMAND_FEATURES
from src.eval.stuff import FEATURES as STUFF_FEATURES
from src.eval.stuff import fit_stuff
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("command-backtest")
ROOT = Path(__file__).resolve().parent.parent

CUTOFF_MONTHS = ("05-01", "07-01", "08-01")
# 2017 is the first season with two prior Statcast seasons and two prior
# seasons in the API season table. 2020 is excluded everywhere in this repo:
# a 60-game season that started July 23 has no May 1 cutoff.
CELL_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
TUNE_THROUGH = 2021

# The four the pre-registration names.
DEFAULT_COMPONENTS = ("p_k_rate", "p_bb_rate", "p_bbhbp_rate", "p_hr_rate")
BASE_ARM = "marcel_pitcher_tuned"
SERVED_ARM = "stuff_additive"
# architecture.md §3's covariate-share rule, in force from BAS-80: a covariate
# is served only if the arm clears the gate *and* its covariate-only share
# clears this on the same cells.
SERVE_MIN_T = 2.5
# The pre-registration's own covariate-only floor for BB/BF, stated before any
# number was read.
PREREG_COVARIATE_MIN_PCT = 1.5
PREREG_COVARIATE_MIN_T = 2.5


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
             stuff_monthly: pd.DataFrame, weights, ballast) -> pd.DataFrame:
    """Merge both covariate blocks onto the cells.

    The stuff block is attached at *its own* pinned hyperparameters
    (`src.eval.stuff.DEFAULT_WINDOW_WEIGHTS` / `DEFAULT_BALLAST`), not at the
    command sweep's, because the incremental comparison has to be against the
    engine that is actually served rather than against a re-tuned version of
    it. The command block moves with the sweep.
    """
    out = command_eval.attach_command_features(cells, command_monthly, weights,
                                               ballast)
    out = stuff_eval.attach_live_features(out, stuff_monthly)
    return out


def shuffle_z(cells: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute the command covariates across pitchers within each cell.

    Every pitcher keeps a real covariate vector — same marginal, same
    shrinkage, same standardization — attached to the wrong pitcher. An arm
    fitted and scored on this must land on the recalibration control; if it
    beats it, the pipeline is fitting the split rather than the covariate.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _, g in cells.groupby(["season", "cutoff"], sort=False):
        g = g.copy()
        perm = rng.permutation(len(g))
        for f in COMMAND_FEATURES:
            g[f] = g[f].to_numpy()[perm]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --- scoring -----------------------------------------------------------------

BOTH_FEATURES = tuple(STUFF_FEATURES) + tuple(COMMAND_FEATURES)


def walk_forward(cells: pd.DataFrame, components, score_seasons,
                 shuffled: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predictions for every arm, refitting coefficients on prior seasons only."""
    frames = []
    for component in components:
        g = cells[cells["component"] == component]
        for season in score_seasons:
            past = g[g["season"] < season]
            here = g[g["season"] == season]
            if past.empty or here.empty:
                continue
            fits = {
                "command_recal": fit_stuff(past, component, features=()),
                "command": fit_stuff(past, component, features=COMMAND_FEATURES),
                "command_additive": fit_stuff(past, component,
                                              features=COMMAND_FEATURES,
                                              fixed_base=True),
                "command_additive_recal": fit_stuff(past, component,
                                                    features=(),
                                                    fixed_base=True),
                SERVED_ARM: fit_stuff(past, component, features=STUFF_FEATURES,
                                      fixed_base=True),
                "stuff_command_additive": fit_stuff(past, component,
                                                    features=BOTH_FEATURES,
                                                    fixed_base=True),
            }
            base = here["base"].to_numpy(dtype="float64")
            preds = {BASE_ARM: base}
            for name, f in fits.items():
                preds[name] = np.clip(f.predict(base, here), 1e-4, 0.999)
            rows = {}
            if shuffled is not None:
                sg = shuffled[shuffled["component"] == component]
                sh = sg[sg["season"] == season]
                sfit = fit_stuff(sg[sg["season"] < season], component,
                                 features=COMMAND_FEATURES)
                preds["command_shuffled"] = np.clip(
                    sfit.predict(sh["base"].to_numpy(dtype="float64"), sh),
                    1e-4, 0.999)
                rows["command_shuffled"] = sh
            for name, p in preds.items():
                f = rows.get(name, here).copy()
                f["model"] = name
                f["predicted"] = p
                f["coef"] = json.dumps(
                    fits[name].coef if name in fits else {})
                frames.append(f)
    return pd.concat(frames, ignore_index=True)


def paired(results: pd.DataFrame, arm: str, base: str, mask=None) -> dict:
    """Paired per-pitcher absolute-error difference, arm minus base.

    Pairing is on (season, cutoff, pitcher); the standard error is clustered
    back on the pitcher, because one pitcher appears at three cutoffs of five
    seasons and those rows are not fifteen independent observations. `pct` is
    the difference as a percent of the base arm's own MAE *on the same slice*.
    """
    cols = ["_key", "player", "predicted", "realized_rate", "trials"]
    r = results if mask is None else results[mask]
    r = r.assign(_key=r["season"].astype(str) + "|" + r["cutoff"] + "|"
                 + r["player"].astype(str))
    a = r[r["model"] == arm][cols]
    b = r[r["model"] == base][cols]
    if a.empty or b.empty:
        return {"n": 0, "n_clusters": 0, "diff": float("nan"),
                "se": float("nan"), "t": float("nan"),
                "win_rate": float("nan"), "base_mae": float("nan"),
                "pct": float("nan")}
    out = paired_abs_error_diff(a, b, id_col="_key", cluster_col="player")
    out["base_mae"] = float(np.average(
        np.abs(b["predicted"] - b["realized_rate"]), weights=b["trials"]))
    out["pct"] = 100.0 * out["diff"] / out["base_mae"]
    return out


def score_table(results: pd.DataFrame) -> pd.DataFrame:
    """Harness `score()` over the pooled holdout, per component and arm."""
    return score(results[["component", "model", "player", "predicted",
                          "realized_successes", "realized_rate", "trials"]])


def tune(cells: pd.DataFrame, command_monthly: pd.DataFrame,
         stuff_monthly: pd.DataFrame, tune_seasons
         ) -> tuple[tuple, float, pd.DataFrame]:
    """Choose the recency weights and the shrinkage ballast on the tuning
    window only, by pooled trials-weighted MAE of the `command` arm over the
    two walk rates — the components this pre-registration expects to move, the
    way stuff tuned on the two it expected to move. Every scored season is
    untouched by this."""
    targets = ["p_bb_rate", "p_bbhbp_rate"]
    rows = []
    for weights in COMMAND_WEIGHT_GRID:
        for ballast in COMMAND_BALLAST_GRID:
            z = attach_z(cells, command_monthly, stuff_monthly, weights, ballast)
            res = walk_forward(z, targets, tune_seasons)
            g = res[res["model"] == "command"]
            mae = float(np.average(np.abs(g["predicted"] - g["realized_rate"]),
                                   weights=g["trials"]))
            rows.append({"weights": weights, "ballast": ballast, "mae": mae,
                         "n": len(g)})
            logger.info("tune %s b=%.0f -> MAE %.6f", weights, ballast, mae)
    grid = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = grid.iloc[0]
    return tuple(best["weights"]), float(best["ballast"]), grid


def tercile_masks(results: pd.DataFrame, mask, axis: str) -> dict:
    """Low/mid/high terciles of one exposure axis, within the given slice."""
    x = results.loc[mask, axis]
    lo, hi = float(x.quantile(1 / 3)), float(x.quantile(2 / 3))
    return {f"{axis} T1 (low)": mask & (results[axis] <= lo),
            f"{axis} T2": mask & (results[axis] > lo) & (results[axis] <= hi),
            f"{axis} T3 (high)": mask & (results[axis] > hi)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    ap.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--command-monthly", type=Path,
                    default=ROOT / "data/features/pitching_command_monthly.parquet")
    ap.add_argument("--stuff-monthly", type=Path,
                    default=ROOT / "data/features/pitching_stuff_monthly.parquet")
    ap.add_argument("--min-trials", type=int, default=100)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--weights", nargs=3, type=float, default=None)
    ap.add_argument("--ballast", type=float, default=None)
    ap.add_argument("--cells-out", type=Path, default=None)
    ap.add_argument("--cells-in", type=Path, default=None)
    ap.add_argument("--shuffle-control", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path,
                    default=ROOT / "data/eval/pitching_command_stage2.json")
    args = ap.parse_args()

    components = tuple(args.components)
    command_monthly = load_command_monthly(args.command_monthly)
    stuff_monthly = load_stuff_monthly(args.stuff_monthly)

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

    weights, ballast, grid = (DEFAULT_COMMAND_WEIGHTS, DEFAULT_COMMAND_BALLAST,
                              None)
    if args.tune:
        weights, ballast, grid = tune(cells, command_monthly, stuff_monthly,
                                      tune_seasons)
        logger.info("tuned: weights=%s ballast=%.0f", weights, ballast)
    if args.weights:
        weights = tuple(args.weights)
    if args.ballast:
        ballast = args.ballast

    z = attach_z(cells, command_monthly, stuff_monthly, weights, ballast)
    shuffled = shuffle_z(z, args.shuffle_seed) if args.shuffle_control else None
    results = walk_forward(z, components, score_seasons, shuffled=shuffled)

    print(f"\n=== pitching command: holdout seasons {score_seasons}, "
          f"cutoffs {CUTOFF_MONTHS} ===")
    print(f"weights={weights} ballast={ballast:.0f} "
          f"(chosen on {tune_seasons}, holdout untouched)")
    print(f"{results['player'].nunique()} pitchers, "
          f"{results[results['model'] == BASE_ARM].shape[0]} cells")
    print("\n--- pooled scores (harness score(), trials-weighted) ---")
    print(score_table(results).round(6).to_string(index=False))

    print("\n--- the gate: paired per-pitcher absolute error, negative = the "
          "arm is better ---")
    prows = []
    arms = [("command", BASE_ARM), ("command_recal", BASE_ARM),
            ("command", "command_recal"),
            ("command_additive", BASE_ARM),
            ("command_additive_recal", BASE_ARM),
            ("command_additive", "command_additive_recal"),
            (SERVED_ARM, BASE_ARM),
            ("stuff_command_additive", BASE_ARM),
            ("stuff_command_additive", SERVED_ARM)]
    if args.shuffle_control:
        arms += [("command_shuffled", BASE_ARM),
                 ("command_shuffled", "command_recal")]
    for component in components:
        m = results["component"] == component
        for arm, base in arms:
            prows.append({"component": component, "arm": arm, "base": base,
                          "scope": "all", **paired(results, arm, base, m)})
    print(pd.DataFrame(prows).round(6).to_string(index=False))

    # The four-component stage-2 table the pre-registration asks for, in one
    # place: the deployable arm against the baseline, and the covariate's own
    # share of it against the same shape with the covariate removed.
    print("\n=== stage 2: command_additive vs the baseline, and the "
          "covariate-only share ===")
    stage2 = []
    for component in components:
        r = next(p for p in prows if p["component"] == component
                 and p["arm"] == "command_additive" and p["base"] == BASE_ARM)
        c = next(p for p in prows if p["component"] == component
                 and p["arm"] == "command_additive"
                 and p["base"] == "command_additive_recal")
        free = next(p for p in prows if p["component"] == component
                    and p["arm"] == "command" and p["base"] == BASE_ARM)
        stage2.append({
            "component": component, "diff": r["diff"], "pct": r["pct"],
            "t": r["t"], "covariate_pct": c["pct"], "covariate_t": c["t"],
            "free_fit_pct": free["pct"], "free_fit_t": free["t"],
            "serves": bool(r["diff"] < 0 and abs(r["t"]) > SERVE_MIN_T
                           and c["diff"] < 0 and abs(c["t"]) > SERVE_MIN_T)})
    s2 = pd.DataFrame(stage2)
    print(s2.round(6).to_string(index=False))
    served = s2[s2["serves"]]["component"].tolist()
    print(f"\nUnder architecture.md §3's covariate-share rule "
          f"(gate cleared AND covariate-only |t| > {SERVE_MIN_T}), the "
          f"components that would be served on command are: "
          f"{', '.join(served) if served else '(none)'}")
    print("Nothing is wired in this pass either way — BAS-76 is a measurement "
          "ticket.")

    print(f"\n=== incremental over the served stuff engine "
          f"({SERVED_ARM} + command vs {SERVED_ARM}) ===")
    inc = pd.DataFrame([
        {"component": c,
         **{k: v for k, v in next(
             p for p in prows if p["component"] == c
             and p["arm"] == "stuff_command_additive"
             and p["base"] == SERVED_ARM).items()
            if k in ("diff", "pct", "t", "n", "n_clusters", "base_mae")}}
        for c in components])
    print(inc.round(6).to_string(index=False))

    # Prediction 3, the §6 split. If the covariate only denoises a small
    # sample, the gain lives in the low-exposure tercile and at the May cutoff
    # and is gone by August; if it carries information the realized rate does
    # not have, it survives everywhere. Read the `pct` column, not `diff`:
    # the error scale itself shrinks as the season goes on.
    print("\n--- information or denoising: by exposure tercile and by cutoff "
          "(command_additive vs baseline) ---")
    srows = []
    for component in components:
        m0 = results["component"] == component
        for axis in ("pre_trials", "cmd_pitches_raw"):
            for label, m in tercile_masks(results, m0, axis).items():
                srows.append({"component": component, "scope": label,
                              **paired(results, "command_additive", BASE_ARM, m)})
        for md in CUTOFF_MONTHS:
            m = m0 & results["cutoff"].str.endswith(md)
            srows.append({"component": component, "scope": f"cutoff {md}",
                          **paired(results, "command_additive", BASE_ARM, m)})
    split = pd.DataFrame(srows)
    print(split.round(6).to_string(index=False))

    print("\n--- per-season paired (command_additive vs baseline) ---")
    yrows = []
    for component in components:
        for season in score_seasons:
            m = ((results["component"] == component)
                 & (results["season"] == season))
            yrows.append({"component": component, "season": season,
                          **paired(results, "command_additive", BASE_ARM, m)})
    print(pd.DataFrame(yrows).round(6).to_string(index=False))

    print("\n--- fitted coefficients on the last holdout season ---")
    for arm in ("command_additive", "stuff_command_additive"):
        last = results[(results["model"] == arm)
                       & (results["season"] == score_seasons[-1])]
        for component in components:
            c = last[last["component"] == component]
            if not c.empty:
                print(f"  {arm} / {component}: {c['coef'].iloc[0]}")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "components": list(components), "weights": list(weights),
            "ballast": ballast, "tune_seasons": tune_seasons,
            "score_seasons": score_seasons, "cutoffs": list(CUTOFF_MONTHS),
            "min_trials": args.min_trials,
            "n_pitchers": int(results["player"].nunique()),
            "n_cells": int(results[results["model"] == BASE_ARM].shape[0]),
            "serve_min_t": SERVE_MIN_T,
            "prereg_covariate_min_pct": PREREG_COVARIATE_MIN_PCT,
            "prereg_covariate_min_t": PREREG_COVARIATE_MIN_T,
            "scores": json.loads(score_table(results).to_json(orient="records")),
            "paired": json.loads(pd.DataFrame(prows).to_json(orient="records")),
            "stage2": json.loads(s2.to_json(orient="records")),
            "incremental": json.loads(inc.to_json(orient="records")),
            "split": json.loads(split.to_json(orient="records")),
            "per_season": json.loads(pd.DataFrame(yrows).to_json(orient="records")),
            "grid": (json.loads(grid.assign(weights=grid["weights"].astype(str))
                                .to_json(orient="records"))
                     if grid is not None else []),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
