"""Walk-forward test of pitching stuff as covariates on the pitcher rates (BAS-71, stage 2).

Five arms at every (component, season, cutoff) cell, scored with the harness's
own metrics on the harness's own common pitcher set:

    marcel_pitcher_tuned   the live baseline, untouched
    stuff_recal            a + b * baseline, coefficients fitted on earlier
                           seasons only — the control that absorbs a pure
                           recalibration gain
    stuff                  the same fit plus standardized stuff covariates
    stuff_additive         the baseline's coefficient pinned at 1 and the
                           covariates added as a correction — the deployable
                           shape, and the one docs/pitching-stuff.md's
                           "Serving" section pre-registers the serving gate on
    stuff_additive_recal   the same shape with no covariate at all: baseline
                           plus a fitted intercept. What is left of a
                           `stuff_additive` gain after this control is the
                           covariate's own

`stuff` vs `marcel_pitcher_tuned` is the gate. `stuff` vs `stuff_recal` is what
the covariate itself is worth, and `stuff_additive` vs `stuff_additive_recal`
is the same question asked of the arm that actually ships. Coefficients for a scored season are fitted on
cells strictly before it; the two hyperparameters (the recency weights over
seasons and the shrinkage ballast) are chosen on a tuning window that ends
before the scored seasons begin.

    python scripts/build_pitching_stuff.py --download     # the artifact
    python -c "from src.data.pa_outcomes_pipeline import build_pa_dataset; \
               build_pa_dataset(data_dir='data/raw')"     # the PA outcomes

    python scripts/run_stuff_backtest.py
    python scripts/run_stuff_backtest.py --tune --json-out data/eval/stuff.json

The §6 split of contact-quality.md is printed unconditionally: the paired gain
by pitcher-exposure tercile and by cutoff, as a percent of the baseline's own
MAE on that slice, so an error scale that shrinks with the season cannot
masquerade as a fading effect. That is what scores the pre-registration's
prediction 3 — information, or denoising.
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

from src.data.pitching_stuff import load_monthly
from src.eval import pitchers as pitcher_eval
from src.eval import stuff as stuff_eval
from src.eval.backtest import score
from src.eval.stuff import (
    DEFAULT_BALLAST,
    DEFAULT_WINDOW_WEIGHTS,
    FEATURES,
    STUFF_BALLAST_GRID,
    STUFF_WEIGHT_GRID,
    features_at_cutoff,
    fit_stuff,
)
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stuff-backtest")
ROOT = Path(__file__).resolve().parent.parent

# The three cutoffs `scripts/run_pitcher_backtest.py` scores, which are also
# the three contact-quality reports on.
CUTOFF_MONTHS = ("05-01", "07-01", "08-01")
# 2017 is the first season with two prior Statcast seasons and two prior
# seasons in the API season table. 2020 is excluded everywhere in this repo:
# a 60-game season that started July 23 has no May 1 cutoff.
CELL_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
TUNE_THROUGH = 2021

# The three the pre-registration names. `p_bbhbp_rate` rides along because it
# is the walk rate station E actually consumes and because contact quality
# showed both walk rates to be pure recalibration — the control has to be able
# to say so again.
DEFAULT_COMPONENTS = ("p_k_rate", "p_bb_rate", "p_bbhbp_rate", "p_hr_rate")
BASE_ARM = "marcel_pitcher_tuned"
# The clustered |t| a component's `stuff_additive` arm has to clear to be
# served, pre-registered in docs/pitching-stuff.md's "Serving" section before
# any additive number was read.
SERVE_MIN_T = 2.5


# --- cells -------------------------------------------------------------------

def build_cells(components, seasons_table: pd.DataFrame, pa_dir: Path,
                min_trials: int) -> pd.DataFrame:
    """One row per (component, season, cutoff, pitcher) with everything the
    arms need: the baseline projection, the realized rest-of-season outcome
    and the pre-cutoff exposure.

    The split is the harness's own — `partial_and_realized` either side of the
    date, `assert_split_clean` on both, the same `min_trials` filter and the
    same intersection with the baseline's coverage that `_run_split` applies.
    Nothing here re-implements a metric, and since BAS-79 it does not even
    build the cells: `src.eval.stuff.build_pitcher_cells` does, so the arm the
    site serves is fitted on exactly the rows the gate below is scored on.
    """
    return stuff_eval.build_pitcher_cells(
        seasons_table, pa_dir, components, seasons=CELL_SEASONS,
        cutoff_months=CUTOFF_MONTHS, min_trials=min_trials)


def attach_z(cells: pd.DataFrame, monthly: pd.DataFrame, weights, ballast
             ) -> pd.DataFrame:
    """Merge the standardized stuff covariates onto the cells.

    Built once per (season, cutoff) — the features do not depend on the
    component. A pitcher with no tracked pitches before the cutoff gets z = 0,
    which makes the stuff arm identical to the recalibration arm for him.
    """
    out = []
    for (season, cutoff), g in cells.groupby(["season", "cutoff"]):
        z = features_at_cutoff(monthly, cutoff, season, weights, ballast)
        zi = z.set_index("player").reindex(g["player"].to_numpy())
        g = g.copy()
        for f in FEATURES:
            g[f] = zi[f].fillna(0.0).to_numpy()
        g["pitches_raw"] = zi["pitches_raw"].fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def shuffle_z(cells: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute the stuff covariates across pitchers within each cell.

    The permuted control (methods.md §5.4). Every pitcher keeps a real
    covariate vector — same marginal, same shrinkage, same standardization —
    attached to the wrong pitcher. An arm fitted and scored on this must land
    on the recalibration control; if it beats it, the pipeline is fitting the
    split rather than the covariate.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _, g in cells.groupby(["season", "cutoff"], sort=False):
        g = g.copy()
        perm = rng.permutation(len(g))
        for f in FEATURES:
            g[f] = g[f].to_numpy()[perm]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --- scoring -----------------------------------------------------------------

def walk_forward(cells: pd.DataFrame, components, score_seasons,
                 features=FEATURES, shuffled: pd.DataFrame | None = None
                 ) -> pd.DataFrame:
    """Predictions for every arm, refitting coefficients on prior seasons only."""
    frames = []
    for component in components:
        g = cells[cells["component"] == component]
        for season in score_seasons:
            past = g[g["season"] < season]
            here = g[g["season"] == season]
            if past.empty or here.empty:
                continue
            recal = fit_stuff(past, component, features=())
            full = fit_stuff(past, component, features=features)
            add = fit_stuff(past, component, features=features, fixed_base=True)
            add_recal = fit_stuff(past, component, features=(), fixed_base=True)
            base = here["base"].to_numpy(dtype="float64")
            preds = {
                BASE_ARM: base,
                "stuff_recal": np.clip(recal.predict(base, None), 1e-4, 0.999),
                "stuff": np.clip(full.predict(base, here), 1e-4, 0.999),
                "stuff_additive": np.clip(add.predict(base, here), 1e-4, 0.999),
                "stuff_additive_recal": np.clip(
                    add_recal.predict(base, None), 1e-4, 0.999),
            }
            rows = {}
            if shuffled is not None:
                sg = shuffled[shuffled["component"] == component]
                sh = sg[sg["season"] == season]
                sfit = fit_stuff(sg[sg["season"] < season], component,
                                 features=features)
                preds["stuff_shuffled"] = np.clip(
                    sfit.predict(sh["base"].to_numpy(dtype="float64"), sh),
                    1e-4, 0.999)
                rows["stuff_shuffled"] = sh
            for name, p in preds.items():
                f = rows.get(name, here).copy()
                f["model"] = name
                f["predicted"] = p
                f["coef"] = json.dumps(
                    full.coef if name == "stuff"
                    else add.coef if name == "stuff_additive"
                    else recal.coef if name == "stuff_recal"
                    else add_recal.coef if name == "stuff_additive_recal"
                    else {})
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


def tune(cells: pd.DataFrame, monthly: pd.DataFrame, components, tune_seasons
         ) -> tuple[tuple, float, pd.DataFrame]:
    """Choose the recency weights and the shrinkage ballast on the tuning
    window only, by pooled trials-weighted MAE of the stuff arm over K/BF and
    HR/BF — the two components the pre-registration expects to move."""
    targets = [c for c in components if c in ("p_k_rate", "p_hr_rate")]
    rows = []
    for weights in STUFF_WEIGHT_GRID:
        for ballast in STUFF_BALLAST_GRID:
            z = attach_z(cells, monthly, weights, ballast)
            res = walk_forward(z, targets, tune_seasons)
            g = res[res["model"] == "stuff"]
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
    ap.add_argument("--monthly", type=Path,
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
                    default=ROOT / "data/eval/pitching_stuff_stage2.json")
    args = ap.parse_args()

    components = tuple(args.components)
    monthly = load_monthly(args.monthly)

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

    weights, ballast, grid = DEFAULT_WINDOW_WEIGHTS, DEFAULT_BALLAST, None
    if args.tune:
        weights, ballast, grid = tune(cells, monthly, components, tune_seasons)
        logger.info("tuned: weights=%s ballast=%.0f", weights, ballast)
    if args.weights:
        weights = tuple(args.weights)
    if args.ballast:
        ballast = args.ballast

    z = attach_z(cells, monthly, weights, ballast)
    shuffled = shuffle_z(z, args.shuffle_seed) if args.shuffle_control else None
    results = walk_forward(z, components, score_seasons, shuffled=shuffled)

    print(f"\n=== pitching stuff: holdout seasons {score_seasons}, "
          f"cutoffs {CUTOFF_MONTHS} ===")
    print(f"weights={weights} ballast={ballast:.0f} "
          f"(chosen on {tune_seasons}, holdout untouched)")
    print("\n--- pooled scores (harness score(), trials-weighted) ---")
    print(score_table(results).round(6).to_string(index=False))

    print("\n--- the gate: paired per-pitcher absolute error, negative = the "
          "arm is better ---")
    prows = []
    arms = [("stuff", BASE_ARM), ("stuff_recal", BASE_ARM),
            ("stuff", "stuff_recal"), ("stuff_additive", BASE_ARM),
            ("stuff_additive_recal", BASE_ARM),
            ("stuff_additive", "stuff_additive_recal")]
    if args.shuffle_control:
        arms += [("stuff_shuffled", BASE_ARM), ("stuff_shuffled", "stuff_recal")]
    for component in components:
        m = results["component"] == component
        for arm, base in arms:
            prows.append({"component": component, "arm": arm, "base": base,
                          "scope": "all", **paired(results, arm, base, m)})
    print(pd.DataFrame(prows).round(6).to_string(index=False))

    # Prediction 3, the §6 split. If the covariate only denoises a small
    # sample, the gain lives in the low-exposure tercile and at the May cutoff
    # and is gone by August; if it carries information the realized rate does
    # not have, it survives everywhere. Read the `pct` column, not `diff`:
    # the error scale itself shrinks as the season goes on.
    print("\n--- information or denoising: by exposure tercile and by cutoff ---")
    srows = []
    for component in components:
        m0 = results["component"] == component
        for axis in ("pre_trials", "pitches_raw"):
            for label, m in tercile_masks(results, m0, axis).items():
                srows.append({"component": component, "scope": label,
                              **paired(results, "stuff", BASE_ARM, m)})
        for md in CUTOFF_MONTHS:
            m = m0 & results["cutoff"].str.endswith(md)
            srows.append({"component": component, "scope": f"cutoff {md}",
                          **paired(results, "stuff", BASE_ARM, m)})
    split = pd.DataFrame(srows)
    print(split.round(6).to_string(index=False))

    print("\n--- per-season paired (stuff vs baseline) ---")
    yrows = []
    for component in components:
        for season in score_seasons:
            m = ((results["component"] == component)
                 & (results["season"] == season))
            yrows.append({"component": component, "season": season,
                          **paired(results, "stuff", BASE_ARM, m)})
    print(pd.DataFrame(yrows).round(6).to_string(index=False))

    print("\n--- fitted coefficients on the last holdout season ---")
    last = results[(results["model"] == "stuff")
                   & (results["season"] == score_seasons[-1])]
    for component in components:
        c = last[last["component"] == component]
        if not c.empty:
            print(f"  {component}: {c['coef'].iloc[0]}")

    # The gate verdict, in `run_pitcher_backtest.py`'s own shape: a component
    # clears when the stuff arm beats the baseline pooled over the holdout.
    verdict = []
    for component in components:
        r = next(p for p in prows if p["component"] == component
                 and p["arm"] == "stuff" and p["base"] == BASE_ARM)
        verdict.append({"component": component, "diff": r["diff"],
                        "pct": r["pct"], "t": r["t"], "n": r["n"],
                        "clusters": r["n_clusters"],
                        "clears": bool(r["diff"] < 0)})
    v = pd.DataFrame(verdict)
    print("\n=== gate: does the stuff arm beat marcel_pitcher_tuned? ===")
    print(v.round(6).to_string(index=False))
    clears = v[v["clears"]]["component"].tolist()
    withheld = v[~v["clears"]]["component"].tolist()
    print(f"\nSERVE: {', '.join(clears) if clears else '(none)'}")
    print(f"WITHHOLD: {', '.join(withheld) if withheld else '(none)'}")

    # The *serving* verdict is a different and stricter question, and the one
    # docs/pitching-stuff.md's "Serving" section pre-registered before BAS-79
    # wired anything: it is asked of `stuff_additive` (the shape that ships,
    # baseline pinned at 1) and it needs a clustered |t| above SERVE_MIN_T,
    # not merely a negative difference. A component that clears the gate on
    # the free fit and misses this bar is withheld whatever its point estimate
    # says. `src/projections/pitcher_ros.py`'s `LIVE_ENGINE` is this table,
    # written down.
    serve_rows = []
    for component in components:
        r = next(p for p in prows if p["component"] == component
                 and p["arm"] == "stuff_additive" and p["base"] == BASE_ARM)
        c = next(p for p in prows if p["component"] == component
                 and p["arm"] == "stuff_additive"
                 and p["base"] == "stuff_additive_recal")
        serve_rows.append({
            "component": component, "diff": r["diff"], "pct": r["pct"],
            "t": r["t"], "covariate_pct": c["pct"], "covariate_t": c["t"],
            "serves": bool(r["diff"] < 0 and abs(r["t"]) > SERVE_MIN_T)})
    sv = pd.DataFrame(serve_rows)
    print(f"\n=== serving gate: stuff_additive vs {BASE_ARM}, |t| > "
          f"{SERVE_MIN_T} ===")
    print(sv.round(6).to_string(index=False))
    served = sv[sv["serves"]]["component"].tolist()
    not_served = sv[~sv["serves"]]["component"].tolist()
    print(f"\nSERVE (additive): {', '.join(served) if served else '(none)'}")
    print("WITHHOLD (additive): "
          f"{', '.join(not_served) if not_served else '(none)'}")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "components": list(components), "weights": list(weights),
            "ballast": ballast, "tune_seasons": tune_seasons,
            "score_seasons": score_seasons, "cutoffs": list(CUTOFF_MONTHS),
            "min_trials": args.min_trials,
            "scores": json.loads(score_table(results).to_json(orient="records")),
            "paired": json.loads(pd.DataFrame(prows).to_json(orient="records")),
            "split": json.loads(split.to_json(orient="records")),
            "per_season": json.loads(pd.DataFrame(yrows).to_json(orient="records")),
            "gate": json.loads(v.to_json(orient="records")),
            "serving_gate": json.loads(sv.to_json(orient="records")),
            "grid": (json.loads(grid.assign(weights=grid["weights"].astype(str))
                                .to_json(orient="records"))
                     if grid is not None else []),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
