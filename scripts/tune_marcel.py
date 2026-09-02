"""Fit Marcel's constants by walk-forward, freeze them, and score the holdout.

Stock Marcel's numbers are Tango's defaults: 5/4/3 recency, one 200-trial
ballast for every component, one age curve. Marcel-with-partial is now the
production rest-of-season projection and the bar every station A model has to
clear, so the question is whether those constants are the *right* constants
for these five components. `marcel_tuned` makes them parameters; this script
fits them and then tries to break the fit on data the fit never saw.

    tune    coordinate search per component on predict years 2020-2024
            (train <= Y-1, score Y), objective = mean trials-weighted MAE.
            Writes src/eval/marcel_params.json.
    score   freeze those params and score the holdout: season-level 2025
            (train <= 2024) and 2026 (train <= 2025), plus the 2026
            intra-season cutoffs 05-01 / 07-01 / 08-01 through the
            cutoff_date path, with the paired per-player difference against
            stock Marcel and its SE.

Ages come from the Chadwick register (`src/data/birthdates.py`, age as of
June 30) — real birthdates, not the `debut - 23` proxy the early Bayesian
fits used. `--api-ages` falls back to the season table's Stats API age.

Usage:
    python scripts/tune_marcel.py                 # tune, then score
    python scripts/tune_marcel.py --skip-tune     # score the committed params
    python scripts/tune_marcel.py --markdown      # doc tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.pa_outcomes import load_pa_outcomes
from src.eval import backtest, score, tuning
from src.eval.backtest import COMPONENTS, frame_provider
from src.eval.baselines import (
    INTRASEASON_BASELINES,
    STOCK_PARAMS,
    load_marcel_params,
    marcel_tuned_provider,
    save_marcel_params,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPONENTS = ["k_rate", "bb_rate", "hr_rate", "babip", "iso"]
TUNE_YEARS = [2020, 2021, 2022, 2023, 2024]     # train <= Y-1, score Y
SEASON_HOLDOUT = [2025, 2026]                    # never seen by the search
CUTOFFS = ["2026-05-01", "2026-07-01", "2026-08-01"]
ARM_ORDER = ["marcel_tuned", "marcel_tuned_noage", "marcel", "marcel_preseason",
             "season_to_date", "bayes_preseason", "previous_season",
             "league_average"]


# --- tuning -----------------------------------------------------------------

def inner_check(seasons: pd.DataFrame, component: str, years: list[int],
                min_trials: int, passes: int, n_validate: int = 2) -> dict:
    """Fit inside the tuning window and validate inside it too.

    Splits the tuning years into fit years and the last `n_validate` of them,
    so "do these parameters generalise at all, or is the search fitting
    noise?" can be asked without spending the holdout. `tune` uses the answer
    as a guard: a component whose fit does not beat stock here keeps stock's
    constants, so the frozen file never ships a component the procedure
    itself could tell was noise.
    """
    spec = COMPONENTS[component]
    fit_years, val_years = years[:-n_validate], years[-n_validate:]
    fit_splits = tuning.make_splits(seasons, component, fit_years, min_trials)
    val_splits = tuning.make_splits(seasons, component, val_years, min_trials)
    stock = STOCK_PARAMS[component]
    full, _ = tuning.coordinate_search(fit_splits, spec, start=stock, passes=passes)
    bw, _ = tuning.coordinate_search(fit_splits, spec, start=stock, passes=passes,
                                     axes=["ballast", "weights"])
    base = tuning.evaluate(val_splits, spec, stock)["mae"]
    full_mae = tuning.evaluate(val_splits, spec, full)["mae"]
    bw_mae = tuning.evaluate(val_splits, spec, bw)["mae"]
    return {
        "fit_years": [fit_years[0], fit_years[-1]],
        "validate_years": [val_years[0], val_years[-1]],
        "stock_mae": base,
        "tuned_mae": full_mae,
        "tuned_ballast_weights_only_mae": bw_mae,
        "tuned_pct": 100.0 * (full_mae - base) / base,
        "tuned_ballast_weights_only_pct": 100.0 * (bw_mae - base) / base,
        "generalises": bool(full_mae < base),
    }


def tune(seasons: pd.DataFrame, components: list[str], years: list[int],
         min_trials: int, passes: int, verbose: bool,
         guard: bool = True) -> tuple[dict, dict, dict]:
    """Coordinate search per component.

    Fits two nested variants and returns (full, restricted, in-sample summary):

        full        all five axes — ballast, the weight ratios, and the three
                    age numbers. This is what gets frozen.
        restricted  ballast and weights only, age curve left at stock.

    The restricted fit exists because the full one has a degeneracy worth
    naming: with the peak age at an end of its grid and both slopes equal,
    the "age curve" is a straight line in age, which is partly a *level*
    correction — Marcel regresses to the last training season's league rate
    while a player's own weighted history spans three, so a league trending
    up or down leaves a bias a linear age term can absorb. Scoring both on
    the holdout says whether the tuned age curve is aging or bookkeeping.
    """
    params, restricted, summary = {}, {}, {}
    for component in components:
        spec = COMPONENTS[component]
        splits = tuning.make_splits(seasons, component, years, min_trials)
        stock = STOCK_PARAMS[component]
        base = tuning.evaluate(splits, spec, stock)
        scored = ", ".join(f"{s.predict_year}:{len(s.realized)}" for s in splits)
        print(f"\n=== {component}: tuning on {years[0]}-{years[-1]} "
              f"(scored players {scored}) ===")
        print(f"    stock       mae={base['mae']:.6f} log_loss={base['log_loss']:.6f}")

        best, trace = tuning.coordinate_search(
            splits, spec, start=stock, passes=passes, verbose=verbose)
        fit = tuning.evaluate(splits, spec, best)
        gain = (fit["mae"] - base["mae"]) / base["mae"]
        print(f"    tuned       mae={fit['mae']:.6f} "
              f"log_loss={fit['log_loss']:.6f} ({gain:+.2%})")
        print(f"      {best.to_dict()}")

        bw, _ = tuning.coordinate_search(
            splits, spec, start=stock, passes=passes,
            axes=["ballast", "weights"], verbose=verbose)
        bw_fit = tuning.evaluate(splits, spec, bw)
        bw_gain = (bw_fit["mae"] - base["mae"]) / base["mae"]
        print(f"    no-age fit  mae={bw_fit['mae']:.6f} "
              f"log_loss={bw_fit['log_loss']:.6f} ({bw_gain:+.2%})")
        print(f"      {bw.to_dict()}")

        inner = inner_check(seasons, component, years, min_trials, passes)
        print(f"    inner validation (fit {inner['fit_years'][0]}-"
              f"{inner['fit_years'][1]}, score {inner['validate_years'][0]}-"
              f"{inner['validate_years'][1]}): tuned {inner['tuned_pct']:+.2f}%, "
              f"no-age {inner['tuned_ballast_weights_only_pct']:+.2f}%")
        if guard and not inner["generalises"]:
            print(f"    GUARD: the fit does not generalise inside the tuning "
                  f"window — {component} keeps stock Marcel's constants")
            best = stock

        params[component] = best
        restricted[component] = bw
        summary[component] = {
            "years": years,
            "n_scored_by_year": {str(s.predict_year): len(s.realized) for s in splits},
            "stock": {k: base[k] for k in ("mae", "rmse", "log_loss")},
            "tuned": {k: fit[k] for k in ("mae", "rmse", "log_loss")},
            "tuned_ballast_weights_only": {
                k: bw_fit[k] for k in ("mae", "rmse", "log_loss")},
            "mae_gain_pct": 100.0 * gain,
            "mae_gain_pct_ballast_weights_only": 100.0 * bw_gain,
            "log_loss_gain_pct": (
                100.0 * (fit["log_loss"] - base["log_loss"]) / base["log_loss"]
                if np.isfinite(base["log_loss"]) else None),
            "mae_by_year": {
                str(a["predict_year"]): {"stock": a["mae"], "tuned": b["mae"]}
                for a, b in zip(base["by_year"], fit["by_year"])
            },
            "inner_validation": inner,
            "kept_stock": bool(guard and not inner["generalises"]),
            "search_trace": trace[-len(tuning.AXES) - 1:],
        }
    return params, restricted, summary


# --- holdout scoring ---------------------------------------------------------

def arms(component: str, params: dict, projections_dir: Path,
         intraseason: bool, restricted: dict | None = None) -> dict:
    """Every arm scored at one cell, including the tuned Marcel."""
    base = dict(INTRASEASON_BASELINES) if intraseason else {
        k: v for k, v in INTRASEASON_BASELINES.items()
        if k not in ("season_to_date", "marcel_preseason")}
    providers = {"marcel_tuned": marcel_tuned_provider(params), **base}
    if restricted:
        providers["marcel_tuned_noage"] = marcel_tuned_provider(restricted)
    path = projections_dir / f"{component}_projections_2026.parquet"
    if path.exists():
        providers["bayes_preseason"] = frame_provider(
            pd.read_parquet(path), pred_col=f"projected_{component}")
    return providers


def score_cells(seasons: pd.DataFrame, pa: pd.DataFrame, components: list[str],
                params: dict, projections_dir: Path, min_trials: int,
                season_years: list[int], cutoffs: list[str],
                restricted: dict | None = None) -> pd.DataFrame:
    """MAE/log-loss per arm and the tuned-vs-stock paired difference, per cell."""
    rows, paired = [], []
    for component in components:
        cells = ([("season", y, None) for y in season_years]
                 + [("cutoff", 2026, c) for c in cutoffs])
        for kind, year, cutoff in cells:
            providers = arms(component, params, projections_dir,
                             intraseason=(kind == "cutoff"),
                             restricted=restricted)
            if kind == "season":
                # bayes_preseason is a 2026 file; it has nothing to say about 2025.
                if year != 2026:
                    providers.pop("bayes_preseason", None)
                results = backtest(component, year - 1, year, seasons=seasons,
                                   providers=providers, min_trials=min_trials)
                label = str(year)
            else:
                results = backtest(component, cutoff_date=cutoff, predict_year=year,
                                   seasons=seasons, pa_frame=pa,
                                   providers=providers, min_trials=min_trials)
                label = cutoff
            rows.append(score(results).assign(cell=label, kind=kind))
            for arm in ([a for a in ("marcel_tuned", "marcel_tuned_noage")
                         if a in providers]):
                d = tuning.paired_from_results(results, arm, "marcel")
                paired.append({"component": component, "cell": label,
                               "kind": kind, "arm": arm, **d})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(paired)


def pooled(paired: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Pool the per-cell paired differences within a component (n-weighted).

    Cells are disjoint player-years, so the pooled SE is the n-weighted
    combination of independent cell SEs.
    """
    stock = (scores[scores["model"] == "marcel"]
             .set_index(["component", "cell"])["mae"])
    out = []
    for (arm, component), g in paired.groupby(["arm", "component"]):
        w = g["n"].to_numpy(dtype="float64")
        diff = float(np.sum(w * g["diff"]) / np.sum(w))
        se = float(np.sqrt(np.sum((w * g["se"]) ** 2)) / np.sum(w))
        ref = float(np.average(
            [stock.loc[(component, c)] for c in g["cell"]], weights=w))
        out.append({"arm": arm, "component": component, "cells": len(g),
                    "n_player_cells": int(g["n"].sum()), "pooled_diff": diff,
                    "se": se, "t": diff / se if se else np.nan,
                    "pct_of_stock_mae": 100.0 * diff / ref,
                    "cells_better": int((g["diff"] < 0).sum())})
    return pd.DataFrame(out).sort_values(["arm", "component"])


def pooled_overall(paired: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One number per arm: every cell's difference as a fraction of that
    cell's stock MAE, pooled n-weighted across components and cells.

    Components live on different scales (ISO's MAE is 4x K%'s), so pooling
    raw differences would just be an ISO measurement. Scaling by the cell's
    own stock MAE makes the units percent-of-Marcel and comparable.
    """
    stock = (scores[scores["model"] == "marcel"]
             .set_index(["component", "cell"])["mae"])
    out = []
    for arm, g in paired.groupby("arm"):
        ref = np.array([stock.loc[(c, cell)]
                        for c, cell in zip(g["component"], g["cell"])])
        w = g["n"].to_numpy(dtype="float64")
        rel = g["diff"].to_numpy() / ref
        rel_se = g["se"].to_numpy() / ref
        diff = float(np.sum(w * rel) / np.sum(w))
        se = float(np.sqrt(np.sum((w * rel_se) ** 2)) / np.sum(w))
        out.append({"arm": arm, "cells": len(g),
                    "pooled_pct_of_stock_mae": 100.0 * diff,
                    "se_pct": 100.0 * se, "t": diff / se if se else np.nan})
    return pd.DataFrame(out)


def inner_validation(seasons: pd.DataFrame, components: list[str],
                     years: list[int], min_trials: int, passes: int) -> pd.DataFrame:
    """`inner_check` for every component, as a table."""
    rows = []
    for component in components:
        r = inner_check(seasons, component, years, min_trials, passes)
        rows.append({"component": component,
                     "fit_years": f"{r['fit_years'][0]}-{r['fit_years'][1]}",
                     "validate_years":
                         f"{r['validate_years'][0]}-{r['validate_years'][1]}",
                     **{k: r[k] for k in
                        ("stock_mae", "tuned_mae", "tuned_pct",
                         "tuned_ballast_weights_only_pct", "generalises")}})
    return pd.DataFrame(rows)


# --- markdown ----------------------------------------------------------------

def mae_markdown(scores: pd.DataFrame, components: list[str],
                 cells: list[str]) -> str:
    lines = ["| Component | Arm | " + " | ".join(cells) + " |",
             "|---|---|" + "---|" * len(cells)]
    for component in components:
        g = scores[scores["component"] == component]
        wide = g.pivot(index="model", columns="cell", values="mae")
        first = True
        for model in [m for m in ARM_ORDER if m in wide.index]:
            label = f"**{component}**" if first else ""
            first = False
            out = []
            for cell in cells:
                v = wide.loc[model, cell] if cell in wide.columns else np.nan
                if not np.isfinite(v):
                    out.append("—")
                    continue
                best = v <= wide[cell].min() + 1e-12
                out.append(f"**{v:.4f}**" if best else f"{v:.4f}")
            lines.append(f"| {label} | {model} | " + " | ".join(out) + " |")
    return "\n".join(lines)


def paired_markdown(paired: pd.DataFrame, components: list[str],
                    cells: list[str], arm: str = "marcel_tuned") -> str:
    paired = paired[paired["arm"] == arm]
    lines = ["| Component | " + " | ".join(cells) + " |",
             "|---|" + "---|" * len(cells)]
    for component in components:
        g = paired[paired["component"] == component].set_index("cell")
        out = []
        for cell in cells:
            if cell not in g.index:
                out.append("—")
                continue
            r = g.loc[cell]
            cell_txt = f"{r['diff']:+.5f} ± {r['se']:.5f}"
            out.append(f"**{cell_txt}**" if r["diff"] < 0 else cell_txt)
        lines.append(f"| {component} | " + " | ".join(out) + " |")
    return "\n".join(lines)


# --- main --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    p.add_argument("--tune-years", nargs="+", type=int, default=TUNE_YEARS)
    p.add_argument("--season-holdout", nargs="+", type=int, default=SEASON_HOLDOUT)
    p.add_argument("--cutoffs", nargs="+", default=CUTOFFS)
    p.add_argument("--min-trials", type=int, default=100)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--skip-tune", action="store_true",
                   help="score the committed params instead of refitting")
    p.add_argument("--skip-score", action="store_true")
    p.add_argument("--api-ages", action="store_true",
                   help="use the season table's Stats API age, not Chadwick")
    p.add_argument("--seasons", type=Path,
                   default=ROOT / "data/parquet/hitter_seasons_api.parquet")
    p.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet")
    p.add_argument("--projections-dir", type=Path, default=ROOT / "data/projections")
    p.add_argument("--params-out", type=Path, default=ROOT / "src/eval/marcel_params.json")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--no-guard", action="store_true",
                   help="freeze the fit even for a component whose inner "
                        "validation says it does not generalise")
    p.add_argument("--inner-validation", action="store_true",
                   help="fit on the first tuning years, validate on the last "
                        "two — a look at whether the age axes overfit that "
                        "does not spend the holdout")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    seasons = pd.read_parquet(args.seasons)
    report = tuning.age_source_report(seasons)
    if args.api_ages:
        print(f"ages: Stats API seasonal age from {args.seasons.name}")
    else:
        seasons = tuning.chadwick_ages(seasons)
        print(f"ages: Chadwick register (June 30), coverage "
              f"{report['register_coverage']:.3%} of {report['rows']} rows; "
              f"floor(age) matches the Stats API age on "
              f"{report.get('floor_matches_api_age', float('nan')):.3%} of them")

    if args.inner_validation:
        print("\n=== inner validation (inside the tuning window) ===")
        print(inner_validation(seasons, args.components, args.tune_years,
                               args.min_trials, args.passes).round(6).to_string(index=False))
        return

    if not args.skip_tune:
        params, restricted, summary = tune(
            seasons, args.components, args.tune_years, args.min_trials,
            args.passes, args.verbose, guard=not args.no_guard)
        path = save_marcel_params(
            params, args.params_out,
            generated="scripts/tune_marcel.py",
            method=(f"coordinate search, walk-forward predict years "
                    f"{args.tune_years[0]}-{args.tune_years[-1]} "
                    f"(train <= Y-1), objective = mean trials-weighted MAE, "
                    f"min_trials={args.min_trials}, "
                    f"ages = Chadwick register (June 30)"
                    + ("; components whose fit does not beat stock on an "
                       "inner validation (fit 2020-2022, score 2023-2024) "
                       "keep stock's constants"
                       if not args.no_guard else "")),
            grid=tuning.grid_summary(),
            in_sample=summary,
            variants={"ballast_weights_only":
                      {k: v.to_dict() for k, v in restricted.items()}},
        )
        print(f"\nwrote {path}")
    params = load_marcel_params(args.params_out)
    restricted = json.loads(args.params_out.read_text()).get(
        "variants", {}).get("ballast_weights_only") if args.params_out.exists() else None
    if args.skip_score:
        return

    pa = load_pa_outcomes(2026, data_dir=args.pa_dir)
    scores, paired = score_cells(
        seasons, pa, args.components, params, args.projections_dir,
        args.min_trials, args.season_holdout, args.cutoffs,
        restricted=restricted)
    cells = [str(y) for y in args.season_holdout] + list(args.cutoffs)

    print("\n=== out of sample: MAE by arm and cell ===")
    if args.markdown:
        print(mae_markdown(scores, args.components, cells))
    else:
        print(scores.round(5).to_string(index=False))

    print("\n=== marcel_tuned - marcel: paired per-player difference in "
          "absolute error (trials-weighted, negative = tuned better) ===")
    if args.markdown:
        print(paired_markdown(paired, args.components, cells))
    else:
        print(paired.round(6).to_string(index=False))

    pool = pooled(paired, scores)
    print("\n=== pooled across the holdout cells ===")
    print(pool.round(6).to_string(index=False))
    print("\n=== pooled overall, as a percent of stock Marcel's MAE ===")
    print(pooled_overall(paired, scores).round(4).to_string(index=False))
    print("(the five cells share players and the three cutoffs are nested "
          "windows of the same season, so these pooled SEs are optimistic; "
          "the per-cell SEs above are the honest ones)")

    overall = pooled_overall(paired, scores).set_index("arm")
    for arm in sorted(paired["arm"].unique()):
        g = paired[paired["arm"] == arm]
        p = pool[pool["arm"] == arm]
        better = int((g["diff"] < 0).sum())
        # The gate (architecture.md section 3, as set for this task): beat
        # stock Marcel on the majority of component x cell cells, with the
        # pooled paired difference below zero.
        cleared = (better > len(g) / 2
                   and overall.loc[arm, "pooled_pct_of_stock_mae"] < 0)
        print(f"\n{arm}: wins {better}/{len(g)} component x cell cells; "
              f"pooled {overall.loc[arm, 'pooled_pct_of_stock_mae']:+.2f}% "
              f"of stock MAE (se {overall.loc[arm, 'se_pct']:.2f}); "
              f"pooled difference below zero in "
              f"{int((p['pooled_diff'] < 0).sum())}/{len(p)} components")
        print(f"GATE ({arm}): " + ("CLEARED" if cleared else "NOT CLEARED"))
        worse = p[p["pooled_diff"] > 0]["component"].tolist()
        if worse:
            print(f"  components still worse than stock: {', '.join(worse)}")


if __name__ == "__main__":
    main()
