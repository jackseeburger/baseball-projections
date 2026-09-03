"""Station A, pitchers: score the rate arms walk-forward against the dumb baselines.

    python scripts/run_pitcher_backtest.py
    python scripts/run_pitcher_backtest.py --markdown
    python scripts/run_pitcher_backtest.py --components p_k_rate --cutoffs 2026-07-01

Five cells, the same shape the hitter side is judged on: the three 2026
intra-season cutoffs (train on everything before the date, score every batter
faced on or after it) plus season-level 2025 and 2026 (train through Y-1,
score Y). Training is `data/parquet/pitcher_seasons_api.parquet` for completed
seasons; at a cutoff the current season comes from the PA parquet aggregated
by pitcher, which is the only source with dates on it.

Arms:

    marcel_pitcher_tuned      the candidate — fitted per-component ballast,
                              recency weights and constrained age curve, frozen
                              in src/eval/marcel_pitcher_params.json
    marcel_pitcher            stock: 5/4/3, 2x the published stabilization
                              point as ballast, no age term. On K, BB+HBP and
                              HR this is exactly what station E already runs.
    marcel_pitcher_*_preseason  the same two with the partial season withheld —
                              the controls that isolate in-season information
    season_to_date            this year's rate regressed to league with the
                              component's stabilization point
    previous_season           last complete season, unregressed
    league_average            league rate through the cutoff

The gate (architecture.md section 3) for serving a component on the site: the
served arm must beat **every** dumb baseline — league average, previous season,
season to date — on the trials-weighted paired per-pitcher difference in
absolute error, pooled across the cells. Per-cell differences and their SEs are
printed too, and they are the honest ones: the five cells share pitchers and
the three cutoffs are nested windows of one season, so the pooled SE is
optimistic. `--markdown` prints the doc tables; the gate verdict prints either
way, naming the components that clear and the components that do not.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.pa_outcomes import load_pa_outcomes
from src.eval import pitchers as P
from src.eval import tuning
from src.eval.backtest import backtest, score

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPONENTS = list(P.COMPONENT_ORDER)
DEFAULT_CUTOFFS = ["2026-05-01", "2026-07-01", "2026-08-01"]
DEFAULT_SEASONS = [2025, 2026]
# The live arm first.
ARM_ORDER = ["marcel_pitcher_tuned", "marcel_pitcher", "season_to_date",
             "marcel_pitcher_tuned_preseason", "marcel_pitcher_preseason",
             "previous_season", "league_average"]
# The arms the gate is measured against: the three that use no model at all.
DUMB_BASELINES = ["league_average", "previous_season", "season_to_date"]
SERVED_ARM = "marcel_pitcher_tuned"

COMPONENT_LABEL = {
    "p_k_rate": "K%", "p_bb_rate": "BB%", "p_bbhbp_rate": "(BB+HBP)%",
    "p_hr_rate": "HR/BF", "p_babip": "BABIP",
}


def load_seasons(path: Path) -> pd.DataFrame:
    """The pitcher season table, with derived counts and Chadwick ages."""
    seasons = P.normalize_pitcher_seasons(pd.read_parquet(path))
    if "age" not in seasons.columns or seasons["age"].isna().all():
        seasons = tuning.chadwick_ages(seasons, id_col="pitcher")
    return seasons


def providers_for(intraseason: bool, params=None) -> dict:
    """The arms for one cell. A season-level cell has no partial season, so the
    two `_preseason` controls and `season_to_date` are dropped there rather
    than scored as duplicates of the arms they control for."""
    arms = dict(P.PITCHER_INTRASEASON_BASELINES)
    if params is not None:
        arms["marcel_pitcher_tuned"] = P.pitcher_tuned_provider(params)
        arms["marcel_pitcher_tuned_preseason"] = (
            lambda train, spec, year, _p=params: P.marcel_pitcher_tuned(
                P.full_seasons(train), spec, year, params=_p))
    if not intraseason:
        for key in ("season_to_date", "marcel_pitcher_preseason",
                    "marcel_pitcher_tuned_preseason"):
            arms.pop(key, None)
    return arms


def run(components: list[str], cutoffs: list[str], season_years: list[int],
        seasons: pd.DataFrame, pa: pd.DataFrame, min_trials: int,
        params=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores, paired). One row per arm per cell, and the served arm's paired
    difference against every other arm in the same cell."""
    rows, paired = [], []
    for component in components:
        cells = ([("season", y, None) for y in season_years]
                 + [("cutoff", 2026, c) for c in cutoffs])
        for kind, year, cutoff in cells:
            arms = providers_for(kind == "cutoff", params)
            if kind == "season":
                results = backtest(component, year - 1, year, seasons=seasons,
                                   providers=arms, min_trials=min_trials)
                label = str(year)
            else:
                results = backtest(component, cutoff_date=cutoff,
                                   predict_year=year, seasons=seasons,
                                   pa_frame=pa, providers=arms,
                                   min_trials=min_trials)
                label = cutoff
            rows.append(score(results).assign(cell=label, kind=kind))
            present = set(results["model"])
            for other in present - {SERVED_ARM}:
                d = tuning.paired_from_results(results, SERVED_ARM, other)
                paired.append({"component": component, "cell": label,
                               "kind": kind, "vs": other, **d})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(paired)


def pool_cells(g: pd.DataFrame) -> dict:
    """Pool one component's per-cell paired differences, n-weighted.

    The cells are disjoint pitcher-years at the season level and nested windows
    of one season at the cutoffs, so this SE is optimistic and the per-cell SEs
    are the honest ones. It is the right summary for the gate all the same: the
    question is whether the arm is better *as a component*, not whether it wins
    a 12-pitcher August cell.
    """
    w = g["n"].to_numpy(dtype="float64")
    diff = float(np.sum(w * g["diff"]) / np.sum(w))
    se = float(np.sqrt(np.sum((w * g["se"]) ** 2)) / np.sum(w))
    return {"cells": len(g), "n_pitcher_cells": int(g["n"].sum()),
            "cells_better": int((g["diff"] < 0).sum()),
            "pooled_diff": diff, "se": se,
            "t": diff / se if se else float("nan")}


def gate(paired: pd.DataFrame, components: list[str]
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-baseline detail, per-component verdict).

    The gate as this task sets it: the served arm must beat **every** dumb
    baseline out of sample on every component it serves. A component clears
    when its pooled paired difference against each of league average, previous
    season and season to date is below zero. Pooling is across cells within the
    component — one thin August cell going the other way is not a verdict, and
    a component that loses a baseline outright is not served.
    """
    detail, verdict = [], []
    for component in components:
        g = paired[(paired["component"] == component)
                   & (paired["vs"].isin(DUMB_BASELINES))]
        if g.empty:
            continue
        rows = []
        for other in DUMB_BASELINES:
            h = g[g["vs"] == other]
            if h.empty:
                continue
            rows.append({"component": component, "vs": other, **pool_cells(h)})
        detail.extend(rows)
        pooled = pd.DataFrame(rows)
        worst = pooled.loc[pooled["pooled_diff"].idxmax()]
        verdict.append({
            "component": component,
            "baselines_beaten": int((pooled["pooled_diff"] < 0).sum()),
            "of": len(pooled),
            "cells_better": int((g["diff"] < 0).sum()),
            "of_cells": len(g),
            "closest_baseline": str(worst["vs"]),
            "closest_diff": float(worst["pooled_diff"]),
            "closest_t": float(worst["t"]),
            "clears": bool((pooled["pooled_diff"] < 0).all()),
        })
    return pd.DataFrame(detail), pd.DataFrame(verdict)


# --- markdown ----------------------------------------------------------------

def mae_markdown(scores: pd.DataFrame, components: list[str],
                 cells: list[str]) -> str:
    lines = ["| Component | Arm | " + " | ".join(cells) + " |",
             "|---|---|" + "---|" * len(cells)]
    for component in components:
        g = scores[scores["component"] == component]
        if g.empty:
            continue
        wide = g.pivot(index="model", columns="cell", values="mae")
        first = True
        for model in [m for m in ARM_ORDER if m in wide.index]:
            label = f"**{COMPONENT_LABEL.get(component, component)}**" if first else ""
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
                    cells: list[str], versus: list[str]) -> str:
    """Served arm minus baseline, per component x cell, one block per baseline."""
    blocks = []
    for other in versus:
        g = paired[paired["vs"] == other]
        if g.empty:
            continue
        lines = [f"**vs `{other}`** (negative = the served arm is better; "
                 "trials-weighted paired difference in absolute error ± SE)",
                 "", "| Component | " + " | ".join(cells) + " |",
                 "|---|" + "---|" * len(cells)]
        for component in components:
            h = g[g["component"] == component].set_index("cell")
            out = []
            for cell in cells:
                if cell not in h.index:
                    out.append("—")
                    continue
                r = h.loc[cell]
                txt = f"{r['diff']:+.5f} ± {r['se']:.5f} (t {r['t']:+.1f})"
                out.append(f"**{txt}**" if r["diff"] < 0 else txt)
            lines.append(f"| {COMPONENT_LABEL.get(component, component)} | "
                         + " | ".join(out) + " |")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    p.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    p.add_argument("--season-years", nargs="+", type=int, default=DEFAULT_SEASONS)
    p.add_argument("--min-trials", type=int, default=100)
    p.add_argument("--seasons", type=Path,
                   default=ROOT / "data/parquet/pitcher_seasons_api.parquet")
    p.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None,
                   help="write the scores to PATH as JSON, for "
                        "scripts/build_accuracy_json.py")
    args = p.parse_args()

    seasons = load_seasons(args.seasons)
    pa = load_pa_outcomes(2026, data_dir=args.pa_dir)
    scores, paired = run(args.components, args.cutoffs, args.season_years,
                         seasons, pa, args.min_trials)
    cells = [str(y) for y in args.season_years] + list(args.cutoffs)

    print("\n=== MAE by arm and cell ===")
    if args.markdown:
        print(mae_markdown(scores, args.components, cells))
    else:
        print(scores.round(5).to_string(index=False))

    print(f"\n=== paired: {SERVED_ARM} minus each baseline ===")
    if args.markdown:
        print(paired_markdown(paired, args.components, cells, DUMB_BASELINES))
    else:
        print(paired[paired["vs"].isin(DUMB_BASELINES)]
              .round(6).to_string(index=False))

    detail, verdict = gate(paired, args.components)
    print("\n=== gate: pooled across cells, does the served arm beat every "
          "dumb baseline? ===")
    print(detail.round(6).to_string(index=False))
    print()
    print(verdict.round(6).to_string(index=False))
    served = verdict[verdict["clears"]]["component"].tolist()
    withheld = verdict[~verdict["clears"]]["component"].tolist()
    print(f"\nSERVE: {', '.join(served) if served else '(none)'}")
    print(f"WITHHOLD: {', '.join(withheld) if withheld else '(none)'}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.csv_out, index=False)
        print(f"\nwrote {args.csv_out}")
    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_trials": args.min_trials,
            "components": args.components,
            "component_labels": {c: COMPONENT_LABEL.get(c, c)
                                 for c in args.components},
            "cells": cells,
            "cutoffs": args.cutoffs,
            "season_years": args.season_years,
            "arms": [a for a in ARM_ORDER if a in set(scores["model"])],
            "served_arm": SERVED_ARM,
            "dumb_baselines": DUMB_BASELINES,
            "last_pa_date": str(pd.to_datetime(pa["game_date"]).max().date()),
            "scores": json.loads(scores.to_json(orient="records")),
            "paired": json.loads(paired.to_json(orient="records")),
            "gate": json.loads(verdict.to_json(orient="records")),
            "gate_detail": json.loads(detail.to_json(orient="records")),
            "served_components": served,
            "withheld_components": withheld,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
