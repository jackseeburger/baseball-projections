"""BAS-83's scoring pass: the covariate arms against their twins and against
Marcel carrying the same covariate (docs/bayes-covariates.md).

`scripts/run_intraseason_backtest_dense.py --stage analyze` already renders
every variant against a fixed list of bases. This script asks the five
questions the BAS-83 pre-registration asks, and only those, so the evidence
JSON it writes maps one-to-one onto predictions 1-5 rather than onto a table
a reader has to re-derive them from:

    1. bayes_walk+contact vs bayes_walk          (does the covariate pay?)
    2. bayes_walk+contact vs contact_additive    (does the hierarchy pay?)
    3. bayes_flat+contact vs contact_additive
    4. sigma_step with and without the covariate
    5. the beta_cov posteriors, per aggregate

Every paired difference reuses the dense sweep's own `variant_comparison`, so
the clustering (by player, primary; by cell, reported; unclustered, labelled
wrong) is the same math every other table on this scoreboard uses.

Reads the checkpoint and fit records the sweep wrote; runs no MCMC.
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

# arm, base — in the order docs/bayes-covariates.md's predictions read.
COMPARISONS = [
    ("bayes_walk+contact", "bayes_walk"),
    ("bayes_flat+contact", "bayes_flat"),
    ("bayes_walk+contact", "contact_additive"),
    ("bayes_flat+contact", "contact_additive"),
    ("bayes_walk+contact", "marcel_tuned"),
    ("bayes_flat+contact", "marcel_tuned"),
    ("bayes_walk", "contact_additive"),
    ("bayes_walk", "marcel_tuned"),
    ("bayes_flat", "marcel_tuned"),
    ("contact_additive", "marcel_tuned"),
]


def arm_mae(cells: pd.DataFrame, model: str, component: str) -> float:
    """Trials-weighted MAE of one arm, for turning a paired diff into a
    percentage of the base's own error."""
    g = cells[(cells["component"] == component) & (cells["model"] == model)]
    if g.empty:
        return float("nan")
    err = (g["predicted"] - g["realized_rate"]).abs().to_numpy("float64")
    w = g["trials"].to_numpy("float64")
    return float(np.average(err, weights=w))


def comparisons(cells: pd.DataFrame, component: str) -> list[dict]:
    present = set(cells.loc[cells["component"] == component, "model"].unique())
    rows = []
    for arm, base in COMPARISONS:
        if arm not in present or base not in present:
            continue
        r = dense.variant_comparison(cells, arm, base, component)
        if not r:
            continue
        base_mae = arm_mae(cells, base, component)
        r["base_mae"] = base_mae
        r["arm_mae"] = arm_mae(cells, arm, component)
        r["pct_of_base_mae"] = (100.0 * r["diff"] / base_mae
                                if base_mae and np.isfinite(base_mae) else float("nan"))
        rows.append(r)
    return rows


def variant_params(fits: list[dict], component: str) -> dict:
    """Posterior summaries of every variant-own parameter, pooled over the
    cutoffs of one component, keyed by arm.

    Pooling is over fits, not over draws: each fit contributes its own
    posterior mean, and what is reported is the mean and range of those — the
    same convention docs/bayes-variants.md's `sigma_step` table uses. The
    `excludes_zero` flag is the fraction of fits whose own 90% interval keeps
    zero out, which is what prediction 5 is actually about.
    """
    out: dict = {}
    for f in fits:
        if (f.get("component") or "k_rate") != component:
            continue
        arm = f.get("arm") or dense.VARIANT_ARM_NAMES.get(f.get("variant", ""), "?")
        for name, summary in (f.get("variant_params") or {}).items():
            slot = out.setdefault(arm, {}).setdefault(
                name, {"means": [], "excludes_zero": 0, "n_fits": 0})
            slot["means"].append(summary["mean"])
            slot["n_fits"] += 1
            lo, hi = summary.get("q05"), summary.get("q95")
            if lo is not None and hi is not None and (lo > 0 or hi < 0):
                slot["excludes_zero"] += 1
    for arm, params in out.items():
        for name, slot in params.items():
            m = np.asarray(slot.pop("means"), dtype="float64")
            slot["mean"] = float(m.mean())
            slot["min"] = float(m.min())
            slot["max"] = float(m.max())
            slot["frac_fits_excluding_zero"] = (slot["excludes_zero"]
                                                / max(slot["n_fits"], 1))
    return out


def render(rows: list[dict]) -> str:
    if not rows:
        return "(nothing scored)"
    aw = max(len(r["arm"]) for r in rows) + 2
    bw = max(len(r["base"]) for r in rows) + 2
    head = (f"{'arm':<{aw}}{'base':<{bw}}{'n':>6}  {'diff':>10}  {'% base':>7}  "
            f"{'t(player)':>10}  {'t(cell)':>8}  {'W-L':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        def num(v, w, p=2):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return f"{'-':>{w}}"
            return f"{'-':>{w}}" if not np.isfinite(v) else f"{v:>{w}.{p}f}"
        lines.append(
            f"{r['arm']:<{aw}}{r['base']:<{bw}}{r['n']:>6}  {r['diff']:>+10.5f}  "
            f"{num(r['pct_of_base_mae'], 7, 2)}  "
            f"{num(r['clustered_by_player_t'], 10)}  "
            f"{num(r['clustered_by_cell_t'], 8)}  "
            f"{r['arm_wins_cells']:>3d}-{r['arm_loses_cells']:<3d}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "data/eval/bas83")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = pd.read_parquet(args.in_dir / "cells_bayes.parquet")
    fits_path = args.in_dir / "bayes_fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else []

    payload: dict = {
        "scope": {
            "seasons": sorted(int(s) for s in cells["season"].unique()),
            "cutoffs": sorted(cells["cutoff"].unique().tolist()),
            "components": sorted(cells["component"].unique().tolist()),
            "arms": sorted(cells["model"].unique().tolist()),
        },
        "wall_time_s": {},
        "comparisons": {},
        "variant_params": {},
    }
    elapsed = [f["elapsed_s"] for f in fits if f.get("elapsed_s")]
    if elapsed:
        e = np.asarray(elapsed, dtype="float64")
        payload["wall_time_s"] = {"n_fits": int(e.size), "median": float(np.median(e)),
                                  "min": float(e.min()), "max": float(e.max()),
                                  "total_hours": float(e.sum() / 3600.0)}
    for component in payload["scope"]["components"]:
        rows = comparisons(cells, component)
        payload["comparisons"][component] = rows
        # Per-cutoff, for the one comparison the pre-registration's first
        # prediction is about. The covariate's current-season window is a
        # month long in April and five months long in August, so if what the
        # block does is import measurement noise rather than information,
        # that shows up here as a gradient and nowhere else.
        g = cells[cells["component"] == component]
        if {"bayes_walk+contact", "bayes_walk"} <= set(g["model"].unique()):
            payload.setdefault("covariate_effect_by_cutoff", {})[component] = (
                json.loads(dense.paired_by_cell(
                    g, "bayes_walk+contact", "bayes_walk"
                ).sort_values("cutoff").to_json(orient="records")))
        payload["variant_params"][component] = variant_params(fits, component)
        print(f"\n=== {component} ===")
        print(render(rows))
        for arm, params in sorted(payload["variant_params"][component].items()):
            for name, s in sorted(params.items()):
                print(f"  {arm:<26} {name:<22} mean {s['mean']:+.4f}  "
                      f"[{s['min']:+.4f}, {s['max']:+.4f}]  "
                      f"{s['frac_fits_excluding_zero']:.0%} of {s['n_fits']} "
                      f"fits exclude 0")

    out = args.out or (args.in_dir / "analysis_bas83.json")
    out.write_text(json.dumps(payload, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
