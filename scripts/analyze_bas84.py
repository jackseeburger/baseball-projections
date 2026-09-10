"""BAS-84's scoring pass: the joint model against its single-component twin,
tuned Marcel and the served engine (docs/bayes-joint.md).

`scripts/run_intraseason_backtest_dense.py --stage analyze` already renders
every variant against a fixed list of bases. This script asks the five
questions the BAS-84 pre-registration asks, and only those, so what it writes
maps one-to-one onto predictions 1-5 instead of onto a table a reader has to
re-derive them from:

    1. the pairwise ability correlations, per cutoff, with their 95% intervals
    2. bayes_joint_walk vs bayes_walk on hr_rate
    3. bayes_joint_walk vs bayes_walk on k_rate and bb_rate
    4. bayes_joint_walk vs contact_additive on hr_rate
    5. vacuity: every sigma_step above 0.02, no correlation pinned at +-1

Every paired difference reuses the dense sweep's own `variant_comparison`, so
the clustering (by player, primary; by cell, reported; unclustered, labelled
wrong) is the same math as every other table on this scoreboard. The
correlations come from the fit records' `joint_params`, written by
`src.models.pa_joint.joint_param_summary`.

Reads the checkpoint and the fit records the sweep wrote; runs no MCMC, and
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

JOINT_ARM = "bayes_joint_walk"
# arm, base -- in the order docs/bayes-joint.md's predictions read them.
COMPARISONS = [
    ("bayes_joint_walk", "bayes_walk"),          # predictions 2 and 3
    ("bayes_joint_walk", "marcel_tuned"),
    ("bayes_joint_walk", "contact_additive"),    # prediction 4
    ("bayes_joint_flat", "bayes_flat"),
    ("bayes_walk", "marcel_tuned"),
    ("bayes_walk", "contact_additive"),
    ("contact_additive", "marcel_tuned"),
]
# docs/bayes-joint.md prediction 5: a walk whose step size sits below this has
# collapsed to the flat model, and the comparison is vacuous rather than lost.
SIGMA_STEP_FLOOR = 0.02


def arm_mae(cells: pd.DataFrame, model: str, component: str) -> float:
    """Trials-weighted MAE of one arm, for turning a paired diff into a
    percentage of the base's own error -- which is the unit predictions 2 and
    4 are stated in."""
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
                                if base_mae and np.isfinite(base_mae)
                                else float("nan"))
        rows.append(r)
    return rows


def diagnostics_summary(fits: list[dict], variant: str) -> dict:
    """Worst r-hat, worst ESS and total divergences across one arm's fits.

    A joint fit is recorded once per component it serves, so the same
    posterior appears up to three times; the summary counts *fits*, keyed on
    (season, cutoff), not fit records, or the divergence total would be
    tripled and the "how many fits were unhealthy" fraction would be
    meaningless.
    """
    seen: dict = {}
    for f in fits:
        if f.get("variant") != variant:
            continue
        d = f.get("diagnostics") or {}
        seen[(f.get("cutoff"),)] = d
    if not seen:
        return {}
    rhat = [d["max_rhat"] for d in seen.values() if d.get("max_rhat") is not None]
    ess = [d["min_ess_bulk"] for d in seen.values()
           if d.get("min_ess_bulk") is not None]
    div = [d.get("divergences", 0) for d in seen.values()]
    healthy = [bool(d.get("healthy")) for d in seen.values()]
    worst_var = {}
    for d in seen.values():
        worst_var[d.get("max_rhat_var")] = worst_var.get(d.get("max_rhat_var"), 0) + 1
    return {
        "n_fits": len(seen),
        "max_rhat": float(np.nanmax(rhat)) if rhat else float("nan"),
        "median_max_rhat": float(np.nanmedian(rhat)) if rhat else float("nan"),
        "n_fits_rhat_above_1_01": int(sum(1 for r in rhat if r >= 1.01)),
        "min_ess_bulk": float(np.nanmin(ess)) if ess else float("nan"),
        "median_min_ess_bulk": float(np.nanmedian(ess)) if ess else float("nan"),
        "total_divergences": int(sum(div)),
        "n_fits_with_divergences": int(sum(1 for x in div if x)),
        "n_fits_healthy": int(sum(healthy)),
        "worst_rhat_variable_counts": worst_var,
    }


def joint_params_by_cutoff(fits: list[dict]) -> dict:
    """`{param: {cutoff: {mean, q2.5, q97.5}}}`, deduplicated across components.

    One joint fit fills three components, so the same posterior is recorded on
    up to three fit records per cell. Keyed on the cutoff, so those three
    collapse to one entry rather than being counted three times -- which
    matters, because prediction 1 is a count over *cutoffs*.
    """
    out: dict = {}
    for f in fits:
        params = f.get("joint_params") or {}
        if not params:
            continue
        cutoff = f.get("cutoff")
        for name, s in params.items():
            out.setdefault(name, {})[cutoff] = {
                "mean": s["mean"], "q2.5": s.get("q2.5"), "q97.5": s.get("q97.5"),
            }
    return out


def score_predictions(by_cutoff: dict) -> dict:
    """Predictions 1 and 5, counted over cutoffs."""
    corr = {k: v for k, v in by_cutoff.items() if k.startswith("corr_")}
    steps = {k: v for k, v in by_cutoff.items() if k.startswith("sigma_step_")}
    p1 = {}
    for name, per_cutoff in sorted(corr.items()):
        n = len(per_cutoff)
        excl = sum(1 for s in per_cutoff.values()
                   if s["q2.5"] is not None and (s["q2.5"] > 0 or s["q97.5"] < 0))
        pos = sum(1 for s in per_cutoff.values() if s["mean"] > 0)
        means = np.asarray([s["mean"] for s in per_cutoff.values()], dtype="float64")
        p1[name] = {
            "n_cutoffs": n, "n_excluding_zero": excl,
            "frac_excluding_zero": excl / max(n, 1),
            "n_positive_mean": pos,
            "mean_of_means": float(means.mean()) if n else float("nan"),
            "min": float(means.min()) if n else float("nan"),
            "max": float(means.max()) if n else float("nan"),
        }
    p5 = {"sigma_step": {}, "correlations_pinned": []}
    for name, per_cutoff in sorted(steps.items()):
        means = np.asarray([s["mean"] for s in per_cutoff.values()], dtype="float64")
        p5["sigma_step"][name] = {
            "n_cutoffs": int(means.size),
            "mean": float(means.mean()) if means.size else float("nan"),
            "min": float(means.min()) if means.size else float("nan"),
            "max": float(means.max()) if means.size else float("nan"),
            "n_below_floor": int((means < SIGMA_STEP_FLOOR).sum()),
            "floor": SIGMA_STEP_FLOOR,
        }
    for name, per_cutoff in sorted(corr.items()):
        for cutoff, s in per_cutoff.items():
            if abs(s["mean"]) > 0.99:
                p5["correlations_pinned"].append({"param": name, "cutoff": cutoff,
                                                  "mean": s["mean"]})
    return {"prediction_1_correlations": p1, "prediction_5_vacuity": p5}


def comparisons_by_season(cells: pd.DataFrame, component: str) -> dict:
    """The same table, one season at a time.

    Pooling four seasons hides a sign flip in a single one, and BAS-84's
    prediction 1 is about whether the correlation structure is a fact about
    hitters or a fact about one year — so the per-season cut is not a
    robustness afterthought here, it is part of the question.
    """
    out = {}
    for season, g in cells.groupby("season"):
        out[int(season)] = comparisons(g, component)
    return out


def joint_params_by_season(fits: list[dict]) -> dict:
    """`{season: {param: {mean, n_cutoffs, n_excluding_zero}}}`, deduplicated
    across the components one joint fit serves."""
    per: dict = {}
    for f in fits:
        params = f.get("joint_params") or {}
        if not params:
            continue
        season = int(str(f.get("cutoff"))[:4])
        for name, s in params.items():
            per.setdefault(season, {}).setdefault(name, {})[f.get("cutoff")] = s
    out: dict = {}
    for season, params in sorted(per.items()):
        for name, by_cutoff in sorted(params.items()):
            means = np.asarray([v["mean"] for v in by_cutoff.values()],
                               dtype="float64")
            excl = sum(1 for v in by_cutoff.values()
                       if v.get("q2.5") is not None
                       and (v["q2.5"] > 0 or v["q97.5"] < 0))
            out.setdefault(season, {})[name] = {
                "n_cutoffs": int(means.size),
                "mean": float(means.mean()),
                "min": float(means.min()), "max": float(means.max()),
                "n_excluding_zero": int(excl),
            }
    return out


def render(rows: list[dict]) -> str:
    if not rows:
        return "(nothing scored)"
    aw = max(len(r["arm"]) for r in rows) + 2
    bw = max(len(r["base"]) for r in rows) + 2
    head = (f"{'arm':<{aw}}{'base':<{bw}}{'n':>6}  {'diff':>10}  {'% base':>7}  "
            f"{'t(player)':>10}  {'t(cell)':>8}  {'W-L':>8}")
    lines = [head, "-" * len(head)]

    def num(v, w, p=2):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return f"{'-':>{w}}"
        return f"{'-':>{w}}" if not np.isfinite(v) else f"{v:>{w}.{p}f}"

    for r in rows:
        lines.append(
            f"{r['arm']:<{aw}}{r['base']:<{bw}}{r['n']:>6}  {r['diff']:>+10.5f}  "
            f"{num(r['pct_of_base_mae'], 7, 2)}  "
            f"{num(r['clustered_by_player_t'], 10)}  "
            f"{num(r['clustered_by_cell_t'], 8)}  "
            f"{r['arm_wins_cells']:>3d}-{r['arm_loses_cells']:<3d}")
    return "\n".join(lines)


def render_corr(by_cutoff: dict) -> str:
    corr = {k: v for k, v in by_cutoff.items() if k.startswith("corr_")}
    if not corr:
        return "(no joint fits recorded)"
    cutoffs = sorted({c for v in corr.values() for c in v})
    lines = []
    for name in sorted(corr):
        lines.append(name)
        for cutoff in cutoffs:
            s = corr[name].get(cutoff)
            if s is None:
                continue
            flag = "*" if (s["q2.5"] > 0 or s["q97.5"] < 0) else " "
            lines.append(f"  {cutoff}  {s['mean']:+.3f}  "
                         f"[{s['q2.5']:+.3f}, {s['q97.5']:+.3f}] {flag}")
    lines.append("(* = 95% interval excludes zero)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "data/eval/bas84")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = pd.read_parquet(args.in_dir / "cells_bayes.parquet")
    fits_path = args.in_dir / "bayes_fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else []

    by_cutoff = joint_params_by_cutoff(fits)
    payload: dict = {
        "scope": {
            "seasons": sorted(int(s) for s in cells["season"].unique()),
            "cutoffs": sorted(cells["cutoff"].unique().tolist()),
            "components": sorted(cells["component"].unique().tolist()),
            "arms": sorted(cells["model"].unique().tolist()),
        },
        "wall_time_s": {},
        "comparisons": {},
        "joint_params_by_cutoff": by_cutoff,
        **score_predictions(by_cutoff),
    }
    elapsed = [f["elapsed_s"] for f in fits if f.get("elapsed_s")]
    if elapsed:
        e = np.asarray(elapsed, dtype="float64")
        payload["wall_time_s"] = {"n_fits": int(e.size), "median": float(np.median(e)),
                                  "min": float(e.min()), "max": float(e.max()),
                                  "total_hours": float(e.sum() / 3600.0)}

    payload["diagnostics"] = {
        "bayes_joint_walk": diagnostics_summary(fits, "joint+ability_walk"),
        "bayes_walk": diagnostics_summary(fits, "ability_walk"),
    }
    payload["joint_params_by_season"] = joint_params_by_season(fits)
    payload["comparisons_by_season"] = {}

    for component in payload["scope"]["components"]:
        rows = comparisons(cells, component)
        payload["comparisons"][component] = rows
        payload["comparisons_by_season"][component] = comparisons_by_season(
            cells, component)
        print(f"\n=== {component} (pooled) ===")
        print(render(rows))
        for season, srows in payload["comparisons_by_season"][component].items():
            print(f"\n--- {component}, {season} ---")
            print(render(srows))

    print("\n=== ability correlations by cutoff ===")
    print(render_corr(by_cutoff))
    print("\n=== vacuity (prediction 5) ===")
    for name, s in payload["prediction_5_vacuity"]["sigma_step"].items():
        print(f"  {name:<24} mean {s['mean']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]  "
              f"{s['n_below_floor']}/{s['n_cutoffs']} below {s['floor']}")
    pinned = payload["prediction_5_vacuity"]["correlations_pinned"]
    print(f"  correlations pinned at +-1: {len(pinned)}")

    print("\n=== correlations by season ===")
    for season, params in sorted(payload["joint_params_by_season"].items()):
        for name, s in sorted(params.items()):
            if not name.startswith("corr_"):
                continue
            print(f"  {season}  {name:<24} mean {s['mean']:+.3f}  "
                  f"[{s['min']:+.3f}, {s['max']:+.3f}]  "
                  f"{s['n_excluding_zero']}/{s['n_cutoffs']} cutoffs exclude 0")

    print("\n=== sampler diagnostics ===")
    for arm, d in payload["diagnostics"].items():
        if not d:
            continue
        print(f"  {arm:<18} {d['n_fits']} fits  max r-hat {d['max_rhat']:.3f} "
              f"(median {d['median_max_rhat']:.3f}, "
              f"{d['n_fits_rhat_above_1_01']} at/above 1.01)  "
              f"min ESS {d['min_ess_bulk']:.0f} "
              f"(median {d['median_min_ess_bulk']:.0f})  "
              f"divergences {d['total_divergences']} "
              f"in {d['n_fits_with_divergences']} fits  "
              f"healthy {d['n_fits_healthy']}/{d['n_fits']}")

    out = args.out or (args.in_dir / "analysis_bas84.json")
    out.write_text(json.dumps(payload, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
