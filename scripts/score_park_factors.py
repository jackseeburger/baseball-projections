"""BAS-86 stage 1: score docs/park-factors.md's four pre-registered predictions.

    python scripts/build_park_factors.py
    python scripts/run_intraseason_backtest_dense.py --stage cheap --park-arm \
        --cheap-seasons 2022 2024 2025 2026 --cheap-cadence biweekly \
        --out-dir data/eval/park_factors_stage1
    python scripts/run_intraseason_backtest_dense.py --stage analyze \
        --out-dir data/eval/park_factors_stage1
    python scripts/score_park_factors.py        # -> data/eval/park_factors_stage1.json

This reads the two artifacts the steps above leave behind — the factor
sidecar (`data/features/park_factors.meta.json`) and the sweep's analysis
payload — and writes one evidence file with every number the pre-registration
asks for and the verdict on each prediction. It computes nothing new: if a
number is not in one of those two files, it is not in the verdict either, so
the scoring cannot quietly acquire a degree of freedom the sweep did not have.

The predictions, verbatim from docs/park-factors.md:

1. persistence: year-over-year correlation of the regressed HR factor across
   parks >= 0.6, K factor >= 0.4 — below that, vacuity, stop;
2. HR/PA: `marcel_tuned_park` beats `marcel_tuned` by >= 0.5% of MAE, |t| > 2.5;
3. K% and BB%: |delta| < 0.3%;
4. under §3's effect floor (1.0% at |t| > 2.0) prediction 2's gain does not
   clear serving on its own.

Vacuity: sd of the regressed log HR factor across the 30 parks in a season
>= 0.05.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "data/features/park_factors.meta.json"
DEFAULT_SWEEP = ROOT / "data/eval/park_factors_stage1"
DEFAULT_OUT = ROOT / "data/eval/park_factors_stage1.json"

# The thresholds, in one place, named after what they are rather than inlined
# into the tests below.
PERSISTENCE_BAR = {"hr_rate": 0.60, "k_rate": 0.40}
VACUITY_LOG_SD = 0.05
GAIN_PCT_BAR = 0.5          # prediction 2: % of the base's MAE
GAIN_T_BAR = 2.5
NULL_PCT_BAR = 0.3          # prediction 3: |delta| under this is "no effect"
FLOOR_PCT = 1.0             # architecture.md §3's effect floor
FLOOR_T = 2.0
ARM = "marcel_tuned_park"
BASE = "marcel_tuned"


def summarise_coverage(coverage: list | None) -> list | None:
    """Per-cell coverage records → one row per (component, season).

    The per-cell file stays where the sweep wrote it; what belongs in the
    evidence is the share of scored batters the arm actually moved and how far
    it moved them, which is a handful of numbers and not two hundred.
    """
    if not coverage:
        return None
    out: dict = {}
    for r in coverage:
        key = (r["component"], int(r["season"]))
        agg = out.setdefault(key, {"component": r["component"],
                                   "season": int(r["season"]), "cells": 0,
                                   "n": 0, "n_with_park": 0, "_logsum": 0.0})
        agg["cells"] += 1
        agg["n"] += int(r["n"])
        agg["n_with_park"] += int(r["n_with_park"])
        agg["_logsum"] += float(r["mean_abs_log_factor"]) * int(r["n"])
    rows = []
    for agg in out.values():
        n = max(agg.pop("n"), 1)
        logsum = agg.pop("_logsum")
        rows.append({**agg, "n": n,
                     "share_with_park": agg["n_with_park"] / n,
                     "mean_abs_log_factor": logsum / n})
    return sorted(rows, key=lambda r: (r["component"], r["season"]))


def _arm_row(park: dict, component: str, arm: str = ARM, season=None) -> dict:
    rows = park["by_season"] if season is not None else park["pooled"]
    for r in rows:
        if (r["arm"] == arm and r["component"] == component
                and (season is None or r.get("season") == season)):
            return r
    return {}


def _t(row: dict) -> float:
    t = row.get("clustered_by_player_t")
    return float("nan") if t is None else float(t)


def score(meta: dict, analysis: dict, coverage: list | None = None) -> dict:
    park = analysis.get("park_arm") or {}
    if not park:
        raise SystemExit("the analysis payload has no park_arm section — run "
                         "the cheap sweep with --park-arm first")

    # ─── vacuity ───
    spread = meta["log_factor_sd_by_season"]
    scored_seasons = sorted({int(r["season"]) for r in park["by_season"]
                             if r.get("season") is not None})
    hr_sd = {int(r["game_year"]): float(r["sd"]) for r in spread
             if r["component"] == "hr_rate"}
    vacuity = {
        "measure": "sd of the regressed log HR factor across the 30 parks",
        "bar": VACUITY_LOG_SD,
        "by_season": hr_sd,
        "scored_seasons": {s: hr_sd.get(s) for s in scored_seasons},
        "min_over_scored_seasons": min((hr_sd[s] for s in scored_seasons
                                        if s in hr_sd), default=None),
    }
    vacuity["verdict"] = ("not vacuous"
                          if (vacuity["min_over_scored_seasons"] or 0) >= VACUITY_LOG_SD
                          else "VACUOUS")

    # ─── prediction 1: persistence ───
    yoy = {r["component"]: float(r["corr"])
           for r in meta["year_over_year_correlation"] if r.get("season") is None}
    single = {r["component"]: float(r["corr"])
              for r in meta["single_season_persistence"]}
    p1 = {
        "claim": ("year-over-year correlation of the regressed factor across "
                  "parks: HR >= 0.60, K >= 0.40"),
        "bars": PERSISTENCE_BAR,
        "stamped_factor_corr": yoy,
        "single_season_raw_corr": single,
        "note": ("the stamped factors for consecutive seasons share two of "
                 "their three window seasons, so the first number is a "
                 "persistence check on the artifact a model consumes; the "
                 "single-season column is the same question asked of windows "
                 "that share nothing"),
        "passes": {c: bool(yoy.get(c, float("nan")) >= bar)
                   for c, bar in PERSISTENCE_BAR.items()},
    }
    p1["verdict"] = "held" if all(p1["passes"].values()) else "FAILED"

    # ─── prediction 2: HR/PA ───
    hr = _arm_row(park, "hr_rate")
    p2 = {
        "claim": f"{ARM} beats {BASE} on HR/PA by >= {GAIN_PCT_BAR}% of MAE, "
                 f"|t| > {GAIN_T_BAR}",
        "diff": hr.get("diff"), "pct_of_base": hr.get("pct_of_base"),
        "t_player": _t(hr), "t_cell": hr.get("clustered_by_cell_t"),
        "n": hr.get("n"), "n_clusters": hr.get("clustered_by_player_n_clusters"),
        "wins_losses": [hr.get("arm_wins_cells"), hr.get("arm_loses_cells")],
        "base_mae": hr.get("base_mae"), "arm_mae": hr.get("arm_mae"),
    }
    beat = (p2["pct_of_base"] is not None and p2["pct_of_base"] <= -GAIN_PCT_BAR
            and abs(p2["t_player"]) > GAIN_T_BAR)
    p2["verdict"] = "held" if beat else "FAILED"
    if p2["pct_of_base"] is not None and p2["pct_of_base"] > 0:
        p2["verdict"] = "FAILED (the arm is worse than the baseline)"

    # ─── prediction 3: K% and BB% ───
    p3 = {"claim": f"K% and BB%: |delta| < {NULL_PCT_BAR}% of MAE", "components": {}}
    for component in ("k_rate", "bb_rate"):
        row = _arm_row(park, component)
        p3["components"][component] = {
            "diff": row.get("diff"), "pct_of_base": row.get("pct_of_base"),
            "t_player": _t(row), "n": row.get("n"),
            "within_bar": bool(abs(row.get("pct_of_base", float("nan")))
                               < NULL_PCT_BAR),
        }
    p3["verdict"] = ("held" if all(v["within_bar"] for v
                                   in p3["components"].values()) else "FAILED")

    # ─── prediction 4: the serving floor ───
    clears = (p2["pct_of_base"] is not None
              and p2["pct_of_base"] <= -FLOOR_PCT
              and abs(p2["t_player"]) > FLOOR_T)
    p4 = {
        "claim": (f"under architecture.md §3's effect floor ({FLOOR_PCT}% of the "
                  f"served baseline's MAE at |t| > {FLOOR_T}), prediction 2's "
                  f"gain does not clear serving on its own"),
        "floor_pct": FLOOR_PCT, "floor_t": FLOOR_T,
        "gain_pct_of_base": p2["pct_of_base"], "t_player": p2["t_player"],
        "clears_the_floor": bool(clears),
        "note": ("the arm has no fitted parameter — the factor is applied as "
                 "measured — so there is no recalibration control to subtract "
                 "and the whole gain is the measurement's share"),
    }
    p4["verdict"] = "held (does not clear)" if not clears else "FAILED (it clears)"
    if not clears and (p2["pct_of_base"] or 0) > 0:
        # Held, but not for the reason the pre-registration had in mind: the
        # floor was there to catch a real-but-small gain, and there is no gain.
        p4["verdict"] = ("held vacuously — the arm has no gain to clear the "
                         "floor with; it is worse than the baseline")

    out = {
        "ticket": "BAS-86",
        "pre_registration": "docs/park-factors.md",
        "arm": ARM, "base": BASE,
        "factors": {
            "path": "data/features/park_factors.parquet",
            "built_at": meta.get("built_at"),
            "seasons_with_data": meta.get("seasons_with_data"),
            "windows": meta.get("windows"),
            "ballast": meta.get("ballast"),
            "ballast_chosen_on": meta.get("ballast_chosen_on"),
            "ballast_selection_rule": meta.get("ballast_selection_rule"),
            "known_park_changes": meta.get("known_park_changes"),
            "notes": meta.get("notes"),
        },
        "loso_persistence": meta.get("loso_persistence"),
        "log_factor_sd_by_season": spread,
        "vacuity": vacuity,
        "scope": {
            "seasons": scored_seasons,
            "cutoffs": sorted({r["cutoff"] for r in analysis.get(
                "cheap_n_by_cutoff", [])}) or None,
            "components": sorted({r["component"] for r in park["pooled"]}),
        },
        "arm_pooled": park["pooled"],
        "arm_by_season": park["by_season"],
        # Where the arm's loss comes from: how much of the park tilt the
        # baseline's own projection already carries. See `park_tilt` in
        # scripts/run_intraseason_backtest_dense.py.
        "park_tilt": park.get("tilt"),
        "arm_coverage": summarise_coverage(coverage),
        "predictions": {"1_persistence": p1, "2_hr_gain": p2,
                        "3_k_bb_null": p3, "4_serving_floor": p4},
    }
    return out


def render(payload: dict) -> str:
    lines = []
    v = payload["vacuity"]
    lines.append(f"vacuity: sd(log HR factor) min over scored seasons = "
                 f"{v['min_over_scored_seasons']:.4f} (bar {v['bar']}) -> "
                 f"{v['verdict']}")
    for key, p in payload["predictions"].items():
        lines.append(f"\n{key}: {p['verdict']}")
        lines.append(f"  {p['claim']}")
        if key == "1_persistence":
            for c, bar in p["bars"].items():
                lines.append(f"  {c}: stamped {p['stamped_factor_corr'][c]:.3f} "
                             f"(bar {bar}); single-season "
                             f"{p['single_season_raw_corr'][c]:.3f}")
        elif key == "2_hr_gain":
            lines.append(f"  diff {p['diff']:+.6f} = {p['pct_of_base']:+.2f}% of "
                         f"base MAE {p['base_mae']:.5f}; t(player) "
                         f"{p['t_player']:+.2f}, n {p['n']}, "
                         f"W-L {p['wins_losses'][0]}-{p['wins_losses'][1]}")
        elif key == "3_k_bb_null":
            for c, r in p["components"].items():
                lines.append(f"  {c}: {r['pct_of_base']:+.2f}% of base MAE, "
                             f"t(player) {r['t_player']:+.2f}")
        elif key == "4_serving_floor":
            lines.append(f"  gain {p['gain_pct_of_base']:+.2f}% at t "
                         f"{p['t_player']:+.2f} -> clears = {p['clears_the_floor']}")
    tilt = payload.get("park_tilt")
    if tilt:
        lines.append("\nhow much park tilt each side already carries "
                     "(slope on the log factor):")
        for r in tilt:
            lines.append(f"  {r['component']:<9} realized {r['slope_realized']:+.2f}"
                         f"  base {r['slope_base']:+.2f}"
                         f"  arm {r['slope_arm']:+.2f}"
                         f"  excess {r['excess_over_realized']:+.2f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text())
    analysis = json.loads((args.sweep_dir / "analysis.json").read_text())
    cov_path = args.sweep_dir / "park_arm_coverage.json"
    coverage = json.loads(cov_path.read_text()) if cov_path.exists() else None

    payload = score(meta, analysis, coverage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(render(payload))
    print(f"\nevidence -> {args.out}")


if __name__ == "__main__":
    main()
