"""BAS-85's scoring pass: the measurement model against the served engine
(docs/bayes-measurement.md).

`scripts/run_intraseason_backtest_dense.py --stage analyze` renders every
variant against a fixed list of bases. This script asks the five questions
the BAS-85 pre-registration asks, and only those, so what it writes maps
one-to-one onto predictions 1-5:

    1. the channel loadings, per cutoff, with their 95% intervals — they have
       to exclude zero at EVERY cutoff
    2. the May/August split: `measurement` beats `contact_additive` on HR/PA
       by >= 3% of MAE (|t| > 2.5) at May, and by <= 1.5% at August
    3. pooled over cutoffs: HR/PA is Delta < 0 with clustered t < -2, and K%
       is a draw within +-0.0005
    4. coverage: `measurement`'s 80% posterior interval covers the realised
       rate 75-85% of the time, `bayes_walk`'s covers < 75% at May
    5. vacuity: latent scale above 0.02, |corr| between any two loadings
       below 0.95

Every paired difference reuses the dense sweep's own `variant_comparison`, so
the clustering (by player, primary; by cell, reported; unclustered, labelled
wrong) is the same math as every other table on this scoreboard, and the
May/August split is that same function applied to a subset of cells rather
than a second statistic. The loadings come from the fit records'
`measurement_params`, written by
`src.models.pa_measurement.measurement_param_summary`.

**The split is on the cutoff's month**, not on a hand-picked list of dates,
so a grid run at a different cadence still splits into the two regimes the
pre-registration names — early season, where outcomes are thin and the
channels are supposed to carry the information, and late season, where the
outcome channel has caught up and the gap is supposed to close.

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

MEASUREMENT_ARM = "bayes_measurement_walk"
WALK_ARM = "bayes_walk"
CONTACT_ARM = "contact_additive"

# arm, base -- in the order docs/bayes-measurement.md's predictions read them.
COMPARISONS = [
    (MEASUREMENT_ARM, CONTACT_ARM),      # predictions 2 and 3
    (MEASUREMENT_ARM, WALK_ARM),
    (MEASUREMENT_ARM, "marcel_tuned"),
    (MEASUREMENT_ARM, "bayes_joint_walk"),
    (WALK_ARM, CONTACT_ARM),
    (CONTACT_ARM, "marcel_tuned"),
]

# The two regimes prediction 2 names, by the cutoff's month. May is where the
# channels are supposed to carry the information (a cell has ~150 PA and ~40
# batted balls); August is where the outcome channel has caught up.
REGIMES = {"may": (5,), "august": (8,)}

# Prediction 2's thresholds, as percentages of the base arm's own MAE. The
# sign convention is `variant_comparison`'s: `diff` is arm minus base, so a
# gain is negative and "beats by >= 3%" is pct <= -3.0.
MAY_GAIN_PCT = -3.0
MAY_T = 2.5
AUGUST_MAX_GAP_PCT = 1.5
# Prediction 3: pooled HR/PA has to be a real gain, pooled K% a draw.
POOLED_T = -2.0
K_DRAW_BAND = 0.0005
# Prediction 4.
COVERAGE_TARGET = (0.75, 0.85)
# Prediction 5.
LATENT_SCALE_FLOOR = 0.02
MAX_LOADING_CORR = 0.95

LOADINGS = ("lambda_barrel", "lambda_ev", "lambda_whiff")


def cutoff_month(cutoff: str) -> int:
    return int(pd.Timestamp(cutoff).month)


def arm_mae(cells: pd.DataFrame, model: str, component: str) -> float:
    """Trials-weighted MAE of one arm, for turning a paired diff into a
    percentage of the base's own error -- the unit predictions 2 and 3 are
    stated in."""
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
    base_mae = arm_mae(cells, base, component)
    r["base_mae"] = base_mae
    r["arm_mae"] = arm_mae(cells, arm, component)
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


# ─── prediction 2: the May/August split ────────────────────────────────────

def regime_split(cells: pd.DataFrame, arm: str, base: str,
                 component: str) -> dict:
    """The same paired comparison, restricted to each regime's cutoffs.

    A subset of cells fed to the same function, not a second statistic: the
    clustering and the win/loss counts mean exactly what they mean in the
    pooled table, which is what lets prediction 2's "3% at May, 1.5% at
    August" be read against prediction 3's pooled number without a
    translation.
    """
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


def score_prediction_2(split: dict) -> dict:
    """"Beats by >= 3% of MAE with |t| > 2.5 at May, gap <= 1.5% at August"."""
    may, aug = split.get("may"), split.get("august")
    out: dict = {"thresholds": {"may_gain_pct": MAY_GAIN_PCT, "may_t": MAY_T,
                                "august_max_gap_pct": AUGUST_MAX_GAP_PCT}}
    if may:
        t = may.get("clustered_by_player_t")
        out["may"] = {
            "pct_of_base_mae": may["pct_of_base_mae"],
            "clustered_by_player_t": t,
            "holds": bool(np.isfinite(may["pct_of_base_mae"])
                          and may["pct_of_base_mae"] <= MAY_GAIN_PCT
                          and t is not None and np.isfinite(t)
                          and abs(t) > MAY_T),
        }
    if aug:
        out["august"] = {
            "pct_of_base_mae": aug["pct_of_base_mae"],
            "clustered_by_player_t": aug.get("clustered_by_player_t"),
            # "the gap is <= 1.5%" is about the *size* of the difference in
            # either direction: a measurement arm 4% WORSE in August is not
            # the pre-registered "it converges", it is a different failure.
            "holds": bool(np.isfinite(aug["pct_of_base_mae"])
                          and abs(aug["pct_of_base_mae"]) <= AUGUST_MAX_GAP_PCT),
        }
    out["holds"] = bool(out.get("may", {}).get("holds")
                        and out.get("august", {}).get("holds"))
    return out


def score_prediction_3(pooled: dict) -> dict:
    """Pooled: HR/PA a real gain, K% a draw within +-0.0005."""
    hr = pooled.get("hr_rate") or {}
    k = pooled.get("k_rate") or {}
    out: dict = {"thresholds": {"pooled_t": POOLED_T, "k_draw_band": K_DRAW_BAND}}
    if hr:
        t = hr.get("clustered_by_player_t")
        out["hr_rate"] = {
            "diff": hr["diff"], "clustered_by_player_t": t,
            "holds": bool(hr["diff"] < 0 and t is not None
                          and np.isfinite(t) and t < POOLED_T),
        }
    if k:
        out["k_rate"] = {
            "diff": k["diff"],
            "holds": bool(abs(k["diff"]) <= K_DRAW_BAND),
        }
    out["holds"] = bool(out.get("hr_rate", {}).get("holds")
                        and out.get("k_rate", {}).get("holds"))
    return out


# ─── prediction 4: posterior coverage ──────────────────────────────────────

def coverage(cells: pd.DataFrame, model: str, component: str,
             months=None) -> dict:
    """How often the 80% posterior interval covers the realised rate.

    `pred_q10`/`pred_q90` are written into the cell parquet by the arms that
    carry an interval (`src.eval.bayes_arm`; the harness carries any `pred_*`
    column through). An arm without them scores `None` rather than 0 —
    `contact_additive` has no interval at all, which the pre-registration
    says in as many words, and reporting that as "0% coverage" would read as
    a catastrophic failure of an arm that simply does not make the claim.

    Unweighted over cells, not trials-weighted: the claim is about how often
    an interval is right, and every batter's interval is one claim.
    """
    g = cells[(cells["component"] == component) & (cells["model"] == model)]
    if months is not None:
        g = g[g["cutoff"].map(cutoff_month).isin(months)]
    if g.empty or not {"pred_q10", "pred_q90"} <= set(g.columns):
        return {"n": 0, "covered": None}
    g = g.dropna(subset=["pred_q10", "pred_q90"])
    if g.empty:
        return {"n": 0, "covered": None}
    inside = ((g["realized_rate"] >= g["pred_q10"])
              & (g["realized_rate"] <= g["pred_q90"]))
    below = (g["realized_rate"] < g["pred_q10"]).mean()
    return {
        "n": int(len(g)), "covered": float(inside.mean()),
        # Which tail the misses fall in says whether the interval is too
        # narrow (both tails) or the point estimate is biased (one tail),
        # and those are different repairs.
        "miss_low": float(below),
        "miss_high": float(1.0 - inside.mean() - below),
        "mean_width": float((g["pred_q90"] - g["pred_q10"]).mean()),
    }


def score_prediction_4(cells: pd.DataFrame, component: str) -> dict:
    lo, hi = COVERAGE_TARGET
    meas = coverage(cells, MEASUREMENT_ARM, component)
    meas_may = coverage(cells, MEASUREMENT_ARM, component, REGIMES["may"])
    walk_may = coverage(cells, WALK_ARM, component, REGIMES["may"])
    out = {
        "target": list(COVERAGE_TARGET),
        MEASUREMENT_ARM: meas,
        f"{MEASUREMENT_ARM}_may": meas_may,
        f"{WALK_ARM}_may": walk_may,
        CONTACT_ARM: coverage(cells, CONTACT_ARM, component),
    }
    out["holds"] = bool(
        meas["covered"] is not None and lo <= meas["covered"] <= hi
        and walk_may["covered"] is not None and walk_may["covered"] < lo)
    return out


# ─── predictions 1 and 5: the loadings ─────────────────────────────────────

def loadings_by_cutoff(fits: list[dict]) -> dict:
    """`{param: {cutoff: {mean, q2.5, q97.5}}}`, deduplicated across components.

    One measurement fit fills both components, so the same posterior is
    recorded on up to two fit records per cell. Keyed on the cutoff so those
    collapse to one entry — prediction 1 is a count over *cutoffs*, and
    counting a cutoff twice would make "at every cutoff" trivially easier to
    read as satisfied.
    """
    out: dict = {}
    for f in fits:
        params = f.get("measurement_params") or {}
        cutoff = f.get("cutoff")
        for name in LOADINGS:
            s = params.get(name)
            if not s:
                continue
            out.setdefault(name, {})[cutoff] = {
                "mean": s["mean"], "q2.5": s.get("q2.5"),
                "q97.5": s.get("q97.5"), "mph_per_sd": s.get("mph_per_sd"),
            }
    return out


def score_prediction_1(by_cutoff: dict) -> dict:
    """"The loadings exclude zero at every cutoff" — so the number that
    matters is `n_excluding_zero` against `n_cutoffs`, not an average."""
    out: dict = {}
    for name in LOADINGS:
        per = by_cutoff.get(name) or {}
        n = len(per)
        excl = sum(1 for s in per.values()
                   if s["q2.5"] is not None and (s["q2.5"] > 0 or s["q97.5"] < 0))
        means = np.asarray([s["mean"] for s in per.values()], dtype="float64")
        out[name] = {
            "n_cutoffs": n, "n_excluding_zero": excl,
            "holds": bool(n > 0 and excl == n),
            "mean_of_means": float(means.mean()) if n else float("nan"),
            "min": float(means.min()) if n else float("nan"),
            "max": float(means.max()) if n else float("nan"),
        }
    out["holds"] = bool(out and all(out[n]["holds"] for n in LOADINGS))
    return out


def score_prediction_5(fits: list[dict]) -> dict:
    """Vacuity: the latent scale is real and no two loadings are the same
    parameter wearing two names."""
    scales: dict = {}
    corrs = []
    for f in fits:
        params = f.get("measurement_params") or {}
        for comp, value in (params.get("sigma_ability") or {}).items():
            scales.setdefault(comp, []).append(float(value))
        m = params.get("max_abs_loading_corr")
        if m is not None:
            corrs.append({"cutoff": f.get("cutoff"), "max_abs_corr": float(m)})
    out: dict = {"latent_scale_floor": LATENT_SCALE_FLOOR,
                 "max_loading_corr": MAX_LOADING_CORR,
                 "sigma_ability": {}, "degenerate": []}
    for comp, values in sorted(scales.items()):
        v = np.asarray(values, dtype="float64")
        out["sigma_ability"][comp] = {
            "n": int(v.size), "mean": float(v.mean()),
            "min": float(v.min()), "max": float(v.max()),
            "n_below_floor": int((v < LATENT_SCALE_FLOOR).sum()),
        }
    out["degenerate"] = [c for c in corrs if c["max_abs_corr"] >= MAX_LOADING_CORR]
    out["holds"] = bool(
        out["sigma_ability"]
        and all(s["n_below_floor"] == 0 for s in out["sigma_ability"].values())
        and not out["degenerate"])
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


def render_loadings(by_cutoff: dict) -> str:
    if not by_cutoff:
        return "(no measurement fits recorded)"
    cutoffs = sorted({c for v in by_cutoff.values() for c in v})
    lines = []
    for name in LOADINGS:
        per = by_cutoff.get(name)
        if not per:
            continue
        lines.append(name)
        for cutoff in cutoffs:
            s = per.get(cutoff)
            if s is None:
                continue
            flag = "*" if (s["q2.5"] > 0 or s["q97.5"] < 0) else " "
            mph = (f"  ({s['mph_per_sd']:+.2f} mph/sd)"
                   if s.get("mph_per_sd") is not None else "")
            lines.append(f"  {cutoff}  {s['mean']:+.3f}  "
                         f"[{s['q2.5']:+.3f}, {s['q97.5']:+.3f}] {flag}{mph}")
    lines.append("(* = 95% interval excludes zero)")
    return "\n".join(lines)


def render_coverage(cov: dict) -> str:
    lo, hi = COVERAGE_TARGET
    lines = [f"target {lo:.0%}-{hi:.0%} of the 80% interval"]
    for name, s in cov.items():
        if not isinstance(s, dict) or "covered" not in s:
            continue
        if s["covered"] is None:
            lines.append(f"  {name:<32} (no interval)")
            continue
        lines.append(f"  {name:<32} {s['covered']:6.1%}  n={s['n']:<6d}  "
                     f"low {s['miss_low']:5.1%}  high {s['miss_high']:5.1%}  "
                     f"width {s['mean_width']:.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "data/eval/bas85")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = pd.read_parquet(args.in_dir / "cells_bayes.parquet")
    fits_path = args.in_dir / "bayes_fits.json"
    fits = json.loads(fits_path.read_text()) if fits_path.exists() else []

    by_cutoff = loadings_by_cutoff(fits)
    pooled: dict = {}
    payload: dict = {
        "scope": {
            "seasons": sorted(int(s) for s in cells["season"].unique()),
            "cutoffs": sorted(cells["cutoff"].unique().tolist()),
            "components": sorted(cells["component"].unique().tolist()),
            "arms": sorted(cells["model"].unique().tolist()),
        },
        "wall_time_s": {},
        "comparisons": {},
        "loadings_by_cutoff": by_cutoff,
        "prediction_1_loadings": score_prediction_1(by_cutoff),
        "prediction_5_vacuity": score_prediction_5(fits),
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
        for r in rows:
            if r["arm"] == MEASUREMENT_ARM and r["base"] == CONTACT_ARM:
                pooled[component] = r
        print(f"\n=== {component} ===")
        print(render(rows))

    split = regime_split(cells, MEASUREMENT_ARM, CONTACT_ARM, "hr_rate")
    payload["regime_split_hr_rate"] = split
    payload["prediction_2_early_season"] = score_prediction_2(split)
    payload["prediction_3_pooled"] = score_prediction_3(pooled)
    payload["prediction_4_coverage"] = {
        c: score_prediction_4(cells, c) for c in payload["scope"]["components"]}

    print("\n=== hr_rate: measurement vs contact_additive by regime ===")
    print(render([r for r in split.values()] or []))
    for name, s in split.items():
        print(f"  {name}: cutoffs {s['cutoffs']}")

    print("\n=== channel loadings by cutoff ===")
    print(render_loadings(by_cutoff))

    for component, cov in payload["prediction_4_coverage"].items():
        print(f"\n=== coverage ({component}) ===")
        print(render_coverage(cov))

    print("\n=== vacuity (prediction 5) ===")
    v = payload["prediction_5_vacuity"]
    for comp, s in v["sigma_ability"].items():
        print(f"  sigma_ability[{comp}] mean {s['mean']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]  "
              f"{s['n_below_floor']}/{s['n']} below {LATENT_SCALE_FLOOR}")
    print(f"  fits with |loading corr| >= {MAX_LOADING_CORR}: {len(v['degenerate'])}")

    print("\n=== pre-registered predictions ===")
    for key in ("prediction_1_loadings", "prediction_2_early_season",
                "prediction_3_pooled", "prediction_5_vacuity"):
        print(f"  {key:<32} {'HOLDS' if payload[key].get('holds') else 'FAILS'}")
    for component, cov in payload["prediction_4_coverage"].items():
        print(f"  prediction_4_coverage[{component}]".ljust(34)
              + ("HOLDS" if cov["holds"] else "FAILS"))

    out = args.out or (args.in_dir / "analysis_bas85.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
