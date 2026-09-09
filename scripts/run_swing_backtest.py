"""Walk-forward test of swing decisions as covariates (BAS-75 / BAS-81, stage 2).

The pre-registration is docs/swing-decisions.md. K% and BB% are the two rate
components with no layer-1 measurement of their own; this asks whether the six
swing-decision aggregates carry anything about the rest of the season beyond
what the realized rates already carry, on the dense harness's own cells.

The arms at every (component, season, cutoff) cell:

    marcel_tuned          the live baseline, untouched
    swing_recal           a + b * baseline, fitted on earlier seasons only —
                          the control that absorbs a pure recalibration gain
    swing                 the same free fit plus the six standardized
                          swing-decision covariates
    swing_additive        the baseline's coefficient pinned at 1, swing
                          decisions added as a pure correction (the
                          deployable shape, docs/contact-quality.md §8)
    swing_additive_recal  the same pinned fit with the covariates removed —
                          an intercept-only correction, which is what
                          `swing_additive` has to beat to have said anything
    swing_shuffled        the permuted control (methods.md §5.4): every batter
                          keeps a real covariate vector attached to the wrong
                          batter

and the **incremental** pair, which answers "does this add to what is already
served" directly rather than by comparing two independent gates:

    contact_additive        the served engine (src/projections/ros.py)
    contact_swing_additive  the same, with swing decisions added alongside

    python scripts/run_swing_backtest.py --json-out data/eval/swing_decisions_stage2.json
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

from src.data.contact_quality import load_monthly as load_contact_monthly
from src.data.swing_decisions import load_monthly as load_swing_monthly
from src.eval.backtest import score
from src.eval.contact import (
    DEFAULT_BALLAST as CONTACT_BALLAST,
    DEFAULT_WINDOW_WEIGHTS as CONTACT_WEIGHTS,
    FEATURES as CONTACT_FEATURES,
    LIVE_CELL_SEASONS,
    LIVE_CUTOFF_MONTHS,
    build_hitter_cells,
    fit_contact,
)
from src.eval.contact import features_at_cutoff as contact_features_at_cutoff
from src.eval.swing import (
    FEATURES,
    SWING_BALLAST_GRID,
    SWING_WEIGHT_GRID,
    TUNED,
    features_at_cutoff,
    pooled_year_over_year,
    year_over_year,
)
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("swing")
ROOT = Path(__file__).resolve().parent.parent

COMPONENTS = ("k_rate", "bb_rate", "hr_rate", "babip", "iso")
# The two components this measurement is *for*. HR/PA, ISO and BABIP are
# pre-registered to move by less than 1% — they are the falsification check,
# not the target.
TARGETS = ("k_rate", "bb_rate")
# Hyperparameters are chosen on cells up to and including this season; every
# scored season is later. Coefficients are refit walk-forward regardless.
TUNE_THROUGH = 2021
VACUITY_SEASONS = tuple(range(2015, 2027))
VACUITY_MIN_PITCHES = 500


# --- features ----------------------------------------------------------------

def attach_z(cells: pd.DataFrame, swing_monthly: pd.DataFrame,
             contact_monthly: pd.DataFrame | None, weights, ballast
             ) -> pd.DataFrame:
    """Merge the standardized covariates onto the cells, one cutoff at a time.

    A batter with no pre-cutoff pitches gets z = 0 on every covariate, which
    makes the covariate arm identical to the recalibration arm for him — so
    the paired comparison runs on the baseline's own player set rather than on
    a quietly different population.
    """
    out = []
    for (season, cutoff), g in cells.groupby(["season", "cutoff"]):
        g = g.copy()
        z = features_at_cutoff(swing_monthly, cutoff, season, weights, ballast
                               ).set_index("player").reindex(g["player"].to_numpy())
        for f in FEATURES:
            g[f] = z[f].fillna(0.0).to_numpy()
        g["pitches_raw"] = z["pitches_raw"].fillna(0.0).to_numpy()
        if contact_monthly is not None:
            zc = contact_features_at_cutoff(
                contact_monthly, "hitter", cutoff, season,
                CONTACT_WEIGHTS, CONTACT_BALLAST
            ).set_index("player").reindex(g["player"].to_numpy())
            for f in CONTACT_FEATURES:
                g[f] = zc[f].fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def shuffle_z(cells: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute the swing covariates across batters within each cell.

    Every batter keeps a real covariate vector — same marginal distribution,
    same shrinkage, same standardization — attached to the wrong batter. An
    arm fitted and scored on this must land on the recalibration control; if
    it beats it, the pipeline is fitting the split rather than the covariate.
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


# --- the arms ----------------------------------------------------------------

ARMS = {
    # name: (features, fixed_base)
    "swing_recal": ((), False),
    "swing": (FEATURES, False),
    "swing_additive_recal": ((), True),
    "swing_additive": (FEATURES, True),
    "contact_additive": (CONTACT_FEATURES, True),
    "contact_swing_additive": (tuple(CONTACT_FEATURES) + tuple(FEATURES), True),
}
CLIP = (1e-4, 0.999)


def walk_forward(cells: pd.DataFrame, components, score_seasons,
                 shuffled: pd.DataFrame | None = None,
                 arms: dict | None = None) -> pd.DataFrame:
    """Predictions for every arm, refitting coefficients on prior seasons only.

    Returns the harness's long results frame, so `score()` and
    `paired_abs_error_diff` read it unchanged.
    """
    arms = arms or ARMS
    frames, coefs = [], []
    for component in components:
        g = cells[cells["component"] == component]
        for season in score_seasons:
            past, here = g[g["season"] < season], g[g["season"] == season]
            if past.empty or here.empty:
                continue
            base = here["base"].to_numpy(dtype="float64")
            preds = {"marcel_tuned": base}
            rows = {}
            for name, (feats, fixed) in arms.items():
                fit = fit_contact(past, component, features=feats,
                                  fixed_base=fixed)
                preds[name] = np.clip(fit.predict(base, here), *CLIP)
                coefs.append({"component": component, "season": season,
                              "arm": name, **fit.coef})
            if shuffled is not None:
                sg = shuffled[shuffled["component"] == component]
                sh = sg[sg["season"] == season]
                sfit = fit_contact(sg[sg["season"] < season], component,
                                   features=FEATURES)
                preds["swing_shuffled"] = np.clip(
                    sfit.predict(sh["base"].to_numpy(dtype="float64"), sh), *CLIP)
                rows["swing_shuffled"] = sh
            for name, p in preds.items():
                f = rows.get(name, here).copy()
                f["model"] = name
                f["predicted"] = p
                frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["coefs"] = coefs
    return out


# --- scoring -----------------------------------------------------------------

def paired(results: pd.DataFrame, arm: str, base: str, mask=None) -> dict:
    """Paired per-batter absolute-error difference, arm minus base.

    Pairing is on (season, cutoff, batter); the SE is clustered back on the
    batter, because one hitter appears at three cutoffs of five seasons and
    those rows are not fifteen independent observations.
    """
    cols = ["_key", "player", "predicted", "realized_rate", "trials"]
    r = results if mask is None else results[mask]
    r = r.assign(_key=r["season"].astype(str) + "|" + r["cutoff"] + "|"
                 + r["player"].astype(str))
    a, b = r[r["model"] == arm][cols], r[r["model"] == base][cols]
    if a.empty or b.empty:
        return {"n": 0, "n_clusters": 0, "diff": float("nan"),
                "se": float("nan"), "t": float("nan"),
                "win_rate": float("nan"), "base_mae": float("nan"),
                "pct": float("nan")}
    out = paired_abs_error_diff(a, b, id_col="_key", cluster_col="player")
    # The base arm's own MAE on this slice, so a difference can be read as a
    # fraction of what it is a difference *of* — a raw delta shrinks with the
    # slice's error scale and would otherwise look like a fading effect.
    out["base_mae"] = float(np.average(
        np.abs(b["predicted"] - b["realized_rate"]), weights=b["trials"]))
    out["pct"] = 100.0 * out["diff"] / out["base_mae"]
    return out


def score_table(results: pd.DataFrame) -> pd.DataFrame:
    r = results.copy()
    r["realized_successes"] = r["realized_rate"] * r["trials"]
    return score(r[["component", "model", "player", "predicted",
                    "realized_successes", "realized_rate", "trials"]])


def tune(cells: pd.DataFrame, swing_monthly: pd.DataFrame, tune_seasons):
    """Choose the recency weights and the shrinkage ballast on the tuning
    window only, by pooled trials-weighted MAE of the `swing` arm over K% and
    BB% — the components the measurement is for."""
    rows = []
    arms = {"swing": (FEATURES, False)}
    for weights in SWING_WEIGHT_GRID:
        for ballast in SWING_BALLAST_GRID:
            z = attach_z(cells, swing_monthly, None, weights, ballast)
            res = walk_forward(z, TARGETS, tune_seasons, arms=arms)
            g = res[res["model"] == "swing"]
            mae = float(np.average(np.abs(g["predicted"] - g["realized_rate"]),
                                   weights=g["trials"]))
            rows.append({"weights": weights, "ballast": ballast,
                         "mae": mae, "n": int(len(g))})
            logger.info("tune %s b=%.0f -> MAE %.6f", weights, ballast, mae)
    grid = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = grid.iloc[0]
    return tuple(best["weights"]), float(best["ballast"]), grid


def tercile_edges(x: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(x, 1 / 3)), float(np.quantile(x, 2 / 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", default=list(COMPONENTS))
    ap.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--swing", type=Path,
                    default=ROOT / "data/features/swing_decisions_monthly.parquet")
    ap.add_argument("--contact", type=Path,
                    default=ROOT / "data/features/contact_quality_monthly.parquet")
    ap.add_argument("--min-trials", type=int, default=100)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--weights", nargs=3, type=float, default=None)
    ap.add_argument("--ballast", type=float, default=None)
    ap.add_argument("--cells-in", type=Path, default=None)
    ap.add_argument("--cells-out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--vacuity-only", action="store_true",
                    help="run the pre-registered vacuity check and stop")
    args = ap.parse_args()

    components = tuple(args.components)
    swing_monthly = load_swing_monthly(args.swing)

    # --- the vacuity check, first, exactly as pre-registered -----------------
    yy = year_over_year(swing_monthly, VACUITY_SEASONS, VACUITY_MIN_PITCHES)
    pooled = pooled_year_over_year(swing_monthly, VACUITY_SEASONS[1:],
                                   VACUITY_MIN_PITCHES)
    print("\n=== vacuity check: year-over-year correlation, batters with "
          f">= {VACUITY_MIN_PITCHES} pitches in both years ===")
    print(yy.round(3).to_string(index=False))
    print(f"\npooled 2016-2026 (n={pooled['n']}): "
          + "  ".join(f"{f}={pooled[f]:.3f}" for f in FEATURES))
    ok = pooled["chase"] > 0.55 and pooled["zcontact"] > 0.50
    print("chase > 0.55 and zone-contact > 0.50: "
          + ("PASS" if ok else "FAIL — docs/swing-decisions.md says stop"))
    if args.vacuity_only or not ok:
        return

    contact_monthly = load_contact_monthly(args.contact)
    if args.cells_in and args.cells_in.exists():
        cells = pd.read_parquet(args.cells_in)
        cells = cells[cells["component"].isin(components)]
    else:
        seasons_table = pd.read_parquet(ROOT / "data/parquet/hitter_seasons_api.parquet")
        cells = build_hitter_cells(seasons_table, args.pa_dir, list(components),
                                   min_trials=args.min_trials)
        if args.cells_out:
            args.cells_out.parent.mkdir(parents=True, exist_ok=True)
            cells.to_parquet(args.cells_out, index=False)
    logger.info("%d cell rows over %s", len(cells),
                sorted(cells["season"].unique()))

    tune_seasons = [s for s in LIVE_CELL_SEASONS if s <= TUNE_THROUGH][2:]
    score_seasons = [s for s in LIVE_CELL_SEASONS if s > TUNE_THROUGH]

    weights, ballast, grid = TUNED["weights"], TUNED["ballast"], None
    if args.tune:
        weights, ballast, grid = tune(cells, swing_monthly, tune_seasons)
        logger.info("tuned: weights=%s ballast=%.0f", weights, ballast)
    if args.weights:
        weights = tuple(args.weights)
    if args.ballast:
        ballast = args.ballast

    z = attach_z(cells, swing_monthly, contact_monthly, weights, ballast)
    shuffled = shuffle_z(z, args.shuffle_seed)
    results = walk_forward(z, components, score_seasons, shuffled=shuffled)

    print(f"\n=== hitters: holdout seasons {score_seasons}, "
          f"cutoffs {list(LIVE_CUTOFF_MONTHS)} ===")
    print(f"weights={weights} ballast={ballast:.0f} "
          f"(chosen on {tune_seasons}, holdout untouched)")
    print("\n--- pooled scores (harness score(), trials-weighted) ---")
    print(score_table(results).round(6).to_string(index=False))

    print("\n--- paired per-batter absolute error, negative = arm is better ---")
    prows = []
    pairs = [
        ("swing", "marcel_tuned"),
        ("swing_recal", "marcel_tuned"),
        ("swing", "swing_recal"),
        ("swing_additive", "marcel_tuned"),
        ("swing_additive", "swing_additive_recal"),
        ("swing_shuffled", "marcel_tuned"),
        ("swing_shuffled", "swing_recal"),
        # the incremental question, asked directly
        ("contact_additive", "marcel_tuned"),
        ("contact_swing_additive", "marcel_tuned"),
        ("contact_swing_additive", "contact_additive"),
    ]
    for component in components:
        m = results["component"] == component
        for arm, base in pairs:
            prows.append({"component": component, "arm": arm, "base": base,
                          "scope": "all", **paired(results, arm, base, m)})
    ptab = pd.DataFrame(prows)
    print(ptab.round(6).to_string(index=False))

    # --- docs/contact-quality.md §6's split, on the two target components ----
    # Information survives at large samples and at the late cutoff; pure
    # variance reduction lives in the low-exposure tercile and at May 1 and is
    # gone by August. Percentages are of the *baseline's own MAE on that
    # slice*, so a shrinking error scale cannot masquerade as a fading effect.
    srows = []
    for component in TARGETS:
        m0 = results["component"] == component
        for axis in ("pitches_raw", "pre_trials"):
            lo, hi = tercile_edges(results.loc[m0, axis].to_numpy())
            for label, m in ((f"low_{axis}", m0 & (results[axis] <= lo)),
                             (f"mid_{axis}", m0 & (results[axis] > lo)
                              & (results[axis] <= hi)),
                             (f"high_{axis}", m0 & (results[axis] > hi))):
                srows.append({"component": component, "scope": label,
                              **paired(results, "swing_additive",
                                       "marcel_tuned", m)})
        for md in LIVE_CUTOFF_MONTHS:
            m = m0 & results["cutoff"].str.endswith(md)
            srows.append({"component": component, "scope": f"cutoff {md}",
                          **paired(results, "swing_additive", "marcel_tuned", m)})
        # terciles crossed with the cutoff — the split the ticket asks for
        lo, hi = tercile_edges(results.loc[m0, "pitches_raw"].to_numpy())
        for tl, tm in (("low", results["pitches_raw"] <= lo),
                       ("mid", (results["pitches_raw"] > lo)
                        & (results["pitches_raw"] <= hi)),
                       ("high", results["pitches_raw"] > hi)):
            for md in LIVE_CUTOFF_MONTHS:
                m = m0 & tm & results["cutoff"].str.endswith(md)
                srows.append({"component": component,
                              "scope": f"{tl} x {md}",
                              **paired(results, "swing_additive",
                                       "marcel_tuned", m)})
    stab = pd.DataFrame(srows)
    print("\n--- §6 split: exposure terciles and cutoff (swing_additive vs "
          "marcel_tuned) ---")
    print(stab.round(6).to_string(index=False))

    print("\n--- per-season paired (swing_additive vs marcel_tuned) ---")
    yrows = []
    for component in components:
        for season in score_seasons:
            m = ((results["component"] == component)
                 & (results["season"] == season))
            yrows.append({"component": component, "season": season,
                          **paired(results, "swing_additive", "marcel_tuned", m)})
    ytab = pd.DataFrame(yrows)
    print(ytab.round(6).to_string(index=False))

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticket": "BAS-81 (implements BAS-75)",
            "preregistration": "docs/swing-decisions.md",
            "side": "hitter", "components": list(components),
            "weights": list(weights), "ballast": ballast,
            "tune_seasons": tune_seasons, "score_seasons": score_seasons,
            "cutoffs": list(LIVE_CUTOFF_MONTHS), "min_trials": args.min_trials,
            "vacuity": {
                "min_pitches": VACUITY_MIN_PITCHES,
                "per_pair": json.loads(yy.to_json(orient="records")),
                "pooled": pooled,
                "passes": bool(ok),
            },
            "scores": json.loads(score_table(results).to_json(orient="records")),
            "paired": json.loads(ptab.to_json(orient="records")),
            "split": json.loads(stab.to_json(orient="records")),
            "per_season": json.loads(ytab.to_json(orient="records")),
            "coefficients": results.attrs.get("coefs", []),
            "grid": (json.loads(grid.assign(
                         weights=grid["weights"].astype(str))
                     .to_json(orient="records")) if grid is not None else []),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
