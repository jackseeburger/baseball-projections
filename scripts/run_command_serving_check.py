"""Score the BAS-88 serving pre-registration for the command level arm.

`docs/pitching-command-level.md`'s "Serving" section pre-registered four
predictions before anything was wired, and this is the runner that scores
them. It answers one question — *would `p_bb_rate` move from `stuff_additive`
to the joint additive fit?* — on the season the move would actually have been
made for, rather than on the pooled 2022-2026 panel BAS-87 cleared its gate
on.

    python scripts/run_command_serving_check.py
    python scripts/run_command_serving_check.py --document-delta

The distinction matters and is the whole reason this script exists separately
from `run_command_level_backtest.py`. That runner pools five holdout seasons;
this one fits **exactly what the serving path fits** — `fit_live_command` on
cell seasons strictly before the predict year, at the pinned hyperparameters
(`src.eval.command.SERVED_WEIGHTS` / `SERVED_BALLAST`, with the six stuff
controls left at stuff's own) — and scores it on the predict year's own
May/Jul/Aug cutoffs. A pooled effect that rests on five seasons is not the
same claim as an effect on the one season being served, and the serving
pre-registration asked for the second.

What each prediction needs:

  1. the served fit beats `stuff_additive` on BB/BF by >= 1.0% of MAE with the
     covariate-only share at |t| > 2.0. With the baseline pinned at 1 the
     stuff-only control *is* the served engine, so those are one number.
  2/3. the served document's own `p_bb_rate` column, and the team walk rates
     built from it — measured only under `--document-delta`, which builds the
     pitcher block twice (as served, and with the map flipped) and needs the
     Stats API and R2.
  4. the vacuity clause: `cmd_csw`'s coefficient in the served fit, with a
     pitcher-clustered standard error. If it straddles zero at |t| < 2 the
     serving is withheld whatever prediction 1 says.

Evidence lands in `data/eval/pitching_command_serving.json`, the mirror of
BAS-79's `pitching_stuff_serving.json`.
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
from src.eval.stuff import FEATURES as STUFF_FEATURES
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("command-serving")
ROOT = Path(__file__).resolve().parent.parent

PREDICT_YEAR = 2026
COMPONENTS = ("p_bb_rate", "p_k_rate", "p_hr_rate", "p_bbhbp_rate")
SERVED_COMPONENT = "p_bb_rate"
BASE_ARM = "marcel_pitcher_tuned"
SERVED_ARM = "stuff_additive"
COMMAND_ARM = "command_additive"

# The pre-registration's own bars, written down before any of this was run.
PREREG = {
    "incremental_min_pct": 1.0,     # over stuff_additive, BB/BF, prediction 1
    "incremental_min_t": 2.0,
    "vacuity_coef": "cmd_csw",      # prediction 4
    "vacuity_min_abs_t": 2.0,
    "doc_share_under": 0.010,       # prediction 2
    "doc_share_min": 0.80,
    "doc_league_shift_max": 0.002,
}


# --- the fit -----------------------------------------------------------------

def clustered_coefficients(cells: pd.DataFrame, features, cluster="player"
                           ) -> dict:
    """The additive fit's coefficients with a cluster-robust standard error.

    The fit itself is `src.eval.stuff.fit_stuff(..., fixed_base=True)` — this
    re-derives it only so the *variance* can be computed on the same design
    matrix. One pitcher appears at three cutoffs of several seasons and those
    rows are not independent, so the cluster is the pitcher, which is the unit
    every paired comparison in this repo already clusters on.
    """
    y = (cells["realized_rate"] - cells["base"]).to_numpy(dtype="float64")
    w = cells["trials"].to_numpy(dtype="float64")
    X = np.column_stack([np.ones(len(cells))]
                        + [cells[f].to_numpy(dtype="float64") for f in features])
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * sw, y * np.sqrt(w), rcond=None)

    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ (X * w[:, None]))
    scores = pd.DataFrame(X * (w * resid)[:, None])
    scores["_c"] = cells[cluster].to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, g in scores.groupby("_c"):
        s = g.drop(columns="_c").to_numpy().sum(axis=0)
        meat += np.outer(s, s)
    n_clusters = int(cells[cluster].nunique())
    n, k = X.shape
    adj = n_clusters / max(n_clusters - 1, 1) * (n - 1) / max(n - k, 1)
    cov = bread @ meat @ bread * adj
    se = np.sqrt(np.diag(cov))

    names = ["intercept"] + list(features)
    out = {"n_rows": int(n), "n_clusters": n_clusters, "coef": {}}
    for i, name in enumerate(names):
        out["coef"][name] = {
            "coef": float(beta[i]), "se": float(se[i]),
            "t": float(beta[i] / se[i]) if se[i] > 0 else float("nan")}
    # Whether the command *block* carries anything jointly, which is a
    # different question from whether any one of its coefficients does — the
    # three covariates overlap each other and the stuff controls heavily.
    idx = [names.index(f) for f in features if f not in STUFF_FEATURES]
    if idx:
        b = beta[idx]
        sub = cov[np.ix_(idx, idx)]
        out["block_wald_chi2"] = float(b @ np.linalg.pinv(sub) @ b)
        out["block_wald_df"] = len(idx)
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, arm: str, base: str) -> dict:
    """Paired per-pitcher absolute-error difference, arm minus base, as a
    percent of the base arm's own MAE on the same rows."""
    out = paired_abs_error_diff(a, b, id_col="_key", cluster_col="player")
    base_mae = float(np.average(np.abs(b["predicted"] - b["realized_rate"]),
                                weights=b["trials"]))
    return {"arm": arm, "base": base, "n": int(out["n"]),
            "n_pitchers": int(out["n_clusters"]), "base_mae": base_mae,
            "diff": float(out["diff"]),
            "pct": 100.0 * float(out["diff"]) / base_mae,
            "t": float(out["t"]), "win_rate": float(out["win_rate"])}


def score_one(z: pd.DataFrame, component: str, predict_year: int) -> dict:
    """Fit the served arm walk-forward and score it on `predict_year`."""
    g = z[z["component"] == component]
    past, here = g[g["season"] < predict_year], g[g["season"] == predict_year]
    if past.empty or here.empty:
        return {}
    both = command_eval.served_features()
    fit_cmd = stuff_eval.fit_stuff(past, component, features=both,
                                   fixed_base=True)
    fit_stf = stuff_eval.fit_stuff(past, component, features=STUFF_FEATURES,
                                   fixed_base=True)

    base = here["base"].to_numpy(dtype="float64")
    key = (here["season"].astype(str) + "|" + here["cutoff"] + "|"
           + here["player"].astype(str)).to_numpy()
    frames = {}
    for name, pred in ((COMMAND_ARM, fit_cmd.predict(base, here)),
                       (SERVED_ARM, fit_stf.predict(base, here)),
                       (BASE_ARM, base)):
        frames[name] = pd.DataFrame({
            "_key": key, "player": here["player"].to_numpy(),
            "predicted": np.clip(pred, 1e-4, 0.999),
            "realized_rate": here["realized_rate"].to_numpy(dtype="float64"),
            "trials": here["trials"].to_numpy(dtype="float64")})

    per_cutoff = []
    for cutoff in sorted(here["cutoff"].unique()):
        m = (here["cutoff"] == cutoff).to_numpy()
        per_cutoff.append({"cutoff": cutoff,
                           **paired(frames[COMMAND_ARM][m],
                                    frames[SERVED_ARM][m],
                                    COMMAND_ARM, SERVED_ARM)})
    return {
        "component": component,
        "fit_seasons": list(fit_cmd.seasons),
        "n_fit_rows": fit_cmd.n_rows,
        "n_scored_cells": int(len(here)),
        "scored_cutoffs": sorted(here["cutoff"].unique().tolist()),
        "coefficients": clustered_coefficients(past, both),
        "coefficients_stuff_only": clustered_coefficients(past,
                                                          STUFF_FEATURES),
        "paired": [
            paired(frames[COMMAND_ARM], frames[SERVED_ARM],
                   COMMAND_ARM, SERVED_ARM),
            paired(frames[COMMAND_ARM], frames[BASE_ARM],
                   COMMAND_ARM, BASE_ARM),
            paired(frames[SERVED_ARM], frames[BASE_ARM],
                   SERVED_ARM, BASE_ARM),
        ],
        "per_cutoff": per_cutoff,
    }


# --- predictions 2 and 3: the document itself --------------------------------

def document_delta(as_of: str, command_monthly: pd.DataFrame) -> dict:
    """Build the pitcher block twice and measure what the move would do.

    Once as served (`LIVE_ENGINE` untouched) and once with `p_bb_rate` flipped
    to the command engine, on the same as-of date and the same inputs, so the
    only difference between the two frames is the arm. Needs the Stats API and
    R2, which is why it is behind a flag.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_ros_projections as B

    from src.data.pa_outcomes import load_pa_outcomes
    from src.projections import pitcher_ros as pr

    B.ensure_training_pa_outcomes(B.training_pa_seasons(), B.PA_OUTCOMES_DIR)
    pa = load_pa_outcomes(B.SEASON, data_dir=ROOT / "data/parquet")
    names = B.load_names()
    stuff = B.load_stuff_monthly()
    seasons = pitcher_eval.normalize_pitcher_seasons(
        pd.read_parquet(B.PITCHER_SEASONS_PARQUET))
    inputs = B.pitcher_inputs(as_of)
    kwargs = dict(team_of=inputs["team_of"],
                  team_games_played=inputs["team_games_played"],
                  team_games_recent=inputs["team_games_recent"],
                  games_remaining=inputs["games_remaining"],
                  active_fraction=inputs["active_fraction"],
                  names=names, teams=inputs["teams"], season=B.SEASON,
                  stuff_monthly=stuff, stuff_pa_dir=B.PA_OUTCOMES_DIR)

    served = pr.build_pitcher_projections(as_of, seasons, pa, **kwargs)
    original = dict(pr.LIVE_ENGINE)
    pr.LIVE_ENGINE[SERVED_COMPONENT] = pr.COMMAND_ENGINE
    try:
        counter = pr.build_pitcher_projections(
            as_of, seasons, pa, command_monthly=command_monthly, **kwargs)
    finally:
        pr.LIVE_ENGINE.clear()
        pr.LIVE_ENGINE.update(original)

    a = served[["pitcher", "team_abbrev", "bb_rate_marcel", "bf_ros"]].rename(
        columns={"bb_rate_marcel": "bb_served"})
    b = counter[["pitcher", "bb_rate_marcel"]].rename(
        columns={"bb_rate_marcel": "bb_command"})
    m = a.merge(b, on="pitcher", how="inner")
    d = (m["bb_command"] - m["bb_served"]).abs()
    league_served = float(np.average(m["bb_served"], weights=m["bf_ros"]))
    league_command = float(np.average(m["bb_command"], weights=m["bf_ros"]))

    teams = m.dropna(subset=["team_abbrev"]).groupby("team_abbrev").apply(
        lambda g: pd.Series({
            "bf_ros": g["bf_ros"].sum(),
            "bb_served": np.average(g["bb_served"], weights=g["bf_ros"]),
            "bb_command": np.average(g["bb_command"], weights=g["bf_ros"])}),
        include_groups=False)
    teams["delta_pp"] = 100.0 * (teams["bb_command"] - teams["bb_served"])

    return {
        "as_of": as_of,
        "served_engine": served.attrs.get("pitcher_engine_used"),
        "counterfactual_engine": counter.attrs.get("pitcher_engine_used"),
        "stuff_features_through": served.attrs.get("stuff_features_through"),
        "command_features_through": counter.attrs.get(
            "command_features_through"),
        "n_pitchers": int(len(m)),
        "share_under_0.010": float((d < PREREG["doc_share_under"]).mean()),
        "median_abs_delta": float(d.median()),
        "p90_abs_delta": float(d.quantile(0.90)),
        "p99_abs_delta": float(d.quantile(0.99)),
        "max_abs_delta": float(d.max()),
        "league_bb_served": league_served,
        "league_bb_command": league_command,
        "league_shift": league_command - league_served,
        "max_abs_team_delta_pp": float(teams["delta_pp"].abs().max()),
        "mean_abs_team_delta_pp": float(teams["delta_pp"].abs().mean()),
        "team_walk_rates": json.loads(
            teams.round(6).sort_values("delta_pp").reset_index().to_json(
                orient="records")),
    }


# --- the verdict -------------------------------------------------------------

def score_predictions(bb: dict, doc: dict | None) -> list[dict]:
    """Predictions 1-4 of the Serving section, mechanically."""
    inc = next(p for p in bb["paired"]
               if p["arm"] == COMMAND_ARM and p["base"] == SERVED_ARM)
    vac = bb["coefficients"]["coef"][PREREG["vacuity_coef"]]
    out = [
        {"prediction": 1,
         "holds": bool(inc["pct"] < 0
                       and -inc["pct"] >= PREREG["incremental_min_pct"]
                       and abs(inc["t"]) > PREREG["incremental_min_t"]),
         "detail": {"pct": inc["pct"], "t": inc["t"],
                    "clears_size": bool(-inc["pct"]
                                        >= PREREG["incremental_min_pct"]),
                    "clears_significance": bool(
                        abs(inc["t"]) > PREREG["incremental_min_t"])}},
    ]
    if doc is not None:
        out.append({"prediction": 2,
                    "holds": bool(doc["share_under_0.010"]
                                  >= PREREG["doc_share_min"]
                                  and abs(doc["league_shift"])
                                  < PREREG["doc_league_shift_max"]),
                    "detail": {k: doc[k] for k in
                               ("share_under_0.010", "median_abs_delta",
                                "max_abs_delta", "league_shift")}})
        out.append({"prediction": 3,
                    "holds": None,
                    "detail": {
                        "max_abs_team_delta_pp": doc["max_abs_team_delta_pp"],
                        "mean_abs_team_delta_pp": doc[
                            "mean_abs_team_delta_pp"],
                        "note": "playoff-odds comparison not run; with the "
                                "serving withheld the board does not move at "
                                "all, so only the counterfactual team walk "
                                "rates are reported"}})
    # Prediction 4 is a gate, not a description: it fires, and when it fires
    # the serving is withheld whatever prediction 1 says.
    fires = abs(vac["t"]) < PREREG["vacuity_min_abs_t"]
    out.append({"prediction": 4, "name": "vacuity",
                "clause_fires": bool(fires),
                "holds": not fires,
                "detail": {"coefficient": PREREG["vacuity_coef"],
                           **vac,
                           "block_wald_chi2": bb["coefficients"].get(
                               "block_wald_chi2"),
                           "block_wald_df": bb["coefficients"].get(
                               "block_wald_df")}})
    out.append({"verdict": "withheld" if (fires or not out[0]["holds"])
                else "serve",
                "reason": ("the vacuity clause fires" if fires else "")
                          + ("; " if fires and not out[0]["holds"] else "")
                          + ("prediction 1 misses the effect floor"
                             if not out[0]["holds"] else "")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components", nargs="+", default=list(COMPONENTS))
    ap.add_argument("--predict-year", type=int, default=PREDICT_YEAR)
    ap.add_argument("--pa-dir", type=Path,
                    default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--command-monthly", type=Path,
                    default=ROOT / "data/features/pitching_command_monthly.parquet")
    ap.add_argument("--stuff-monthly", type=Path,
                    default=ROOT / "data/features/pitching_stuff_monthly.parquet")
    ap.add_argument("--min-trials", type=int, default=100)
    ap.add_argument("--cells-in", type=Path, default=None)
    ap.add_argument("--cells-out", type=Path, default=None)
    ap.add_argument("--document-delta", action="store_true",
                    help="also build the pitcher block twice to measure "
                         "predictions 2 and 3 (needs the Stats API and R2)")
    ap.add_argument("--as-of", default=None,
                    help="as-of date for --document-delta (default: today)")
    ap.add_argument("--json-out", type=Path,
                    default=ROOT / "data/eval/pitching_command_serving.json")
    args = ap.parse_args()

    components = tuple(args.components)
    command_monthly = load_command_monthly(args.command_monthly)
    stuff_monthly = load_stuff_monthly(args.stuff_monthly)

    if args.cells_in and args.cells_in.exists():
        cells = pd.read_parquet(args.cells_in)
        cells = cells[cells["component"].isin(components)]
    else:
        seasons_table = pitcher_eval.normalize_pitcher_seasons(
            pd.read_parquet(ROOT / "data/parquet/pitcher_seasons_api.parquet"))
        cells = stuff_eval.build_pitcher_cells(
            seasons_table, args.pa_dir, components,
            seasons=stuff_eval.LIVE_CELL_SEASONS,
            min_trials=args.min_trials)
        if args.cells_out:
            args.cells_out.parent.mkdir(parents=True, exist_ok=True)
            cells.to_parquet(args.cells_out, index=False)
    logger.info("%d cell rows over %s", len(cells),
                sorted(cells["season"].unique().tolist()))

    z = command_eval.attach_both_blocks(cells, command_monthly, stuff_monthly)
    scored = {c: score_one(z, c, args.predict_year) for c in components}
    scored = {c: r for c, r in scored.items() if r}

    bb = scored[SERVED_COMPONENT]
    print(f"\n=== the {args.predict_year} served fit: {COMMAND_ARM} on "
          f"{SERVED_COMPONENT} ===")
    print(f"fitted walk-forward on {bb['fit_seasons']} "
          f"({bb['n_fit_rows']} cells, "
          f"{bb['coefficients']['n_clusters']} pitchers), scored on "
          f"{bb['n_scored_cells']} cells at {bb['scored_cutoffs']}")
    print(f"weights={command_eval.SERVED_WEIGHTS} "
          f"ballast={command_eval.SERVED_BALLAST:.0f}, stuff controls at "
          f"{command_eval.SERVED_STUFF_WEIGHTS}/"
          f"{command_eval.SERVED_STUFF_BALLAST:.0f}")
    print("\n--- coefficients, SE clustered by pitcher ---")
    print(pd.DataFrame(bb["coefficients"]["coef"]).T.round(6).to_string())
    print(f"command block jointly: chi2 {bb['coefficients']['block_wald_chi2']:.1f}"
          f" on {bb['coefficients']['block_wald_df']} df")

    print("\n--- paired on the served season, negative = the arm is better ---")
    rows = [{**p, "component": c} for c, r in scored.items()
            for p in r["paired"]]
    print(pd.DataFrame(rows)[["component", "arm", "base", "n", "base_mae",
                              "diff", "pct", "t", "win_rate"]
                             ].round(6).to_string(index=False))
    print("\n--- BB/BF by cutoff ---")
    print(pd.DataFrame(bb["per_cutoff"])[["cutoff", "n", "pct", "t"]
                                         ].round(4).to_string(index=False))

    doc = None
    if args.document_delta:
        as_of = args.as_of or datetime.now(timezone.utc).date().isoformat()
        doc = document_delta(as_of, command_monthly)
        print(f"\n=== the document, {as_of}: what the move would change ===")
        print(f"{doc['n_pitchers']} pitchers; "
              f"{100 * doc['share_under_0.010']:.1f}% move < 0.010, median "
              f"{doc['median_abs_delta']:.5f}, max {doc['max_abs_delta']:.5f}")
        print(f"league BB/BF {doc['league_bb_served']:.5f} -> "
              f"{doc['league_bb_command']:.5f} "
              f"(shift {doc['league_shift']:+.5f})")
        print(f"team walk rates move at most "
              f"{doc['max_abs_team_delta_pp']:.3f} percentage points")

    print("\n=== scoring the serving pre-registration ===")
    predictions = score_predictions(bb, doc)
    for row in predictions:
        if "verdict" in row:
            print(f"  VERDICT: {row['verdict'].upper()} — {row['reason']}")
            continue
        if "clause_fires" in row:
            # The vacuity clause is a trapdoor, not a claim: it either fires
            # or it does not, and "did not fire" is the good outcome.
            state = "FIRES" if row["clause_fires"] else "does not fire"
        elif row["holds"] is None:
            state = "NOT SCORED"
        else:
            state = "HOLDS" if row["holds"] else "FAILS"
        print(f"  prediction {row['prediction']}: {state}  {row['detail']}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticket": "BAS-88",
        "predict_year": args.predict_year,
        "served_component": SERVED_COMPONENT,
        "weights": list(command_eval.SERVED_WEIGHTS),
        "ballast": command_eval.SERVED_BALLAST,
        "stuff_weights": list(command_eval.SERVED_STUFF_WEIGHTS),
        "stuff_ballast": command_eval.SERVED_STUFF_BALLAST,
        "level_features": list(command_eval.LEVEL_FEATURES),
        "stuff_controls": list(STUFF_FEATURES),
        "min_trials": args.min_trials,
        "prereg": PREREG,
        "components": scored,
        "document_delta": doc,
        "predictions": predictions,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
