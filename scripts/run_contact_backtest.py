"""Walk-forward test of Statcast contact quality as covariates (BAS-58, stage 1).

Three arms at every (component, season, cutoff) cell, scored with the harness's
own metrics on the harness's own common player set:

    marcel_tuned    the live baseline, untouched
    contact_recal   a + b * baseline, coefficients fitted on earlier seasons
                    only — the control that absorbs a pure recalibration gain
    contact         the same fit plus standardized exit-velocity /
                    launch-angle covariates

`contact` vs `marcel_tuned` is the gate. `contact` vs `contact_recal` is what
the covariate itself is worth. Coefficients for a scored season are fitted on
cells strictly before it; the two hyperparameters (the recency weights over
seasons and the shrinkage ballast on the covariates) are chosen on a tuning
window that ends before the scored seasons begin.

    # build the inputs once
    python scripts/build_contact_quality.py --download
    python -c "from src.data.pa_outcomes_pipeline import build_pa_dataset; \
               build_pa_dataset(data_dir='data/raw')"

    python scripts/run_contact_backtest.py --side hitter
    python scripts/run_contact_backtest.py --side pitcher --tune
    python scripts/run_contact_backtest.py --side hitter --json-out out.json
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

from src.data.contact_quality import load_monthly
from src.eval import pitchers as pitcher_eval
from src.eval.backtest import COMPONENTS, score
from src.eval.baselines import marcel_tuned
from src.eval.contact import (
    CONTACT_BALLAST_GRID,
    CONTACT_WEIGHT_GRID,
    TUNED,
    FEATURES,
    features_at_cutoff,
    fit_contact,
)
from src.eval.intraseason import (
    assert_split_clean,
    build_training_frame,
    partial_and_realized,
)
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("contact")
ROOT = Path(__file__).resolve().parent.parent

CUTOFF_MONTHS = ("05-01", "07-01", "08-01")
# 2017 is the first season with two full prior Statcast seasons *and* two prior
# seasons in the API season table. 2020 is excluded everywhere in this repo:
# a 60-game season that started on July 23 has no May 1 cutoff.
CELL_SEASONS = (2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
# Hyperparameters are chosen on cells up to and including this season; every
# season after it is holdout. Coefficients are refit walk-forward regardless.
TUNE_THROUGH = 2021

HITTER_COMPONENTS = ("k_rate", "bb_rate", "hr_rate", "babip", "iso")
PITCHER_COMPONENTS = ("p_k_rate", "p_bb_rate", "p_bbhbp_rate", "p_hr_rate",
                      "p_babip")
# Pre-registered: contact quality is a measurement of batted balls, so it can
# only speak to components a batted ball is part of. K% and BB% are the
# falsification checks — a "gain" there is a warning, not a win.
CONTACT_DEPENDENT = ("hr_rate", "babip", "iso", "p_hr_rate", "p_babip")

SIDE_CONFIG = {
    "hitter": {
        "components": HITTER_COMPONENTS,
        "seasons_path": "data/parquet/hitter_seasons_api.parquet",
        "id_col": "batter",
        "base_name": "marcel_tuned",
        "aggregate": partial_and_realized,
        "base": lambda train, spec, year: marcel_tuned(train, spec, year),
    },
    "pitcher": {
        "components": PITCHER_COMPONENTS,
        "seasons_path": "data/parquet/pitcher_seasons_api.parquet",
        "id_col": "pitcher",
        "base_name": "marcel_pitcher_tuned",
        "aggregate": pitcher_eval.partial_and_realized,
        "base": lambda train, spec, year: pitcher_eval.marcel_pitcher_tuned(
            train, spec, year),
    },
}


# --- cells -------------------------------------------------------------------

def build_cells(side: str, components, seasons_table: pd.DataFrame,
                pa_dir: Path, min_trials: int) -> pd.DataFrame:
    """One row per (component, season, cutoff, player) with everything the
    arms need: the baseline projection, the realized rest-of-season outcome
    and the pre-cutoff exposure.

    The split itself is the harness's — `partial_and_realized` either side of
    the date, `assert_split_clean` on both, the same `min_trials` filter and
    the same intersection with the baseline's coverage that `_run_split`
    applies. Nothing here re-implements a metric.
    """
    cfg = SIDE_CONFIG[side]
    id_col = cfg["id_col"]
    rows = []
    for season in CELL_SEASONS:
        pa = pd.read_parquet(pa_dir / f"pa_outcomes_{season}.parquet",
                             columns=["batter", "pitcher", "game_pk",
                                      "game_date", "game_year", "event",
                                      "is_k", "is_bb", "is_hbp", "is_hit",
                                      "is_hr", "is_single", "is_double",
                                      "is_triple"])
        pa["game_date"] = pd.to_datetime(pa["game_date"])
        for md in CUTOFF_MONTHS:
            cutoff = f"{season}-{md}"
            partial, realized = cfg["aggregate"](pa, cutoff, season)
            train = build_training_frame(seasons_table, partial, season, id_col)
            assert_split_clean(train, realized, cutoff, season)
            pre = partial.set_index(id_col)
            for component in components:
                spec = COMPONENTS[component]
                real = realized[realized[spec.trials] >= min_trials]
                if real.empty:
                    continue
                base = cfg["base"](train, spec, season)[[id_col, "predicted"]]
                base = base.dropna(subset=["predicted"])
                j = real[[id_col, spec.successes, spec.trials]].merge(
                    base, on=id_col, how="inner")
                if j.empty:
                    continue
                rows.append(pd.DataFrame({
                    "component": component, "side": side, "season": season,
                    "cutoff": cutoff, "player": j[id_col].to_numpy(),
                    "base": j["predicted"].to_numpy(dtype="float64"),
                    "realized_successes": j[spec.successes].to_numpy(dtype="float64"),
                    "trials": j[spec.trials].to_numpy(dtype="float64"),
                    "realized_rate": (j[spec.successes] / j[spec.trials]
                                      ).to_numpy(dtype="float64"),
                    "pre_trials": pre[spec.trials].reindex(
                        j[id_col].to_numpy()).fillna(0.0).to_numpy(dtype="float64"),
                }))
            logger.info("%s %s: %d component frames so far", side, cutoff, len(rows))
    return pd.concat(rows, ignore_index=True)


def attach_z(cells: pd.DataFrame, monthly: pd.DataFrame, weights, ballast
             ) -> pd.DataFrame:
    """Merge the standardized contact covariates onto the cells.

    Built once per (side, season, cutoff) — the features do not depend on the
    component. A player with no tracked contact before the cutoff gets z = 0,
    which makes the contact arm identical to the recalibration arm for him.
    """
    out = []
    for (side, season, cutoff), g in cells.groupby(["side", "season", "cutoff"]):
        z = features_at_cutoff(monthly, side, cutoff, season, weights, ballast)
        zi = z.set_index("player").reindex(g["player"].to_numpy())
        g = g.copy()
        for f in FEATURES:
            g[f] = zi[f].fillna(0.0).to_numpy()
        g["bbe_raw"] = zi["bbe_raw"].fillna(0.0).to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --- scoring -----------------------------------------------------------------

def shuffle_z(cells: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute the contact covariates across players within each cell.

    The permuted control methods.md §5 asks for. Every player keeps a real
    covariate vector — the same marginal distribution, the same shrinkage, the
    same standardization — attached to the wrong player. An arm fitted and
    scored on this must land at the recalibration control; if it beats it,
    something in the pipeline is fitting the split rather than the covariate.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _, g in cells.groupby(["side", "season", "cutoff"], sort=False):
        g = g.copy()
        perm = rng.permutation(len(g))
        for f in FEATURES:
            g[f] = g[f].to_numpy()[perm]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def walk_forward(cells: pd.DataFrame, components, score_seasons,
                 features=FEATURES, shuffled: pd.DataFrame | None = None,
                 hsgp: dict | None = None) -> pd.DataFrame:
    """Predictions for every arm, refitting coefficients on prior seasons only.

    Returns the harness's long results frame — [component, model, player,
    predicted, realized_successes, realized_rate, trials] plus the season,
    cutoff and exposure columns the splits are cut on — so `score()` and
    `paired_abs_error_diff` read it unchanged.

    `hsgp` maps a scored season to the same cells carrying an `xcon` column
    from a surface fitted on seasons before it — a separate frame per scored
    season, because the surface itself is refitted at every fold.
    """
    frames = []
    for component in components:
        g = cells[cells["component"] == component]
        for season in score_seasons:
            past = g[g["season"] < season]
            here = g[g["season"] == season]
            if past.empty or here.empty:
                continue
            recal = fit_contact(past, component, features=())
            full = fit_contact(past, component, features=features)
            add = fit_contact(past, component, features=features,
                              fixed_base=True)
            base = here["base"].to_numpy(dtype="float64")
            preds = {
                "marcel_tuned": base,
                "contact_recal": np.clip(recal.predict(base, None), 1e-4, 0.999),
                "contact": np.clip(full.predict(base, here), 1e-4, 0.999),
                "contact_additive": np.clip(add.predict(base, here), 1e-4, 0.999),
            }
            rows = {}
            if shuffled is not None:
                sg = shuffled[shuffled["component"] == component]
                sh = sg[sg["season"] == season]
                sfit = fit_contact(sg[sg["season"] < season], component,
                                   features=features)
                preds["contact_shuffled"] = np.clip(
                    sfit.predict(sh["base"].to_numpy(dtype="float64"), sh),
                    1e-4, 0.999)
                rows["contact_shuffled"] = sh
            if hsgp is not None and season in hsgp:
                hg = hsgp[season]
                hg = hg[hg["component"] == component]
                hp, hh = hg[hg["season"] < season], hg[hg["season"] == season]
                for name, feats in (("contact_hsgp", ("xcon",)),
                                    ("contact_both", (*features, "xcon"))):
                    hfit = fit_contact(hp, component, features=feats)
                    preds[name] = np.clip(
                        hfit.predict(hh["base"].to_numpy(dtype="float64"), hh),
                        1e-4, 0.999)
                    rows[name] = hh
            for name, p in preds.items():
                f = rows.get(name, here).copy()
                f["model"] = name
                f["predicted"] = p
                f["coef"] = json.dumps(full.coef if name == "contact" else
                                       recal.coef if name == "contact_recal" else {})
                frames.append(f)
    return pd.concat(frames, ignore_index=True)


def build_hsgp(cells: pd.DataFrame, side: str, pa_dir: Path, score_seasons,
               weights, ballast: float, cache: Path | None = None,
               **fit_kwargs) -> tuple[dict, list[dict]]:
    """One HSGP contact-quality surface per scored season, and the covariate
    it produces for every cell.

    The surface for season Y is fitted on batted balls from every cell season
    before Y and on nothing else, so the fold's own season is outside the fit
    as well as outside the regression. `xcon` for a cell is then the
    surface-weighted mean over that player's pre-cutoff batted balls, shrunk
    and standardized exactly as the stage-1 covariates are.
    """
    from src.eval.hsgp_contact import (
        BATTED_BALL_COLUMNS,
        Surface,
        batted_balls_from_pa,
        fit_surface,
        grid_cells,
        player_values,
        shrink_and_standardize,
        window_batted_balls,
    )

    id_col = SIDE_CONFIG[side]["id_col"]
    bb = pd.concat(
        [batted_balls_from_pa(pd.read_parquet(
            pa_dir / f"pa_outcomes_{s}.parquet", columns=BATTED_BALL_COLUMNS))
         for s in CELL_SEASONS], ignore_index=True)
    logger.info("%d tracked batted balls over %s", len(bb), list(CELL_SEASONS))

    out, fits = {}, []
    for season in score_seasons:
        fit_seasons = tuple(s for s in CELL_SEASONS if s < season)
        npz = (cache / f"surface_{season}.npz") if cache else None
        if npz is not None and npz.exists():
            blob = np.load(npz, allow_pickle=True)
            surface = Surface(values=blob["values"],
                              diagnostics=dict(blob["diagnostics"].item()),
                              seasons=fit_seasons)
            logger.info("surface %d: cached", season)
        else:
            train = bb[bb["game_year"].isin(fit_seasons)]
            surface = fit_surface(grid_cells(train), seasons=fit_seasons,
                                  **fit_kwargs)
            if npz is not None:
                cache.mkdir(parents=True, exist_ok=True)
                np.savez(npz, values=surface.values,
                         diagnostics=np.array(surface.diagnostics, dtype=object))
        logger.info("surface %d fitted on %s: %s", season, fit_seasons,
                    surface.diagnostics)
        fits.append({"season": season, "fit_seasons": list(fit_seasons),
                     **surface.diagnostics})

        frames = []
        for (cell_season, cutoff), g in cells.groupby(["season", "cutoff"]):
            w = {cell_season - i: float(x) for i, x in enumerate(weights)}
            window = window_batted_balls(bb, cutoff, cell_season, weights)
            vals = player_values(window, surface, id_col, w)
            z = shrink_and_standardize(vals, ballast).set_index("player")
            g = g.copy()
            g["xcon"] = z["xcon"].reindex(g["player"].to_numpy()).fillna(
                0.0).to_numpy()
            frames.append(g)
        out[season] = pd.concat(frames, ignore_index=True)
    return out, fits


def paired(results: pd.DataFrame, arm: str, base: str, mask=None) -> dict:
    """Paired per-player absolute-error difference, arm minus base.

    Pairing is on (season, cutoff, player) rather than player alone: the same
    hitter appears in five seasons and three cutoffs, and collapsing him would
    silently average different splits together. The standard error is then
    clustered *back* on the player, because those fifteen rows share a hitter
    and are not fifteen independent observations — without that the t reported
    here would be inflated by roughly the square root of the cutoffs.
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
    # Clustered on the player: one hitter is scored at three cutoffs of five
    # seasons, and those fifteen rows are not fifteen independent draws.
    out = paired_abs_error_diff(a, b, id_col="_key", cluster_col="player")
    # The base arm's own MAE on this same slice, so a difference can be read
    # as a fraction of what it is a difference *of* — a raw delta shrinks with
    # the slice's error scale and would otherwise look like a fading effect.
    out["base_mae"] = float(np.average(
        np.abs(b["predicted"] - b["realized_rate"]), weights=b["trials"]))
    out["pct"] = 100.0 * out["diff"] / out["base_mae"]
    return out


def score_table(results: pd.DataFrame) -> pd.DataFrame:
    """Harness `score()` over the pooled holdout, per component and arm."""
    return score(results[["component", "model", "player", "predicted",
                          "realized_successes", "realized_rate", "trials"]])


def tune(cells: pd.DataFrame, monthly: pd.DataFrame, components,
         tune_seasons) -> tuple[tuple, float, pd.DataFrame]:
    """Choose the recency weights and the shrinkage ballast on the tuning
    window only, by pooled trials-weighted MAE over the contact-dependent
    components."""
    rows = []
    targets = [c for c in components if c in CONTACT_DEPENDENT]
    for weights in CONTACT_WEIGHT_GRID:
        for ballast in CONTACT_BALLAST_GRID:
            z = attach_z(cells, monthly, weights, ballast)
            res = walk_forward(z, targets, tune_seasons)
            g = res[res["model"] == "contact"]
            mae = float(np.average(np.abs(g["predicted"] - g["realized_rate"]),
                                   weights=g["trials"]))
            rows.append({"weights": weights, "ballast": ballast, "mae": mae,
                         "n": len(g)})
            logger.info("tune %s b=%.0f -> MAE %.6f", weights, ballast, mae)
    grid = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = grid.iloc[0]
    return tuple(best["weights"]), float(best["ballast"]), grid


# --- main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=("hitter", "pitcher"), default="hitter")
    ap.add_argument("--components", nargs="+", default=None)
    ap.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--monthly", type=Path,
                    default=ROOT / "data/features/contact_quality_monthly.parquet")
    ap.add_argument("--min-trials", type=int, default=100)
    ap.add_argument("--tune", action="store_true",
                    help="re-derive the two hyperparameters on the tuning window")
    ap.add_argument("--weights", nargs=3, type=float, default=None)
    ap.add_argument("--ballast", type=float, default=None)
    ap.add_argument("--cells-out", type=Path, default=None)
    ap.add_argument("--cells-in", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--shuffle-control", action="store_true",
                    help="add the permuted-covariate arm (must land at recal)")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    g2 = ap.add_argument_group(
        "stage 2: the HSGP surface",
        "Replace the six hand-chosen aggregates with a nonparametric contact-"
        "quality surface over (EV, LA), refitted at every fold. Needs pymc.")
    g2.add_argument("--hsgp", action="store_true")
    g2.add_argument("--hsgp-draws", type=int, default=500)
    g2.add_argument("--hsgp-tune", type=int, default=500)
    g2.add_argument("--hsgp-chains", type=int, default=4)
    g2.add_argument("--hsgp-m", type=int, nargs=2, default=[12, 12])
    g2.add_argument("--hsgp-c", type=float, default=1.5)
    g2.add_argument("--hsgp-sampler", default="numpyro")
    g2.add_argument("--hsgp-cache", type=Path, default=None,
                    help="directory to cache fitted surfaces in")
    args = ap.parse_args()

    cfg = SIDE_CONFIG[args.side]
    components = tuple(args.components or cfg["components"])
    monthly = load_monthly(args.monthly)

    if args.cells_in and args.cells_in.exists():
        cells = pd.read_parquet(args.cells_in)
        cells = cells[cells["component"].isin(components)]
        logger.info("loaded %d cell rows from %s", len(cells), args.cells_in)
    else:
        seasons_table = pd.read_parquet(ROOT / cfg["seasons_path"])
        if args.side == "pitcher":
            seasons_table = pitcher_eval.normalize_pitcher_seasons(seasons_table)
        cells = build_cells(args.side, components, seasons_table, args.pa_dir,
                            args.min_trials)
        if args.cells_out:
            args.cells_out.parent.mkdir(parents=True, exist_ok=True)
            cells.to_parquet(args.cells_out, index=False)
            logger.info("wrote %s (%d rows)", args.cells_out, len(cells))

    tune_seasons = [s for s in CELL_SEASONS if s <= TUNE_THROUGH][2:]
    score_seasons = [s for s in CELL_SEASONS if s > TUNE_THROUGH]

    weights = TUNED[args.side]["weights"]
    ballast = TUNED[args.side]["ballast"]
    grid = None
    if args.tune:
        weights, ballast, grid = tune(cells, monthly, components, tune_seasons)
        logger.info("tuned: weights=%s ballast=%.0f", weights, ballast)
    if args.weights:
        weights = tuple(args.weights)
    if args.ballast:
        ballast = args.ballast

    z = attach_z(cells, monthly, weights, ballast)
    shuffled = shuffle_z(z, args.shuffle_seed) if args.shuffle_control else None
    hsgp, surface_fits = None, []
    if args.hsgp:
        hsgp, surface_fits = build_hsgp(
            z, args.side, args.pa_dir, score_seasons, weights, ballast,
            cache=args.hsgp_cache, m=tuple(args.hsgp_m), c=args.hsgp_c,
            draws=args.hsgp_draws, tune=args.hsgp_tune,
            chains=args.hsgp_chains, sampler=args.hsgp_sampler)
    results = walk_forward(z, components, score_seasons, shuffled=shuffled,
                           hsgp=hsgp)

    print(f"\n=== {args.side}: holdout seasons {score_seasons}, "
          f"cutoffs {CUTOFF_MONTHS} ===")
    print(f"weights={weights} ballast={ballast:.0f} "
          f"(chosen on {tune_seasons}, holdout untouched)")
    print("\n--- pooled scores (harness score(), trials-weighted) ---")
    print(score_table(results).round(6).to_string(index=False))

    print("\n--- paired per-player absolute error, negative = arm is better ---")
    prows = []
    for component in components:
        m = results["component"] == component
        arms = [("contact", "marcel_tuned"), ("contact_recal", "marcel_tuned"),
                ("contact", "contact_recal"),
                ("contact_additive", "marcel_tuned")]
        if args.shuffle_control:
            arms += [("contact_shuffled", "marcel_tuned"),
                     ("contact_shuffled", "contact_recal")]
        if args.hsgp:
            # The stage-2 question is not "does the surface beat the
            # baseline" — it is "does the surface beat the six hand-chosen
            # aggregates it was built to replace".
            arms += [("contact_hsgp", "marcel_tuned"),
                     ("contact_hsgp", "contact_recal"),
                     ("contact_hsgp", "contact"),
                     ("contact_both", "marcel_tuned"),
                     ("contact_both", "contact")]
        for arm, base in arms:
            prows.append({"component": component, "arm": arm, "base": base,
                          "scope": "all", **paired(results, arm, base, m)})
    # The variance-reduction test, on three axes. If the covariate only
    # denoises a small sample, the gain lives in the low-exposure half and at
    # the early cutoff and vanishes in the other one; if it carries
    # information the baseline does not have, it survives everywhere.
    #
    #   pre_trials  the player's own current-season exposure at the cutoff —
    #               what the baseline's in-season term is built from
    #   bbe_raw     the batted balls behind the covariate itself, over the
    #               whole three-season window
    #   cutoff      May 1 is the small-sample end of the season, Aug 1 the
    #               large-sample one
    for component in components:
        m0 = results["component"] == component
        for axis in ("pre_trials", "bbe_raw"):
            cut = float(results.loc[m0, axis].median())
            for half, m in ((f"low_{axis}", m0 & (results[axis] <= cut)),
                            (f"high_{axis}", m0 & (results[axis] > cut))):
                prows.append({"component": component, "arm": "contact",
                              "base": "marcel_tuned", "scope": half,
                              **paired(results, "contact", "marcel_tuned", m)})
        for md in CUTOFF_MONTHS:
            m = m0 & results["cutoff"].str.endswith(md)
            prows.append({"component": component, "arm": "contact",
                          "base": "marcel_tuned", "scope": f"cutoff {md}",
                          **paired(results, "contact", "marcel_tuned", m)})
    ptab = pd.DataFrame(prows)
    print(ptab.round(6).to_string(index=False))

    print("\n--- per-season paired (contact vs marcel_tuned) ---")
    srows = []
    for component in components:
        for season in score_seasons:
            m = ((results["component"] == component)
                 & (results["season"] == season))
            srows.append({"component": component, "season": season,
                          **paired(results, "contact", "marcel_tuned", m)})
    print(pd.DataFrame(srows).round(6).to_string(index=False))

    print("\n--- the tail of the holdout, where the baseline's own constants "
          "are also out of sample ---")
    trows = []
    for component in components:
        for label, seasons in (("2025-2026", [2025, 2026]), ("2026", [2026])):
            m = ((results["component"] == component)
                 & results["season"].isin(seasons))
            trows.append({"component": component, "scope": label,
                          **paired(results, "contact", "marcel_tuned", m)})
    print(pd.DataFrame(trows).round(6).to_string(index=False))

    print("\n--- fitted coefficients on the last holdout season ---")
    last = results[(results["model"] == "contact")
                   & (results["season"] == score_seasons[-1])]
    for component in components:
        c = last[last["component"] == component]
        if not c.empty:
            print(f"  {component}: {c['coef'].iloc[0]}")

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "side": args.side, "components": list(components),
            "weights": list(weights), "ballast": ballast,
            "tune_seasons": tune_seasons, "score_seasons": score_seasons,
            "cutoffs": list(CUTOFF_MONTHS), "min_trials": args.min_trials,
            "scores": json.loads(score_table(results).to_json(orient="records")),
            "paired": json.loads(ptab.to_json(orient="records")),
            "per_season": json.loads(pd.DataFrame(srows).to_json(orient="records")),
            "tail": json.loads(pd.DataFrame(trows).to_json(orient="records")),
            "grid": (json.loads(grid.assign(weights=grid["weights"].astype(str))
                                .to_json(orient="records")) if grid is not None
                     else []),
            "surface_fits": surface_fits,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
