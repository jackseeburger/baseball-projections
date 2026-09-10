"""BAS-92: persist the hierarchical posterior's *width* at the props cutoffs.

`docs/posterior-width.md` asks for one thing this repo has never written down:
the `bayes_walk` arm's per-player posterior standard deviation, at a handful
of 2026 cutoffs, for every batter a prop can be priced on. The point estimate
has been scored to death (docs/bayes-components.md); the width has only ever
lived inside a fit that was thrown away as soon as its mean was read.

What this script does, per (component, cutoff):

* fits the single-component ability-walk arm — `src.eval.bayes_arm`, the same
  `BayesArmConfig` the dense harness builds, with the dense harness's own
  training seasons (`bayes_prior_seasons`), `min_pa`, sampler scale and
  `include_pitcher=False`;
* projects to 2026 for every batter the fit saw, then again for the batters it
  did not (the population projection, `generate_projections(unseen=...)`), so
  a September call-up has a width rather than a hole;
* writes `batter, component, cutoff, mean, sd, lower, upper, unseen` to
  `data/eval/bas92/bayes_width_<cutoff>.parquet` and every fit's diagnostics
  to `data/eval/bas92/bayes_width_fits.json`.

The cutoffs `docs/posterior-width.md` names (07-15, 08-01, 08-15, 09-01) are
biweekly but they are **not** the dense harness's own biweekly grid, which
runs 04-15 + 14n through 08-05 with 05-01/07-01/08-01 pinned in. 2026-08-01
is the only one of the four that is in `BIWEEKLY_MMDD`, so it is the only
cutoff where the checkpoint comparison in `--check-checkpoint` has rows to
compare against.

Usage:
    python scripts/run_bayes_width.py                       # the four cutoffs
    python scripts/run_bayes_width.py --cutoffs 2026-08-01  # one
    python scripts/run_bayes_width.py --check-checkpoint    # reproduction only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_intraseason_backtest_dense import (
    CHEAP_SEASONS, bayes_prior_seasons,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bayes_width")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/eval/bas92"
PA_DIR = ROOT / "data/parquet/pa_outcomes"
CHECKPOINT = ROOT / "data/eval/dense_intraseason/cells_bayes.parquet"
CLOSES = ROOT / "data/market/prop_closes_2026.parquet"

# docs/posterior-width.md, "What gets built".
CUTOFFS = ("2026-07-15", "2026-08-01", "2026-08-15", "2026-09-01")
COMPONENTS = ("k_rate", "bb_rate", "hr_rate")
PREDICT_YEAR = 2026
# The arm docs/bayes-components.md calls `bayes_walk`: one component, ability
# on a random walk across seasons, no pitcher effect.
VARIANT = "ability_walk"
CHECKPOINT_ARM = "bayes_walk"


def out_path(cutoff: str, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"bayes_width_{cutoff}.parquet"


def props_batters(closes_path: Path = CLOSES) -> list[int]:
    """Every batter the props archive carries a priceable contract for.

    The dense harness's unseen set comes from *its* training frame; this
    exam's consumer is `src.market.props.price`, so the set that matters is
    the one the archive can ask about. A batter with no Bayes row at all
    falls back to the Beta (`props.BayesWidth`), and the whole point of the
    population projection is that this should not happen.
    """
    if not closes_path.exists():
        logger.warning("no prop closes at %s — unseen set is the fit's own "
                       "coverage only", closes_path)
        return []
    closes = pd.read_parquet(closes_path, columns=["prop_stat", "player_id"])
    ids = closes.loc[closes["prop_stat"] != "k", "player_id"].dropna()
    return sorted({int(p) for p in ids})


def unseen_frame(batters, fitted, predict_year: int = PREDICT_YEAR) -> pd.DataFrame:
    """[batter, age] for the batters a fit did not cover, ages from the register.

    `src.eval.bayes_arm.unseen_from_train` reads ages off the harness's
    training frame; there is no such frame here, so the ages come from the
    same Chadwick register `prepare_model_data` uses, on the same June-30
    seasonal-age convention. A batter the register misses gets NaN, which
    `_project_unseen` reads as "no age adjustment" rather than guessing.
    """
    from src.data.birthdates import (
        BIRTHDATES_PARQUET, load_birthdates, seasonal_age,
    )

    missing = sorted(set(int(b) for b in batters) - {int(b) for b in fitted})
    out = pd.DataFrame({"batter": missing})
    if not missing:
        out["age"] = pd.Series(dtype="float64")
        return out
    if BIRTHDATES_PARQUET.exists():
        bd = load_birthdates()
        out["age"] = seasonal_age(bd, out["batter"], predict_year)
    else:
        logger.warning("no birthdates parquet — unseen batters get no age term")
        out["age"] = np.nan
    return out


def fit_one(component: str, cutoff: str, seasons: tuple[int, ...],
            batters: list[int], sampler: str, draws: int, tune: int,
            chains: int, cores: int, pa_dir: Path,
            park_factors: bool = True) -> tuple[pd.DataFrame, dict]:
    """One `bayes_walk` fit, projected over seen and unseen batters.

    Returns the long per-batter frame and the fit record (diagnostics, scale,
    data summary) that goes into `bayes_width_fits.json`.
    """
    from src.eval.bayes_arm import BayesArmConfig, fit_bayes_rate
    from src.models.pa_components import get_component

    config = BayesArmConfig(
        pa_dir=pa_dir, component=component, seasons=seasons, min_pa=50,
        include_pitcher=False, max_batters=None, ability_walk=True,
        draws=draws, tune=tune, chains=chains, cores=cores,
        target_accept=0.9, nuts_sampler=sampler,
    )
    assert config.variant() == VARIANT, config.variant()

    t0 = time.time()
    if park_factors:
        fit = fit_bayes_rate(cutoff, PREDICT_YEAR, config)
    else:
        # The checkpoint control: the dense grid was fit before BAS-86's park
        # artifact existed, so `load_park_factors()` returned None and every
        # cell ran at a neutral offset. Reproducing those numbers means
        # reproducing that, which is a monkeypatch and not a config flag
        # because the loader takes its path from the module, not the arm.
        from src.models import pa_rate as pa_rate_mod

        real = pa_rate_mod.load_park_factors
        pa_rate_mod.load_park_factors = lambda *a, **k: None
        try:
            fit = fit_bayes_rate(cutoff, PREDICT_YEAR, config)
        finally:
            pa_rate_mod.load_park_factors = real
    elapsed = time.time() - t0

    seen = set(int(b) for b in fit.projections["batter"])
    unseen = unseen_frame(batters, seen)
    if len(unseen):
        logger.info("%s @ %s: %d batters projected from the fitted population",
                    component, cutoff, len(unseen))
        fit.project(unseen)

    comp = get_component(component)
    cols = comp.out_columns()
    proj = fit.projections
    frame = pd.DataFrame({
        "batter": proj["batter"].astype("int64"),
        "component": component,
        "cutoff": cutoff,
        "predict_year": PREDICT_YEAR,
        "mean": proj[cols["mean"]].astype(float),
        "sd": proj[cols["std"]].astype(float),
        "lower": proj[cols["lower"]].astype(float),
        "upper": proj[cols["upper"]].astype(float),
        "unseen": proj["unseen"].astype(bool),
    })
    record = {
        "component": component, "cutoff": cutoff, "variant": VARIANT,
        "arm": CHECKPOINT_ARM, "scale": config.label(),
        "park_factors": bool(park_factors),
        "seasons": [int(s) for s in seasons],
        "elapsed_s": round(elapsed, 1),
        "diagnostics": fit.diagnostics,
        "n_seen": int((~frame["unseen"]).sum()),
        "n_unseen": int(frame["unseen"].sum()),
        "sd_median": float(frame["sd"].median()),
        "sd_median_seen": float(frame.loc[~frame["unseen"], "sd"].median()),
        "sd_median_unseen": (float(frame.loc[frame["unseen"], "sd"].median())
                             if frame["unseen"].any() else None),
        **fit.data_summary,
    }
    # `sigma_step` is the ability walk's own scalar; docs/bayes-variants.md's
    # vacuity check reads it, and it is the parameter that makes this arm's
    # width different from the flat arm's.
    post = getattr(fit.trace, "posterior", None)
    if post is not None and "sigma_step" in post:
        vals = np.asarray(post["sigma_step"].values, dtype="float64").ravel()
        record["sigma_step"] = {"mean": float(vals.mean()),
                                "sd": float(vals.std()),
                                "q05": float(np.percentile(vals, 5)),
                                "q95": float(np.percentile(vals, 95))}
    return frame, record


def checkpoint_check(widths: pd.DataFrame,
                     checkpoint: Path = CHECKPOINT) -> list[dict]:
    """Max abs difference between these posterior means and the dense grid's.

    The dense checkpoint keeps one `predicted` per (component, season, cutoff,
    model, batter) for the batters that cell scored — the common-player set
    with at least `MIN_TRIALS` realized trials after the cutoff — so this is a
    comparison on the intersection, reported with its own n.
    """
    if not checkpoint.exists():
        logger.warning("no dense checkpoint at %s", checkpoint)
        return []
    cells = pd.read_parquet(checkpoint)
    cells = cells[(cells["model"] == CHECKPOINT_ARM) & (cells["season"] == PREDICT_YEAR)]
    rows = []
    for (component, cutoff), have in widths.groupby(["component", "cutoff"]):
        ref = cells[(cells["component"] == component) & (cells["cutoff"] == cutoff)]
        if ref.empty:
            rows.append({"component": component, "cutoff": cutoff,
                         "n_checkpoint": 0, "n_matched": 0,
                         "max_abs_diff": None, "mean_abs_diff": None,
                         "note": "cutoff not in the dense harness's biweekly grid"})
            continue
        j = ref[["batter", "predicted"]].merge(
            have[["batter", "mean"]], on="batter", how="inner")
        d = (j["predicted"].astype(float) - j["mean"].astype(float)).abs()
        rows.append({"component": component, "cutoff": cutoff,
                     "n_checkpoint": int(len(ref)), "n_matched": int(len(j)),
                     "max_abs_diff": float(d.max()) if len(d) else None,
                     "mean_abs_diff": float(d.mean()) if len(d) else None,
                     "note": None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoffs", nargs="+", default=list(CUTOFFS))
    ap.add_argument("--components", nargs="+", default=list(COMPONENTS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--pa-dir", type=Path, default=PA_DIR)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--sampler", default="numpyro")
    ap.add_argument("--no-park-factors", action="store_true",
                    help="fit at a neutral park offset, which is what the "
                         "dense checkpoint's own fits ran under (its grid "
                         "predates BAS-86's park artifact)")
    ap.add_argument("--check-only", action="store_true",
                    help="skip fitting; re-run the checkpoint comparison on "
                         "the parquet files already on disk")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits_path = args.out_dir / f"bayes_width_fits.json"

    available = set(y for y in CHEAP_SEASONS
                    if (args.pa_dir / f"pa_outcomes_{y}.parquet").exists())
    seasons = bayes_prior_seasons(PREDICT_YEAR, available)
    logger.info("training seasons: %s", list(seasons))

    batters = props_batters()
    logger.info("%d batters in the props archive", len(batters))

    fits: list[dict] = json.loads(fits_path.read_text()) if fits_path.exists() else []
    done = {(f["component"], f["cutoff"]) for f in fits}

    for cutoff in args.cutoffs:
        path = args.out_dir / f"bayes_width_{cutoff}.parquet"
        have = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if args.check_only:
            continue
        frames = [have] if not have.empty else []
        for component in args.components:
            if (component, cutoff) in done and not have.empty and \
                    component in set(have["component"]):
                logger.info("%s @ %s already done", component, cutoff)
                continue
            frame, record = fit_one(
                component, cutoff, seasons, batters, args.sampler,
                args.draws, args.tune, args.chains, args.cores, args.pa_dir,
                park_factors=not args.no_park_factors)
            frames = [f for f in frames if not f.empty]
            frames = [f[f["component"] != component] for f in frames] + [frame]
            fits = [f for f in fits
                    if not (f["component"] == component and f["cutoff"] == cutoff)]
            fits.append(record)
            out = pd.concat(frames, ignore_index=True)
            out.to_parquet(path, index=False)
            fits_path.write_text(json.dumps(fits, indent=1))
            logger.info("%s @ %s: %d rows -> %s (%.1fs)", component, cutoff,
                        len(frame), path, record["elapsed_s"])

    widths = []
    for cutoff in args.cutoffs:
        path = args.out_dir / f"bayes_width_{cutoff}.parquet"
        if path.exists():
            widths.append(pd.read_parquet(path))
    if not widths:
        print("nothing written")
        return
    widths = pd.concat(widths, ignore_index=True)
    check = checkpoint_check(widths)
    (args.out_dir / f"bayes_width_checkpoint_check.json").write_text(
        json.dumps(check, indent=1))
    print("\n== checkpoint reproduction (posterior means vs "
          f"{CHECKPOINT_ARM} in the dense grid) ==")
    print(pd.DataFrame(check).to_string(index=False))
    print("\n== widths written ==")
    print(widths.groupby(["cutoff", "component"])
          .agg(n=("batter", "size"), unseen=("unseen", "sum"),
               sd_median=("sd", "median"), mean_median=("mean", "median"))
          .to_string())


if __name__ == "__main__":
    main()
