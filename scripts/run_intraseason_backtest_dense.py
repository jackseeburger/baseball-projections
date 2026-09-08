"""BAS-63: densify the player-rate walk-forward backtest from 3 cutoffs to many.

`scripts/run_intraseason_backtest.py` scores three cutoffs (May 1, Jul 1, Aug
1) in one season, 2026 only. The headline "Bayesian K% is a dead heat with
tuned Marcel" number rests on n=126 hitters at a single cutoff — the thinnest
evidence on the scoreboard, next to station G's 249 weekly as-of dates over
ten seasons and the pitcher-workload harness's 44 biweekly cutoffs over five.

This script reuses the harness unchanged (`src.eval.backtest.backtest`,
`src.eval.intraseason`) and just calls it at many more (season, cutoff)
pairs, split into two sweeps at different cadences because the arms have very
different costs:

    cheap sweep   marcel_tuned, marcel, marcel_tuned_preseason,
                  marcel_preseason, season_to_date, previous_season,
                  league_average (plus bayes_preseason where a preseason file
                  exists, which today is 2026 only) — closed-form, so this
                  runs WEEKLY across every season 2019-2026 (2020 excluded;
                  a 60-game season starting July 23 has no May 1 cutoff, the
                  same convention `run_contact_backtest.py` uses).

    bayes sweep   adds the `bayes` arm — the PA-level K% model refit at the
                  cutoff (`src.eval.bayes_arm`) — on top of the same cheap
                  arms. One fit is an MCMC run (~75-90s here with NumPyro on
                  4 cores; the honest number for whatever machine runs this
                  is printed at the end of the run), so this sweep is
                  BIWEEKLY on a 4-season subset (2022, 2024, 2025, 2026) —
                  scoped deliberately per the task's cost-control guidance
                  rather than run to completion on all 7 seasons.

Pre-registered prediction (recorded before the densified numbers were read,
in the commit that added this script): partial pooling should help most when
samples are smallest, so the Bayesian arm's advantage over tuned Marcel
should be LARGEST in April/May and shrink roughly monotonically through
September. The three existing cutoffs show the opposite — Bayes worse in May
and July, only a tie by August. See docs/densified-intraseason-backtest.md
for the verdict.

Usage:
    # one-time data prep (writes gitignored data/parquet/pa_outcomes/*)
    python -c "from src.data.pa_outcomes_pipeline import build_pa_dataset; \\
               build_pa_dataset(years=[2019,2021,2022,2023,2024,2025])"
    python -c "from src.data.pa_outcomes import download; download(2026, 'data/parquet/pa_outcomes')"

    python scripts/run_intraseason_backtest_dense.py --stage cheap
    python scripts/run_intraseason_backtest_dense.py --stage bayes
    python scripts/run_intraseason_backtest_dense.py --stage analyze
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.eval.backtest import backtest, score as harness_score
from src.eval.baselines import INTRASEASON_BASELINES
from src.eval.tuning import paired_abs_error_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dense_backtest")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/eval/dense_intraseason"

# 2020 excluded everywhere in this repo: a 60-game season that started July
# 23 has no May 1 cutoff (see scripts/run_contact_backtest.py, and
# scripts/run_team_backtest.py's EXCLUDED_SEASONS).
EXCLUDED_SEASONS = (2020,)
CHEAP_SEASONS = (2019, 2021, 2022, 2023, 2024, 2025, 2026)
# The bayes sweep needs an MCMC fit per (season, cutoff); scoped to a subset
# per the task's cost-control note. 2022 for a season outside the 2024-2026
# window the existing "fair fight" doc already covers, so the densified
# result is not just re-slicing the same one season.
BAYES_SEASONS = (2022, 2024, 2025, 2026)

DEFAULT_COMPONENTS = ["k_rate", "bb_rate", "hr_rate", "babip", "iso"]
MIN_TRIALS = 100
PAIRED_BASE = "marcel_tuned"


def _mmdd_range(start: str, end: str, step_days: int) -> list[str]:
    cur = pd.Timestamp(f"2001-{start}")
    stop = pd.Timestamp(f"2001-{end}")
    out = []
    while cur <= stop:
        out.append(cur.strftime("%m-%d"))
        cur += pd.Timedelta(days=step_days)
    return out


# Weekly grid for the cheap sweep: every 7 days from two weeks after the
# earliest opening day in the window through Sept 2 (2026's local PA parquet
# ends Sept 1, so cutoffs past that would train on a "future" that does not
# exist for the in-progress season). Anchored on the legacy three cutoffs so
# the old numbers are exactly reproduced as a subset of the new ones.
WEEKLY_MMDD = sorted(set(_mmdd_range("04-08", "09-02", 7))
                     | {"05-01", "07-01", "08-01"})
# Biweekly grid for the bayes sweep: coarser, same anchors.
BIWEEKLY_MMDD = sorted(set(_mmdd_range("04-15", "08-15", 14))
                       | {"05-01", "07-01", "08-01"})


def load_pa_by_year(seasons: tuple[int, ...],
                    pa_dir: Path) -> dict[int, pd.DataFrame]:
    out = {}
    for year in seasons:
        path = pa_dir / f"pa_outcomes_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — build it first (see module docstring)")
        df = pd.read_parquet(path, columns=[
            "batter", "pitcher", "game_pk", "game_date", "game_year", "event",
            "is_k", "is_bb", "is_hbp", "is_hit", "is_hr", "is_single",
            "is_double", "is_triple",
        ])
        df["game_date"] = pd.to_datetime(df["game_date"])
        out[year] = df
    return out


def bayes_prior_seasons(year: int, available: set[int], max_priors: int = 2) -> tuple[int, ...]:
    """Up to `max_priors` immediately preceding non-excluded seasons with PA data."""
    priors = []
    y = year - 1
    while len(priors) < max_priors and y > 2000:
        if y not in EXCLUDED_SEASONS and y in available:
            priors.append(y)
        y -= 1
    return tuple(sorted({*priors, year}))


def preseason_bayes_provider(component: str, projections_dir: Path, year: int):
    path = projections_dir / f"{component}_projections_{year}.parquet"
    if not path.exists():
        return None
    from src.eval.backtest import frame_provider
    df = pd.read_parquet(path)
    return frame_provider(df, pred_col=f"projected_{component}")


# ─── cheap sweep ───

def run_cheap(seasons_table: pd.DataFrame, pa_by_year: dict[int, pd.DataFrame],
             components: list[str], seasons: tuple[int, ...],
             cutoffs_mmdd: list[str], projections_dir: Path,
             min_trials: int = MIN_TRIALS,
             checkpoint: Path | None = None) -> pd.DataFrame:
    done = set()
    frames = []
    if checkpoint is not None and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        frames.append(prev)
        done = set(zip(prev["component"], prev["season"], prev["cutoff"]))
        logger.info("resuming cheap sweep: %d cells already checkpointed", len(prev))

    for year in seasons:
        pa = pa_by_year[year]
        last_pa_date = pa["game_date"].max()
        for md in cutoffs_mmdd:
            cutoff = f"{year}-{md}"
            if pd.Timestamp(cutoff) >= last_pa_date:
                continue
            for component in components:
                if (component, year, cutoff) in done:
                    continue
                providers = dict(INTRASEASON_BASELINES)
                pre = preseason_bayes_provider(component, projections_dir, year)
                if pre is not None:
                    providers["bayes_preseason"] = pre
                try:
                    results = backtest(
                        component, cutoff_date=cutoff, predict_year=year,
                        seasons=seasons_table, pa_frame=pa, providers=providers,
                        min_trials=min_trials,
                    )
                except ValueError as e:
                    logger.warning("skip %s %s: %s", component, cutoff, e)
                    continue
                results = results.assign(season=year, cutoff=cutoff)
                frames.append(results)
                if checkpoint is not None:
                    pd.concat(frames, ignore_index=True).to_parquet(checkpoint, index=False)
            logger.info("cheap: %s done", cutoff)
    return pd.concat(frames, ignore_index=True)


# ─── bayes sweep ───

def run_bayes(seasons_table: pd.DataFrame, pa_by_year: dict[int, pd.DataFrame],
             seasons: tuple[int, ...], cutoffs_mmdd: list[str],
             min_trials: int = MIN_TRIALS, checkpoint: Path | None = None,
             fits_path: Path | None = None,
             draws: int = 500, tune: int = 500, chains: int = 2,
             sampler: str = "numpyro", include_pitcher: bool = False,
             pa_dir: Path = ROOT / "data/parquet/pa_outcomes") -> tuple[pd.DataFrame, list[dict]]:
    from src.eval.bayes_arm import BayesArmConfig, bayes_k_rate_provider

    done = set()
    frames, fits = [], []
    if checkpoint is not None and checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        frames.append(prev)
        done = set(zip(prev["season"], prev["cutoff"]))
        logger.info("resuming bayes sweep: %d cells already checkpointed", len(prev))
    if fits_path is not None and fits_path.exists():
        fits = json.loads(fits_path.read_text())

    available = set(pa_by_year)
    for year in seasons:
        pa = pa_by_year[year]
        last_pa_date = pa["game_date"].max()
        bayes_seasons = bayes_prior_seasons(year, available)
        for md in cutoffs_mmdd:
            cutoff = f"{year}-{md}"
            if (year, cutoff) in done:
                continue
            if pd.Timestamp(cutoff) >= last_pa_date:
                continue
            t0 = time.time()
            config = BayesArmConfig(
                pa_dir=pa_dir, seasons=bayes_seasons, min_pa=50,
                include_pitcher=include_pitcher, max_batters=None,
                draws=draws, tune=tune, chains=chains, cores=chains,
                target_accept=0.9, nuts_sampler=sampler,
            )
            providers = dict(INTRASEASON_BASELINES)
            fit_record = {}

            def on_fit(fit, _rec=fit_record):
                _rec.update({
                    "cutoff": fit.cutoff_date, "scale": fit.config.label(),
                    "diagnostics": fit.diagnostics, **fit.data_summary,
                })

            providers["bayes"] = bayes_k_rate_provider(cutoff, year, config, on_fit=on_fit)
            try:
                results = backtest(
                    "k_rate", cutoff_date=cutoff, predict_year=year,
                    seasons=seasons_table, pa_frame=pa, providers=providers,
                    min_trials=min_trials,
                )
            except ValueError as e:
                logger.warning("skip bayes %s: %s", cutoff, e)
                continue
            elapsed = time.time() - t0
            fit_record["elapsed_s"] = round(elapsed, 1)
            fit_record["bayes_seasons"] = list(bayes_seasons)
            fits.append(fit_record)
            results = results.assign(season=year, cutoff=cutoff)
            frames.append(results)
            out = pd.concat(frames, ignore_index=True)
            if checkpoint is not None:
                out.to_parquet(checkpoint, index=False)
            if fits_path is not None:
                fits_path.write_text(json.dumps(fits, indent=1))
            logger.info("bayes: %s done in %.1fs (%d cells so far)",
                       cutoff, elapsed, len(out))
    return pd.concat(frames, ignore_index=True), fits


# ─── analysis ───

def paired_by_cell(cells: pd.DataFrame, arm: str, base: str = PAIRED_BASE) -> pd.DataFrame:
    """Per (component, season, cutoff) paired diff, arm minus base — no
    clustering needed here since within one cell a player appears once."""
    rows = []
    for (component, season, cutoff), g in cells.groupby(["component", "season", "cutoff"]):
        a = g[g["model"] == arm]
        b = g[g["model"] == base]
        if a.empty or b.empty:
            continue
        cols = ["batter", "predicted", "realized_rate", "trials"]
        r = paired_abs_error_diff(a[cols], b[cols], id_col="batter")
        rows.append({"component": component, "season": season, "cutoff": cutoff,
                     "arm": arm, "base": base, **r})
    return pd.DataFrame(rows)


def common_player_sets(cells: pd.DataFrame, model: str = PAIRED_BASE) -> dict:
    """season -> set of batters scored at that season's LATEST cutoff.

    Rest-of-season trials only shrink as the cutoff advances, so a batter who
    clears `min_trials` at the last cutoff clears it at every earlier one
    too — this set is a valid fixed population at every cutoff in the season.
    """
    out = {}
    g = cells[cells["model"] == model]
    for season, sg in g.groupby("season"):
        last = max(sg["cutoff"].unique())
        out[season] = set(sg.loc[sg["cutoff"] == last, "batter"])
    return out


def _keyed(g: pd.DataFrame, model: str) -> pd.DataFrame:
    """Rows for one model with a (season, cutoff, batter)-unique id column
    and the raw batter id kept alongside as the cluster key. Pooling across
    seasons means the same real batter id recurs at the same calendar date in
    different years — a plain `batter` id_col would collide those rows (and
    `paired_abs_error_diff`'s indexed join would cross-multiply the
    duplicates), so pairing runs on the unique key and only the SE clusters
    on the real player.
    """
    m = g[g["model"] == model]
    key = m["season"].astype(str) + "|" + m["cutoff"] + "|" + m["batter"].astype(str)
    return m.assign(_key=key, _cluster=m["batter"])[
        ["_key", "_cluster", "predicted", "realized_rate", "trials"]]


def pooled_by_calendar_date(cells: pd.DataFrame, arm: str, base: str = PAIRED_BASE,
                            component: str = "k_rate",
                            common_sets: dict | None = None) -> pd.DataFrame:
    """Pool every season at the same MM-DD cutoff and pair, clustered by
    player (the same batter appears at several seasons' worth of this
    calendar date, and those rows are not independent draws).

    Returns one row per MM-DD with both the natural population and, when
    `common_sets` is given, the fixed common-player-set population.
    """
    g = cells[cells["component"] == component]
    rows = []
    for md in sorted(g["cutoff"].str[5:].unique()):
        gg = g[g["cutoff"].str.endswith(md)]
        a, b = _keyed(gg, arm), _keyed(gg, base)
        if a.empty or b.empty:
            continue
        r_nat = paired_abs_error_diff(a, b, id_col="_key", cluster_col="_cluster")
        n_seasons = int(gg["season"].nunique())
        rows.append({"cutoff_mmdd": md, "component": component, "arm": arm,
                     "base": base, "scope": "natural", "n_seasons": n_seasons,
                     **r_nat})
        if common_sets is not None:
            keep = set()
            for season in gg["season"].unique():
                keep |= common_sets.get(season, set())
            ac = a[a["_cluster"].isin(keep)]
            bc = b[b["_cluster"].isin(keep)]
            if not ac.empty and not bc.empty:
                r_com = paired_abs_error_diff(ac, bc, id_col="_key", cluster_col="_cluster")
                rows.append({"cutoff_mmdd": md, "component": component, "arm": arm,
                            "base": base, "scope": "common", "n_seasons": n_seasons,
                            **r_com})
    return pd.DataFrame(rows)


def overall_clustered(cells: pd.DataFrame, arm: str, base: str = PAIRED_BASE,
                      component: str = "k_rate") -> dict:
    """One pooled paired stat over every (season, cutoff) cell, clustered by
    player, against the naive unclustered version — the check the
    contact-quality work found inflates t by ~30% when skipped."""
    g = cells[(cells["component"] == component)]
    a = g[g["model"] == arm]
    b = g[g["model"] == base]
    key_a = a["season"].astype(str) + "|" + a["cutoff"] + "|" + a["batter"].astype(str)
    key_b = b["season"].astype(str) + "|" + b["cutoff"] + "|" + b["batter"].astype(str)
    a = a.assign(_key=key_a, _cluster=a["batter"])
    b = b.assign(_key=key_b)
    cols = ["_key", "predicted", "realized_rate", "trials", "_cluster"]
    clustered = paired_abs_error_diff(a[cols], b[["_key", "predicted", "realized_rate", "trials"]],
                                      id_col="_key", cluster_col="_cluster")
    unclustered = paired_abs_error_diff(a[cols], b[["_key", "predicted", "realized_rate", "trials"]],
                                        id_col="_key")
    return {"clustered": clustered, "unclustered": unclustered}


def build_analysis(cheap_path: Path, bayes_path: Path, out_json: Path) -> dict:
    cheap = pd.read_parquet(cheap_path) if cheap_path.exists() else pd.DataFrame()
    bayes = pd.read_parquet(bayes_path) if bayes_path.exists() else pd.DataFrame()

    payload: dict = {}

    if not cheap.empty:
        payload["cheap_scope"] = {
            "seasons": sorted(int(s) for s in cheap["season"].unique()),
            "n_cutoffs": int(cheap.groupby("season")["cutoff"].nunique().sum()),
            "components": sorted(cheap["component"].unique().tolist()),
        }
        payload["cheap_paired_marcel_vs_marcel_tuned"] = json.loads(
            paired_by_cell(cheap, "marcel", "marcel_tuned").to_json(orient="records"))
        payload["cheap_n_by_cutoff"] = json.loads(
            cheap[cheap["model"] == "marcel_tuned"]
            .groupby(["component", "season", "cutoff"]).size()
            .rename("n").reset_index().to_json(orient="records"))

    if not bayes.empty:
        payload["bayes_scope"] = {
            "seasons": sorted(int(s) for s in bayes["season"].unique()),
            "n_cutoffs": int(bayes.groupby("season")["cutoff"].nunique().sum()),
        }
        cs = common_player_sets(bayes, "marcel_tuned")
        payload["common_set_sizes"] = {int(k): len(v) for k, v in cs.items()}

        by_cell = paired_by_cell(bayes, "bayes", "marcel_tuned")
        payload["bayes_paired_by_cell"] = json.loads(by_cell.to_json(orient="records"))

        by_date = pooled_by_calendar_date(bayes, "bayes", "marcel_tuned",
                                          "k_rate", common_sets=cs)
        payload["bayes_gap_by_calendar_date"] = json.loads(by_date.to_json(orient="records"))

        overall = overall_clustered(bayes, "bayes", "marcel_tuned", "k_rate")
        payload["bayes_overall_clustered_vs_unclustered"] = overall

        # Also the stock-marcel-with-partial control, same pooling, for the
        # same reason the 3-cutoff doc reports it: the arm bayes is really
        # racing (marcel, not marcel_tuned, since marcel_tuned's own
        # constants were fitted on 2020-2024, which overlaps this densified
        # holdout — see the caveat in the doc).
        by_date_stock = pooled_by_calendar_date(bayes, "bayes", "marcel",
                                                "k_rate", common_sets=cs)
        payload["bayes_gap_by_calendar_date_vs_stock_marcel"] = json.loads(
            by_date_stock.to_json(orient="records"))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=1))
    return payload


# ─── cli ───

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("cheap", "bayes", "analyze", "all"), default="all")
    ap.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    ap.add_argument("--seasons-table", type=Path,
                    default=ROOT / "data/parquet/hitter_seasons_api.parquet")
    ap.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet/pa_outcomes")
    ap.add_argument("--projections-dir", type=Path, default=ROOT / "data/projections")
    ap.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--bayes-draws", type=int, default=500)
    ap.add_argument("--bayes-tune", type=int, default=500)
    ap.add_argument("--bayes-chains", type=int, default=2)
    ap.add_argument("--bayes-sampler", default="numpyro")
    ap.add_argument("--bayes-seasons", nargs="+", type=int, default=list(BAYES_SEASONS))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cheap_ckpt = args.out_dir / "cells_cheap.parquet"
    bayes_ckpt = args.out_dir / "cells_bayes.parquet"
    fits_path = args.out_dir / "bayes_fits.json"
    analysis_path = args.out_dir / "analysis.json"

    all_seasons = tuple(sorted(set(CHEAP_SEASONS) | set(args.bayes_seasons)))
    seasons_table = pd.read_parquet(args.seasons_table)

    if args.stage in ("cheap", "all"):
        pa_by_year = load_pa_by_year(CHEAP_SEASONS, args.pa_dir)
        cheap = run_cheap(seasons_table, pa_by_year, args.components, CHEAP_SEASONS,
                          WEEKLY_MMDD, args.projections_dir, args.min_trials,
                          checkpoint=cheap_ckpt)
        print(f"cheap sweep: {len(cheap)} rows -> {cheap_ckpt}")

    if args.stage in ("bayes", "all"):
        # bayes_prior_seasons only ever reaches into CHEAP_SEASONS (every
        # BAYES_SEASONS entry is a subset of CHEAP_SEASONS by construction),
        # so the cheap sweep's PA frames already cover every prior a fit
        # needs.
        pa_by_year = load_pa_by_year(CHEAP_SEASONS, args.pa_dir)
        bayes, fits = run_bayes(seasons_table, pa_by_year, tuple(args.bayes_seasons),
                                BIWEEKLY_MMDD, args.min_trials, checkpoint=bayes_ckpt,
                                fits_path=fits_path, draws=args.bayes_draws,
                                tune=args.bayes_tune, chains=args.bayes_chains,
                                sampler=args.bayes_sampler, pa_dir=args.pa_dir)
        print(f"bayes sweep: {len(bayes)} rows, {len(fits)} fits -> {bayes_ckpt}")

    if args.stage in ("analyze", "all"):
        payload = build_analysis(cheap_ckpt, bayes_ckpt, analysis_path)
        print(f"analysis -> {analysis_path}")
        print(json.dumps({k: v for k, v in payload.items()
                          if k in ("cheap_scope", "bayes_scope", "common_set_sizes",
                                   "bayes_overall_clustered_vs_unclustered")},
                         indent=1))


if __name__ == "__main__":
    main()
