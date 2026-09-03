"""Fit the pitcher Marcel's constants by walk-forward, freeze them, score the holdout.

The pitcher side of station A starts from constants nobody fit: 5/4/3 recency
and twice the published stabilization point as ballast, which is what station E
has been running since the starter term shipped. This script asks whether those
are the right constants for these five components, using the same procedure —
and the same `src/eval/tuning.py` — the hitter side used.

    tune    coordinate search per component on predict years 2020-2024
            (train <= Y-1, score Y), objective = mean trials-weighted MAE,
            age term constrained to a peak in 25-31 with slopes of opposite
            signs, inner-validation guard on top. Writes
            src/eval/marcel_pitcher_params.json.
    score   freeze those params and score the holdout the search never saw:
            season-level 2025 and 2026, plus the 2026 intra-season cutoffs
            05-01 / 07-01 / 08-01, with the paired per-pitcher difference
            against stock and its SE.

Everything here except `PITCHER_STOCK_PARAMS` and the season table is imported
from `scripts/tune_marcel.py`, so the two runs cannot diverge in method.

Ages are the Chadwick register's, as of June 30 — the same age of record the
hitter fit uses. The pitching endpoint's own `age` is the player's age *today*,
which is useless on a historical row, so it is never read.

Usage:
    python scripts/tune_marcel_pitchers.py                # tune, then score
    python scripts/tune_marcel_pitchers.py --skip-tune    # score the frozen file
    python scripts/tune_marcel_pitchers.py --markdown     # doc tables
    python scripts/tune_marcel_pitchers.py --inner-validation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.data.pa_outcomes import load_pa_outcomes
from src.eval import backtest, pitchers, score, tuning
from src.eval.backtest import COMPONENTS

from tune_marcel import (                                   # noqa: E402
    inner_check,
    league_mode_choice,
    league_picks,
    mae_markdown,
    paired_markdown,
    pooled,
    pooled_overall,
    tune,
)

DEFAULT_COMPONENTS = list(pitchers.COMPONENT_ORDER)
TUNE_YEARS = [2020, 2021, 2022, 2023, 2024]      # train <= Y-1, score Y
SEASON_HOLDOUT = [2025, 2026]                     # never seen by the search
CUTOFFS = ["2026-05-01", "2026-07-01", "2026-08-01"]
STOCK_ARM = "marcel_pitcher"
TUNED_ARM = "marcel_pitcher_tuned"
ARM_ORDER = [TUNED_ARM, "marcel_pitcher_tuned_noage", STOCK_ARM,
             "marcel_pitcher_tuned_preseason", "marcel_pitcher_preseason",
             "season_to_date", "previous_season", "league_average"]


def load_seasons(path: Path) -> pd.DataFrame:
    seasons = pitchers.normalize_pitcher_seasons(pd.read_parquet(path))
    return tuning.chadwick_ages(seasons, id_col="pitcher")


def arms(params: dict, intraseason: bool, restricted: dict | None = None) -> dict:
    """Every arm scored at one cell.

    `PITCHER_INTRASEASON_BASELINES` registers the tuned pair bound to *no*
    params, i.e. reading the committed file. They are dropped and re-added
    bound to `params`, so scoring a fit that has not been frozen yet really
    scores that fit.
    """
    base = dict(pitchers.PITCHER_INTRASEASON_BASELINES)
    for key in ("marcel_pitcher_tuned", "marcel_pitcher_tuned_preseason"):
        base.pop(key, None)
    if not intraseason:
        for key in ("season_to_date", "marcel_pitcher_preseason"):
            base.pop(key, None)
    out = {**base, TUNED_ARM: pitchers.pitcher_tuned_provider(params)}
    if intraseason:
        out["marcel_pitcher_tuned_preseason"] = (
            lambda train, spec, year, _p=params: pitchers.marcel_pitcher_tuned(
                pitchers.full_seasons(train), spec, year, params=_p))
    if restricted:
        out["marcel_pitcher_tuned_noage"] = pitchers.pitcher_tuned_provider(restricted)
    return out


def score_cells(seasons: pd.DataFrame, pa: pd.DataFrame, components: list[str],
                params: dict, min_trials: int, season_years: list[int],
                cutoffs: list[str], restricted: dict | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores, paired): MAE per arm, and the tuned-minus-stock paired
    difference, for every component x cell of the holdout."""
    rows, paired = [], []
    for component in components:
        cells = ([("season", y, None) for y in season_years]
                 + [("cutoff", 2026, c) for c in cutoffs])
        for kind, year, cutoff in cells:
            providers = arms(params, intraseason=(kind == "cutoff"),
                             restricted=restricted)
            if kind == "season":
                results = backtest(component, year - 1, year, seasons=seasons,
                                   providers=providers, min_trials=min_trials)
                label = str(year)
            else:
                results = backtest(component, cutoff_date=cutoff,
                                   predict_year=year, seasons=seasons,
                                   pa_frame=pa, providers=providers,
                                   min_trials=min_trials)
                label = cutoff
            rows.append(score(results).assign(cell=label, kind=kind))
            for arm in (TUNED_ARM, "marcel_pitcher_tuned_noage"):
                if arm not in providers:
                    continue
                d = tuning.paired_from_results(results, arm, STOCK_ARM)
                paired.append({"component": component, "cell": label,
                               "kind": kind, "arm": arm, **d})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(paired)


def inner_validation(seasons: pd.DataFrame, components: list[str],
                     years: list[int], min_trials: int, passes: int,
                     constrained: bool = True) -> pd.DataFrame:
    rows = []
    for component in components:
        r = inner_check(seasons, component, years, min_trials, passes,
                        constrained=constrained,
                        stock_params=pitchers.PITCHER_STOCK_PARAMS)
        rows.append({"component": component,
                     "fit_years": f"{r['fit_years'][0]}-{r['fit_years'][1]}",
                     "validate_years":
                         f"{r['validate_years'][0]}-{r['validate_years'][1]}",
                     **{k: r[k] for k in
                        ("stock_mae", "tuned_mae", "tuned_pct",
                         "tuned_ballast_weights_only_pct", "generalises")}})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    p.add_argument("--tune-years", nargs="+", type=int, default=TUNE_YEARS)
    p.add_argument("--season-holdout", nargs="+", type=int, default=SEASON_HOLDOUT)
    p.add_argument("--cutoffs", nargs="+", default=CUTOFFS)
    p.add_argument("--min-trials", type=int, default=100)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--skip-tune", action="store_true")
    p.add_argument("--skip-score", action="store_true")
    p.add_argument("--seasons", type=Path,
                   default=ROOT / "data/parquet/pitcher_seasons_api.parquet")
    p.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet")
    p.add_argument("--params-out", type=Path,
                   default=ROOT / "src/eval/marcel_pitcher_params.json")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--inner-validation", action="store_true")
    p.add_argument("--unconstrained-age", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    seasons = load_seasons(args.seasons)
    report = tuning.age_source_report(seasons, id_col="pitcher")
    print(f"ages: Chadwick register (June 30), coverage "
          f"{report['register_coverage']:.3%} of {report['rows']} pitcher-seasons")
    constrained = not args.unconstrained_age
    stock = pitchers.PITCHER_STOCK_PARAMS

    if args.inner_validation:
        print("\n=== inner validation (inside the tuning window) ===")
        print(inner_validation(seasons, args.components, args.tune_years,
                               args.min_trials, args.passes,
                               constrained=constrained)
              .round(6).to_string(index=False))
        return

    if not args.skip_tune:
        print("\n=== projected league rate: inner validation "
              f"(fit {args.tune_years[0]}-{args.tune_years[-3]}, "
              f"score {args.tune_years[-2]}-{args.tune_years[-1]}) ===")
        mode_scores, mode_levels = league_mode_choice(
            seasons, args.components, args.tune_years, args.min_trials,
            args.passes, constrained=constrained, stock_params=stock)
        print(mode_scores.drop(columns="params").round(6).to_string(index=False))
        picks = league_picks(mode_scores)
        print("\npicked: " + ", ".join(
            f"{c}={m}" + (f"@{d:g}" if m == "drift" else "")
            for c, (m, d) in sorted(picks.items())))

        params, restricted, summary = tune(
            seasons, args.components, args.tune_years, args.min_trials,
            args.passes, args.verbose, guard=not args.no_guard,
            constrained=constrained, picks=picks, stock_params=stock)
        path = pitchers.save_pitcher_params(
            params, args.params_out,
            generated="scripts/tune_marcel_pitchers.py",
            method=(f"coordinate search, walk-forward predict years "
                    f"{args.tune_years[0]}-{args.tune_years[-1]} "
                    f"(train <= Y-1), objective = mean trials-weighted MAE, "
                    f"min_trials={args.min_trials} batters faced, "
                    f"ages = Chadwick register (June 30)"
                    + ("; age term constrained to a peak in "
                       f"{int(tuning.AGE_PEAK_WINDOW[0])}-"
                       f"{int(tuning.AGE_PEAK_WINDOW[1])} with slopes of "
                       "opposite signs, signed so a pitcher's strikeout rate "
                       "peaks there and his walk, home-run and BABIP rates "
                       "trough there" if constrained else
                       "; age term unconstrained")
                    + ("; components whose fit does not beat stock on an "
                       "inner validation keep stock's constants"
                       if not args.no_guard else "")),
            stock=("5/4/3 recency, ballast = 2x the published stabilization "
                   "point (K 70 BF, BB 170, BB+HBP 170, HR 1300, BABIP 2000 "
                   "BIP), no age term — the constants src/sim/starters.py has "
                   "run since the station E starter term shipped"),
            grid=tuning.grid_summary(constrained),
            league_rate={
                "picked": {c: {"mode": m, "damp": d}
                           for c, (m, d) in sorted(picks.items())},
                "chosen_on": ("inner validation inside the tuning window, with "
                              "the league option pinned; the holdout is never "
                              "consulted"),
                "inner_validation":
                    mode_scores.drop(columns="params").to_dict("records"),
            },
            in_sample=summary,
            variants={"ballast_weights_only":
                      {k: v.to_dict() for k, v in restricted.items()}},
        )
        print(f"\nwrote {path}")

    params = pitchers.load_pitcher_params(args.params_out)
    restricted = (json.loads(args.params_out.read_text())
                  .get("variants", {}).get("ballast_weights_only")
                  if args.params_out.exists() else None)
    if args.skip_score:
        return

    pa = load_pa_outcomes(2026, data_dir=args.pa_dir)
    scores, paired = score_cells(seasons, pa, args.components, params,
                                 args.min_trials, args.season_holdout,
                                 args.cutoffs, restricted=restricted)
    cells = [str(y) for y in args.season_holdout] + list(args.cutoffs)

    print("\n=== out of sample: MAE by arm and cell ===")
    if args.markdown:
        print(mae_markdown(scores, args.components, cells, ARM_ORDER))
    else:
        print(scores.round(5).to_string(index=False))

    print(f"\n=== {TUNED_ARM} - {STOCK_ARM}: paired per-pitcher difference in "
          "absolute error (trials-weighted, negative = tuned better) ===")
    if args.markdown:
        print(paired_markdown(paired, args.components, cells, arm=TUNED_ARM))
    else:
        print(paired.round(6).to_string(index=False))

    pool = pooled(paired, scores.assign(
        model=scores["model"].replace({STOCK_ARM: "marcel"})))
    print("\n=== pooled across the holdout cells ===")
    print(pool.round(6).to_string(index=False))
    overall = pooled_overall(paired, scores.assign(
        model=scores["model"].replace({STOCK_ARM: "marcel"})))
    print("\n=== pooled overall, as a percent of stock's MAE ===")
    print(overall.round(4).to_string(index=False))
    print("(the five cells share pitchers and the three cutoffs are nested "
          "windows of the same season, so these pooled SEs are optimistic; "
          "the per-cell SEs above are the honest ones)")

    idx = overall.set_index("arm")
    for arm in sorted(paired["arm"].unique()):
        g = paired[paired["arm"] == arm]
        pl = pool[pool["arm"] == arm]
        better = int((g["diff"] < 0).sum())
        tied = int((g["diff"] == 0).sum())
        worse = pl[pl["pooled_diff"] > 0]["component"].tolist()
        # A component the guard sent back to stock ties every cell by
        # construction, so counting strict wins would call the frozen file a
        # failure for containing an honest "tuning found nothing here". The
        # test is: nothing got worse, and the total moved the right way.
        cleared = (not worse) and idx.loc[arm, "pooled_pct_of_stock_mae"] < 0
        print(f"\n{arm}: wins {better}/{len(g)} component x cell cells "
              f"({tied} tied, i.e. components the guard kept at stock); "
              f"pooled {idx.loc[arm, 'pooled_pct_of_stock_mae']:+.2f}% of "
              f"stock MAE (se {idx.loc[arm, 'se_pct']:.2f}); pooled difference "
              f"below zero in {int((pl['pooled_diff'] < 0).sum())}/{len(pl)} "
              "components")
        print(f"TUNING VERDICT ({arm}): "
              + ("no component worse than stock, and the pooled difference is "
                 "negative" if cleared else "at least one component is worse "
                 "than stock, or the pooled difference is not negative"))
        if worse:
            print(f"  components still worse than stock: {', '.join(worse)}")
    print("\nThe serving gate is not this table — it is "
          "scripts/run_pitcher_backtest.py, which asks whether the served arm "
          "beats every dumb baseline.")


if __name__ == "__main__":
    main()
