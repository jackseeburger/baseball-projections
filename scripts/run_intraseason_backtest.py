"""Intra-season walk-forward: cut the season at a date, project the rest.

Answers the question the product actually asks — *given everything through
2026-07-01, how good are the rest-of-season rate projections?* — with the
same scoring path as the season-level harness (`src/eval/backtest.py`).

Arms at every cutoff:

    marcel_tuned      the live model (src/projections/ros.py): the same
                      estimator with per-component ballast, recency weights
                      and age curve fitted walk-forward on 2020-2024 and
                      frozen in src/eval/marcel_params.json
    marcel            stock Marcel — 5/4/3 with the partial current season as
                      the most recent year (Marcel weights by PA, so it scales
                      itself). The constants Tango published, unfitted.
    marcel_tuned_preseason / marcel_preseason
                      the same two with the partial season withheld — the
                      controls that isolate the value of in-season data
    season_to_date    this year's rate regressed to league with the
                      component's stabilization-point ballast
    previous_season   the last complete season, unregressed
    league_average    league rate through the cutoff
    bayes_preseason   our Bayesian components, fit through 2025 (a fixed
                      preseason file, so also a no-in-season-info arm)

Usage:
    python scripts/run_intraseason_backtest.py
    python scripts/run_intraseason_backtest.py --cutoffs 2026-07-01 --components k_rate
    python scripts/run_intraseason_backtest.py --markdown            # doc table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.pa_outcomes import load_pa_outcomes
from src.eval import backtest, calibration, score
from src.eval.backtest import frame_provider
from src.eval.baselines import INTRASEASON_BASELINES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMPONENTS = ["k_rate", "bb_rate", "hr_rate", "babip", "iso"]
DEFAULT_CUTOFFS = ["2026-05-01", "2026-07-01", "2026-08-01"]
# The live arm first: `marcel_tuned` is what src/projections/ros.py serves.
ARM_ORDER = ["marcel_tuned", "marcel", "season_to_date",
             "marcel_tuned_preseason", "marcel_preseason",
             "bayes_preseason", "previous_season", "league_average"]


def bayes_provider(component: str, projections_dir: Path):
    """Our preseason Bayesian projection for one component, or None if absent.

    These files were fit through 2025 and carry a `projection_year` column;
    `frame_provider` picks the predict year's row, so the same file is the
    no-in-season-information arm at every cutoff.
    """
    path = projections_dir / f"{component}_projections_2026.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return frame_provider(df, pred_col=f"projected_{component}")


def run(
    components: list[str],
    cutoffs: list[str],
    seasons: pd.DataFrame,
    pa: pd.DataFrame,
    projections_dir: Path,
    min_trials: int,
    predict_year: int,
) -> pd.DataFrame:
    tables = []
    for component in components:
        providers = dict(INTRASEASON_BASELINES)
        bayes = bayes_provider(component, projections_dir)
        if bayes is not None:
            providers["bayes_preseason"] = bayes
        for cutoff in cutoffs:
            results = backtest(
                component, cutoff_date=cutoff, predict_year=predict_year,
                seasons=seasons, pa_frame=pa, providers=providers,
                min_trials=min_trials,
            )
            tables.append(score(results).assign(cutoff=cutoff))
    return pd.concat(tables, ignore_index=True)


def to_markdown(scores: pd.DataFrame, components: list[str]) -> str:
    """One block per component: rows are arms, columns are cutoffs (MAE)."""
    lines = ["| Component | Arm | " + " | ".join(
        f"MAE {c[5:]}" for c in sorted(scores["cutoff"].unique())) + " |"]
    lines.append("|---|---|" + "---|" * scores["cutoff"].nunique())
    for component in components:
        g = scores[scores["component"] == component]
        if g.empty:
            continue
        wide = g.pivot(index="model", columns="cutoff", values="mae")
        order = [m for m in ARM_ORDER if m in wide.index]
        first = True
        for model in order:
            label = f"**{component}**" if first else ""
            first = False
            cells = []
            for cutoff in sorted(wide.columns):
                v = wide.loc[model, cutoff]
                best = v == wide[cutoff].min()
                cells.append(f"**{v:.4f}**" if best else f"{v:.4f}")
            lines.append(f"| {label} | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    parser.add_argument("--cutoffs", nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--predict-year", type=int, default=2026)
    parser.add_argument("--min-trials", type=int, default=100)
    parser.add_argument("--seasons", type=Path,
                        default=ROOT / "data/parquet/hitter_seasons_api.parquet")
    parser.add_argument("--pa-dir", type=Path, default=ROOT / "data/parquet",
                        help="cache dir for pa_outcomes_<year>.parquet (pulled from R2)")
    parser.add_argument("--projections-dir", type=Path,
                        default=ROOT / "data/projections")
    parser.add_argument("--markdown", action="store_true",
                        help="print the docs table instead of the wide scores")
    parser.add_argument("--calibration-model", default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="also write the scores to PATH as JSON, for "
                             "scripts/build_accuracy_json.py")
    args = parser.parse_args()

    seasons = pd.read_parquet(args.seasons)
    pa = load_pa_outcomes(args.predict_year, data_dir=args.pa_dir)

    scores = run(args.components, args.cutoffs, seasons, pa,
                 args.projections_dir, args.min_trials, args.predict_year)

    if args.markdown:
        print(to_markdown(scores, args.components))
    else:
        for cutoff in args.cutoffs:
            print(f"\n=== cutoff {cutoff} → rest of {args.predict_year} ===")
            g = scores[scores["cutoff"] == cutoff]
            print(g.drop(columns="cutoff").round(5).to_string(index=False))

    if args.calibration_model:
        results = backtest(
            args.components[0], cutoff_date=args.cutoffs[-1],
            predict_year=args.predict_year, seasons=seasons, pa_frame=pa,
            min_trials=args.min_trials,
        )
        print(f"\nCalibration ({args.calibration_model}, {args.components[0]}, "
              f"{args.cutoffs[-1]}):")
        print(calibration(results, args.calibration_model).round(4).to_string(index=False))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.csv_out, index=False)
        print(f"\nwrote {args.csv_out}")

    if args.json_out:
        import json
        from datetime import datetime, timezone

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predict_year": args.predict_year,
            "min_trials": args.min_trials,
            "components": args.components,
            "cutoffs": args.cutoffs,
            # Arms in the order the doc table uses, restricted to the ones this
            # run actually produced (bayes_preseason is absent without the
            # projection files).
            "arms": [a for a in ARM_ORDER if a in set(scores["model"])],
            "last_pa_date": str(pd.to_datetime(pa["game_date"]).max().date()),
            "scores": json.loads(scores.to_json(orient="records")),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
