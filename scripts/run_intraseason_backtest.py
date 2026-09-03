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
    bayes             opt-in (`--bayes`): the PA-level Bayesian K% model
                      **refit at the cutoff** on exactly the plate
                      appearances the baselines see. This is the arm that
                      makes the comparison a fair fight; `bayes_preseason`
                      never was one, because the harness itself measures
                      in-season information at 5-6% of K% MAE, the same order
                      as the deficit it was being charged with. Needs pymc and
                      MCMC time, hence opt-in — and it serves k_rate only.

Usage:
    python scripts/run_intraseason_backtest.py
    python scripts/run_intraseason_backtest.py --cutoffs 2026-07-01 --components k_rate
    python scripts/run_intraseason_backtest.py --markdown            # doc table
    # the fair fight, at whatever scale you can sample:
    python scripts/run_intraseason_backtest.py --components k_rate --bayes \\
        --bayes-draws 500 --bayes-tune 500 --bayes-max-batters 300
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
# `bayes` sits next to it because they are the two arms fed the same partial
# season; `bayes_preseason` sits down with the other withheld-season arms,
# where it belongs.
ARM_ORDER = ["marcel_tuned", "bayes", "marcel", "season_to_date",
             "marcel_tuned_preseason", "marcel_preseason",
             "bayes_preseason", "previous_season", "league_average"]
# The arm the paired test measures everything against by default: the one the
# site actually serves.
PAIRED_BASE = "marcel_tuned"


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
    bayes_config=None,
    paired_base: str = PAIRED_BASE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """(scores, paired, fits).

    `paired` is the within-player paired difference in absolute error against
    `paired_base` for every other arm at every cell — the same statistic
    `scripts/tune_marcel.py` reports, and the only honest way to say whether a
    0.0003 MAE gap means anything. `fits` records what each Bayesian refit
    actually was (scale, diagnostics, exposure), so a table can be labelled
    with its own scale.
    """
    from src.eval.tuning import paired_from_results

    tables, paired, fits = [], [], []
    for component in components:
        for cutoff in cutoffs:
            providers = dict(INTRASEASON_BASELINES)
            preseason = bayes_provider(component, projections_dir)
            if preseason is not None:
                providers["bayes_preseason"] = preseason
            if bayes_config is not None and component == "k_rate":
                # Imported here, not at module scope: it pulls in pymc, which
                # CI does not install.
                from src.eval.bayes_arm import bayes_k_rate_provider

                providers["bayes"] = bayes_k_rate_provider(
                    cutoff, predict_year, bayes_config,
                    on_fit=lambda fit: fits.append({
                        "cutoff": fit.cutoff_date,
                        "scale": fit.config.label(),
                        "diagnostics": fit.diagnostics,
                        **fit.data_summary,
                    }),
                )
            results = backtest(
                component, cutoff_date=cutoff, predict_year=predict_year,
                seasons=seasons, pa_frame=pa, providers=providers,
                min_trials=min_trials,
            )
            tables.append(score(results).assign(cutoff=cutoff))
            arms = set(results["model"])
            if paired_base in arms:
                for arm in sorted(arms - {paired_base}):
                    paired.append({
                        "component": component, "cutoff": cutoff, "arm": arm,
                        "base": paired_base,
                        **paired_from_results(results, arm, paired_base),
                    })
    return (pd.concat(tables, ignore_index=True),
            pd.DataFrame(paired), fits)


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
    parser.add_argument("--paired-base", default=PAIRED_BASE,
                        help="arm the paired per-hitter test measures against")
    # The refit Bayesian arm. Opt-in because it runs MCMC.
    bayes = parser.add_argument_group(
        "bayes arm (k_rate only)",
        "Refit the PA-level Bayesian K% model at each cutoff on exactly the "
        "plate appearances the baselines see. Report the scale you ran: a "
        "reduced fit is evidence about a reduced fit.")
    bayes.add_argument("--bayes", action="store_true",
                       help="add the refit `bayes` arm (needs pymc)")
    bayes.add_argument("--bayes-pa-dir", type=Path,
                       default=ROOT / "data/parquet/pa_outcomes",
                       help="directory of pa_outcomes_<year>.parquet for the fit")
    bayes.add_argument("--bayes-seasons", nargs="+", type=int, default=None,
                       help="seasons to fit on (default: every parquet found)")
    bayes.add_argument("--bayes-min-pa", type=int, default=50)
    bayes.add_argument("--bayes-max-batters", type=int, default=None,
                       help="reduced scale: fit only the N busiest batters")
    bayes.add_argument("--bayes-no-pitcher", action="store_true",
                       help="drop the opposing-pitcher term (ablation)")
    bayes.add_argument("--bayes-draws", type=int, default=500)
    bayes.add_argument("--bayes-tune", type=int, default=500)
    bayes.add_argument("--bayes-chains", type=int, default=2)
    bayes.add_argument("--bayes-sampler", default="numpyro",
                       help="'numpyro' where JAX is installed, else 'pymc'")
    bayes.add_argument("--bayes-target-accept", type=float, default=0.9)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None,
                        help="also write the scores to PATH as JSON, for "
                             "scripts/build_accuracy_json.py")
    args = parser.parse_args()

    seasons = pd.read_parquet(args.seasons)
    pa = load_pa_outcomes(args.predict_year, data_dir=args.pa_dir)

    bayes_config = None
    if args.bayes:
        from src.eval.bayes_arm import BayesArmConfig

        bayes_config = BayesArmConfig(
            pa_dir=args.bayes_pa_dir,
            seasons=tuple(args.bayes_seasons) if args.bayes_seasons else None,
            min_pa=args.bayes_min_pa,
            include_pitcher=not args.bayes_no_pitcher,
            max_batters=args.bayes_max_batters,
            draws=args.bayes_draws, tune=args.bayes_tune,
            chains=args.bayes_chains, cores=args.bayes_chains,
            target_accept=args.bayes_target_accept,
            nuts_sampler=args.bayes_sampler,
        )

    scores, paired, fits = run(
        args.components, args.cutoffs, seasons, pa,
        args.projections_dir, args.min_trials, args.predict_year,
        bayes_config=bayes_config, paired_base=args.paired_base,
    )

    if args.markdown:
        print(to_markdown(scores, args.components))
    else:
        for cutoff in args.cutoffs:
            print(f"\n=== cutoff {cutoff} → rest of {args.predict_year} ===")
            g = scores[scores["cutoff"] == cutoff]
            print(g.drop(columns="cutoff").round(5).to_string(index=False))

    if not paired.empty:
        print(f"\n=== paired per-hitter absolute error vs {args.paired_base} "
              f"(negative = the arm is better) ===")
        print(paired[["component", "cutoff", "arm", "n", "diff", "se", "t",
                      "win_rate"]].round(5).to_string(index=False))

    for fit in fits:
        print(f"\nbayes fit @ {fit['cutoff']}: {fit['scale']}")
        print(f"  data: {fit['n_pa']:,} PA in {fit['n_cells']:,} cells, "
              f"{fit['n_batters']} batters, {fit['n_pitchers']} pitchers, "
              f"seasons {fit['seasons']}, partial {fit['partial_pa']:,} PA "
              f"through {fit['last_game']}")
        d = fit["diagnostics"]
        print(f"  diagnostics: max r-hat {d['max_rhat']:.4f} ({d['max_rhat_var']}), "
              f"min ESS {d['min_ess_bulk']:.0f} ({d['min_ess_var']}), "
              f"divergences {d['divergences']}, "
              f"BFMI {', '.join(f'{b:.2f}' for b in d['bfmi'])}")

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
            "paired_base": args.paired_base,
            "paired": json.loads(paired.to_json(orient="records")) if not paired.empty else [],
            # What the `bayes` arm actually was in this run. The accuracy page
            # reads this to label the row with its own scale rather than
            # letting a reduced local fit pass for the full refit.
            "bayes_fits": fits,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
