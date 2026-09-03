"""Fit the PA-level K% model at a cutoff and report whether the fit is trustworthy.

The workflow this follows is the standard one — visualize, prior predictive
check, fit, assess convergence, posterior predictive check — reported as
numbers rather than plots so it can run headless and be pasted into a doc.

Two questions it answers, and they are different kinds of evidence:

  * **Is this fit usable?** R-hat, worst bulk ESS, divergences and BFMI, plus a
    prior predictive check (does the prior put mass on possible strikeout
    rates?) and a posterior predictive check (does the fitted model reproduce
    the K rate and its spread across hitters?).
  * **Is the opposing-pitcher term worth its cost?** `--ablation` fits the same
    data twice, with and without the term, and compares them by PSIS-LOO. Both
    models are built on the *same* cells — the pitcher-keyed partition — so
    the pointwise log-likelihoods are over the same observations and `az.compare`
    means something. LOO is an in-fit predictive estimate: it says the term
    earns its parameters, not that the arm wins the walk-forward. That second
    claim only comes from `scripts/run_intraseason_backtest.py --bayes`.

Needs pymc and arviz, and MCMC time. Nothing in CI runs it.

Usage:
    python scripts/validate_pa_k_rate.py --cutoff 2026-07-01 \\
        --seasons 2024 2025 2026 --max-batters 150 --draws 300 --tune 300
    python scripts/validate_pa_k_rate.py --cutoff 2026-07-01 --ablation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _rates_by_batter(replicate: np.ndarray, data: dict, min_pa: int = 100):
    """Roll a per-cell replicate up to per-batter rates at the real exposure.

    Cells are ~2 PA each once the pitcher is in the key, so a per-cell rate is
    almost always 0, 0.5 or 1 and tells you nothing. The quantity worth
    checking is a hitter's rate over the PA he actually took.
    """
    n = np.asarray(data["n_trials"], dtype="float64")
    b = np.asarray(data["batter_idx"])
    nb = int(b.max()) + 1
    pa = np.bincount(b, weights=n, minlength=nb)
    rate = np.bincount(b, weights=replicate, minlength=nb) / np.maximum(pa, 1)
    return rate[pa >= min_pa], pa


def summarize_prior_predictive(model, data, samples: int = 200, seed: int = 59) -> dict:
    """Where the prior puts a hitter's strikeout rate, before seeing any data.

    Reported at the batter level with real exposure (see `_rates_by_batter`).
    A prior that puts real mass on a 2% or a 70% strikeout hitter is telling
    you something before any Modal time is spent.
    """
    import pymc as pm

    with model:
        prior = pm.sample_prior_predictive(draws=samples, random_seed=seed)
    obs = np.asarray(prior.prior_predictive["obs_k"].values, dtype="float64")
    obs = obs.reshape(-1, obs.shape[-1])                  # (draws, cells)
    n = np.asarray(data["n_trials"], dtype="float64")

    per_draw_league = obs.sum(axis=1) / n.sum()
    by_batter = np.concatenate([_rates_by_batter(obs[i], data)[0]
                                for i in range(obs.shape[0])])
    q = np.percentile(by_batter, [1, 5, 25, 50, 75, 95, 99])
    return {
        "hitter_rate_quantiles": {f"p{p}": round(float(v), 4) for p, v in
                                  zip((1, 5, 25, 50, 75, 95, 99), q)},
        "league_rate_mean": round(float(per_draw_league.mean()), 4),
        "league_rate_p5_p95": [round(float(np.percentile(per_draw_league, 5)), 4),
                               round(float(np.percentile(per_draw_league, 95)), 4)],
        "implausible_share": round(float(np.mean(
            (by_batter < 0.05) | (by_batter > 0.55))), 4),
    }


def summarize_posterior_predictive(trace, model, data, seed: int = 59) -> dict:
    """Does the fitted model reproduce the league rate and the spread of hitters?

    Cell-level replicates are rolled up per batter with the real exposure, so
    the check is on the quantity the projection is about — a hitter's rate —
    not on a two-PA cell nobody cares about.
    """
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(
            trace, random_seed=seed, progressbar=False,
            extend_inferencedata=False)
    rep = np.asarray(ppc.posterior_predictive["obs_k"].values, dtype="float64")
    rep = rep.reshape(-1, rep.shape[-1])                 # (draws, cells)
    n = np.asarray(data["n_trials"], dtype="float64")
    k = np.asarray(data["k"], dtype="float64")

    obs_by_batter, pa_by_batter = _rates_by_batter(k, data)

    # A subsample of draws is plenty for a spread check and keeps this cheap.
    idx = np.linspace(0, rep.shape[0] - 1, min(100, rep.shape[0])).astype(int)
    rep_sd, rep_rate = [], []
    for i in idx:
        rates, _ = _rates_by_batter(rep[i], data)
        rep_sd.append(float(rates.std()))
        rep_rate.append(float(rep[i].sum() / n.sum()))
    return {
        "observed_league_rate": round(float(k.sum() / n.sum()), 4),
        "replicated_league_rate_mean": round(float(np.mean(rep_rate)), 4),
        "replicated_league_rate_p5_p95": [round(float(np.percentile(rep_rate, 5)), 4),
                                          round(float(np.percentile(rep_rate, 95)), 4)],
        "observed_sd_across_hitters": round(float(obs_by_batter.std()), 4),
        "replicated_sd_across_hitters_mean": round(float(np.mean(rep_sd)), 4),
        "replicated_sd_p5_p95": [round(float(np.percentile(rep_sd, 5)), 4),
                                 round(float(np.percentile(rep_sd, 95)), 4)],
        "n_hitters_over_100_pa": int(len(obs_by_batter)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", default="2026-07-01")
    parser.add_argument("--pa-dir", type=Path,
                        default=ROOT / "data/parquet/pa_outcomes")
    parser.add_argument("--seasons", nargs="+", type=int, default=None)
    parser.add_argument("--max-batters", type=int, default=None,
                        help="reduced scale: fit only the N busiest batters")
    parser.add_argument("--min-pa", type=int, default=50)
    parser.add_argument("--no-pitcher", action="store_true")
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--sampler", default="numpyro")
    parser.add_argument("--ablation", action="store_true",
                        help="also fit without the pitcher term and LOO-compare")
    parser.add_argument("--no-ppc", action="store_true",
                        help="skip the posterior predictive check (memory)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import arviz as az
    import pymc as pm

    from src.eval.bayes_arm import BayesArmConfig, _load_cut_pa
    from src.models.pa_k_rate import (
        build_model, load_park_factors, model_diagnostics, prepare_model_data,
        sample_model,
    )

    config = BayesArmConfig(
        pa_dir=args.pa_dir,
        seasons=tuple(args.seasons) if args.seasons else None,
        min_pa=args.min_pa,
        include_pitcher=not args.no_pitcher,
        max_batters=args.max_batters,
        draws=args.draws, tune=args.tune, chains=args.chains,
        cores=args.chains, target_accept=args.target_accept,
        nuts_sampler=args.sampler,
    )
    print(f"scale: {config.label()}  cutoff {args.cutoff}")

    pa = _load_cut_pa(config, args.cutoff)
    # The ablation needs both models over the same observations, so the cells
    # are always keyed by pitcher when it is available; the term is what gets
    # switched off, not the partition.
    data = prepare_model_data(
        pa, load_park_factors(), min_pa=args.min_pa,
        cutoff_date=args.cutoff, include_pitcher=not args.no_pitcher,
    )
    report = {
        "cutoff": args.cutoff,
        "scale": config.label(),
        "data": {
            "n_pa": int(data["n_pa"]), "n_cells": int(data["n_obs"]),
            "n_batters": int(data["n_batters"]),
            "n_pitchers": int(data["n_pitchers"]),
            "seasons": [int(s) for s in data["seasons"]],
            "pa_per_cell": round(data["n_pa"] / max(data["n_obs"], 1), 2),
        },
    }
    print(json.dumps(report["data"], indent=1))

    model = build_model(data)
    report["prior_predictive"] = summarize_prior_predictive(model, data)
    print("prior predictive (implied K rate):",
          json.dumps(report["prior_predictive"], indent=1))

    t0 = time.time()
    trace = sample_model(model, **{**config.sampler_kwargs(),
                                   "idata_kwargs": {"log_likelihood": args.ablation}})
    report["seconds"] = round(time.time() - t0, 1)
    report["diagnostics"] = model_diagnostics(trace)
    print("diagnostics:", json.dumps(report["diagnostics"], indent=1))

    summary = az.summary(trace, var_names=[
        v for v in ("league_init", "mu_ability", "sigma_ability", "sigma_pitcher",
                    "beta_hand", "beta_age", "beta_age2")
        if v in trace.posterior])
    print("\nscalar parameters:")
    print(summary.to_string())
    report["scalars"] = json.loads(summary.to_json(orient="index"))

    if not args.no_ppc:
        report["posterior_predictive"] = summarize_posterior_predictive(
            trace, model, data)
        print("posterior predictive:",
              json.dumps(report["posterior_predictive"], indent=1))

    if args.ablation:
        if not data["include_pitcher"]:
            raise SystemExit("--ablation needs the pitcher term on")
        print("\n--- ablation: the same cells, without the pitcher term ---")
        flat = {**data, "include_pitcher": False}
        model_flat = build_model(flat)
        trace_flat = sample_model(model_flat, **{
            **config.sampler_kwargs(), "idata_kwargs": {"log_likelihood": True}})
        report["ablation_diagnostics"] = model_diagnostics(trace_flat)
        print("no-pitcher diagnostics:",
              json.dumps(report["ablation_diagnostics"], indent=1))
        comparison = az.compare({"with_pitcher": trace, "no_pitcher": trace_flat})
        print("\nPSIS-LOO (same cells, so comparable):")
        print(comparison.to_string())
        report["loo_compare"] = json.loads(comparison.to_json(orient="index"))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
