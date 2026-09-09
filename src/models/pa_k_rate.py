"""
PA-level Bayesian Strikeout Rate Model — the K% face of `src.models.pa_rate`

The model this module used to contain now lives in `src.models.pa_rate`,
parameterised by component (BAS-73): BB/PA and HR/PA are the same
hierarchical binomial over the same (batter, season, team, stand [, pitcher])
cells with a different numerator column and a different league-level prior,
so keeping three copies of a 1,300-line file was the wrong shape.

Nothing about K% changed. This module re-exports `pa_rate`'s surface and
pins `component="k_rate"` on the four functions that take one, so every
existing call site — `src.eval.bayes_arm`, `modal_functions/app.py`, the
tests, `python -m src.models.pa_k_rate` — keeps working with the same
signatures and the same numbers.

**"The same numbers" is a checked claim.** `tests/test_models/test_pa_rate.py`
fits the fixture in `tests/test_models/reference_k_rate.py` through this
module in all four `ModelOptions` combinations and compares every posterior
mean and every projected rate, exactly (`rtol=0, atol=0`), against
`tests/test_models/k_rate_reference.json` — which was generated from the
pre-split `pa_k_rate.py`, before `pa_rate.py` existed. The fixture uses the
pure-Python `pymc` NUTS sampler at a fixed seed precisely so a fresh process
reproduces the same draws bit-for-bit; the JIT backends make no such promise.

Two things K% keeps that the other components do not, both documented where
they live in `pa_rate.py`: the `league_init` prior mean stays pinned at
-1.27 rather than being read off the earliest training season (see
`RateComponent.league_init_mu` — the data-driven value would be -1.21 to
-1.26, within a fifth of the prior's own sd, but pinning is what makes the
bit-for-bit check above possible), and the likelihood variable stays named
`obs_k`.

Usage:
    python -m src.models.pa_k_rate              # local test
    from src.models.pa_k_rate import run_model  # programmatic
"""
from __future__ import annotations

import functools
import os

import pandas as pd

from src.models import pa_rate
from src.models.pa_rate import (  # noqa: F401  (re-exported surface)
    ABILITY_STEP_SIGMA_PRIOR,
    AGE_PEAK_WINDOW,
    BASE_PA_COLUMNS,
    DATA_DIR,
    MIN_PA_THRESHOLD,
    PARQUET_DIR,
    PITCHER_SIGMA_PRIOR,
    PROJECT_ROOT,
    PROJECTION_YEAR,
    REFERENCE_AGE,
    SAMPLER_KWARGS,
    ModelOptions,
    _age_term_function,
    _project_unseen,
    load_park_factors,
    logger,
    model_diagnostics,
    sample_model,
)

# The one component this module speaks for. Everything below is `pa_rate`
# with this bound in; nothing else differs.
K_RATE = pa_rate.RATE_COMPONENTS["k_rate"]

load_pa_data = functools.partial(pa_rate.load_pa_data, component=K_RATE)
prepare_model_data = functools.partial(pa_rate.prepare_model_data,
                                       component=K_RATE)
run_model = functools.partial(pa_rate.run_model, component=K_RATE)

# `build_model`, `generate_projections` and `log_to_wandb` read the component
# off `data["component"]`, which `prepare_model_data` above already stamped,
# so they need no binding — and must not get one, or a caller who prepared
# BB% data and imported `build_model` from here would silently get a K% graph
# instead of the KeyError it deserves.
build_model = pa_rate.build_model
generate_projections = pa_rate.generate_projections
log_to_wandb = pa_rate.log_to_wandb

# `functools.partial` has no `__doc__` of its own worth reading; point it at
# the function it wraps so `help()` and IDEs still work.
for _bound, _target in ((load_pa_data, pa_rate.load_pa_data),
                        (prepare_model_data, pa_rate.prepare_model_data),
                        (run_model, pa_rate.run_model)):
    _bound.__doc__ = _target.__doc__


if __name__ == "__main__":
    """Run the K% model locally with reduced sampling for testing.

    Set environment variables to override defaults:
        PA_DIR:    Path to PA outcomes parquet directory
        PF_PATH:   Path to park_factors.parquet
        MIN_PA:    Minimum PA threshold (default 50)
        FAST:      If '1', use minimal sampling for quick test
        NO_WANDB:  If '1', skip wandb logging
    """
    fast_mode = os.environ.get("FAST", "0") == "1"
    no_wandb = os.environ.get("NO_WANDB", "0") == "1"

    overrides: dict = {}
    if fast_mode:
        logger.info("FAST MODE: minimal sampling for testing")
        overrides = dict(draws=100, tune=100, chains=2, cores=1)

    trace, projections, model_data = run_model(
        pa_dir=os.environ.get("PA_DIR"),
        pf_path=os.environ.get("PF_PATH"),
        min_pa=int(os.environ.get("MIN_PA", MIN_PA_THRESHOLD)),
        log_wandb=not no_wandb,
        wandb_offline=True,  # default to offline for local runs
        **overrides,
    )

    pd.set_option("display.width", 200)
    print("\n" + "=" * 60)
    print(f"K-Rate Projections for {PROJECTION_YEAR}")
    print("=" * 60)
    print("\nLowest projected K% (best contact):")
    print(projections.head(15).to_string(index=False))
    print("\nHighest projected K% (most Ks):")
    print(projections.tail(15).to_string(index=False))
    print(f"\nMedian projected K%: {projections['projected_k_rate'].median():.1%}")
    print(f"Mean projected K%:   {projections['projected_k_rate'].mean():.1%}")
