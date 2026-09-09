"""The K% fixture the bit-for-bit equivalence test is pinned to.

Shared by `tests/test_models/test_pa_rate.py` and by the one-off generator
that produced `k_rate_reference.json` from the *pre-BAS-73* `pa_k_rate.py`
(the version that owned the whole model, before `pa_rate.py` existed). Both
sides therefore fit the same rows with the same seed through the same
sampler, which is the only way the comparison means anything.

Deliberately tiny — 12 batters x 3 seasons, 25 draws, one chain, the `pymc`
NUTS sampler — because the point is reproducing an exact number, not a good
posterior. The `pymc` sampler (not `numpyro`) is chosen because it is pure
Python/NumPy seeded from `random_seed`, so the same point sequence comes back
in a fresh process; the JIT-compiled backends do not promise that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Same shape as `tests/test_models/test_pa_k_rate_options.py`'s `_pa_rows`,
# frozen here so a change to that helper cannot silently move the pin.
FIXTURE_SEASONS = (2023, 2024, 2025)
FIXTURE_SEED = 3
FIXTURE_N_BATTERS = 12

FAST_SAMPLER = dict(draws=25, tune=25, chains=1, cores=1,
                    nuts_sampler="pymc", progressbar=False, random_seed=59,
                    idata_kwargs={"log_likelihood": False})


def fixture_pa_rows() -> pd.DataFrame:
    rng = np.random.default_rng(FIXTURE_SEED)
    rows = []
    for b in range(FIXTURE_N_BATTERS):
        true_rate = rng.uniform(0.15, 0.35)
        for s in FIXTURE_SEASONS:
            for _ in range(rng.integers(60, 100)):
                rows.append({
                    "batter": 1000 + b,
                    "game_year": s,
                    "game_date": pd.Timestamp(f"{s}-06-15"),
                    "stand": "R" if b % 2 == 0 else "L",
                    "is_k": int(rng.random() < true_rate),
                    "is_bb": int(rng.random() < 0.09),
                    "is_hr": int(rng.random() < 0.03),
                    "home_team": "HOU",
                    "away_team": "SEA",
                    "inning_topbot": "Top",
                })
    return pd.DataFrame(rows)


OPTION_COMBOS = (
    ("flat", dict(ability_walk=False, constrained_age=False)),
    ("walk", dict(ability_walk=True, constrained_age=False)),
    ("age", dict(ability_walk=False, constrained_age=True)),
    ("walk_age", dict(ability_walk=True, constrained_age=True)),
)


def summarise_k_rate(module) -> dict:
    """Posterior means and projections for every option combo, from `module`.

    `module` is whatever exposes the pre-BAS-73 `pa_k_rate` surface —
    `prepare_model_data`, `build_model`, `sample_model`,
    `generate_projections`, `ModelOptions`. The reference snapshot and the
    shipped wrapper both satisfy it, which is the whole trick: one function,
    two modules, and any difference between them shows up as a float that
    does not match.
    """
    pa = fixture_pa_rows()
    out: dict = {}
    for name, flags in OPTION_COMBOS:
        data = module.prepare_model_data(pa, None, min_pa=1,
                                         include_pitcher=False)
        model = module.build_model(data, module.ModelOptions(**flags))
        trace = module.sample_model(model, **FAST_SAMPLER)
        proj = module.generate_projections(trace, data, projection_year=2026)
        post = trace.posterior
        means = {
            str(v): float(np.asarray(post[v].values, dtype="float64").mean())
            for v in post.data_vars
        }
        out[name] = {
            "posterior_means": means,
            "projected_k_rate": [float(x) for x in proj["projected_k_rate"]],
            "batters": [int(b) for b in proj["batter"]],
        }
    return out
