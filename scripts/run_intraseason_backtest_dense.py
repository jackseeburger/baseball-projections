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

    bayes sweep   adds the `bayes` arm(s) — the PA-level K% model refit at
                  the cutoff (`src.eval.bayes_arm`) — on top of the same
                  cheap arms. One fit is an MCMC run (~75-90s here with
                  NumPyro on 4 cores; the honest number for whatever machine
                  runs this is printed at the end of the run), so this sweep
                  is BIWEEKLY on a 4-season subset (2022, 2024, 2025, 2026) —
                  scoped deliberately per the task's cost-control guidance
                  rather than run to completion on all 7 seasons.

Pre-registered prediction (recorded before the densified numbers were read,
in the commit that added this script): partial pooling should help most when
samples are smallest, so the Bayesian arm's advantage over tuned Marcel
should be LARGEST in April/May and shrink roughly monotonically through
September. The three existing cutoffs show the opposite — Bayes worse in May
and July, only a tie by August. See docs/densified-intraseason-backtest.md
for the verdict.

**Structural variants (BAS-69, docs/bayes-variants.md).** The bayes sweep
scores several structures of the same arm in one pass — `flat` (the model as
scored above), `ability_walk`, `constrained_age`, and their combination —
selected by flags on `src.eval.bayes_arm.BayesArmConfig`. Each variant gets
its own name in the results frame (`bayes_flat`, `bayes_walk`, `bayes_age`,
`bayes_walk_age` — see `VARIANT_ARM_NAMES`) and its own `bayes_k_rate_provider`
call, so one variant's MCMC fit can never be served under another variant's
label — see `make_variant_providers`'s docstring for exactly why that would
otherwise be a live risk. `bayes` is kept as an exact alias for `bayes_flat`
(same fit, zero extra cost — see the module's "alias" comment in
`make_variant_providers`) so every number this doc already reports under the
name `bayes` keeps meaning the same thing.

**Components (BAS-73, docs/bayes-components.md).** The bayes arm served K%
only until BB/PA and HR/PA turned out to be the same hierarchical binomial
with a different numerator (`src.models.pa_rate`). `--bayes-components`
selects which; the default is `k_rate` alone, so a command that does not name
any runs exactly the sweep it always ran. The checkpoint and the fit records
key on the component as well as (season, cutoff, variant) — three components
x four variants is twelve MCMC fits per cutoff, and the one-season-per-process
rule below matters proportionally more.

**The joint model (BAS-84, docs/bayes-joint.md).** `--variants joint_walk`
(alias for `joint+ability_walk`; `joint_flat` for `joint`) fits K%, BB% and
HR/PA in *one* model, with a per-batter ability vector across components under
an LKJ(2) correlation prior (`src.models.pa_joint`). The checkpoint still keys
on (component, season, cutoff) exactly as before — a joint arm's rows are
written per component like every other arm's — but the arithmetic underneath
runs the other way: one MCMC fit per (season, cutoff) fills all three
components, memoized in `src.eval.bayes_arm`, so
`--bayes-components k_rate bb_rate hr_rate --variants joint_walk` costs one
fit per cutoff and not three. Every fit record carries `joint_params`: the
posterior mean and 95% interval of each pairwise ability correlation and each
per-component `sigma_step`.

Usage:
    # one-time data prep (writes gitignored data/parquet/pa_outcomes/*)
    python -c "from src.data.pa_outcomes_pipeline import build_pa_dataset; \\
               build_pa_dataset(years=[2019,2021,2022,2023,2024,2025])"
    python -c "from src.data.pa_outcomes import download; download(2026, 'data/parquet/pa_outcomes')"

    python scripts/run_intraseason_backtest_dense.py --stage cheap
    python scripts/run_intraseason_backtest_dense.py --stage bayes
    python scripts/run_intraseason_backtest_dense.py --stage analyze

**Run the bayes stage one season per process.** With four variants it does
four MCMC fits per cutoff, and NumPyro compiles fresh XLA kernels for each
one; the JIT's memory is never reclaimed within a process. At around 70
compilations the run dies with `LLVM ERROR: Unable to allocate section
memory!` — not an out-of-memory in the usual sense (15 GB was free), and not
catchable, because LLVM aborts rather than raising. It happened at the 19th
cell of a ~40-cell grid on 2026-09-08.

    for yr in 2022 2024 2025 2026; do
        python scripts/run_intraseason_backtest_dense.py --stage bayes \
            --bayes-seasons $yr --bayes-components k_rate bb_rate hr_rate
    done

Each season is a fresh process and therefore a fresh JIT cache. The
checkpoint is keyed on (component, season, cutoff), so this is exactly
equivalent to one long run and a run that dies mid-grid loses only its
current cell. Splitting by component as well as by season is the same trick
again when the grid is wide enough to need it.
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

# pymc/arviz are not imported here (nor at import time by src.eval.bayes_arm —
# see its own docstring): this module has to be importable, and its non-bayes
# functions runnable, in CI, which installs requirements-ci.txt and has
# neither.

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

# ─── bayes structural variants (BAS-69, docs/bayes-variants.md) ───
# Keys are `BayesArmConfig.variant()`'s own vocabulary — "+".join of whichever
# of ("ability_walk", "constrained_age") are on, "flat" if neither — so a
# config built from one of these strings and `.variant()` called on the
# result always agree; `_variant_config` below asserts that on every config
# it builds, rather than trusting the two vocabularies to stay in sync by
# convention alone.
VARIANT_ARM_NAMES = {
    "flat": "bayes_flat",
    "ability_walk": "bayes_walk",
    "constrained_age": "bayes_age",
    "ability_walk+constrained_age": "bayes_walk_age",
    # BAS-84, docs/bayes-joint.md: one fit over K%, BB% and HR/PA with a
    # per-batter ability vector and an LKJ(2) correlation between them. The
    # flag joins the same "+"-slug vocabulary, first because
    # `BayesArmConfig.variant()` lists it first, and the fit behind a joint
    # arm is one MCMC run per (season, cutoff) that fills all three
    # components -- the checkpoint still keys on (component, season, cutoff),
    # so nothing downstream has to know that.
    "joint": "bayes_joint",
    "joint+ability_walk": "bayes_joint_walk",
}
ARM_NAME_VARIANT = {arm: variant for variant, arm in VARIANT_ARM_NAMES.items()}
# Spellings `--variants` accepts for the joint arms, because "joint_walk" is
# what docs/bayes-joint.md calls the arm and "joint+ability_walk" is what
# `BayesArmConfig.variant()` calls the same structure. Resolved once, on the
# way in, so everything downstream (arm names, the checkpoint, the fit
# records) speaks the config's own vocabulary and only one of the two names
# can ever appear in a results table.
VARIANT_ALIASES = {
    "joint_flat": "joint",
    "joint_walk": "joint+ability_walk",
}


def resolve_variant(name: str) -> str:
    """`VARIANT_ALIASES` applied, unknown names left alone so the caller's
    own error message is the one that fires."""
    return VARIANT_ALIASES.get(name, name)


# BAS-69's full design: the two single-component flags each on their own, and
# together. Default sweep scope, overridable with --variants. Spelled out
# rather than read off `VARIANT_ARM_NAMES` so that adding an arm to that
# registry (the joint ones, BAS-84) does not silently quadruple what a
# command with no `--variants` fits.
DEFAULT_VARIANTS = ["flat", "ability_walk", "constrained_age",
                    "ability_walk+constrained_age"]
assert set(DEFAULT_VARIANTS) <= set(VARIANT_ARM_NAMES)

# Posterior scalars a variant's own structure adds, named exactly as
# docs/bayes-variants.md's math names them. `model_diagnostics()` (src/models/
# pa_k_rate.py) reports the single worst r-hat/ESS across *every* variable in
# the trace — useful for "did this fit sample cleanly", useless for "what did
# sigma_step land on", which is what the doc's vacuity check needs: if
# sigma_step's posterior concentrates near zero the walk collapsed to the flat
# model and the doc's prediction 1 is untestable, not false. Kept as data here
# (not logic in bayes_arm.py/pa_k_rate.py) so it can be extended once the
# variants are actually implemented without touching either file.
VARIANT_OWN_PARAMS = {
    "ability_walk": ["sigma_step"],
    # No entry for "joint" (BAS-84). Its own scalars are per component --
    # three `sigma_step`s and three pairwise ability correlations, named after
    # the components they belong to -- so a flat list of names here could not
    # say which is which, and a joint trace carries no plain `sigma_step` at
    # all (it is a vector; averaging three components' step sizes into one
    # number is worse than reporting none), so the walk entry above finds
    # nothing and is skipped exactly as this function's docstring says it
    # should be. What the joint fit records instead is `joint_params` in its
    # data summary: mean and 95% interval per correlation and per sigma_step,
    # written by `src.models.pa_joint.joint_param_summary`, which is what
    # docs/bayes-joint.md's predictions 1 and 5 are read off.
    # `peak_age`, not `peak`: the model names the Deterministic that scales
    # `peak_frac` onto AGE_PEAK_WINDOW `peak_age` (src/models/pa_k_rate.py).
    # The mismatch cost the first sweep the single most interesting number
    # this variant produces — where the model actually puts the K% peak,
    # which is directly comparable to the peak `scripts/tune_marcel.py` fits
    # for tuned Marcel — and cost it silently, because a name missing from
    # the trace is skipped rather than raised on. The test below pins the
    # names against the model module so the next mismatch is loud.
    "constrained_age": ["peak_age", "slope_young", "slope_old"],
}


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


# ─── contact_additive, the served covariate engine, as a sweep arm (BAS-83) ───

# `src.eval.contact.LIVE_CELL_SEASONS` is the full walk-forward training set
# for the additive fit. Only the seasons whose PA parquet is actually on disk
# can be used, so the arm reports which ones it fitted on rather than failing
# or, worse, quietly training on fewer cells than a reader assumes.
CONTACT_ARM = "contact_additive"


def _available_cell_seasons(pa_dir: Path, predict_year: int) -> tuple[int, ...]:
    from src.eval.contact import LIVE_CELL_SEASONS

    return tuple(s for s in LIVE_CELL_SEASONS
                 if s < predict_year
                 and (pa_dir / f"pa_outcomes_{s}.parquet").exists())


def make_contact_provider(component: str, cutoff: str, year: int,
                          seasons_table: pd.DataFrame, monthly: pd.DataFrame,
                          pa_dir: Path, sink: dict | None = None):
    """The `contact_additive` engine (docs/contact-quality.md §8) as a harness
    provider at one cutoff — the comparator BAS-83 pre-registered.

    This is the *served* shape, not the free fit: `fit_contact(...,
    fixed_base=True)` pins `marcel_tuned`'s coefficient at exactly 1 and fits
    only the intercept and the six covariate coefficients, on cell seasons
    strictly before `year`. It is the same construction
    `src.projections.ros.contact_engine_provider` serves live, down to the
    month-lagged cutoff (`ros.contact_cutoff`), so what the sweep scores and
    what the board serves are one estimator.

    Returns `None` — the arm is simply absent from that cell — when there are
    no training cell seasons on disk before `year`, rather than fitting on
    nothing.
    """
    from src.eval import contact as contact_eval
    from src.projections.ros import contact_cutoff

    cell_seasons = _available_cell_seasons(pa_dir, year)
    if not cell_seasons:
        logger.warning("contact_additive: no PA parquets before %d — arm skipped",
                       year)
        return None
    cells = contact_eval.build_hitter_cells(
        seasons_table, pa_dir, [component], seasons=cell_seasons)
    if cells.empty:
        logger.warning("contact_additive: no training cells for %s before %d",
                       component, year)
        return None
    cells = contact_eval.attach_live_features(cells, monthly)
    fit = contact_eval.fit_contact(cells, component,
                                   features=contact_eval.FEATURES,
                                   fixed_base=True)
    if sink is not None:
        sink.update({
            "component": component, "cutoff": cutoff,
            "train_seasons": [int(s) for s in cell_seasons],
            "n_rows": fit.n_rows, "n_cells": fit.n_cells,
            "coef": dict(fit.coef),
            "contact_cutoff": str(contact_cutoff(cutoff).date()),
        })
    config = contact_eval.ContactProviderConfig(
        monthly=monthly, cutoff=contact_cutoff(cutoff), predict_year=year,
        fit=fit, base_provider=INTRASEASON_BASELINES["marcel_tuned"],
        side="hitter",
    )
    return contact_eval.contact_provider(config)


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

def _variant_config(variant: str, **kwargs):
    """A `BayesArmConfig` for one variant name, self-checked against
    `BayesArmConfig.variant()` so the two vocabularies (this module's short
    names, the config's own flags) cannot silently drift apart."""
    from src.eval.bayes_arm import BayesArmConfig

    variant = resolve_variant(variant)
    on = set(variant.split("+"))
    config = BayesArmConfig(
        ability_walk="ability_walk" in on, constrained_age="constrained_age" in on,
        joint="joint" in on,
        **kwargs,
    )
    assert config.variant() == variant, (
        f"variant name {variant!r} does not round-trip through "
        f"BayesArmConfig.variant() (got {config.variant()!r}) — "
        f"VARIANT_ARM_NAMES and BayesArmConfig's flags have drifted apart"
    )
    return config


def variant_param_summary(trace, config) -> dict:
    """Posterior mean/sd for every scalar a variant's own structure adds.

    This is the fits-file half of the vacuity check in
    docs/bayes-variants.md: `sigma_step`'s posterior mean is the number that
    says whether `ability_walk` collapsed to the flat model at this window
    length, and that number lives nowhere else in what gets written down —
    `model_diagnostics()` only ever reports the single worst r-hat/ESS across
    every variable. Reads straight off the trace's posterior group rather
    than through arviz's summary machinery, so this needs nothing beyond
    `.posterior[name].values` and stays testable without pymc/arviz installed
    (a plain object with a `.posterior` mapping is enough — see
    tests/test_scripts/test_run_intraseason_backtest_dense.py).

    A name from VARIANT_OWN_PARAMS that isn't in the trace is skipped, not
    raised on: the model side may land under a different name than the
    pre-registration's math used, and this table should say what it found,
    not assume a naming convention that turns out to be wrong.
    """
    on = [n for n in ("ability_walk", "constrained_age") if getattr(config, n)]
    names = [p for n in on for p in VARIANT_OWN_PARAMS.get(n, [])]
    out: dict = {}
    posterior = getattr(trace, "posterior", None) if trace is not None else None
    if posterior is None:
        return out
    for name in names:
        if name not in posterior:
            continue
        vals = np.asarray(posterior[name].values, dtype="float64").ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[name] = {"mean": float(vals.mean()), "sd": float(vals.std())}
    return out


def make_variant_providers(cutoff: str, year: int, variants: list[str],
                           config_kwargs: dict, fits_sink: list[dict],
                           component: str = "k_rate") -> dict:
    """One `bayes_k_rate_provider` call per variant — the cache isolation guard.

    `bayes_k_rate_provider`'s memoization cache keys on `(component,
    cutoff_date, predict_year)` (see src/eval/bayes_arm.py) — no variant in
    the key. The component *is* in the key, because one provider object can
    legitimately be asked for a component more than once at the same cutoff
    and a date-only key would hand a K% frame back for a BB% request; the
    variant is not, and does not need to be, because the cache is a fresh
    dict created
    by that call's own closure, so it cannot see another call's fits. The bug
    this function exists to prevent is upstream of the cache entirely: handing
    the *same provider object* to two different variant names in the
    `providers` dict passed to `backtest()`, which would silently score one
    variant's fit under another variant's label without the cache itself ever
    doing anything wrong. So this is the one place in the sweep allowed to
    construct these providers — every variant gets its own
    `bayes_k_rate_provider(...)` call and its own object — and `on_fit` below
    re-checks the fit's own `config.variant()` against what was asked for, so
    a violation of that invariant raises instead of quietly mislabeling a fit.
    `tests/test_scripts/test_run_intraseason_backtest_dense.py` exercises both
    guards without needing pymc, by monkeypatching `fit_bayes_k_rate`.

    The one deliberate exception: "flat" is also served under the plain
    "bayes" key, as the *same* provider object (so the *same* cache) rather
    than a second call — this is the continuity alias described in the module
    docstring, and it costs zero extra MCMC because it is a cache hit by
    construction, never a second fit.

    Per-variant wall time is attached to `fits_sink`'s record for that fit,
    measured around the provider call rather than inside `on_fit` (which
    fires before the model's projection pass and before the call returns), so
    "elapsed_s" is only ever set on a fit that actually ran a real MCMC
    fit — a cache hit (a second call to an already-fit provider, such as the
    "bayes" alias resolving after "bayes_flat" already fit) does not overwrite
    it or add a spurious near-zero timing.
    """
    from src.eval.bayes_arm import bayes_k_rate_provider

    providers: dict = {}
    seen_ids: set[int] = set()
    for variant in variants:
        config = _variant_config(variant, component=component, **config_kwargs)
        arm_name = VARIANT_ARM_NAMES[variant]
        holder: dict = {}

        def on_fit(fit, _variant=variant, _arm=arm_name, _holder=holder):
            if fit.config.variant() != _variant:
                raise RuntimeError(
                    f"cache isolation broken: provider {_arm!r} was asked "
                    f"for variant {_variant!r} but the fit that came back "
                    f"reports {fit.config.variant()!r} — a "
                    f"bayes_k_rate_provider cache must never be shared "
                    f"across variants"
                )
            if fit.config.component != component:
                raise RuntimeError(
                    f"component isolation broken: provider {_arm!r} was "
                    f"asked for {component!r} but the fit that came back "
                    f"reports {fit.config.component!r} — a "
                    f"bayes_k_rate_provider cache must never be shared "
                    f"across components either"
                )
            record = {
                "cutoff": fit.cutoff_date, "component": component,
                "variant": _variant, "arm": _arm,
                "scale": fit.config.label(), "diagnostics": fit.diagnostics,
                "variant_params": variant_param_summary(fit.trace, fit.config),
                **fit.data_summary,
            }
            fits_sink.append(record)
            _holder["record"] = record

        raw = bayes_k_rate_provider(cutoff, year, config, on_fit=on_fit)
        assert id(raw) not in seen_ids, "two variants got the same provider object"
        seen_ids.add(id(raw))

        def timed(train, spec, predict_year, _raw=raw, _holder=holder):
            t0 = time.time()
            out = _raw(train, spec, predict_year)
            if "record" in _holder:  # a real fit just happened, not a cache hit
                _holder["record"]["elapsed_s"] = round(time.time() - t0, 1)
                del _holder["record"]
            return out

        providers[arm_name] = timed

    if "flat" in variants:
        providers["bayes"] = providers[VARIANT_ARM_NAMES["flat"]]
    return providers


def _load_bayes_checkpoint(path: Path) -> pd.DataFrame:
    """Load a bayes-sweep checkpoint, refusing to reinterpret a pre-variant one.

    Before variants existed, every bayes-arm row here was `model="bayes"` and
    meant exactly one thing: the flat model, fit under whatever CLI scope
    (`--bayes-seasons`, sampler settings) that run used. Silently relabelling
    those rows "bayes_flat" and treating the cell as done would risk two
    things going wrong without any error to catch them: (1) this run's scope
    may differ from that old run's, so "already done" would compare a stale,
    differently-scoped fit against the rest of a table computed under today's
    settings, and (2) nothing downstream would ever be able to tell an old
    "bayes" row and a new "bayes_flat" row apart as the same measurement or a
    different one. A pre-variant checkpoint is therefore a hard stop: move it
    aside (e.g. rename to cells_bayes.legacy.parquet) and start a fresh one,
    or rerun --stage bayes from scratch if you want its numbers folded back
    in as the flat variant.
    """
    if not path.exists():
        return pd.DataFrame()
    prev = pd.read_parquet(path)
    if prev.empty or "model" not in prev.columns:
        return prev
    present = set(prev["model"].unique())
    if "bayes" in present and not (set(VARIANT_ARM_NAMES.values()) & present):
        raise RuntimeError(
            f"{path} is a pre-variant-sweep checkpoint (only a 'bayes' "
            f"model column, none of {sorted(VARIANT_ARM_NAMES.values())} "
            "present) — see _load_bayes_checkpoint's docstring for why this "
            "isn't reinterpreted automatically. Move it aside and start a "
            "fresh checkpoint, or rerun --stage bayes from scratch."
        )
    return prev


def _load_bayes_fits(path: Path) -> list[dict]:
    """Same refusal as `_load_bayes_checkpoint`, for the diagnostics file."""
    if not path.exists():
        return []
    fits = json.loads(path.read_text())
    if fits and not all("variant" in f for f in fits):
        raise RuntimeError(
            f"{path} has fit records from before variants existed (missing "
            "'variant'). Move it aside or delete it — see "
            "_load_bayes_checkpoint's docstring for the same reasoning."
        )
    return fits


BAYES_COMPONENTS = ("k_rate", "bb_rate", "hr_rate")
DEFAULT_BAYES_COMPONENTS = ["k_rate"]


def _done_variants_for_cell(have: pd.DataFrame | None) -> set[str]:
    """Which variants a checkpointed (season, cutoff) cell's rows already
    cover, read off the `model` column via `ARM_NAME_VARIANT`. `have=None`
    (nothing checkpointed for this cell yet) is the empty set."""
    if have is None:
        return set()
    return {ARM_NAME_VARIANT[m] for m in have["model"].unique() if m in ARM_NAME_VARIANT}


def run_bayes(seasons_table: pd.DataFrame, pa_by_year: dict[int, pd.DataFrame],
             seasons: tuple[int, ...], cutoffs_mmdd: list[str],
             min_trials: int = MIN_TRIALS, checkpoint: Path | None = None,
             fits_path: Path | None = None,
             draws: int = 500, tune: int = 500, chains: int = 2,
             cores: int | None = None,
             sampler: str = "numpyro", include_pitcher: bool = False,
             pa_dir: Path = ROOT / "data/parquet/pa_outcomes",
             variants: list[str] | None = None,
             components: list[str] | None = None,
             seasons_table_for_contact: pd.DataFrame | None = None,
             monthly: pd.DataFrame | None = None,
             contact_arm: bool = False,
             contact_fits: list[dict] | None = None,
             ) -> tuple[pd.DataFrame, list[dict]]:
    """Score every requested bayes variant, plus the cheap baselines, at each
    (component, season, cutoff) — one fit per (component, variant, season,
    cutoff), memoized across resumed runs at cell granularity.

    **Components (BAS-73).** `components` defaults to `["k_rate"]`, so a
    command that does not name any runs exactly the sweep it always ran. BB%
    and HR/PA are the same hierarchical binomial with a different numerator
    (`src.models.pa_rate`) and cost the same per fit, so the grid multiplies:
    three components x four variants is twelve MCMC fits per cutoff, which is
    why the module docstring's one-season-per-process rule matters more here
    than it did — the JIT dies at ~70 compilations.

    Resume is all-or-nothing *per cell*, not per variant: if a checkpointed
    cell already carries every arm in `variants`, it is skipped outright; if
    it is missing even one, the whole cell is recomputed (baselines and every
    requested variant, not just the missing one) and its old rows are
    replaced rather than appended to. The alternative — fitting only the
    missing variant(s) and splicing their rows into an already-checkpointed
    cell — would score them under a different `common_players` intersection
    than what is already on disk for that cell (`backtest()` intersects
    predicted coverage across whatever `providers` dict it is given *in that
    call*), which is exactly the kind of silent inconsistency this sweep
    exists to avoid. Refitting an already-done variant is the accepted cost
    of that guarantee; it only bites a resume that also changes `--variants`
    mid-sweep, which is rare next to Ctrl-C-and-resume with a fixed variant
    list (the common case, where every touched cell is already complete and
    costs nothing to skip).
    """
    variants = [resolve_variant(v) for v in variants] if variants else list(DEFAULT_VARIANTS)
    unknown = [v for v in variants if v not in VARIANT_ARM_NAMES]
    if unknown:
        raise ValueError(f"unknown bayes variant(s) {unknown}; "
                         f"known: {sorted(VARIANT_ARM_NAMES)}")
    components = list(components) if components else list(DEFAULT_BAYES_COMPONENTS)
    unknown_c = [c for c in components if c not in BAYES_COMPONENTS]
    if unknown_c:
        raise ValueError(f"unknown bayes component(s) {unknown_c}; the "
                         f"PA-level binomial serves {list(BAYES_COMPONENTS)}")

    cells: dict[tuple, pd.DataFrame] = {}
    fits: list[dict] = []
    if checkpoint is not None:
        prev = _load_bayes_checkpoint(checkpoint)
        if not prev.empty:
            # Keyed on the component too: a 2024-05-01 BB% cell and a
            # 2024-05-01 K% cell are different measurements, and before
            # BAS-73 a two-key groupby would have collapsed them into one
            # entry whose second write silently replaced the first.
            for (component, season, cutoff), g in prev.groupby(
                    ["component", "season", "cutoff"]):
                cells[(component, season, cutoff)] = g
            n_variant_cells = sum(
                1 for g in cells.values()
                for m in g["model"].unique() if m in ARM_NAME_VARIANT
            )
            logger.info("resuming bayes sweep: %d cells checkpointed, "
                       "%d arm-cells among them", len(cells), n_variant_cells)
    if fits_path is not None:
        fits = _load_bayes_fits(fits_path)

    available = set(pa_by_year)
    for year in seasons:
        pa = pa_by_year[year]
        last_pa_date = pa["game_date"].max()
        bayes_seasons = bayes_prior_seasons(year, available)
        for md in cutoffs_mmdd:
            cutoff = f"{year}-{md}"
            if pd.Timestamp(cutoff) >= last_pa_date:
                continue
            for component in components:
                key = (component, year, cutoff)
                if set(variants) <= _done_variants_for_cell(cells.get(key)):
                    continue

                t0 = time.time()
                config_kwargs = dict(
                    pa_dir=pa_dir, seasons=bayes_seasons, min_pa=50,
                    include_pitcher=include_pitcher, max_batters=None,
                    draws=draws, tune=tune, chains=chains,
                    cores=(chains if cores is None else cores),
                    target_accept=0.9, nuts_sampler=sampler,
                )
                # Recomputing the cell invalidates any fit records already
                # written for it (see the docstring above), so drop them
                # before `make_variant_providers`'s on_fit hooks append the
                # fresh ones. Records written before BAS-73 carry no
                # "component" key at all; those are k_rate by construction,
                # which is what the `or "k_rate"` says.
                fits[:] = [
                    f for f in fits
                    if not (f.get("cutoff") == cutoff
                            and (f.get("component") or "k_rate") == component)
                ]

                providers = dict(INTRASEASON_BASELINES)
                if contact_arm:
                    sink: dict = {}
                    cp = make_contact_provider(
                        component, cutoff, year, seasons_table_for_contact,
                        monthly, pa_dir, sink)
                    if cp is not None:
                        providers[CONTACT_ARM] = cp
                        if contact_fits is not None:
                            contact_fits.append(sink)
                providers.update(make_variant_providers(
                    cutoff, year, variants, config_kwargs, fits,
                    component=component))
                try:
                    results = backtest(
                        component, cutoff_date=cutoff, predict_year=year,
                        seasons=seasons_table, pa_frame=pa, providers=providers,
                        min_trials=min_trials,
                    )
                except ValueError as e:
                    logger.warning("skip bayes %s %s (%s): %s",
                                   component, cutoff, variants, e)
                    continue
                elapsed = time.time() - t0
                cells[key] = results.assign(season=year, cutoff=cutoff)
                out = pd.concat(cells.values(), ignore_index=True)
                if checkpoint is not None:
                    out.to_parquet(checkpoint, index=False)
                if fits_path is not None:
                    fits_path.write_text(json.dumps(fits, indent=1))
                logger.info("bayes: %s %s done in %.1fs (%d variants: %s, "
                            "%d rows so far)", component, cutoff, elapsed,
                            len(variants), ",".join(variants), len(out))
    if not cells:
        return pd.DataFrame(), fits
    return pd.concat(cells.values(), ignore_index=True), fits


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


MIN_CLUSTERS_FOR_T = 2


def _suppress_degenerate_t(result: dict) -> dict:
    """NaN out a t-statistic backed by fewer than two clusters.

    `paired_abs_error_diff` reports `n_clusters` alongside the SE. With one
    cluster the clustered SE is a sum of one residual and the ratio is
    meaningless — but finite, and often enormous, which is worse than
    missing. Returns a copy so the caller's other numbers (diff, n) survive.
    """
    if result.get("n_clusters", 0) >= MIN_CLUSTERS_FOR_T:
        return result
    out = dict(result)
    out["t"] = float("nan")
    out["se"] = float("nan")
    return out


def variant_comparison(cells: pd.DataFrame, arm: str, base: str,
                       component: str = "k_rate") -> dict:
    """One pooled arm-vs-base comparison, reported the honest way and the
    wrong way side by side, plus a per-cell win/loss count.

    docs/densified-intraseason-backtest.md found unclustered t running
    **3.71x** the player-clustered one on this exact harness (46,591 rows,
    870 real hitters averaging 54 rows apiece) — not a rounding difference, a
    number that changes which comparisons look significant. So every
    comparison here reports three numbers rather than one:

    - clustered by player (the primary read — the same hitter scored at
      several cutoffs/seasons is not several independent draws about him);
    - clustered by (season, cutoff) cell — the other defensible grouping,
      since every row in one cell shares that cutoff's single MCMC fit and
      its idiosyncrasies. Reported alongside because the two disagreeing
      would itself be worth knowing, not folded into one "the" clustered
      number;
    - unclustered, explicitly labelled `_WRONG` so nobody downstream can
      quote it by accident, with the ratio to the player-clustered t
      attached so the inflation is visible without a second lookup.

    `paired_abs_error_diff` (src/eval/tuning.py) already does the clustered
    math via its `cluster_col` argument — this function only picks the keys
    and reuses it three times, once per clustering choice, rather than
    reimplementing anything.

    Win/loss reuses `paired_by_cell`: its `diff` *is* MAE(arm) - MAE(base) on
    that cell's common population (trials-weighted mean of |err_a| - |err_b|
    equals the difference of trials-weighted MAEs), so `diff < 0` is exactly
    "arm's MAE was lower here" with no separate MAE computation needed.
    """
    g = cells[cells["component"] == component]
    a, b = g[g["model"] == arm], g[g["model"] == base]
    if a.empty or b.empty:
        return {}

    key_a = a["season"].astype(str) + "|" + a["cutoff"] + "|" + a["batter"].astype(str)
    key_b = b["season"].astype(str) + "|" + b["cutoff"] + "|" + b["batter"].astype(str)
    cell_a = a["season"].astype(str) + "|" + a["cutoff"]
    a = a.assign(_key=key_a, _player=a["batter"], _cell=cell_a)
    b = b.assign(_key=key_b)
    cols_a = ["_key", "predicted", "realized_rate", "trials", "_player", "_cell"]
    cols_b = ["_key", "predicted", "realized_rate", "trials"]

    by_player = paired_abs_error_diff(a[cols_a], b[cols_b], id_col="_key", cluster_col="_player")
    by_cell = paired_abs_error_diff(a[cols_a], b[cols_b], id_col="_key", cluster_col="_cell")
    unclustered = paired_abs_error_diff(a[cols_a], b[cols_b], id_col="_key")

    per_cell = paired_by_cell(g, arm, base)
    wins = int((per_cell["diff"] < 0).sum())
    losses = int((per_cell["diff"] > 0).sum())

    # A clustered t computed from a single cluster is not a large number, it
    # is not a number: the between-cluster variance it divides by has no
    # degrees of freedom left, and floating point returns whatever the last
    # rounding error happened to be. Scoring one (season, cutoff) pair does
    # exactly this to the by-cell clustering, and the first run of this table
    # duly printed t = 2.3e15 next to an honest 1.68. Anything that reads as a
    # real statistic at a glance and is not one has to be suppressed at the
    # source, not formatted away.
    by_cell = _suppress_degenerate_t(by_cell)
    by_player = _suppress_degenerate_t(by_player)

    t_player = by_player["t"]
    ratio = (unclustered["t"] / t_player
            if np.isfinite(t_player) and t_player != 0 else float("nan"))

    return {
        "arm": arm, "base": base, "component": component,
        "diff": by_player["diff"], "n": by_player["n"],
        "clustered_by_player_se": by_player["se"], "clustered_by_player_t": t_player,
        "clustered_by_player_n_clusters": by_player["n_clusters"],
        "clustered_by_cell_se": by_cell["se"], "clustered_by_cell_t": by_cell["t"],
        "clustered_by_cell_n_clusters": by_cell["n_clusters"],
        "unclustered_se_WRONG": unclustered["se"], "unclustered_t_WRONG": unclustered["t"],
        "unclustered_over_player_clustered_t_ratio": ratio,
        "n_cells_scored": int(len(per_cell)),
        "arm_wins_cells": wins, "arm_loses_cells": losses,
    }


def build_variant_comparison_table(bayes: pd.DataFrame, component: str = "k_rate",
                                   bases: tuple[str, ...] = ("marcel_tuned", "marcel",
                                                             "bayes_flat", "bayes_walk",
                                                             CONTACT_ARM),
                                   ) -> pd.DataFrame:
    """`variant_comparison` for every scored variant against every base in
    `bases`, skipping a variant against itself (diff is identically zero and
    says nothing).

    `bayes_walk` and `contact_additive` joined the base list for BAS-83/84:
    a new arm has to be read against its own no-covariate, single-component
    twin (does the structure pay?) and against the served engine (does it beat
    what is on the board?), and a base absent from the checkpoint is skipped
    by `variant_comparison` returning `{}` rather than erroring."""
    present = set(bayes["model"].unique()) if not bayes.empty else set()
    arms = [a for a in VARIANT_ARM_NAMES.values() if a in present]
    rows = []
    for arm in arms:
        for base in bases:
            if arm == base:
                continue
            r = variant_comparison(bayes, arm, base, component)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def render_variant_table(df: pd.DataFrame) -> str:
    """Terminal-friendly rendering of `build_variant_comparison_table`'s
    output — the clustered-by-player t is the one to trust; the unclustered
    one is printed only so it's visible how wrong it would be to quote."""
    if df.empty:
        return "(no variant comparisons — bayes checkpoint has no scored variants)"
    # Widths come from the data, not from a guess: "bayes_walk_age" against
    # "marcel_tuned" ran the two columns together in the first rendering.
    arm_w = max([len(str(v)) for v in df["arm"]] + [len("arm")]) + 2
    base_w = max([len(str(v)) for v in df["base"]] + [len("base")]) + 2

    def num(v, width, places=2):
        """`nan` prints as a dash. A suppressed t is missing, not zero, and a
        table that renders it as a number invites someone to quote it.

        Handles `None` as well as `nan` because this table is rendered from
        the analysis payload after a JSON round-trip, and JSON has no NaN —
        `to_json` writes `null` and it comes back as `None`.
        """
        if v is None:
            return f"{'-':>{width}}"
        try:
            v = float(v)
        except (TypeError, ValueError):
            return f"{'-':>{width}}"
        return f"{'-':>{width}}" if not np.isfinite(v) else f"{v:>{width}.{places}f}"

    header = (f"{'arm':<{arm_w}}{'base':<{base_w}}{'n':>6}  {'cells':>5}  "
             f"{'diff':>9}  {'t(player)':>10}  {'t(cell)':>9}  "
             f"{'t(none) WRONG':>14}  {'ratio':>6}  {'W-L':>9}")
    lines = [header, "-" * len(header)]
    for _, r in df.iterrows():
        lines.append(
            f"{r['arm']:<{arm_w}}{r['base']:<{base_w}}{r['n']:>6}  "
            f"{r['n_cells_scored']:>5}  {r['diff']:>+9.5f}  "
            f"{num(r['clustered_by_player_t'], 10)}  "
            f"{num(r['clustered_by_cell_t'], 9)}  "
            f"{num(r['unclustered_t_WRONG'], 14)}  "
            f"{num(r['unclustered_over_player_clustered_t_ratio'], 6)}  "
            f"{r['arm_wins_cells']:>4d}-{r['arm_loses_cells']:<4d}"
        )
    return "\n".join(lines)


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
        bayes_components = sorted(bayes["component"].unique().tolist())
        payload["bayes_scope"] = {
            "seasons": sorted(int(s) for s in bayes["season"].unique()),
            "n_cutoffs": int(bayes.groupby("season")["cutoff"].nunique().sum()),
            "components": bayes_components,
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

        # BAS-69: every structural variant present in the checkpoint against
        # marcel_tuned, stock marcel, and the flat variant (its own control),
        # clustered by player and by cell, with the unclustered number
        # labelled as the wrong one to quote — see variant_comparison's
        # docstring.
        variant_table = build_variant_comparison_table(bayes)
        payload["bayes_variant_comparison"] = json.loads(
            variant_table.to_json(orient="records"))

        # BAS-73: the same table per component. Kept as its own key rather
        # than folded into the one above so the K% numbers every existing
        # reader quotes stay exactly where they were, under the same key,
        # while BB% and HR/PA arrive alongside them.
        per_component = {}
        for component in bayes_components:
            table = build_variant_comparison_table(bayes, component=component)
            if not table.empty:
                per_component[component] = json.loads(
                    table.to_json(orient="records"))
        payload["bayes_variant_comparison_by_component"] = per_component

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
    ap.add_argument("--bayes-cores", type=int, default=None,
                    help="sampler processes; defaults to --bayes-chains. Set 1 "
                         "to run the chains sequentially when this box is "
                         "shared with another job")
    ap.add_argument("--bayes-sampler", default="numpyro")
    ap.add_argument("--bayes-seasons", nargs="+", type=int, default=list(BAYES_SEASONS))
    ap.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS),
                    help="comma-separated bayes structural variants to sweep; "
                         f"known: {','.join(VARIANT_ARM_NAMES)} "
                         f"(aliases: {','.join(VARIANT_ALIASES)})")
    # Separate from --components, which scopes the *cheap* sweep across all
    # five hitter components. The bayes arm only serves the three per-PA
    # binomials, and its default stays k_rate alone so a command that names
    # neither runs exactly the sweep it always ran.
    ap.add_argument("--contact-arm", action="store_true",
                    help="also score `contact_additive` — Marcel plus the same "
                         "contact aggregates, the served shape — at every "
                         "bayes cell, so the covariate comparison is paired")
    ap.add_argument("--bayes-components", nargs="+",
                    default=list(DEFAULT_BAYES_COMPONENTS),
                    choices=list(BAYES_COMPONENTS),
                    help="components to fit the bayes arm for "
                         f"(default: {' '.join(DEFAULT_BAYES_COMPONENTS)})")
    args = ap.parse_args()

    # Aliases resolve here, before validation, so `--variants joint_walk` (the
    # spelling docs/bayes-joint.md uses) reaches the same config as
    # `joint+ability_walk` and everything downstream sees one vocabulary.
    variants = [resolve_variant(v.strip())
                for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANT_ARM_NAMES]
    if unknown:
        ap.error(f"unknown --variants {unknown}; known: "
                 f"{sorted(VARIANT_ARM_NAMES)} (aliases: "
                 f"{sorted(VARIANT_ALIASES)})")

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
        available = tuple(y for y in CHEAP_SEASONS
                          if (args.pa_dir / f"pa_outcomes_{y}.parquet").exists())
        pa_by_year = load_pa_by_year(
            tuple(sorted(set(available) | set(args.bayes_seasons))), args.pa_dir)
        monthly = None
        contact_fits: list[dict] = []
        if args.contact_arm:
            from src.data.contact_quality import load_monthly

            monthly = load_monthly()
        bayes, fits = run_bayes(seasons_table, pa_by_year, tuple(args.bayes_seasons),
                                BIWEEKLY_MMDD, args.min_trials, checkpoint=bayes_ckpt,
                                fits_path=fits_path, draws=args.bayes_draws,
                                tune=args.bayes_tune, chains=args.bayes_chains,
                                cores=args.bayes_cores,
                                sampler=args.bayes_sampler, pa_dir=args.pa_dir,
                                variants=variants,
                                components=args.bayes_components,
                                seasons_table_for_contact=seasons_table,
                                monthly=monthly,
                                contact_arm=args.contact_arm,
                                contact_fits=contact_fits)
        if contact_fits:
            (args.out_dir / "contact_additive_fits.json").write_text(
                json.dumps(contact_fits, indent=1))
        print(f"bayes sweep: {len(bayes)} rows, {len(fits)} fits, "
             f"components {args.bayes_components}, variants {variants} "
             f"-> {bayes_ckpt}")

    if args.stage in ("analyze", "all"):
        payload = build_analysis(cheap_ckpt, bayes_ckpt, analysis_path)
        print(f"analysis -> {analysis_path}")
        print(json.dumps({k: v for k, v in payload.items()
                          if k in ("cheap_scope", "bayes_scope", "common_set_sizes",
                                   "bayes_overall_clustered_vs_unclustered")},
                         indent=1))
        for component, table in payload.get(
                "bayes_variant_comparison_by_component", {}).items():
            print(f"\nvariant comparison ({component}, pooled; t(player) is "
                  f"the one to trust):")
            print(render_variant_table(pd.DataFrame(table)))


if __name__ == "__main__":
    main()
