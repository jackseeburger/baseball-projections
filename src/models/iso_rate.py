"""
Batter-season Bayesian ISO (Isolated Power) model.

    ISO = (1x2B + 2x3B + 3xHR) / AB

Unlike K%/BABIP, ISO is a continuous rate rather than a per-PA binary
outcome, so the likelihood is Normal on the natural scale rather than
Binomial on the logit scale:

    ISO_obs ~ Normal(ISO_true, sigma_obs / sqrt(AB))

where `sigma_obs` is a free parameter for single-season measurement noise
and the `1/sqrt(AB)` term means a batter-season built on more at-bats gets
a tighter observation, the natural-scale analogue of a Binomial's own
variance shrinking with the trial count.

    ISO_true = league_trend[season] + player[batter] + hand + park
             + age_effect(age) + park_factor

Age enters through an HSGP (Hilbert Space Gaussian Process, Matern-5/2)
surface rather than a parametric curve — the model learns the shape from
data instead of assuming a quadratic.

**Provenance (issue #86).** This module is a same-day extraction of
`modal_functions/app.py::train_iso_model`, committed 2026-04-05 (`ee31c53`)
and unchanged since. Nothing in `src/` gated this model before the
extraction — there was no ISO component here to diverge from, which is
itself the Phase 1 finding: `train_iso_model` has no pre-existing `src/`
counterpart the way `pa_k_rate.py` does. The extraction changes nothing
about priors, the likelihood, the age model, data preparation, or the
batter-season aggregation; it only gives this code a home outside
`modal_functions/app.py` so Modal can import it instead of inlining a
second copy. The April 10, 2026 `bayes_preseason` ISO numbers published in
docs/accuracy-2026.md and docs/ros-projections.md came from exactly this
code (see docs/modal-src-divergence.md) — the extraction does not
regenerate or alter that file.

**Not gated.** Nothing in `src/eval/` scores this component against Marcel
the way `src/eval/bayes_arm.py` does for K% (BAS-59); there is no ISO
"fair fight." Whether an HSGP age surface beats a parametric one, or
whether this model should read a pitcher effect, are modelling questions
with their own gate (architecture.md #3) — out of scope here.

Designed for Modal deployment (8GB RAM, 4 CPU, NumPyro backend), same as
`pa_k_rate.py`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.pa_k_rate import model_diagnostics  # noqa: F401  (re-exported for callers)
from src.models.rate_hsgp import attach_calendar_age, gp_age_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REFERENCE_AGE = 27.0
LEAGUE_AVG_ISO = 0.150       # ~.145-.155 league average
MIN_AB_THRESHOLD = 50        # minimum career ABs to include a batter
MIN_SEASON_AB = 50           # minimum ABs within a season to keep the cell
MIN_SEASON_AB_FAST = 30      # relaxed floor for fast_mode smoke runs
PROJECTION_YEAR = 2026

# HSGP config: m = number of spectral basis functions, c = boundary factor.
# c=1.5 with REFERENCE_AGE=27 covers training ages through roughly 45,
# which is where projections extrapolate to.
HSGP_M = 20
HSGP_C = 1.5

SAMPLER_KWARGS = dict(
    draws=2000,
    tune=1500,
    chains=4,
    cores=1,
    target_accept=0.95,
    nuts_sampler="numpyro",
    random_seed=42,
    idata_kwargs={"log_likelihood": False},
)

NON_AB_EVENTS = [
    "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "sac_fly_double_play", "sac_bunt_double_play",
    "catcher_interf", "intent_walk",
]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading and preparation
# ═══════════════════════════════════════════════════════════════════════════

def load_ab_data(pa_dir: Path | str) -> pd.DataFrame:
    """Load PA parquet files and return the columns the ISO model needs."""
    import pyarrow.parquet as pq

    pa_dir = Path(pa_dir)
    parquet_files = sorted(pa_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pa_dir}")

    schema = pq.read_schema(parquet_files[0])
    keep_cols = ["batter", "game_year", "stand", "event",
                 "is_hr", "is_double", "is_triple",
                 "is_bb", "is_hbp",
                 "home_team", "away_team", "inning_topbot"]
    if "birth_year" in schema.names:
        keep_cols.append("birth_year")
    available = set(schema.names)
    keep_cols = [c for c in keep_cols if c in available]

    frames = [pd.read_parquet(f, columns=keep_cols) for f in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df):,} PAs ({df['game_year'].min()}-{df['game_year'].max()})")
    return df


def prepare_model_data(
    df: pd.DataFrame,
    park_factors: pd.DataFrame | None = None,
    min_ab: int = MIN_AB_THRESHOLD,
    fast_mode: bool = False,
    hitter_seasons: pd.DataFrame | None = None,
) -> dict:
    """Filter to at-bats, aggregate to batter-season, compute ISO cells.

    Filtering, exactly as it has run since April 2026:
      1. Drop non-AB events (walks, HBP, sac bunts/flies) to get at-bats.
      2. Drop likely pitchers: batters whose best single season is < 80 AB.
      3. Drop batters under `min_ab` career ABs.
      4. Aggregate to batter-season; drop seasons under the AB floor.

    Age comes from the PA row's own `birth_year` column when present, else
    a median estimate from `hitter_seasons.parquet`, else `first_year - 24`
    — the calendar-year convention this model has always used (not the
    June-30 seasonal-age convention `src.data.birthdates` provides; see
    docs/modal-src-divergence.md for why this was not "fixed" here).
    """
    event_col = "event" if "event" in df.columns else "events"
    df = df[df[event_col].notna() & (df[event_col] != "") & (df[event_col] != "0")].copy()
    n_pa_total = len(df)

    if "is_bb" in df.columns:
        df_ab = df[(df["is_bb"] == 0) & (df.get("is_hbp", pd.Series(0)) == 0)].copy()
        df_ab = df_ab[~df_ab[event_col].isin(NON_AB_EVENTS)].copy()
    else:
        df_ab = df[~df[event_col].isin(NON_AB_EVENTS)].copy()
    logger.info(f"Filtered to {len(df_ab):,} ABs from {n_pa_total:,} PAs")

    if "is_double" in df_ab.columns and "is_triple" in df_ab.columns:
        df_ab["extra_bases"] = (
            df_ab["is_double"].astype(int) * 1
            + df_ab["is_triple"].astype(int) * 2
            + df_ab["is_hr"].astype(int) * 3
        )
    else:
        df_ab["extra_bases"] = (
            (df_ab[event_col] == "double").astype(int) * 1
            + (df_ab[event_col] == "triple").astype(int) * 2
            + (df_ab[event_col].isin(["home_run"])).astype(int) * 3
        )

    df_ab["bat_team"] = np.where(
        df_ab["inning_topbot"] == "Top", df_ab["away_team"], df_ab["home_team"]
    )

    n_before = df_ab["batter"].nunique()
    season_ab = df_ab.groupby(["batter", "game_year"]).size().reset_index(name="season_ab")
    max_season_ab = season_ab.groupby("batter")["season_ab"].max()
    likely_hitters = max_season_ab[max_season_ab >= 80].index
    n_pitchers_removed = n_before - len(likely_hitters)
    df_ab = df_ab[df_ab["batter"].isin(likely_hitters)].copy()

    ab_counts = df_ab.groupby("batter").size()
    qualified = ab_counts[ab_counts >= min_ab].index
    df_ab = df_ab[df_ab["batter"].isin(qualified)].copy()
    logger.info(f"Batters: {n_before} -> {df_ab['batter'].nunique()} "
                f"(removed {n_pitchers_removed} likely pitchers, >= {min_ab} career AB)")

    seasons = np.sort(df_ab["game_year"].unique())
    season_map = {yr: i for i, yr in enumerate(seasons)}
    df_ab["season_idx"] = df_ab["game_year"].map(season_map).astype(np.int64)

    batters = np.sort(df_ab["batter"].unique())
    batter_map = {b: i for i, b in enumerate(batters)}
    df_ab["batter_idx"] = df_ab["batter"].map(batter_map).astype(np.int64)

    teams = np.sort(df_ab["bat_team"].unique())
    team_map = {t: i for i, t in enumerate(teams)}
    df_ab["team_idx"] = df_ab["bat_team"].map(team_map).astype(np.int64)

    df_ab["stand_idx"] = (df_ab["stand"] == "R").astype(np.int64)

    attach_calendar_age(df_ab, "batter", "game_year", hitter_seasons=hitter_seasons)

    n_teams = len(teams)
    n_seasons = len(seasons)
    log_pf = np.zeros((n_teams, n_seasons), dtype=np.float64)
    if park_factors is not None:
        yr_col = "game_year" if "game_year" in park_factors.columns else "year"
        # ISO's own factor when the table carries one (BAS-86 does), the HR
        # factor otherwise: ISO is extra-base points per AB, so a park's HR
        # effect is the biggest part of it but not the whole of it — the
        # doubles a deep gap gives back are in `iso_park_factor` and not in
        # `hr_park_factor`.
        pf_col = next(
            (c for c in park_factors.columns if c.lower() == "iso_park_factor"),
            next((c for c in park_factors.columns if c.lower() in ("pf_hr", "hr_park_factor")),
                 next((c for c in park_factors.columns
                       if "hr" in c.lower() and ("park" in c.lower() or "pf" in c.lower() or "factor" in c.lower())),
                      None),
                 ),
        )
        if pf_col:
            for _, row in park_factors.iterrows():
                t = team_map.get(row["team"])
                s = season_map.get(int(row[yr_col]))
                if t is not None and s is not None:
                    log_pf[t, s] = np.log(float(row[pf_col]))
    df_ab["log_pf_hr"] = log_pf[df_ab["team_idx"].values, df_ab["season_idx"].values].astype(np.float64)

    batter_meta = (
        df_ab.groupby("batter")
        .agg(stand=("stand", "first"), birth_year=("birth_year", "first"),
             last_season=("game_year", "max"), total_ab=("extra_bases", "size"),
             career_iso=("extra_bases", lambda x: x.sum() / len(x)))
        .reset_index()
    )

    team_mode = (
        df_ab.groupby(["batter", "game_year", "team_idx"])
        .size().reset_index(name="ab_count")
        .sort_values("ab_count", ascending=False)
        .drop_duplicates(subset=["batter", "game_year"], keep="first")
        .set_index(["batter", "game_year"])["team_idx"]
    )

    agg_df = (
        df_ab.groupby(["batter", "game_year"])
        .agg(
            total_extra_bases=("extra_bases", "sum"),
            total_ab=("extra_bases", "size"),
            stand_idx=("stand_idx", "first"),
            batter_idx=("batter_idx", "first"),
            season_idx=("season_idx", "first"),
            age=("age", "first"),
            age_centered=("age_centered", "first"),
            log_pf_hr=("log_pf_hr", "mean"),
        )
        .reset_index()
    )
    agg_df["iso_obs"] = agg_df["total_extra_bases"] / agg_df["total_ab"]

    min_season_ab = MIN_SEASON_AB_FAST if fast_mode else MIN_SEASON_AB
    agg_df = agg_df[agg_df["total_ab"] >= min_season_ab].copy()

    team_mode_df = team_mode.reset_index()
    team_mode_df.columns = ["batter", "game_year", "team_idx"]
    agg_df = agg_df.merge(team_mode_df, on=["batter", "game_year"], how="left",
                           suffixes=("_drop", ""))
    if "team_idx_drop" in agg_df.columns:
        agg_df = agg_df.drop(columns=["team_idx_drop"])
    agg_df["team_idx"] = agg_df["team_idx"].astype(np.int64)

    remaining_batters = np.sort(agg_df["batter"].unique())
    new_batter_map = {b: i for i, b in enumerate(remaining_batters)}
    agg_df["batter_idx"] = agg_df["batter"].map(new_batter_map).astype(np.int64)

    n_obs = len(agg_df)
    n_total_ab = int(agg_df["total_ab"].sum())
    logger.info(f"Aggregated to {n_obs:,} batter-seasons ({n_total_ab:,} total ABs, "
                f"min {min_season_ab} AB/season)")

    return {
        "n_obs": n_obs,
        "n_batters": len(remaining_batters),
        "n_seasons": n_seasons,
        "n_teams": n_teams,
        "df": agg_df,
        "seasons": seasons,
        "batters": remaining_batters,
        "teams": teams,
        "batter_map": new_batter_map,
        "team_map": team_map,
        "season_map": season_map,
        "batter_meta": batter_meta[batter_meta["batter"].isin(remaining_batters)].copy(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PyMC model (HSGP age curve, Normal likelihood on the natural scale)
# ═══════════════════════════════════════════════════════════════════════════

def build_model(data: dict) -> pm.Model:
    agg_df = data["df"]
    coords = {
        "batter": data["batters"],
        "season": data["seasons"],
        "team": data["teams"],
        "obs_id": np.arange(data["n_obs"]),
    }
    age_centered_vals = agg_df["age_centered"].values.astype(np.float64)
    obs_weights = (1.0 / np.sqrt(agg_df["total_ab"].values)).astype(np.float64)

    with pm.Model(coords=coords) as model:
        batter_idx_d = pm.Data("batter_idx", agg_df["batter_idx"].values, dims="obs_id")
        season_idx_d = pm.Data("season_idx", agg_df["season_idx"].values, dims="obs_id")
        team_idx_d = pm.Data("team_idx", agg_df["team_idx"].values, dims="obs_id")
        stand_idx_d = pm.Data("stand_idx", agg_df["stand_idx"].values, dims="obs_id")
        log_pf_d = pm.Data("log_pf_hr", agg_df["log_pf_hr"].values, dims="obs_id")
        obs_weight_d = pm.Data("obs_weights", obs_weights, dims="obs_id")
        age_c_d = pm.Data("age_centered", age_centered_vals, dims="obs_id")

        league_init = pm.Normal("league_init", mu=LEAGUE_AVG_ISO, sigma=0.03)
        league_innovations = pm.Normal("league_innovations", mu=0, sigma=0.005, dims="season")
        league_trend = pm.Deterministic(
            "league_trend", league_init + pt.cumsum(league_innovations), dims="season")

        mu_ability = pm.Normal("mu_ability", mu=0.0, sigma=0.03)
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=0.06)
        z_ability = pm.Normal("z_ability", mu=0, sigma=1, dims="batter")
        player_ability = pm.Deterministic(
            "player_ability", mu_ability + sigma_ability * z_ability, dims="batter")

        beta_hand = pm.Normal("beta_hand", mu=0.0, sigma=0.02)
        park_effect = pm.ZeroSumNormal("park_effect", sigma=0.01, dims="team")

        eta_age = pm.HalfNormal("eta_age", sigma=0.03)
        ell_age = pm.InverseGamma("ell_age", mu=5.0, sigma=2.0)
        cov_age = eta_age**2 * pm.gp.cov.Matern52(1, ls=ell_age)
        gp_age = pm.gp.HSGP(m=[HSGP_M], c=HSGP_C, cov_func=cov_age)
        age_effect = gp_age.prior("age_effect", X=age_c_d[:, None])

        mu_iso = (
            league_trend[season_idx_d]
            + player_ability[batter_idx_d]
            + beta_hand * stand_idx_d
            + park_effect[team_idx_d]
            + age_effect
            + LEAGUE_AVG_ISO * log_pf_d
        )

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.15)
        pm.Normal("obs_iso", mu=mu_iso, sigma=sigma_obs * obs_weight_d,
                  observed=agg_df["iso_obs"].values, dims="obs_id")

    logger.info(f"Model built: {data['n_obs']:,} obs, {data['n_batters']:,} batters, "
                f"{data['n_seasons']} seasons, {data['n_teams']} teams")
    return model


def sample_model(model: pm.Model, **sampler_overrides) -> az.InferenceData:
    kwargs = {**SAMPLER_KWARGS, **sampler_overrides}
    logger.info(f"Sampling: {kwargs['chains']} chains x {kwargs['draws']} draws "
                f"(tune={kwargs['tune']})")
    t0 = time.time()
    with model:
        trace = pm.sample(**kwargs)
    logger.info(f"Sampling done in {time.time() - t0:.0f}s")
    return trace


# ═══════════════════════════════════════════════════════════════════════════
# Projections (multi-year, HSGP age effect via posterior interpolation)
# ═══════════════════════════════════════════════════════════════════════════

def generate_projections(
    trace: az.InferenceData,
    data: dict,
    projection_year: int = PROJECTION_YEAR,
    n_years: int = 5,
) -> pd.DataFrame:
    """Multi-year ISO projections from the posterior, `n_years` seasons out."""
    post = trace.posterior
    obs_age_values = data["df"]["age"].values
    eval_gp_age_effect, ns = gp_age_evaluator(trace, obs_age_values)

    lt = post["league_trend"].values
    pa_vals = post["player_ability"].values
    bh = post["beta_hand"].values
    innov = post["league_innovations"].values

    lt_flat = lt.reshape(ns, -1)
    pa_flat = pa_vals.reshape(ns, -1)
    bh_flat = bh.reshape(ns)
    innov_flat = innov.reshape(ns, -1)

    last_trend = lt_flat[:, -1]
    innov_std = innov_flat.std(axis=1)

    seasons = data["seasons"]
    batter_map = data["batter_map"]
    cutoff_year = int(seasons[-1]) - 2
    active = data["batter_meta"][data["batter_meta"]["last_season"] >= cutoff_year].copy()
    active = active[active["batter"].isin(data["batters"])].copy()
    logger.info(f"Projecting {len(active)} batters active since {cutoff_year}")

    rows = []
    for proj_year in range(projection_year, projection_year + n_years):
        years_ahead = proj_year - int(seasons[-1])
        projected_trend = last_trend.copy()
        rng_year = np.random.default_rng(42 + proj_year)
        for _ in range(years_ahead):
            projected_trend = projected_trend + rng_year.normal(0, innov_std)

        for _, row in active.iterrows():
            batter_id = int(row["batter"])
            if batter_id not in batter_map:
                continue
            b_idx = batter_map[batter_id]
            proj_age = proj_year - float(row["birth_year"])
            s_idx = 1 if row["stand"] == "R" else 0
            age_eff = eval_gp_age_effect(np.array([proj_age])).squeeze()

            iso_proj = projected_trend + pa_flat[:, b_idx] + bh_flat * s_idx + age_eff
            iso_proj = np.clip(iso_proj, 0.0, 0.6)

            rows.append({
                "batter": batter_id,
                "projection_year": proj_year,
                "projected_age": proj_age,
                "stand": row["stand"],
                "projected_iso": float(np.mean(iso_proj)),
                "iso_std": float(np.std(iso_proj)),
                "iso_lower": float(np.percentile(iso_proj, 5)),
                "iso_upper": float(np.percentile(iso_proj, 95)),
                "iso_10": float(np.percentile(iso_proj, 10)),
                "iso_90": float(np.percentile(iso_proj, 90)),
                "posterior_mean_ability": float(np.mean(pa_flat[:, b_idx])),
                "total_ab": int(row["total_ab"]),
                "career_iso": float(row["career_iso"]),
                "last_season": int(row["last_season"]),
            })

    proj_df = pd.DataFrame(rows).sort_values(["batter", "projection_year"]).reset_index(drop=True)
    logger.info(f"Projections generated: {len(proj_df)} rows")
    return proj_df


def compute_aging_curve(trace: az.InferenceData, data: dict) -> pd.DataFrame:
    """Age grid 20-42 with the posterior mean/5th/95th age effect, for logging."""
    obs_age_values = data["df"]["age"].values
    eval_gp_age_effect, _ = gp_age_evaluator(trace, obs_age_values)
    age_grid = np.linspace(20, 42, 100)
    curves = eval_gp_age_effect(age_grid)
    return pd.DataFrame({
        "age": age_grid,
        "age_effect_mean": curves.mean(axis=1),
        "age_effect_lower": np.percentile(curves, 5, axis=1),
        "age_effect_upper": np.percentile(curves, 95, axis=1),
    })
