"""
Batter-season Bayesian BABIP (Batting Average on Balls in Play) model.

    BABIP = (H - HR) / (AB - K - HR + SF)

Binomial on batter-season cells, logit link:

    logit(p_BABIP) = league_trend[season] + player[batter] + hand + park
                    + age_effect(age) + log(park_factor)

MLB average BABIP ~.300 (logit ~ -0.847). Age enters through an HSGP
(Hilbert Space Gaussian Process, Matern-5/2) surface, same construction as
`iso_rate.py` and the Modal K% model this repo used to run.

**Provenance (issue #86).** Extracted from
`modal_functions/app.py::train_babip_model`, committed 2026-04-05
(`54af327`) and unchanged since, with no change to priors, likelihood, age
model, or data preparation. As with `iso_rate.py`, there was no `src/`
BABIP model before this extraction to diverge from — the extraction exists
to give Modal something to import instead of a second inlined copy, not to
correct anything. The April 10, 2026 `bayes_preseason` BABIP numbers in
docs/ros-projections.md came from this code; see
docs/modal-src-divergence.md.

**Not gated.** BABIP is the one component the harness has found no
information advantage in even walk-forward (docs/ros-projections.md); this
module exists to be called, not to win an argument about whether it should
be. Modelling changes (a different age curve, a pitcher effect) have their
own gate per architecture.md #3.
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

from src.models.iso_rate import HSGP_C, HSGP_M
from src.models.pa_k_rate import model_diagnostics  # noqa: F401  (re-exported for callers)
from src.models.rate_hsgp import REFERENCE_AGE, attach_calendar_age, gp_age_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LEAGUE_INIT_LOGIT = -0.847   # BABIP ~.300 on the logit scale
MIN_AB_THRESHOLD = 50        # minimum career balls in play to include a batter
MIN_SEASON_BIP = 50
MIN_SEASON_BIP_FAST = 30
PROJECTION_YEAR = 2026

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

NON_BIP_EVENTS = [
    "strikeout", "strikeout_double_play", "walk",
    "hit_by_pitch", "sac_bunt", "sac_bunt_double_play",
    "catcher_interf", "intent_walk", "home_run",
]
HIT_EVENTS = ["single", "double", "triple"]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading and preparation
# ═══════════════════════════════════════════════════════════════════════════

def load_bip_data(pa_dir: Path | str) -> pd.DataFrame:
    """Load PA parquet files and return the columns the BABIP model needs."""
    import pyarrow.parquet as pq

    pa_dir = Path(pa_dir)
    parquet_files = sorted(pa_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pa_dir}")

    schema = pq.read_schema(parquet_files[0])
    keep_cols = ["batter", "game_year", "stand", "event",
                 "is_hit", "is_hr", "is_k", "is_in_play",
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
    """Filter to balls in play (excl. HR), aggregate to batter-season BABIP cells."""
    event_col = "event" if "event" in df.columns else "events"
    df = df[df[event_col].notna() & (df[event_col] != "") & (df[event_col] != "0")].copy()
    n_total = len(df)

    if "is_in_play" in df.columns:
        df["is_bip_no_hr"] = ((df["is_in_play"] == 1) & (df["is_hr"] == 0)).astype(np.int8)
        df["is_hit_no_hr"] = ((df["is_hit"] == 1) & (df["is_hr"] == 0)).astype(np.int8)
    else:
        df["is_bip_no_hr"] = (~df[event_col].isin(NON_BIP_EVENTS)).astype(np.int8)
        df["is_hit_no_hr"] = df[event_col].isin(HIT_EVENTS).astype(np.int8)

    df_bip = df[df["is_bip_no_hr"] == 1].copy()
    logger.info(f"Balls in play (excl HR): {len(df_bip):,} from {n_total:,} PAs")

    df_bip["bat_team"] = np.where(
        df_bip["inning_topbot"] == "Top", df_bip["away_team"], df_bip["home_team"]
    )

    n_before = df_bip["batter"].nunique()
    season_pa = df.groupby(["batter", "game_year"]).size().reset_index(name="season_pa")
    max_season_pa = season_pa.groupby("batter")["season_pa"].max()
    likely_hitters = max_season_pa[max_season_pa >= 80].index
    n_pitchers_removed = n_before - len(likely_hitters)
    df_bip = df_bip[df_bip["batter"].isin(likely_hitters)].copy()

    bip_counts = df_bip.groupby("batter").size()
    qualified = bip_counts[bip_counts >= min_ab].index
    df_bip = df_bip[df_bip["batter"].isin(qualified)].copy()
    logger.info(f"Batters: {n_before} -> {df_bip['batter'].nunique()} "
                f"(removed {n_pitchers_removed} likely pitchers, >= {min_ab} career BIP)")

    seasons = np.sort(df_bip["game_year"].unique())
    season_map = {yr: i for i, yr in enumerate(seasons)}
    df_bip["season_idx"] = df_bip["game_year"].map(season_map).astype(np.int64)

    batters = np.sort(df_bip["batter"].unique())
    batter_map = {b: i for i, b in enumerate(batters)}
    df_bip["batter_idx"] = df_bip["batter"].map(batter_map).astype(np.int64)

    teams = np.sort(df_bip["bat_team"].unique())
    team_map = {t: i for i, t in enumerate(teams)}
    df_bip["team_idx"] = df_bip["bat_team"].map(team_map).astype(np.int64)

    df_bip["stand_idx"] = (df_bip["stand"] == "R").astype(np.int64)

    attach_calendar_age(df_bip, "batter", "game_year", hitter_seasons=hitter_seasons)

    n_teams = len(teams)
    n_seasons = len(seasons)
    log_pf = np.zeros((n_teams, n_seasons), dtype=np.float64)
    if park_factors is not None:
        yr_col = "game_year" if "game_year" in park_factors.columns else "year"
        pf_col = next(
            (c for c in park_factors.columns if c.lower() in ("pf_babip", "babip_park_factor")),
            next((c for c in park_factors.columns
                  if c.lower() in ("pf_h", "h_park_factor", "hit_park_factor")), None),
        )
        if pf_col:
            for _, row in park_factors.iterrows():
                t = team_map.get(row["team"])
                s = season_map.get(int(row[yr_col]))
                if t is not None and s is not None:
                    log_pf[t, s] = np.log(float(row[pf_col]))
    df_bip["log_pf_babip"] = log_pf[df_bip["team_idx"].values, df_bip["season_idx"].values].astype(np.float64)

    batter_meta = (
        df_bip.groupby("batter")
        .agg(stand=("stand", "first"), birth_year=("birth_year", "first"),
             last_season=("game_year", "max"), total_bip=("is_bip_no_hr", "size"),
             career_babip=("is_hit_no_hr", "mean"))
        .reset_index()
    )

    team_mode = (
        df_bip.groupby(["batter", "game_year", "team_idx"])
        .size().reset_index(name="bip_count")
        .sort_values("bip_count", ascending=False)
        .drop_duplicates(subset=["batter", "game_year"], keep="first")
        .set_index(["batter", "game_year"])["team_idx"]
    )

    agg_df = (
        df_bip.groupby(["batter", "game_year"])
        .agg(
            n_hits_bip=("is_hit_no_hr", "sum"),
            n_bip=("is_bip_no_hr", "size"),
            stand_idx=("stand_idx", "first"),
            batter_idx=("batter_idx", "first"),
            season_idx=("season_idx", "first"),
            age=("age", "first"),
            age_centered=("age_centered", "first"),
            log_pf_babip=("log_pf_babip", "mean"),
        )
        .reset_index()
    )

    min_season_bip = MIN_SEASON_BIP_FAST if fast_mode else MIN_SEASON_BIP
    agg_df = agg_df[agg_df["n_bip"] >= min_season_bip].copy()

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

    agg_df["n_hits_bip"] = agg_df["n_hits_bip"].astype(np.int64)
    agg_df["n_bip"] = agg_df["n_bip"].astype(np.int64)

    n_obs = len(agg_df)
    n_total_bip = int(agg_df["n_bip"].sum())
    logger.info(f"Aggregated to {n_obs:,} batter-seasons ({n_total_bip:,} total BIP, "
                f"min {min_season_bip} BIP/season)")

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
# PyMC model (HSGP age curve, Binomial likelihood, logit link)
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

    with pm.Model(coords=coords) as model:
        batter_idx_d = pm.Data("batter_idx", agg_df["batter_idx"].values, dims="obs_id")
        season_idx_d = pm.Data("season_idx", agg_df["season_idx"].values, dims="obs_id")
        team_idx_d = pm.Data("team_idx", agg_df["team_idx"].values, dims="obs_id")
        stand_idx_d = pm.Data("stand_idx", agg_df["stand_idx"].values, dims="obs_id")
        log_pf_d = pm.Data("log_pf_babip", agg_df["log_pf_babip"].values, dims="obs_id")
        n_bip_d = pm.Data("n_bip", agg_df["n_bip"].values, dims="obs_id")
        age_c_d = pm.Data("age_centered", age_centered_vals, dims="obs_id")

        league_init = pm.Normal("league_init", mu=LEAGUE_INIT_LOGIT, sigma=0.3)
        league_innovations = pm.Normal("league_innovations", mu=0, sigma=0.05, dims="season")
        league_trend = pm.Deterministic(
            "league_trend", league_init + pt.cumsum(league_innovations), dims="season")

        mu_ability = pm.Normal("mu_ability", mu=0.0, sigma=0.3)
        sigma_ability = pm.HalfNormal("sigma_ability", sigma=0.3)
        z_ability = pm.Normal("z_ability", mu=0, sigma=1, dims="batter")
        player_ability = pm.Deterministic(
            "player_ability", mu_ability + sigma_ability * z_ability, dims="batter")

        beta_hand = pm.Normal("beta_hand", mu=0.0, sigma=0.2)
        park_effect = pm.ZeroSumNormal("park_effect", sigma=0.05, dims="team")

        eta_age = pm.HalfNormal("eta_age", sigma=0.3)
        ell_age = pm.InverseGamma("ell_age", mu=5.0, sigma=2.0)
        cov_age = eta_age**2 * pm.gp.cov.Matern52(1, ls=ell_age)
        gp_age = pm.gp.HSGP(m=[HSGP_M], c=HSGP_C, cov_func=cov_age)
        age_effect = gp_age.prior("age_effect", X=age_c_d[:, None])

        eta = (
            league_trend[season_idx_d]
            + player_ability[batter_idx_d]
            + beta_hand * stand_idx_d
            + park_effect[team_idx_d]
            + age_effect
            + log_pf_d
        )

        p = pm.math.invlogit(eta)
        pm.Binomial("obs_babip", n=n_bip_d, p=p, observed=agg_df["n_hits_bip"].values, dims="obs_id")

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
# Projections
# ═══════════════════════════════════════════════════════════════════════════

def generate_projections(
    trace: az.InferenceData,
    data: dict,
    projection_year: int = PROJECTION_YEAR,
    n_years: int = 5,
) -> pd.DataFrame:
    """Multi-year BABIP projections from the posterior, `n_years` seasons out."""
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

            eta_proj = projected_trend + pa_flat[:, b_idx] + bh_flat * s_idx + age_eff
            p_babip = 1.0 / (1.0 + np.exp(-eta_proj))

            rows.append({
                "batter": batter_id,
                "projection_year": proj_year,
                "projected_age": proj_age,
                "stand": row["stand"],
                "projected_babip": float(np.mean(p_babip)),
                "babip_std": float(np.std(p_babip)),
                "babip_lower": float(np.percentile(p_babip, 5)),
                "babip_upper": float(np.percentile(p_babip, 95)),
                "babip_10": float(np.percentile(p_babip, 10)),
                "babip_90": float(np.percentile(p_babip, 90)),
                "posterior_mean_ability": float(np.mean(pa_flat[:, b_idx])),
                "total_bip": int(row["total_bip"]),
                "career_babip": float(row["career_babip"]),
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
