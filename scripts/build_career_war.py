"""Build career WAR projections with uncertainty from Bayesian component models.

Uses Monte Carlo simulation to propagate uncertainty from the 5 component
rate posteriors (K%, BB%, HR rate, ISO, BABIP) through the assembly chain:
    rates → counting stats → slash line → wOBA → wRC+ → oWAR

Each player gets posterior draws of career WAR trajectories with percentile bands.

Output: public/data/career_war.json with structure:
{
  "592450": {
    "name": "Aaron Judge",
    "mlbam": 592450,
    "fg_id": 15640,
    "historical": [{"year": 2017, "age": 25, "war": 8.7, ...}, ...],
    "projected": [
      {"year": 2026, "age": 34, "war_p50": 6.2, "war_p10": 4.1, "war_p90": 8.3,
       "war_p25": 5.0, "war_p75": 7.4, "war_p5": 3.2, "war_p95": 9.1},
      ...
    ]
  }
}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
PROJ_DIR = BASE / "data" / "projections"
DATA_DIR = BASE / "data" / "parquet"
OUT_DIR = BASE / "public" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DRAWS = 2000  # Monte Carlo draws per player-year

# ─── 2024 wOBA constants (FanGraphs) ─────────────────────────────────────
W_BB = 0.690
W_HBP = 0.722
W_1B = 0.883
W_2B = 1.244
W_3B = 1.569
W_HR = 2.015
WOBA_SCALE = 1.185
LG_WOBA = 0.310
LG_R_PA = 0.116
RUNS_PER_WIN = 10.0
REPLACEMENT_PER_600 = 20.0

# Approximate league-average rates for fallback
HBP_RATE = 0.012
SF_RATE = 0.008


def load_component_projections():
    """Load all 5 component rate projections with uncertainty."""
    components = {}
    for stat, rate_col in [
        ("k_rate", "projected_k_rate"),
        ("bb_rate", "projected_bb_rate"),
        ("hr_rate", "projected_hr_rate"),
        ("iso", "projected_iso"),
        ("babip", "projected_babip"),
    ]:
        path = PROJ_DIR / f"{stat}_projections_2026.parquet"
        if not path.exists():
            print(f"❌ Missing: {path}")
            sys.exit(1)
        df = pd.read_parquet(str(path))
        components[stat] = df
        print(f"  {stat}: {len(df)} rows, {df['batter'].nunique()} players, "
              f"years {sorted(df['projection_year'].unique())}")
    return components


def draw_correlated_rates(player_rates, n_draws=N_DRAWS):
    """Generate Monte Carlo draws for a single player-year.

    Uses truncated normal draws for each component rate, respecting
    the posterior mean and std from the Bayesian models.

    For rates bounded in [0, 1] (K%, BB%, HR rate, BABIP), we draw in
    logit space and transform back to ensure valid rates.
    """
    draws = {}
    for stat in ["k_rate", "bb_rate", "hr_rate", "iso", "babip"]:
        mean = player_rates[f"{stat}_mean"]
        std = player_rates[f"{stat}_std"]

        if std <= 0 or np.isnan(std):
            draws[stat] = np.full(n_draws, mean)
            continue

        if stat == "iso":
            # ISO is on natural scale (Normal model), can be negative in theory
            raw = np.random.normal(mean, std, n_draws)
            draws[stat] = np.maximum(raw, 0.0)  # floor at 0
        else:
            # Rate stats: draw in logit space for proper bounds
            # logit(p) = log(p / (1-p))
            p = np.clip(mean, 0.001, 0.999)
            logit_mean = np.log(p / (1 - p))
            # Approximate logit-space std via delta method: std_logit ≈ std / (p*(1-p))
            logit_std = std / (p * (1 - p))
            logit_draws = np.random.normal(logit_mean, logit_std, n_draws)
            draws[stat] = 1.0 / (1.0 + np.exp(-logit_draws))

    return draws


def assemble_war_draws(draws, pa=550):
    """Convert Monte Carlo rate draws to WAR draws via the assembly chain.

    This replicates the assembly logic from modal_functions/app.py but
    operates on arrays of draws rather than point estimates.
    """
    k_rate = draws["k_rate"]
    bb_rate = draws["bb_rate"]
    hr_rate = draws["hr_rate"]
    iso = draws["iso"]
    babip = draws["babip"]

    # At-bats fraction
    ab_frac = 1.0 - bb_rate - HBP_RATE
    ab = pa * ab_frac

    # Counting stats
    hr = hr_rate * pa
    k = k_rate * pa
    sf = SF_RATE * pa
    bip = np.maximum(ab - k - hr + sf, 1)

    h_minus_hr = babip * bip
    h = h_minus_hr + hr

    # Slash line
    avg = h / np.maximum(ab, 1)
    slg = avg + iso

    bb = bb_rate * pa
    hbp = HBP_RATE * pa
    obp = (h + bb + hbp) / (ab + bb + hbp + sf)

    # Hit decomposition for wOBA
    xb = iso * ab
    hr_eb = 3.0 * hr
    non_hr_xb = np.maximum(xb - hr_eb, 0)
    triples = 0.12 * non_hr_xb / 1.12
    doubles = np.maximum(non_hr_xb - 2.0 * triples, 0)
    singles = np.maximum(h - hr - doubles - triples, 0)

    denom = ab + bb + hbp + sf
    woba = (W_BB * bb + W_HBP * hbp + W_1B * singles +
            W_2B * doubles + W_3B * triples + W_HR * hr) / np.maximum(denom, 1)

    # Value stats
    wraa = (woba - LG_WOBA) / WOBA_SCALE * pa
    wrc_plus = 100.0 * ((woba - LG_WOBA) / WOBA_SCALE + LG_R_PA) / LG_R_PA

    # oWAR (no positional adjustment — use default 0)
    replacement = REPLACEMENT_PER_600 * (pa / 600.0)
    owar = (wraa + replacement) / RUNS_PER_WIN

    return {
        "owar": owar,
        "woba": woba,
        "wrc_plus": wrc_plus,
        "avg": avg,
        "obp": obp,
        "slg": slg,
    }


def build_career_war():
    """Main pipeline: build career WAR with uncertainty for all players."""
    print("Loading component projections...")
    components = load_component_projections()

    # Load historical hitter seasons for actual WAR
    hs_path = DATA_DIR / "hitter_seasons.parquet"
    if not hs_path.exists():
        print(f"❌ Missing historical data: {hs_path}")
        sys.exit(1)
    hs = pd.read_parquet(str(hs_path))
    print(f"Historical seasons: {len(hs)} rows")

    # Load FG crosswalk for player names and fg_id
    fg_data = {}
    for sys_name in ["steamer", "zips"]:
        path = PROJ_DIR / f"fg_{sys_name}_bat_2026.json"
        if path.exists():
            with open(path) as f:
                fg_data[sys_name] = json.load(f)

    fg_to_mlbam = {}
    mlbam_to_fg = {}
    mlbam_to_name = {}
    for sys_name, players in fg_data.items():
        for p in players:
            fg_id = str(p.get("playerids", ""))
            mlbam = p.get("xMLBAMID")
            name = p.get("PlayerName", "")
            if fg_id and mlbam:
                try:
                    fg_to_mlbam[fg_id] = int(mlbam)
                    mlbam_to_fg[int(mlbam)] = int(fg_id)
                    mlbam_to_name[int(mlbam)] = name
                except (ValueError, TypeError):
                    continue  # skip non-numeric IDs like 'sa3022923'

    print(f"Crosswalk: {len(mlbam_to_fg)} MLBAM→FG mappings")

    # ── Merge component projections per batter-year ──────────────────────
    # Build a unified dataframe with mean + std for each rate
    k = components["k_rate"][["batter", "projection_year", "projected_age",
                               "projected_k_rate", "k_rate_std"]].rename(
        columns={"projected_k_rate": "k_rate_mean"})
    bb = components["bb_rate"][["batter", "projection_year",
                                 "projected_bb_rate", "bb_rate_std"]].rename(
        columns={"projected_bb_rate": "bb_rate_mean"})
    hr = components["hr_rate"][["batter", "projection_year",
                                 "projected_hr_rate", "hr_rate_std"]].rename(
        columns={"projected_hr_rate": "hr_rate_mean"})
    iso = components["iso"][["batter", "projection_year",
                              "projected_iso", "iso_std"]].rename(
        columns={"projected_iso": "iso_mean"})
    babip = components["babip"][["batter", "projection_year",
                                  "projected_babip", "babip_std"]].rename(
        columns={"projected_babip": "babip_mean"})

    merged = k
    for df in [bb, hr, iso, babip]:
        merged = merged.merge(df, on=["batter", "projection_year"], how="inner")

    print(f"Merged projections: {len(merged)} batter-years, "
          f"{merged['batter'].nunique()} unique batters")

    # ── Build career WAR for each player ─────────────────────────────────
    career_data = {}
    batters = merged["batter"].unique()
    n_processed = 0

    for mlbam in batters:
        mlbam = int(mlbam)
        fg_id = mlbam_to_fg.get(mlbam)
        if fg_id is None:
            continue

        # Historical seasons
        player_hist = hs[hs["fg_id"] == fg_id].sort_values("year")
        if len(player_hist) < 2:
            continue

        name = mlbam_to_name.get(mlbam, player_hist.iloc[0].get("name", f"Player {mlbam}"))

        historical = []
        for _, row in player_hist.iterrows():
            historical.append({
                "year": int(row["year"]),
                "age": int(row["age"]),
                "war": round(float(row["war"]), 1),
                "pa": int(row["pa"]),
                "team": row["team"],
                "woba": round(float(row["woba"]), 3) if pd.notna(row.get("woba")) else None,
                "wrc_plus": round(float(row["wrc_plus"]), 0) if pd.notna(row.get("wrc_plus")) else None,
                "off": round(float(row["off"]), 1) if pd.notna(row.get("off")) else None,
            })

        # Projected seasons with Monte Carlo uncertainty
        player_proj = merged[merged["batter"] == mlbam].sort_values("projection_year")
        projected = []

        for _, prow in player_proj.iterrows():
            rates = {
                "k_rate_mean": prow["k_rate_mean"],
                "k_rate_std": prow["k_rate_std"],
                "bb_rate_mean": prow["bb_rate_mean"],
                "bb_rate_std": prow["bb_rate_std"],
                "hr_rate_mean": prow["hr_rate_mean"],
                "hr_rate_std": prow["hr_rate_std"],
                "iso_mean": prow["iso_mean"],
                "iso_std": prow["iso_std"],
                "babip_mean": prow["babip_mean"],
                "babip_std": prow["babip_std"],
            }

            draws = draw_correlated_rates(rates)
            result = assemble_war_draws(draws, pa=550)
            war_draws = result["owar"]

            projected.append({
                "year": int(prow["projection_year"]),
                "age": int(prow["projected_age"]),
                "war_p50": round(float(np.percentile(war_draws, 50)), 1),
                "war_p10": round(float(np.percentile(war_draws, 10)), 1),
                "war_p90": round(float(np.percentile(war_draws, 90)), 1),
                "war_p25": round(float(np.percentile(war_draws, 25)), 1),
                "war_p75": round(float(np.percentile(war_draws, 75)), 1),
                "war_p5": round(float(np.percentile(war_draws, 5)), 1),
                "war_p95": round(float(np.percentile(war_draws, 95)), 1),
                # Also store component medians for tooltip detail
                "woba_p50": round(float(np.median(result["woba"])), 3),
                "wrc_plus_p50": round(float(np.median(result["wrc_plus"])), 0),
                "type": "projected",
            })

        career_data[mlbam] = {
            "name": name,
            "mlbam": mlbam,
            "fg_id": fg_id,
            "historical": historical,
            "projected": projected,
        }
        n_processed += 1

    print(f"\nCareer data: {n_processed} players")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "career_war.json"
    with open(str(out_path), "w") as f:
        json.dump(career_data, f)
    size = out_path.stat().st_size
    print(f"Saved: {out_path} ({size/1024:.0f}KB)")

    # ── Sanity checks ────────────────────────────────────────────────────
    judge = career_data.get(592450)
    if judge:
        print(f"\n{'='*60}")
        print(f"Aaron Judge career WAR:")
        for s in judge["historical"][-3:]:
            print(f"  {s['year']} (age {s['age']}): {s['war']} WAR actual")
        print("  --- projected (with uncertainty) ---")
        for s in judge["projected"]:
            print(f"  {s['year']} (age {s['age']}): "
                  f"p50={s['war_p50']} [{s['war_p10']}–{s['war_p90']}] WAR "
                  f"(wOBA={s['woba_p50']}, wRC+={s['wrc_plus_p50']})")

    soto = career_data.get(665742)
    if soto:
        print(f"\nJuan Soto career WAR:")
        for s in soto["historical"][-3:]:
            print(f"  {s['year']} (age {s['age']}): {s['war']} WAR actual")
        print("  --- projected ---")
        for s in soto["projected"]:
            print(f"  {s['year']} (age {s['age']}): "
                  f"p50={s['war_p50']} [{s['war_p10']}–{s['war_p90']}] WAR")

    return career_data


if __name__ == "__main__":
    np.random.seed(42)
    build_career_war()
