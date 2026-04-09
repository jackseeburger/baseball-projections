#!/usr/bin/env python3
"""Assemble our model projections and create comparison dataset with FanGraphs systems."""

import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJ_DIR = Path("/home/hermes/projects/baseball-assembly/data/projections")
FG_DIR = Path("/home/hermes/projects/baseball-dashboard/data/projections")
OUT_DIR = FG_DIR

# ══════════════════════════════════════════════════════════════════════
# 1. Load and assemble our component projections
# ══════════════════════════════════════════════════════════════════════
print("=== Loading our component projections ===")
components = {}
for stat, rate_col in [
    ("k_rate", "projected_k_rate"),
    ("bb_rate", "projected_bb_rate"),
    ("hr_rate", "projected_hr_rate"),
    ("iso", "projected_iso"),
    ("babip", "projected_babip"),
]:
    df = pd.read_parquet(str(PROJ_DIR / f"{stat}_projections_2026.parquet"))
    components[stat] = df

# Merge all components
merged = components["k_rate"][
    ["batter", "projection_year", "projected_age", "stand",
     "projected_k_rate", "k_rate_std", "total_pa", "last_season"]
].copy()
merged = merged.rename(columns={"projected_k_rate": "k_rate"})

for stat, rate_col, std_col in [
    ("bb_rate", "projected_bb_rate", "bb_rate_std"),
    ("hr_rate", "projected_hr_rate", "hr_rate_std"),
    ("iso", "projected_iso", "iso_std"),
    ("babip", "projected_babip", "babip_std"),
]:
    df = components[stat][["batter", "projection_year", rate_col, std_col]]
    df = df.rename(columns={rate_col: stat, std_col: f"{stat}_std"})
    merged = merged.merge(df, on=["batter", "projection_year"], how="inner")

# Filter to 2026 + recently active
m2026 = merged[(merged["projection_year"] == 2026) & (merged["last_season"] >= 2024)].copy()
print(f"Active hitters for 2026: {len(m2026)}")

# Assembly: component rates -> slash line -> value stats
pa = 550
hbp_rate = 0.012
sf_rate = 0.008
ab_frac = 1.0 - m2026["bb_rate"] - hbp_rate
ab = pa * ab_frac
hr = m2026["hr_rate"] * pa
k = m2026["k_rate"] * pa
bb = m2026["bb_rate"] * pa
hbp = hbp_rate * pa
sf = sf_rate * pa
bip = np.maximum(ab - k - hr + sf, 1)
h_minus_hr = m2026["babip"] * bip
h = h_minus_hr + hr
avg = h / np.maximum(ab, 1)
slg = avg + m2026["iso"]
obp = (h + bb + hbp) / (ab + bb + hbp + sf)

# wOBA via FanGraphs linear weights (2024)
W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.690, 0.722, 0.883, 1.244, 1.569, 2.015
WOBA_SCALE, LG_WOBA, LG_R_PA, RUNS_PER_WIN = 1.185, 0.310, 0.116, 10.0

xb = m2026["iso"] * ab
non_hr_xb = np.maximum(xb - 3.0 * hr, 0)
triples = 0.12 * non_hr_xb / 1.12
doubles = np.maximum(non_hr_xb - 2.0 * triples, 0)
singles = np.maximum(h - hr - doubles - triples, 0)
denom = ab + bb + hbp + sf
woba = (W_BB*bb + W_HBP*hbp + W_1B*singles + W_2B*doubles + W_3B*triples + W_HR*hr) / np.maximum(denom, 1)
wraa = (woba - LG_WOBA) / WOBA_SCALE * pa
wrc_plus = 100.0 * ((woba - LG_WOBA) / WOBA_SCALE + LG_R_PA) / LG_R_PA

m2026["avg"] = avg
m2026["obp"] = obp
m2026["slg"] = slg
m2026["woba"] = woba
m2026["wraa"] = wraa
m2026["wrc_plus"] = wrc_plus
m2026["off"] = wraa  # offensive runs = wRAA
m2026["hr_count"] = hr
m2026["pa"] = pa

m2026 = m2026.sort_values("woba", ascending=False).reset_index(drop=True)
m2026.to_parquet(str(OUT_DIR / "our_model_2026.parquet"), index=False)
print(f"Saved our_model_2026.parquet ({len(m2026)} players)")

# ══════════════════════════════════════════════════════════════════════
# 2. Load FanGraphs projections and normalize
# ══════════════════════════════════════════════════════════════════════
print("\n=== Loading FanGraphs projections ===")

fg_systems = {}
for system in ["steamer", "zips", "depthcharts"]:
    with open(str(FG_DIR / f"fg_{system}_bat_2026.json")) as f:
        raw = json.load(f)
    
    df = pd.DataFrame(raw)
    # Normalize column names
    df = df.rename(columns={
        "xMLBAMID": "batter",
        "PlayerName": "name",
        "playerids": "fg_id",
        "K%": "k_rate",
        "BB%": "bb_rate",
    })
    
    # Filter to qualified batters (PA >= 200)
    df = df[df["PA"] >= 200].copy()
    
    # Compute HR rate (HR per PA) to match our model
    df["hr_rate"] = df["HR"] / df["PA"]
    
    fg_systems[system] = df
    print(f"  {system}: {len(df)} qualified hitters (PA >= 200)")

# ══════════════════════════════════════════════════════════════════════
# 3. Build comparison dataset
# ══════════════════════════════════════════════════════════════════════
print("\n=== Building comparison dataset ===")

# Our model uses MLBAM IDs as 'batter' — same as FanGraphs xMLBAMID
our = m2026[["batter", "projected_age", "stand", "k_rate", "bb_rate", "hr_rate",
             "iso", "babip", "avg", "obp", "slg", "woba", "wraa", "wrc_plus",
             "off", "hr_count", "pa",
             "k_rate_std", "bb_rate_std", "hr_rate_std", "iso_std", "babip_std"]].copy()

# Build name crosswalk from FanGraphs data
name_map = {}
for sys_name, df in fg_systems.items():
    for _, row in df.iterrows():
        if row["batter"] not in name_map:
            name_map[int(row["batter"])] = {
                "name": row["name"],
                "fg_id": row["fg_id"],
                "team": row["Team"],
            }

# Add names to our projections
our["name"] = our["batter"].map(lambda x: name_map.get(int(x), {}).get("name", f"MLBAM-{int(x)}"))
our["team"] = our["batter"].map(lambda x: name_map.get(int(x), {}).get("team", "???"))

# Create unified comparison table
compare_stats = ["k_rate", "bb_rate", "hr_rate", "ISO", "BABIP", "AVG", "OBP", "SLG", "wOBA", "wRC+", "Off"]

records = []
for _, row in our.iterrows():
    mlbam = int(row["batter"])
    rec = {
        "batter": mlbam,
        "name": row["name"],
        "team": row["team"],
        "age": row["projected_age"],
        "stand": row["stand"],
        # Our model
        "our_k_rate": row["k_rate"],
        "our_bb_rate": row["bb_rate"],
        "our_hr_rate": row["hr_rate"],
        "our_iso": row["iso"],
        "our_babip": row["babip"],
        "our_avg": row["avg"],
        "our_obp": row["obp"],
        "our_slg": row["slg"],
        "our_woba": row["woba"],
        "our_wrc_plus": row["wrc_plus"],
        "our_off": row["off"],
        "our_hr": row["hr_count"],
        # Uncertainty
        "our_k_rate_std": row["k_rate_std"],
        "our_bb_rate_std": row["bb_rate_std"],
        "our_hr_rate_std": row["hr_rate_std"],
        "our_iso_std": row["iso_std"],
        "our_babip_std": row["babip_std"],
    }
    
    # Add FanGraphs systems
    for sys_name, df in fg_systems.items():
        match = df[df["batter"] == mlbam]
        if len(match) > 0:
            m = match.iloc[0]
            prefix = sys_name[:4]  # stea, zips, dept
            rec[f"{prefix}_k_rate"] = m["k_rate"]
            rec[f"{prefix}_bb_rate"] = m["bb_rate"]
            rec[f"{prefix}_hr_rate"] = m["hr_rate"]
            rec[f"{prefix}_iso"] = m["ISO"]
            rec[f"{prefix}_babip"] = m["BABIP"]
            rec[f"{prefix}_avg"] = m["AVG"]
            rec[f"{prefix}_obp"] = m["OBP"]
            rec[f"{prefix}_slg"] = m["SLG"]
            rec[f"{prefix}_woba"] = m["wOBA"]
            rec[f"{prefix}_wrc_plus"] = m["wRC+"]
            rec[f"{prefix}_off"] = m.get("Off", np.nan)
            rec[f"{prefix}_hr"] = m["HR"]
            rec[f"{prefix}_pa"] = m["PA"]
    
    records.append(rec)

comparison = pd.DataFrame(records)

# Only keep players that exist in at least one FG system
has_fg = comparison[[c for c in comparison.columns if c.startswith(("stea_", "zips_", "dept_"))]].notna().any(axis=1)
comparison = comparison[has_fg].reset_index(drop=True)

comparison.to_parquet(str(OUT_DIR / "comparison_2026.parquet"), index=False)
print(f"Comparison dataset: {len(comparison)} players matched")

# ══════════════════════════════════════════════════════════════════════
# 4. Summary statistics
# ══════════════════════════════════════════════════════════════════════
print("\n=== System Comparison Summary (matched players) ===")
for stat, our_col in [("K%", "k_rate"), ("BB%", "bb_rate"), ("HR/PA", "hr_rate"),
                       ("ISO", "iso"), ("BABIP", "babip"), ("wOBA", "woba"), ("wRC+", "wrc_plus")]:
    our_vals = comparison[f"our_{our_col}"]
    print(f"\n{stat}:")
    print(f"  Ours:    mean={our_vals.mean():.3f}  std={our_vals.std():.3f}")
    for sys_name, prefix in [("Steamer", "stea"), ("ZiPS", "zips"), ("DC", "dept")]:
        col = f"{prefix}_{our_col}"
        if col in comparison.columns:
            vals = comparison[col].dropna()
            if len(vals) > 0:
                # Correlation with our model
                common = comparison[[f"our_{our_col}", col]].dropna()
                corr = common[f"our_{our_col}"].corr(common[col])
                mae = (common[f"our_{our_col}"] - common[col]).abs().mean()
                print(f"  {sys_name:8s}: mean={vals.mean():.3f}  std={vals.std():.3f}  corr={corr:.3f}  MAE={mae:.3f}")

# Top 10 comparison
print("\n=== Top 10 by wOBA (Our Model vs Steamer vs ZiPS) ===")
top = comparison.sort_values("our_woba", ascending=False).head(10)
for _, r in top.iterrows():
    stea = f"{r.get('stea_woba', np.nan):.3f}" if pd.notna(r.get("stea_woba")) else "  N/A"
    zips = f"{r.get('zips_woba', np.nan):.3f}" if pd.notna(r.get("zips_woba")) else "  N/A"
    print(f"  {r['name']:22s} Ours={r['our_woba']:.3f}  Steamer={stea}  ZiPS={zips}")
