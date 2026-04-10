"""Build career WAR data with historical actuals + projected future."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("/home/hermes/projects/baseball-dashboard/public/data")

# Load historical hitter seasons
hs = pd.read_parquet("/home/hermes/projects/baseball-projections/data/parquet/hitter_seasons.parquet")

# Load FanGraphs projection data to get FG→MLBAM crosswalk
fg_data = {}
for sys in ["steamer", "zips"]:
    with open(f"/home/hermes/projects/baseball-dashboard/data/projections/fg_{sys}_bat_2026.json") as f:
        fg_data[sys] = json.load(f)

# Build FG ID → MLBAM crosswalk
fg_to_mlbam = {}
mlbam_to_name = {}
for sys_name, players in fg_data.items():
    for p in players:
        fg_id = str(p.get("playerids", ""))
        mlbam = p.get("xMLBAMID")
        name = p.get("PlayerName", "")
        if fg_id and mlbam:
            fg_to_mlbam[fg_id] = int(mlbam)
            mlbam_to_name[int(mlbam)] = name

print(f"Crosswalk: {len(fg_to_mlbam)} FG→MLBAM mappings")

# Load our projections
our = pd.read_parquet("/home/hermes/projects/baseball-dashboard/data/projections/our_model_2026.parquet")

# Build career data for each player in our projections
career_data = {}

for _, proj in our.iterrows():
    mlbam = int(proj["batter"])
    fg_id = None
    
    # Find FG ID for this MLBAM
    for fid, mid in fg_to_mlbam.items():
        if mid == mlbam:
            fg_id = fid
            break
    
    if fg_id is None:
        continue
    
    # Get historical seasons
    player_hist = hs[hs["fg_id"] == int(fg_id)].sort_values("year")
    if len(player_hist) == 0:
        continue
    
    name = player_hist.iloc[0]["name"]
    
    # Historical WAR
    historical = []
    for _, row in player_hist.iterrows():
        historical.append({
            "year": int(row["year"]),
            "age": int(row["age"]),
            "war": round(float(row["war"]), 1),
            "pa": int(row["pa"]),
            "team": row["team"],
            "woba": round(float(row["woba"]), 3) if pd.notna(row["woba"]) else None,
            "wrc_plus": round(float(row["wrc_plus"]), 0) if pd.notna(row["wrc_plus"]) else None,
            "off": round(float(row["off"]), 1) if pd.notna(row["off"]) else None,
        })
    
    # Projected WAR for future years
    # Use our 2026 oWAR projection, then apply aging curve decline
    # For simplicity, project WAR declining from our 2026 estimate based on age
    proj_age = proj["projected_age"]
    proj_woba = proj["woba"]
    proj_wrc_plus = proj["wrc_plus"]
    proj_off = proj["wraa"]
    
    # Simple WAR approximation from offensive runs
    # WAR ≈ (Off + positional + replacement) / 10
    # We'll use Off + ~20 replacement runs per 600PA as proxy
    pa_est = 550
    replacement = 20.0 * (pa_est / 600.0)
    proj_war_2026 = (proj_off + replacement) / 10.0
    
    # Project future years using a simple aging decline
    # ~0.5 WAR decline per year after 30, ~0.3 before
    projected = []
    last_year = max(r["year"] for r in historical)
    
    for future_year in range(2026, 2033):
        future_age = proj_age + (future_year - 2026)
        if future_age > 42:
            break
        
        # Age-based decline from 2026 projection
        years_out = future_year - 2026
        if future_age <= 28:
            decline = years_out * 0.1  # still improving or plateau
        elif future_age <= 32:
            decline = years_out * 0.3
        elif future_age <= 36:
            decline = years_out * 0.5
        else:
            decline = years_out * 0.8
        
        future_war = max(proj_war_2026 - decline, -1.0)
        
        projected.append({
            "year": future_year,
            "age": int(future_age),
            "war": round(future_war, 1),
            "type": "projected",
        })
    
    # Only include players with at least 2 historical seasons
    if len(historical) >= 2:
        career_data[mlbam] = {
            "name": name,
            "mlbam": mlbam,
            "fg_id": int(fg_id),
            "historical": historical,
            "projected": projected,
        }

print(f"Career data: {len(career_data)} players")

# Save
with open(str(OUT / "career_war.json"), "w") as f:
    json.dump(career_data, f)

size = (OUT / "career_war.json").stat().st_size
print(f"career_war.json: {size/1024:.0f}KB")

# Quick sanity check
judge = career_data.get(592450)
if judge:
    print(f"\nAaron Judge career:")
    for s in judge["historical"]:
        print(f"  {s['year']} (age {s['age']}): {s['war']} WAR, {s['pa']} PA")
    print("  --- projected ---")
    for s in judge["projected"]:
        print(f"  {s['year']} (age {s['age']}): {s['war']} WAR")
