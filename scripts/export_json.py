#!/usr/bin/env python3
"""Export parquet/json data to browser-friendly JSON for D3 dashboard."""

import json
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path("/home/hermes/projects/baseball-dashboard/public/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ_DIR = Path("/home/hermes/projects/baseball-assembly/data/projections")
FG_DIR = Path("/home/hermes/projects/baseball-dashboard/data/projections")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 4)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# 1. Comparison dataset
print("Exporting comparison data...")
comp = pd.read_parquet(str(FG_DIR / "comparison_2026.parquet"))
# Round floats for smaller file size
for col in comp.select_dtypes(include=[np.floating]).columns:
    comp[col] = comp[col].round(4)
comp = comp.fillna("")
comp.to_json(str(OUT_DIR / "comparison.json"), orient="records")
print(f"  comparison.json: {len(comp)} players")

# 2. Our full model
print("Exporting our model data...")
our = pd.read_parquet(str(FG_DIR / "our_model_2026.parquet"))
for col in our.select_dtypes(include=[np.floating]).columns:
    our[col] = our[col].round(4)
our.to_json(str(OUT_DIR / "our_model.json"), orient="records")
print(f"  our_model.json: {len(our)} players")

# 3. Aging curves
print("Exporting aging curves...")
aging = {}
for stat in ["k_rate", "bb_rate", "hr_rate", "iso", "babip"]:
    path = PROJ_DIR / f"{stat}_aging_curve_2026.parquet"
    if path.exists():
        df = pd.read_parquet(str(path))
        print(f"  {stat}: {df.shape}, cols={list(df.columns)}")
        
        if "age" in df.columns:
            # Column might be aging_effect or age_effect_mean
            effect_col = None
            for candidate in ["aging_effect", "age_effect_mean", "effect"]:
                if candidate in df.columns:
                    effect_col = candidate
                    break
            
            if effect_col:
                records = []
                for _, row in df.iterrows():
                    rec = {"age": round(float(row["age"]), 1), "mean": round(float(row[effect_col]), 4)}
                    if "age_effect_lower" in df.columns:
                        rec["lower"] = round(float(row["age_effect_lower"]), 4)
                    if "age_effect_upper" in df.columns:
                        rec["upper"] = round(float(row["age_effect_upper"]), 4)
                    records.append(rec)
                aging[stat] = records
            else:
                print(f"    WARNING: no effect column found in {list(df.columns)}")
        else:
            print(f"    WARNING: no age column")
    else:
        print(f"  {stat}: NO FILE")

with open(str(OUT_DIR / "aging_curves.json"), "w") as f:
    json.dump(aging, f, cls=NpEncoder)
print(f"  aging_curves.json: {len(aging)} stats")

# 4. Summary stats for quick loading
print("Computing summary stats...")
summary = {
    "total_players": len(our),
    "matched_players": len(comp),
    "systems": {}
}

for prefix, name in [("our", "Bayesian"), ("stea", "Steamer"), ("zips", "ZiPS"), ("dept", "Depth Charts")]:
    stats = {}
    for stat in ["k_rate", "bb_rate", "hr_rate", "iso", "babip", "avg", "obp", "slg", "woba", "wrc_plus", "off"]:
        col = f"{prefix}_{stat}"
        if col in comp.columns:
            vals = pd.to_numeric(comp[col], errors='coerce').dropna()
            if len(vals) > 0:
                stats[stat] = {
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4),
                    "n": int(len(vals)),
                }
    summary["systems"][name] = stats

# Correlations
corrs = {}
for stat in ["k_rate", "bb_rate", "hr_rate", "iso", "babip", "woba", "wrc_plus"]:
    corrs[stat] = {}
    for prefix, name in [("stea", "Steamer"), ("zips", "ZiPS"), ("dept", "Depth Charts")]:
        our_col = f"our_{stat}"
        other_col = f"{prefix}_{stat}"
        if our_col in comp.columns and other_col in comp.columns:
            common = comp[[our_col, other_col]].replace("", np.nan).dropna()
            common[our_col] = pd.to_numeric(common[our_col])
            common[other_col] = pd.to_numeric(common[other_col])
            if len(common) > 10:
                corrs[stat][name] = round(float(common[our_col].corr(common[other_col])), 3)
                
summary["correlations"] = corrs

with open(str(OUT_DIR / "summary.json"), "w") as f:
    json.dump(summary, f, cls=NpEncoder, indent=2)

print("\nDone! Files in", OUT_DIR)
for p in sorted(OUT_DIR.glob("*.json")):
    size = p.stat().st_size
    print(f"  {p.name}: {size/1024:.0f}KB")
