
import modal
import os

app = modal.App("upload-fix")
volume = modal.Volume.from_name("baseball-data")
image = modal.Image.debian_slim().pip_install("pandas", "pyarrow")

@app.function(image=image, volumes={"/data": volume}, timeout=300)
def rewrite_pa_files():
    """Read PA files from volume, add birth_year, write back."""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    pa_dir = Path("/data/parquet/pa_outcomes")
    files = sorted(pa_dir.glob("*.parquet"))
    print(f"Found {len(files)} PA files")
    
    # First check current schema
    df0 = pd.read_parquet(files[0]).head(1)
    print(f"Current columns: {list(df0.columns)}")
    
    if "birth_year" in df0.columns:
        print("birth_year already present! Done.")
        return
    
    # Build birth year map from debut year - 24 (same fallback)
    # Load all data to get first appearances
    all_batters = set()
    first_year = {}
    for f in files:
        df = pd.read_parquet(f, columns=["batter", "game_year"])
        for batter, year in zip(df["batter"], df["game_year"]):
            if batter not in first_year or year < first_year[batter]:
                first_year[batter] = year
    
    birth_map = {b: y - 24 for b, y in first_year.items()}
    print(f"Computed birth years for {len(birth_map)} batters")
    
    # Now rewrite each file with birth_year added
    for f in files:
        df = pd.read_parquet(f)
        df["birth_year"] = df["batter"].map(birth_map).astype(int)
        df.to_parquet(f, index=False)
        print(f"  Rewrote {f.name}: {len(df)} rows, {len(df.columns)} cols")
    
    # Commit the volume
    volume.commit()
    print("\nDone! Volume committed.")

@app.local_entrypoint()
def main():
    rewrite_pa_files.remote()
