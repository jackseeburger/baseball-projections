#!/usr/bin/env python3
import pandas as pd, os
parquet_dir = "data/parquet"
for f in sorted(os.listdir(parquet_dir)):
    if f.endswith('.parquet'):
        df = pd.read_parquet(f"{parquet_dir}/{f}")
        print(f"\n{f}: {len(df):,} rows x {len(df.columns)} cols")
        if 'year' in df.columns:
            print(f"  Years: {int(df['year'].min())}-{int(df['year'].max())}")
        if 'name' in df.columns:
            print(f"  Unique players: {df['name'].nunique()}")
        if 'team' in df.columns:
            print(f"  Teams: {df['team'].nunique()}")
        # Show a few sample rows for Marcel projections
        if 'marcel' in f and 'war' in df.columns:
            print(f"\n  Top 5 by WAR:")
            top = df.nlargest(5, 'war')
            for _, r in top.iterrows():
                if 'era' in df.columns:
                    print(f"    {r['name']:25s} {r['team']:4s} age {r['age']:2.0f} | {r['ip']:.0f} IP, {r['era']:.2f} ERA, {r['fip']:.2f} FIP, {r['war']:.1f} WAR")
                else:
                    print(f"    {r['name']:25s} {r['team']:4s} age {r['age']:2.0f} | {r['pa']:.0f} PA, .{r['avg']:.3f} AVG, {r['hr']:.0f} HR, {r['wrc_plus']:.0f} wRC+, {r['war']:.1f} WAR")
