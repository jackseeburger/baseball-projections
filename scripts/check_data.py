import pandas as pd

df = pd.read_parquet('data/parquet/pa_outcomes/pa_outcomes_2024.parquet')
print('PA Columns:', list(df.columns))
print('Shape:', df.shape)
print('Sample:')
print(df.head(3).to_string())
print()

pf = pd.read_parquet('data/parquet/park_factors.parquet')
print('Park factors columns:', list(pf.columns))
print('PF sample:')
print(pf.head(3).to_string())
print()

# Check if model-expected columns exist
needed = ['batter', 'game_year', 'stand', 'is_k', 'home_team', 'away_team', 'inning_topbot']
missing = [c for c in needed if c not in df.columns]
print(f'Missing columns for model: {missing}')
# Check if inning_topbot exists
if 'inning_topbot' not in df.columns:
    print(f'Available inning-related cols: {[c for c in df.columns if "inning" in c.lower()]}')
print(f'\npf_k in park_factors: {"k_park_factor" in pf.columns or "pf_k" in pf.columns}')
print(f'Park factor columns: {list(pf.columns)}')
