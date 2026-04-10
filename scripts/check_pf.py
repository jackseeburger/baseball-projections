import pandas as pd
pf = pd.read_parquet('data/parquet/park_factors.parquet')
print('Local columns:', list(pf.columns))
print(pf.head(2))
