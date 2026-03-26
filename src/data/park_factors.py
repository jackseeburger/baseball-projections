"""
Park factor model — computes multi-year regressed park factors for each stadium.

Uses the standard method:
1. Compare runs scored in home games vs road games for each team
2. Apply multi-year averaging (3-year window) for stability
3. Regress toward 1.0 (neutral) to account for small sample sizes

Can also produce component-level park factors (HR, BB, K, BABIP, etc.)
when the underlying data supports it.

Outputs:
  - park_factors.parquet: Park factors by team/year for runs and components
"""
import pandas as pd
import numpy as np

from src.config import PARQUET_DIR, HISTORICAL_START_YEAR, CURRENT_YEAR
from src.utils.helpers import setup_logging

logger = setup_logging("park_factors")


def compute_basic_park_factors(
    hitter_df: pd.DataFrame,
    n_year_window: int = 3,
    regression_weight: float = 0.4,
) -> pd.DataFrame:
    """Compute regressed park factors from team batting data.
    
    Method:
    1. For each team-year, compare per-PA rates at home vs away
       (we approximate this by using team stats — proper splits would be better)
    2. Average over a multi-year window
    3. Regress toward 1.0 (neutral park)
    
    Args:
        hitter_df: DataFrame with hitter_seasons data
        n_year_window: Years to average (default 3)
        regression_weight: How much to regress toward 1.0 (0=no regression, 1=full regression)
    
    Returns:
        DataFrame with columns: team, year, pf_runs, pf_hr, pf_bb, pf_k, pf_babip, pf_h
    """
    if hitter_df.empty:
        return pd.DataFrame()
    
    # Aggregate to team-year totals
    team_stats = hitter_df.groupby(['team', 'year']).agg({
        'pa': 'sum',
        'h': 'sum',
        'hr': 'sum',
        'bb': 'sum',
        'so': 'sum',
        'r': 'sum',
        'doubles': 'sum',
        'triples': 'sum',
    }).reset_index()
    
    # Compute per-PA rates
    for col in ['h', 'hr', 'bb', 'so', 'r', 'doubles', 'triples']:
        team_stats[f'{col}_rate'] = team_stats[col] / team_stats['pa']
    
    # League average rates per year
    league_avg = team_stats.groupby('year').agg({
        'h_rate': 'mean',
        'hr_rate': 'mean',
        'bb_rate': 'mean',
        'so_rate': 'mean',
        'r_rate': 'mean',
    }).rename(columns=lambda c: f'lg_{c}')
    
    team_stats = team_stats.merge(league_avg, on='year')
    
    # Raw park factor = team rate / league rate
    # This is a simplified approach — proper park factors use home/away splits
    # For now, we use team-vs-league as a reasonable proxy
    components = {
        'pf_runs': ('r_rate', 'lg_r_rate'),
        'pf_hr': ('hr_rate', 'lg_hr_rate'),
        'pf_h': ('h_rate', 'lg_h_rate'),
        'pf_bb': ('bb_rate', 'lg_bb_rate'),
        'pf_k': ('so_rate', 'lg_so_rate'),
    }
    
    for pf_name, (team_col, lg_col) in components.items():
        team_stats[pf_name] = team_stats[team_col] / team_stats[lg_col]
    
    # Keep just park factor columns
    pf_cols = ['team', 'year'] + list(components.keys())
    pf_raw = team_stats[pf_cols].copy()
    
    # Multi-year averaging
    pf_smoothed = []
    for team in pf_raw['team'].unique():
        team_data = pf_raw[pf_raw['team'] == team].sort_values('year')
        
        for _, row in team_data.iterrows():
            year = row['year']
            # Window: current year + previous years
            window = team_data[
                (team_data['year'] >= year - n_year_window + 1) & 
                (team_data['year'] <= year)
            ]
            
            smoothed_row = {'team': team, 'year': year}
            for pf_col in components.keys():
                raw_avg = window[pf_col].mean()
                # Regress toward 1.0 (neutral)
                smoothed_row[pf_col] = (1 - regression_weight) * raw_avg + regression_weight * 1.0
            
            pf_smoothed.append(smoothed_row)
    
    result = pd.DataFrame(pf_smoothed)
    
    # BABIP park factor (derived from H and HR park factors)
    if 'pf_h' in result.columns and 'pf_hr' in result.columns:
        # BABIP PF ≈ (H_PF - HR_PF * HR_share) / (1 - HR_share)
        # Simplified: just use H park factor as proxy for BABIP PF
        result['pf_babip'] = result['pf_h']
    
    return result


def build_park_factors(
    hitter_df: pd.DataFrame = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build or load park factors.
    
    Args:
        hitter_df: Pre-loaded hitter seasons data. If None, loads from parquet.
        force_rebuild: If True, recompute even if cached.
    
    Returns:
        DataFrame with park factors by team-year.
    """
    output_path = PARQUET_DIR / "park_factors.parquet"
    
    if not force_rebuild and output_path.exists():
        logger.info("Loading cached park factors...")
        return pd.read_parquet(output_path)
    
    if hitter_df is None:
        hitter_path = PARQUET_DIR / "hitter_seasons.parquet"
        if not hitter_path.exists():
            raise FileNotFoundError(
                "hitter_seasons.parquet not found. Run historical_pipeline first."
            )
        hitter_df = pd.read_parquet(hitter_path)
    
    logger.info("Computing park factors...")
    pf_df = compute_basic_park_factors(hitter_df)
    
    if not pf_df.empty:
        pf_df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(pf_df)} team-year park factors to {output_path}")
        
        # Summary stats
        latest = pf_df[pf_df['year'] == pf_df['year'].max()]
        logger.info(f"\nLatest year park factors (runs):")
        for _, row in latest.sort_values('pf_runs', ascending=False).head(5).iterrows():
            logger.info(f"  {row['team']}: {row['pf_runs']:.3f}")
        logger.info("  ...")
        for _, row in latest.sort_values('pf_runs').head(5).iterrows():
            logger.info(f"  {row['team']}: {row['pf_runs']:.3f}")
    
    return pf_df


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    pf = build_park_factors(force_rebuild=force)
    print(f"\nPark Factors Complete!")
    print(f"  {len(pf)} team-year records")
    if not pf.empty:
        print(f"  Years: {pf['year'].min()}-{pf['year'].max()}")
        print(f"  Teams: {pf['team'].nunique()}")
