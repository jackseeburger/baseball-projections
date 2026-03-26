"""
Marcel projection system — the "minimum intelligence" baseline.

Marcel is Tom Tango's intentionally simple projection system that serves as the
"must beat" benchmark. Any projection system that can't beat Marcel isn't worth
its complexity.

The Marcel method:
1. Weight 3 years of data: 5/4/3 (most recent / previous / two years ago)
2. Regress toward league average using a reliability weighting
3. Apply an age adjustment
4. Project playing time based on recent trends

For hitters, we project component rates (K%, BB%, ISO, BABIP, HR/FB)
and then assemble them into aggregate stats (AVG, OBP, SLG, wOBA, wRC+).

For pitchers, we project K%, BB%, HR/FB, BABIP, LOB% and derive ERA/FIP/xFIP.

Reference: https://www.tangotiger.net/marcel/
"""
import pandas as pd
import numpy as np
from typing import Optional

from src.config import (
    MARCEL_WEIGHTS,
    MARCEL_LEAGUE_PA,
    MARCEL_LEAGUE_IP,
    PARQUET_DIR,
    CURRENT_YEAR,
)
from src.utils.helpers import setup_logging

logger = setup_logging("marcel")


# ─── wOBA constants (approximate 2024 values) ───
# These should ideally be loaded per-year from FanGraphs
WOBA_WEIGHTS = {
    'bb': 0.690,
    'hbp': 0.720,
    'singles': 0.880,
    'doubles': 1.245,
    'triples': 1.580,
    'hr': 2.030,
}
WOBA_SCALE = 1.185
LG_WOBA = 0.315
LG_OBP = 0.315
LG_R_PA = 0.120  # league runs per PA


def compute_league_averages(df: pd.DataFrame, year: int) -> dict:
    """Compute league-average rates for a given year."""
    season = df[df['year'] == year]
    if season.empty:
        return {}
    
    total_pa = season['pa'].sum()
    if total_pa == 0:
        return {}
    
    avgs = {}
    for col in ['k_rate', 'bb_rate', 'babip', 'iso', 'hr_fb', 'avg', 'obp', 'slg', 'woba']:
        if col in season.columns:
            # PA-weighted average
            valid = season[season[col].notna() & (season['pa'] > 0)]
            if not valid.empty:
                avgs[col] = np.average(valid[col], weights=valid['pa'])
    
    return avgs


def compute_league_averages_pitchers(df: pd.DataFrame, year: int) -> dict:
    """Compute league-average rates for pitchers in a given year."""
    season = df[df['year'] == year]
    if season.empty:
        return {}
    
    total_ip = season['ip'].sum()
    if total_ip == 0:
        return {}
    
    avgs = {}
    for col in ['k_pct', 'bb_pct', 'hr_fb', 'babip', 'lob_pct', 'era', 'fip', 'xfip', 'k_9', 'bb_9', 'hr_9']:
        if col in season.columns:
            valid = season[season[col].notna() & (season['ip'] > 0)]
            if not valid.empty:
                avgs[col] = np.average(valid[col], weights=valid['ip'])
    
    return avgs


def marcel_rate(
    player_seasons: pd.DataFrame,
    stat_col: str,
    weight_col: str,
    projection_year: int,
    league_avgs: dict,
    regression_pa: int = MARCEL_LEAGUE_PA,
) -> Optional[float]:
    """Compute Marcel-weighted rate stat for a player.
    
    Formula:
        weighted_stat = sum(year_weight * PA * rate) / sum(year_weight * PA)
        reliability = sum(year_weight * PA) / (sum(year_weight * PA) + regression_PA)
        projected = reliability * weighted_stat + (1 - reliability) * league_avg
    
    Args:
        player_seasons: DataFrame of the player's historical seasons
        stat_col: Column name for the rate stat (e.g., 'k_rate')
        weight_col: Column name for the weighting denominator (e.g., 'pa')
        projection_year: Year we're projecting
        league_avgs: Dict of league-average rates by stat name
        regression_pa: PA equivalent for regression toward league avg
    
    Returns:
        Projected rate, or None if insufficient data
    """
    lg_avg = league_avgs.get(stat_col, np.nan)
    if np.isnan(lg_avg):
        return None
    
    weighted_num = 0.0  # weighted sum of (weight * rate * pa)
    weighted_den = 0.0  # weighted sum of (weight * pa)
    
    for years_ago, year_weight in MARCEL_WEIGHTS.items():
        year = projection_year - years_ago
        season = player_seasons[player_seasons['year'] == year]
        
        if season.empty:
            continue
        
        row = season.iloc[0]
        pa = row.get(weight_col, 0)
        rate = row.get(stat_col, np.nan)
        
        if pd.isna(rate) or pa <= 0:
            continue
        
        weighted_num += year_weight * pa * rate
        weighted_den += year_weight * pa
    
    if weighted_den == 0:
        return None
    
    weighted_rate = weighted_num / weighted_den
    reliability = weighted_den / (weighted_den + regression_pa)
    
    return reliability * weighted_rate + (1 - reliability) * lg_avg


def marcel_playing_time(
    player_seasons: pd.DataFrame,
    weight_col: str,
    projection_year: int,
) -> float:
    """Project playing time (PA or IP) using Marcel method.
    
    Formula: 0.5 * Y1 + 0.1 * Y2 (simple weighted recent history)
    With a floor of 200 PA for hitters, 50 IP for pitchers.
    """
    y1 = player_seasons[player_seasons['year'] == projection_year - 1]
    y2 = player_seasons[player_seasons['year'] == projection_year - 2]
    
    pa1 = y1.iloc[0][weight_col] if not y1.empty else 0
    pa2 = y2.iloc[0][weight_col] if not y2.empty else 0
    
    projected = 0.5 * pa1 + 0.1 * pa2
    
    # Floor
    if weight_col == 'pa':
        return max(projected, 200)
    else:  # IP
        return max(projected, 50)


def age_adjustment(age: int, stat_col: str) -> float:
    """Apply age-based adjustment to projected rate.
    
    Simple aging curve: peak at 27-28, linear decline after.
    Returns a multiplicative factor (e.g., 1.02 for a 25-year-old's K rate).
    
    Positive stats (power, speed) decline after peak.
    Negative stats (K%) increase after peak.
    """
    peak_age = 27
    
    # Positive stats (higher is better): small boost if young, penalty if old
    positive_stats = {'bb_rate', 'iso', 'babip', 'avg', 'obp', 'slg', 'woba'}
    # Negative stats (lower is better for hitters): K rate increases with age
    negative_stats = {'k_rate'}
    
    age_diff = age - peak_age
    
    if stat_col in positive_stats:
        # ~0.3% decline per year past peak
        return 1.0 - (age_diff * 0.003) if age_diff > 0 else 1.0 + (abs(age_diff) * 0.001)
    elif stat_col in negative_stats:
        # K rate increases ~0.5% per year past peak
        return 1.0 + (age_diff * 0.005) if age_diff > 0 else 1.0 - (abs(age_diff) * 0.002)
    else:
        return 1.0


def project_hitters(
    hitter_df: pd.DataFrame,
    projection_year: int = CURRENT_YEAR + 1,
    min_pa_recent: int = 50,
) -> pd.DataFrame:
    """Generate Marcel hitter projections.
    
    Projects: PA, K%, BB%, ISO, BABIP, HR/FB, AVG, OBP, SLG, wOBA, HR, R, RBI
    
    Args:
        hitter_df: Historical hitter seasons from hitter_seasons.parquet
        projection_year: Year to project (default: next year)
        min_pa_recent: Minimum PA in most recent 2 years to include a player
    
    Returns:
        DataFrame with one row per player, projected stats
    """
    logger.info(f"Projecting hitters for {projection_year}...")
    
    # Compute league averages for each of the 3 prior years
    lg_avgs = {}
    for y in range(projection_year - 3, projection_year):
        lg_avgs[y] = compute_league_averages(hitter_df, y)
    
    # Use most recent year's league averages as baseline for regression
    recent_lg = lg_avgs.get(projection_year - 1, {})
    if not recent_lg:
        # Fallback to any available year
        for y in sorted(lg_avgs.keys(), reverse=True):
            if lg_avgs[y]:
                recent_lg = lg_avgs[y]
                break
    
    # Get players who had meaningful PA in recent years
    recent_years = hitter_df[
        hitter_df['year'].isin(range(projection_year - 3, projection_year))
    ]
    recent_pa = recent_years.groupby('fg_id')['pa'].sum()
    eligible = recent_pa[recent_pa >= min_pa_recent].index
    
    projections = []
    rate_stats = ['k_rate', 'bb_rate', 'iso', 'babip', 'hr_fb']
    
    for fg_id in eligible:
        player = hitter_df[hitter_df['fg_id'] == fg_id].sort_values('year')
        latest = player.iloc[-1]
        
        proj = {
            'fg_id': fg_id,
            'name': latest['name'],
            'team': latest['team'],
            'age': latest['age'] + (projection_year - latest['year']),
            'year': projection_year,
        }
        
        # Project playing time
        proj['pa'] = marcel_playing_time(player, 'pa', projection_year)
        
        # Project rate stats
        for stat in rate_stats:
            rate = marcel_rate(player, stat, 'pa', projection_year, recent_lg)
            if rate is not None:
                # Apply age adjustment
                adj = age_adjustment(proj['age'], stat)
                proj[stat] = np.clip(rate * adj, 0, 1)  # Clip to valid range
            else:
                proj[stat] = recent_lg.get(stat, np.nan)
        
        # Derive counting and aggregate stats from components
        pa = proj['pa']
        bb_rate = proj.get('bb_rate', 0)
        k_rate = proj.get('k_rate', 0)
        iso = proj.get('iso', 0)
        babip = proj.get('babip', 0)
        hr_fb = proj.get('hr_fb', 0)
        
        # Derive AVG from BABIP and K%
        # AVG ≈ BABIP * (1 - K%) * (1 - HR_share) + HR_share
        # Simplified: AVG ≈ BABIP * (1 - K% - HR/PA) + HR/PA
        ab = pa * (1 - bb_rate - 0.01)  # approximate AB
        
        # HR from HR/FB and estimated FB count
        fb_rate = 0.35  # league average FB%
        proj['ab'] = ab
        proj['bb'] = int(pa * bb_rate)
        proj['so'] = int(pa * k_rate)
        
        # Balls in play
        bip = ab - proj['so']
        fb_count = bip * fb_rate
        proj['hr'] = int(fb_count * hr_fb) if hr_fb > 0 else int(pa * 0.03)
        
        # Hits from BABIP (BABIP = (H - HR) / (AB - SO - HR + SF))
        h_minus_hr = babip * (ab - proj['so'] - proj['hr'])
        proj['h'] = int(h_minus_hr + proj['hr'])
        
        # AVG, OBP, SLG
        proj['avg'] = proj['h'] / ab if ab > 0 else 0
        proj['obp'] = (proj['h'] + proj['bb']) / pa if pa > 0 else 0
        proj['slg'] = proj['avg'] + iso
        
        # wOBA (simplified)
        singles = proj['h'] - proj['hr'] - int(bip * 0.20 * 0.5)  # rough doubles
        doubles = int(bip * 0.20 * 0.5)
        triples = int(bip * 0.03)
        
        woba_num = (WOBA_WEIGHTS['bb'] * proj['bb'] +
                    WOBA_WEIGHTS['singles'] * max(singles, 0) +
                    WOBA_WEIGHTS['doubles'] * doubles +
                    WOBA_WEIGHTS['triples'] * triples +
                    WOBA_WEIGHTS['hr'] * proj['hr'])
        proj['woba'] = woba_num / pa if pa > 0 else 0
        
        # wRC+
        if LG_WOBA > 0:
            proj['wrc_plus'] = int(100 * ((proj['woba'] - LG_WOBA) / WOBA_SCALE + LG_R_PA) / LG_R_PA)
        
        # R and RBI (rough estimates from wOBA and PA)
        proj['r'] = int(pa * LG_R_PA * (proj['woba'] / LG_WOBA) * 0.9)
        proj['rbi'] = int(pa * LG_R_PA * (proj['woba'] / LG_WOBA) * 0.85)
        
        # WAR estimate (very rough: (wRC+ - 100) / 600 * PA * 0.1 + replacement)
        proj['war'] = round(((proj.get('wrc_plus', 100) - 100) / 600) * pa * 0.1 + pa / 600 * 0.5, 1)
        
        projections.append(proj)
    
    result = pd.DataFrame(projections)
    logger.info(f"  Projected {len(result)} hitters")
    return result


def project_pitchers(
    pitcher_df: pd.DataFrame,
    projection_year: int = CURRENT_YEAR + 1,
    min_ip_recent: float = 20,
) -> pd.DataFrame:
    """Generate Marcel pitcher projections.
    
    Projects: IP, K%, BB%, HR/FB, BABIP, LOB%, ERA, FIP, xFIP, K/9, BB/9
    
    Args:
        pitcher_df: Historical pitcher seasons from pitcher_seasons.parquet
        projection_year: Year to project
        min_ip_recent: Minimum IP in recent 2 years to include
    
    Returns:
        DataFrame with one row per pitcher, projected stats
    """
    logger.info(f"Projecting pitchers for {projection_year}...")
    
    # League averages
    lg_avgs = {}
    for y in range(projection_year - 3, projection_year):
        lg_avgs[y] = compute_league_averages_pitchers(pitcher_df, y)
    
    recent_lg = lg_avgs.get(projection_year - 1, {})
    if not recent_lg:
        for y in sorted(lg_avgs.keys(), reverse=True):
            if lg_avgs[y]:
                recent_lg = lg_avgs[y]
                break
    
    # Eligible pitchers
    recent = pitcher_df[pitcher_df['year'].isin(range(projection_year - 3, projection_year))]
    recent_ip = recent.groupby('fg_id')['ip'].sum()
    eligible = recent_ip[recent_ip >= min_ip_recent].index
    
    projections = []
    rate_stats = ['k_pct', 'bb_pct', 'hr_fb', 'babip', 'lob_pct']
    
    for fg_id in eligible:
        player = pitcher_df[pitcher_df['fg_id'] == fg_id].sort_values('year')
        latest = player.iloc[-1]
        
        proj = {
            'fg_id': fg_id,
            'name': latest['name'],
            'team': latest['team'],
            'age': latest['age'] + (projection_year - latest['year']),
            'year': projection_year,
        }
        
        # Playing time
        proj['ip'] = marcel_playing_time(player, 'ip', projection_year)
        
        # Rate stats
        for stat in rate_stats:
            rate = marcel_rate(player, stat, 'ip', projection_year, recent_lg,
                             regression_pa=MARCEL_LEAGUE_IP)
            if rate is not None:
                proj[stat] = np.clip(rate, 0, 1)
            else:
                proj[stat] = recent_lg.get(stat, np.nan)
        
        # Derive counting stats
        ip = proj['ip']
        k_pct = proj.get('k_pct', 0.20)
        bb_pct = proj.get('bb_pct', 0.08)
        hr_fb = proj.get('hr_fb', 0.10)
        babip = proj.get('babip', 0.300)
        lob_pct = proj.get('lob_pct', 0.72)
        
        # Estimate batters faced from IP (roughly 4.3 BF per IP in modern era)
        bf = ip * 4.3
        proj['so'] = int(bf * k_pct)
        proj['bb'] = int(bf * bb_pct)
        
        # K/9 and BB/9
        proj['k_9'] = (proj['so'] / ip * 9) if ip > 0 else 0
        proj['bb_9'] = (proj['bb'] / ip * 9) if ip > 0 else 0
        
        # HR from HR/FB
        bip = bf - proj['so'] - proj['bb']
        fb = bip * 0.35  # ~35% FB rate
        proj['hr'] = int(fb * hr_fb)
        proj['hr_9'] = (proj['hr'] / ip * 9) if ip > 0 else 0
        
        # Hits from BABIP
        proj['h'] = int(babip * (bip - proj['hr']) + proj['hr'])
        
        # FIP = ((13*HR + 3*BB - 2*K) / IP) + FIP_constant
        FIP_CONSTANT = 3.20  # approximate
        proj['fip'] = round(((13 * proj['hr'] + 3 * proj['bb'] - 2 * proj['so']) / ip + FIP_CONSTANT), 2) if ip > 0 else 0
        
        # ERA from FIP and LOB%
        # Rough approximation: ERA ≈ FIP * (1 + (0.72 - LOB%)) when LOB% deviates
        proj['era'] = round(proj['fip'] * (1 + (0.72 - lob_pct) * 2), 2)
        proj['era'] = max(proj['era'], 1.50)  # Floor
        
        # WHIP
        proj['whip'] = round((proj['h'] + proj['bb']) / ip, 2) if ip > 0 else 0
        
        # WAR estimate (very rough)
        lg_era = recent_lg.get('era', 4.20)
        runs_saved = (lg_era - proj['era']) / 9 * ip
        proj['war'] = round(runs_saved / 10 + ip / 200 * 0.5, 1)
        
        # Starter vs reliever (based on recent GS rate)
        recent_gs = player[player['year'] >= projection_year - 2]
        if not recent_gs.empty:
            gs_rate = recent_gs['gs'].sum() / recent_gs['g'].sum() if recent_gs['g'].sum() > 0 else 0
            proj['role'] = 'SP' if gs_rate > 0.5 else 'RP'
        else:
            proj['role'] = 'SP' if ip > 100 else 'RP'
        
        projections.append(proj)
    
    result = pd.DataFrame(projections)
    logger.info(f"  Projected {len(result)} pitchers")
    return result


def run_marcel(
    projection_year: int = CURRENT_YEAR + 1,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full Marcel projections for hitters and pitchers.
    
    Returns:
        (hitter_projections, pitcher_projections) DataFrames
    """
    h_output = PARQUET_DIR / f"marcel_hitters_{projection_year}.parquet"
    p_output = PARQUET_DIR / f"marcel_pitchers_{projection_year}.parquet"
    
    if not force_rebuild and h_output.exists() and p_output.exists():
        logger.info("Loading cached Marcel projections...")
        return pd.read_parquet(h_output), pd.read_parquet(p_output)
    
    # Load historical data
    hitter_path = PARQUET_DIR / "hitter_seasons.parquet"
    pitcher_path = PARQUET_DIR / "pitcher_seasons.parquet"
    
    if not hitter_path.exists() or not pitcher_path.exists():
        raise FileNotFoundError(
            "Historical data not found. Run historical_pipeline first."
        )
    
    hitter_df = pd.read_parquet(hitter_path)
    pitcher_df = pd.read_parquet(pitcher_path)
    
    # Project
    h_proj = project_hitters(hitter_df, projection_year)
    p_proj = project_pitchers(pitcher_df, projection_year)
    
    # Save
    if not h_proj.empty:
        h_proj.to_parquet(h_output, index=False)
        logger.info(f"Saved hitter projections to {h_output}")
    
    if not p_proj.empty:
        p_proj.to_parquet(p_output, index=False)
        logger.info(f"Saved pitcher projections to {p_output}")
    
    return h_proj, p_proj


if __name__ == "__main__":
    import sys
    
    year = CURRENT_YEAR + 1
    force = "--force" in sys.argv
    
    # Allow specifying year
    for arg in sys.argv[1:]:
        if arg.isdigit():
            year = int(arg)
    
    hitters, pitchers = run_marcel(projection_year=year, force_rebuild=force)
    
    print(f"\n{'='*60}")
    print(f"Marcel Projections for {year}")
    print(f"{'='*60}")
    
    if not hitters.empty:
        print(f"\n--- Top 20 Hitters by WAR ---")
        top_h = hitters.nlargest(20, 'war')[['name', 'team', 'age', 'pa', 'avg', 'obp', 'slg', 'hr', 'woba', 'wrc_plus', 'war']]
        print(top_h.to_string(index=False))
    
    if not pitchers.empty:
        print(f"\n--- Top 20 Pitchers by WAR ---")
        top_p = pitchers.nlargest(20, 'war')[['name', 'team', 'age', 'role', 'ip', 'era', 'fip', 'k_9', 'bb_9', 'whip', 'war']]
        print(top_p.to_string(index=False))
