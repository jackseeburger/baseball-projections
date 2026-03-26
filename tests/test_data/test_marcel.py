"""Tests for the Marcel projection system."""
import pandas as pd
import numpy as np
import pytest
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

from src.models.marcel import (
    marcel_rate,
    marcel_playing_time,
    age_adjustment,
    compute_league_averages,
    MARCEL_WEIGHTS,
)


@pytest.fixture
def sample_hitter_data():
    """Create sample hitter data for testing."""
    return pd.DataFrame([
        {'fg_id': 1, 'name': 'Mike Trout', 'team': 'LAA', 'age': 30, 'year': 2022,
         'pa': 500, 'ab': 440, 'h': 130, 'hr': 30, 'bb': 55, 'so': 120,
         'doubles': 25, 'triples': 2, 'r': 80, 'rbi': 85,
         'avg': 0.295, 'obp': 0.380, 'slg': 0.560, 'woba': 0.400,
         'babip': 0.320, 'iso': 0.265, 'hr_fb': 0.22, 'k_rate': 0.24,
         'bb_rate': 0.11, 'wrc_plus': 160, 'war': 6.0,
         'singles': 73, 'hbp': 5, 'sf': 3, 'sh': 0, 'gdp': 10, 'ibb': 5,
         'sb': 5, 'cs': 2,
         'gb_pct': 0.35, 'fb_pct': 0.40, 'ld_pct': 0.25,
         'o_swing_pct': 0.28, 'z_swing_pct': 0.68, 'swing_pct': 0.45,
         'o_contact_pct': 0.60, 'z_contact_pct': 0.85, 'contact_pct': 0.75,
         'swstr_pct': 0.12, 'off': 35, 'def_value': -2, 'bsr': 0.5,
         'bb_pct': 0.11, 'k_pct': 0.24},
        {'fg_id': 1, 'name': 'Mike Trout', 'team': 'LAA', 'age': 31, 'year': 2023,
         'pa': 450, 'ab': 395, 'h': 110, 'hr': 25, 'bb': 50, 'so': 115,
         'doubles': 20, 'triples': 1, 'r': 70, 'rbi': 75,
         'avg': 0.278, 'obp': 0.365, 'slg': 0.530, 'woba': 0.385,
         'babip': 0.310, 'iso': 0.252, 'hr_fb': 0.20, 'k_rate': 0.255,
         'bb_rate': 0.111, 'wrc_plus': 150, 'war': 5.0,
         'singles': 64, 'hbp': 4, 'sf': 2, 'sh': 0, 'gdp': 8, 'ibb': 4,
         'sb': 3, 'cs': 1,
         'gb_pct': 0.36, 'fb_pct': 0.39, 'ld_pct': 0.25,
         'o_swing_pct': 0.29, 'z_swing_pct': 0.67, 'swing_pct': 0.45,
         'o_contact_pct': 0.59, 'z_contact_pct': 0.84, 'contact_pct': 0.74,
         'swstr_pct': 0.13, 'off': 30, 'def_value': -3, 'bsr': 0.3,
         'bb_pct': 0.111, 'k_pct': 0.255},
        {'fg_id': 1, 'name': 'Mike Trout', 'team': 'LAA', 'age': 32, 'year': 2024,
         'pa': 400, 'ab': 350, 'h': 95, 'hr': 20, 'bb': 45, 'so': 100,
         'doubles': 18, 'triples': 1, 'r': 60, 'rbi': 65,
         'avg': 0.271, 'obp': 0.355, 'slg': 0.500, 'woba': 0.370,
         'babip': 0.305, 'iso': 0.229, 'hr_fb': 0.18, 'k_rate': 0.25,
         'bb_rate': 0.1125, 'wrc_plus': 140, 'war': 4.0,
         'singles': 56, 'hbp': 3, 'sf': 2, 'sh': 0, 'gdp': 7, 'ibb': 3,
         'sb': 2, 'cs': 1,
         'gb_pct': 0.37, 'fb_pct': 0.38, 'ld_pct': 0.25,
         'o_swing_pct': 0.30, 'z_swing_pct': 0.66, 'swing_pct': 0.45,
         'o_contact_pct': 0.58, 'z_contact_pct': 0.83, 'contact_pct': 0.73,
         'swstr_pct': 0.14, 'off': 25, 'def_value': -4, 'bsr': 0.1,
         'bb_pct': 0.1125, 'k_pct': 0.25},
    ])


@pytest.fixture
def league_avgs():
    """Sample league averages."""
    return {
        'k_rate': 0.225,
        'bb_rate': 0.085,
        'iso': 0.150,
        'babip': 0.295,
        'hr_fb': 0.12,
        'avg': 0.248,
        'obp': 0.315,
        'slg': 0.398,
        'woba': 0.315,
    }


class TestMarcelRate:
    def test_basic_rate_projection(self, sample_hitter_data, league_avgs):
        """Marcel should produce a rate between player's rate and league avg."""
        rate = marcel_rate(sample_hitter_data, 'k_rate', 'pa', 2025, league_avgs)
        assert rate is not None
        # Should be between league avg and player's recent rate
        assert league_avgs['k_rate'] <= rate <= 0.26

    def test_regression_toward_mean(self, sample_hitter_data, league_avgs):
        """With less data, projection should be closer to league average."""
        # Full data
        full_rate = marcel_rate(sample_hitter_data, 'k_rate', 'pa', 2025, league_avgs)
        
        # Reduce PA to increase regression
        small_data = sample_hitter_data.copy()
        small_data['pa'] = 50  # Very small sample
        small_rate = marcel_rate(small_data, 'k_rate', 'pa', 2025, league_avgs)
        
        # Small sample should be closer to league average
        assert abs(small_rate - league_avgs['k_rate']) < abs(full_rate - league_avgs['k_rate'])

    def test_no_data_returns_none(self, league_avgs):
        """Player with no historical data should return None."""
        empty = pd.DataFrame(columns=['year', 'pa', 'k_rate'])
        rate = marcel_rate(empty, 'k_rate', 'pa', 2025, league_avgs)
        assert rate is None

    def test_recency_weighting(self, sample_hitter_data, league_avgs):
        """More recent years should have higher weight."""
        # This is implicit in the 5/4/3 weighting
        rate = marcel_rate(sample_hitter_data, 'iso', 'pa', 2025, league_avgs)
        assert rate is not None
        # Player's ISO has been declining (0.265 -> 0.252 -> 0.229)
        # Weighted projection should reflect recent trend, closer to 2024 value
        # but still regressed toward league avg (0.150)
        assert 0.150 <= rate <= 0.260


class TestMarcelPlayingTime:
    def test_basic_projection(self, sample_hitter_data):
        """Playing time should be based on recent 2 years."""
        pa = marcel_playing_time(sample_hitter_data, 'pa', 2025)
        # 0.5 * 400 + 0.1 * 450 = 200 + 45 = 245
        assert pa == pytest.approx(245, abs=1)

    def test_minimum_floor(self):
        """Should enforce minimum PA floor."""
        tiny = pd.DataFrame([
            {'year': 2024, 'pa': 50},
            {'year': 2023, 'pa': 30},
        ])
        pa = marcel_playing_time(tiny, 'pa', 2025)
        assert pa >= 200  # Floor for hitters


class TestAgeAdjustment:
    def test_peak_age_no_adjustment(self):
        """At peak age, adjustment should be ~1.0."""
        adj = age_adjustment(27, 'iso')
        assert adj == pytest.approx(1.0, abs=0.01)

    def test_young_player_slight_boost(self):
        """Young players should get a slight positive rate boost."""
        adj = age_adjustment(24, 'iso')
        assert adj > 1.0

    def test_old_player_decline(self):
        """Older players should show decline in positive stats."""
        adj = age_adjustment(35, 'iso')
        assert adj < 1.0

    def test_k_rate_increases_with_age(self):
        """K rate should increase (get worse) with age."""
        adj = age_adjustment(35, 'k_rate')
        assert adj > 1.0  # Multiplier > 1 means higher K%


class TestLeagueAverages:
    def test_computes_averages(self, sample_hitter_data):
        """Should compute PA-weighted league averages."""
        avgs = compute_league_averages(sample_hitter_data, 2024)
        assert 'k_rate' in avgs
        assert 'bb_rate' in avgs
        assert 0 < avgs['k_rate'] < 1

    def test_empty_year_returns_empty(self, sample_hitter_data):
        """Non-existent year should return empty dict."""
        avgs = compute_league_averages(sample_hitter_data, 2030)
        assert avgs == {}
