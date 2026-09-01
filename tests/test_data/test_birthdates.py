"""Tests for Chadwick register birthdate handling (roadmap 0.1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.birthdates import (
    birth_year_map,
    build_batter_birth_years,
    parse_register,
    seasonal_age,
)


@pytest.fixture
def register():
    """Small register-shaped frame: full birthdate, year-only, and no-date rows."""
    return pd.DataFrame({
        "batter": [545361, 660271, 111111, 222222],
        "name_first": ["Mike", "Shohei", "Year", "Nodate"],
        "name_last": ["Trout", "Ohtani", "Only", "Player"],
        "birth_year": pd.array([1991, 1994, 1990, None], dtype="Int64"),
        "birth_month": pd.array([8, 7, None, None], dtype="Int64"),
        "birth_day": pd.array([7, 5, None, None], dtype="Int64"),
        "mlb_played_first": pd.array([2011, 2018, 2015, None], dtype="Int64"),
        "mlb_played_last": pd.array([2026, 2026, 2020, None], dtype="Int64"),
    })


class TestSeasonalAge:
    def test_full_birthdate_after_june30(self, register):
        # Trout born 1991-08-07: on 2021-06-30 he has not turned 30 yet.
        age = seasonal_age(register, np.array([545361]), 2021)
        assert 29.8 < age[0] < 30.0

    def test_full_birthdate_before_june30(self, register):
        # Born July 5 → just under N years on June 30. Ohtani on 2024-06-30
        # is a few days short of 30.
        age = seasonal_age(register, np.array([660271]), 2024)
        assert 29.9 < age[0] < 30.0

    def test_year_only_falls_back_to_integer_age(self, register):
        age = seasonal_age(register, np.array([111111]), 2020)
        assert age[0] == pytest.approx(30.0, abs=0.01)

    def test_unknown_batter_is_nan(self, register):
        age = seasonal_age(register, np.array([999999]), 2020)
        assert np.isnan(age[0])

    def test_vectorized_seasons(self, register):
        ages = seasonal_age(
            register, np.array([545361, 545361]), np.array([2020, 2021])
        )
        assert ages[1] - ages[0] == pytest.approx(1.0, abs=1e-9)


class TestBirthYearMap:
    def test_register_value_wins_over_fallback(self, register):
        first_year = pd.Series({545361: 2011, 111111: 2015})
        by = birth_year_map(register, fallback_first_year=first_year)
        assert by.loc[545361] == 1991
        assert by.loc[111111] == 1990

    def test_missing_id_uses_first_year_minus_23(self, register):
        first_year = pd.Series({999999: 2018})
        by = birth_year_map(register, fallback_first_year=first_year)
        assert by.loc[999999] == 2018 - 23

    def test_null_birth_year_uses_fallback(self, register):
        first_year = pd.Series({222222: 2019})
        by = birth_year_map(register, fallback_first_year=first_year)
        assert by.loc[222222] == 2019 - 23


class TestParseRegister:
    def test_dedupes_keeping_most_complete(self):
        raw = pd.DataFrame({
            "key_mlbam": pd.array([545361, 545361], dtype="Int64"),
            "name_first": ["Mike", "Mike"],
            "name_last": ["Trout", "Trout"],
            "birth_year": pd.array([None, 1991], dtype="Int64"),
            "birth_month": pd.array([None, 8], dtype="Int64"),
            "birth_day": pd.array([None, 7], dtype="Int64"),
            "mlb_played_first": pd.array([2011, 2011], dtype="Int64"),
            "mlb_played_last": pd.array([2026, 2026], dtype="Int64"),
        })
        out = parse_register(raw)
        assert len(out) == 1
        assert out.iloc[0]["birth_year"] == 1991
        assert out.iloc[0]["batter"] == 545361

    def test_drops_rows_without_mlbam(self):
        raw = pd.DataFrame({
            "key_mlbam": pd.array([None, 545361], dtype="Int64"),
            "name_first": ["Ghost", "Mike"],
            "name_last": ["Player", "Trout"],
            "birth_year": pd.array([1980, 1991], dtype="Int64"),
            "birth_month": pd.array([1, 8], dtype="Int64"),
            "birth_day": pd.array([1, 7], dtype="Int64"),
            "mlb_played_first": pd.array([None, 2011], dtype="Int64"),
            "mlb_played_last": pd.array([None, 2026], dtype="Int64"),
        })
        out = parse_register(raw)
        assert list(out["batter"]) == [545361]


class TestBuildBatterBirthYears:
    def test_schema_matches_modal_volume_file(self, register):
        first_year = pd.Series({545361: 2011, 999999: 2018})
        table = build_batter_birth_years(first_year, register)
        assert list(table.columns) == ["batter", "birth_year"]
        assert set(table["batter"]) == {545361, 999999}
        assert table.set_index("batter").loc[545361, "birth_year"] == 1991
