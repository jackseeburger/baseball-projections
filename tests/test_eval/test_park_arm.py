"""`marcel_tuned_park`, the pre-registered park arm (BAS-86).

No pymc, no network: the arm is `marcel_tuned` times a number, and every test
here is about which number, for whom, and where it came from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval import park_arm  # noqa: E402
from src.eval.backtest import COMPONENTS  # noqa: E402


def pa_rows(rows) -> pd.DataFrame:
    """(batter, date, home, away, topbot) tuples → a PA-shaped frame."""
    return pd.DataFrame(
        [{"batter": b, "game_date": pd.Timestamp(d), "home_team": h,
          "away_team": a, "inning_topbot": t} for b, d, h, a, t in rows])


PF = pd.DataFrame({
    "team": ["COL", "SF", "HOU"], "game_year": [2024, 2024, 2024],
    "k_park_factor": [0.85, 1.02, 1.00],
    "hr_park_factor": [1.20, 0.80, 1.05],
    "babip_park_factor": [1.10, 0.99, 1.00],
})


class TestHomeTeam:
    def test_the_club_with_the_most_pa_wins(self):
        rows = [(1, "2024-04-05", "COL", "SF", "Bot"),
                (1, "2024-04-06", "COL", "SF", "Bot"),
                (1, "2024-04-07", "SF", "COL", "Top")]
        assert park_arm.home_team_map(pa_rows(rows))[1] == "COL"

    def test_a_traded_batter_gets_the_club_he_played_most_for(self):
        rows = ([(1, "2024-04-05", "SF", "COL", "Top")]          # 1 PA for COL
                + [(1, f"2024-06-0{i}", "HOU", "SF", "Bot") for i in range(1, 5)])
        assert park_arm.home_team_map(pa_rows(rows))[1] == "HOU"

    def test_only_pa_before_the_cutoff_count(self):
        """The club a batter is traded to *after* the cutoff is not knowable
        at the cutoff, and an arm that used it would be reading the future."""
        rows = [(1, "2024-04-05", "COL", "SF", "Bot"),
                (1, "2024-08-05", "HOU", "SF", "Bot"),
                (1, "2024-08-06", "HOU", "SF", "Bot"),
                (1, "2024-08-07", "HOU", "SF", "Bot")]
        assert park_arm.home_team_map(pa_rows(rows), "2024-07-01")[1] == "COL"

    def test_a_batter_with_no_pa_before_the_cutoff_is_absent(self):
        rows = [(1, "2024-08-05", "COL", "SF", "Bot")]
        assert 1 not in park_arm.home_team_map(pa_rows(rows), "2024-07-01").index


class TestMultipliers:
    def test_one_season_and_one_component(self):
        assert park_arm.park_multipliers(PF, 2024, "hr_rate") == {
            "COL": 1.20, "SF": 0.80, "HOU": 1.05}

    def test_a_component_with_no_column_is_empty_not_borrowed(self):
        """ISO has no column in this table. Falling back to the HR factor
        would be a different measurement served under the ISO arm's name."""
        assert park_arm.park_multipliers(PF, 2024, "iso") == {}

    def test_a_season_the_table_does_not_cover_is_empty(self):
        assert park_arm.park_multipliers(PF, 2019, "hr_rate") == {}

    def test_no_table_at_all_is_empty(self):
        assert park_arm.park_multipliers(None, 2024, "hr_rate") == {}

    def test_the_column_names_match_the_model_registry(self):
        from src.models.pa_components import RATE_COMPONENTS

        for name, comp in RATE_COMPONENTS.items():
            assert park_arm.PARK_FACTOR_COLUMNS[name] == comp.park_factor_col

    def test_every_harness_hitter_component_is_named(self):
        from src.eval.backtest import HITTER_COMPONENTS

        assert set(park_arm.PARK_FACTOR_COLUMNS) == set(HITTER_COMPONENTS)


class TestProvider:
    def _train(self):
        return pd.DataFrame({
            "batter": [1, 2, 3] * 2,
            "season": [2023] * 3 + [2024] * 3,
            "pa": [500, 500, 500, 300, 300, 300],
            "hr": [20, 10, 15, 12, 6, 9],
            "partial": [False] * 3 + [True] * 3,
        })

    def _map(self):
        return pd.Series({1: "COL", 2: "SF", 3: "NYM"}, name="team")

    def test_the_prediction_is_the_base_times_the_park_factor(self):
        from src.eval.baselines import marcel_tuned

        spec = COMPONENTS["hr_rate"]
        train = self._train()
        base = marcel_tuned(train, spec, 2024).set_index("batter")["predicted"]
        arm = park_arm.park_provider(
            park_arm.park_multipliers(PF, 2024, "hr_rate"), self._map(),
        )(train, spec, 2024).set_index("batter")["predicted"]
        assert arm[1] == pytest.approx(base[1] * 1.20)
        assert arm[2] == pytest.approx(base[2] * 0.80)
        assert arm[3] == pytest.approx(base[3])  # NYM is not in the table

    def test_zero_strength_is_the_base_provider_exactly(self):
        from src.eval.baselines import marcel_tuned

        spec = COMPONENTS["hr_rate"]
        train = self._train()
        base = marcel_tuned(train, spec, 2024)
        arm = park_arm.park_provider(
            park_arm.park_multipliers(PF, 2024, "hr_rate"), self._map(),
            strength=0.0)(train, spec, 2024)
        assert arm["predicted"].to_numpy() == pytest.approx(
            base["predicted"].to_numpy())

    def test_half_strength_is_the_square_root_of_the_factor(self):
        from src.eval.baselines import marcel_tuned

        spec = COMPONENTS["hr_rate"]
        train = self._train()
        base = marcel_tuned(train, spec, 2024).set_index("batter")["predicted"]
        arm = park_arm.park_provider(
            park_arm.park_multipliers(PF, 2024, "hr_rate"), self._map(),
            strength=park_arm.HALF_STRENGTH)(train, spec, 2024
                                             ).set_index("batter")["predicted"]
        assert arm[1] == pytest.approx(base[1] * np.sqrt(1.20))

    def test_coverage_is_the_base_arm_s_coverage(self):
        """Adding this arm to a cell must not shrink the common-player set
        the other arms are scored on, so it has to project every batter the
        base projects — including the ones it has no park for."""
        from src.eval.baselines import marcel_tuned

        spec = COMPONENTS["hr_rate"]
        train = self._train()
        base = marcel_tuned(train, spec, 2024)
        arm = park_arm.park_provider({}, pd.Series(dtype="object"))(
            train, spec, 2024)
        assert set(arm["batter"]) == set(base["batter"])
        assert arm["predicted"].notna().all()

    def test_park_providers_is_empty_without_factors(self):
        pa = pa_rows([(1, "2024-04-05", "COL", "SF", "Bot")])
        assert park_arm.park_providers(PF, pa, "2024-07-01", 2019, "hr_rate") == {}
        got = park_arm.park_providers(PF, pa, "2024-07-01", 2024, "hr_rate")
        assert set(got) == {park_arm.ARM, park_arm.HALF_ARM}


class TestCoverage:
    def test_it_counts_the_batters_the_arm_actually_moved(self):
        team_map = pd.Series({1: "COL", 2: "NYM"}, name="team")
        cov = park_arm.coverage(team_map, {"COL": 1.2}, [1, 2, 3])
        assert cov == {"n": 3, "n_with_park": 1,
                       "share_with_park": pytest.approx(1 / 3),
                       "mean_abs_log_factor": pytest.approx(
                           abs(np.log(1.2)) / 3)}
