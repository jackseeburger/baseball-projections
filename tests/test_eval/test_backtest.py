"""Tests for the backtest harness (roadmap 0.2) on synthetic seasons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval import backtest, calibration, score
from src.eval.backtest import COMPONENTS, parquet_provider
from src.eval.baselines import league_average, marcel, previous_season
from src.eval.metrics import binomial_log_loss, calibration_table, weighted_mae, weighted_rmse


@pytest.fixture
def seasons():
    """Three players with stable true K rates, seasons 2021-2024.

    Player 1: true .150, player 2: true .250, player 3: true .350.
    Counts are exact expectation so baseline behavior is deterministic.
    """
    rows = []
    for season in [2021, 2022, 2023, 2024]:
        for batter, rate in [(1, 0.15), (2, 0.25), (3, 0.35)]:
            pa = 600
            rows.append({
                "batter": batter, "season": season, "age": 24 + (season - 2021),
                "pa": pa, "k": int(round(rate * pa)),
            })
    return pd.DataFrame(rows)


class TestMetrics:
    def test_log_loss_perfect_vs_wrong(self):
        good = binomial_log_loss([0.25], [150], [600])
        bad = binomial_log_loss([0.10], [150], [600])
        assert good < bad

    def test_log_loss_matches_bernoulli_formula(self):
        p, k, n = 0.3, 30, 100
        expected = -(k * np.log(p) + (n - k) * np.log(1 - p)) / n
        assert binomial_log_loss([p], [k], [n]) == pytest.approx(expected)

    def test_weighted_mae_and_rmse(self):
        # errors 0.01 (w=100) and 0.03 (w=300) → MAE = 0.025
        assert weighted_mae([0.21, 0.27], [0.20, 0.30], [100, 300]) == pytest.approx(0.025)
        rmse = weighted_rmse([0.21, 0.27], [0.20, 0.30], [100, 300])
        assert rmse == pytest.approx(np.sqrt((100 * 0.01**2 + 300 * 0.03**2) / 400))

    def test_calibration_buckets_track_diagonal_for_perfect_preds(self):
        rng = np.random.default_rng(7)
        true = rng.uniform(0.1, 0.4, 500)
        tab = calibration_table(true, true, np.full(500, 600.0), n_bins=10)
        assert len(tab) == 10
        assert np.allclose(tab["mean_predicted"], tab["mean_realized"])


class TestBaselines:
    def test_previous_season_is_last_train_year_rate(self, seasons):
        spec = COMPONENTS["k_rate"]
        pred = previous_season(seasons[seasons.season <= 2023], spec, 2024)
        assert pred.set_index("batter").loc[1, "predicted"] == pytest.approx(0.15)

    def test_league_average_is_flat(self, seasons):
        spec = COMPONENTS["k_rate"]
        pred = league_average(seasons[seasons.season <= 2023], spec, 2024)
        assert pred["predicted"].nunique() == 1
        assert pred["predicted"].iloc[0] == pytest.approx(0.25)

    def test_marcel_regresses_extremes_toward_league(self, seasons):
        spec = COMPONENTS["k_rate"]
        pred = marcel(seasons[seasons.season <= 2023], spec, 2024).set_index("batter")
        # Low-K player pulled up toward .250, high-K player pulled down —
        # and the pull is symmetric before the (asymmetric) age adjustment.
        assert 0.15 < pred.loc[1, "predicted"] < 0.25
        assert 0.25 < pred.loc[3, "predicted"] < 0.35


class TestBacktest:
    def test_end_to_end_shapes_and_scores(self, seasons):
        results = backtest("k_rate", 2023, seasons=seasons)
        assert set(results["model"]) == {"marcel", "previous_season", "league_average"}
        scores = score(results)
        assert len(scores) == 3
        # Stable players: player-specific baselines must beat league average.
        s = scores.set_index("model")
        assert s.loc["previous_season", "mae"] < s.loc["league_average", "mae"]
        assert s.loc["previous_season", "log_loss"] < s.loc["league_average", "log_loss"]

    def test_providers_never_see_predict_year(self, seasons):
        seen = {}

        def spy(train, spec, predict_year):
            seen["max_season"] = int(train["season"].max())
            return previous_season(train, spec, predict_year)

        backtest("k_rate", 2023, seasons=seasons, providers={"spy": spy})
        assert seen["max_season"] == 2023

    def test_min_trials_filters_small_samples(self, seasons):
        tiny = seasons.copy()
        tiny.loc[(tiny.batter == 3) & (tiny.season == 2024), "pa"] = 30
        results = backtest("k_rate", 2023, seasons=tiny, min_trials=100)
        assert 3 not in results["batter"].values

    def test_rejects_backwards_predict_year(self, seasons):
        with pytest.raises(ValueError):
            backtest("k_rate", 2023, 2022, seasons=seasons)

    def test_calibration_from_results(self, seasons):
        results = backtest("k_rate", 2023, seasons=seasons)
        tab = calibration(results, "previous_season", n_bins=3)
        assert len(tab) == 3
        assert abs(tab["gap"]).max() < 0.01

    def test_parquet_provider(self, seasons, tmp_path):
        path = tmp_path / "preds.parquet"
        pd.DataFrame({"batter": [1, 2, 3], "proj_k": [0.15, 0.25, 0.35]}).to_parquet(path)
        results = backtest(
            "k_rate", 2023, seasons=seasons,
            providers={"bayes": parquet_provider(path, pred_col="proj_k")},
        )
        assert score(results).iloc[0]["mae"] == pytest.approx(0.0, abs=1e-9)
