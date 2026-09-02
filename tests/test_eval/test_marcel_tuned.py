"""Tests for the tuned Marcel provider and the walk-forward tuner.

The load-bearing one is `test_stock_params_reproduce_marcel_exactly`: the
whole point of `marcel_tuned` is that it is stock Marcel with the constants
pulled out, so any measured difference between them is a difference in
*parameters*. If that test ever fails, every number in the tuning doc is
about the refactor rather than the fit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval import score, tuning
from src.eval.backtest import COMPONENTS, backtest
from src.eval.baselines import (
    MARCEL_BALLAST,
    STOCK_PARAMS,
    MarcelParams,
    load_marcel_params,
    marcel,
    marcel_tuned,
    marcel_tuned_preseason,
    marcel_tuned_provider,
    save_marcel_params,
    tuned_age_adjustment,
)
from src.models.marcel import age_adjustment

FLAT = MarcelParams(ballast=200.0, weights=(5.0, 4.0, 3.0), peak_age=27.0,
                    age_slope_young=0.0, age_slope_old=0.0)


@pytest.fixture
def seasons():
    """Five players with stable true K rates, seasons 2021-2024, exact counts.

    Ages are spread either side of 27 so the age curve is exercised, and the
    league rate is .250 in every season.
    """
    rows = []
    for season in [2021, 2022, 2023, 2024]:
        for batter, rate, age0 in [(1, 0.15, 22), (2, 0.20, 25), (3, 0.25, 27),
                                   (4, 0.30, 30), (5, 0.35, 33)]:
            pa = 600
            rows.append({
                "batter": batter, "season": season,
                "age": age0 + (season - 2021), "pa": pa,
                "k": int(round(rate * pa)), "hr": int(round(0.04 * pa)),
                "bb": int(round(0.09 * pa)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def spec():
    return COMPONENTS["k_rate"]


# --- the reduction ----------------------------------------------------------

class TestReducesToMarcel:
    def test_stock_params_reproduce_marcel_exactly(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        a = marcel(train, spec, 2024)
        b = marcel_tuned(train, spec, 2024, params=STOCK_PARAMS["k_rate"])
        assert list(a["batter"]) == list(b["batter"])
        # Bitwise, not approx: same arithmetic in the same order.
        assert (a["predicted"].to_numpy() == b["predicted"].to_numpy()).all()

    @pytest.mark.parametrize("component", sorted(COMPONENTS))
    def test_every_component_reduces(self, seasons, component):
        """Same check on all five components, with each one's own columns."""
        s = seasons.assign(ab=550, bip=400, hits_in_play=120, xb_points=180)
        cspec = COMPONENTS[component]
        train = s[s.season <= 2023]
        a = marcel(train, cspec, 2024)["predicted"].to_numpy()
        b = marcel_tuned(train, cspec, 2024,
                         params=STOCK_PARAMS[component])["predicted"].to_numpy()
        assert (a == b).all()

    @pytest.mark.parametrize("component", sorted(COMPONENTS))
    @pytest.mark.parametrize("age", list(range(19, 43)))
    def test_age_curve_matches_the_stock_one(self, component, age):
        stock = age_adjustment(age, component)
        tuned = float(tuned_age_adjustment(np.array([age]),
                                           STOCK_PARAMS[component])[0])
        assert tuned == stock

    def test_missing_age_gets_no_adjustment(self):
        adj = tuned_age_adjustment(np.array([np.nan, 30.0]),
                                   STOCK_PARAMS["k_rate"])
        assert adj[0] == 1.0
        assert adj[1] != 1.0

    def test_no_age_column_is_no_adjustment(self, seasons, spec):
        train = seasons[seasons.season <= 2023].drop(columns="age")
        steep = FLAT.replace(age_slope_young=0.05, age_slope_old=0.05)
        a = marcel_tuned(train, spec, 2024, params=FLAT)
        b = marcel_tuned(train, spec, 2024, params=steep)
        assert (a["predicted"].to_numpy() == b["predicted"].to_numpy()).all()


# --- the knobs --------------------------------------------------------------

class TestBallast:
    def test_bigger_ballast_pulls_extremes_toward_league(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        league = 0.25
        light = marcel_tuned(train, spec, 2024,
                             params=FLAT.replace(ballast=50.0)).set_index("batter")
        heavy = marcel_tuned(train, spec, 2024,
                             params=FLAT.replace(ballast=2000.0)).set_index("batter")
        for batter in (1, 5):   # the .150 and .350 players
            assert (abs(heavy.loc[batter, "predicted"] - league)
                    < abs(light.loc[batter, "predicted"] - league))
        # ...and a huge ballast takes everyone essentially to league.
        assert heavy["predicted"].std() < light["predicted"].std()

    def test_zero_ballast_is_the_raw_weighted_rate(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        pred = marcel_tuned(train, spec, 2024,
                            params=FLAT.replace(ballast=0.0)).set_index("batter")
        assert pred.loc[1, "predicted"] == pytest.approx(0.15)
        assert pred.loc[5, "predicted"] == pytest.approx(0.35)


class TestWeights:
    def test_zeroing_years_two_and_three_is_last_season_plus_regression(
            self, seasons, spec):
        """With (w1, 0, 0) only 2023 counts, and the estimator is exactly
        (s + b_eff*league) / (t + b_eff) on that season alone.

        b_eff is `ballast * mean(w)` divided by w1 — the ballast is quoted at
        the average year weight (see the module note in baselines.py), so
        zeroing two of three years leaves a third of it.
        """
        train = seasons[seasons.season <= 2023]
        params = FLAT.replace(weights=(1.0, 0.0, 0.0), ballast=300.0)
        pred = marcel_tuned(train, spec, 2024, params=params).set_index("batter")
        last = train[train.season == 2023].set_index("batter")
        b_eff = 300.0 * np.mean([1.0, 0.0, 0.0]) / 1.0
        expected = ((last["k"] + b_eff * 0.25) / (last["pa"] + b_eff))
        for batter in last.index:
            assert pred.loc[batter, "predicted"] == pytest.approx(expected[batter])

    def test_only_the_ratios_matter(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        a = marcel_tuned(train, spec, 2024,
                         params=FLAT.replace(weights=(5.0, 4.0, 3.0)))
        b = marcel_tuned(train, spec, 2024,
                         params=FLAT.replace(weights=(1.0, 0.8, 0.6)))
        assert np.allclose(a["predicted"], b["predicted"])

    def test_recency_moves_a_player_who_changed(self, spec):
        """A player who jumped in the last season is projected higher when
        the recent year carries more of the weight."""
        rows = []
        for season, rate in [(2021, 0.20), (2022, 0.20), (2023, 0.35)]:
            rows.append({"batter": 1, "season": season, "age": 27, "pa": 600,
                         "k": int(round(rate * 600))})
            rows.append({"batter": 2, "season": season, "age": 27, "pa": 600,
                         "k": 150})   # a .250 anchor so league stays .250-ish
        train = pd.DataFrame(rows)
        flat = marcel_tuned(train, spec, 2024,
                            params=FLAT.replace(weights=(1.0, 1.0, 1.0)))
        recent = marcel_tuned(train, spec, 2024,
                              params=FLAT.replace(weights=(1.0, 0.2, 0.0)))
        assert (recent.set_index("batter").loc[1, "predicted"]
                > flat.set_index("batter").loc[1, "predicted"])

    def test_all_zero_weights_is_an_error(self, seasons, spec):
        with pytest.raises(ValueError, match="positive year weight"):
            marcel_tuned(seasons, spec, 2025,
                         params=FLAT.replace(weights=(0.0, 0.0, 0.0)))


class TestAgeCurve:
    def test_sign_below_and_above_the_peak(self):
        rising = MarcelParams(peak_age=27.0, age_slope_young=0.002,
                              age_slope_old=0.005)
        assert tuned_age_adjustment(np.array([24.0]), rising)[0] < 1.0
        assert tuned_age_adjustment(np.array([27.0]), rising)[0] == 1.0
        assert tuned_age_adjustment(np.array([32.0]), rising)[0] > 1.0

        falling = rising.replace(age_slope_young=-0.002, age_slope_old=-0.005)
        assert tuned_age_adjustment(np.array([24.0]), falling)[0] > 1.0
        assert tuned_age_adjustment(np.array([32.0]), falling)[0] < 1.0

    def test_slopes_are_per_year(self):
        p = MarcelParams(peak_age=27.0, age_slope_young=0.0, age_slope_old=0.01)
        assert tuned_age_adjustment(np.array([30.0]), p)[0] == pytest.approx(1.03)

    def test_the_peak_moves(self):
        early = MarcelParams(peak_age=25.0, age_slope_young=0.0, age_slope_old=-0.01)
        late = early.replace(peak_age=31.0)
        assert tuned_age_adjustment(np.array([28.0]), early)[0] < 1.0
        assert tuned_age_adjustment(np.array([28.0]), late)[0] == 1.0

    def test_it_reaches_the_projection(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        flat = marcel_tuned(train, spec, 2024, params=FLAT).set_index("batter")
        aging = marcel_tuned(
            train, spec, 2024,
            params=FLAT.replace(age_slope_old=0.01, age_slope_young=0.01),
        ).set_index("batter")
        assert aging.loc[1, "predicted"] < flat.loc[1, "predicted"]    # age 24
        assert aging.loc[5, "predicted"] > flat.loc[5, "predicted"]    # age 35

    def test_fractional_ages_floor_like_marcel(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        fuzzy = train.assign(age=train["age"] + 0.4)
        a = marcel_tuned(train, spec, 2024, params=STOCK_PARAMS["k_rate"])
        b = marcel_tuned(fuzzy, spec, 2024, params=STOCK_PARAMS["k_rate"])
        assert (a["predicted"].to_numpy() == b["predicted"].to_numpy()).all()


# --- partial seasons and the harness ----------------------------------------

class TestPartialSeasons:
    def test_partial_season_is_treated_as_marcel_treats_it(self, seasons, spec):
        train = seasons[seasons.season <= 2023].copy()
        train["partial"] = False
        partial = pd.DataFrame([
            {"batter": b, "season": 2024, "age": 27, "pa": 200,
             "k": 40, "partial": True} for b in [1, 2, 3, 4, 5]
        ])
        train = pd.concat([train, partial], ignore_index=True)
        a = marcel(train, spec, 2024)
        b = marcel_tuned(train, spec, 2024, params=STOCK_PARAMS["k_rate"])
        assert (a["predicted"].to_numpy() == b["predicted"].to_numpy()).all()

    def test_preseason_arm_withholds_the_partial_season(self, seasons, spec):
        train = seasons[seasons.season <= 2023].copy()
        train["partial"] = False
        partial = pd.DataFrame([
            {"batter": b, "season": 2024, "age": 27, "pa": 400,
             "k": 20, "partial": True} for b in [1, 2, 3, 4, 5]
        ])
        withheld = marcel_tuned_preseason(
            pd.concat([train, partial], ignore_index=True), spec, 2024, params=FLAT)
        control = marcel_tuned(train, spec, 2024, params=FLAT)
        assert np.allclose(withheld["predicted"], control["predicted"])


class TestHarnessIntegration:
    def test_provider_runs_through_backtest(self, seasons, spec):
        s = pd.concat([seasons, seasons.assign(season=2025)], ignore_index=True)
        results = backtest(
            "k_rate", 2024, 2025, seasons=s,
            providers={"marcel": marcel,
                       "tuned": marcel_tuned_provider(STOCK_PARAMS)},
        )
        table = score(results).set_index("model")
        assert table.loc["marcel", "mae"] == pytest.approx(table.loc["tuned", "mae"])

    def test_tuning_scoring_path_matches_the_harness(self, seasons, spec):
        s = pd.concat([seasons, seasons.assign(season=2025)], ignore_index=True)
        split = tuning.make_split(s, spec, 2025)
        mine = tuning.score_split(split, spec, STOCK_PARAMS["k_rate"])
        theirs = score(backtest("k_rate", 2024, 2025, seasons=s,
                                providers={"marcel": marcel})).iloc[0]
        assert mine["mae"] == pytest.approx(theirs["mae"])
        assert mine["log_loss"] == pytest.approx(theirs["log_loss"])
        assert mine["n_players"] == theirs["n_players"]


# --- the params file ---------------------------------------------------------

class TestParamsFile:
    def test_round_trip(self, tmp_path):
        params = {
            "k_rate": MarcelParams(120.0, (1.0, 0.4, 0.2), 29.0, 0.004, 0.006),
            "babip": MarcelParams(700.0, (1.0, 1.0, 0.8), 26.0, -0.002, -0.001),
        }
        path = save_marcel_params(params, tmp_path / "p.json", generated="test")
        back = load_marcel_params(path)
        for name, p in params.items():
            assert back[name] == p
        assert json.loads(path.read_text())["generated"] == "test"

    def test_unfit_components_fall_back_to_stock(self, tmp_path):
        path = save_marcel_params(
            {"k_rate": MarcelParams(120.0, (1.0, 0.4, 0.2), 29.0, 0.004, 0.006)},
            tmp_path / "p.json")
        back = load_marcel_params(path)
        assert back["bb_rate"] == STOCK_PARAMS["bb_rate"]

    def test_missing_file_is_stock_unless_strict(self, tmp_path):
        assert load_marcel_params(tmp_path / "absent.json") == STOCK_PARAMS
        with pytest.raises(FileNotFoundError):
            load_marcel_params(tmp_path / "absent.json", strict=True)

    def test_the_committed_file_has_every_component(self):
        params = load_marcel_params(strict=True)
        for component in COMPONENTS:
            assert component in params
            p = params[component]
            assert p.ballast > 0
            assert sum(p.weights) > 0
            assert 20 <= p.peak_age <= 40
        blob = json.loads(
            (Path(__file__).parent.parent.parent
             / "src/eval/marcel_params.json").read_text())
        assert set(blob["components"]) == set(COMPONENTS)
        assert blob["in_sample"].keys() == blob["components"].keys()

    def test_default_params_come_from_the_committed_file(self, seasons, spec):
        train = seasons[seasons.season <= 2023]
        default = marcel_tuned(train, spec, 2024)
        explicit = marcel_tuned(train, spec, 2024,
                                params=load_marcel_params()["k_rate"])
        assert (default["predicted"].to_numpy()
                == explicit["predicted"].to_numpy()).all()


# --- the paired statistic ----------------------------------------------------

class TestPairedDiff:
    def test_sign_and_zero(self):
        realized = pd.DataFrame({"batter": [1, 2, 3],
                                 "realized_rate": [0.2, 0.3, 0.4],
                                 "trials": [600.0, 600.0, 600.0]})
        good = realized.assign(predicted=realized["realized_rate"] + 0.01)
        bad = realized.assign(predicted=realized["realized_rate"] + 0.03)
        d = tuning.paired_abs_error_diff(good, bad)
        assert d["diff"] == pytest.approx(-0.02)
        assert d["n"] == 3
        assert d["win_rate"] == pytest.approx(1.0)
        assert tuning.paired_abs_error_diff(good, good)["diff"] == 0.0

    def test_only_common_players_are_paired(self):
        a = pd.DataFrame({"batter": [1, 2], "predicted": [0.2, 0.2],
                          "realized_rate": [0.2, 0.2], "trials": [600.0, 600.0]})
        b = pd.DataFrame({"batter": [2, 3], "predicted": [0.3, 0.3],
                          "realized_rate": [0.2, 0.2], "trials": [600.0, 600.0]})
        assert tuning.paired_abs_error_diff(a, b)["n"] == 1

    def test_se_shrinks_with_more_players(self):
        rng = np.random.default_rng(11)
        n = 400
        realized = pd.DataFrame({
            "batter": np.arange(n), "realized_rate": rng.uniform(0.15, 0.35, n),
            "trials": np.full(n, 600.0)})
        a = realized.assign(predicted=realized["realized_rate"]
                            + rng.normal(0, 0.01, n))
        b = realized.assign(predicted=realized["realized_rate"]
                            + rng.normal(0, 0.02, n))
        big = tuning.paired_abs_error_diff(a, b)
        small = tuning.paired_abs_error_diff(a.head(40), b.head(40))
        assert big["diff"] < 0
        assert big["se"] < small["se"]


# --- the search ---------------------------------------------------------------

class TestSearch:
    def test_it_finds_a_planted_ballast(self, spec):
        """A league where every player's true rate IS the league rate wants a
        very large ballast; the search should move that way from stock."""
        rng = np.random.default_rng(3)
        rows = []
        for season in range(2019, 2025):
            for batter in range(120):
                pa = 600
                rows.append({"batter": batter, "season": season, "age": 27,
                             "pa": pa, "k": int(rng.binomial(pa, 0.25))})
        seasons = pd.DataFrame(rows)
        splits = tuning.make_splits(seasons, "k_rate", [2023, 2024])
        best, trace = tuning.coordinate_search(
            splits, spec, start=STOCK_PARAMS["k_rate"], passes=1,
            axes=["ballast"])
        assert best.ballast > MARCEL_BALLAST
        assert trace[-1]["mae"] <= trace[0]["mae"]

    def test_search_never_returns_something_worse(self, seasons, spec):
        s = pd.concat([seasons, seasons.assign(season=2025)], ignore_index=True)
        splits = tuning.make_splits(s, "k_rate", [2024, 2025])
        stock = STOCK_PARAMS["k_rate"]
        best, _ = tuning.coordinate_search(splits, spec, start=stock, passes=1)
        assert (tuning.evaluate(splits, spec, best)["mae"]
                <= tuning.evaluate(splits, spec, stock)["mae"])
