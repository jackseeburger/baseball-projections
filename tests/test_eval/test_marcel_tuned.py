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
from src.eval.backtest import COMPONENTS, HITTER_COMPONENTS, backtest
from src.eval.baselines import (
    MARCEL_BALLAST,
    STOCK_PARAMS,
    MarcelParams,
    load_marcel_params,
    marcel,
    marcel_tuned,
    marcel_tuned_preseason,
    marcel_tuned_provider,
    projected_league_rate,
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

    @pytest.mark.parametrize("component", sorted(HITTER_COMPONENTS))
    def test_every_component_reduces(self, seasons, component):
        """Same check on all five components, with each one's own columns."""
        s = seasons.assign(ab=550, bip=400, hits_in_play=120, xb_points=180)
        cspec = COMPONENTS[component]
        train = s[s.season <= 2023]
        a = marcel(train, cspec, 2024)["predicted"].to_numpy()
        b = marcel_tuned(train, cspec, 2024,
                         params=STOCK_PARAMS[component])["predicted"].to_numpy()
        assert (a == b).all()

    @pytest.mark.parametrize("component", sorted(HITTER_COMPONENTS))
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


class TestProjectedLeagueRate:
    """The three "regress toward what?" options, on a frame whose answer is
    arithmetic rather than a fit.

    Three seasons, one player each — so every season's league rate *is* that
    player's rate — and the rates climb by exactly .010 a year: 2021 .200,
    2022 .210, 2023 .220. The trials differ per season so a wrong weighting
    shows up as a wrong number rather than as the same number by luck.
    """

    @pytest.fixture
    def trend(self):
        return pd.DataFrame([
            {"batter": 1, "season": 2021, "age": 27, "pa": 1000, "k": 200},
            {"batter": 1, "season": 2022, "age": 28, "pa": 500, "k": 105},
            {"batter": 1, "season": 2023, "age": 29, "pa": 2000, "k": 440},
        ])

    def test_last_is_the_most_recent_season(self, trend, spec):
        p = FLAT.replace(league_mode="last")
        assert projected_league_rate(trend, spec, p, 2024) == pytest.approx(0.220)

    def test_weighted3_uses_the_components_own_recency_weights(self, trend, spec):
        p = FLAT.replace(league_mode="weighted3", weights=(3.0, 2.0, 1.0))
        # (3*440 + 2*105 + 1*200) / (3*2000 + 2*500 + 1*1000) = 1730/8000
        assert projected_league_rate(trend, spec, p, 2024) == pytest.approx(1730 / 8000)

    def test_weighted3_with_flat_weights_is_the_pooled_three_year_rate(
            self, trend, spec):
        p = FLAT.replace(league_mode="weighted3", weights=(1.0, 1.0, 1.0))
        assert projected_league_rate(trend, spec, p, 2024) == pytest.approx(
            745 / 3500)

    @pytest.mark.parametrize("damp,expected", [
        (0.0, 0.220), (0.5, 0.225), (1.0, 0.230),
    ])
    def test_drift_extrapolates_the_one_season_change(self, trend, spec,
                                                      damp, expected):
        p = FLAT.replace(league_mode="drift", league_damp=damp)
        assert projected_league_rate(trend, spec, p, 2024) == pytest.approx(expected)

    def test_drift_scales_with_the_horizon(self, trend, spec):
        """Two years out is two years of drift; zero years out is none."""
        p = FLAT.replace(league_mode="drift", league_damp=1.0)
        assert projected_league_rate(trend, spec, p, 2025) == pytest.approx(0.240)
        assert projected_league_rate(trend, spec, p, 2023) == pytest.approx(0.220)

    def test_drift_needs_a_previous_season(self, spec):
        one = pd.DataFrame([{"batter": 1, "season": 2023, "age": 27,
                             "pa": 1000, "k": 220}])
        p = FLAT.replace(league_mode="drift", league_damp=1.0)
        assert projected_league_rate(one, spec, p, 2024) == pytest.approx(0.220)

    def test_an_unknown_mode_is_an_error(self, trend, spec):
        with pytest.raises(ValueError, match="unknown league_mode"):
            projected_league_rate(trend, spec, FLAT.replace(league_mode="ouija"),
                                  2024)

    def test_the_mode_reaches_the_projection(self, seasons, spec):
        """A rising league pulls every projection up, because the ballast is
        league average and league average is what moved."""
        train = seasons[seasons.season <= 2023].copy()
        # Make the league rate climb: scale each season's strikeouts.
        train["k"] = (train["k"] * (1.0 + 0.1 * (train["season"] - 2021))).round()
        last = marcel_tuned(train, spec, 2024,
                            params=FLAT.replace(league_mode="last"))
        drift = marcel_tuned(train, spec, 2024,
                             params=FLAT.replace(league_mode="drift",
                                                 league_damp=1.0))
        assert (drift["predicted"].to_numpy() > last["predicted"].to_numpy()).all()

    def test_stock_params_still_mean_last_season(self, seasons, spec):
        assert STOCK_PARAMS["k_rate"].league_mode == "last"
        train = seasons[seasons.season <= 2023]
        assert projected_league_rate(train, spec, STOCK_PARAMS["k_rate"],
                                     2024) == pytest.approx(0.25)


class TestAgeConstraint:
    """The tuner's age term cannot land outside the constrained family.

    The unconstrained fit put the peak at a grid end (23 or 31) with equal
    slopes either side — a straight line in age that was half level
    correction. These are the rules that make that shape unreachable.
    """

    @pytest.mark.parametrize("component", sorted(HITTER_COMPONENTS))
    def test_every_candidate_the_search_can_propose_is_inside(self, component):
        start = tuning.constrain(STOCK_PARAMS[component], component)
        for axis in tuning.AXES:
            for cand in tuning._candidates(axis, start, component,
                                           constrained=True):
                assert tuning.age_curve_ok(cand, component), (axis, cand)

    @pytest.mark.parametrize("component", sorted(HITTER_COMPONENTS))
    def test_a_fit_lands_inside_the_window(self, component):
        """The real search on real-shaped noise, from a start outside the
        family, still returns something inside it."""
        rng = np.random.default_rng(7)
        rows = []
        for season in range(2019, 2025):
            for batter in range(80):
                pa = 600
                rows.append({"batter": batter, "season": season,
                             "age": 20 + batter % 20, "pa": pa,
                             "k": int(rng.binomial(pa, 0.24)),
                             "bb": int(rng.binomial(pa, 0.09)),
                             "hr": int(rng.binomial(pa, 0.035)),
                             "ab": 550, "bip": 400,
                             "hits_in_play": int(rng.binomial(400, 0.29)),
                             "xb_points": int(rng.binomial(550, 0.16))})
        frame = pd.DataFrame(rows)
        spec = COMPONENTS[component]
        splits = tuning.make_splits(frame, component, [2023, 2024])
        # Start from a deliberately illegal age term: peak off the window with
        # both slopes the same sign, i.e. the straight line we are outlawing.
        bad = STOCK_PARAMS[component].replace(
            peak_age=23.0, age_slope_young=0.012, age_slope_old=0.012)
        assert not tuning.age_curve_ok(bad, component)
        best, _ = tuning.coordinate_search(splits, spec, start=bad, passes=1)
        assert tuning.age_curve_ok(best, component)
        lo, hi = tuning.AGE_PEAK_WINDOW
        assert lo <= best.peak_age <= hi

    def test_the_slopes_turn_the_curve_over(self):
        """Inside the family the multiplier has a genuine extremum at the peak
        — it cannot be monotone across the age range, which is the shape that
        doubles as a level."""
        for component in HITTER_COMPONENTS:
            for young in tuning.constrained_slope_grid(component, "young"):
                for old in tuning.constrained_slope_grid(component, "old"):
                    p = MarcelParams(peak_age=28.0, age_slope_young=young,
                                     age_slope_old=old)
                    adj = tuned_age_adjustment(np.arange(20.0, 41.0), p)
                    peak = tuned_age_adjustment(np.array([28.0]), p)[0]
                    assert peak == 1.0
                    if tuning.AGE_DIRECTION[component] > 0:
                        assert adj.max() == pytest.approx(1.0)
                    else:
                        assert adj.min() == pytest.approx(1.0)

    def test_k_rate_is_mirrored(self):
        """K% is the component where a bigger number is a worse hitter, so its
        constrained curve troughs at the peak age instead of cresting."""
        assert tuning.AGE_DIRECTION["k_rate"] < 0
        assert all(s <= 0 for s in tuning.constrained_slope_grid("k_rate", "young"))
        assert all(s >= 0 for s in tuning.constrained_slope_grid("k_rate", "old"))
        assert all(s >= 0 for s in tuning.constrained_slope_grid("iso", "young"))
        assert all(s <= 0 for s in tuning.constrained_slope_grid("iso", "old"))

    def test_constrain_is_idempotent_and_clips(self):
        p = MarcelParams(peak_age=23.0, age_slope_young=0.012,
                         age_slope_old=0.012)
        once = tuning.constrain(p, "iso")
        assert once.peak_age == tuning.AGE_PEAK_WINDOW[0]
        assert once.age_slope_young == 0.012      # right sign for iso, kept
        assert once.age_slope_old == 0.0          # wrong sign, zeroed
        assert tuning.constrain(once, "iso") == once

    def test_the_frozen_age_curves_are_inside_the_family(self):
        """Whatever is committed must satisfy the rule it was fit under —
        except a component the guard sent back to stock, whose curve is
        Tango's monotone one by design."""
        fitted = load_marcel_params(strict=True)
        for component, p in fitted.items():
            if p == STOCK_PARAMS[component]:
                continue
            assert tuning.age_curve_ok(p, component), component

    def test_unconstrained_search_can_still_leave_the_window(self):
        """The constraint is a choice, not a law of the code: the old
        behaviour is still reachable, which is what makes the comparison in
        the doc runnable."""
        cands = tuning._candidates("peak_age", STOCK_PARAMS["iso"], "iso",
                                   constrained=False)
        assert any(c.peak_age < tuning.AGE_PEAK_WINDOW[0] for c in cands)


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

    def test_the_league_mode_round_trips(self, tmp_path):
        params = {"iso": MarcelParams(300.0, (1.0, 0.6, 0.4), 25.0, 0.0, -0.012,
                                      league_mode="drift", league_damp=0.5)}
        back = load_marcel_params(save_marcel_params(params, tmp_path / "p.json"))
        assert back["iso"] == params["iso"]

    def test_a_file_written_before_league_mode_existed_still_loads(self, tmp_path):
        """Back-compat: no `league_mode` key means stock Marcel's "last", so
        an older params file means exactly what it meant when it was written."""
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"components": {"k_rate": {
            "ballast": 100.0, "weights": [1.0, 0.4, 0.2], "peak_age": 31.0,
            "age_slope_young": 0.006, "age_slope_old": 0.012}}}))
        p = load_marcel_params(path)["k_rate"]
        assert p.league_mode == "last"
        assert p.league_damp == 0.0

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
        # HITTER_COMPONENTS, not COMPONENTS: importing src.eval.pitchers adds
        # the pitcher components to the shared registry, and they are fitted
        # and frozen separately (src/eval/marcel_pitcher_params.json).
        params = load_marcel_params(strict=True)
        for component in HITTER_COMPONENTS:
            assert component in params
            p = params[component]
            assert p.ballast > 0
            assert sum(p.weights) > 0
            assert 20 <= p.peak_age <= 40
        blob = json.loads(
            (Path(__file__).parent.parent.parent
             / "src/eval/marcel_params.json").read_text())
        assert set(blob["components"]) == set(HITTER_COMPONENTS)
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
