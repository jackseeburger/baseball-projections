"""The verdicts on docs/park-factors.md's four predictions (BAS-86).

A verdict function is exactly the kind of code that can invert silently — a
sign, a `<` for a `>`, a "held" printed next to numbers that say the opposite —
and be believed anyway, because it is the last thing anybody reads. So each
prediction is exercised in both directions here, on synthetic inputs shaped
like the real sidecar and the real analysis payload.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "score_park_factors", ROOT / "scripts/score_park_factors.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score_mod = _load()


def meta(hr_corr=0.92, k_corr=0.95, hr_sd=0.10):
    return {
        "built_at": "2026-09-10T00:00:00+00:00",
        "ballast": {"hr_rate": 3000.0},
        "log_factor_sd_by_season": [
            {"component": "hr_rate", "game_year": y, "sd": hr_sd, "n_parks": 30}
            for y in (2022, 2024, 2025, 2026)],
        "year_over_year_correlation": [
            {"component": "hr_rate", "season": None, "corr": hr_corr},
            {"component": "k_rate", "season": None, "corr": k_corr},
            {"component": "hr_rate", "season": 2025, "corr": 0.9},
        ],
        "single_season_persistence": [
            {"component": "hr_rate", "corr": 0.44},
            {"component": "k_rate", "corr": 0.70},
        ],
        "loso_persistence": [{"component": "hr_rate", "ballast": 3000.0,
                              "season": None, "rmse": 0.10, "corr": 0.75}],
    }


def arm_row(component, diff, pct, t, season=None, arm="marcel_tuned_park"):
    return {
        "arm": arm, "base": "marcel_tuned", "component": component,
        "diff": diff, "pct_of_base": pct, "n": 14294,
        "base_mae": 0.01, "arm_mae": 0.01 + diff,
        "clustered_by_player_t": t, "clustered_by_cell_t": t * 3,
        "clustered_by_player_n_clusters": 700,
        "arm_wins_cells": 0, "arm_loses_cells": 48, "season": season,
    }


def analysis(hr=(0.0003, 3.36, 4.39), k=(0.002, 6.83, 5.21),
             bb=(0.0003, 1.43, 3.04)):
    pooled = [arm_row("hr_rate", *hr), arm_row("k_rate", *k),
              arm_row("bb_rate", *bb)]
    by_season = [arm_row("hr_rate", *hr, season=s) for s in (2022, 2024)]
    return {"park_arm": {"pooled": pooled, "by_season": by_season,
                         "base": "marcel_tuned",
                         "tilt": [{"component": "hr_rate", "n": 14294,
                                   "sd_log_factor": 0.11,
                                   "slope_realized": 0.21, "slope_base": 0.24,
                                   "slope_arm": 1.22,
                                   "excess_over_realized": 1.01,
                                   "base_share_of_realized": 1.14}]}}


class TestVacuity:
    def test_a_spread_above_the_bar_is_not_vacuous(self):
        out = score_mod.score(meta(hr_sd=0.10), analysis())
        assert out["vacuity"]["verdict"] == "not vacuous"

    def test_a_flat_factor_is_vacuous(self):
        out = score_mod.score(meta(hr_sd=0.01), analysis())
        assert out["vacuity"]["verdict"] == "VACUOUS"


class TestPredictionOne:
    def test_it_holds_above_both_bars(self):
        out = score_mod.score(meta(), analysis())
        p1 = out["predictions"]["1_persistence"]
        assert p1["verdict"] == "held"
        assert p1["passes"] == {"hr_rate": True, "k_rate": True}

    def test_hr_below_its_bar_fails_it(self):
        out = score_mod.score(meta(hr_corr=0.55), analysis())
        assert out["predictions"]["1_persistence"]["verdict"] == "FAILED"

    def test_the_harsher_single_season_reading_rides_along(self):
        p1 = score_mod.score(meta(), analysis())["predictions"]["1_persistence"]
        assert p1["single_season_raw_corr"] == {"hr_rate": 0.44, "k_rate": 0.70}


class TestPredictionTwo:
    def test_a_real_gain_holds_it(self):
        out = score_mod.score(meta(), analysis(hr=(-0.0001, -0.9, -3.0)))
        assert out["predictions"]["2_hr_gain"]["verdict"] == "held"

    def test_a_gain_too_small_fails_it(self):
        out = score_mod.score(meta(), analysis(hr=(-0.00001, -0.2, -3.0)))
        assert out["predictions"]["2_hr_gain"]["verdict"] == "FAILED"

    def test_a_gain_without_significance_fails_it(self):
        out = score_mod.score(meta(), analysis(hr=(-0.0001, -0.9, -1.2)))
        assert out["predictions"]["2_hr_gain"]["verdict"] == "FAILED"

    def test_an_arm_that_is_worse_says_so_rather_than_just_failing(self):
        """The distinction that matters when reading the verdict later: the
        arm did not merely fall short of the bar, it lost."""
        out = score_mod.score(meta(), analysis())
        assert out["predictions"]["2_hr_gain"]["verdict"] == (
            "FAILED (the arm is worse than the baseline)")


class TestPredictionThree:
    def test_two_small_deltas_hold_it(self):
        out = score_mod.score(meta(), analysis(k=(0.0, 0.1, 0.3),
                                               bb=(0.0, -0.2, -0.5)))
        assert out["predictions"]["3_k_bb_null"]["verdict"] == "held"

    def test_one_big_delta_fails_it(self):
        out = score_mod.score(meta(), analysis(k=(0.002, 6.83, 5.21),
                                               bb=(0.0, -0.2, -0.5)))
        p3 = out["predictions"]["3_k_bb_null"]
        assert p3["verdict"] == "FAILED"
        assert p3["components"]["k_rate"]["within_bar"] is False
        assert p3["components"]["bb_rate"]["within_bar"] is True


class TestPredictionFour:
    def test_a_gain_under_the_floor_holds_it(self):
        out = score_mod.score(meta(), analysis(hr=(-0.0001, -0.8, -3.0)))
        p4 = out["predictions"]["4_serving_floor"]
        assert p4["clears_the_floor"] is False
        assert p4["verdict"] == "held (does not clear)"

    def test_a_gain_over_the_floor_fails_it(self):
        out = score_mod.score(meta(), analysis(hr=(-0.0002, -1.5, -3.0)))
        p4 = out["predictions"]["4_serving_floor"]
        assert p4["clears_the_floor"] is True
        assert p4["verdict"].startswith("FAILED")

    def test_no_gain_at_all_is_held_vacuously_and_says_which(self):
        """Holding because the arm lost is not the same claim as holding
        because a real gain was too small to serve, and a verdict that read
        the same either way would be the misleading half of this file."""
        out = score_mod.score(meta(), analysis())
        assert out["predictions"]["4_serving_floor"]["verdict"].startswith(
            "held vacuously")


class TestPayload:
    def test_a_payload_with_no_park_arm_is_refused(self):
        with pytest.raises(SystemExit):
            score_mod.score(meta(), {"cheap_scope": {}})

    def test_the_evidence_carries_the_tilt_and_the_scope(self):
        out = score_mod.score(meta(), analysis())
        assert out["park_tilt"][0]["excess_over_realized"] == 1.01
        assert out["scope"]["seasons"] == [2022, 2024]
        assert set(out["scope"]["components"]) == {"hr_rate", "k_rate", "bb_rate"}

    def test_coverage_is_summarised_per_component_and_season(self):
        coverage = [
            {"component": "hr_rate", "season": 2024, "cutoff": c, "n": 100,
             "n_with_park": 90, "share_with_park": 0.9,
             "mean_abs_log_factor": 0.10}
            for c in ("2024-05-01", "2024-07-01")]
        out = score_mod.score(meta(), analysis(), coverage)
        assert out["arm_coverage"] == [{
            "component": "hr_rate", "season": 2024, "cells": 2, "n": 200,
            "n_with_park": 180, "share_with_park": 0.9,
            "mean_abs_log_factor": pytest.approx(0.10)}]

    def test_render_mentions_every_prediction(self):
        out = score_mod.score(meta(), analysis())
        text = score_mod.render(out)
        for key in out["predictions"]:
            assert key in text
