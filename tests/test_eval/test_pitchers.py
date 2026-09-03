"""Station A, pitcher side: the provider, the cutoff path, and the leakage guard.

The interesting claims are (1) the pitcher components ride the harness's own
estimator rather than a second one, (2) the cutoff path cannot see a batter
faced on or after the cutoff, and (3) the aggregation identities are the same
ones the hitter side uses, read off the other id on the row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import pitchers as P
from src.eval import tuning
from src.eval.backtest import COMPONENTS, backtest, score
from src.eval.baselines import MarcelParams
from src.eval.intraseason import assert_split_clean, build_training_frame

SEASON = 2026


# ─── fixtures ────────────────────────────────────────────────────

def pa_rows(pitcher: int, batter: int, date: str, n: int, k: int = 0,
            bb: int = 0, hbp: int = 0, hr: int = 0, hits: int = 0,
            game_pk: int = 1) -> pd.DataFrame:
    """`n` plate appearances on one date, with the given outcome counts."""
    rows = []
    for i in range(n):
        event = "field_out"
        is_k = int(i < k)
        is_bb = int(k <= i < k + bb)
        is_hbp = int(k + bb <= i < k + bb + hbp)
        is_hr = int(k + bb + hbp <= i < k + bb + hbp + hr)
        is_hit = int(k + bb + hbp <= i < k + bb + hbp + hits)
        if is_k:
            event = "strikeout"
        elif is_bb:
            event = "walk"
        elif is_hbp:
            event = "hit_by_pitch"
        elif is_hr:
            event = "home_run"
        elif is_hit:
            event = "single"
        rows.append({
            "batter": batter + i, "pitcher": pitcher, "game_pk": game_pk,
            "game_date": date, "game_year": SEASON, "event": event,
            "is_k": is_k, "is_bb": is_bb, "is_hbp": is_hbp, "is_hit": is_hit,
            "is_hr": is_hr, "is_single": int(is_hit and not is_hr),
            "is_double": 0, "is_triple": 0,
        })
    return pd.DataFrame(rows)


def season_row(pitcher: int, season: int, bf: int, k: int, bb: int, hr: int,
               age: float = 28.0) -> dict:
    """One pitcher-season in the table `normalize_pitcher_seasons` produces."""
    ab = bf - bb
    return {"pitcher": pitcher, "season": season, "bf": bf, "k": k, "bb": bb,
            "hbp": 0, "hr": hr, "ab": ab, "h": hr + 40, "sf": 0, "age": age}


def seasons_frame(rows) -> pd.DataFrame:
    return P.normalize_pitcher_seasons(pd.DataFrame(rows))


# ─── components and constants ────────────────────────────────────

def test_the_pitcher_components_are_registered_with_the_harness():
    for name in P.COMPONENT_ORDER:
        assert name in COMPONENTS
        assert COMPONENTS[name].id_col == "pitcher"
    # and the hitter ones are untouched
    assert COMPONENTS["k_rate"].id_col == "batter"


def test_stock_ballasts_are_twice_the_stabilization_point_in_real_trials():
    """`MarcelParams.ballast` is at the average year weight; the published
    points are real batters faced, which land on the most recent season."""
    for name, stab in P.STABILIZATION.items():
        params = P.PITCHER_STOCK_PARAMS[name]
        real = params.ballast * float(np.mean(params.weights)) / params.weights[0]
        assert real == pytest.approx(2.0 * stab)


def test_stock_has_no_age_term():
    """Station E never aged a pitcher, so stock must not either — every age
    effect in the tuned arm has to be something the search found."""
    for params in P.PITCHER_STOCK_PARAMS.values():
        assert params.age_slope_young == 0.0
        assert params.age_slope_old == 0.0


def test_the_aging_direction_flips_on_strikeouts_only():
    """A high K% is good for a pitcher and bad for a hitter; walks, homers and
    hits on balls in play are bad for the pitcher either way."""
    assert tuning.AGE_DIRECTION["p_k_rate"] == 1.0
    assert tuning.AGE_DIRECTION["k_rate"] == -1.0
    for name in ("p_bb_rate", "p_bbhbp_rate", "p_hr_rate", "p_babip"):
        assert tuning.AGE_DIRECTION[name] == -1.0


# ─── aggregation ─────────────────────────────────────────────────

def test_aggregation_counts_the_same_pa_from_the_other_end():
    pa = pd.concat([
        pa_rows(100, 500, "2026-04-01", 20, k=6, bb=2, hbp=1, hr=1, hits=4),
        pa_rows(100, 600, "2026-04-08", 10, k=3, bb=1),
    ], ignore_index=True)
    g = P.aggregate_pa_pitchers(pa, SEASON)
    assert len(g) == 1
    row = g.iloc[0]
    assert row["pitcher"] == 100
    assert row["bf"] == 30            # a batter faced is a PA seen from the mound
    assert row["k"] == 9
    assert row["bb"] == 3
    assert row["hbp"] == 1
    assert row["bbhbp"] == 4
    assert row["ab"] == 30 - 3 - 1    # BF - BB - HBP (no SF/SH/CI here)
    assert row["bip"] == row["ab"] - row["k"] - row["hr"] + row["sf"]
    assert row["hits_in_play"] == row["h"] - row["hr"]


def test_normalize_fills_the_derived_columns_the_api_table_lacks():
    raw = pd.DataFrame([{"pitcher": 1, "season": 2025, "bf": 700, "k": 180,
                         "bb": 50, "hbp": 5, "hr": 20, "ab": 640, "h": 150,
                         "sf": 4}])
    out = P.normalize_pitcher_seasons(raw)
    assert out.loc[0, "bbhbp"] == 55
    assert out.loc[0, "bip"] == 640 - 180 - 20 + 4
    assert out.loc[0, "hits_in_play"] == 130


# ─── the estimator ───────────────────────────────────────────────

def test_stock_pitcher_marcel_is_the_regression_it_claims_to_be():
    """One season, one pitcher: the closed form, with the ballast in real BF."""
    train = seasons_frame([season_row(1, 2025, 400, 120, 30, 12)])
    out = P.marcel_pitcher(train, COMPONENTS["p_k_rate"], 2026)
    ballast = 2.0 * P.STABILIZATION["p_k_rate"]          # 140 real BF
    lg = float(train["k"].sum() / train["bf"].sum())     # one pitcher *is* the league
    assert lg == pytest.approx(0.30)
    expected = (120 + ballast * lg) / (400 + ballast)
    assert out.loc[0, "predicted"] == pytest.approx(expected)


def test_a_pitcher_with_no_history_is_absent_not_league_average():
    train = seasons_frame([season_row(1, 2025, 400, 120, 30, 12)])
    out = P.marcel_pitcher(train, COMPONENTS["p_k_rate"], 2026)
    assert set(out["pitcher"]) == {1}


def test_the_tuned_arm_falls_back_to_stock_for_an_unfit_component(tmp_path):
    path = tmp_path / "params.json"
    P.save_pitcher_params({"p_k_rate": MarcelParams(ballast=1.0)}, path)
    loaded = P.load_pitcher_params(path)
    assert loaded["p_k_rate"].ballast == 1.0
    assert loaded["p_hr_rate"] == P.PITCHER_STOCK_PARAMS["p_hr_rate"]


def test_a_missing_params_file_is_stock_not_an_error(tmp_path):
    assert P.load_pitcher_params(tmp_path / "absent.json") == P.PITCHER_STOCK_PARAMS
    with pytest.raises(FileNotFoundError):
        P.load_pitcher_params(tmp_path / "absent.json", strict=True)


# ─── the cutoff path and its leakage guard ───────────────────────

def cutoff_frames(cutoff: str):
    """A pitcher with a clean April and a catastrophic July, split at `cutoff`."""
    pa = pd.concat([
        pa_rows(100, 500, "2026-04-10", 60, k=20, bb=4, game_pk=1),
        pa_rows(100, 700, "2026-04-20", 60, k=20, bb=4, game_pk=2),
        pa_rows(100, 900, "2026-07-10", 80, k=2, bb=30, hr=10, game_pk=3),
        pa_rows(101, 1100, "2026-04-11", 60, k=12, bb=8, game_pk=4),
        pa_rows(101, 1300, "2026-07-11", 80, k=16, bb=8, game_pk=5),
    ], ignore_index=True)
    return pa, P.partial_and_realized(pa, cutoff, SEASON)


def test_the_split_puts_every_pa_on_exactly_one_side_of_the_cutoff():
    pa, (partial, realized) = cutoff_frames("2026-06-01")
    assert partial["last_game_date"].max() < pd.Timestamp("2026-06-01")
    assert realized["first_game_date"].min() >= pd.Timestamp("2026-06-01")
    assert int(partial["bf"].sum()) + int(realized["bf"].sum()) == len(pa)


def test_post_cutoff_batters_faced_cannot_move_the_projection():
    """The claim in one assertion: delete every PA on or after the cutoff and
    the projection is bit-for-bit what it was."""
    prior = seasons_frame([season_row(100, 2025, 600, 180, 45, 18),
                           season_row(101, 2025, 600, 140, 55, 22)])
    pa, _ = cutoff_frames("2026-06-01")
    truncated = pa[pd.to_datetime(pa["game_date"]) < pd.Timestamp("2026-06-01")]

    def project(frame):
        partial, _ = P.partial_and_realized(frame, "2026-06-01", SEASON)
        train = build_training_frame(prior, partial, SEASON, "pitcher")
        return P.marcel_pitcher(train, COMPONENTS["p_k_rate"], SEASON)

    full = project(pa).set_index("pitcher")["predicted"]
    cut = project(truncated).set_index("pitcher")["predicted"]
    assert list(full.index) == list(cut.index)
    assert np.allclose(full.to_numpy(), cut.to_numpy(), rtol=0, atol=0)


def test_moving_the_cutoff_later_does_change_it():
    """The guard above would also pass if the provider ignored 2026 entirely."""
    prior = seasons_frame([season_row(100, 2025, 600, 180, 45, 18),
                           season_row(101, 2025, 600, 140, 55, 22)])
    pa, _ = cutoff_frames("2026-06-01")

    def project(cutoff):
        partial, _ = P.partial_and_realized(pa, cutoff, SEASON)
        train = build_training_frame(prior, partial, SEASON, "pitcher")
        return P.marcel_pitcher(train, COMPONENTS["p_k_rate"], SEASON
                                ).set_index("pitcher")["predicted"]

    june, august = project("2026-06-01"), project("2026-08-01")
    # 100's July was a disaster; seeing it must drop his projected K rate.
    assert august.loc[100] < june.loc[100]


def test_the_leakage_guard_rejects_a_training_row_from_after_the_cutoff():
    pa, (partial, realized) = cutoff_frames("2026-06-01")
    late, _ = P.partial_and_realized(pa, "2026-08-01", SEASON)   # sees July
    assert_split_clean(partial, realized, "2026-06-01", SEASON)  # the clean one
    with pytest.raises(ValueError, match="on or after the cutoff"):
        assert_split_clean(late, realized, "2026-06-01", SEASON)


def test_the_leakage_guard_rejects_realized_rows_from_before_the_cutoff():
    pa, (partial, _) = cutoff_frames("2026-06-01")
    _, early_realized = P.partial_and_realized(pa, "2026-04-15", SEASON)
    with pytest.raises(ValueError, match="realized row"):
        assert_split_clean(partial, early_realized, "2026-06-01", SEASON)


def test_the_backtest_cutoff_path_runs_end_to_end_on_pitchers():
    prior = seasons_frame([season_row(100, 2024, 600, 170, 45, 18),
                           season_row(101, 2024, 600, 130, 55, 22),
                           season_row(100, 2025, 600, 180, 45, 18),
                           season_row(101, 2025, 600, 140, 55, 22)])
    pa, _ = cutoff_frames("2026-06-01")
    results = backtest("p_k_rate", cutoff_date="2026-06-01", predict_year=SEASON,
                       seasons=prior, pa_frame=pa,
                       providers=dict(P.PITCHER_INTRASEASON_BASELINES),
                       min_trials=50)
    assert "pitcher" in results.columns and "batter" not in results.columns
    assert set(results["pitcher"]) == {100, 101}
    table = score(results)
    assert set(table["model"]) >= {"marcel_pitcher", "league_average",
                                   "previous_season", "season_to_date"}
    assert (table["mae"] >= 0).all()


def test_the_paired_test_pairs_on_pitchers():
    prior = seasons_frame([season_row(100, 2025, 600, 180, 45, 18),
                           season_row(101, 2025, 600, 140, 55, 22)])
    pa, _ = cutoff_frames("2026-06-01")
    results = backtest("p_k_rate", cutoff_date="2026-06-01", predict_year=SEASON,
                       seasons=prior, pa_frame=pa,
                       providers=dict(P.PITCHER_INTRASEASON_BASELINES),
                       min_trials=50)
    d = tuning.paired_from_results(results, "marcel_pitcher", "league_average")
    assert d["n"] == 2
    assert np.isfinite(d["diff"])


# ─── the shared rate table ───────────────────────────────────────

LEAGUE = {"rate_k": 0.22, "rate_bbhbp": 0.09, "rate_hr": 0.03, "bf_per_ip": 4.3}


def counts_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["pitcher", "season", "bf", "k",
                                       "bbhbp", "hr", "outs"])


def test_pitcher_rates_anchors_on_the_season_not_the_last_row():
    """Opening day: the partial season is empty, and the weights must not
    shift a slot — last year is still the second weight, not the first."""
    counts = counts_frame([[1, 2025, 600, 180, 54, 18, 1200],
                           [1, 2024, 600, 120, 54, 18, 1200]])
    rates = P.pitcher_rates(counts, 2026, LEAGUE)
    assert rates.loc[1, "bf_weighted"] == pytest.approx(600 * 0.8 + 600 * 0.6)
    expected = ((180 * 0.8 + 120 * 0.6 + 140 * LEAGUE["rate_k"])
                / (600 * 0.8 + 600 * 0.6 + 140))
    assert rates.loc[1, "rate_k"] == pytest.approx(expected)


def test_pitcher_rates_is_empty_without_a_usable_season():
    assert P.pitcher_rates(counts_frame([]), 2026, LEAGUE).empty
    old = counts_frame([[1, 2019, 600, 180, 54, 18, 1200]])
    assert P.pitcher_rates(old, 2026, LEAGUE).empty


# ─── the frozen file ─────────────────────────────────────────────

def test_the_committed_params_file_covers_every_component():
    import json

    params = P.load_pitcher_params(strict=True)
    for component in P.COMPONENT_ORDER:
        assert component in params
        p = params[component]
        assert p.ballast > 0
        assert sum(p.weights) > 0
        assert tuning.age_curve_ok(p, component), component
    blob = json.loads(P.PITCHER_PARAMS_PATH.read_text())
    assert set(blob["components"]) == set(P.COMPONENT_ORDER)
    assert blob["in_sample"].keys() == blob["components"].keys()


def test_the_guard_left_the_unfit_components_at_stock():
    """Three of the five did not beat stock on the inner validation, so the
    frozen file has to hold stock's constants for them — not some equally-good
    arbitrary point the search wandered to."""
    params = P.load_pitcher_params(strict=True)
    kept = [c for c in P.COMPONENT_ORDER
            if params[c] == P.PITCHER_STOCK_PARAMS[c]]
    assert set(kept) == {"p_bb_rate", "p_bbhbp_rate", "p_hr_rate"}


def test_station_e_reads_stock_constants_not_the_tuned_file():
    """The walks-plus-hit-batsmen component is the one station E consumes, and
    station E deliberately runs it on stock: a refit of station A must not move
    the odds without the odds being re-scored."""
    from src.sim import starters as st

    station_e = st.marcel_params()["p_bbhbp_rate"]
    assert station_e.ballast == P.PITCHER_STOCK_PARAMS["p_bbhbp_rate"].ballast
    assert station_e.weights == P.PITCHER_STOCK_PARAMS["p_bbhbp_rate"].weights
