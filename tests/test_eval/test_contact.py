"""Contact-quality covariates: the leakage guard first, then the estimator.

The leakage test is the one that matters. `synthetic_monthly` builds a season
where every bucket at or after the cutoff is an *extreme* — a thousand batted
balls, all barrels, all at 120 mph — so any post-cutoff row that reaches the
feature moves it enormously and no rounding or off-by-one can hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.contact_quality import COUNT_COLUMNS, EV_BIN_COLUMNS
from src.eval.backtest import COMPONENTS
from src.eval.contact import (
    ContactProviderConfig,
    assert_month_boundary,
    assert_window_clean,
    contact_metrics,
    contact_provider,
    features_at_cutoff,
    fit_contact,
    standardize,
    window_counts,
)
from src.eval.tuning import paired_abs_error_diff


def bucket(side, player, season, month, bbe, ev, la, barrel_frac=0.0,
           ev_bin=8):
    """One monthly bucket with hand-set sufficient statistics."""
    row = {"side": side, "player": player, "season": season, "month": month}
    row.update({c: 0.0 for c in COUNT_COLUMNS})
    row["bbe"] = float(bbe)
    row["sum_ev"] = float(bbe) * ev
    row["sum_ev2"] = float(bbe) * ev * ev
    row["sum_la"] = float(bbe) * la
    row["n_barrel"] = float(bbe) * barrel_frac
    row["n_hardhit"] = float(bbe) * (1.0 if ev >= 95 else 0.0)
    row["n_sweetspot"] = float(bbe) * (1.0 if 8 <= la <= 32 else 0.0)
    row[EV_BIN_COLUMNS[ev_bin]] = float(bbe)
    return row


@pytest.fixture
def synthetic_monthly():
    """Two hitters, a normal April and a monstrous May onward.

    Every bucket from May (the cutoff month) on is 1,000 batted balls of
    120 mph barrels. A feature built for a June 1 cutoff must contain none of
    it; a feature built for a July 1 cutoff must contain May and June.
    """
    rows = []
    for player in (1, 2):
        for season in (2024, 2025, 2026):
            rows.append(bucket("hitter", player, season, 4, 50, 88.0, 12.0,
                               barrel_frac=0.05, ev_bin=6))
            for month in (5, 6, 7, 8, 9):
                rows.append(bucket("hitter", player, season, month, 1000,
                                   120.0, 25.0, barrel_frac=1.0,
                                   ev_bin=len(EV_BIN_COLUMNS) - 1))
    return pd.DataFrame(rows)


# --- the leakage guard -------------------------------------------------------

def test_window_excludes_every_post_cutoff_month(synthetic_monthly):
    """April only, at a May 1 cutoff — even though May onward is extreme."""
    counts = window_counts(synthetic_monthly, "hitter", "2026-05-01", 2026,
                           weights=(1.0, 0.0, 0.0))
    assert counts["bbe"].tolist() == [50.0, 50.0]
    assert (counts["sum_ev"] / counts["bbe"]).tolist() == [88.0, 88.0]
    assert counts["n_barrel"].tolist() == [2.5, 2.5]


def test_prior_seasons_enter_whole_but_the_current_one_is_cut(synthetic_monthly):
    """A prior season contributes all twelve months; the predict year does not."""
    counts = window_counts(synthetic_monthly, "hitter", "2026-05-01", 2026,
                           weights=(1.0, 1.0, 0.0))
    # 2026: April only (50). 2025: April plus five monster months (5050).
    assert counts["bbe"].tolist() == [5100.0, 5100.0]


def test_moving_the_cutoff_forward_admits_exactly_the_months_between(
        synthetic_monthly):
    may = window_counts(synthetic_monthly, "hitter", "2026-05-01", 2026,
                        weights=(1.0, 0.0, 0.0))["bbe"].iloc[0]
    july = window_counts(synthetic_monthly, "hitter", "2026-07-01", 2026,
                         weights=(1.0, 0.0, 0.0))["bbe"].iloc[0]
    assert may == 50.0
    assert july == 50.0 + 2 * 1000.0   # May and June, and nothing later


def test_features_are_unchanged_when_post_cutoff_rows_are_extreme(
        synthetic_monthly):
    """The whole point: deleting the post-cutoff rows changes nothing."""
    before = features_at_cutoff(synthetic_monthly, "hitter", "2026-05-01", 2026)
    clean = synthetic_monthly[~((synthetic_monthly["season"] == 2026)
                                & (synthetic_monthly["month"] >= 5))]
    after = features_at_cutoff(clean, "hitter", "2026-05-01", 2026)
    pd.testing.assert_frame_equal(before, after)


def test_a_mid_month_cutoff_is_refused_not_rounded():
    with pytest.raises(ValueError, match="not the first of a month"):
        assert_month_boundary(pd.Timestamp("2026-05-15"))
    with pytest.raises(ValueError, match="not the first of a month"):
        window_counts(pd.DataFrame(), "hitter", "2026-05-15", 2026)


def test_assert_window_clean_catches_a_broken_filter(synthetic_monthly):
    """Hand the guard rows the filter should have removed; it must object."""
    with pytest.raises(ValueError, match="on or after the cutoff"):
        assert_window_clean(synthetic_monthly[synthetic_monthly["season"] == 2026],
                            "2026-05-01", 2026)


def test_assert_window_clean_catches_a_future_season(synthetic_monthly):
    with pytest.raises(ValueError, match="> predict year"):
        assert_window_clean(synthetic_monthly, "2025-05-01", 2025)


def test_a_future_season_never_enters_the_window(synthetic_monthly):
    counts = window_counts(synthetic_monthly, "hitter", "2025-05-01", 2025,
                           weights=(1.0, 1.0, 1.0))
    # 2026 must contribute nothing: 2025 April + 2024 whole + 2023 (absent).
    assert counts["bbe"].tolist() == [50.0 + 5050.0, 50.0 + 5050.0]


def test_the_other_side_of_the_ball_is_not_mixed_in(synthetic_monthly):
    extra = synthetic_monthly.copy()
    extra["side"] = "pitcher"
    both = pd.concat([synthetic_monthly, extra], ignore_index=True)
    a = window_counts(both, "hitter", "2026-05-01", 2026, weights=(1.0, 0.0, 0.0))
    b = window_counts(synthetic_monthly, "hitter", "2026-05-01", 2026,
                      weights=(1.0, 0.0, 0.0))
    pd.testing.assert_frame_equal(a, b)


# --- the metrics -------------------------------------------------------------

def test_shrinkage_pulls_a_one_ball_sample_to_the_league():
    counts = pd.DataFrame([
        {"player": 1, "bbe": 1.0, "bbe_raw": 1.0, "sum_ev": 120.0,
         "sum_la": 25.0, "n_barrel": 1.0, "n_hardhit": 1.0, "n_sweetspot": 1.0,
         **{c: 0.0 for c in COUNT_COLUMNS if c not in
            ("bbe", "sum_ev", "sum_la", "n_barrel", "n_hardhit", "n_sweetspot")}},
        {"player": 2, "bbe": 999.0, "bbe_raw": 999.0, "sum_ev": 999 * 88.0,
         "sum_la": 999 * 10.0, "n_barrel": 0.0, "n_hardhit": 0.0,
         "n_sweetspot": 0.0,
         **{c: 0.0 for c in COUNT_COLUMNS if c not in
            ("bbe", "sum_ev", "sum_la", "n_barrel", "n_hardhit", "n_sweetspot")}},
    ])
    hard = contact_metrics(counts, ballast=0.0)
    soft = contact_metrics(counts, ballast=200.0)
    assert hard.loc[0, "ev_mean"] == pytest.approx(120.0)
    # With 200 balls of a league that averages ~88, one 120 mph ball barely moves.
    assert soft.loc[0, "ev_mean"] < 90.0
    assert soft.loc[0, "barrel"] < 0.02


def test_standardize_is_weighted_mean_zero():
    m = pd.DataFrame({"player": [1, 2, 3], "bbe_raw": [10.0, 10.0, 80.0],
                      "ev_mean": [80.0, 90.0, 100.0], "ev90": [1.0, 2.0, 3.0],
                      "barrel": [0.0, 0.1, 0.2], "hardhit": [0.1, 0.2, 0.3],
                      "sweetspot": [0.2, 0.3, 0.4], "la_mean": [5.0, 10.0, 15.0]})
    z = standardize(m)
    assert np.average(z["ev_mean"], weights=m["bbe_raw"]) == pytest.approx(0.0)
    assert np.sqrt(np.average(z["ev_mean"] ** 2,
                              weights=m["bbe_raw"])) == pytest.approx(1.0)


def test_a_constant_metric_standardizes_to_zero_not_to_nan():
    m = pd.DataFrame({"player": [1, 2], "bbe_raw": [10.0, 10.0],
                      "ev_mean": [90.0, 90.0], "ev90": [1.0, 1.0],
                      "barrel": [0.1, 0.1], "hardhit": [0.1, 0.1],
                      "sweetspot": [0.1, 0.1], "la_mean": [5.0, 5.0]})
    assert standardize(m)["ev_mean"].tolist() == [0.0, 0.0]


# --- the estimator -----------------------------------------------------------

def cells_with_known_signal(g=0.5, n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.15, 0.30, n)
    z = rng.normal(size=n)
    frame = pd.DataFrame({
        "component": "k_rate", "season": rng.integers(2017, 2020, n),
        "cutoff": "2019-05-01", "player": np.arange(n),
        "base": base, "trials": 400.0,
        "realized_rate": 0.05 + 0.8 * base + g * 0.01 * z,
    })
    for f in ("ev_mean", "ev90", "barrel", "hardhit", "sweetspot", "la_mean"):
        frame[f] = 0.0
    frame["ev_mean"] = z
    return frame


def test_fit_recovers_a_planted_coefficient():
    fit = fit_contact(cells_with_known_signal(), "k_rate")
    assert fit.coef["base"] == pytest.approx(0.8, abs=1e-6)
    assert fit.coef["intercept"] == pytest.approx(0.05, abs=1e-6)
    assert fit.coef["ev_mean"] == pytest.approx(0.005, abs=1e-6)
    assert fit.coef["barrel"] == pytest.approx(0.0, abs=1e-6)


def test_the_recalibration_control_carries_no_covariate():
    fit = fit_contact(cells_with_known_signal(), "k_rate", features=())
    assert fit.features == ()
    assert "ev_mean" not in fit.coef


def test_fit_refuses_a_component_it_has_no_cells_for():
    with pytest.raises(ValueError, match="no training cells"):
        fit_contact(cells_with_known_signal(), "babip")


def test_provider_covers_exactly_the_baseline_and_falls_back_to_recal(
        synthetic_monthly):
    """A player with no tracked contact is kept, at z = 0, not dropped."""
    base_frame = pd.DataFrame({"batter": [1, 2, 999], "predicted": [0.2, 0.25, 0.3]})
    fit = fit_contact(cells_with_known_signal(), "k_rate")
    cfg = ContactProviderConfig(
        monthly=synthetic_monthly, cutoff="2026-05-01", predict_year=2026,
        fit=fit, base_provider=lambda train, spec, year: base_frame,
        side="hitter",
    )
    out = contact_provider(cfg)(pd.DataFrame(), COMPONENTS["k_rate"], 2026)
    assert out["batter"].tolist() == [1, 2, 999]
    assert out["predicted"].notna().all()
    # 999 has no contact anywhere, so his prediction is the pure recalibration.
    recal = fit.coef["intercept"] + fit.coef["base"] * 0.3
    assert out.loc[2, "predicted"] == pytest.approx(recal)


def test_provider_predictions_stay_inside_the_clip():
    base_frame = pd.DataFrame({"batter": [1], "predicted": [50.0]})
    fit = fit_contact(cells_with_known_signal(), "k_rate")
    cfg = ContactProviderConfig(
        monthly=pd.DataFrame(columns=["side", "player", "season", "month",
                                      *COUNT_COLUMNS]),
        cutoff="2026-05-01", predict_year=2026, fit=fit,
        base_provider=lambda train, spec, year: base_frame)
    out = contact_provider(cfg)(pd.DataFrame(), COMPONENTS["k_rate"], 2026)
    assert 0.0 < out["predicted"].iloc[0] <= 0.999


# --- the clustered standard error -------------------------------------------

def paired_frames(d, reps=1):
    """Two scored frames whose per-row error difference is exactly `d`,
    each row repeated `reps` times under the same player id."""
    rows_a, rows_b = [], []
    for i, di in enumerate(d):
        for r in range(reps):
            rows_a.append({"key": f"{i}-{r}", "player": i, "predicted": 0.1 + di,
                           "realized_rate": 0.1, "trials": 1.0})
            rows_b.append({"key": f"{i}-{r}", "player": i, "predicted": 0.1,
                           "realized_rate": 0.1, "trials": 1.0})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def test_clustering_matches_the_plain_se_when_every_row_is_its_own_cluster():
    a, b = paired_frames(np.linspace(-0.01, 0.01, 20))
    plain = paired_abs_error_diff(a, b, id_col="key")
    clustered = paired_abs_error_diff(a, b, id_col="key", cluster_col="player")
    assert clustered["se"] == pytest.approx(plain["se"])
    assert plain["n_clusters"] == plain["n"]


def test_clustering_widens_the_se_when_rows_are_repeats_of_one_player():
    """Three copies of the same player is one observation, not three."""
    a, b = paired_frames(np.linspace(-0.01, 0.01, 20), reps=3)
    plain = paired_abs_error_diff(a, b, id_col="key")
    clustered = paired_abs_error_diff(a, b, id_col="key", cluster_col="player")
    assert clustered["diff"] == pytest.approx(plain["diff"])
    assert clustered["n_clusters"] == 20 and clustered["n"] == 60
    assert clustered["se"] == pytest.approx(plain["se"] * np.sqrt(3), rel=1e-6)
