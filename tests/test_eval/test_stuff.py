"""Stuff covariates: the two leakage guards first, then the estimator.

The leakage tests are the ones that matter. `synthetic_monthly` builds a season
where every bucket at or after the cutoff is an *extreme* — thousands of
pitches, every one of them a predicted whiff — so any post-cutoff row that
reaches the feature moves it enormously and no rounding or off-by-one can hide.
The stage-1 guard is the other half: a pitch from season Y must not be able to
enter the model that scores season Y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pitching_stuff import COUNT_COLUMNS
from src.eval.stuff import (
    FEATURES,
    features_at_cutoff,
    fit_stuff,
    standardize,
    stuff_metrics,
    window_counts,
)
from src.models.stuff import assert_no_leak


def bucket(pitcher, season, month, pitches, whiff_rate, csw_rate, velo,
           fb_share=0.5):
    """One monthly bucket with hand-set sufficient statistics."""
    row = {"pitcher": pitcher, "season": season, "month": month}
    row.update({c: 0.0 for c in COUNT_COLUMNS})
    swings = pitches * 0.47
    row["pitches"] = float(pitches)
    row["swings"] = swings
    row["p_whiff_sum"] = swings * whiff_rate
    row["p_csw_sum"] = pitches * csw_rate
    row["sum_velo"] = pitches * velo
    for pre, share in (("fb", fb_share), ("nfb", 1.0 - fb_share)):
        row[f"{pre}_pitches"] = pitches * share
        row[f"{pre}_swings"] = swings * share
        row[f"{pre}_p_whiff_sum"] = swings * share * whiff_rate
        row[f"{pre}_p_csw_sum"] = pitches * share * csw_rate
        row[f"{pre}_sum_velo"] = pitches * share * velo
    row["n_ff"] = pitches * fb_share
    row["n_sl"] = pitches * (1.0 - fb_share)
    return row


@pytest.fixture
def synthetic_monthly():
    """Two pitchers, an ordinary April and an impossible May onward.

    Every bucket from May (the cutoff month) on is 5,000 pitches at a predicted
    whiff rate of 1.0 and 105 mph. A feature built for a May 1 cutoff must
    contain none of it; one built for a July 1 cutoff must contain May and June.
    """
    rows = []
    for pitcher in (1, 2):
        for season in (2024, 2025, 2026):
            rows.append(bucket(pitcher, season, 4, 400, 0.25, 0.28, 93.0))
            for month in (5, 6, 7, 8, 9):
                rows.append(bucket(pitcher, season, month, 5000, 1.0, 1.0,
                                   105.0))
    return pd.DataFrame(rows)


# --- the stage-2 leakage guard ------------------------------------------------

def test_a_may_first_cutoff_never_sees_may(synthetic_monthly):
    """The pre-registered guard: a cutoff on the 1st excludes that month."""
    counts = window_counts(synthetic_monthly, "2026-05-01", 2026,
                           (1.0, 0.0, 0.0))
    assert float(counts["pitches"].max()) == pytest.approx(400.0)
    z = stuff_metrics(counts, ballast=0.0)
    assert float(z["velo"].max()) == pytest.approx(93.0, abs=1e-6)
    assert float(z["xwhiff"].max()) == pytest.approx(0.25, abs=1e-6)


def test_the_window_is_identical_to_deleting_the_future(synthetic_monthly):
    """Stronger than the filter test: the feature at a July 1 cutoff equals the
    feature built from a frame with every July-onward row physically removed."""
    kept = synthetic_monthly[~((synthetic_monthly["season"] == 2026)
                               & (synthetic_monthly["month"] >= 7))]
    a = features_at_cutoff(synthetic_monthly, "2026-07-01", 2026)
    b = features_at_cutoff(kept, "2026-07-01", 2026)
    pd.testing.assert_frame_equal(a, b)


def test_a_mid_month_cutoff_is_refused_not_rounded(synthetic_monthly):
    with pytest.raises(ValueError, match="first of a month"):
        window_counts(synthetic_monthly, "2026-05-15", 2026)


def test_a_zero_weight_season_leaves_the_window_entirely(synthetic_monthly):
    """Exposure has to drop with the season, or the split is cut on pitches
    that are not behind the covariate."""
    one = window_counts(synthetic_monthly, "2026-05-01", 2026, (1.0, 0.0, 0.0))
    three = window_counts(synthetic_monthly, "2026-05-01", 2026, (1.0, 1.0, 1.0))
    assert float(one["pitches_raw"].max()) < float(three["pitches_raw"].max())


# --- the stage-1 leakage guard ------------------------------------------------

def test_stage_one_refuses_a_training_pitch_from_the_scored_season():
    train = pd.DataFrame({"game_year": [2022, 2023, 2024]})
    assert_no_leak(train[train["game_year"] < 2024], 2024)
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leak(train, 2024)
    with pytest.raises(ValueError, match="no training pitches"):
        assert_no_leak(train.iloc[:0], 2024)


# --- the metrics --------------------------------------------------------------

def test_shrinkage_pulls_a_thin_pitcher_toward_the_league():
    counts = pd.DataFrame([
        {"player": 1, "pitches_raw": 3000.0,
         **{k: v for k, v in bucket(1, 2026, 4, 3000, 0.20, 0.28, 92.0).items()
            if k in COUNT_COLUMNS}},
        {"player": 2, "pitches_raw": 30.0,
         **{k: v for k, v in bucket(2, 2026, 4, 30, 0.50, 0.40, 99.0).items()
            if k in COUNT_COLUMNS}},
    ])
    raw = stuff_metrics(counts, ballast=0.0)
    shrunk = stuff_metrics(counts, ballast=250.0)
    thin_raw = float(raw.loc[raw["player"] == 2, "xwhiff"].iloc[0])
    thin_shrunk = float(shrunk.loc[shrunk["player"] == 2, "xwhiff"].iloc[0])
    thick_shrunk = float(shrunk.loc[shrunk["player"] == 1, "xwhiff"].iloc[0])
    assert thin_shrunk < thin_raw
    # 30 pitches against 250 of ballast: the thin pitcher ends up nearer the
    # league (which is the thick pitcher, who is 99% of it) than his own rate.
    assert abs(thin_shrunk - thick_shrunk) < abs(thin_raw - thick_shrunk)


def test_standardize_is_pitch_weighted_and_centred():
    m = pd.DataFrame({"player": [1, 2, 3], "pitches_raw": [3000.0, 3000.0, 10.0],
                      **{f: [0.2, 0.3, 9.0] for f in FEATURES}})
    z = standardize(m)
    w = m["pitches_raw"].to_numpy()
    for f in FEATURES:
        assert float(np.average(z[f].to_numpy(), weights=w)) == pytest.approx(0.0, abs=1e-9)


# --- the estimator ------------------------------------------------------------

def cells(n=400, seed=0):
    """Cells where the realized rate is the baseline plus a real xwhiff effect."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.22, 0.03, n)
    z = rng.normal(0.0, 1.0, n)
    y = base + 0.01 * z + rng.normal(0.0, 0.005, n)
    d = pd.DataFrame({"component": "p_k_rate", "season": rng.integers(2022, 2025, n),
                      "cutoff": "2024-05-01", "player": np.arange(n),
                      "base": base, "realized_rate": y, "trials": 400.0})
    for f in FEATURES:
        d[f] = 0.0
    d["xwhiff"] = z
    return d


def test_fit_recovers_the_covariate_and_the_control_does_not_see_it():
    c = cells()
    full = fit_stuff(c, "p_k_rate")
    recal = fit_stuff(c, "p_k_rate", features=())
    assert full.coef["xwhiff"] == pytest.approx(0.01, abs=0.002)
    assert "xwhiff" not in recal.coef
    # The covariate arm has to explain the part the recalibration cannot.
    err = lambda fit, z: np.mean(np.abs(
        fit.predict(c["base"].to_numpy(), z) - c["realized_rate"].to_numpy()))
    assert err(full, c) < err(recal, None)


def test_fixed_base_pins_the_baseline_coefficient_at_one():
    fit = fit_stuff(cells(), "p_k_rate", fixed_base=True)
    assert fit.coef["base"] == 1.0
    assert fit.coef["xwhiff"] == pytest.approx(0.01, abs=0.002)


def test_fit_refuses_a_component_with_no_training_cells():
    with pytest.raises(ValueError, match="no training cells"):
        fit_stuff(cells(), "p_hr_rate")
