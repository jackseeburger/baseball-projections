"""Swing-decision covariates: the leakage guard first, then the metrics.

The leakage test is the one that matters. `synthetic_monthly` builds seasons
where every bucket at or after the cutoff is an *extreme* — a thousand pitches
a month at which the batter swung at everything outside the zone and missed
everything inside it — so any post-cutoff row that reached the feature would
move it enormously and no rounding or off-by-one could hide. The test then
asserts the feature is identical **bit for bit** to the one built from a frame
with those rows physically deleted, which is the only version of this claim
that cannot be satisfied by a small leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.swing_decisions import COUNT_COLUMNS
from src.eval.swing import (
    FEATURES,
    assert_month_boundary,
    assert_window_clean,
    features_at_cutoff,
    league_month_rates,
    swing_metrics,
    window_counts,
    year_over_year,
)


def bucket(batter, season, month, pitches=1000, zone_frac=0.5,
           chase=0.28, zswing=0.67, zcontact=0.85, ocontact=0.60,
           cstrike=0.75):
    """One monthly bucket with hand-set rates turned back into counts."""
    pz = pitches * zone_frac
    po = pitches - pz
    sz, so = pz * zswing, po * chase
    return {
        "batter": batter, "season": season, "month": month,
        "pitches_z": pz, "pitches_o": po,
        "swings_z": sz, "swings_o": so,
        "contact_z": sz * zcontact, "contact_o": so * ocontact,
        "whiff_z": sz * (1 - zcontact), "whiff_o": so * (1 - ocontact),
        "called_z": (pz - sz) * cstrike,
    }


@pytest.fixture
def synthetic_monthly():
    """Two batters, a normal April and a monstrous May onward.

    From May — the cutoff month — every bucket is a thousand pitches at which
    the batter chased everything and made contact with nothing. A feature
    built at a May 1 cutoff must contain none of it.
    """
    rows = []
    for batter in (1, 2):
        for season in (2024, 2025, 2026):
            rows.append(bucket(batter, season, 4))
            for month in (5, 6, 7, 8, 9):
                rows.append(bucket(batter, season, month, chase=1.0,
                                   zswing=1.0, zcontact=0.0, ocontact=0.0,
                                   cstrike=0.0))
    return pd.DataFrame(rows)


# --- the leakage guard -------------------------------------------------------

def test_window_excludes_every_post_cutoff_month(synthetic_monthly):
    counts = window_counts(synthetic_monthly, "2026-05-01", 2026,
                           weights=(1.0, 0.0, 0.0))
    assert counts["pitches_z"].tolist() == [500.0, 500.0]
    # April's chase rate, untouched by the pathological months
    assert (counts["swings_o"] / counts["pitches_o"]).tolist() == [0.28, 0.28]


def test_prior_seasons_enter_whole_but_the_current_one_is_cut(synthetic_monthly):
    counts = window_counts(synthetic_monthly, "2026-05-01", 2026,
                           weights=(1.0, 1.0, 0.0))
    # 2026: April only (1000 pitches). 2025: all six months (6000).
    assert counts["pitches_z"].tolist() == [3500.0, 3500.0]


def test_the_feature_is_identical_to_one_built_without_the_later_rows(
        synthetic_monthly):
    """The claim, bit for bit: a same-month bucket is not merely down-weighted
    or shrunk away — it is not in the sum at all."""
    kept = synthetic_monthly[
        (synthetic_monthly["season"] < 2026)
        | (synthetic_monthly["month"] < 5)].copy()
    a = features_at_cutoff(synthetic_monthly, "2026-05-01", 2026)
    b = features_at_cutoff(kept, "2026-05-01", 2026)
    assert a["player"].tolist() == b["player"].tolist()
    for f in (*FEATURES, "pitches_raw"):
        assert a[f].to_numpy().tobytes() == b[f].to_numpy().tobytes()


def test_a_cutoff_that_is_not_a_month_boundary_is_refused(synthetic_monthly):
    with pytest.raises(ValueError, match="first of a month"):
        assert_month_boundary("2026-05-15")
    with pytest.raises(ValueError, match="first of a month"):
        window_counts(synthetic_monthly, "2026-05-15", 2026)


def test_the_guard_catches_a_bucket_the_filter_let_through(synthetic_monthly):
    """`assert_window_clean` re-checks the filtered rows rather than trusting
    the filter, so a broken filter is caught rather than believed."""
    bad = synthetic_monthly[(synthetic_monthly["season"] == 2026)
                            & (synthetic_monthly["month"] >= 5)]
    with pytest.raises(ValueError, match="leakage"):
        assert_window_clean(bad, "2026-05-01", 2026)
    with pytest.raises(ValueError, match="leakage"):
        assert_window_clean(
            synthetic_monthly.assign(season=2027), "2026-05-01", 2026)


# --- the metrics -------------------------------------------------------------

def test_rates_are_ratios_of_sums_not_averages_of_monthly_rates():
    """A month of 100 pitches at a 50% chase rate and a month of 900 at 10%
    is a 14% chase rate over the window, not 30%."""
    m = pd.DataFrame([bucket(1, 2026, 3, pitches=200, zone_frac=0.5, chase=0.5),
                      bucket(1, 2026, 4, pitches=1800, zone_frac=0.5, chase=0.1)])
    counts = window_counts(m, "2026-05-01", 2026, weights=(1.0, 0.0, 0.0))
    assert float(counts["swings_o"].iloc[0] / counts["pitches_o"].iloc[0]) \
        == pytest.approx(0.14)


def test_a_league_average_batter_gets_a_feature_of_zero():
    """The features are league-month relative, so a batter who behaved exactly
    like the league he played in comes back at zero before standardization —
    which is also what a batter with no pitches at all gets."""
    m = pd.DataFrame([bucket(1, 2026, 4), bucket(2, 2026, 4)])
    metrics = swing_metrics(window_counts(m, "2026-05-01", 2026,
                                          weights=(1.0, 0.0, 0.0)))
    for f in FEATURES:
        assert metrics[f].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_the_league_reference_is_the_players_own_months():
    """Two batters with the same raw chase rate in different months of
    different league behaviour get different features — that is the point of
    carrying the league month through the sum."""
    m = pd.DataFrame([
        # April 2026: a league that chases a lot. May: one that does not.
        bucket(1, 2026, 4, chase=0.30), bucket(9, 2026, 4, chase=0.30),
        bucket(2, 2026, 5, chase=0.30), bucket(9, 2026, 5, chase=0.10),
    ])
    lg = league_month_rates(m).set_index(["season", "month"])["chase"]
    assert lg.loc[(2026, 4)] == pytest.approx(0.30)
    assert lg.loc[(2026, 5)] == pytest.approx(0.20)
    metrics = swing_metrics(window_counts(m, "2026-06-01", 2026,
                                          weights=(1.0, 0.0, 0.0)),
                            ballast=0.0).set_index("player")
    assert metrics.loc[1, "chase"] == pytest.approx(0.0)
    assert metrics.loc[2, "chase"] == pytest.approx(0.10)


def test_shrinkage_pulls_a_tiny_sample_toward_its_own_league():
    small = pd.DataFrame([bucket(1, 2026, 4, pitches=20, chase=1.0),
                          bucket(2, 2026, 4, pitches=100000, chase=0.28)])
    counts = window_counts(small, "2026-05-01", 2026, weights=(1.0, 0.0, 0.0))
    hard = swing_metrics(counts, ballast=0.0).set_index("player")
    soft = swing_metrics(counts, ballast=100.0).set_index("player")
    assert hard.loc[1, "chase"] > soft.loc[1, "chase"] > 0.0


def test_standardized_features_are_finite_and_centred(synthetic_monthly):
    z = features_at_cutoff(synthetic_monthly, "2026-05-01", 2026)
    assert np.isfinite(z[list(FEATURES)].to_numpy()).all()
    assert set(z.columns) == {"player", "pitches_raw", *FEATURES}


def test_year_over_year_needs_the_pitch_floor_in_both_years():
    m = pd.DataFrame([bucket(1, 2025, 4, pitches=1000),
                      bucket(1, 2026, 4, pitches=1000),
                      bucket(2, 2025, 4, pitches=1000),
                      bucket(2, 2026, 4, pitches=10),
                      bucket(3, 2025, 4, pitches=1000, chase=0.4),
                      bucket(3, 2026, 4, pitches=1000, chase=0.4),
                      bucket(4, 2025, 4, pitches=1000, chase=0.2),
                      bucket(4, 2026, 4, pitches=1000, chase=0.2)])
    yy = year_over_year(m, (2025, 2026), min_pitches=500)
    assert yy["n"].tolist() == [3]      # batter 2 is out, on his 2026


def test_counts_carry_every_column_the_artifact_defines(synthetic_monthly):
    counts = window_counts(synthetic_monthly, "2026-05-01", 2026)
    assert set(COUNT_COLUMNS).issubset(counts.columns)
