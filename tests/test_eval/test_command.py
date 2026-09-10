"""Command covariates: the stage-2 leakage guard first, then the metrics.

Same shape as `tests/test_eval/test_stuff.py`, and for the same reason: the
leakage test is the one that matters. `synthetic_monthly` builds a season where
every bucket at or after the cutoff is an *extreme* — thousands of pitches,
every one of them carrying a huge command residual and landing in the heart of
the zone — so any post-cutoff row that reaches the feature moves it enormously
and no rounding or off-by-one can hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pitching_command import COUNT_COLUMNS, REGIONS, season_aggregate
from src.eval.command import (
    FEATURES,
    LEVEL_FEATURES,
    LEVEL_FEATURES_WITH_EDGE,
    attach_command_features,
    command_metrics,
    features_at_cutoff,
    window_counts,
)


def bucket(pitcher, season, month, pitches, resid, cs_resid, heart_share,
           zone_share=0.45):
    """One monthly bucket with hand-set sufficient statistics."""
    row = {"pitcher": pitcher, "season": season, "month": month}
    row.update({c: 0.0 for c in COUNT_COLUMNS})
    takens = pitches * 0.53
    row["pitches"] = float(pitches)
    row["takens"] = takens
    row["cmd_csw_sum"] = pitches * 0.30
    row["cmd_resid_sum"] = pitches * resid
    row["cmd_resid_fb_sum"] = pitches * resid * 0.5
    row["cmd_resid_nfb_sum"] = pitches * resid * 0.5
    row["cs_taken_sum"] = takens * 0.32
    row["cs_resid_sum"] = takens * cs_resid
    row["in_zone"] = pitches * zone_share
    row["n_heart"] = pitches * heart_share
    rest = (1.0 - heart_share) / 3.0
    for r in ("shadow", "chase", "waste"):
        row[f"n_{r}"] = pitches * rest
    row["fb_pitches"] = pitches * 0.5
    return row


@pytest.fixture
def synthetic_monthly():
    """Two pitchers, an ordinary April and an impossible May onward.

    Every bucket from May (the cutoff month) on is 5,000 pitches at a command
    residual of 0.5 per pitch, every one of them in the heart of the zone. A
    feature built for a May 1 cutoff must contain none of it; one built for a
    July 1 cutoff must contain May and June.
    """
    rows = []
    for pitcher in (1, 2):
        for season in (2024, 2025, 2026):
            # The two pitchers differ in April so the standardization has
            # something to standardize; pitcher 1 is the better-located one.
            scale = 1.0 if pitcher == 1 else 0.5
            rows.append(bucket(pitcher, season, 4, 400, 0.01 * scale,
                               0.02 * scale, 0.30 * scale,
                               zone_share=0.45 * scale))
            for month in (5, 6, 7, 8, 9):
                rows.append(bucket(pitcher, season, month, 5000, 0.5, 0.5, 1.0,
                                   zone_share=1.0))
    return pd.DataFrame(rows)


# --- the stage-2 leakage guard ------------------------------------------------

def test_a_may_first_cutoff_never_sees_may(synthetic_monthly):
    """The pre-registered guard: a cutoff on the 1st excludes that month."""
    counts = window_counts(synthetic_monthly, "2026-05-01", 2026,
                           (1.0, 0.0, 0.0))
    assert float(counts["pitches"].max()) == pytest.approx(400.0)
    m = command_metrics(counts, ballast=0.0)
    assert float(m["cmd_resid"].max()) == pytest.approx(0.01, abs=1e-6)
    assert float(m["cs_resid"].max()) == pytest.approx(0.02, abs=1e-6)
    assert float(m["zone_share"].max()) == pytest.approx(0.45, abs=1e-6)


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


# --- the metrics --------------------------------------------------------------

def test_region_shares_are_fractions_of_the_pitches_with_a_known_region():
    """A pitch with no tracked strike zone has no region, so it is out of the
    denominator rather than silently in `heart`."""
    counts = pd.DataFrame([{
        "player": 1, "pitches_raw": 1000.0,
        **{c: 0.0 for c in COUNT_COLUMNS},
    }])
    counts.loc[0, "pitches"] = 1000.0
    counts.loc[0, "takens"] = 500.0
    counts.loc[0, "in_zone"] = 450.0
    for r, n in zip(REGIONS, (400.0, 200.0, 200.0, 100.0)):
        counts.loc[0, f"n_{r}"] = n
    m = command_metrics(counts, ballast=0.0)
    # 900 pitches have a region; 100 do not.
    assert m["shadow_share"].iloc[0] == pytest.approx(200 / 900)
    assert m["chase_share"].iloc[0] == pytest.approx(200 / 900)
    assert m["waste_share"].iloc[0] == pytest.approx(100 / 900)
    # `zone_share` is the rulebook zone, out of every pitch.
    assert m["zone_share"].iloc[0] == pytest.approx(0.45)


def test_the_residual_is_shrunk_toward_the_league_not_toward_zero():
    """Shrinkage adds `ballast` pitches of the league's own profile, so a
    pitcher with almost no pitches lands on the league, not on zero."""
    rows = [bucket(1, 2026, 4, 20000, 0.05, 0.05, 0.4),
            bucket(2, 2026, 4, 20, 0.9, 0.9, 0.4)]
    counts = window_counts(pd.DataFrame(rows), "2026-05-01", 2026)
    b = 1000.0
    league = float((20000 * 0.05 + 20 * 0.9) / 20020)
    heavy = command_metrics(counts, ballast=b).set_index("player")
    # 20 of his own pitches against 1,000 of the league's: he keeps 2% of his
    # own signal, and the rest of him is the league.
    assert heavy.loc[2, "cmd_resid"] == pytest.approx(
        (20 * 0.9 + b * league) / (20 + b), abs=1e-6)
    assert abs(heavy.loc[2, "cmd_resid"] - league) < abs(0.9 - league) / 10
    light = command_metrics(counts, ballast=0.0).set_index("player")
    assert light.loc[2, "cmd_resid"] == pytest.approx(0.9, abs=1e-6)


def test_a_pitcher_with_no_tracked_pitches_gets_a_zero_covariate(
        synthetic_monthly):
    """The common pitcher set stays the baseline's: an untracked pitcher is
    given z = 0 rather than dropped, which makes the command arm identical to
    the recalibration arm for him."""
    cells = pd.DataFrame({
        "component": "p_bb_rate", "season": 2026, "cutoff": "2026-05-01",
        "player": [1, 2, 999], "base": 0.08, "realized_rate": 0.08,
        "trials": 300.0, "realized_successes": 24.0, "pre_trials": 100.0,
    })
    out = attach_command_features(cells, synthetic_monthly)
    got = out.set_index("player")
    for f in FEATURES:
        assert got.loc[999, f] == 0.0
    assert got.loc[999, "cmd_pitches_raw"] == 0.0
    assert not np.allclose([got.loc[1, f] for f in FEATURES], 0.0)


# --- the level block (BAS-87) -------------------------------------------------

def test_the_level_block_is_a_level_not_a_residual():
    """`cmd_csw` is the location-aware model's own CSW per pitch. The bucket
    helper writes 0.30 of a CSW per pitch and a residual of `resid` on top of
    it, so the level must come back 0.30 whatever the residual is."""
    rows = [bucket(1, 2026, 4, 5000, 0.05, 0.05, 0.4),
            bucket(2, 2026, 4, 5000, -0.05, -0.05, 0.4)]
    m = command_metrics(window_counts(pd.DataFrame(rows), "2026-05-01", 2026),
                        ballast=0.0).set_index("player")
    assert m.loc[1, "cmd_csw"] == pytest.approx(0.30, abs=1e-9)
    assert m.loc[2, "cmd_csw"] == pytest.approx(0.30, abs=1e-9)
    assert m.loc[1, "cmd_resid"] == pytest.approx(0.05, abs=1e-9)
    assert m.loc[2, "cmd_resid"] == pytest.approx(-0.05, abs=1e-9)


def test_the_level_block_obeys_the_same_cutoff_guard(synthetic_monthly):
    """The level features go through the same `window_counts`, so the May 1
    guard has to hold for them too — the fixture's May onward is every pitch
    in the zone, which would move `zone_share` from 0.45 to nearly 1."""
    z = features_at_cutoff(synthetic_monthly, "2026-05-01", 2026,
                           features=LEVEL_FEATURES)
    assert list(z.columns) == ["player", "pitches_raw", *LEVEL_FEATURES]
    kept = synthetic_monthly[~((synthetic_monthly["season"] == 2026)
                               & (synthetic_monthly["month"] >= 5))]
    pd.testing.assert_frame_equal(
        z, features_at_cutoff(kept, "2026-05-01", 2026,
                              features=LEVEL_FEATURES))


def test_attaching_the_level_block_attaches_only_the_level_block(
        synthetic_monthly):
    """Selecting the block changes which columns land on the cells, and an
    untracked pitcher still gets z = 0 rather than being dropped."""
    cells = pd.DataFrame({
        "component": "p_bb_rate", "season": 2026, "cutoff": "2026-05-01",
        "player": [1, 2, 999], "base": 0.08, "realized_rate": 0.08,
        "trials": 300.0, "realized_successes": 24.0, "pre_trials": 100.0,
    })
    out = attach_command_features(cells, synthetic_monthly,
                                  features=LEVEL_FEATURES)
    assert set(LEVEL_FEATURES) <= set(out.columns)
    assert "cmd_resid" not in out.columns
    got = out.set_index("player")
    for f in LEVEL_FEATURES:
        assert got.loc[999, f] == 0.0
    wide = attach_command_features(cells, synthetic_monthly,
                                   features=LEVEL_FEATURES_WITH_EDGE)
    assert "shadow_share" in wide.columns


def test_the_season_aggregate_carries_the_levels_the_vacuity_check_needs():
    """The vacuity check is re-registered on the levels, so `season_aggregate`
    has to produce them from the additive sums and not from anything the
    monthly reduction does not already carry."""
    rows = [bucket(1, 2025, 4, 1000, 0.02, 0.02, 0.4),
            bucket(1, 2025, 5, 1000, 0.02, 0.02, 0.4)]
    g = season_aggregate(pd.DataFrame(rows)).set_index(["pitcher", "season"])
    assert g.loc[(1, 2025), "pitches"] == 2000
    assert g.loc[(1, 2025), "cmd_csw"] == pytest.approx(0.30, abs=1e-9)
    assert g.loc[(1, 2025), "zone_share"] == pytest.approx(0.45, abs=1e-9)
    # heart is 0.4 of the pitches and the other three split the remaining 0.6.
    assert g.loc[(1, 2025), "waste_share"] == pytest.approx(0.2, abs=1e-9)
    assert g.loc[(1, 2025), "heart_share"] == pytest.approx(0.4, abs=1e-9)
