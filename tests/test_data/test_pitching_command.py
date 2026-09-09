"""The location block, the called-strike-given-taken mapping, and the leakage guards.

Two things matter most at this end. The **mapping**: a command model is only
measuring location if the called-strike target's denominator is takes and its
numerator is takes the umpire rang up — put a foul tip or a swinging strike on
the wrong side of that line and the model is fitted on a different quantity
wearing command's name. And the **guards**: the stage-1 walk-forward guard,
which refuses a training frame containing a pitch from the season being
scored, and the stage-2 window guard, which refuses a cutoff that is not the
first of a month and re-checks the rows it summed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pitching_command import (
    BALL_WIDTH,
    COUNT_COLUMNS,
    MIN_ZONE_HEIGHT,
    REGIONS,
    ZONE_HALF_WIDTH,
    location_features,
    monthly_buckets,
    season_aggregate,
    year_over_year,
)
from src.data.pitching_stuff import competitive_pitches, pitch_features


def pitch(description="ball", pitch_type="FF", pitcher=1, date="2024-04-10",
          plate_x=0.0, plate_z=2.5, sz_top=3.4, sz_bot=1.6, balls=0, strikes=0,
          stand="R", **over):
    """One Statcast pitch row, physics and zone filled with plausible values."""
    row = {
        "game_type": "R", "game_date": date, "game_year": int(date[:4]),
        "pitcher": pitcher, "p_throws": "R", "stand": stand,
        "description": description, "pitch_type": pitch_type,
        "release_speed": 94.0, "effective_speed": 94.5,
        "pfx_x": -0.5, "pfx_z": 1.4, "release_pos_x": -1.8,
        "release_pos_y": 54.0, "release_pos_z": 6.0, "release_extension": 6.4,
        "release_spin_rate": 2300.0, "spin_axis": 210.0, "arm_angle": 45.0,
        "vx0": 5.0, "vy0": -136.0, "vz0": -5.0, "ax": -8.0, "ay": 28.0,
        "az": -14.0, "plate_x": plate_x, "plate_z": plate_z,
        "sz_top": sz_top, "sz_bot": sz_bot, "balls": balls, "strikes": strikes,
        "zone": 5,
    }
    row.update(over)
    return row


def feats(rows):
    df = competitive_pitches(pd.DataFrame(rows))
    return location_features(df, pitch_features(df))


# --- the mapping -------------------------------------------------------------

@pytest.mark.parametrize("description,taken,called", [
    ("called_strike", True, True),
    ("ball", True, False),
    ("blocked_ball", True, False),
    ("hit_by_pitch", True, False),
    # A swing is not a take, whatever it did to the ball — including the two
    # that are strikes.
    ("swinging_strike", False, False),
    ("swinging_strike_blocked", False, False),
    ("foul_tip", False, False),
    ("foul", False, False),
    ("hit_into_play", False, False),
])
def test_called_strike_given_taken_mapping(description, taken, called):
    f = feats([pitch(description=description)])
    assert bool(f["is_taken"].iloc[0]) is taken
    assert bool(f["is_called"].iloc[0]) is called
    # The target is only ever read on its own denominator, so a called strike
    # can never be scored on a pitch that was swung at.
    assert not (bool(f["is_called"].iloc[0]) and not bool(f["is_taken"].iloc[0]))


def test_target_rows_restricts_called_strike_to_takes():
    from src.models.command import TARGETS, target_rows

    f = feats([pitch(description="called_strike"), pitch(description="ball"),
               pitch(description="swinging_strike"), pitch(description="foul")])
    rows = target_rows(f, "called_taken")
    assert len(rows) == 2
    assert rows[TARGETS["called_taken"][0]].sum() == 1
    # CSW is per pitch and keeps everything.
    assert len(target_rows(f, "csw")) == 4


# --- the geometry ------------------------------------------------------------

def test_zone_relative_height_is_in_the_batters_own_units():
    tall = feats([pitch(plate_z=2.5, sz_bot=1.6, sz_top=3.6)])
    short = feats([pitch(plate_z=2.0, sz_bot=1.4, sz_top=2.6)])
    assert tall["plate_z_sz"].iloc[0] == pytest.approx(0.45, abs=1e-5)
    assert short["plate_z_sz"].iloc[0] == pytest.approx(0.5, abs=1e-5)
    # ... and the raw height is still there, because an umpire's zone is not
    # exactly the batter's.
    assert tall["plate_z"].iloc[0] == pytest.approx(2.5)


def test_plate_x_is_mirrored_to_the_arm_and_to_the_batters_box():
    """Away-from-the-batter is the same pitch to either side of the plate."""
    r = feats([pitch(stand="R", plate_x=-0.7)])
    l = feats([pitch(stand="L", plate_x=0.7)])
    assert r["plate_x_bat"].iloc[0] == pytest.approx(l["plate_x_bat"].iloc[0])
    assert r["plate_x_bat"].iloc[0] > 0
    # The arm-side mirror is the pitcher's, and both of these are righties.
    assert r["plate_x_arm"].iloc[0] == pytest.approx(-0.7)


@pytest.mark.parametrize("plate_x,plate_z,region", [
    (0.0, 2.5, "heart"),                                # middle
    (ZONE_HALF_WIDTH - 0.05, 2.5, "shadow"),            # just inside the edge
    (ZONE_HALF_WIDTH + 0.1, 2.5, "shadow"),             # just outside it
    (ZONE_HALF_WIDTH + BALL_WIDTH + 0.2, 2.5, "chase"),
    (3.0, 2.5, "waste"),
    (0.0, 1.65, "shadow"),                              # just above the knees
    (0.0, 0.4, "waste"),                                # in the dirt
])
def test_attack_regions_partition_the_plate(plate_x, plate_z, region):
    f = feats([pitch(plate_x=plate_x, plate_z=plate_z)])
    assert f[f"region_{region}"].iloc[0] == 1.0
    assert sum(f[f"region_{r}"].iloc[0] for r in REGIONS) == 1.0


def test_in_zone_is_the_rulebook_zone_and_crosses_heart_and_shadow():
    inside_edge = feats([pitch(plate_x=ZONE_HALF_WIDTH - 0.05, plate_z=2.5)])
    outside_edge = feats([pitch(plate_x=ZONE_HALF_WIDTH + 0.05, plate_z=2.5)])
    assert inside_edge["in_zone"].iloc[0] == 1.0
    assert inside_edge["region_shadow"].iloc[0] == 1.0
    assert outside_edge["in_zone"].iloc[0] == 0.0
    assert outside_edge["region_shadow"].iloc[0] == 1.0


def test_a_missing_strike_zone_is_missing_not_a_region():
    """The archive zero-fills `sz_top`/`sz_bot` on an untracked pitch. A pitch
    with no zone has no region — dropping it into `heart` by default would
    make a missing measurement look like a pitch down the middle."""
    f = feats([pitch(sz_top=0.0, sz_bot=0.0)])
    assert np.isnan(f["plate_z_sz"].iloc[0])
    assert np.isnan(f["edge_dist"].iloc[0])
    assert np.isnan(f["in_zone"].iloc[0])
    for r in REGIONS:
        assert np.isnan(f[f"region_{r}"].iloc[0])
    assert MIN_ZONE_HEIGHT > 0


def test_the_count_is_a_feature():
    f = feats([pitch(balls=3, strikes=0)])
    assert f["balls"].iloc[0] == 3 and f["strikes"].iloc[0] == 0
    assert f["count_diff"].iloc[0] == 3


def test_same_hand_matchup():
    assert feats([pitch(stand="R")])["same_hand"].iloc[0] == 1.0
    assert feats([pitch(stand="L")])["same_hand"].iloc[0] == 0.0


# --- the monthly reduction ---------------------------------------------------

def scored(rows, cmd=0.30, stuff=0.25, cs_cmd=0.55, cs_stuff=0.50):
    return feats(rows).assign(p_csw_pitching=cmd, p_csw_stuff=stuff,
                              p_cs_pitching=cs_cmd, p_cs_stuff=cs_stuff)


def test_the_bucket_is_the_residual_not_the_level():
    """The served engine already carries the stuff score; what this artifact
    holds is what location adds on top of it."""
    g = monthly_buckets(scored([pitch()] * 4, cmd=0.31, stuff=0.25)).iloc[0]
    assert g["pitches"] == 4
    assert g["cmd_resid_sum"] == pytest.approx(4 * 0.06)
    assert g["cmd_csw_sum"] == pytest.approx(4 * 0.31)


def test_called_strike_residual_is_summed_over_takes_only():
    df = scored([pitch(description="called_strike"), pitch(description="ball"),
                 pitch(description="swinging_strike"),
                 pitch(description="hit_into_play")],
                cs_cmd=0.6, cs_stuff=0.5)
    g = monthly_buckets(df).iloc[0]
    assert g["pitches"] == 4 and g["takens"] == 2
    assert g["cs_resid_sum"] == pytest.approx(2 * 0.1)
    assert g["cs_taken_sum"] == pytest.approx(2 * 0.6)
    # The CSW residual is per pitch and keeps all four.
    assert g["cmd_resid_sum"] == pytest.approx(4 * 0.05)


def test_buckets_are_additive_over_months():
    """A bucket is a sum, so two months summed equal one bucket of both."""
    april = scored([pitch(date="2024-04-10")] * 3)
    may = scored([pitch(date="2024-05-10", plate_x=2.0)] * 2)
    parts = monthly_buckets(pd.concat([april, may], ignore_index=True))
    assert sorted(parts["month"]) == [4, 5]
    both = parts[COUNT_COLUMNS].sum()
    one = monthly_buckets(pd.concat([april, may], ignore_index=True).assign(
        game_date=pd.Timestamp("2024-04-10")))[COUNT_COLUMNS].sum()
    pd.testing.assert_series_equal(both, one)


def test_region_counts_sum_to_the_pitches_with_a_known_zone():
    df = scored([pitch(plate_x=0.0), pitch(plate_x=0.9), pitch(plate_x=1.3),
                 pitch(plate_x=3.0), pitch(sz_top=0.0, sz_bot=0.0)])
    g = monthly_buckets(df).iloc[0]
    assert g["pitches"] == 5
    assert sum(g[f"n_{r}"] for r in REGIONS) == 4


def test_the_fastball_split_partitions_the_csw_residual():
    df = scored([pitch(pitch_type="FF"), pitch(pitch_type="SL"),
                 pitch(pitch_type="CH")])
    g = monthly_buckets(df).iloc[0]
    assert g["cmd_resid_fb_sum"] + g["cmd_resid_nfb_sum"] == pytest.approx(
        g["cmd_resid_sum"])
    assert g["fb_pitches"] == 1


# --- the vacuity quantity ----------------------------------------------------

def test_year_over_year_uses_only_pitchers_over_the_floor_in_both_years():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(40):
        skill = rng.normal()
        for season in (2023, 2024):
            n = 2000 if p < 30 else 100
            rows.append({"pitcher": p, "season": season, "month": 5,
                         "pitches": n,
                         "cmd_resid_sum": n * (skill * 0.01
                                               + rng.normal(0, 0.001)),
                         "takens": n * 0.5, "cs_resid_sum": 0.0})
    monthly = pd.DataFrame(rows)
    v = year_over_year(monthly, "cmd_resid", min_pitches=1000)
    assert v["n_pairs"] == 30            # the 10 short pitcher-seasons are out
    assert v["r"] > 0.9                  # the construction is nearly noiseless
    agg = season_aggregate(monthly)
    assert set(agg["season"]) == {2023, 2024}


# --- leakage guard 1: the stage-1 walk-forward fit ---------------------------

def test_a_pitch_from_the_scored_season_cannot_enter_its_own_model():
    """`fit_command_model` re-checks the training frame itself rather than
    trusting the caller's season filter."""
    from src.models.command import fit_command_model

    train = feats([pitch(date="2023-05-01"), pitch(date="2024-05-01")])
    with pytest.raises(ValueError, match="leakage"):
        fit_command_model(train, "csw", "pitch_type", 2024)
    with pytest.raises(ValueError, match="no training pitches"):
        fit_command_model(train.iloc[:0], "csw", "pitch_type", 2024)
