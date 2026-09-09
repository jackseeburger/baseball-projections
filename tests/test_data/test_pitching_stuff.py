"""Pitch selection, the swing/whiff mapping, and the monthly reduction.

The swing mapping is the test that matters most at this end. A stuff model is
only measuring "misses bats" if the denominator is swings and the numerator is
swings that missed; get `foul_tip` on the wrong side of that line and the
model is fitted on a different quantity with the same name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.pitching_stuff import (
    COUNT_COLUMNS,
    FASTBALL_TYPES,
    MIN_FASTBALLS,
    competitive_pitches,
    fastball_reference,
    monthly_buckets,
    pitch_features,
)


def pitch(description="ball", pitch_type="FF", pitcher=1, date="2024-04-10",
          speed=94.0, **over):
    """One Statcast pitch row, physics filled with plausible values."""
    row = {
        "game_type": "R", "game_date": date, "game_year": int(date[:4]),
        "pitcher": pitcher, "p_throws": "R", "stand": "R",
        "description": description, "pitch_type": pitch_type,
        "release_speed": speed, "effective_speed": speed + 0.5,
        "pfx_x": -0.5, "pfx_z": 1.4, "release_pos_x": -1.8,
        "release_pos_y": 54.0, "release_pos_z": 6.0, "release_extension": 6.4,
        "release_spin_rate": 2300.0, "spin_axis": 210.0, "arm_angle": 45.0,
        "vx0": 5.0, "vy0": -136.0, "vz0": -5.0, "ax": -8.0, "ay": 28.0,
        "az": -14.0, "plate_x": 0.1, "plate_z": 2.5,
    }
    row.update(over)
    return row


def frame(rows):
    return pd.DataFrame(rows)


# --- selection and the swing mapping -----------------------------------------

def test_pitchouts_bunts_and_position_players_are_dropped():
    df = frame([pitch(description="ball"),
                pitch(description="pitchout"),
                pitch(description="intent_ball"),
                pitch(description="foul_bunt"),
                pitch(description="missed_bunt"),
                pitch(pitch_type="IN"),
                pitch(game_type="S"),
                pitch(speed=0.0)])
    assert len(competitive_pitches(df)) == 1


@pytest.mark.parametrize("description,swing,whiff,csw", [
    ("swinging_strike", True, True, True),
    ("swinging_strike_blocked", True, True, True),
    # A foul tip is a swing the bat *touched*. It is a strike, but the bat did
    # not miss, so it is not a whiff and it is not a called strike either.
    ("foul_tip", True, False, False),
    ("foul", True, False, False),
    ("hit_into_play", True, False, False),
    ("called_strike", False, False, True),
    ("ball", False, False, False),
    ("blocked_ball", False, False, False),
    ("hit_by_pitch", False, False, False),
])
def test_target_mapping(description, swing, whiff, csw):
    f = pitch_features(competitive_pitches(frame([pitch(description=description)])))
    assert bool(f["is_swing"].iloc[0]) is swing
    assert bool(f["is_whiff"].iloc[0]) is whiff
    assert bool(f["is_csw"].iloc[0]) is csw


def test_zero_filled_tracking_reads_as_missing():
    """The archive writes 0 for a field that does not exist that season."""
    df = frame([pitch(spin_axis=0.0, arm_angle=0.0, release_spin_rate=0.0)])
    f = pitch_features(competitive_pitches(df))
    assert np.isnan(f["arm_angle"].iloc[0])
    assert np.isnan(f["spin_axis_sin"].iloc[0])
    assert np.isnan(f["release_spin_rate"].iloc[0])


def test_horizontal_quantities_are_mirrored_to_the_arm_side():
    """A lefty's arm-side run and a righty's are the same pitch."""
    r = pitch_features(competitive_pitches(frame([pitch(p_throws="R", pfx_x=-0.9)])))
    l = pitch_features(competitive_pitches(frame([pitch(p_throws="L", pfx_x=0.9)])))
    assert r["pfx_x_arm"].iloc[0] == pytest.approx(l["pfx_x_arm"].iloc[0])
    assert r["is_rhp"].iloc[0] == 1.0 and l["is_rhp"].iloc[0] == 0.0


# --- the fastball reference ---------------------------------------------------

def test_fastball_reference_needs_enough_fastballs():
    few = pitch_features(competitive_pitches(frame(
        [pitch(pitch_type="FF")] * (MIN_FASTBALLS - 1)
        + [pitch(pitch_type="SL", speed=85.0)] * 50)))
    assert fastball_reference(few).empty
    # ... and with no reference the relative features are missing, not zero.
    assert few["d_velo_fb"].isna().all()

    enough = pitch_features(competitive_pitches(frame(
        [pitch(pitch_type="FF", speed=95.0)] * MIN_FASTBALLS
        + [pitch(pitch_type="SL", speed=85.0)] * 10)))
    sl = enough[enough["pitch_type"] == "SL"]
    assert sl["d_velo_fb"].iloc[0] == pytest.approx(-10.0, abs=1e-3)


def test_cutters_are_not_fastballs_for_the_reference():
    assert "FC" not in FASTBALL_TYPES
    df = pitch_features(competitive_pitches(frame(
        [pitch(pitch_type="FF", speed=95.0)] * MIN_FASTBALLS
        + [pitch(pitch_type="FC", speed=90.0)] * MIN_FASTBALLS)))
    assert df["fb_velo"].iloc[0] == pytest.approx(95.0, abs=1e-3)


# --- the monthly reduction ---------------------------------------------------

def scored(rows, p_whiff=0.3, p_csw=0.28):
    f = pitch_features(competitive_pitches(frame(rows)))
    return f.assign(p_whiff=p_whiff, p_csw=p_csw)


def test_buckets_are_additive_over_months():
    """A bucket is a sum, so two months summed equal one bucket of both."""
    april = scored([pitch(date="2024-04-10", description="swinging_strike")] * 3)
    may = scored([pitch(date="2024-05-10", description="foul")] * 2)
    parts = monthly_buckets(pd.concat([april, may], ignore_index=True))
    assert sorted(parts["month"]) == [4, 5]
    both = parts[COUNT_COLUMNS].sum()
    one = monthly_buckets(pd.concat([april, may], ignore_index=True).assign(
        game_date=pd.Timestamp("2024-04-10")))[COUNT_COLUMNS].sum()
    pd.testing.assert_series_equal(both, one)


def test_predicted_whiff_is_summed_over_swings_only():
    """A whiff rate is per swing; summing it over takes would make the
    aggregate a swing-rate measurement wearing a whiff's name."""
    df = scored([pitch(description="swinging_strike"),
                 pitch(description="ball"),
                 pitch(description="called_strike")], p_whiff=0.5, p_csw=0.4)
    g = monthly_buckets(df).iloc[0]
    assert g["pitches"] == 3 and g["swings"] == 1
    assert g["p_whiff_sum"] == pytest.approx(0.5)
    assert g["p_csw_sum"] == pytest.approx(1.2)


def test_fastball_split_partitions_the_totals():
    df = scored([pitch(pitch_type="FF", description="swinging_strike"),
                 pitch(pitch_type="SL", description="foul"),
                 pitch(pitch_type="CH", description="ball")])
    g = monthly_buckets(df).iloc[0]
    for stem in ("pitches", "swings", "p_whiff_sum", "p_csw_sum", "sum_velo"):
        assert g[f"fb_{stem}"] + g[f"nfb_{stem}"] == pytest.approx(g[stem])
    assert g["n_ff"] == 1 and g["n_sl"] == 1 and g["n_ch"] == 1


def test_the_sidecar_records_when_the_build_ran_and_what_it_touched(tmp_path):
    """`scripts/check_freshness.py` is stdlib-only, so the build timestamp
    lives in a JSON sidecar rather than inside the parquet (BAS-79)."""
    import json

    from src.data.pitching_stuff import meta_path, write_meta

    parquet = tmp_path / "pitching_stuff_monthly.parquet"
    assert meta_path(parquet) == tmp_path / "pitching_stuff_monthly.meta.json"

    out = write_meta(parquet, built_at="2026-09-09T04:11:00+00:00",
                     seasons_built=[2026])
    payload = json.loads(out.read_text())
    assert payload["built_at"] == "2026-09-09T04:11:00+00:00"
    assert payload["seasons_built"] == [2026]
