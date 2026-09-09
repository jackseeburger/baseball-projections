"""Pitch selection, the zone decision, and the monthly reduction.

Two things here would quietly change what is being measured if they were
wrong. The swing mapping is BAS-71's and a `foul_tip` on the wrong side of it
turns a contact rate into a whiff rate with the same name. And the zone
fallback is what stands between "pitches outside the zone" and "pitches whose
`zone` field happened to be missing".
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.swing_decisions import (
    COUNT_COLUMNS,
    HALF_BALL_FT,
    ZONE_HALF_WIDTH_FT,
    competitive_pitches,
    load_monthly,
    monthly_buckets,
    pitch_flags,
    save_monthly,
    zone_flags,
)


def pitch(description="ball", zone=5, date="2024-04-10", batter=1, **over):
    row = {
        "game_type": "R", "game_date": date, "game_year": int(date[:4]),
        "batter": batter, "description": description, "pitch_type": "FF",
        "zone": zone, "plate_x": 0.0, "plate_z": 2.5,
        "sz_top": 3.4, "sz_bot": 1.6,
    }
    row.update(over)
    return row


def frame(rows):
    return pd.DataFrame(rows)


# --- selection ---------------------------------------------------------------

def test_bunts_pitchouts_and_intentional_balls_are_dropped():
    df = frame([pitch(description="ball"),
                pitch(description="pitchout"),
                pitch(description="intent_ball"),
                pitch(description="automatic_ball"),
                pitch(description="foul_bunt"),
                pitch(description="missed_bunt"),
                pitch(description="bunt_foul_tip"),
                pitch(pitch_type="IN"),
                pitch(game_type="S")])
    assert len(competitive_pitches(df)) == 1


def test_a_pitch_without_tracking_is_still_a_swing_decision():
    """Unlike the stuff model, this one does not need the physics block: a
    batter decided whether to swing whether or not the spin rate was measured."""
    out = competitive_pitches(frame([pitch(description="called_strike")]))
    assert len(out) == 1
    assert out["game_date"].dtype.kind == "M"


# --- the zone ----------------------------------------------------------------

def test_statcast_zone_decides_where_it_exists():
    df = frame([pitch(zone=z) for z in (1, 5, 9, 11, 12, 13, 14)])
    in_zone, located = zone_flags(df)
    assert located.all()
    assert list(in_zone) == [True, True, True, False, False, False, False]


def test_zone_falls_back_to_coordinates_with_a_half_ball_margin():
    # `zone` missing (the archive's 0 fill and a real null both count).
    just_in = pitch(zone=0, plate_x=ZONE_HALF_WIDTH_FT - 0.01, plate_z=2.5)
    just_out = pitch(zone=None, plate_x=ZONE_HALF_WIDTH_FT + 0.01, plate_z=2.5)
    top_edge = pitch(zone=0, plate_x=0.0, plate_z=3.4 + HALF_BALL_FT - 0.01)
    over_top = pitch(zone=0, plate_x=0.0, plate_z=3.4 + HALF_BALL_FT + 0.01)
    bot_edge = pitch(zone=0, plate_x=0.0, plate_z=1.6 - HALF_BALL_FT + 0.01)
    under = pitch(zone=0, plate_x=0.0, plate_z=1.6 - HALF_BALL_FT - 0.01)
    in_zone, located = zone_flags(frame([just_in, just_out, top_edge,
                                         over_top, bot_edge, under]))
    assert located.all()
    assert list(in_zone) == [True, False, True, False, True, False]


def test_the_fallback_uses_this_batters_own_zone():
    """Same coordinates, two batters: a pitch at 3.5 ft is a strike to the
    taller zone and a ball to the shorter one."""
    tall = pitch(zone=0, plate_z=3.5, sz_top=3.8, sz_bot=1.8)
    short = pitch(zone=0, plate_z=3.5, sz_top=3.1, sz_bot=1.5)
    in_zone, _ = zone_flags(frame([tall, short]))
    assert list(in_zone) == [True, False]


def test_a_pitch_with_no_usable_location_is_dropped_not_guessed():
    df = frame([pitch(zone=0, plate_x=None, plate_z=None),
                pitch(zone=0, sz_top=0.0, sz_bot=0.0)])
    _, located = zone_flags(df)
    assert not located.any()
    assert pitch_flags(df).empty


# --- the swing mapping and the counts ----------------------------------------

def test_the_swing_mapping_is_the_pinned_one():
    rows = [pitch(description=d) for d in
            ("ball", "called_strike", "hit_into_play", "foul", "foul_tip",
             "swinging_strike", "swinging_strike_blocked", "blocked_ball")]
    f = pitch_flags(frame(rows))
    assert f["pitches_z"].sum() == 8
    # five swings: in play, foul, foul tip, and the two swinging strikes
    assert f["swings_z"].sum() == 5
    # a foul tip touched the ball, so it is contact and not a whiff
    assert f["contact_z"].sum() == 3
    assert f["whiff_z"].sum() == 2
    assert f["called_z"].sum() == 1


def test_out_of_zone_counts_are_the_mirror_image():
    f = pitch_flags(frame([pitch(zone=13, description="swinging_strike"),
                           pitch(zone=13, description="foul"),
                           pitch(zone=13, description="ball")]))
    assert f["pitches_o"].sum() == 3 and f["pitches_z"].sum() == 0
    assert f["swings_o"].sum() == 2
    assert f["whiff_o"].sum() == 1 and f["contact_o"].sum() == 1
    # a called strike outside the zone is not a "called strike taken in zone"
    assert f["called_z"].sum() == 0


def test_monthly_buckets_are_additive_counts():
    rows = ([pitch(date="2024-04-05", description="swinging_strike")] * 3
            + [pitch(date="2024-05-05", description="ball", zone=13)] * 2)
    g = monthly_buckets(pitch_flags(frame(rows)))
    april = g[g["month"] == 4].iloc[0]
    may = g[g["month"] == 5].iloc[0]
    assert april["pitches_z"] == 3 and april["whiff_z"] == 3
    assert may["pitches_o"] == 2 and may["swings_o"] == 0
    assert set(COUNT_COLUMNS).issubset(g.columns)


def test_empty_input_keeps_the_schema():
    g = monthly_buckets(pitch_flags(competitive_pitches(
        frame([pitch(game_type="S")]))))
    assert g.empty
    assert set(COUNT_COLUMNS).issubset(g.columns)


def test_save_and_load_round_trip(tmp_path):
    g = monthly_buckets(pitch_flags(frame([pitch()])))
    path = save_monthly(g, tmp_path / "sd.parquet")
    back = load_monthly(path)
    assert back["pitches_z"].dtype.kind == "f"
    assert back["pitches_z"].sum() == pytest.approx(g["pitches_z"].sum())
