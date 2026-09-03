"""The batted-ball filter and the monthly buckets it feeds.

The archive fills missing values with 0 / "0" rather than nulls, so "no
tracked exit velocity" and "a batted ball hit at 0 mph" look identical on
disk. Every test here exists because getting that wrong would quietly put
zeros into a mean exit velocity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.contact_quality import (
    COUNT_COLUMNS,
    EV_BIN_COLUMNS,
    batted_balls,
    ev_bin_index,
    load_monthly,
    monthly_buckets,
    save_monthly,
)


def pitch(**kw):
    row = {
        "game_type": "R", "type": "X", "game_date": "2026-04-05",
        "game_year": 2026, "batter": 1, "pitcher": 100,
        "events": "single", "bb_type": "line_drive",
        "launch_speed": 100.0, "launch_angle": 15.0, "launch_speed_angle": 6.0,
    }
    row.update(kw)
    return row


def test_batted_balls_keeps_only_tracked_in_play_regular_season():
    df = pd.DataFrame([
        pitch(),                                   # keep
        pitch(game_type="S"),                      # spring training
        pitch(type="S"),                           # a swinging strike
        pitch(launch_speed=0.0),                   # the archive's "missing"
        pitch(events="sac_bunt"),                  # bunts are not BBE
        pitch(launch_angle=0.0),                   # a real 0-degree liner
    ])
    out = batted_balls(df)
    assert len(out) == 2
    assert set(out["launch_angle"]) == {15.0, 0.0}


def test_batted_balls_returns_datetimes():
    out = batted_balls(pd.DataFrame([pitch()]))
    assert out["game_date"].dtype.kind == "M"


def test_monthly_buckets_are_additive_counts():
    df = pd.DataFrame([
        pitch(game_date="2026-04-05", launch_speed=100.0, launch_angle=20.0),
        pitch(game_date="2026-04-20", launch_speed=90.0, launch_angle=40.0,
              launch_speed_angle=3.0, bb_type="fly_ball"),
        pitch(game_date="2026-05-02", launch_speed=80.0, launch_angle=-5.0,
              launch_speed_angle=2.0, bb_type="ground_ball"),
    ])
    g = monthly_buckets(batted_balls(df), "batter")
    april = g[g["month"] == 4].iloc[0]
    assert april["bbe"] == 2
    assert april["sum_ev"] == pytest.approx(190.0)
    assert april["n_barrel"] == 1            # only the 6
    assert april["n_hardhit"] == 1           # only the 100 mph
    assert april["n_sweetspot"] == 1         # 20 is in [8, 32], 40 is not
    assert april["n_ld"] == 1 and april["n_fb"] == 1
    may = g[g["month"] == 5].iloc[0]
    assert may["bbe"] == 1 and may["n_gb"] == 1 and may["n_sweetspot"] == 0
    # The histogram carries every batted ball exactly once.
    for _, row in g.iterrows():
        assert row[EV_BIN_COLUMNS].sum() == row["bbe"]


def test_monthly_buckets_both_ids_see_the_same_batted_ball():
    df = pd.DataFrame([pitch(batter=7, pitcher=99)])
    bb = batted_balls(df)
    h = monthly_buckets(bb, "batter")
    p = monthly_buckets(bb, "pitcher")
    assert h["bbe"].sum() == p["bbe"].sum() == 1
    assert int(h["batter"].iloc[0]) == 7 and int(p["pitcher"].iloc[0]) == 99


def test_ev_bin_index_is_monotone_and_bounded():
    idx = ev_bin_index(np.array([10.0, 61.0, 100.0, 200.0]))
    assert list(idx) == sorted(idx)
    assert idx[0] == 0
    assert idx[-1] == len(EV_BIN_COLUMNS) - 1


def test_empty_input_keeps_the_schema():
    g = monthly_buckets(batted_balls(pd.DataFrame([pitch(type="S")])), "batter")
    assert g.empty
    assert set(COUNT_COLUMNS).issubset(g.columns)


def test_save_and_load_round_trip(tmp_path):
    df = monthly_buckets(batted_balls(pd.DataFrame([pitch()])), "batter")
    df = df.rename(columns={"batter": "player"})
    df.insert(0, "side", "hitter")
    path = save_monthly(df, tmp_path / "cq.parquet")
    back = load_monthly(path)
    assert back["bbe"].dtype.kind == "f"
    assert back["bbe"].sum() == df["bbe"].sum()
    assert back["sum_ev"].iloc[0] == pytest.approx(df["sum_ev"].iloc[0])
