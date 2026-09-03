"""The stage-2 surface: its grid, its window, and its leakage guard.

`fit_surface` itself runs MCMC and is not exercised here — the same reason
`tests/test_eval/test_bayes_arm.py` does not sample. Everything around it
that could silently be wrong *is* exercised: which batted balls count, which
cell one lands in, which of them are inside a cutoff's window, and what a
player's number is once a surface exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.hsgp_contact import (
    EV_EDGES,
    LA_EDGES,
    Surface,
    batted_balls_from_pa,
    cell_centres,
    cell_index,
    grid_cells,
    player_values,
    shrink_and_standardize,
    window_batted_balls,
)


def pa_row(**kw):
    row = {"batter": 1, "pitcher": 100, "game_date": "2026-04-10",
           "game_year": 2026, "event": "single", "launch_speed": 100.0,
           "launch_angle": 15.0, "woba_value": 0.9, "woba_denom": 1.0}
    row.update(kw)
    return row


def test_only_tracked_non_bunt_batted_balls_survive():
    pa = pd.DataFrame([
        pa_row(),                              # keep
        pa_row(launch_speed=0.0),              # the archive's "missing"
        pa_row(event="sac_bunt"),              # not a batted-ball event
        pa_row(woba_denom=0.0),                # no wOBA denominator
        pa_row(launch_angle=None),             # untracked angle
    ])
    out = batted_balls_from_pa(pa)
    assert len(out) == 1
    assert out["month"].iloc[0] == 4
    assert out["value"].iloc[0] == pytest.approx(0.9)


def test_cell_index_is_inside_the_grid_for_anything():
    i, j = cell_index(np.array([-50.0, 95.0, 500.0]),
                      np.array([-200.0, 12.0, 200.0]))
    assert i.min() >= 0 and i.max() <= len(EV_EDGES) - 2
    assert j.min() >= 0 and j.max() <= len(LA_EDGES) - 2


def test_cell_centres_line_up_with_the_index():
    ev_c, la_c = cell_centres()
    i, j = cell_index(np.array([100.1]), np.array([12.0]))
    assert abs(ev_c[i[0]] - 100.1) <= 1.25 + 1e-9
    assert abs(la_c[j[0]] - 12.0) <= 2.5 + 1e-9


def test_grid_cells_drops_the_thin_ones_and_keeps_the_counts():
    bb = pd.DataFrame({
        "launch_speed": [100.0] * 30 + [60.0] * 3,
        "launch_angle": [15.0] * 30 + [-20.0] * 3,
        "value": [1.0] * 30 + [0.0] * 3,
    })
    g = grid_cells(bb, min_bbe=25)
    assert len(g) == 1
    assert int(g["n"].iloc[0]) == 30
    assert g["mean"].iloc[0] == pytest.approx(1.0)


# --- the leakage guard -------------------------------------------------------

def bb_frame():
    """One batter, three seasons, every month, one ball a month."""
    rows = []
    for season in (2024, 2025, 2026):
        for month in range(3, 11):
            rows.append({"batter": 1, "pitcher": 9, "game_year": season,
                         "month": month, "launch_speed": 95.0,
                         "launch_angle": 10.0, "value": 1.0})
    return pd.DataFrame(rows)


def test_window_takes_prior_seasons_whole_and_the_current_one_cut():
    w = window_batted_balls(bb_frame(), "2026-06-01", 2026, (1.0, 1.0, 1.0))
    assert (w["game_year"] == 2026).sum() == 3       # March, April, May
    assert (w["game_year"] == 2025).sum() == 8       # all of it
    assert (w["game_year"] == 2024).sum() == 8


def test_window_never_admits_the_cutoff_month_or_later():
    for month, expected in ((5, 2), (7, 4), (8, 5)):
        w = window_batted_balls(bb_frame(), f"2026-0{month}-01", 2026,
                                (1.0, 0.0, 0.0))
        assert len(w) == expected
        assert w["month"].max() < month


def test_window_refuses_a_mid_month_cutoff():
    with pytest.raises(ValueError, match="not the first of a month"):
        window_batted_balls(bb_frame(), "2026-06-15", 2026, (1.0, 0.0, 0.0))


def test_window_excludes_seasons_outside_the_three():
    w = window_batted_balls(bb_frame(), "2026-04-01", 2026, (1.0, 1.0, 0.0))
    assert set(w["game_year"]) == {2025, 2026}


# --- the covariate -----------------------------------------------------------

def flat_surface(value=0.5):
    ev_c, la_c = cell_centres()
    return Surface(values=np.full((len(ev_c), len(la_c)), value),
                   diagnostics={}, seasons=())


def test_player_value_is_the_recency_weighted_mean_of_the_surface():
    ev_c, la_c = cell_centres()
    values = np.zeros((len(ev_c), len(la_c)))
    i_hot, j_hot = cell_index(np.array([110.0]), np.array([25.0]))
    values[i_hot[0], j_hot[0]] = 2.0
    surface = Surface(values=values, diagnostics={})
    bb = pd.DataFrame({
        "batter": [1, 1, 2], "pitcher": [9, 9, 9],
        "game_year": [2026, 2025, 2026], "month": [4, 4, 4],
        "launch_speed": [110.0, 110.0, 70.0],
        "launch_angle": [25.0, 25.0, 25.0], "value": [2.0, 2.0, 0.0],
    })
    vals = player_values(bb, surface, "batter", {2026: 1.0, 2025: 0.5}).set_index("player")
    assert vals.loc[1, "w"] == pytest.approx(1.5)
    assert vals.loc[1, "wv"] == pytest.approx(3.0)     # 2.0 * (1 + 0.5)
    assert vals.loc[1, "bbe_raw"] == 2.0
    assert vals.loc[2, "wv"] == pytest.approx(0.0)


def test_a_season_outside_the_weights_contributes_nothing():
    bb = pd.DataFrame({
        "batter": [1, 1], "pitcher": [9, 9], "game_year": [2026, 2019],
        "month": [4, 4], "launch_speed": [95.0, 95.0],
        "launch_angle": [10.0, 10.0], "value": [1.0, 1.0]})
    vals = player_values(bb, flat_surface(), "batter", {2026: 1.0})
    assert vals["bbe_raw"].iloc[0] == 1.0


def test_shrinkage_and_standardization_of_the_single_covariate():
    """Three players, because two standardize to the same pair whatever their
    values are — the z of a two-point set depends only on its weights."""
    vals = pd.DataFrame({"player": [1, 2, 3], "w": [1.0, 400.0, 400.0],
                         "wv": [2.0, 400.0 * 0.30, 400.0 * 0.45],
                         "bbe_raw": [1.0, 400.0, 400.0]})
    hard = shrink_and_standardize(vals, ballast=0.0)
    soft = shrink_and_standardize(vals, ballast=500.0)
    # With no ballast the one-ball player is the extreme of the three; with
    # 500 balls of league average behind him he is pulled inside them.
    assert hard["xcon"].iloc[0] > hard["xcon"].max() - 1e-9
    assert soft["xcon"].iloc[0] < soft["xcon"].max()
    assert np.average(soft["xcon"], weights=vals["bbe_raw"]) == pytest.approx(0.0)


def test_the_pitcher_side_reads_the_same_batted_balls():
    bb = pd.DataFrame({
        "batter": [1, 2], "pitcher": [9, 9], "game_year": [2026, 2026],
        "month": [4, 4], "launch_speed": [95.0, 95.0],
        "launch_angle": [10.0, 10.0], "value": [1.0, 0.0]})
    h = player_values(bb, flat_surface(), "batter", {2026: 1.0})
    p = player_values(bb, flat_surface(), "pitcher", {2026: 1.0})
    assert len(h) == 2 and len(p) == 1
    assert p["bbe_raw"].iloc[0] == 2.0
