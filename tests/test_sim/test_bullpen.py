"""Unit tests for the station E bullpen-availability term (src/sim/bullpen.py).

All synthetic — no network. What has to hold:
  * a starter's own outing is not a bullpen appearance
  * the pen is who threw for *this* club inside the trailing window, weighted
    by how much he threw
  * back-to-back work makes a reliever unavailable, an off day rests everyone
  * excluding the arms a manager leans on moves the pen's RA/9 the most
  * a fully available pen leaves the team's runs-allowed rate exactly untouched
  * the starter and bullpen deltas partition the nine innings
  * appearances on or after the date being predicted never enter (leakage)
"""
import numpy as np
import pandas as pd
import pytest

from src.sim import bullpen as bp
from src.sim import starters as sp

LG_RA9 = 4.40


def log(pitcher, team, date, bf=4, outs=3, gs=0):
    return {"pitcher": pitcher, "team": team, "date": date, "bf": bf,
            "outs": outs, "gs": gs}


def frame(rows):
    return pd.DataFrame(rows)


# A three-man pen for club 100: 1 is the closer (most work), 2 the setup man,
# 3 the mop-up arm. Club 200 has one reliever, 9.
PEN = frame([
    log(1, 100, "2026-05-01"), log(1, 100, "2026-05-03"), log(1, 100, "2026-05-05"),
    log(2, 100, "2026-05-02"), log(2, 100, "2026-05-05"),
    log(3, 100, "2026-05-04", bf=2),
    log(9, 200, "2026-05-05"),
])
RA9 = {1: 3.0, 2: 4.4, 3: 6.0, 9: 4.4}


# ─── relief_appearances ───

def test_a_start_is_not_a_bullpen_appearance():
    out = bp.relief_appearances(frame([log(1, 100, "2026-05-01", gs=1),
                                       log(2, 100, "2026-05-01")]))
    assert out["pitcher"].tolist() == [2]


def test_a_swingman_counts_only_on_the_days_he_relieved():
    out = bp.relief_appearances(frame([log(7, 100, "2026-05-01", gs=1),
                                       log(7, 100, "2026-05-07")]))
    assert out["date"].tolist() == ["2026-05-07"]


def test_an_appearance_with_no_club_is_dropped():
    out = bp.relief_appearances(frame([log(1, None, "2026-05-01"),
                                       log(2, 100, "2026-05-01")]))
    assert out["pitcher"].tolist() == [2]


def test_an_empty_log_gives_an_empty_frame():
    assert bp.relief_appearances(pd.DataFrame()).empty


# ─── pen_window ───

def test_the_pen_is_grouped_by_club_and_weighted_by_work():
    w = bp.pen_window(bp.relief_appearances(PEN), "2026-05-06", days=21)
    club = w[w["team"] == 100].set_index("pitcher")["bf"]
    assert club[1] == 12.0 and club[2] == 8.0 and club[3] == 2.0
    assert w[w["team"] == 200]["pitcher"].tolist() == [9]


def test_appearances_outside_the_window_do_not_make_the_pen():
    rel = bp.relief_appearances(frame([log(4, 100, "2026-04-01"),
                                       log(1, 100, "2026-05-05")]))
    w = bp.pen_window(rel, "2026-05-06", days=21)
    assert w["pitcher"].tolist() == [1]


def test_a_shorter_window_keeps_only_the_most_recent_work():
    w = bp.pen_window(bp.relief_appearances(PEN), "2026-05-06", days=1)
    assert set(w["pitcher"]) == {1, 2, 9}          # 3 last threw on 05-04


def test_no_history_yet_is_an_empty_pen():
    assert bp.pen_window(bp.relief_appearances(PEN), "2026-04-01").empty
    assert bp.pen_window(pd.DataFrame(columns=bp.APPEARANCE_COLS), "2026-05-06").empty


# ─── unavailable ───

def test_three_straight_days_of_work_makes_a_reliever_unavailable():
    rel = bp.relief_appearances(frame([log(1, 100, "2026-05-03"),
                                       log(1, 100, "2026-05-04"),
                                       log(1, 100, "2026-05-05"),
                                       log(2, 100, "2026-05-05")]))
    assert bp.unavailable(rel, "2026-05-06") == {1}


def test_back_to_back_alone_still_leaves_him_available():
    """The default rule is three straight days, not two."""
    rel = bp.relief_appearances(frame([log(2, 100, "2026-05-04"),
                                       log(2, 100, "2026-05-05")]))
    assert bp.unavailable(rel, "2026-05-06") == set()
    # ...but the stricter readings of the rule do take him out
    assert bp.unavailable(rel, "2026-05-06", days=2, min_days=2) == {2}
    assert bp.unavailable(rel, "2026-05-06", days=1, min_days=1) == {2}


def test_an_off_day_rests_everyone():
    rel = bp.relief_appearances(frame([log(1, 100, "2026-05-01"),
                                       log(1, 100, "2026-05-02"),
                                       log(1, 100, "2026-05-03")]))
    assert bp.unavailable(rel, "2026-05-06") == set()


def test_two_outings_on_the_same_day_are_one_day_of_work():
    """A doubleheader is one calendar day, not two days of work."""
    rel = bp.relief_appearances(frame([log(1, 100, "2026-05-05"),
                                       log(1, 100, "2026-05-05")]))
    assert bp.unavailable(rel, "2026-05-06", days=2, min_days=2) == set()


def test_the_game_being_predicted_never_counts_as_work():
    """Leakage: an appearance on the date itself is the game we are pricing."""
    rel = bp.relief_appearances(frame([log(1, 100, "2026-05-04"),
                                       log(1, 100, "2026-05-05"),
                                       log(1, 100, "2026-05-06")]))
    assert bp.unavailable(rel, "2026-05-06") == set()
    assert bp.unavailable(rel, "2026-05-07") == {1}


# ─── pen_ra9 ───

def club_pen(as_of="2026-05-06", days=21, team=100):
    w = bp.pen_window(bp.relief_appearances(PEN), as_of, days=days)
    return w[w["team"] == team]


def test_the_pen_rate_is_weighted_by_trailing_workload():
    got = bp.pen_ra9(club_pen(), RA9, LG_RA9)
    want = (12 * 3.0 + 8 * 4.4 + 2 * 6.0) / 22
    assert got == pytest.approx(want)


def test_losing_the_best_arm_makes_the_pen_worse():
    full = bp.pen_ra9(club_pen(), RA9, LG_RA9)
    short = bp.pen_ra9(club_pen(), RA9, LG_RA9, exclude=[1])
    assert short > full
    assert short == pytest.approx((8 * 4.4 + 2 * 6.0) / 10)


def test_losing_the_mop_up_man_makes_the_pen_better():
    assert bp.pen_ra9(club_pen(), RA9, LG_RA9, exclude=[3]) < \
        bp.pen_ra9(club_pen(), RA9, LG_RA9)


def test_a_reliever_with_no_history_is_scored_at_league_average():
    assert bp.pen_ra9(club_pen(), {}, LG_RA9) == pytest.approx(LG_RA9)


def test_an_empty_or_fully_excluded_pen_falls_back_to_league_average():
    assert bp.pen_ra9(club_pen(), RA9, LG_RA9, exclude=[1, 2, 3]) == pytest.approx(LG_RA9)
    assert bp.pen_ra9(pd.DataFrame(columns=["team", "pitcher", "bf"]),
                      RA9, LG_RA9) == pytest.approx(LG_RA9)


def test_the_league_baseline_pools_every_club():
    w = bp.pen_window(bp.relief_appearances(PEN), "2026-05-06")
    pooled = bp.league_pen_ra9(w, RA9, LG_RA9)
    assert min(RA9.values()) < pooled < max(RA9.values())


# ─── blend_bullpen_team ───

def test_a_fully_available_pen_leaves_the_team_rate_untouched():
    for team_ra in (3.6, 4.4, 5.2):
        assert bp.blend_bullpen_team(4.1, team_ra, 4.1) == pytest.approx(team_ra)


def test_a_depleted_pen_costs_exactly_the_relief_share_of_the_gap():
    got = bp.blend_bullpen_team(4.9, 4.4, 4.3)
    assert got == pytest.approx(4.4 + (3.5 / 9.0) * 0.6)
    assert bp.blend_bullpen_team(4.0, 4.4, 4.3) < 4.4


def test_the_starter_and_the_pen_partition_the_nine_innings():
    assert bp.RELIEF_IP + sp.STARTER_IP == pytest.approx(sp.GAME_IP)
    ra = bp.blend_bullpen_team(
        5.4, sp.blend_starter_team(5.4, 4.4, 4.4), 4.4)
    # a staff a full run worse than league everywhere costs a full run
    assert ra == pytest.approx(4.4 + 1.0)


def test_team_context_survives_the_adjustment():
    good, bad = 3.6, 5.2
    assert (bp.blend_bullpen_team(4.9, bad, 4.4)
            - bp.blend_bullpen_team(4.9, good, 4.4)) == pytest.approx(bad - good)


def test_blend_is_vectorized():
    out = bp.blend_bullpen_team(np.array([4.0, 5.0]), np.array([4.4, 4.4]), 4.4)
    assert out.shape == (2,)
    assert out[0] < 4.4 < out[1]
