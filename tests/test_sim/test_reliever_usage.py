"""Unit tests for the reliever-availability term (src/sim/reliever_usage.py).

All synthetic — no network. What has to hold:
  * pitch counts come from the game log, and fall back to batters faced only
    when the log has none
  * a start counts as work (an opener's arm is just as tired), even though it
    is not bullpen work
  * d1/d2/d3 are calendar days before the game, doubleheaders summed
  * the three hard stops fire exactly where they are documented, and the taper
    between them is monotone and bounded
  * a rested pen is scored exactly as the whole pen; a tired closer's innings
    fall to whoever is left
  * tonight's starter is not in the pen behind himself
  * **usage from the game's own date or later never reaches the weights**
"""
import numpy as np
import pandas as pd
import pytest

from src.sim import bullpen as bp
from src.sim import reliever_usage as ru

LG_RA9 = 4.40


def log(pitcher, team, date, pitches=15, bf=4, outs=3, gs=0):
    return {"pitcher": pitcher, "team": team, "date": date, "pitches": pitches,
            "bf": bf, "outs": outs, "gs": gs}


def frame(rows):
    return pd.DataFrame(rows)


# A three-man pen for club 100: 1 is the closer (most work), 2 the setup man,
# 3 the mop-up arm. Club 200 has one reliever, 9.
PEN = bp.pen_window(bp.relief_appearances(frame([
    log(1, 100, "2026-05-01"), log(1, 100, "2026-05-03"), log(1, 100, "2026-05-05"),
    log(2, 100, "2026-05-02"), log(2, 100, "2026-05-05"),
    log(3, 100, "2026-05-04", bf=2),
    log(9, 200, "2026-05-05"),
])), "2026-05-06")
PEN_100 = PEN[PEN["team"] == 100]
RA9 = {1: 3.0, 2: 4.4, 3: 6.0, 9: 4.4}


# ─── pitch counts off the game log ───

def test_the_log_s_own_pitch_count_is_used():
    out = ru.appearance_pitches(frame([log(1, 100, "2026-05-01", pitches=37, bf=8)]))
    assert out["pitches"].tolist() == [37.0]


def test_a_missing_pitch_count_falls_back_to_batters_faced():
    # 30 pitches over 10 batters faced sets the league rate at 3.0; the second
    # appearance has no count and is estimated at 4 x 3.0.
    logs = frame([log(1, 100, "2026-05-01", pitches=30, bf=10),
                  log(2, 100, "2026-05-01", pitches=0, bf=4)])
    out = ru.appearance_pitches(logs)
    assert out["pitches"].tolist() == [30.0, 12.0]


def test_a_log_with_no_pitch_column_at_all_still_works():
    logs = frame([{"pitcher": 1, "team": 100, "date": "2026-05-01", "bf": 10}])
    out = ru.appearance_pitches(logs)
    assert out["pitches"].tolist() == [10 * ru.DEFAULT_PITCHES_PER_BF]


def test_pitches_per_bf_ignores_rows_with_no_count():
    logs = frame([log(1, 100, "2026-05-01", pitches=40, bf=10),
                  log(2, 100, "2026-05-01", pitches=0, bf=99)])
    assert ru.pitches_per_bf(logs) == pytest.approx(4.0)


def test_pitches_per_bf_falls_back_when_nothing_is_measurable():
    logs = frame([log(1, 100, "2026-05-01", pitches=0, bf=0)])
    assert ru.pitches_per_bf(logs) == ru.DEFAULT_PITCHES_PER_BF


def test_a_start_is_work_even_though_it_is_not_bullpen_work():
    logs = frame([log(7, 100, "2026-05-01", gs=1, pitches=95)])
    assert ru.appearance_pitches(logs)["pitches"].tolist() == [95.0]
    # ...and the same outing is *not* a bullpen appearance.
    assert bp.relief_appearances(logs).empty


def test_an_empty_log_gives_an_empty_frame():
    out = ru.appearance_pitches(frame([]))
    assert out.empty and list(out.columns) == ru.APPEARANCE_COLS


# ─── the trailing pitch window ───

def test_days_back_are_calendar_days_before_the_game():
    app = ru.appearance_pitches(frame([
        log(1, 100, "2026-05-05", pitches=20),
        log(1, 100, "2026-05-04", pitches=10),
        log(1, 100, "2026-05-03", pitches=5),
    ]))
    row = ru.recent_pitches(app, "2026-05-06").set_index("pitcher").loc[1]
    assert (row["d1"], row["d2"], row["d3"]) == (20.0, 10.0, 5.0)


def test_two_outings_on_the_same_day_are_summed():
    app = ru.appearance_pitches(frame([log(1, 100, "2026-05-05", pitches=12),
                                       log(1, 100, "2026-05-05", pitches=9)]))
    assert ru.recent_pitches(app, "2026-05-06")["d1"].tolist() == [21.0]


def test_an_off_day_pushes_the_work_further_back():
    app = ru.appearance_pitches(frame([log(1, 100, "2026-05-04", pitches=30)]))
    row = ru.recent_pitches(app, "2026-05-06").set_index("pitcher").loc[1]
    assert (row["d1"], row["d2"]) == (0.0, 30.0)


def test_work_older_than_the_window_does_not_appear():
    app = ru.appearance_pitches(frame([log(1, 100, "2026-05-01", pitches=45)]))
    assert ru.recent_pitches(app, "2026-05-06").empty


def test_an_empty_window_has_the_day_columns():
    out = ru.recent_pitches(ru.appearance_pitches(frame([])), "2026-05-06")
    assert list(out.columns) == ["pitcher", *ru.DAY_COLS]


# ─── the availability rule ───

def test_a_rested_arm_is_fully_available():
    assert ru.availability_weight(0.0, 0.0, 0.0) == 1.0


def test_heavy_work_yesterday_rules_him_out():
    assert ru.availability_weight(ru.HARD_1D_PITCHES, 0.0, 0.0) == 0.0
    assert ru.availability_weight(ru.HARD_1D_PITCHES - 1, 0.0, 0.0) > 0.0


def test_heavy_work_across_two_days_rules_him_out():
    half = ru.HARD_2D_PITCHES / 2.0
    assert half < ru.HARD_1D_PITCHES        # neither day alone is a hard stop
    assert ru.availability_weight(half, half, 0.0) == 0.0


def test_three_days_running_rules_him_out_however_light():
    assert ru.availability_weight(1.0, 1.0, 1.0) == 0.0
    assert ru.availability_weight(1.0, 1.0, 0.0) > 0.0


def test_the_taper_is_monotone_between_the_hard_stops():
    w = [ru.availability_weight(p, 0.0, 0.0) for p in (0, 5, 10, 20, 30)]
    assert all(a > b for a, b in zip(w, w[1:]))
    assert all(0.0 <= x <= 1.0 for x in w)


def test_older_work_counts_less_than_yesterday_s():
    assert (ru.availability_weight(0.0, 20.0, 0.0)
            > ru.availability_weight(20.0, 0.0, 0.0))
    assert (ru.availability_weight(0.0, 0.0, 20.0)
            > ru.availability_weight(0.0, 20.0, 0.0))


def test_the_weight_is_vectorized():
    w = ru.availability_weight(np.array([0.0, 20.0, 99.0]), np.zeros(3), np.zeros(3))
    assert w.shape == (3,)
    assert w[0] == 1.0 and 0.0 < w[1] < 1.0 and w[2] == 0.0


def test_only_arms_that_worked_recently_are_in_the_map():
    app = ru.appearance_pitches(frame([log(1, 100, "2026-05-05", pitches=20),
                                       log(2, 100, "2026-04-01", pitches=20)]))
    assert set(ru.availability(app, "2026-05-06")) == {1}


# ─── the available pen's rate ───

def test_a_fully_rested_pen_scores_exactly_as_the_whole_pen():
    assert ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {}) == pytest.approx(
        bp.pen_ra9(PEN_100, RA9, LG_RA9))


def test_a_tired_closer_makes_the_pen_worse():
    rested = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {})
    tired = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {1: 0.0})
    assert tired > rested


def test_a_tired_mop_up_man_makes_the_pen_better():
    rested = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {})
    assert ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {3: 0.0}) < rested


def test_a_half_available_arm_lands_between_full_and_none():
    full = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {})
    none = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {1: 0.0})
    half = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {1: 0.5})
    assert full < half < none


def test_tonight_s_starter_is_not_in_the_pen_behind_himself():
    with_him = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {})
    without = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {}, exclude=[1])
    assert without > with_him


def test_a_reliever_with_no_history_is_scored_at_league_average():
    pen = pd.DataFrame({"team": [100], "pitcher": [42], "bf": [10.0]})
    assert ru.available_pen_ra9(pen, RA9, LG_RA9, {}) == pytest.approx(LG_RA9)


def test_an_empty_pen_falls_back_to_league_average():
    assert ru.available_pen_ra9(PEN_100.iloc[:0], RA9, LG_RA9, {}) == LG_RA9
    assert ru.available_pen_ra9(None, RA9, LG_RA9, {}) == LG_RA9


def test_a_wholly_exhausted_pen_falls_back_to_the_whole_pen():
    """Everyone worked last night; the club still has to pitch the game."""
    out = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {1: 0.0, 2: 0.0, 3: 0.0})
    assert out == pytest.approx(bp.pen_ra9(PEN_100, RA9, LG_RA9))


def test_the_league_baseline_pools_every_club():
    lg = ru.league_available_pen_ra9(PEN, RA9, LG_RA9, {})
    one = ru.available_pen_ra9(PEN_100, RA9, LG_RA9, {})
    assert lg != one
    assert min(RA9.values()) <= lg <= max(RA9.values())


def test_a_league_average_pen_leaves_the_team_rate_untouched():
    """The term is a delta: same rate in, same rate out."""
    pen = pd.DataFrame({"team": [100], "pitcher": [2], "bf": [10.0]})
    avail = ru.available_pen_ra9(pen, RA9, LG_RA9, {})
    assert bp.blend_bullpen_team(avail, 4.10, avail) == pytest.approx(4.10)


# ─── the leakage guard ───

def test_usage_on_the_game_s_own_date_never_reaches_the_weights():
    """The outing being predicted — or an earlier game of the same
    doubleheader — has not happened when the line is priced."""
    before = frame([log(1, 100, "2026-05-05", pitches=20)])
    after = frame([log(1, 100, "2026-05-05", pitches=20),
                   log(1, 100, "2026-05-06", pitches=45),   # the game itself
                   log(1, 100, "2026-05-07", pitches=45)])  # tomorrow
    as_of = "2026-05-06"
    assert (ru.availability(ru.appearance_pitches(after), as_of)
            == ru.availability(ru.appearance_pitches(before), as_of))


def test_a_future_outing_cannot_make_a_rested_arm_unavailable():
    app = ru.appearance_pitches(frame([log(1, 100, "2026-05-06", pitches=99),
                                       log(1, 100, "2026-05-09", pitches=99)]))
    assert ru.availability(app, "2026-05-06") == {}
    assert ru.available_pen_ra9(
        PEN_100, RA9, LG_RA9, ru.availability(app, "2026-05-06")
    ) == pytest.approx(bp.pen_ra9(PEN_100, RA9, LG_RA9))


def test_the_window_moves_forward_with_the_date():
    """The same log read one day later sees the work one day further back."""
    heavy = ru.HARD_1D_PITCHES
    app = ru.appearance_pitches(frame([log(1, 100, "2026-05-05", pitches=heavy)]))
    assert ru.availability(app, "2026-05-06")[1] == 0.0
    assert ru.availability(app, "2026-05-08")[1] > 0.0
    assert ru.availability(app, "2026-05-09") == {}
