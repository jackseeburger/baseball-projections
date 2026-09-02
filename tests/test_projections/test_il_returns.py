"""Return-time distribution and the expected-share projection it feeds.

Everything here is synthetic and hand-checkable: a handful of spells whose
Kaplan-Meier curve can be worked out on paper, and a two-club roster where one
regular is on the injured list. The point of the file is the arithmetic and
the walk-forward guard — a transaction filed on or after the cutoff must not
be able to move a projection made at that cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projections import il_returns as ilr
from src.projections.playing_time import project_playing_time

SEASON_END = "2026-09-27"
CUTOFF = "2026-08-01"


def tx(player_id, date, type_code, description=""):
    return {"player_id": player_id, "date": date, "type_code": type_code,
            "description": description, "type_desc": "", "transaction_id": 0,
            "effective_date": date, "resolution_date": date,
            "from_team_id": None, "to_team_id": 100, "season": 2026}


def events(rows) -> pd.DataFrame:
    return ilr.parse_events(pd.DataFrame(rows))


# --- reading the feed ---

def test_classify_reads_the_three_injured_list_sentences():
    assert ilr.classify("SC", "Reds placed CF Ed on the 10-day injured list.") == (
        ilr.IL_PLACEMENT, "IL10")
    assert ilr.classify("SC", "Reds activated CF Ed from the 10-day injured list.") == (
        ilr.IL_ACTIVATION, "IL10")
    assert ilr.classify(
        "SC", "Reds transferred RHP Al from the 15-day injured list to the "
              "60-day injured list.") == (ilr.IL_TRANSFER, "IL60")


def test_classify_reads_options_recalls_and_departures():
    assert ilr.classify("OPT", "Reds optioned CF Ed to Louisville.") == (
        ilr.OPTION, ilr.OPTION_TYPE)
    assert ilr.classify("CU", "Reds recalled CF Ed from Louisville.") == (
        ilr.RECALL, ilr.OPTION_TYPE)
    assert ilr.classify("SE", "Reds selected the contract of CF Ed.") == (
        ilr.RECALL, ilr.OPTION_TYPE)
    assert ilr.classify("REL", "Reds released CF Ed.") == (ilr.DEPARTURE, None)
    # A bare activation closes a stint the feed did not label.
    assert ilr.classify("SC", "Reds activated CF Ed.") == (ilr.IL_ACTIVATION, None)


def test_classify_ignores_transactions_that_are_not_spell_events():
    assert ilr.classify("TR", "Reds traded CF Ed to the Cubs.") is None
    assert ilr.classify("NUM", "CF Ed changed his uniform number.") is None
    # A rehab assignment does *not* end an injured-list stint.
    assert ilr.classify("ASG", "Reds sent CF Ed on a rehab assignment to Dayton.") is None


def test_status_code_maps_to_a_spell_type():
    assert ilr.status_spell_type("D10") == "IL10"
    assert ilr.status_spell_type("D60") == "IL60"
    assert ilr.status_spell_type("RM") == ilr.OPTION_TYPE
    # Paternity, bereavement, restricted, suspended and active have no curve.
    for code in ("A", "PL", "BRV", "RL", "SU"):
        assert ilr.status_spell_type(code) is None


# --- spells ---

def test_a_placement_pairs_with_its_activation():
    ev = events([
        tx(1, "2026-05-01", "SC", "Reds placed CF Ed on the 10-day injured list."),
        tx(1, "2026-05-21", "SC", "Reds activated CF Ed from the 10-day injured list."),
    ])
    spells = ilr.build_spells(ev, SEASON_END)
    assert len(spells) == 1
    row = spells.iloc[0]
    assert (row["type"], row["entry_day"], row["exit_day"], row["returned"]) == (
        "IL10", 0, 20, True)


def test_a_spell_still_open_at_the_end_of_the_season_is_censored():
    ev = events([tx(1, "2026-09-07", "SC",
                    "Reds placed CF Ed on the 10-day injured list.")])
    spells = ilr.build_spells(ev, SEASON_END)
    assert len(spells) == 1
    assert spells.iloc[0]["exit_day"] == 20
    assert not spells.iloc[0]["returned"]


def test_an_option_pairs_with_its_recall():
    ev = events([
        tx(1, "2026-05-01", "OPT", "Reds optioned CF Ed to Louisville."),
        tx(1, "2026-05-15", "CU", "Reds recalled CF Ed from Louisville."),
    ])
    spells = ilr.build_spells(ev, SEASON_END)
    assert list(spells["type"]) == [ilr.OPTION_TYPE]
    assert spells.iloc[0]["exit_day"] == 14 and spells.iloc[0]["returned"]


def test_a_transfer_censors_the_short_list_and_left_truncates_the_long_one():
    ev = events([
        tx(1, "2026-05-01", "SC", "Reds placed RHP Al on the 15-day injured list."),
        tx(1, "2026-05-21", "SC", "Reds transferred RHP Al from the 15-day "
                                  "injured list to the 60-day injured list."),
        tx(1, "2026-08-01", "SC", "Reds activated RHP Al from the 60-day injured list."),
    ])
    spells = ilr.build_spells(ev, SEASON_END).set_index("type")
    # The 15-day half is watched for 20 days and then stops being a 15-day case.
    assert spells.loc["IL15", "entry_day"] == 0
    assert spells.loc["IL15", "exit_day"] == 20
    assert not spells.loc["IL15", "returned"]
    # The 60-day half is dated from the *original* placement — that is the
    # elapsed time a projection sees — but only enters the risk set on the day
    # of the transfer.
    assert spells.loc["IL60", "entry_day"] == 20
    assert spells.loc["IL60", "exit_day"] == 92
    assert spells.loc["IL60", "returned"]


def test_an_injured_list_spell_and_an_option_spell_can_run_at_once():
    ev = events([
        tx(1, "2026-05-01", "OPT", "Reds optioned CF Ed to Louisville."),
        tx(1, "2026-05-10", "SC", "Reds placed CF Ed on the 10-day injured list."),
        tx(1, "2026-05-30", "SC", "Reds activated CF Ed from the 10-day injured list."),
        tx(1, "2026-06-01", "CU", "Reds recalled CF Ed from Louisville."),
    ])
    spells = ilr.build_spells(ev, SEASON_END).set_index("type")
    assert spells.loc["IL10", "exit_day"] == 20
    assert spells.loc[ilr.OPTION_TYPE, "exit_day"] == 31
    assert bool(spells.loc["IL10", "returned"]) and bool(
        spells.loc[ilr.OPTION_TYPE, "returned"])


# --- the survival table ---

def _spells(rows, spell_type="IL10") -> pd.DataFrame:
    """(entry_day, exit_day, returned) triples as a spells frame."""
    return pd.DataFrame([
        {"player_id": i, "kind": ilr.IL, "type": spell_type,
         "start": pd.Timestamp("2026-05-01"), "entry_day": e, "exit_day": x,
         "returned": r}
        for i, (e, x, r) in enumerate(rows)])


def test_survival_table_matches_a_hand_computed_kaplan_meier():
    # Four spells: back on day 10, day 20 and day 30, and one censored on 15.
    table = ilr.survival_table(_spells([(0, 10, True), (0, 20, True),
                                        (0, 15, False), (0, 30, True)]))
    assert list(table["day"]) == [10, 20, 30]
    # Day 10: all four still out, one back  -> S = 3/4.
    # Day 20: only the 20 and the 30 are still at risk (the censored one left
    #         at 15 and the first returned) -> S = 3/4 x 1/2.
    # Day 30: one at risk, one back -> S = 0.
    assert list(table["at_risk"]) == [4, 2, 1]
    assert list(table["returns"]) == [1, 1, 1]
    np.testing.assert_allclose(table["survival"], [0.75, 0.375, 0.0])


def test_censoring_never_counts_as_a_return():
    # Every spell censored: nobody is recorded as coming back, so the table is
    # empty and the survival curve stays at one.
    table = ilr.survival_table(_spells([(0, 10, False), (0, 40, False)]))
    assert table.empty
    assert ilr.survival_at(table, "IL10", 60) == 1.0


def test_survival_is_a_step_function_read_at_any_day():
    table = ilr.survival_table(_spells([(0, 10, True), (0, 20, True),
                                        (0, 15, False), (0, 30, True)]))
    assert ilr.survival_at(table, "IL10", 0) == 1.0
    assert ilr.survival_at(table, "IL10", 9) == 1.0
    assert ilr.survival_at(table, "IL10", 10) == 0.75
    assert ilr.survival_at(table, "IL10", 19) == 0.75
    np.testing.assert_allclose(ilr.survival_at(table, "IL10", [10, 25, 40]),
                               [0.75, 0.375, 0.0])
    # A type with no curve at all is the old hard gate: nobody comes back.
    assert ilr.survival_at(table, "IL60", 30) == 1.0


# --- reading the curve conditionally ---

def test_conditional_probability_divides_out_the_time_already_served():
    table = ilr.survival_table(_spells([(0, 10, True), (0, 20, True),
                                        (0, 15, False), (0, 30, True)]))
    # P(back by 20 | still out at 10) = 1 - S(20)/S(10) = 1 - .375/.75.
    probs = ilr.return_probability(table, "IL10", elapsed=10, horizon_days=10)
    assert len(probs) == 10
    np.testing.assert_allclose(probs[:9], 0.0)
    np.testing.assert_allclose(probs[9], 0.5)
    # Unconditionally the same day is a much better bet.
    np.testing.assert_allclose(
        ilr.return_probability(table, "IL10", 0, 20)[-1], 0.625)


def test_expected_active_fraction_is_the_mean_of_the_daily_probabilities():
    table = ilr.survival_table(_spells([(0, 10, True), (0, 20, True),
                                        (0, 15, False), (0, 30, True)]))
    # Twelve days from day 10: nine at P = 0, then three at P = 0.5.
    assert ilr.expected_active_fraction(table, "IL10", 10, 12) == pytest.approx(0.125)
    assert ilr.expected_active_fraction(table, "IL10", 10, 0) == 0.0


def test_a_sixty_day_list_late_in_the_season_projects_to_nothing():
    # Nobody on this list came back inside three months.
    table = ilr.survival_table(_spells([(0, 92, True), (0, 100, True),
                                        (0, 120, False)], spell_type="IL60"))
    # Twenty days left in the season, a fortnight already served: no chance.
    assert ilr.expected_active_fraction(table, "IL60", 14, 20) == 0.0
    # Over a long enough horizon he is worth something again.
    assert ilr.expected_active_fraction(table, "IL60", 14, 120) > 0.1


def test_an_unknown_spell_type_keeps_the_old_hard_zero():
    table = ilr.survival_table(_spells([(0, 10, True)]))
    assert ilr.expected_active_fraction(table, "IL60", 5, 60) == 0.0


# --- who is out at the cutoff, and since when ---

def test_open_spells_carry_the_elapsed_days_at_the_cutoff():
    ev = events([tx(1, "2026-07-22", "SC",
                    "Reds placed CF Ed on the 10-day injured list.")])
    open_now = ilr.open_spells_at(ev, CUTOFF)
    assert len(open_now) == 1
    assert open_now.iloc[0]["type"] == "IL10"
    assert open_now.iloc[0]["elapsed_days"] == 10


def test_a_transfer_keeps_the_original_placement_date():
    ev = events([
        tx(1, "2026-06-01", "SC", "Reds placed RHP Al on the 15-day injured list."),
        tx(1, "2026-07-02", "SC", "Reds transferred RHP Al from the 15-day "
                                  "injured list to the 60-day injured list."),
    ])
    open_now = ilr.open_spells_at(ev, CUTOFF).iloc[0]
    assert open_now["type"] == "IL60"
    assert open_now["elapsed_days"] == 61


def test_an_activation_before_the_cutoff_closes_the_spell():
    ev = events([
        tx(1, "2026-07-01", "SC", "Reds placed CF Ed on the 10-day injured list."),
        tx(1, "2026-07-20", "SC", "Reds activated CF Ed from the 10-day injured list."),
    ])
    assert ilr.open_spells_at(ev, CUTOFF).empty


def test_transactions_on_or_after_the_cutoff_are_invisible():
    placed = tx(1, "2026-07-22", "SC",
                "Reds placed CF Ed on the 10-day injured list.")
    # Activated on the cutoff date itself: that is the cutoff morning's news,
    # not yesterday's, so the projection must still see him as injured.
    same_day = tx(1, CUTOFF, "SC",
                  "Reds activated CF Ed from the 10-day injured list.")
    later = tx(1, "2026-08-09", "SC",
               "Reds activated CF Ed from the 10-day injured list.")
    for extra in ([], [same_day], [same_day, later]):
        open_now = ilr.open_spells_at(events([placed] + extra), CUTOFF)
        assert len(open_now) == 1
        assert open_now.iloc[0]["elapsed_days"] == 10


# --- the expected share, end to end through the projection ---
#
# Two clubs of nine identical regulars, four plate appearances a day for the
# ninety days before the cutoff, plus one tenth hitter on club 100 who is
# identical to them until the day he goes on the ten-day list and stops
# playing. Everything in this section turns on one comparison: what he is
# projected for against what a hitter who never got hurt is projected for.

REGULARS = {100: list(range(101, 110)), 200: list(range(201, 210))}
INJURED = 110
HORIZON_DAYS = 30
# A list on which every stint lasts exactly 25 days, so the expected fraction
# of a horizon is a fraction of whole days that can be counted on fingers.
FIXED_STINT_DAYS = 25


def _fixed_stint_table(spell_type="IL10", days=FIXED_STINT_DAYS):
    return ilr.survival_table(_spells([(0, days, True)] * 20, spell_type=spell_type))


def make_logs(placed_on, days: int = 90) -> pd.DataFrame:
    """Ninety days of four-PA games; the injured hitter stops when placed."""
    dates = pd.date_range(end=pd.Timestamp(CUTOFF) - pd.Timedelta(days=1), periods=days)
    rows = []
    for date in dates:
        for team, ids in REGULARS.items():
            for b in ids:
                rows.append({"batter": b, "team_id": team,
                             "date": date.date().isoformat(), "pa": 4})
        if date < pd.Timestamp(placed_on):
            rows.append({"batter": INJURED, "team_id": 100,
                         "date": date.date().isoformat(), "pa": 4})
    return pd.DataFrame(rows)


def make_roster() -> pd.DataFrame:
    rows = [{"batter": b, "team_id": team, "status_code": "A"}
            for team, ids in REGULARS.items() for b in ids]
    rows.append({"batter": INJURED, "team_id": 100, "status_code": "D10"})
    return pd.DataFrame(rows)


def _project(placed_on, transactions=None, fractions=None, method="blend_il"):
    roster, logs = make_roster(), make_logs(placed_on)
    team_logs = logs.groupby(["team_id", "date"], as_index=False)["pa"].sum()
    remaining = pd.DataFrame({"team_id": [100, 200], "games_remaining": [30, 30]})
    if transactions is not None:
        fractions = ilr.expected_active_fractions(
            roster, ilr.parse_events(pd.DataFrame(transactions)),
            _fixed_stint_table(), CUTOFF, HORIZON_DAYS)
    proj = project_playing_time(roster, logs, remaining, CUTOFF,
                                team_logs=team_logs, method=method,
                                active_fractions=fractions)
    return proj.set_index("batter")["pa_share"]


def _placement(date):
    return [tx(INJURED, date, "SC", "Reds placed CF Ed on the 10-day injured list.")]


def test_the_injured_hitter_gets_his_preinjury_share_times_his_return_fraction():
    placed_on = "2026-07-12"          # twenty days before the cutoff
    fraction = ilr.expected_active_fraction(_fixed_stint_table(), "IL10", 20,
                                            HORIZON_DAYS)
    # Back on day 25, so available for 26 of the 30 remaining days.
    assert fraction == pytest.approx(26 / 30)

    share = _project(placed_on, transactions=_placement(placed_on))
    # He batted exactly as often as the nine regulars until the day he was
    # placed, so his pre-injury share is theirs and the only thing separating
    # them is the fraction of the horizon he is expected back for.
    assert share[INJURED] == pytest.approx(fraction * share[101])
    # The club's shares still add to one and the untouched club is untouched.
    assert share.loc[[*REGULARS[100], INJURED]].sum() == pytest.approx(1.0)
    assert share[201] == pytest.approx(1 / 9)


def test_a_hitter_placed_the_day_before_the_cutoff_is_still_a_regular():
    placed_on = pd.Timestamp(CUTOFF) - pd.Timedelta(days=1)
    fraction = ilr.expected_active_fraction(_fixed_stint_table(), "IL10", 1,
                                            HORIZON_DAYS)
    assert fraction == pytest.approx(7 / 30)      # back on day 25 of 31

    share = _project(placed_on, transactions=_placement(placed_on.date().isoformat()))
    # His trailing window is empty of the last day only; weighing him at the
    # cutoff instead would have made a regular look like a bench bat.
    assert share[INJURED] == pytest.approx(fraction * share[101])
    assert share[INJURED] > 0.2 * share[101]


def test_a_sixty_day_list_late_in_the_season_projects_to_zero_through_the_model():
    placed_on = "2026-07-12"
    late = ilr.expected_active_fractions(
        make_roster().assign(status_code=lambda d: d["status_code"].replace("D10", "D60")),
        ilr.parse_events(pd.DataFrame([tx(
            INJURED, placed_on, "SC",
            "Reds placed CF Ed on the 60-day injured list.")])),
        _fixed_stint_table("IL60", days=120), CUTOFF, HORIZON_DAYS)
    assert late["active_fraction"].iloc[0] == 0.0
    share = _project(placed_on, fractions=late)
    assert share[INJURED] == 0.0
    assert share[101] == pytest.approx(1 / 9)


def test_a_transaction_on_or_after_the_cutoff_cannot_move_the_projection():
    placed_on = "2026-07-12"
    base = _project(placed_on, transactions=_placement(placed_on))
    future = [
        # He is activated on the cutoff morning and plays out the season —
        # true, and unknowable at the cutoff.
        tx(INJURED, CUTOFF, "SC",
           "Reds activated CF Ed from the 10-day injured list."),
        tx(INJURED, "2026-08-20", "SC",
           "Reds placed CF Ed on the 60-day injured list."),
        # ... and a regular gets hurt three weeks later.
        tx(101, "2026-08-22", "SC", "Reds placed CF Al on the 10-day injured list."),
    ]
    after = _project(placed_on, transactions=_placement(placed_on) + future)
    pd.testing.assert_series_equal(base, after)


def test_blend_il_without_any_fractions_is_exactly_blend():
    placed_on = "2026-07-12"
    pd.testing.assert_series_equal(_project(placed_on, method="blend_il"),
                                   _project(placed_on, method="blend"))
    # And with them it is not — the whole point.
    with_returns = _project(placed_on, transactions=_placement(placed_on))
    assert with_returns[INJURED] > 0
    assert _project(placed_on, method="blend")[INJURED] == 0.0


def test_an_undated_spell_keeps_the_hard_zero():
    # Nothing in the transaction feed says when he went on the list, so the
    # projection has no elapsed time to condition on and declines to guess.
    fractions = ilr.expected_active_fractions(
        make_roster(), ilr.parse_events(pd.DataFrame([])), _fixed_stint_table(),
        CUTOFF, HORIZON_DAYS)
    assert fractions.empty
    assert _project("2026-07-12", fractions=fractions)[INJURED] == 0.0
