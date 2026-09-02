"""The posted lineup must be the nine a club started, not the nine it ended with.

`liveData.boxscore.teams.{side}.battingOrder` is always nine long but holds the
*last* occupant of each slot, so about one entry per team per game is a pinch
hitter or defensive replacement. Who pinch-hit is a fact about how the game
went, so reading that array leaks the result backwards into a walk-forward
backtest (measured on 2025: the ending lineup's distance from a club's own norm
correlates -0.06 with the runs that club scored — the wrong sign).
`posted_lineup` reads the per-player codes instead: "300" is the number-three
hitter who started, "301" the man who took over that slot.
"""
import pytest

from src.data.mlb_stats_api import posted_lineup


def box(codes: dict, with_person: bool = True) -> dict:
    """A boxscore team block: {player id: battingOrder code}."""
    players = {}
    for pid, code in codes.items():
        entry = {"battingOrder": code} if code else {}
        if with_person:
            entry["person"] = {"id": pid}
        players[f"ID{pid}"] = entry
    return {"players": players}


STARTERS = {100 + i: f"{i}00" for i in range(1, 10)}


def test_the_nine_starters_come_back_in_batting_order():
    assert posted_lineup(box(STARTERS)) == [101, 102, 103, 104, 105, 106, 107,
                                            108, 109]


def test_players_who_never_batted_are_ignored():
    codes = dict(STARTERS)
    codes.update({500: None, 501: None})       # relievers, unused bench
    assert posted_lineup(box(codes)) == list(range(101, 110))


def test_a_pinch_hitter_does_not_displace_the_starter():
    """The whole point: `301` batted third, but `103` was the posted three."""
    codes = dict(STARTERS)
    codes[301] = "301"                          # pinch hit for the 3 hole
    codes[702] = "702"                          # second man through the 7 hole
    assert posted_lineup(box(codes)) == list(range(101, 110))


def test_ids_are_recovered_from_the_player_key_when_person_is_trimmed():
    """The cached feed is fetched with a `fields` filter that empties
    `person`, so the id has to come off the "ID<id>" key."""
    assert posted_lineup(box(STARTERS, with_person=False)) == list(range(101, 110))


def test_an_incomplete_or_missing_order_yields_nothing():
    assert posted_lineup({}) == []
    assert posted_lineup({"players": {}}) == []
    short = {k: v for k, v in list(STARTERS.items())[:8]}
    assert posted_lineup(box(short)) == []


def test_a_malformed_code_is_skipped_rather_than_crashing():
    codes = dict(STARTERS)
    codes[900] = "notanumber00"
    assert posted_lineup(box(codes)) == list(range(101, 110))


def test_slots_are_ordered_numerically_not_lexically():
    """Codes are strings; "1000" must never sort before "900"."""
    order = posted_lineup(box(STARTERS))
    assert order == sorted(order, key=lambda p: int(STARTERS[p][:-2]))
    assert order[0] == 101 and order[-1] == 109


@pytest.mark.parametrize("code", ["100", "900"])
def test_only_codes_ending_in_double_zero_are_starters(code):
    single = {101: code}
    assert posted_lineup(box(single)) == []      # nine required
    assert posted_lineup(box({101: code[:-2] + "01"})) == []
