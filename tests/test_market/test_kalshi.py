"""Kalshi normalization on real (trimmed) market payloads — no network."""
import json
import sys
from itertools import permutations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import kalshi
from src.market import teams as T
from src.market.schema import validate

FIXTURE = Path(__file__).parent.parent / "fixtures/market/kalshi_markets.json"
TS = "2026-09-02T18:00:00+00:00"


@pytest.fixture(scope="module")
def markets():
    return json.load(open(FIXTURE))


@pytest.fixture(scope="module")
def records(markets):
    return {r["market_id"]: r for r in (kalshi.normalize(m, TS) for m in markets)}


def test_all_records_pass_schema(records):
    for r in records.values():
        validate(r)


def test_game_event_ticker_decodes_time_and_teams():
    ev = kalshi.parse_event("KXMLBGAME-26SEP042210WSHLAD")
    assert ev["season"] == 2026
    assert ev["game_date"] == "2026-09-04"
    assert ev["game_start"] == "2026-09-05T02:10:00+00:00"   # 22:10 EDT → 02:10Z
    assert (ev["away_abbrev"], ev["home_abbrev"]) == ("WSH", "LAD")


def test_doubleheader_suffix_is_stripped_from_pair():
    ev = kalshi.parse_event("KXMLBGAME-26SEP041915DETCLEG2")
    assert (ev["away_abbrev"], ev["home_abbrev"]) == ("DET", "CLE")
    assert ev["game_number"] == 2 and ev["game_start"] == "2026-09-04T23:15:00+00:00"
    assert kalshi.parse_event("KXMLBGAME-26SEP042210WSHLAD")["game_number"] is None


def test_futures_event_ticker():
    ev = kalshi.parse_event("KXMLBALEAST-26")
    assert ev == {"series": "KXMLBALEAST", "season": 2026}


def test_every_ordered_team_pair_splits_uniquely():
    """The ticker glues two abbreviations with no separator; the split must
    be unambiguous for every possible matchup or game mapping breaks."""
    for a, b in permutations(T.KALSHI_ABBREV_TO_ID, 2):
        assert T.split_kalshi_pair(a + b) == (a, b), a + b


def test_moneyline_pair_prices_are_consistent(records):
    wsh = records["KXMLBGAME-26SEP042210WSHLAD-WSH"]
    lad = records["KXMLBGAME-26SEP042210WSHLAD-LAD"]
    assert wsh["market_type"] == lad["market_type"] == "moneyline"
    assert wsh["team_abbrev"] == "WSH" and lad["team_abbrev"] == "LAD"
    assert wsh["home_id"] == lad["home_id"] == T.ABBREV_TO_ID["LAD"]
    assert wsh["away_id"] == T.ABBREV_TO_ID["WSH"]
    # Two YES markets on one game: their mids sum to ≈ 1 (within the spread).
    if wsh["mid"] is not None and lad["mid"] is not None:
        assert abs(wsh["mid"] + lad["mid"] - 1) < 0.05


def test_total_and_spread_lines(records):
    tot = records["KXMLBTOTAL-26SEP032210STLLAD-9"]
    assert tot["market_type"] == "total" and tot["outcome"] == "Over"
    assert tot["line"] == 8.5 and tot["team_id"] is None
    spr = records["KXMLBSPREAD-26SEP032210STLLAD-STL4"]
    assert spr["market_type"] == "spread" and spr["team_abbrev"] == "STL"
    assert spr["line"] == -3.5


def test_first5_tie_has_no_team(records):
    tie = records["KXMLBF5-26SEP032210STLLAD-TIE"]
    assert tie["market_type"] == "first5_moneyline"
    assert tie["outcome"] == "Tie" and tie["team_id"] is None


def test_futures_markets_map_to_team(records):
    ws = records["KXMLB-26-WSH"]
    assert ws["market_type"] == "futures_ws" and ws["team_abbrev"] == "WSH"
    assert ws["game_date"] is None and ws["home_id"] is None
    div = records["KXMLBALEAST-26-TOR"]
    assert div["market_type"] == "futures_division" and div["team_abbrev"] == "TOR"


def test_settled_market_carries_result(records):
    settled = [r for r in records.values() if r["status"] == "finalized" or r["result"]]
    assert settled, "fixture should include settled markets"
    assert {r["result"] for r in settled} <= {"yes", "no"}


def test_zero_quotes_become_none():
    m = {"ticker": "KXMLBGAME-26SEP042210WSHLAD-WSH", "event_ticker": "KXMLBGAME-26SEP042210WSHLAD",
         "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.2600", "last_price_dollars": "0.2500",
         "yes_sub_title": "Washington", "status": "active"}
    r = kalshi.normalize(m, TS)
    assert r["bid"] is None and r["ask"] == 0.26 and r["mid"] is None
