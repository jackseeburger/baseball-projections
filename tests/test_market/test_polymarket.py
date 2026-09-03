"""Polymarket normalization on real (trimmed) Gamma events — no network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import polymarket
from src.market import teams as T
from src.market.schema import validate

FIXTURE = Path(__file__).parent.parent / "fixtures/market/polymarket_events.json"
TS = "2026-09-02T18:00:00+00:00"


@pytest.fixture(scope="module")
def events():
    return json.load(open(FIXTURE))


@pytest.fixture(scope="module")
def by_event(events):
    return {e["slug"]: polymarket.normalize_event(e, TS) for e in events}


def test_all_records_pass_schema(by_event):
    for recs in by_event.values():
        assert recs
        for r in recs:
            validate(r)


def test_postponed_game_uses_start_time_not_slug(by_event):
    recs = by_event["mlb-tb-nyy-2026-05-23"]
    ml = next(r for r in recs if r["market_type"] == "moneyline")
    assert ml["game_date"] == "2026-09-22"          # slug says May 23
    assert ml["game_start"] == "2026-09-22T17:05:00+00:00"
    assert ml["home_id"] == T.ABBREV_TO_ID["NYY"] and ml["away_id"] == T.ABBREV_TO_ID["TB"]
    assert ml["outcome"] == "Tampa Bay Rays" and ml["team_abbrev"] == "TB"
    assert 0 < ml["mid"] < 1


def test_totals_and_spreads_carry_lines(by_event):
    recs = by_event["mlb-tb-nyy-2026-05-23"]
    tot = next(r for r in recs if r["market_type"] == "total")
    assert tot["outcome"] == "Over" and tot["line"] and tot["team_id"] is None
    spr = next(r for r in recs if r["market_type"] == "spread")
    assert spr["line"] < 0 and spr["team_id"] is not None


def test_first_five_type(by_event):
    recs = by_event["mlb-det-cle-2026-06-14-first-five-winner"]
    assert {r["market_type"] for r in recs} <= {"first5_moneyline", "first5_total", "first5_spread"}
    assert all(r["game_date"] == "2026-09-04" for r in recs)


def test_futures_from_question_text(by_event):
    recs = by_event["mlb-world-series-champion-2026"]
    assert all(r["market_type"] == "futures_ws" for r in recs)
    assert all(r["team_id"] is not None for r in recs)
    assert all(r["game_date"] is None and r["home_id"] is None for r in recs)
    assert all(r["season"] == 2026 for r in recs)


def test_closed_market_has_result(by_event):
    recs = by_event["mlb-stl-lad-2026-09-01"]
    ml = recs[0]
    assert ml["status"] == "closed" and ml["result"] in ("yes", "no")


def test_late_game_date_is_eastern(by_event):
    # 02:10Z on Sep 2 is a Sep 1 game in ET.
    recs = by_event["mlb-stl-lad-2026-09-01"]
    assert recs[0]["game_date"] == "2026-09-01"


def test_event_start_time_beats_stale_market_game_start():
    event = {"slug": "mlb-det-cle-2026-06-14-player-props", "startTime": "2026-09-04T18:10:00Z",
             "teams": [{"name": "Detroit Tigers", "abbreviation": "det", "ordering": "away"},
                       {"name": "Cleveland Guardians", "abbreviation": "cle", "ordering": "home"}],
             "markets": [{"id": "1", "question": "Riley Greene 1+ HR?", "outcomes": '["Yes","No"]',
                          "outcomePrices": '["0.2","0.8"]', "sportsMarketType": "baseball_player_home_runs",
                          "gameStartTime": "2026-06-14 17:40:00+00", "active": True}]}
    r = polymarket.normalize_event(event, TS)[0]
    assert r["market_type"] == "prop_hr"
    assert r["game_date"] == "2026-09-04" and r["team_id"] is None


def test_team_id_from_text_prefers_longest_name():
    assert T.team_id_from_text("Will the New York Mets win?") == T.ABBREV_TO_ID["NYM"]
    assert T.team_id_from_text("Will the Athletics win the 2026 World Series?") == 133
    assert T.team_id_from_text("Will Shohei Ohtani hit 60 HR?") is None
