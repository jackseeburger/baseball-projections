"""Sportsbook normalization and de-vig math — no network."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import devig, oddsapi
from src.market import teams as T
from src.market.schema import validate

FIXTURE = Path(__file__).parent.parent / "fixtures/market/oddsapi_events.json"
TS = "2026-09-02T20:00:00+00:00"


@pytest.fixture(scope="module")
def events():
    return json.load(open(FIXTURE))


@pytest.fixture(scope="module")
def records(events):
    return oddsapi.normalize(events, TS)


class TestDevig:
    def test_multiplicative_two_way_sums_to_one(self):
        fair = devig.multiplicative([1.53, 2.55])
        assert sum(fair) == pytest.approx(1.0)
        assert fair[0] > fair[1]

    def test_power_two_way_sums_to_one_and_shades_longshot_more(self):
        odds = [1.08, 7.3]                      # heavy favourite
        mult, pw = devig.multiplicative(odds), devig.power(odds)
        assert sum(pw) == pytest.approx(1.0)
        # Power method gives the favourite a larger share than multiplicative.
        assert pw[0] > mult[0]

    def test_fair_book_is_unchanged(self):
        assert devig.power([2.0, 2.0]) == pytest.approx([0.5, 0.5])
        assert devig.overround([2.0, 2.0]) == pytest.approx(1.0)

    def test_bad_odds_rejected(self):
        with pytest.raises(ValueError):
            devig.implied(1.0)


class TestNormalize:
    def test_schema(self, records):
        assert records
        for r in records:
            validate(r)
            assert r["venue"] == "oddsapi" and r["book"]

    def test_moneyline_pair_devigs_within_book(self, records):
        by_book = {}
        for r in records:
            if r["market_type"] == "moneyline":
                by_book.setdefault((r["event_id"], r["book"]), []).append(r)
        assert by_book
        for pair in by_book.values():
            assert len(pair) == 2
            assert sum(r["mid"] for r in pair) == pytest.approx(1.0, abs=1e-3)
            assert sum(r["last"] for r in pair) > 1.0          # raw implied carries the vig
            assert {r["team_id"] for r in pair} == {pair[0]["home_id"], pair[0]["away_id"]}

    def test_totals_carry_line_and_no_team(self, records):
        tot = [r for r in records if r["market_type"] == "total"]
        assert tot and all(r["line"] and r["team_id"] is None for r in tot)
        assert {r["outcome"] for r in tot} == {"Over", "Under"}

    def test_live_flag_uses_snapshot_time(self, events):
        e = events[0]
        before = oddsapi.normalize_event(e, "2026-01-01T00:00:00+00:00")
        after = oddsapi.normalize_event(e, "2027-01-01T00:00:00+00:00")
        assert {r["status"] for r in before} == {"pregame"}
        assert {r["status"] for r in after} == {"live"}

    def test_pinnacle_present_and_teams_resolve(self, records):
        assert any(r["book"] == "pinnacle" for r in records)
        assert all(r["home_id"] and r["away_id"] for r in records)
        assert all(r["game_date"] for r in records)
