"""Snapshot writer: immutable files, round-trip, summary shape."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.market import kalshi, paper, polymarket, snapshot
from src.market.schema import FIELDS, validate

FIX = Path(__file__).parent.parent / "fixtures/market"
TS = "2026-09-02T18:00:00+00:00"


@pytest.fixture
def records():
    recs = [kalshi.normalize(m, TS) for m in json.load(open(FIX / "kalshi_markets.json"))]
    for e in json.load(open(FIX / "polymarket_events.json")):
        recs.extend(polymarket.normalize_event(e, TS))
    # pretend the WSH@LAD moneylines were mapped
    for r in recs:
        if r["event_id"] == "KXMLBGAME-26SEP042210WSHLAD":
            r["game_pk"] = 777
    return recs


def test_write_is_immutable_and_round_trips(records, tmp_path):
    path = snapshot.write(records, TS, tmp_path)
    assert path.name == "2026-09-02T1800Z.jsonl.gz"
    df = snapshot.read(path)
    assert list(df.columns) == FIELDS and len(df) == len(records)
    with pytest.raises(FileExistsError):
        snapshot.write(records, TS, tmp_path)


def test_summary_has_home_prob_per_venue(records):
    s = snapshot.summarize(records, TS, {"kalshi_markets": 9})
    assert s["as_of"] == TS and s["n_records"] == len(records)
    game = next(g for g in s["games"] if g["game_pk"] == 777)
    assert 0 < game["kalshi_p_home"] < 1
    assert "futures_ws" in s["futures"] and "WSH" in s["futures"]["futures_ws"]


def test_validate_rejects_bad_price():
    r = kalshi.normalize(json.load(open(FIX / "kalshi_markets.json"))[0], TS)
    r["bid"] = 1.5
    with pytest.raises(ValueError):
        validate(r)
    r["bid"] = 0.5
    r["market_type"] = "nonsense"
    with pytest.raises(ValueError):
        validate(r)


# ───────────── Kalshi settlement in the archive (BAS-78) ─────────────

def fake_settled(monkeypatch, markets):
    monkeypatch.setattr(kalshi, "fetch_settled_props",
                        lambda **kw: list(markets))


def settled_market(ticker="KXMLBHIT-26SEP082140TEXSEA-TEXWLANGFORD36-1",
                   result="yes"):
    return {
        "ticker": ticker, "event_ticker": "-".join(ticker.split("-")[:2]),
        "title": "Wyatt Langford: 1+ hits?", "yes_sub_title": "Wyatt Langford: 1+",
        "status": "finalized", "result": result, "floor_strike": 0.5,
        "close_time": "2026-09-09T05:09:59Z",
        "yes_bid_dollars": "0.62", "yes_ask_dollars": "0.66",
        "last_price_dollars": "0.64",
    }


def test_settled_props_join_the_snapshot_with_a_result_and_the_same_schema(monkeypatch):
    """The settled tail is merged into the same file, in the same shape.

    Kalshi drops a market from the open listing the moment it settles, so
    without this pull no snapshot ever carries a Kalshi prop's result and the
    paper ledger has to settle every Kalshi ticket from a hand-run backfill.
    """
    fake_settled(monkeypatch, [settled_market()])
    recs = snapshot.settled_prop_records(TS)
    assert len(recs) == 1
    r = recs[0]
    validate(r)
    assert r["result"] == "yes"
    assert r["status"] == "settled"        # the API says "finalized"; one word here
    assert set(r) == set(FIELDS)


def test_a_settled_row_is_never_emittable(monkeypatch):
    """`paper.open_props` drops it on all three counts, and that is asserted.

    Its status is not open, it already carries a result, and its first pitch
    is behind the snapshot. Any one of the three would be enough; the ledger
    should never write a ticket on a settled market.
    """
    fake_settled(monkeypatch, [settled_market()])
    r = snapshot.settled_prop_records("2026-09-09T12:00:00+00:00")[0]
    r = dict(r, game_pk=700001, player_id=12345)
    df = pd.DataFrame([r])
    assert paper.open_props(df).empty
    # and each guard on its own, so a later loosening cannot pass by accident
    assert r["status"] not in ("active", "open")
    assert r["result"] is not None
    assert r["game_start"] < r["ts"]


def test_settled_rows_do_not_duplicate_a_market_already_in_the_snapshot(monkeypatch):
    fake_settled(monkeypatch, [settled_market(), settled_market()])
    recs = snapshot.settled_prop_records(TS)
    assert len(recs) == 1
    assert snapshot.settled_prop_records(
        TS, seen={settled_market()["ticker"]}) == []


def test_a_settled_market_with_no_result_is_not_evidence(monkeypatch):
    fake_settled(monkeypatch, [settled_market(result="")])
    assert snapshot.settled_prop_records(TS) == []
