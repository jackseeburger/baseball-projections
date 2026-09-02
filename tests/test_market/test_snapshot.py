"""Snapshot writer: immutable files, round-trip, summary shape."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import kalshi, polymarket, snapshot
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
