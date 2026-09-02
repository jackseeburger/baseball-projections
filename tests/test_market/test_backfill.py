"""Pre-game close reconstruction — pure logic, no network."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import backfill
from src.market import teams as T

LAD, WSH, NYY, TB = 119, 120, 147, 139
FIRST_PITCH = 1_800_000_000


def candle(end_ts, price, bid, ask, vol=10.0):
    return {"end_period_ts": end_ts,
            "price": {"close_dollars": f"{price:.4f}"},
            "yes_bid": {"close_dollars": f"{bid:.4f}"},
            "yes_ask": {"close_dollars": f"{ask:.4f}"},
            "volume_fp": f"{vol}"}


def test_last_before_picks_latest_pre_pitch_observation():
    candles = [candle(FIRST_PITCH - 7200, .50, .49, .51),
               candle(FIRST_PITCH - 3600, .55, .54, .56),
               candle(FIRST_PITCH + 3600, .90, .89, .91)]   # in-game, must be ignored
    last = backfill.last_before(candles, FIRST_PITCH, key="end_period_ts")
    assert last["price"]["close_dollars"] == "0.5500"
    assert last["_n_pre"] == 2 and last["_volume_pre"] == 20.0


def test_last_before_returns_none_without_pre_pitch_data():
    assert backfill.last_before([candle(FIRST_PITCH + 60, .9, .9, .9)], FIRST_PITCH, "end_period_ts") is None


def test_kalshi_closes_use_home_market_only(monkeypatch):
    markets = [
        {"ticker": "KXMLBGAME-26SEP042210WSHLAD-WSH", "event_ticker": "KXMLBGAME-26SEP042210WSHLAD",
         "open_time": "2026-09-02T00:00:00Z", "close_time": "2026-09-05T06:00:00Z", "result": "no"},
        {"ticker": "KXMLBGAME-26SEP042210WSHLAD-LAD", "event_ticker": "KXMLBGAME-26SEP042210WSHLAD",
         "open_time": "2026-09-02T00:00:00Z", "close_time": "2026-09-05T06:00:00Z", "result": "yes"},
    ]
    fp = backfill._ts("2026-09-05T02:10:00+00:00")
    asked = []

    def fake_candles(series, ticker, start, end, period_minutes=60, session=None):
        asked.append(ticker)
        return [candle(fp - 3600, .72, .71, .73)]

    monkeypatch.setattr(backfill.kalshi, "fetch_candlesticks", fake_candles)
    rows = backfill.kalshi_closes(2026, markets=markets)
    assert asked == ["KXMLBGAME-26SEP042210WSHLAD-LAD"], "only the home YES market is priced"
    r = rows[0]
    assert r["home_id"] == LAD and r["away_id"] == WSH
    assert r["p_home_close"] == .72 and r["home_won"] is True
    assert r["minutes_before_pitch"] == 60.0


def test_polymarket_close_is_oriented_to_home(monkeypatch):
    event = {
        "slug": "mlb-tb-nyy-2026-07-10", "startTime": "2026-07-10T23:05:00Z",
        "teams": [{"name": "Tampa Bay Rays", "abbreviation": "tb", "ordering": "away"},
                  {"name": "New York Yankees", "abbreviation": "nyy", "ordering": "home"}],
        "markets": [{"id": "1", "sportsMarketType": "moneyline",
                     "outcomes": '["Tampa Bay Rays","New York Yankees"]',
                     "outcomePrices": '["0","1"]',            # Yankees won
                     "clobTokenIds": '["tokA","tokB"]'}],
    }
    fp = backfill._ts(event["startTime"])
    monkeypatch.setattr(backfill, "polymarket_price_history",
                        lambda tok, s, e, session=None: [{"t": fp - 1800, "p": 0.41},
                                                         {"t": fp + 600, "p": 0.05}])
    r = backfill.polymarket_closes(2026, events=[event])[0]
    # First outcome is the away team at .41 → P(home) = .59
    assert r["p_home_close"] == 0.59
    assert r["home_id"] == NYY and r["away_id"] == TB
    assert r["home_won"] is True and r["minutes_before_pitch"] == 30.0


def test_to_frame_maps_game_pk_and_drops_unmapped():
    schedule = pd.DataFrame([{"game_pk": 42, "date": "2026-09-04", "home_id": LAD, "away_id": WSH,
                              "game_datetime": "2026-09-05T02:10:00Z"}])
    rows = [
        {"venue": "kalshi", "game_date": "2026-09-04", "game_start": "2026-09-05T02:10:00+00:00",
         "home_id": LAD, "away_id": WSH, "p_home_close": .7, "bid": .69, "ask": .71,
         "close_ts": 1, "minutes_before_pitch": 60.0, "volume_pre": 5.0, "n_obs": 3,
         "market_id": "m1", "home_won": True},
        {"venue": "kalshi", "game_date": "2026-09-09", "game_start": "2026-09-09T23:00:00+00:00",
         "home_id": NYY, "away_id": TB, "p_home_close": .5, "bid": .49, "ask": .51,
         "close_ts": 1, "minutes_before_pitch": 60.0, "volume_pre": 5.0, "n_obs": 3,
         "market_id": "m2", "home_won": False},
    ]
    df = backfill.to_frame(rows, schedule)
    assert list(df.columns) == backfill.CLOSE_COLUMNS
    assert df["game_pk"].tolist() == [42]
