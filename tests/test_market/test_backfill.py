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


# ─────────────────── the candlestick archive (maker exam) ───────────────────

def ohlc(end_ts, low, high, open_=None, close=None, bid=None, ask=None, vol=10.0):
    """A Kalshi hourly candle with the full price node."""
    return {"end_period_ts": end_ts,
            "price": {"open_dollars": f"{open_ if open_ is not None else low:.4f}",
                      "high_dollars": f"{high:.4f}", "low_dollars": f"{low:.4f}",
                      "close_dollars": f"{close if close is not None else high:.4f}"},
            "yes_bid": {"close_dollars": f"{bid if bid is not None else low:.4f}"},
            "yes_ask": {"close_dollars": f"{ask if ask is not None else high:.4f}"},
            "volume_fp": f"{vol}"}


def test_candle_rows_flatten_the_nested_payload_in_time_order():
    rows = backfill.candle_rows(
        [ohlc(FIRST_PITCH - 3600, .50, .55), ohlc(FIRST_PITCH - 7200, .40, .45)],
        "KXMLBGAME-X-LAD", 42)
    assert [r["end_period_ts"] for r in rows] == [FIRST_PITCH - 7200, FIRST_PITCH - 3600]
    assert list(rows[0]) == backfill.CANDLE_COLUMNS
    assert rows[0]["price_low"] == .40 and rows[0]["price_high"] == .45
    assert rows[0]["market_id"] == "KXMLBGAME-X-LAD" and rows[0]["game_pk"] == 42
    assert rows[0]["volume"] == 10.0


def test_candle_rows_survive_a_missing_price_node():
    # An hour with no trades can come back without a price block at all.
    c = {"end_period_ts": FIRST_PITCH - 3600, "yes_bid": {"close_dollars": "0.5000"},
         "yes_ask": {"close_dollars": "0.5100"}}
    r = backfill.candle_rows([c], "M", 1)[0]
    assert r["price_low"] is None and r["price_close"] is None
    assert r["yes_bid_close"] == .50 and r["volume"] == 0.0


def test_candles_for_a_market_ask_for_the_pregame_window_only(monkeypatch):
    asked = {}

    def fake(series, ticker, start, end, period_minutes=60, session=None):
        asked.update(series=series, ticker=ticker, start=start, end=end,
                     period=period_minutes)
        return [ohlc(FIRST_PITCH - 3600, .50, .55),
                ohlc(FIRST_PITCH + 3600, .90, .95)]      # in-game, must be dropped

    monkeypatch.setattr(backfill.kalshi, "fetch_candlesticks", fake)
    rows = backfill.kalshi_candles_for_market("M", 7, FIRST_PITCH)
    assert asked["start"] == FIRST_PITCH - 24 * 3600 and asked["end"] == FIRST_PITCH
    assert asked["period"] == 60 and asked["series"] == "KXMLBGAME"
    assert [r["end_period_ts"] for r in rows] == [FIRST_PITCH - 3600]


def test_candle_archive_skips_a_failing_market_and_reports_it(monkeypatch):
    closes = pd.DataFrame([
        {"venue": "kalshi", "market_id": "GOOD", "game_pk": 1,
         "game_start": "2027-01-15T20:00:00+00:00"},
        {"venue": "kalshi", "market_id": "BAD", "game_pk": 2,
         "game_start": "2027-01-15T20:00:00+00:00"},
        {"venue": "polymarket", "market_id": "PM", "game_pk": 3,
         "game_start": "2027-01-15T20:00:00+00:00"},
    ])

    def fake(series, ticker, start, end, period_minutes=60, session=None):
        if ticker == "BAD":
            raise RuntimeError("429 forever")
        return [ohlc(end - 3600, .50, .55)]

    monkeypatch.setattr(backfill.kalshi, "fetch_candlesticks", fake)
    rows, failures = backfill.kalshi_candle_archive(closes, pace_seconds=0)
    assert failures == ["BAD"]                       # one bad market, not a dead run
    assert {r["market_id"] for r in rows} == {"GOOD"}   # and no Polymarket rows


def test_candle_archive_resumes_and_checkpoints(monkeypatch):
    closes = pd.DataFrame([
        {"venue": "kalshi", "market_id": f"M{i}", "game_pk": i,
         "game_start": "2027-01-15T20:00:00+00:00"} for i in range(3)])
    monkeypatch.setattr(backfill.kalshi, "fetch_candlesticks",
                        lambda *a, **k: [ohlc(a[3] - 3600, .5, .55)])
    seen = []
    rows, failures = backfill.kalshi_candle_archive(
        closes, pace_seconds=0, skip_markets={"M0"},
        on_market=lambda mid, got: seen.append(mid))
    assert seen == ["M1", "M2"] and not failures      # M0 was already archived
    assert len(rows) == 2


def test_candle_frame_dedupes_and_sorts():
    rows = backfill.candle_rows([ohlc(20, .5, .55), ohlc(10, .4, .45)], "M", 2) \
        + backfill.candle_rows([ohlc(10, .4, .45)], "M", 2) \
        + backfill.candle_rows([ohlc(15, .3, .35)], "N", 1)
    df = backfill.candle_frame(rows)
    assert list(df.columns) == backfill.CANDLE_COLUMNS
    assert len(df) == 3                                   # the repeat is one row
    assert df["game_pk"].tolist() == [1, 2, 2]
    assert backfill.candle_frame([]).empty
