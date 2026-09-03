"""Pre-game closing prices for settled games — the market benchmark for E.

For an exchange the "closing line" is the last price before first pitch.
Both venues keep enough history to reconstruct it after the fact:

    Kalshi      hourly candlesticks per market (price / yes_bid / yes_ask)
    Polymarket  CLOB prices-history per outcome token, hourly with an
                explicit [startTs, endTs] window (the interval=max form
                returns nothing for closed markets)

Coverage limits found 2026-09-02: Kalshi's KXMLBGAME series has settled
markets from 2026-06-22; the default listing only shows ~1 month, so we
page by close-time windows. Polymarket's closed-event listing caps at
offset 2100, which newest-first reaches back to early July.

One row per (venue, game): P(home) at the close, the quote around it, and
how long before first pitch the last observation was.

**Player props** (`kalshi_prop_closes`) work the same way one level down: one
row per settled *contract* rather than per game, because a prop is a player at
a line and a game carries dozens of them. Kalshi's prop series start
2026-06-27 and list every hitter at every strike whether or not anyone traded
it, so the listing pass keeps only contracts with volume and the candlestick
pass — one request each, no bulk endpoint — runs on a small thread pool.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.market import kalshi, polymarket
from src.market import teams as T
from src.market.games import assign_game_pk
from src.market.http import get_json
from src.market.schema import empty_record

logger = logging.getLogger(__name__)

CLOSE_COLUMNS = [
    "venue", "game_pk", "game_date", "game_start", "home_id", "away_id",
    "p_home_close", "bid", "ask", "close_ts", "minutes_before_pitch",
    "volume_pre", "n_obs", "market_id", "home_won",
]


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


# ───────────────────────────── Kalshi ─────────────────────────────

def kalshi_settled(series_ticker: str, season: int, session=None,
                   window_days: int = 7, start: str | None = None,
                   end: str | None = None, follow_cursor: bool = False) -> list[dict]:
    """Every settled market in a series, paged by close-time window.

    The default listing only reaches back about a month, so the window is the
    only way to walk a season. `follow_cursor` also pages *within* a window,
    which the moneyline series never needs (30 games a day) and the prop series
    always do (several thousand).
    """
    lo = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start \
        else datetime(season, 3, 15, tzinfo=timezone.utc)
    hi = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end \
        else datetime.now(timezone.utc) + timedelta(days=1)
    out: list[dict] = []
    cursor = lo
    while cursor < hi:
        nxt = min(cursor + timedelta(days=window_days), hi)
        page_cursor = None
        while True:
            params = {"series_ticker": series_ticker, "status": "settled", "limit": 1000,
                      "min_close_ts": int(cursor.timestamp()),
                      "max_close_ts": int(nxt.timestamp())}
            if page_cursor:
                params["cursor"] = page_cursor
            page = get_json(f"{kalshi.BASE}/markets", params, session=session)
            markets = page.get("markets", [])
            out.extend(markets)
            page_cursor = page.get("cursor")
            if not follow_cursor:
                if page_cursor:
                    logger.warning("Kalshi %s window %s..%s had a cursor; shrink window_days",
                                   series_ticker, cursor.date(), nxt.date())
                break
            if not page_cursor or not markets:
                break
        cursor = nxt
    return out


def kalshi_settled_games(season: int, session=None, window_days: int = 7) -> list[dict]:
    """Every settled KXMLBGAME market for a season, paged by close-time window."""
    return kalshi_settled("KXMLBGAME", season, session=session, window_days=window_days)


def kalshi_close_for_market(market: dict, first_pitch_ts: int, session=None) -> dict | None:
    """Last hourly candle ending at or before first pitch."""
    candles = kalshi.fetch_candlesticks(
        "KXMLBGAME", market["ticker"], _ts(market["open_time"]), _ts(market["close_time"]),
        period_minutes=60, session=session,
    )
    return last_before(candles, first_pitch_ts, key="end_period_ts")


def last_before(points: list[dict], cutoff_ts: int, key: str) -> dict | None:
    pre = [p for p in points if int(p[key]) <= cutoff_ts]
    if not pre:
        return None
    pre.sort(key=lambda p: int(p[key]))
    last = dict(pre[-1])
    last["_n_pre"] = len(pre)
    last["_volume_pre"] = sum(float(p.get("volume_fp") or 0) for p in pre)
    return last


def kalshi_closes(season: int, session=None, markets: list[dict] | None = None,
                  pace_seconds: float = 0.25) -> list[dict]:
    """One row per game: the home team's YES market at the close."""
    markets = markets if markets is not None else kalshi_settled_games(season, session)
    rows, seen = [], set()
    for m in markets:
        ev = kalshi.parse_event(m.get("event_ticker", ""))
        home = ev.get("home_abbrev")
        if not home or m["ticker"].rsplit("-", 1)[-1] != home or m["event_ticker"] in seen:
            continue
        seen.add(m["event_ticker"])
        fp = _ts(ev["game_start"])
        time.sleep(pace_seconds)      # Kalshi rate-limits bursts of candlestick calls
        try:
            c = kalshi_close_for_market(m, fp, session)
        except Exception as exc:  # one bad market must not sink the season
            logger.warning("candlesticks failed for %s: %s", m["ticker"], exc)
            continue
        if c is None:
            continue
        rows.append({
            "venue": "kalshi",
            "game_date": ev["game_date"], "game_start": ev["game_start"],
            "home_id": T.KALSHI_ABBREV_TO_ID.get(home),
            "away_id": T.KALSHI_ABBREV_TO_ID.get(ev.get("away_abbrev")),
            "p_home_close": float(c["price"]["close_dollars"]),
            "bid": float(c["yes_bid"]["close_dollars"]),
            "ask": float(c["yes_ask"]["close_dollars"]),
            "close_ts": int(c["end_period_ts"]),
            "minutes_before_pitch": round((fp - int(c["end_period_ts"])) / 60, 1),
            "volume_pre": c["_volume_pre"], "n_obs": c["_n_pre"],
            "market_id": m["ticker"],
            "home_won": {"yes": True, "no": False}.get(m.get("result")),
        })
    return rows


# ──────────────────────── Kalshi player props ────────────────────────

PROP_CLOSE_COLUMNS = [
    "venue", "game_pk", "game_date", "game_start", "team_id",
    "player_id", "player_name", "prop_stat", "prop_line",
    "p_over_close", "bid", "ask", "last_trade", "close_ts", "minutes_before_pitch",
    "volume_pre", "volume_total", "n_obs", "market_id", "result", "over_hit",
]


def kalshi_settled_props(season: int, series: dict | None = None, session=None,
                         window_days: int = 1, start: str | None = None,
                         end: str | None = None, min_volume: float = 1.0) -> list[dict]:
    """Settled prop markets across every prop series, one listing pass.

    `min_volume` drops the markets that never traded. An untraded contract has
    a quote but no price anyone paid, and Kalshi lists a line for every hitter
    in the game at every strike, so most of the universe is untraded noise that
    would cost one candlestick request each to discover.
    """
    series = series if series is not None else kalshi.PROP_SERIES
    out = []
    for ticker in series:
        markets = kalshi_settled(ticker, season, session=session,
                                 window_days=window_days, start=start, end=end,
                                 follow_cursor=True)
        kept = [m for m in markets if float(m.get("volume_fp") or 0) >= min_volume]
        logger.info("%s: %d settled, %d traded", ticker, len(markets), len(kept))
        out.extend(kept)
    return out


def _first_dollars(block: dict | None, *keys) -> float | None:
    for k in keys:
        v = (block or {}).get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def candle_price(candle: dict) -> tuple:
    """(price, bid, ask, last_trade) in dollars from one candle, or Nones.

    A prop contract can go a whole hour without a trade — most of them do —
    and Kalshi then returns `price: {}` or only `previous_dollars`, the last
    trade carried forward. So the price here is **the last trade when it still
    lies inside the closing quote, and the midpoint of that quote otherwise**,
    which differs from the moneyline reconstruction (`kalshi_closes`, last
    trade full stop) for a reason: a game market trades every hour and its
    last print is current, while a prop's last print can be hours stale and
    sit right outside a book that has since moved. On one day of props the
    two rules disagree on 12% of contracts, and where they disagree the stale
    print is *outside* the book — which would hand the P&L a free edge against
    a price nobody was showing.

    A zero bid or ask means no quote on that side rather than a free option,
    so it becomes None.
    """
    last = _first_dollars(candle.get("price"), "close_dollars", "previous_dollars")
    bid = _first_dollars(candle.get("yes_bid"), "close_dollars")
    ask = _first_dollars(candle.get("yes_ask"), "close_dollars")
    if ask is not None and ask <= 0:
        ask = None
    if bid is not None and bid <= 0:
        bid = None
    if bid is not None and ask is not None:
        inside = last is not None and bid <= last <= ask
        price = last if inside else round((bid + ask) / 2, 4)
    else:
        price = last
    return price, bid, ask, last


def _prop_row(market: dict, candles: list[dict]) -> dict | None:
    """One settled prop market + its candles → a close row (None if no price)."""
    ev = kalshi.parse_event(market.get("event_ticker", ""))
    if not ev.get("game_start"):
        return None
    fp = _ts(ev["game_start"])
    c = last_before(candles, fp, key="end_period_ts")
    if c is None:
        return None
    price, bid, ask, last = candle_price(c)
    if price is None:
        return None
    stat = kalshi.PROP_SERIES.get(ev["series"])
    name, line = kalshi.parse_prop_label(
        market.get("yes_sub_title") or market.get("title"))
    floor = market.get("floor_strike")
    result = (market.get("result") or "").lower() or None
    return {
        "venue": "kalshi",
        "game_date": ev["game_date"], "game_start": ev["game_start"],
        "home_id": T.KALSHI_ABBREV_TO_ID.get(ev.get("home_abbrev")),
        "away_id": T.KALSHI_ABBREV_TO_ID.get(ev.get("away_abbrev")),
        "team_id": kalshi.prop_team_id(market["ticker"]),
        "player_name": name,
        "prop_stat": kalshi.PROP_MARKET_TYPES.get(stat) if stat else None,
        "prop_line": float(floor) if floor is not None else line,
        "p_over_close": price, "bid": bid, "ask": ask, "last_trade": last,
        "close_ts": int(c["end_period_ts"]),
        "minutes_before_pitch": round((fp - int(c["end_period_ts"])) / 60, 1),
        "volume_pre": c["_volume_pre"], "n_obs": c["_n_pre"],
        "volume_total": float(market.get("volume_fp") or 0),
        "market_id": market["ticker"], "result": result,
        "over_hit": {"yes": True, "no": False}.get(result),
    }


def kalshi_prop_closes(season: int, session=None, markets: list[dict] | None = None,
                       workers: int = 6, pace_seconds: float = 0.0,
                       **kwargs) -> list[dict]:
    """Pre-first-pitch close for every settled prop market that traded.

    One candlestick request per market — there is no bulk endpoint — so this
    is the expensive half of the archive: tens of thousands of calls. Kalshi
    429s a burst and `http.get_json` backs off, and a small thread pool holds
    the sustained rate around 10-15/s without tripping it for long.

    YES on every prop series is "at least N", i.e. **over** `prop_line`, so
    `p_over_close` needs no orientation the way a moneyline does.
    """
    markets = markets if markets is not None else kalshi_settled_props(
        season, session=session, **kwargs)
    local = threading.local()

    def fetch(m: dict) -> dict | None:
        sess = session
        if sess is None:
            if not hasattr(local, "session"):
                local.session = requests.Session()
            sess = local.session
        series = m["ticker"].split("-", 1)[0]
        try:
            candles = kalshi.fetch_candlesticks(
                series, m["ticker"], _ts(m["open_time"]), _ts(m["close_time"]),
                period_minutes=60, session=sess)
        except Exception as exc:       # one bad market must not sink the run
            logger.warning("candlesticks failed for %s: %s", m["ticker"], exc)
            return None
        if pace_seconds:
            time.sleep(pace_seconds)
        return _prop_row(m, candles)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(fetch, markets), 1):
            if row is not None:
                rows.append(row)
            if i % 2000 == 0:
                logger.info("props: %d/%d markets, %d with a pre-pitch close",
                            i, len(markets), len(rows))
    return rows


def prop_frame(rows: list[dict], schedule: pd.DataFrame, resolver=None) -> pd.DataFrame:
    """Attach `game_pk` and `player_id`, and return the PROP_CLOSE_COLUMNS frame."""
    recs = []
    for r in rows:
        rec = empty_record()
        rec.update({"ts": "backfill", "venue": r["venue"], "market_id": r["market_id"],
                    "market_type": "moneyline", "game_date": r["game_date"],
                    "game_start": r["game_start"], "home_id": r["home_id"],
                    "away_id": r["away_id"]})
        recs.append(rec)
    logger.info("game_pk mapping: %s", assign_game_pk(recs, schedule))
    for r, rec in zip(rows, recs):
        r["game_pk"] = rec["game_pk"]
        r["player_id"] = resolver.resolve(r["player_name"]) if resolver else None
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PROP_CLOSE_COLUMNS)
    df = df[df["game_pk"].notna()].copy()
    df["game_pk"] = df["game_pk"].astype(int)
    df = df.drop_duplicates(subset=["venue", "market_id"], keep="first")
    return (df[PROP_CLOSE_COLUMNS]
            .sort_values(["game_date", "game_pk", "prop_stat", "player_id", "prop_line"])
            .reset_index(drop=True))


# ─────────────────────────── Polymarket ───────────────────────────

def polymarket_closed_events(season: int, session=None, page_size: int = 100,
                             max_offset: int = 2100) -> list[dict]:
    """Closed MLB moneyline events for a season, newest first."""
    out, offset = [], 0
    while offset < max_offset:
        params = {"tag_slug": "mlb", "closed": "true", "limit": page_size, "offset": offset,
                  "order": "endDate", "ascending": "false"}
        try:
            page = get_json(f"{polymarket.GAMMA}/events", params, session=session)
        except RuntimeError:
            break                         # offset cap reached
        if not page:
            break
        for e in page:
            st = e.get("startTime") or ""
            if st.startswith(str(season)) and any(
                    mk.get("sportsMarketType") == "moneyline" for mk in e.get("markets", [])):
                out.append(e)
        oldest = min((e.get("startTime") or "9") for e in page)
        if len(page) < page_size or oldest < f"{season}-03-01":
            break
        offset += page_size
    return out


def polymarket_price_history(token_id: str, start_ts: int, end_ts: int,
                             session=None) -> list[dict]:
    data = get_json(f"{polymarket.CLOB}/prices-history",
                    {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
                    session=session)
    return data.get("history", [])


def polymarket_closes(season: int, session=None, events: list[dict] | None = None) -> list[dict]:
    """One row per game: P(home) from the first outcome's token, oriented."""
    events = events if events is not None else polymarket_closed_events(season, session)
    rows = []
    for e in events:
        home_id = away_id = None
        for t in e.get("teams") or []:
            tid = T.POLYMARKET_ABBREV_TO_ID.get((t.get("abbreviation") or "").lower()) \
                or T.NAME_TO_ID.get(t.get("name"))
            if t.get("ordering") == "home":
                home_id = tid
            elif t.get("ordering") == "away":
                away_id = tid
        mk = next((m for m in e.get("markets", []) if m.get("sportsMarketType") == "moneyline"), None)
        if mk is None or not home_id or not away_id or not e.get("startTime"):
            continue
        outcomes = polymarket._loads(mk.get("outcomes"), [])
        tokens = polymarket._loads(mk.get("clobTokenIds"), [])
        if len(outcomes) < 2 or len(tokens) < 2:
            continue
        first_id = T.NAME_TO_ID.get(outcomes[0])
        if first_id not in (home_id, away_id):
            continue
        fp = _ts(e["startTime"])
        try:
            hist = polymarket_price_history(tokens[0], fp - 3 * 86400, fp + 6 * 3600, session)
        except Exception as exc:
            logger.warning("prices-history failed for %s: %s", e.get("slug"), exc)
            continue
        last = last_before(hist, fp, key="t")
        if last is None:
            continue
        p_first = float(last["p"])
        p_home = p_first if first_id == home_id else round(1 - p_first, 4)
        prices = [polymarket._f(p) for p in polymarket._loads(mk.get("outcomePrices"), [])]
        home_won = None
        if prices and prices[0] is not None:
            first_won = prices[0] >= 0.99 if prices[0] >= 0.99 or prices[0] <= 0.01 else None
            if first_won is not None:
                home_won = first_won if first_id == home_id else not first_won
        start_iso = datetime.fromtimestamp(fp, timezone.utc).isoformat()
        rows.append({
            "venue": "polymarket",
            "game_date": datetime.fromtimestamp(fp, polymarket.ET).date().isoformat(),
            "game_start": start_iso, "home_id": home_id, "away_id": away_id,
            "p_home_close": p_home, "bid": None, "ask": None,
            "close_ts": int(last["t"]),
            "minutes_before_pitch": round((fp - int(last["t"])) / 60, 1),
            "volume_pre": None, "n_obs": last["_n_pre"],
            "market_id": str(mk.get("id")), "home_won": home_won,
        })
    return rows


# ───────────────────────────── Assemble ─────────────────────────────

def to_frame(rows: list[dict], schedule: pd.DataFrame) -> pd.DataFrame:
    """Attach game_pk via the shared mapper and return the CLOSE_COLUMNS frame."""
    recs = []
    for r in rows:
        rec = empty_record()
        rec.update({"ts": "backfill", "venue": r["venue"], "market_id": r["market_id"],
                    "market_type": "moneyline", "game_date": r["game_date"],
                    "game_start": r["game_start"], "home_id": r["home_id"], "away_id": r["away_id"]})
        recs.append(rec)
    stats = assign_game_pk(recs, schedule)
    logger.info("game_pk mapping: %s", stats)
    for r, rec in zip(rows, recs):
        r["game_pk"] = rec["game_pk"]
        r["home_id"], r["away_id"] = rec["home_id"], rec["away_id"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=CLOSE_COLUMNS)
    df = df[df["game_pk"].notna()].copy()
    df["game_pk"] = df["game_pk"].astype(int)
    df = df.drop_duplicates(subset=["venue", "game_pk"], keep="first")
    return df[CLOSE_COLUMNS].sort_values(["game_date", "game_pk", "venue"]).reset_index(drop=True)


# ─────────────────── Kalshi candlestick archive (maker exam) ───────────────────
#
# The close alone can only price a *taker*: it says what one quote was at one
# instant, so the only trade it can simulate is crossing that quote. A maker
# rests a limit order and is filled only if the market comes to it, which is a
# question about the whole pre-game price *path*, not about its last point.
# Kalshi keeps hourly OHLC per market for as long as the market existed, so the
# path is reconstructable after the fact — but only until we need it, hence the
# archive. One row per (market, hour) over the last 24 hours before first pitch,
# which is the window a pre-game limit order can plausibly rest for.

CANDLE_COLUMNS = [
    "market_id", "game_pk", "end_period_ts",
    "yes_bid_close", "yes_ask_close",
    "price_open", "price_high", "price_low", "price_close", "volume",
]
CANDLE_HOURS_BEFORE = 24        # how long before first pitch the order can rest


def _dollars(node, field: str):
    """`{"close_dollars": "0.5100"}` → 0.51; missing or unparseable → None."""
    if not isinstance(node, dict):
        return None
    v = node.get(f"{field}_dollars")
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def candle_rows(candles: list[dict], market_id: str, game_pk: int) -> list[dict]:
    """Kalshi's nested OHLC → flat rows in CANDLE_COLUMNS order.

    `price` is the traded price (carried forward in an hour with no trades);
    `yes_bid` / `yes_ask` are the quote at the end of the hour. Volume is
    contracts traded during the hour, and it is what tells a fill simulation
    whether the low was a real print or a stale carry-forward.
    """
    rows = []
    for c in candles:
        try:
            end_ts = int(c["end_period_ts"])
        except (KeyError, TypeError, ValueError):
            continue
        price = c.get("price") or {}
        rows.append({
            "market_id": market_id,
            "game_pk": int(game_pk),
            "end_period_ts": end_ts,
            "yes_bid_close": _dollars(c.get("yes_bid"), "close"),
            "yes_ask_close": _dollars(c.get("yes_ask"), "close"),
            "price_open": _dollars(price, "open"),
            "price_high": _dollars(price, "high"),
            "price_low": _dollars(price, "low"),
            "price_close": _dollars(price, "close"),
            "volume": float(c.get("volume_fp") or 0.0),
        })
    rows.sort(key=lambda r: r["end_period_ts"])
    return rows


def kalshi_candles_for_market(market_id: str, game_pk: int, first_pitch_ts: int,
                              hours_before: int = CANDLE_HOURS_BEFORE,
                              session=None) -> list[dict]:
    """Hourly candles for one market from T−`hours_before` to first pitch.

    The exchange returns nothing before the market's own `open_time`, so the
    start is a floor and not a promise: a market that opened twelve hours out
    yields twelve candles. Nothing after first pitch is requested at all — an
    in-game price is not information a pre-game order could have acted on.
    """
    candles = kalshi.fetch_candlesticks(
        "KXMLBGAME", market_id, first_pitch_ts - hours_before * 3600,
        first_pitch_ts, period_minutes=60, session=session)
    rows = candle_rows(candles, market_id, game_pk)
    return [r for r in rows if r["end_period_ts"] <= first_pitch_ts]


def kalshi_candle_archive(closes: pd.DataFrame, session=None,
                          pace_seconds: float = 0.25,
                          hours_before: int = CANDLE_HOURS_BEFORE,
                          skip_markets: set[str] | None = None,
                          on_market=None) -> tuple[list[dict], list[str]]:
    """Candles for every Kalshi market in a closes frame, one market at a time.

    Returns `(rows, failures)`. A market whose candles cannot be fetched is
    logged and skipped — an archive of 800-odd markets must not be lost to one
    of them — and its ticker comes back in `failures` so the caller can report
    and retry it. `on_market(market_id, rows)` is called after each success so
    the caller can checkpoint partial progress: Kalshi has rate-limited this
    client before, and a run that dies mid-flight should cost minutes of work,
    not hours.
    """
    skip = skip_markets or set()
    rows: list[dict] = []
    failures: list[str] = []
    k = closes[closes["venue"] == "kalshi"] if "venue" in closes.columns else closes
    for r in k.itertuples():
        market_id = str(r.market_id)
        if market_id in skip:
            continue
        first_pitch = _ts(str(r.game_start))
        time.sleep(pace_seconds)      # Kalshi rate-limits bursts of candle calls
        try:
            got = kalshi_candles_for_market(market_id, int(r.game_pk), first_pitch,
                                            hours_before, session)
        except Exception as exc:      # one bad market must not sink the archive
            logger.warning("candles failed for %s: %s", market_id, exc)
            failures.append(market_id)
            continue
        rows.extend(got)
        if on_market is not None:
            on_market(market_id, got)
    return rows, failures


def candle_frame(rows: list[dict]) -> pd.DataFrame:
    """CANDLE_COLUMNS frame, one row per (market, hour), de-duplicated."""
    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df = pd.DataFrame(rows)[CANDLE_COLUMNS]
    df = df.drop_duplicates(subset=["market_id", "end_period_ts"], keep="last")
    return df.sort_values(["game_pk", "market_id", "end_period_ts"]).reset_index(drop=True)
