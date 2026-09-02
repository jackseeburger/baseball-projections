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
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

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

def kalshi_settled_games(season: int, session=None, window_days: int = 7) -> list[dict]:
    """Every settled KXMLBGAME market for a season, paged by close-time window."""
    start = datetime(season, 3, 15, tzinfo=timezone.utc)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    out: list[dict] = []
    cursor = start
    while cursor < end:
        nxt = cursor + timedelta(days=window_days)
        params = {"series_ticker": "KXMLBGAME", "status": "settled", "limit": 1000,
                  "min_close_ts": int(cursor.timestamp()), "max_close_ts": int(nxt.timestamp())}
        page = get_json(f"{kalshi.BASE}/markets", params, session=session)
        markets = page.get("markets", [])
        out.extend(markets)
        if page.get("cursor"):
            logger.warning("Kalshi window %s..%s had a cursor; shrink window_days", cursor.date(), nxt.date())
        cursor = nxt
    return out


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
