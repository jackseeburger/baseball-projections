"""Polymarket — Gamma (metadata + top of book) and CLOB (history), no key.

    gamma-api.polymarket.com/events?tag_slug=mlb   events with nested markets
    clob.polymarket.com/prices-history?market=<token>   price series

Game events carry `teams` (with home/away ordering) and `gameId`; each
market has `outcomes` / `outcomePrices` as JSON strings, `bestBid` /
`bestAsk` for the first outcome's token, and `sportsMarketType`.
Slugs keep the *original* date of a postponed game, so game dates come
from the event's `startTime` (falling back to the market's
`gameStartTime`), never from the slug.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.market import teams as T
from src.market.http import get_json
from src.market.schema import empty_record, mid_price

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
ET = ZoneInfo("America/New_York")

SPORTS_TYPES = {
    "moneyline": "moneyline",
    "totals": "total",
    "spreads": "spread",
    "nrfi": "nrfi",
    "baseball_team_first_five_winner": "first5_moneyline",
    "baseball_team_first_five_total": "first5_total",
    "baseball_team_first_five_spread": "first5_spread",
    "baseball_game_extra_innings": "extra_innings",
}

_SLUG_TYPES = [
    (re.compile(r"world-series-champion"), "futures_ws"),
    (re.compile(r"(american|national)-league-champion"), "futures_pennant"),
    (re.compile(r"(al|nl)-(east|central|west)-champion"), "futures_division"),
    (re.compile(r"make-postseason"), "futures_playoffs"),
    (re.compile(r"win-totals"), "futures_wins"),
    (re.compile(r"(mvp|cy-young|rookie|manager|hank-aaron|comeback|glove|dh-winner|"
                r"-leader)"), "futures_award"),
]


def fetch_events(tag_slug: str = "mlb", closed: bool = False, page_size: int = 100,
                 session=None) -> list[dict]:
    """Every event under a tag (offset-paginated)."""
    out, offset = [], 0
    while True:
        params = {"tag_slug": tag_slug, "closed": str(closed).lower(),
                  "limit": page_size, "offset": offset}
        if not closed:
            params["active"] = "true"
        page = get_json(f"{GAMMA}/events", params, session=session)
        out.extend(page)
        if len(page) < page_size:
            return out
        offset += page_size


def _loads(x, default):
    if x is None:
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (TypeError, ValueError):
        return default


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _game_start(market: dict, event: dict) -> str | None:
    # The event's startTime is updated when a game is postponed; a market's
    # gameStartTime can keep the original date (seen on player props).
    raw = event.get("startTime") or market.get("gameStartTime")
    if not raw:
        return None
    raw = raw.replace(" ", "T")
    if raw.endswith("+00"):
        raw = raw[:-3] + "+00:00"
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def market_type_for(market: dict, event: dict) -> str:
    st = market.get("sportsMarketType")
    if st in SPORTS_TYPES:
        return SPORTS_TYPES[st]
    if st and st.startswith("baseball_player"):
        return "player_prop"
    slug = event.get("slug") or ""
    for pat, mtype in _SLUG_TYPES:
        if pat.search(slug):
            return mtype
    return "other"


def normalize_event(event: dict, ts: str) -> list[dict]:
    """All markets of one Gamma event → schema records (first outcome)."""
    home_id = away_id = None
    for t in event.get("teams") or []:
        tid = T.POLYMARKET_ABBREV_TO_ID.get((t.get("abbreviation") or "").lower()) \
            or T.NAME_TO_ID.get(t.get("name"))
        if t.get("ordering") == "home":
            home_id = tid
        elif t.get("ordering") == "away":
            away_id = tid
    season = None
    m = re.search(r"(20\d{2})", event.get("slug") or "")
    if m:
        season = int(m.group(1))

    records = []
    for mk in event.get("markets") or []:
        outcomes = _loads(mk.get("outcomes"), [])
        prices = [_f(p) for p in _loads(mk.get("outcomePrices"), [])]
        if not outcomes:
            continue
        r = empty_record()
        start = _game_start(mk, event)
        mtype = market_type_for(mk, event)
        r.update({
            "ts": ts, "venue": "polymarket",
            "market_id": str(mk.get("id")), "event_id": event.get("slug"),
            "market_type": mtype, "title": mk.get("question"),
            "outcome": outcomes[0],
            "line": _f(mk.get("line")),
            "season": season,
            "game_start": start if mtype in _GAME_TYPES else None,
            "home_id": home_id if mtype in _GAME_TYPES else None,
            "away_id": away_id if mtype in _GAME_TYPES else None,
            "bid": _f(mk.get("bestBid")), "ask": _f(mk.get("bestAsk")),
            "last": _f(mk.get("lastTradePrice")),
            "volume": _f(mk.get("volumeNum")), "liquidity": _f(mk.get("liquidityNum")),
            "open_interest": None,
            "status": "closed" if mk.get("closed") else ("active" if mk.get("active") else "inactive"),
            "close_time": mk.get("endDate"),
        })
        if r["game_start"]:
            r["game_date"] = datetime.fromisoformat(r["game_start"]).astimezone(ET).date().isoformat()
        r["mid"] = mid_price(r["bid"], r["ask"])
        # Gamma's outcomePrices is the CLOB midpoint per outcome — a usable
        # price when the top of book is one-sided.
        if r["mid"] is None and prices and prices[0] is not None:
            r["mid"] = round(prices[0], 4)
        if mk.get("closed") and prices and prices[0] is not None:
            r["result"] = "yes" if prices[0] >= 0.99 else ("no" if prices[0] <= 0.01 else None)
        # Which team does the first outcome name?
        tid = T.NAME_TO_ID.get(outcomes[0])
        if tid is None and mtype.startswith("futures"):
            tid = T.team_id_from_text(mk.get("question") or "")
        if tid:
            r["team_id"], r["team_abbrev"] = tid, T.ID_TO_ABBREV[tid]
        records.append(r)
    return records


_GAME_TYPES = set(SPORTS_TYPES.values()) | {"player_prop"}


def fetch_price_history(token_id: str, fidelity_minutes: int = 60, session=None) -> list[dict]:
    """CLOB price series for one outcome token: [{t: unix, p: price}, ...]."""
    data = get_json(f"{CLOB}/prices-history",
                    {"market": token_id, "interval": "max", "fidelity": fidelity_minutes},
                    session=session)
    return data.get("history", [])
