"""The Odds API — sportsbook lines (needs ODDS_API_KEY; free tier 500 req/month).

Sportsbooks are the *reference* price (docs/architecture.md §0): Pinnacle's
close is the sharpest public number, so it is the benchmark models are
judged against. They are not the money venue — that is the exchanges.

Quota: one call costs (markets × regions) requests. `h2h,totals` over
`us,eu` is 4; two snapshots a day is ~240/month.

Record shape: one row per (bookmaker, market, outcome). `last` carries
the book's raw implied probability (with vig), `mid` the de-vigged fair
probability (multiplicative, within that book's two-way market), and
`odds_decimal` the quoted price.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from src.market import teams as T
from src.market.devig import implied, multiplicative
from src.market.schema import empty_record

BASE = "https://api.the-odds-api.com/v4"
ET = ZoneInfo("America/New_York")
DEFAULT_REGIONS = "us,eu"
DEFAULT_MARKETS = "h2h,totals"
MARKET_TYPES = {"h2h": "moneyline", "totals": "total", "spreads": "spread"}


def api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY") or None


def fetch_odds(sport: str = "baseball_mlb", regions: str = DEFAULT_REGIONS,
               markets: str = DEFAULT_MARKETS, key: str | None = None,
               session: requests.Session | None = None, timeout: float = 60.0
               ) -> tuple[list[dict], dict]:
    """Return (events, quota) where quota has requests_used / requests_remaining."""
    key = key or api_key()
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    sess = session or requests
    r = sess.get(f"{BASE}/sports/{sport}/odds/", timeout=timeout, params={
        "apiKey": key, "regions": regions, "markets": markets,
        "oddsFormat": "decimal", "dateFormat": "iso",
    })
    r.raise_for_status()
    quota = {
        "requests_used": _int(r.headers.get("x-requests-used")),
        "requests_remaining": _int(r.headers.get("x-requests-remaining")),
        "requests_last": _int(r.headers.get("x-requests-last")),
    }
    return r.json(), quota


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _iso_utc(s: str) -> str:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()


def normalize_event(event: dict, ts: str) -> list[dict]:
    """All (book, market, outcome) rows for one Odds API event."""
    home_id = T.NAME_TO_ID.get(event.get("home_team"))
    away_id = T.NAME_TO_ID.get(event.get("away_team"))
    start = _iso_utc(event["commence_time"])
    start_dt = datetime.fromisoformat(start)
    game_date = start_dt.astimezone(ET).date().isoformat()
    live = start_dt <= datetime.fromisoformat(ts)
    season = start_dt.year

    rows = []
    for book in event.get("bookmakers", []):
        for mk in book.get("markets", []):
            mtype = MARKET_TYPES.get(mk.get("key"))
            if mtype is None:
                continue
            outcomes = mk.get("outcomes") or []
            odds = [o.get("price") for o in outcomes]
            if len(outcomes) != 2 or any(not o or o <= 1 for o in odds):
                continue
            fair = multiplicative(odds)
            for o, price, p_fair in zip(outcomes, odds, fair):
                r = empty_record()
                team_id = T.NAME_TO_ID.get(o.get("name"))
                r.update({
                    "ts": ts, "venue": "oddsapi", "book": book["key"],
                    "market_id": f"{event['id']}:{book['key']}:{mk['key']}:{o.get('name')}",
                    "event_id": event["id"], "market_type": mtype,
                    "title": f"{event.get('away_team')} @ {event.get('home_team')} {mk['key']}",
                    "outcome": o.get("name"),
                    "team_id": team_id, "team_abbrev": T.ID_TO_ABBREV.get(team_id),
                    "line": o.get("point"), "season": season,
                    "game_date": game_date, "game_start": start,
                    "home_id": home_id, "away_id": away_id,
                    "odds_decimal": float(price),
                    "last": round(implied(price), 4), "mid": round(p_fair, 4),
                    "status": "live" if live else "pregame",
                    "close_time": book.get("last_update"),
                })
                rows.append(r)
    return rows


def normalize(events: list[dict], ts: str) -> list[dict]:
    out = []
    for e in events:
        out.extend(normalize_event(e, ts))
    return out
