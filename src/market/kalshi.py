"""Kalshi trade-api v2 — public market data, no credentials.

Docs: https://trading-api.readme.io/reference  (markets, candlesticks)

Series we archive (verified live 2026-09-02):
    KXMLBGAME     moneyline, one YES market per team per game
    KXMLBTOTAL    game totals, one market per strike ("Over 8.5 runs scored")
    KXMLBSPREAD   run lines ("St. Louis wins by over 3.5 runs?")
    KXMLBF5       first-5-innings winner (has a TIE market)
    KXMLB         World Series winner            KXMLBPLAYOFFS  make playoffs
    KXMLBAL/NL    pennant                        KXMLB{AL,NL}{EAST,CENT,WEST}
    KXMLBBESTRECORD

Player props (verified live 2026-09-02) reuse the game event ticker and add a
player-and-strike suffix to the market ticker:

    KXMLBHIT-26SEP022010CWSHOU-HOULWADE31-3   "LaMonte Wade Jr.: 3+ hits?"

The trailing `-3` is the integer threshold; `floor_strike` is 2.5 and
`strike_type` is "greater", so YES is *at least* that many. The player is
named only in `title` / `yes_sub_title` — `custom_strike.baseball_player` is a
vendor UUID, not an MLBAM id — so `src/market/players.py` resolves the name.

Game event tickers encode the ET first pitch and the away/home pair:
    KXMLBGAME-26SEP042210WSHLAD  →  2026-09-04 22:10 ET, WSH @ LAD
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.market import teams as T
from src.market.http import get_json
from src.market.schema import PROP_MARKET_TYPES, empty_record, mid_price

BASE = "https://api.elections.kalshi.com/trade-api/v2"
ET = ZoneInfo("America/New_York")

# One series per prop stat. Kept apart from SERIES so the snapshot job and the
# prop backfill can ask for props alone (they are ~10x the market count of
# everything else put together).
PROP_SERIES = {
    "KXMLBHR": "prop_hr",
    "KXMLBKS": "prop_k",
    "KXMLBHIT": "prop_hits",
    "KXMLBTB": "prop_tb",
    "KXMLBRBI": "prop_rbi",
    "KXMLBSB": "prop_sb",
    "KXMLBOUTS": "prop_outs",
}

SERIES = {
    "KXMLBGAME": "moneyline",
    "KXMLBTOTAL": "total",
    "KXMLBSPREAD": "spread",
    "KXMLBF5": "first5_moneyline",
    "KXMLB": "futures_ws",
    "KXMLBPLAYOFFS": "futures_playoffs",
    "KXMLBAL": "futures_pennant",
    "KXMLBNL": "futures_pennant",
    "KXMLBALEAST": "futures_division",
    "KXMLBALCENT": "futures_division",
    "KXMLBALWEST": "futures_division",
    "KXMLBNLEAST": "futures_division",
    "KXMLBNLCENT": "futures_division",
    "KXMLBNLWEST": "futures_division",
    "KXMLBBESTRECORD": "futures_best_record",
    **PROP_SERIES,
}

_GAME_EVENT = re.compile(
    r"^(?P<series>KX[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<hh>\d{2})(?P<mm>\d{2})(?P<pair>[A-Z]+?)(G(?P<game>\d))?$"
)   # trailing G1/G2 marks doubleheader games
_FUTURES_EVENT = re.compile(r"^(?P<series>KX[A-Z0-9]+)-(?P<yy>\d{2})$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _f(x):
    """Kalshi returns dollars as strings; blank/zero-sided quotes → None."""
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_markets(series_ticker: str, status: str = "open", limit: int = 200,
                  session=None) -> list[dict]:
    """All markets in a series (cursor-paginated)."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": series_ticker, "status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        page = get_json(f"{BASE}/markets", params, session=session)
        out.extend(page.get("markets", []))
        cursor = page.get("cursor")
        if not cursor or not page.get("markets"):
            return out


def fetch_all(series: dict[str, str] | None = None, status: str = "open",
              session=None) -> list[dict]:
    series = series or SERIES
    raw = []
    for ticker in series:
        raw.extend(fetch_markets(ticker, status=status, session=session))
    return raw


def parse_event(event_ticker: str) -> dict:
    """Decode the pieces we need from an event ticker."""
    m = _GAME_EVENT.match(event_ticker)
    if m:
        year = 2000 + int(m["yy"])
        start_et = datetime(year, _MONTHS[m["mon"]], int(m["dd"]),
                            int(m["hh"]), int(m["mm"]), tzinfo=ET)
        pair = T.split_kalshi_pair(m["pair"])
        away, home = pair if pair else (None, None)
        return {
            "series": m["series"], "season": year,
            "game_date": start_et.date().isoformat(),
            "game_start": start_et.astimezone(timezone.utc).isoformat(),
            "away_abbrev": away, "home_abbrev": home,
            "game_number": int(m["game"]) if m["game"] else None,
        }
    m = _FUTURES_EVENT.match(event_ticker)
    if m:
        return {"series": m["series"], "season": 2000 + int(m["yy"])}
    return {"series": event_ticker.split("-")[0], "season": None}


_OVER = re.compile(r"over ([\d.]+)", re.I)
# "LaMonte Wade Jr.: 3+" / "Bubba Chandler: 16+ Outs Recorded?" — the player is
# everything before the last colon, the threshold the integer after it.
_PROP_LABEL = re.compile(r"^(?P<name>.+):\s*(?P<n>\d+)\+")


def parse_prop_label(text: str | None) -> tuple[str | None, float | None]:
    """('LaMonte Wade Jr.: 3+', ...) → ("LaMonte Wade Jr.", 2.5).

    The line returned is the *over/under* number (one below the "N+"
    threshold), so YES pays on strictly more than `prop_line` — the same
    convention as a total and as Polymarket's `line` field.
    """
    m = _PROP_LABEL.match((text or "").strip())
    if not m:
        return None, None
    return m["name"].strip(), float(m["n"]) - 0.5


def prop_team_id(ticker: str) -> int | None:
    """The club the prop's player is on, from the ticker's player segment.

    `KXMLBHIT-26SEP022010CWSHOU-HOULWADE31-3` → the `HOULWADE31` segment is the
    team abbreviation followed by an opaque player code, so the abbreviation is
    the longest known prefix. Useful because a prop is a *player* market with
    no home/away side of its own.
    """
    parts = ticker.split("-")
    if len(parts) < 4:
        return None
    seg = parts[-2]
    for n in (3, 2):
        tid = T.KALSHI_ABBREV_TO_ID.get(seg[:n])
        if tid:
            return tid
    return None


def normalize(market: dict, ts: str) -> dict:
    """One Kalshi market → schema record. Prices are for the YES side."""
    r = empty_record()
    ev = parse_event(market.get("event_ticker", ""))
    series = ev["series"]
    r.update({
        "ts": ts, "venue": "kalshi",
        "market_id": market["ticker"], "event_id": market.get("event_ticker"),
        "market_type": SERIES.get(series, "other"),
        "title": market.get("title"),
        "season": ev.get("season"),
        "game_date": ev.get("game_date"), "game_start": ev.get("game_start"),
        "home_id": T.KALSHI_ABBREV_TO_ID.get(ev.get("home_abbrev")),
        "away_id": T.KALSHI_ABBREV_TO_ID.get(ev.get("away_abbrev")),
        "bid": _f(market.get("yes_bid_dollars")),
        "ask": _f(market.get("yes_ask_dollars")),
        "last": _f(market.get("last_price_dollars")),
        "volume": _f(market.get("volume_fp")),
        "liquidity": _f(market.get("liquidity_dollars")),
        "open_interest": _f(market.get("open_interest_fp")),
        "status": market.get("status"),
        "result": market.get("result") or None,
        "close_time": market.get("close_time"),
    })
    # Zero means "no quote" on Kalshi, not a free option.
    if r["bid"] == 0:
        r["bid"] = None
    if r["ask"] == 0:
        r["ask"] = None
    r["mid"] = mid_price(r["bid"], r["ask"])

    suffix = market["ticker"].rsplit("-", 1)[-1]
    label = market.get("yes_sub_title") or ""
    mtype = r["market_type"]
    if mtype in PROP_MARKET_TYPES:
        name, line = parse_prop_label(label or market.get("title"))
        # `floor_strike` is the same number and is authoritative when the
        # label ever changes shape; the label is the only source of the name.
        floor = _f(market.get("floor_strike"))
        r["prop_stat"] = PROP_MARKET_TYPES[mtype]
        r["prop_line"] = floor if floor is not None else line
        r["player_name"] = name
        r["outcome"] = "Over"
        r["team_id"] = prop_team_id(market["ticker"])
    elif mtype == "total":
        m = _OVER.search(market.get("title") or "")
        r["outcome"] = "Over"
        r["line"] = float(m.group(1)) if m else None
    elif mtype == "spread":
        abbrev = re.sub(r"\d+$", "", suffix)
        r["outcome"] = label
        r["team_id"] = T.KALSHI_ABBREV_TO_ID.get(abbrev) or T.KALSHI_NAME_TO_ID.get(label)
        m = _OVER.search(market.get("title") or "")
        r["line"] = -float(m.group(1)) if m else None   # favorite's run line
    elif suffix == "TIE":
        r["outcome"] = "Tie"
    else:
        r["outcome"] = label or suffix
        r["team_id"] = T.KALSHI_ABBREV_TO_ID.get(suffix) or T.KALSHI_NAME_TO_ID.get(label)
    if r["team_id"]:
        r["team_abbrev"] = T.ID_TO_ABBREV[r["team_id"]]
    return r


def fetch_candlesticks(series_ticker: str, market_ticker: str, start_ts: int,
                       end_ts: int, period_minutes: int = 60, session=None) -> list[dict]:
    """Hourly OHLC of price / yes_bid / yes_ask — the pre-game close for
    settled markets lives here (part 3 of the archive ticket)."""
    data = get_json(
        f"{BASE}/series/{series_ticker}/markets/{market_ticker}/candlesticks",
        {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_minutes},
        session=session,
    )
    return data.get("candlesticks", [])
