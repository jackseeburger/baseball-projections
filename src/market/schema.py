"""One record per (venue, market) per snapshot. All prices are P(outcome)
in [0, 1] from the point of view of `outcome` — the YES side on Kalshi,
the first listed outcome on Polymarket.
"""
from __future__ import annotations

FIELDS = [
    "ts",             # snapshot time, ISO-8601 UTC (same for every row)
    "venue",          # "kalshi" | "polymarket"
    "market_id",      # venue's market identifier (Kalshi ticker / Polymarket id)
    "event_id",       # venue's event grouping (Kalshi event_ticker / Polymarket slug)
    "market_type",    # see MARKET_TYPES
    "title",
    "outcome",        # label the prices refer to (team name, "Over", "Tie", ...)
    "team_id",        # MLB Stats API team id for `outcome` when it is a team
    "team_abbrev",    # MLB Stats API abbreviation for the same
    "line",           # total / spread number, else None
    "season",
    "game_pk",        # MLB Stats API game id for game markets (None until mapped)
    "game_date",      # ET calendar date of the game (YYYY-MM-DD)
    "game_start",     # scheduled first pitch, ISO UTC (venue's belief)
    "home_id",
    "away_id",
    "bid",
    "ask",
    "mid",
    "last",
    "volume",
    "liquidity",
    "open_interest",
    "status",         # venue status string
    "result",         # settlement ("yes"/"no"/outcome label) once resolved
    "close_time",     # venue's market close / expiry, ISO
]

MARKET_TYPES = {
    "moneyline", "total", "spread",
    "first5_moneyline", "first5_total", "first5_spread",
    "nrfi", "extra_innings", "player_prop",
    "futures_ws", "futures_pennant", "futures_division", "futures_playoffs",
    "futures_best_record", "futures_wins", "futures_award",
    "other",
}

GAME_MARKET_TYPES = {
    "moneyline", "total", "spread",
    "first5_moneyline", "first5_total", "first5_spread",
    "nrfi", "extra_innings", "player_prop",
}


def empty_record() -> dict:
    return {f: None for f in FIELDS}


def mid_price(bid, ask):
    """Midpoint when both sides are quoted, else None (a one-sided book is
    not a price)."""
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0 or ask > 1:
        return None
    return round((bid + ask) / 2, 4)


def validate(record: dict) -> None:
    """Raise if a record violates the schema — the snapshot guard."""
    missing = set(FIELDS) - set(record)
    extra = set(record) - set(FIELDS)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    if record["venue"] not in ("kalshi", "polymarket"):
        raise ValueError(f"bad venue {record['venue']!r}")
    if record["market_type"] not in MARKET_TYPES:
        raise ValueError(f"bad market_type {record['market_type']!r}")
    for f in ("bid", "ask", "mid", "last"):
        v = record[f]
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"{f}={v} out of [0,1] for {record['market_id']}")
    if not record["market_id"] or not record["ts"]:
        raise ValueError("market_id and ts are required")
