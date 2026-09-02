"""Venue team labels → MLB Stats API team ids.

Static so tests and snapshots never depend on a network call to resolve a
name. `scripts/snapshot_market.py` asserts this table still matches
`fetch_teams()` at run time, so a franchise rename fails loudly.
"""
from __future__ import annotations

import re

# team_id, MLB abbrev, MLB full name (as of 2026)
MLB_TEAMS = [
    (108, "LAA", "Los Angeles Angels"),
    (109, "AZ", "Arizona Diamondbacks"),
    (110, "BAL", "Baltimore Orioles"),
    (111, "BOS", "Boston Red Sox"),
    (112, "CHC", "Chicago Cubs"),
    (113, "CIN", "Cincinnati Reds"),
    (114, "CLE", "Cleveland Guardians"),
    (115, "COL", "Colorado Rockies"),
    (116, "DET", "Detroit Tigers"),
    (117, "HOU", "Houston Astros"),
    (118, "KC", "Kansas City Royals"),
    (119, "LAD", "Los Angeles Dodgers"),
    (120, "WSH", "Washington Nationals"),
    (121, "NYM", "New York Mets"),
    (133, "ATH", "Athletics"),
    (134, "PIT", "Pittsburgh Pirates"),
    (135, "SD", "San Diego Padres"),
    (136, "SEA", "Seattle Mariners"),
    (137, "SF", "San Francisco Giants"),
    (138, "STL", "St. Louis Cardinals"),
    (139, "TB", "Tampa Bay Rays"),
    (140, "TEX", "Texas Rangers"),
    (141, "TOR", "Toronto Blue Jays"),
    (142, "MIN", "Minnesota Twins"),
    (143, "PHI", "Philadelphia Phillies"),
    (144, "ATL", "Atlanta Braves"),
    (145, "CWS", "Chicago White Sox"),
    (146, "MIA", "Miami Marlins"),
    (147, "NYY", "New York Yankees"),
    (158, "MIL", "Milwaukee Brewers"),
]

ID_TO_ABBREV = {tid: ab for tid, ab, _ in MLB_TEAMS}
ABBREV_TO_ID = {ab: tid for tid, ab, _ in MLB_TEAMS}
NAME_TO_ID = {name: tid for tid, _, name in MLB_TEAMS}

# Kalshi uses MLB abbreviations verbatim in tickers (verified 2026-09-02).
KALSHI_ABBREV_TO_ID = dict(ABBREV_TO_ID)

# Kalshi yes_sub_title / futures labels are city-style short names.
KALSHI_NAME_TO_ID = {
    "A's": 133, "Athletics": 133, "Arizona": 109, "Atlanta": 144,
    "Baltimore": 110, "Boston": 111, "Chicago C": 112, "Chicago WS": 145,
    "Cincinnati": 113, "Cleveland": 114, "Colorado": 115, "Detroit": 116,
    "Houston": 117, "Kansas City": 118, "Los Angeles A": 108,
    "Los Angeles D": 119, "Miami": 146, "Milwaukee": 158, "Minnesota": 142,
    "New York M": 121, "New York Y": 147, "Philadelphia": 143,
    "Pittsburgh": 134, "San Diego": 135, "San Francisco": 137,
    "Seattle": 136, "St. Louis": 138, "Tampa Bay": 139, "Texas": 140,
    "Toronto": 141, "Washington": 120,
}

# Polymarket lowercases and keeps a few legacy codes.
POLYMARKET_ABBREV_TO_ID = {ab.lower(): tid for tid, ab, _ in MLB_TEAMS}
POLYMARKET_ABBREV_TO_ID.update({"ari": 109, "oak": 133, "was": 120, "chw": 145})

_NAMES_LONGEST_FIRST = sorted(NAME_TO_ID, key=len, reverse=True)


def team_id_from_text(text: str) -> int | None:
    """Find an MLB full team name inside free text (futures questions like
    'Will the Toronto Blue Jays win the 2026 World Series?')."""
    if not text:
        return None
    for name in _NAMES_LONGEST_FIRST:
        if re.search(r"\b" + re.escape(name) + r"\b", text):
            return NAME_TO_ID[name]
    return None


def split_kalshi_pair(pair: str) -> tuple[str, str] | None:
    """'WSHLAD' → ('WSH', 'LAD'). Returns None when the split is not unique."""
    hits = []
    for i in range(2, len(pair) - 1):
        a, b = pair[:i], pair[i:]
        if a in KALSHI_ABBREV_TO_ID and b in KALSHI_ABBREV_TO_ID and a != b:
            hits.append((a, b))
    return hits[0] if len(hits) == 1 else None
