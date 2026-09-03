"""Venue player names → MLBAM ids.

Neither exchange publishes an MLBAM id on a player prop. Kalshi's
`custom_strike.baseball_player` is a vendor UUID and Polymarket's question is
free text, so the only join key either venue offers is the **printed name**.
Every downstream number — the Marcel rate, the posted lineup slot, the
settlement check — is keyed on MLBAM, so a prop that cannot be resolved cannot
be priced, and the resolution rate is a headline coverage number rather than a
detail.

The map is built once per season from
``statsapi.mlb.com/api/v1/sports/1/players?season=YYYY`` — one request for
every player on a 40-man roster that season (~1,450 in 2026) — and cached under
``data/cache/market/`` (gitignored) so a rerun is offline and identical. Names
that stay unresolved fall back to ``people/search?names=``, one request each,
which is what catches a September call-up who is not yet in the season list.

Matching is deliberately forgiving in the ways venue text differs from the
Stats API and strict everywhere else:

* accents folded (``Christian Vázquez`` → ``vazquez``), because Polymarket
  drops them and Kalshi keeps them;
* punctuation dropped (``Jr.`` → ``jr``, ``O'Neill`` → ``oneill``);
* generational suffixes dropped on a second pass only, so ``Bobby Witt Jr.``
  still prefers the Witt who is actually Jr. when both exist.

A normalized name that maps to **two different ids** is left unresolved rather
than guessed — silently picking one would attach a prop to the wrong player's
rates, which is worse than dropping it.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

from src.config import DATA_DIR
from src.market.http import get_json
from src.market.schema import PROP_MARKET_TYPES

logger = logging.getLogger(__name__)

STATS_API = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = DATA_DIR / "cache" / "market"

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalize_name(name: str | None, drop_suffix: bool = False) -> str:
    """Fold a printed name to the form both venues and the Stats API share."""
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", str(name))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = _PUNCT.sub(" ", folded.lower())
    parts = folded.split()
    if drop_suffix:
        parts = [p for p in parts if p not in _SUFFIXES] or parts
    return " ".join(parts)


def _index(people: list[dict]) -> dict[str, int | None]:
    """{normalized name: id}, with ambiguous names mapped to None.

    Both the exact and the suffix-stripped spelling are indexed; the exact one
    wins on lookup, so a suffix only ever breaks a tie that the exact spelling
    could not.
    """
    exact: dict[str, int | None] = {}
    loose: dict[str, int | None] = {}
    for p in people:
        pid, name = p.get("id"), p.get("fullName")
        if pid is None or not name:
            continue
        for idx, key in ((exact, normalize_name(name)),
                         (loose, normalize_name(name, drop_suffix=True))):
            if not key:
                continue
            if key in idx and idx[key] != int(pid):
                idx[key] = None              # ambiguous — refuse to guess
            else:
                idx.setdefault(key, int(pid))
    return {**loose, **exact}                # the exact spelling wins a tie


def fetch_season_players(season: int, refresh: bool = False, session=None) -> list[dict]:
    """Every player who appeared on a roster in `season` (id + fullName)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"players_{season}.json"
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            logger.warning("corrupt %s; re-fetching", cache)
    data = get_json(f"{STATS_API}/sports/1/players", {"season": season}, session=session)
    people = [{"id": p["id"], "fullName": p.get("fullName")}
              for p in data.get("people", []) if p.get("id")]
    cache.write_text(json.dumps(people))
    return people


class NameResolver:
    """Name → MLBAM id, with a per-instance memo and a search fallback.

    `search=False` makes the resolver a pure function of the cached season
    list, which is what the tests use.
    """

    def __init__(self, season: int, people: list[dict] | None = None,
                 search: bool = True, session=None):
        self.season = season
        self.search = search
        self.session = session
        if people is None:
            people = fetch_season_players(season, session=session)
        self.index = _index(people)
        self.misses: set[str] = set()
        self.ambiguous: set[str] = set()
        self._memo: dict[str, int | None] = {}

    def _lookup(self, name: str) -> int | None:
        for key in (normalize_name(name), normalize_name(name, drop_suffix=True)):
            if key in self.index:
                if self.index[key] is None:
                    self.ambiguous.add(key)
                    return None
                return self.index[key]
        return None

    def _search(self, name: str) -> int | None:
        """`people/search` — one request, only for a name the season list missed."""
        try:
            data = get_json(f"{STATS_API}/people/search",
                            {"names": name, "sportIds": 1}, session=self.session)
        except RuntimeError as exc:                     # pragma: no cover - network
            logger.warning("people/search failed for %r: %s", name, exc)
            return None
        hits = _index([{"id": p.get("id"), "fullName": p.get("fullName")}
                       for p in data.get("people", [])])
        for key in (normalize_name(name), normalize_name(name, drop_suffix=True)):
            if hits.get(key):
                pid = hits[key]
                self.index[key] = pid
                return pid
        return None

    def resolve(self, name: str | None) -> int | None:
        if not name:
            return None
        if name in self._memo:
            return self._memo[name]
        pid = self._lookup(name)
        if pid is None and self.search:
            pid = self._search(name)
        if pid is None:
            self.misses.add(name)
        self._memo[name] = pid
        return pid


def assign_player_ids(records: list[dict], resolver: NameResolver) -> dict:
    """Fill `player_id` in place on every prop record. Returns coverage stats."""
    stats = {"prop_markets": 0, "named": 0, "resolved": 0,
             "unresolved_names": 0, "players": 0}
    ids = set()
    for r in records:
        if r["market_type"] not in PROP_MARKET_TYPES:
            continue
        stats["prop_markets"] += 1
        if not r["player_name"]:
            continue
        stats["named"] += 1
        pid = resolver.resolve(r["player_name"])
        if pid is not None:
            r["player_id"] = pid
            stats["resolved"] += 1
            ids.add(pid)
    stats["players"] = len(ids)
    stats["unresolved_names"] = len(resolver.misses)
    stats["resolution_rate"] = round(stats["resolved"] / stats["prop_markets"], 4) \
        if stats["prop_markets"] else None
    return stats
