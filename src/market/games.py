"""Map venue game markets onto MLB Stats API `game_pk`s.

A schedule frame (from `mlb_stats_api.fetch_schedule`) has one row per
game with `game_pk, date, home_id, away_id` and, when available,
`game_datetime` (UTC first pitch) for doubleheader disambiguation.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.market.schema import GAME_MARKET_TYPES


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_index(schedule: pd.DataFrame) -> dict[tuple, list[dict]]:
    """(date, home_id, away_id) → [games] (a list because of doubleheaders)."""
    idx: dict[tuple, list[dict]] = {}
    has_dt = "game_datetime" in schedule.columns
    for row in schedule.itertuples(index=False):
        key = (str(row.date), int(row.home_id), int(row.away_id))
        idx.setdefault(key, []).append({
            "game_pk": int(row.game_pk),
            "start": _parse(getattr(row, "game_datetime", None)) if has_dt else None,
        })
    return idx


def _pick(candidates: list[dict], start: datetime | None) -> int:
    if len(candidates) == 1 or start is None:
        return candidates[0]["game_pk"]
    timed = [c for c in candidates if c["start"] is not None]
    if not timed:
        return candidates[0]["game_pk"]
    return min(timed, key=lambda c: abs((c["start"] - start).total_seconds()))["game_pk"]


def assign_game_pk(records: list[dict], schedule: pd.DataFrame) -> dict:
    """Fill `game_pk` in place for game-market records. Returns coverage stats.

    Match order: exact (date, home, away) → teams swapped on the same date
    (venue disagreed about home/away) → either orientation ±1 day (late
    West Coast games, venue clock skew).
    """
    idx = build_index(schedule)
    stats = {"game_markets": 0, "mapped": 0, "swapped": 0, "day_shift": 0, "unmapped": 0}
    for r in records:
        if r["market_type"] not in GAME_MARKET_TYPES:
            continue
        stats["game_markets"] += 1
        if not (r["game_date"] and r["home_id"] and r["away_id"]):
            stats["unmapped"] += 1
            continue
        start = _parse(r["game_start"])
        date = datetime.fromisoformat(r["game_date"]).date()
        found = None
        for shift in (0, -1, 1):
            d = (date + timedelta(days=shift)).isoformat()
            for swapped in (False, True):
                h, a = (r["away_id"], r["home_id"]) if swapped else (r["home_id"], r["away_id"])
                cands = idx.get((d, h, a))
                if cands:
                    found = _pick(cands, start)
                    if swapped:
                        stats["swapped"] += 1
                        r["home_id"], r["away_id"] = h, a
                    if shift:
                        stats["day_shift"] += 1
                    break
            if found is not None:
                break
        if found is None:
            stats["unmapped"] += 1
        else:
            r["game_pk"] = found
            stats["mapped"] += 1
    return stats
