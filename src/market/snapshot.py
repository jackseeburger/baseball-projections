"""Collect one snapshot from every venue and write it, never overwriting.

Layout:
    data/market/snapshots/YYYY-MM-DDTHHMMZ.jsonl.gz   one line per record
    public/data/market/latest.json                     small game/futures summary

Dated files are immutable (roadmap 3.1). The summary is for the site and
for eyeballing; analysis reads the gzipped snapshots.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.market import kalshi, oddsapi, polymarket
from src.market.games import assign_game_pk
from src.market.schema import FIELDS, validate

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_DIR = ROOT / "data/market/snapshots"
LATEST_PATH = ROOT / "public/data/market/latest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collect(ts: str | None = None, schedule: pd.DataFrame | None = None,
            venues: tuple[str, ...] = ("kalshi", "polymarket"), session=None) -> tuple[list[dict], dict]:
    """Pull every venue, normalize, map to game_pk. Returns (records, stats)."""
    ts = ts or now_iso()
    records: list[dict] = []
    stats: dict = {"ts": ts}
    if "kalshi" in venues:
        raw = kalshi.fetch_all(session=session)
        records.extend(kalshi.normalize(m, ts) for m in raw)
        stats["kalshi_markets"] = len(raw)
    if "polymarket" in venues:
        events = polymarket.fetch_events(session=session)
        n = 0
        for ev in events:
            recs = polymarket.normalize_event(ev, ts)
            records.extend(recs)
            n += len(recs)
        stats["polymarket_events"] = len(events)
        stats["polymarket_markets"] = n
    if "oddsapi" in venues:
        events, quota = oddsapi.fetch_odds(session=session)
        recs = oddsapi.normalize(events, ts)
        records.extend(recs)
        stats["oddsapi_events"] = len(events)
        stats["oddsapi_rows"] = len(recs)
        stats["oddsapi_quota"] = quota
    for r in records:
        validate(r)
    if schedule is not None:
        stats["mapping"] = assign_game_pk(records, schedule)
    return records, stats


def snapshot_path(ts: str, out_dir: Path = SNAPSHOT_DIR) -> Path:
    stamp = datetime.fromisoformat(ts).astimezone(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    return out_dir / f"{stamp}.jsonl.gz"


def write(records: list[dict], ts: str, out_dir: Path = SNAPSHOT_DIR) -> Path:
    path = snapshot_path(ts, out_dir)
    if path.exists():
        raise FileExistsError(f"{path} exists; snapshots are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps({f: r[f] for f in FIELDS}, separators=(",", ":")) + "\n")
    return path


def read(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh], columns=FIELDS)


def summarize(records: list[dict], ts: str, stats: dict | None = None) -> dict:
    """Game-level moneyline table (P(home) by venue) + team futures."""
    games: dict = {}
    for r in records:
        if r["market_type"] != "moneyline" or not r["game_pk"] or not r["team_id"]:
            continue
        g = games.setdefault(r["game_pk"], {
            "game_pk": r["game_pk"], "date": r["game_date"],
            "home_id": r["home_id"], "away_id": r["away_id"], "game_start": r["game_start"],
        })
        is_home = r["team_id"] == r["home_id"]
        price = r["mid"] if r["mid"] is not None else r["last"]
        if price is None:
            continue
        p_home = price if is_home else round(1 - price, 4)
        if r["venue"] == "oddsapi":
            if not is_home or r["status"] == "live":
                continue                       # one row per book, pregame only
            g.setdefault("_books", {})[r["book"]] = p_home
            continue
        key = f"{r['venue']}_p_home"
        # Prefer the home-side market's own quote; fall back to 1 - away.
        if is_home or key not in g:
            g[key] = p_home
            g[f"{r['venue']}_volume"] = r["volume"]
    for g in games.values():
        books = g.pop("_books", None)
        if books:
            g["books_n"] = len(books)
            g["books_consensus_p_home"] = round(sum(books.values()) / len(books), 4)
            if "pinnacle" in books:
                g["pinnacle_p_home"] = books["pinnacle"]

    futures: dict = defaultdict(dict)
    for r in records:
        if not r["market_type"].startswith("futures") or not r["team_abbrev"]:
            continue
        price = r["mid"] if r["mid"] is not None else r["last"]
        if price is None:
            continue
        futures[r["market_type"]].setdefault(r["team_abbrev"], {})[r["venue"]] = price

    return {
        "as_of": ts,
        "n_records": len(records),
        "stats": stats or {},
        "games": sorted(games.values(), key=lambda g: (g["date"] or "", g["game_pk"])),
        "futures": {k: dict(sorted(v.items())) for k, v in sorted(futures.items())},
    }


def write_latest(summary: dict, path: Path = LATEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    return path
