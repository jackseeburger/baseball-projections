"""Archive one snapshot of every open MLB market on Kalshi + Polymarket.

No credentials. Writes an immutable gzipped JSONL under
data/market/snapshots/ and refreshes public/data/market/latest.json.

Usage:
    python scripts/snapshot_market.py               # full run
    python scripts/snapshot_market.py --dry-run     # print stats only
    python scripts/snapshot_market.py --venues kalshi
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mlb_stats_api import fetch_schedule
from src.market import snapshot
from src.market.teams import MLB_TEAMS
from src.sim.teams import fetch_teams


def check_team_table(season: int) -> None:
    live = {(int(r.team_id), r.abbrev, r.name) for r in fetch_teams(season).itertuples()}
    static = set(MLB_TEAMS)
    if live != static:
        raise SystemExit(f"src/market/teams.py is stale vs Stats API: "
                         f"live-static={sorted(live - static)} static-live={sorted(static - live)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venues", nargs="+", default=["kalshi", "polymarket"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=snapshot.SNAPSHOT_DIR)
    ap.add_argument("--latest", type=Path, default=snapshot.LATEST_PATH)
    args = ap.parse_args()

    today = date.today()
    check_team_table(today.year)
    # Window wide enough for postponed games rescheduled weeks out — venues
    # keep those markets open the whole time.
    schedule = fetch_schedule((today - timedelta(days=3)).isoformat(),
                              (today + timedelta(days=35)).isoformat())
    schedule = schedule[schedule["game_type"].isin(["R", "F", "D", "L", "W"])]

    ts = snapshot.now_iso()
    records, stats = snapshot.collect(ts, schedule=schedule, venues=tuple(args.venues))
    print(json.dumps(stats, indent=1))
    if args.dry_run:
        return
    path = snapshot.write(records, ts, args.out_dir)
    summary = snapshot.summarize(records, ts, stats)
    snapshot.write_latest(summary, args.latest)
    print(f"wrote {len(records)} records → {path}")
    print(f"{len(summary['games'])} games with moneylines → {args.latest}")


if __name__ == "__main__":
    main()
