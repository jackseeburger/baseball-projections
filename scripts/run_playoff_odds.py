"""Compute playoff odds for all 30 teams and write a dated JSON snapshot.

Reads live standings + schedule from the MLB Stats API (no credentials),
runs the season Monte Carlo + bracket, and writes:

    public/data/playoff_odds/YYYY-MM-DD.json   (never overwritten — roadmap 3.1)
    public/data/playoff_odds/latest.json

Usage:
    python scripts/run_playoff_odds.py --sims 20000
    python scripts/run_playoff_odds.py --sims 2000 --dry-run   # print only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mlb_stats_api import fetch_schedule, fetch_standings
from src.sim.odds import run_playoff_odds
from src.sim.season import from_schedule
from src.sim.strength import estimate_hfa, regressed_strength
from src.sim.teams import DIVISION_NAMES, fetch_teams

OUT_DIR = Path(__file__).resolve().parent.parent / "public/data/playoff_odds"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--sims", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: day-of-year, so reruns on a "
                             "day reproduce)")
    parser.add_argument("--regress-games", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    today = date.today()
    seed = args.seed if args.seed is not None else today.timetuple().tm_yday

    teams = fetch_teams(args.season)
    standings = fetch_standings(args.season)
    schedule = fetch_schedule(f"{args.season}-03-01", f"{args.season}-10-15")
    state = from_schedule(schedule, teams)
    strength = regressed_strength(standings, regress_games=args.regress_games)
    hfa = estimate_hfa(state.completed)
    print(f"{len(state.completed)} games played, {len(state.remaining)} remaining, "
          f"HFA={hfa:.4f}, sims={args.sims}, seed={seed}")

    odds = run_playoff_odds(state, strength, hfa, n_sims=args.sims, seed=seed)
    odds["division"] = odds["division_id"].map(DIVISION_NAMES)

    show = odds[["abbrev", "division", "wins", "losses", "strength", "mean_wins",
                 "p_playoffs", "p_division", "p_bye", "p_pennant", "p_ws"]]
    with_fmt = show.copy()
    for c in ("p_playoffs", "p_division", "p_bye", "p_pennant", "p_ws"):
        with_fmt[c] = (with_fmt[c] * 100).round(1)
    with_fmt["strength"] = with_fmt["strength"].round(3)
    with_fmt["mean_wins"] = with_fmt["mean_wins"].round(1)
    print(with_fmt.to_string(index=False))

    if args.dry_run:
        return

    payload = {
        "season": args.season,
        "as_of": today.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sims": args.sims,
        "seed": seed,
        "hfa": hfa,
        "games_played": int(len(state.completed)),
        "games_remaining": int(len(state.remaining)),
        "method": "Regressed Pythagenpat strength, log5 + HFA, MLB tiebreakers",
        "teams": json.loads(odds.drop(columns=["division_id"]).to_json(orient="records")),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dated = args.out_dir / f"{today.isoformat()}.json"
    if dated.exists():
        print(f"snapshot {dated.name} already exists; not overwriting (updating latest.json only)")
    else:
        dated.write_text(json.dumps(payload, indent=1))
        print(f"wrote {dated}")
    (args.out_dir / "latest.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
