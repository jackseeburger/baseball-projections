"""Compare our latest playoff odds snapshot with FanGraphs (roadmap 2.6 acceptance).

Usage:
    python scripts/compare_public_odds.py                # uses latest.json
    python scripts/compare_public_odds.py --date 2026-09-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

ODDS_DIR = Path(__file__).resolve().parent.parent / "public/data/playoff_odds"
FG_URL = ("https://www.fangraphs.com/api/playoff-odds/odds"
          "?dateEnd={date}&dateDelta=&projectionMode=2&standingsType=div")

FIELDS = [("p_playoffs", "poffTitle"), ("p_division", "divTitle"),
          ("p_bye", "div2Title"), ("p_pennant", "csWin"), ("p_ws", "wsWin")]


def fetch_fangraphs(date: str) -> pd.DataFrame:
    rows = requests.get(FG_URL.format(date=date), timeout=60).json()
    return pd.DataFrame([{
        "short": r["shortName"], "fg_exp_wins": r["endData"]["ExpW"],
        **{f"fg_{ours[2:]}": r["endData"][theirs] for ours, theirs in FIELDS},
    } for r in rows])


def compare(snapshot: dict) -> pd.DataFrame:
    ours = pd.DataFrame(snapshot["teams"])
    fg = fetch_fangraphs(snapshot["as_of"])
    ours["short"] = ours["name"].map(
        lambda n: next((s for s in fg["short"] if n.endswith(s)), None))
    m = ours.merge(fg, on="short", how="left")
    missing = m[m["fg_exp_wins"].isna()]["name"].tolist()
    if missing:
        raise RuntimeError(f"could not match to FanGraphs: {missing}")
    for ours_col, _ in FIELDS:
        m[f"diff_{ours_col[2:]}"] = (m[ours_col] - m[f"fg_{ours_col[2:]}"]) * 100
    m["diff_exp_wins"] = m["mean_wins"] - m["fg_exp_wins"]
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="snapshot date (default latest)")
    args = parser.parse_args()
    path = ODDS_DIR / (f"{args.date}.json" if args.date else "latest.json")
    m = compare(json.loads(path.read_text()))

    show = m[["abbrev", "wins", "losses", "mean_wins", "fg_exp_wins",
              "p_playoffs", "fg_playoffs", "diff_playoffs",
              "p_division", "fg_division", "diff_division",
              "p_ws", "fg_ws", "diff_ws"]].copy()
    for c in ("p_playoffs", "fg_playoffs", "p_division", "fg_division", "p_ws", "fg_ws"):
        show[c] = show[c] * 100
    print(show.round(1).sort_values("fg_playoffs", ascending=False).to_string(index=False))
    print("\nmean |diff| (points):",
          {k[2:]: round(float(m[f"diff_{k[2:]}"].abs().mean()), 2) for k, _ in FIELDS},
          f"| exp wins: {m['diff_exp_wins'].abs().mean():.2f}")
    worst = m.reindex(m["diff_playoffs"].abs().sort_values(ascending=False).index).head(3)
    print("largest playoff-odds gaps:",
          [(r.abbrev, round(r.diff_playoffs, 1)) for r in worst.itertuples()])


if __name__ == "__main__":
    main()
