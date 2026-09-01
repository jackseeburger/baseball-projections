"""Team metadata (league / division membership) from the Stats API."""
from __future__ import annotations

import pandas as pd

from src.data.mlb_stats_api import _get

AL, NL = 103, 104
DIVISION_NAMES = {
    200: "AL West", 201: "AL East", 202: "AL Central",
    203: "NL West", 204: "NL East", 205: "NL Central",
}


def fetch_teams(season: int) -> pd.DataFrame:
    """One row per MLB team: team_id, abbrev, name, league_id, division_id."""
    rows = []
    for t in _get("teams", sportId=1, season=season)["teams"]:
        rows.append({
            "team_id": t["id"],
            "abbrev": t["abbreviation"],
            "name": t["name"],
            "league_id": t["league"]["id"],
            "division_id": t["division"]["id"],
        })
    df = pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)
    assert len(df) == 30, f"expected 30 teams, got {len(df)}"
    return df
