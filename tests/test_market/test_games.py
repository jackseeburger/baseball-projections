"""game_pk mapping against a synthetic schedule."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market.games import assign_game_pk
from src.market.schema import empty_record

LAD, WSH, NYY, TB = 119, 120, 147, 139


def rec(**kw):
    r = empty_record()
    r.update({"ts": "t", "venue": "kalshi", "market_id": kw.pop("market_id", "x"),
              "market_type": "moneyline"})
    r.update(kw)
    return r


def schedule():
    return pd.DataFrame([
        {"game_pk": 1, "date": "2026-09-04", "home_id": LAD, "away_id": WSH,
         "game_datetime": "2026-09-05T02:10:00Z"},
        {"game_pk": 2, "date": "2026-09-22", "home_id": NYY, "away_id": TB,
         "game_datetime": "2026-09-22T17:05:00Z"},
        {"game_pk": 3, "date": "2026-09-22", "home_id": NYY, "away_id": TB,
         "game_datetime": "2026-09-22T23:05:00Z"},   # doubleheader nightcap
    ])


def test_exact_match():
    r = rec(game_date="2026-09-04", home_id=LAD, away_id=WSH)
    stats = assign_game_pk([r], schedule())
    assert r["game_pk"] == 1 and stats["mapped"] == 1 and stats["swapped"] == 0


def test_swapped_home_away_is_corrected():
    r = rec(game_date="2026-09-04", home_id=WSH, away_id=LAD)
    stats = assign_game_pk([r], schedule())
    assert r["game_pk"] == 1 and stats["swapped"] == 1
    assert (r["home_id"], r["away_id"]) == (LAD, WSH)


def test_day_shift_for_late_games():
    r = rec(game_date="2026-09-05", home_id=LAD, away_id=WSH)
    stats = assign_game_pk([r], schedule())
    assert r["game_pk"] == 1 and stats["day_shift"] == 1


def test_doubleheader_picks_closest_start():
    night = rec(game_date="2026-09-22", home_id=NYY, away_id=TB,
                game_start="2026-09-22T23:00:00+00:00")
    day = rec(game_date="2026-09-22", home_id=NYY, away_id=TB,
              game_start="2026-09-22T17:05:00+00:00")
    assign_game_pk([night, day], schedule())
    assert night["game_pk"] == 3 and day["game_pk"] == 2


def test_unmapped_and_non_game_markets():
    fut = rec(market_type="futures_ws", team_id=LAD)
    ghost = rec(game_date="2026-09-10", home_id=LAD, away_id=WSH)
    stats = assign_game_pk([fut, ghost], schedule())
    assert fut["game_pk"] is None and ghost["game_pk"] is None
    assert stats == {"game_markets": 1, "mapped": 0, "swapped": 0, "day_shift": 0, "unmapped": 1}
