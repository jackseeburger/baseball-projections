"""Savant fetch logic — chunking, truncation splitting, dedup. No network."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import statcast_savant as ss


class FakeSession:
    """Returns a fixed number of rows per requested window."""

    def __init__(self, rows_for, unique=True):
        self.rows_for = rows_for
        self.unique = unique      # False → every window returns the same pitches
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        start, end = params["game_date_gt"], params["game_date_lt"]
        self.calls.append((start, end))
        n = self.rows_for(start, end)
        offset = hash(start) % 100_000 if self.unique else 0
        df = pd.DataFrame({
            "game_pk": [offset + i for i in range(n)],
            "at_bat_number": [1] * n, "pitch_number": [1] * n,
            "game_date": [start] * n,
        })
        return FakeResponse(df.to_csv(index=False))


class FakeResponse:
    def __init__(self, text, status=200):
        self.content = text.encode()
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(self.status)


def test_params_cover_the_window():
    p = ss._params(date(2026, 4, 1), date(2026, 4, 3), 2026)
    assert p["game_date_gt"] == "2026-04-01" and p["game_date_lt"] == "2026-04-03"
    assert p["hfSea"] == "2026|" and p["type"] == "details"


def test_fetch_season_chunks_the_range():
    sess = FakeSession(lambda s, e: 10)
    df = ss.fetch_season(2026, date(2026, 4, 1), date(2026, 4, 6), chunk_days=3, session=sess)
    assert sess.calls == [("2026-04-01", "2026-04-03"), ("2026-04-04", "2026-04-06")]
    assert len(df) == 20


def test_truncated_chunk_is_split():
    # The first window looks capped; its halves do not.
    def rows(s, e):
        return ss.ROW_CAP if (s, e) == ("2026-04-01", "2026-04-04") else 5

    sess = FakeSession(rows)
    ss.fetch_season(2026, date(2026, 4, 1), date(2026, 4, 4), chunk_days=4, session=sess)
    assert ("2026-04-01", "2026-04-04") in sess.calls
    assert len(sess.calls) > 1, "a capped window must be split"


def test_duplicate_pitches_dropped_at_chunk_boundaries():
    # A game straddling a chunk boundary is returned by both requests.
    sess = FakeSession(lambda s, e: 4, unique=False)
    df = ss.fetch_season(2026, date(2026, 4, 1), date(2026, 4, 4), chunk_days=2, session=sess)
    assert len(sess.calls) == 2
    assert len(df) == 4, "the same pitch must not be counted twice"


def test_html_error_page_is_rejected():
    class Bad(FakeSession):
        def get(self, *a, **k):
            return FakeResponse("<!DOCTYPE html><html>error</html>")

    with pytest.raises(ValueError, match="non-CSV"):
        ss.fetch_range(date(2026, 4, 1), date(2026, 4, 1), 2026, session=Bad(lambda s, e: 0))


def test_backwards_range_rejected():
    with pytest.raises(ValueError):
        ss.fetch_season(2026, date(2026, 5, 1), date(2026, 4, 1))
