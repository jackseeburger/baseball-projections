"""Schema guard for the committed season-level fixture.

The parquet is small (≈180 KB) and committed so CI and fresh cloud sessions
can run real backtests without network. Regenerate with
src/data/mlb_stats_api.build_seasons_table.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

FIXTURE = Path(__file__).parent.parent.parent / "data/parquet/hitter_seasons_api.parquet"

REQUIRED = {"batter", "season", "age", "pa", "ab", "k", "bb", "hr",
            "xb_points", "bip", "hits_in_play"}


@pytest.mark.skipif(not FIXTURE.exists(), reason="seasons fixture not present")
def test_seasons_fixture_schema_and_sanity():
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(FIXTURE)
    assert REQUIRED <= set(df.columns)
    assert df["season"].min() <= 2015 and df["season"].max() >= 2025
    # Derived columns stay internally consistent.
    full = df[df["pa"] >= 400]
    assert ((full["bip"] == full["ab"] - full["k"] - full["hr"] + full["sf"]).all())
    assert (full["hits_in_play"] <= full["bip"]).all()
    league_k = full["k"].sum() / full["pa"].sum()
    assert 0.15 < league_k < 0.30
