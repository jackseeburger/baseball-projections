"""The `--json-out` document from scripts/backtest_game_odds.py.

One row per model. The backtest assembles the scored model list in pieces —
the ballast sweep, the station models, then one column per market venue — and
the venue columns belong to two of those pieces at once (they are appended to
`models` when `--market` is joined, and kept separately for the payload's
`market_models`). Adding the pieces together emitted every venue twice, which
the site then had to work around when it drew the table. These tests pin the
emission itself: assembling the list twice over cannot produce a repeated row.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _backtest_module():
    spec = importlib.util.spec_from_file_location(
        "backtest_game_odds", ROOT / "scripts/backtest_game_odds.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MODELS = ["home_constant", "pythag_60", "pythag_C_sp"]
VENUES = ["kalshi_close", "polymarket_close"]


@pytest.fixture(scope="module")
def bt():
    return _backtest_module()


@pytest.fixture
def preds():
    """A small walk-forward frame with every scored column present."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "game_pk": np.arange(n),
        "date": pd.date_range("2026-05-01", periods=n, freq="D").astype(str),
        "home_win": rng.random(n) < 0.54,
    })
    for i, m in enumerate(MODELS + VENUES):
        df[m] = np.clip(0.5 + 0.02 * (i + 1) + 0.05 * rng.standard_normal(n), 0.05, 0.95)
    return df


def _payload(bt, preds, models, venues):
    return bt.score_payload(preds, models, venues, generated_at="2026-09-02T00:00:00Z",
                            season=2026, min_games=20,
                            market_file="market_closes_2026.parquet",
                            sp_fallback_games=0, sp_no_history_slots=0)


def test_model_names_keeps_first_position_and_drops_repeats(bt):
    assert bt.model_names(["a", "b"], ["b", "c"], ["a"]) == ["a", "b", "c"]
    assert bt.model_names([], []) == []


def test_json_output_names_every_model_once(bt, preds, tmp_path):
    """The venues are in `models` *and* in `venues`; the file must not say so twice."""
    joined = bt.model_names(MODELS, VENUES)          # what main() passes as `models`
    payload = _payload(bt, preds, joined, VENUES)
    out = tmp_path / "game_odds.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")

    doc = json.loads(out.read_text())
    names = [row["model"] for row in doc["scores"]]
    assert len(names) == len(set(names)), f"repeated model rows: {names}"
    assert set(names) == set(MODELS) | set(VENUES)
    assert doc["market_models"] == VENUES
    assert doc["n_games"] == len(preds)


def test_venue_rows_survive_when_the_lists_are_assembled_naively(bt, preds, tmp_path):
    """Deduping must not drop a venue that only the second list contributes."""
    payload = _payload(bt, preds, MODELS, VENUES)
    names = [row["model"] for row in payload["scores"]]
    assert len(names) == len(set(names))
    assert set(VENUES) <= set(names)


def test_dedupe_changes_no_number(bt, preds):
    """Every surviving row is the row the duplicated list would have produced."""
    doubled = bt.score(preds, MODELS + VENUES + VENUES).drop_duplicates("model")
    once = bt.score(preds, bt.model_names(MODELS, VENUES))
    pd.testing.assert_frame_equal(
        doubled.sort_values("model").reset_index(drop=True),
        once.sort_values("model").reset_index(drop=True))
