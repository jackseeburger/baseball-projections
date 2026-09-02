"""Schema tests for the live rest-of-season projection the site loads.

`public/data/projections/latest.json` is the only thing standing between
`src/projections/ros.py` and the browser. The page renders whatever it is
handed, so the contract that matters is the shape: the framing sentence that
names the model, the `stale` flag with a reason when true, and one row per
hitter whose numbers are finite JSON — never a bare NaN, which is invalid JSON
and blanks the page.

The document is built from a synthetic projection frame here; the committed
file is checked against the same contract at the bottom, and skipped in a
checkout that has never run the builder.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "public/data/projections/latest.json"

RATE_COLUMNS = [f"{stat}_rate_{arm}"
                for stat in ("k", "bb", "hr", "babip", "iso")
                for arm in ("marcel", "marcel_preseason", "bayes")]
REQUIRED_PLAYER_KEYS = (["batter", "name", "team_id", "team_abbrev", "as_of", "pa_ros"]
                        + RATE_COLUMNS + ["k_ros", "bb_ros", "hr_ros", "woba_ros"])


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_ros_projections", ROOT / "scripts/build_ros_projections.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_builder()


def make_projections() -> pd.DataFrame:
    """Two hitters in the frame `build_ros_projections` returns, one of whom has
    no preseason Bayesian projection (a rookie) — the null path the page must
    survive."""
    rows = []
    for i, (batter, name, woba) in enumerate([(1, "First Hitter", 0.360),
                                              (2, "Second Hitter", 0.310)]):
        row = {"batter": batter, "name": name, "team_id": 100, "team_abbrev": "NYY",
               "as_of": "2026-09-02", "pa_ros": 100.0 - 10 * i,
               "k_ros": 22.0, "bb_ros": 9.0, "hr_ros": 3.0, "woba_ros": woba}
        for column in RATE_COLUMNS:
            row[column] = None if (batter == 2 and column.endswith("_bayes")) else 0.2
        rows.append(row)
    return pd.DataFrame(rows)[REQUIRED_PLAYER_KEYS]


@pytest.fixture
def doc():
    return build.to_document(make_projections(), "2026-09-02",
                             git_sha="0123456789abcdef", season_end="2026-09-27")


def walk_numbers(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


def assert_contract(doc: dict) -> None:
    """Everything the page reads, checked in one place."""
    for key in ("as_of", "n_hitters", "method", "framing", "stale", "players"):
        assert key in doc, key
    assert isinstance(doc["stale"], bool)
    if doc["stale"]:
        assert doc["stale_reason"], "a stale projection must say why"
    assert doc["n_hitters"] == len(doc["players"])
    assert isinstance(doc["players"], list)
    for player in doc["players"]:
        for key in REQUIRED_PLAYER_KEYS:
            assert key in player, key
        assert player["pa_ros"] is not None and player["pa_ros"] > 0
    for path, value in walk_numbers(doc):
        assert not math.isnan(value) and not math.isinf(value), path


# ─── document shape ──────────────────────────────────────────────

def test_the_document_matches_the_contract(doc):
    assert_contract(doc)


def test_the_framing_names_the_model_and_the_date_it_saw(doc):
    assert doc["framing"] == ("Live projection: Marcel with 2026 through "
                              "2026-09-01. Preseason Bayesian shown for "
                              "comparison — see Model Accuracy.")
    assert doc["through"] == "2026-09-01", "the as-of day itself is not in the data"


def test_the_method_says_what_the_model_is(doc):
    assert "Marcel" in doc["method"]
    assert "plate appearances" in doc["method"]


def test_the_arms_are_labelled_with_the_live_one_marked(doc):
    arms = {a["key"]: a for a in doc["arms"]}
    assert set(arms) == {"marcel", "bayes", "marcel_preseason"}
    assert arms["marcel"]["is_live"] is True
    assert arms["bayes"]["is_live"] is False
    assert all(a["label"] and a["note"] for a in doc["arms"])


def test_the_woba_weights_travel_with_the_numbers(doc):
    """The page shows a wOBA; the file has to say which wOBA."""
    assert doc["woba"]["weights"]["hr"] == 2.015
    assert doc["woba"]["triples_per_double"] == 0.12


def test_a_missing_comparison_rate_is_null_not_nan(doc):
    rookie = next(p for p in doc["players"] if p["batter"] == 2)
    assert rookie["k_rate_bayes"] is None
    assert rookie["k_rate_marcel"] is not None


def test_the_numbers_are_rounded_for_the_archive():
    """A dated snapshot is written every night and never rewritten."""
    frame = make_projections()
    frame.loc[0, "woba_ros"] = 0.36123456789
    frame.loc[0, "pa_ros"] = 100.123456789
    doc = build.to_document(frame, "2026-09-02", git_sha="x")
    player = doc["players"][0]
    assert player["woba_ros"] == pytest.approx(0.36123)
    assert player["pa_ros"] == pytest.approx(100.12)


# ─── the stale path ──────────────────────────────────────────────

def test_a_stale_document_carries_yesterdays_players_and_says_why(doc):
    stale = build.stale_document(doc, "2026-09-01.json",
                                 "R2_ACCESS_KEY_ID not set on this runner",
                                 "2026-09-03")
    assert stale["stale"] is True
    assert "R2_ACCESS_KEY_ID" in stale["stale_reason"]
    assert "carried over from 2026-09-01.json" in stale["stale_reason"]
    assert stale["players"] == doc["players"]
    assert stale["as_of"] == doc["as_of"], "the as-of date is the data's, not today's"
    assert stale["requested_as_of"] == "2026-09-03"
    assert_contract(stale)


def test_an_empty_document_still_has_the_shape():
    empty = build.empty_document("no PA parquet and no R2", "2026-09-02")
    assert empty["players"] == [] and empty["n_hitters"] == 0
    assert "No previous projection" in empty["stale_reason"]
    assert_contract(empty)


def test_a_build_with_no_inputs_falls_back_instead_of_raising(tmp_path):
    """The nightly job must not fail because R2 was unreachable."""
    doc = build.build(
        "2026-09-02", out_dir=tmp_path,
        seasons_path=tmp_path / "absent.parquet", pa_dir=tmp_path,
        projections_dir=tmp_path)
    assert doc["stale"] is True and doc["stale_reason"]
    assert_contract(doc)


def test_the_dated_snapshot_is_never_overwritten(tmp_path, doc):
    build.write_document(doc, tmp_path)
    dated = tmp_path / "2026-09-02.json"
    first = json.loads(dated.read_text())
    build.write_document(dict(doc, git_sha="changed"), tmp_path)
    assert json.loads(dated.read_text()) == first
    assert json.loads((tmp_path / "latest.json").read_text())["git_sha"] == "changed"


# ─── the file the site actually loads ────────────────────────────

def test_committed_latest_json_matches_the_contract():
    if not LATEST.exists():                    # not built in this checkout
        pytest.skip("public/data/projections/latest.json not present")
    doc = json.loads(LATEST.read_text())       # json.loads rejects bare NaN
    assert_contract(doc)
    assert doc["n_hitters"] > 0, "a committed projection with no hitters is a bug"


def test_committed_projection_is_internally_consistent():
    if not LATEST.exists():
        pytest.skip("public/data/projections/latest.json not present")
    doc = json.loads(LATEST.read_text())
    players = pd.DataFrame(doc["players"])
    # Counts are the rate times the playing time, to the rounding written out.
    for stat in ("k", "bb", "hr"):
        expected = players[f"{stat}_rate_marcel"] * players["pa_ros"]
        assert (players[f"{stat}_ros"] - expected).abs().max() < 0.01, stat
    # Nobody projects outside the range of a plate appearance.
    assert players["k_rate_marcel"].between(0, 1).all()
    assert players["woba_ros"].between(0, 1).all()
    assert players["batter"].is_unique
