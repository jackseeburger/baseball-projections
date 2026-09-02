"""Schema tests for the Model Accuracy page's data (station H).

The page renders nothing it is not handed, so the contract that matters is the
shape of `public/data/accuracy/latest.json`: every section present, every
`stale` flag a real boolean with a reason when true, every metric a finite
number or null — never a bare NaN, which is not valid JSON and would blank the
page in the browser.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/accuracy"
SECTIONS = ("components", "ros_backtest", "game_odds", "playoff_odds_control")


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_accuracy_json", ROOT / "scripts/build_accuracy_json.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_builder()


@pytest.fixture
def doc(tmp_path):
    """A full document built from fixture score files (no network, no scripts)."""
    return build.build_document(
        out_dir=tmp_path,
        components_json=FIXTURES / "components_scores.json",
        game_odds_json=FIXTURES / "game_odds_market.json",
        ros_json=FIXTURES / "ros_backtest.json",
        git_sha="0123456789abcdef",
    )


def walk_numbers(obj, path="$"):
    """Every leaf number in the document, with a path for the failure message."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield path, obj


# ─── document-level schema ───────────────────────────────────────

def test_every_section_is_present(doc):
    assert set(doc["sections"]) == set(SECTIONS)


def test_top_level_fields(doc):
    for key in ("generated_at", "as_of", "git_sha", "title", "subtitle", "sections", "meta"):
        assert key in doc, key
    assert doc["git_sha"] == "0123456789abcdef"
    assert doc["meta"]["generated_by"] == "scripts/build_accuracy_json.py"


def test_sections_carry_the_contract_fields(doc):
    for name, section in doc["sections"].items():
        assert isinstance(section["stale"], bool), name
        assert section["as_of"], name
        assert section["n"] is None or isinstance(section["n"], int), name
        assert section["framing"], f"{name} must state how to read it"
        assert section["source"], f"{name} must name where it came from"
        assert section["rows"] and section["columns"], name
        if section["stale"]:
            assert section["stale_reason"], f"{name} is stale without a reason"


def test_every_number_is_finite_and_json_safe(doc):
    for path, value in walk_numbers(doc):
        assert not math.isnan(value) and not math.isinf(value), path
    # json.dumps with allow_nan=False is what a browser's JSON.parse enforces.
    reparsed = json.loads(json.dumps(doc, allow_nan=False))
    assert reparsed == doc


def test_every_row_metric_is_numeric_or_null(doc):
    for name, section in doc["sections"].items():
        keys = {c["key"] for c in section["columns"] if c["type"] != "text"}
        for row in section["rows"]:
            for key in keys:
                value = row.get(key, (row.get("metrics") or {}).get(key))
                assert value is None or isinstance(value, (int, float)), f"{name}.{key}"


def test_meta_status_mirrors_the_sections(doc):
    status = {s["section"]: s for s in doc["meta"]["status"]}
    assert set(status) == set(SECTIONS)
    for name, section in doc["sections"].items():
        assert status[name]["fresh"] is (not section["stale"])
        assert status[name]["as_of"] == section["as_of"]


def test_glossary_explains_the_metrics_and_the_gate(doc):
    terms = [g["term"] for g in doc["meta"]["glossary"]]
    assert {"Brier score", "Log loss", "MAE", "The gate rule"} <= set(terms)
    assert all(g["text"] for g in doc["meta"]["glossary"])


# ─── section content ─────────────────────────────────────────────

def test_components_ranks_ours_behind_the_public_systems(doc):
    rows = doc["sections"]["components"]["rows"]
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    by_model = {r["model"]: r for r in rows}
    assert by_model["bayes_preseason"]["is_ours"] is True
    assert by_model["bayes_preseason"]["rank"] > by_model["marcel"]["rank"], (
        "the fixture has us behind Marcel; the page must say so")
    assert doc["sections"]["components"]["n"] == 263


def test_components_keeps_a_missing_metric_as_null_not_nan():
    payload = json.loads((FIXTURES / "components_scores.json").read_text())
    payload["scores"][0]["mae"] = float("nan")
    section = build.section_components(payload)
    row = next(r for r in section["rows"] if r["model"] == "depth_charts")
    assert row["metrics"]["k_rate"] is None


def test_game_odds_marks_the_market_rows(doc):
    section = doc["sections"]["game_odds"]
    assert section["stale"] is False
    market = [r for r in section["rows"] if r["is_market"]]
    assert {r["model"] for r in market} == {"kalshi_close", "polymarket_close"}
    assert market[0]["rank"] == 1, "the market is the bar and currently leads"
    assert section["n"] == 756


def test_game_odds_without_market_is_stale_with_a_reason():
    payload = json.loads((FIXTURES / "game_odds_no_market.json").read_text())
    section = build.section_game_odds(payload, "R2_ACCESS_KEY_ID not set on this runner")
    assert section["stale"] is True
    assert "R2_ACCESS_KEY_ID" in section["stale_reason"]
    assert not any(r["is_market"] for r in section["rows"])


def test_game_odds_lists_each_price_once(doc):
    """The backtester emits a market close once per model subset it scores.
    The same Kalshi price twice reads as two contenders tying with the bar."""
    payload = json.loads((FIXTURES / "game_odds_market.json").read_text())
    doubled = dict(payload, scores=payload["scores"] + payload["scores"])
    section = build.section_game_odds(doubled, None)
    models = [r["model"] for r in section["rows"]]
    assert len(models) == len(set(models))
    assert models == [r["model"] for r in doc["sections"]["game_odds"]["rows"]]
    assert [r["rank"] for r in section["rows"]] == list(range(1, len(models) + 1))


def test_control_section_parses_the_coin_flip_table(doc):
    section = doc["sections"]["playoff_odds_control"]
    models = {r["model"] for r in section["rows"]}
    assert models == {"coin_flip", "ours"}
    control = next(r for r in section["rows"] if r["model"] == "coin_flip")
    assert control["is_control"] is True
    assert all(isinstance(v, float) for v in control["metrics"].values())
    assert section["stale"] is True and section["stale_reason"]


def test_control_section_reads_a_table_out_of_markdown():
    md = ("# Doc\n\n**As of Sept 1, 2026**\n\n"
          "## 2b. The control\n\nRun with 8,000 sims each:\n\n"
          "| Strength model | P(playoffs) |\n|---|---|\n"
          "| **No model — every team is a .500 coin flip** | 1.94 |\n"
          "| Ours | 1.63 |\n\nProse after the table.\n")
    section = build.section_control(md, 30)
    assert section["as_of"] == "2026-09-01"
    assert section["n"] == 30
    assert [r["metrics"]["P(playoffs)"] for r in section["rows"]] == [1.94, 1.63]


def test_ros_section_ranks_the_live_arm_ahead_of_our_preseason_model(doc):
    """The section exists to show why the site swapped models; if the fixture
    stopped saying that, the swap is no longer justified."""
    section = doc["sections"]["ros_backtest"]
    assert section["stale"] is False
    assert section["live_arm"] == "marcel"
    live = next(r for r in section["rows"] if r["is_production"])
    assert live["is_production"] is True
    by_cutoff = {}
    for row in section["rows"]:
        by_cutoff.setdefault(row["cutoff_date"], {})[row["model"]] = row["metrics"]
    for cutoff, arms in by_cutoff.items():
        assert arms["marcel"]["k_rate"] < arms["bayes_preseason"]["k_rate"], cutoff
        assert arms["marcel"]["k_rate"] < arms["marcel_preseason"]["k_rate"], cutoff


def test_ros_section_marks_the_winner_inside_its_own_cutoff(doc):
    """MAE rises with every cutoff, so a column-wide minimum would always be
    the May 1 row. The builder marks the best arm per cutoff instead, and tells
    the page not to rank the column itself."""
    section = doc["sections"]["ros_backtest"]
    assert section["highlight_best"] is False
    for cutoff in {r["cutoff_date"] for r in section["rows"]}:
        rows = [r for r in section["rows"] if r["cutoff_date"] == cutoff]
        for component in ("k_rate", "bb_rate", "hr_rate", "iso"):
            values = [(r["metrics"][component], r["model"]) for r in rows]
            winner = min(values)[1]
            marked = [r["model"] for r in rows if component in r["best"]]
            assert marked == [winner], f"{cutoff} {component}"


def test_ros_section_framing_is_counted_not_asserted(doc):
    """The claim on the page is recomputed from the table, so it flips on its
    own if the result ever does."""
    section = doc["sections"]["ros_backtest"]
    assert "11 of 12" in section["framing"]
    assert "component-cutoff cells" in section["framing"]


def test_ros_section_is_stale_when_the_pa_parquet_and_r2_are_both_missing(
        tmp_path, monkeypatch):
    for var in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    note = build.ros_input_note(tmp_path / "absent.parquet")
    assert note and "R2_ACCESS_KEY_ID" in note


def test_ros_input_note_is_silent_when_the_parquet_is_there(tmp_path):
    path = tmp_path / "pa_outcomes_2026.parquet"
    path.write_bytes(b"not really a parquet, but present")
    assert build.ros_input_note(path) is None


# ─── staleness and archiving ─────────────────────────────────────

def test_a_skipped_section_falls_back_to_the_previous_snapshot(tmp_path, doc):
    build.write_document(doc, tmp_path)
    later = build.build_document(
        out_dir=tmp_path, skip=("game_odds",),
        components_json=FIXTURES / "components_scores.json",
        game_odds_json=FIXTURES / "game_odds_market.json",
        ros_json=FIXTURES / "ros_backtest.json",
        git_sha="deadbeef")
    section = later["sections"]["game_odds"]
    assert section["stale"] is True
    assert "carried over" in section["stale_reason"]
    assert section["rows"] == doc["sections"]["game_odds"]["rows"]


def test_a_missing_section_with_no_history_still_has_the_shape(tmp_path):
    later = build.build_document(
        out_dir=tmp_path, skip=("components", "ros_backtest", "game_odds"),
        components_json=None, game_odds_json=None, ros_json=None,
        git_sha="deadbeef")
    for name in ("components", "ros_backtest", "game_odds"):
        section = later["sections"][name]
        assert section["stale"] is True
        assert "No previous snapshot" in section["stale_reason"]
        assert section["rows"] == []


def test_dated_snapshot_is_never_overwritten(tmp_path, doc):
    build.write_document(doc, tmp_path)
    dated = tmp_path / f"{doc['as_of']}.json"
    first = json.loads(dated.read_text())
    doc2 = dict(doc, git_sha="changed")
    build.write_document(doc2, tmp_path)
    assert json.loads(dated.read_text()) == first
    assert json.loads((tmp_path / "latest.json").read_text())["git_sha"] == "changed"


# ─── the file the site actually loads ────────────────────────────

def test_committed_latest_json_matches_the_contract():
    path = ROOT / "public/data/accuracy/latest.json"
    if not path.exists():                      # not built in this checkout
        pytest.skip("public/data/accuracy/latest.json not present")
    doc = json.loads(path.read_text())         # json.loads rejects bare NaN
    assert set(doc["sections"]) == set(SECTIONS)
    for name, section in doc["sections"].items():
        assert isinstance(section["stale"], bool), name
        assert section["framing"], name
    for path_, value in walk_numbers(doc):
        assert not math.isnan(value) and not math.isinf(value), path_
