"""The staleness watchdog: budgets, exit codes, and where the clock comes from.

The one that matters most is `test_embedded_timestamp_beats_mtime`. A fresh
`git clone` stamps every file with the checkout time, so on a CI runner mtime
says "seconds old" for data written a week ago — a watchdog that trusts mtime
passes forever and the outage stays invisible, which is the failure this whole
thing exists to prevent.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_freshness", ROOT / "scripts/check_freshness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fresh = _load()


def write_tree(root: Path, ages_hours: dict[str, float]) -> None:
    """Build a checkout-shaped tree whose artifacts are `ages_hours` old.

    Keys are artifact names from the module's table; anything left out is not
    written at all, so its directory is absent.
    """
    by_name = {a.name: a for a in fresh.ARTIFACTS}
    for name, age in ages_hours.items():
        artifact = by_name[name]
        stamp = NOW - timedelta(hours=age)
        target = root / artifact.path
        if artifact.kind == "json_field":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({artifact.field: stamp.isoformat()}))
        else:
            target.mkdir(parents=True, exist_ok=True)
            filename = stamp.strftime(artifact.ts_format) + ".jsonl.gz"
            (target / filename).write_bytes(b"")


ALL_FRESH = {
    "playoff odds board": 5,
    "rest-of-season projections": 5,
    "accuracy page data": 5,
    "market snapshot archive": 3,
    "market latest.json": 3,
}


def status_by_name(results) -> dict[str, str]:
    return {r.artifact.name: r.status for r in results}


def test_fresh_tree_passes(tmp_path):
    write_tree(tmp_path, ALL_FRESH)
    results = fresh.check_all(tmp_path, NOW)
    assert set(status_by_name(results).values()) == {"OK"}
    assert fresh.main(["--root", str(tmp_path), "--now", NOW.isoformat()]) == 0


def test_artifact_just_inside_its_budget_is_ok(tmp_path):
    # 35h59m against the 36h nightly budget, 19h59m against the 20h market
    # budget: the boundary is not the alarm.
    write_tree(tmp_path, {**ALL_FRESH,
                          "playoff odds board": 35.98,
                          "market snapshot archive": 19.98})
    assert set(status_by_name(fresh.check_all(tmp_path, NOW)).values()) == {"OK"}


def test_missed_day_of_nightly_odds_is_stale(tmp_path):
    # A single dropped nightly run puts the board ~48h behind.
    write_tree(tmp_path, {**ALL_FRESH, "playoff odds board": 48})
    results = fresh.check_all(tmp_path, NOW)
    assert status_by_name(results)["playoff odds board"] == "STALE"
    assert [r.artifact.name for r in results if r.failed] == ["playoff odds board"]
    assert fresh.main(["--root", str(tmp_path), "--now", NOW.isoformat()]) == 1


def test_one_dropped_market_slot_does_not_cry_wolf(tmp_path):
    # 23:00 dropped, so the newest snapshot is the 16:30 one and the gap to
    # the next 10:00 slot is 17h30m — routine, not an outage.
    write_tree(tmp_path, {**ALL_FRESH, "market snapshot archive": 17.5})
    assert status_by_name(fresh.check_all(tmp_path, NOW))["market snapshot archive"] == "OK"


def test_missed_day_of_market_snapshots_is_stale(tmp_path):
    write_tree(tmp_path, {**ALL_FRESH, "market snapshot archive": 24})
    results = fresh.check_all(tmp_path, NOW)
    assert status_by_name(results)["market snapshot archive"] == "STALE"
    assert fresh.main(["--root", str(tmp_path), "--now", NOW.isoformat()]) == 1


def test_embedded_timestamp_beats_mtime(tmp_path):
    """Old data in a file touched a second ago still reports STALE."""
    write_tree(tmp_path, {**ALL_FRESH, "playoff odds board": 100})
    board = tmp_path / "public/data/playoff_odds/latest.json"
    os.utime(board, (time.time(), time.time()))
    assert board.stat().st_mtime == pytest.approx(time.time(), abs=5)

    result = next(r for r in fresh.check_all(tmp_path, NOW)
                  if r.artifact.name == "playoff odds board")
    assert result.status == "STALE"
    assert result.age_hours == pytest.approx(100)


def test_newest_snapshot_comes_from_the_name_not_the_mtime(tmp_path):
    """Same trap for the snapshot directory, which is aged by filename."""
    write_tree(tmp_path, ALL_FRESH)
    snapshots = tmp_path / "data/market/snapshots"
    for entry in snapshots.iterdir():
        entry.unlink()
    old = snapshots / "2026-09-01T1000Z.jsonl.gz"   # 50h before NOW
    new = snapshots / "2026-09-03T1000Z.jsonl.gz"   # 2h before NOW
    old.write_bytes(b"")
    new.write_bytes(b"")
    # The stale file is the one that looks newest on disk.
    os.utime(new, (time.time() - 86400, time.time() - 86400))
    os.utime(old, (time.time(), time.time()))

    result = next(r for r in fresh.check_all(tmp_path, NOW)
                  if r.artifact.name == "market snapshot archive")
    assert result.status == "OK"
    assert result.age_hours == pytest.approx(2)
    assert "2026-09-03T1000Z" in result.detail


def test_absent_directory_reports_absent_and_fails_when_required(tmp_path):
    missing = {k: v for k, v in ALL_FRESH.items() if k != "market snapshot archive"}
    write_tree(tmp_path, missing)
    result = next(r for r in fresh.check_all(tmp_path, NOW)
                  if r.artifact.name == "market snapshot archive")
    assert result.status == "ABSENT"
    assert result.detail == "directory does not exist"
    assert result.age_hours is None
    # Every artifact in the table exists in the repo today, so all are required
    # and an absent one is a real regression.
    assert result.failed
    assert fresh.main(["--root", str(tmp_path), "--now", NOW.isoformat()]) == 1


def test_absent_optional_artifact_is_skipped_not_failed(tmp_path):
    optional = fresh.Artifact(
        name="future R2 snapshot mirror",
        path="public/data/not_built_yet/latest.json",
        kind="json_field",
        field="generated_at",
        budget_hours=36,
        required=False,
        workflow="not-shipped-yet.yml",
    )
    result = fresh.check_artifact(optional, tmp_path, NOW)
    assert result.status == "ABSENT"
    assert not result.failed


def test_empty_directory_is_missing_not_absent(tmp_path):
    """The directory is there and the job left nothing in it — always a failure."""
    write_tree(tmp_path, ALL_FRESH)
    snapshots = tmp_path / "data/market/snapshots"
    for entry in snapshots.iterdir():
        entry.unlink()
    result = next(r for r in fresh.check_all(tmp_path, NOW)
                  if r.artifact.name == "market snapshot archive")
    assert result.status == "MISSING"
    assert result.failed


def test_unreadable_timestamp_reports_error(tmp_path):
    write_tree(tmp_path, ALL_FRESH)
    board = tmp_path / "public/data/playoff_odds/latest.json"
    board.write_text(json.dumps({"season": 2026}))
    result = next(r for r in fresh.check_all(tmp_path, NOW)
                  if r.artifact.name == "playoff odds board")
    assert result.status == "ERROR"
    assert result.failed


def test_json_output_is_machine_readable(tmp_path, capsys):
    write_tree(tmp_path, {**ALL_FRESH, "playoff odds board": 48})
    code = fresh.main(["--root", str(tmp_path), "--now", NOW.isoformat(), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and payload["ok"] is False
    assert payload["now"] == NOW.isoformat()
    board = next(a for a in payload["artifacts"] if a["name"] == "playoff odds board")
    assert board["status"] == "STALE"
    assert board["age_hours"] == pytest.approx(48)
    assert board["budget_hours"] == 36
    assert board["workflow"] == "nightly-odds.yml"


def test_report_names_the_workflow_to_go_look_at(tmp_path, capsys):
    write_tree(tmp_path, {**ALL_FRESH, "market snapshot archive": 30})
    fresh.main(["--root", str(tmp_path), "--now", NOW.isoformat()])
    out = capsys.readouterr().out
    assert "STALE" in out and "market-snapshot.yml" in out
    assert "1 of 5 artifacts out of budget" in out


def test_naive_and_zulu_timestamps_are_read_as_utc():
    assert fresh.parse_time("2026-09-03T00:58:55Z") == \
        datetime(2026, 9, 3, 0, 58, 55, tzinfo=timezone.utc)
    assert fresh.parse_time("2026-09-03T00:58:55") == \
        datetime(2026, 9, 3, 0, 58, 55, tzinfo=timezone.utc)
    assert fresh.parse_time("2026-09-03T02:58:55+02:00") == \
        datetime(2026, 9, 3, 0, 58, 55, tzinfo=timezone.utc)


def test_budgets_are_under_a_day_where_the_job_runs_more_than_daily():
    """A budget of 24h or more could never catch a whole missed day."""
    for artifact in fresh.ARTIFACTS:
        if artifact.workflow == "market-snapshot.yml":
            assert artifact.budget_hours < 24
        if artifact.workflow == "nightly-odds.yml":
            # Daily job: a missed day is ~48h, so the budget must sit between
            # one clean day and two.
            assert 24 < artifact.budget_hours < 48


def test_table_matches_this_checkout():
    """Every artifact in the table has a directory in the repo right now.

    Freshness of the real files is not asserted — that is the watchdog's job in
    production, not CI's — but a path that has been renamed or deleted should
    fail here rather than page someone at 05:53 UTC.
    """
    for artifact in fresh.ARTIFACTS:
        target = ROOT / artifact.path
        directory = target if artifact.kind == "newest_filename" else target.parent
        assert directory.is_dir(), f"{artifact.name}: {directory} is gone"
