"""The schedule watchdog: cron arithmetic, due slots, and mocked Actions runs.

Three tests carry the design and the rest support them:

* `test_the_outage_of_2026_09_03_would_have_been_caught` — the fixture is the
  real evidence from that morning, and the check has to call it a drop.
* `test_statcast_and_modal_refit_are_pending_not_dropped` — the same fixture,
  the mistake we nearly made by hand: two workflows had never had a slot come
  due, and "never due" must not read as "dropped".
* `test_merely_late_does_not_fire` — a 4h25m-late run is the worst delay this
  repo has measured, and it is not an outage. An alarm that fires here is an
  alarm that gets muted.

Nothing here touches the network: the API is a fake fed from JSON fixtures
under `tests/fixtures/actions/`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/actions"


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_schedules", ROOT / "scripts/check_schedules.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sched = _load()


def utc(text: str) -> datetime:
    return sched.parse_time(text)


# ---------------------------------------------------------------------------
# Fixtures and a fake API
# ---------------------------------------------------------------------------
class FakeAPI:
    """The two reads `check_all` makes, served from a fixture.

    `fail_list` and `fail_runs` model the API refusing to answer, which the
    watchdog must treat as an alarm rather than a pass.
    """

    def __init__(self, workflows: list[dict], fail_list: bool = False,
                 fail_runs: tuple[int, ...] = ()):
        self.workflows = workflows
        self.fail_list = fail_list
        self.fail_runs = fail_runs
        self.run_calls: list[int] = []

    def list_workflows(self):
        if self.fail_list:
            raise sched.ApiError("HTTP 503 Service Unavailable")
        return [{k: v for k, v in w.items() if k not in ("runs", "crons")}
                for w in self.workflows]

    def list_runs(self, workflow_id: int, per_page: int = 50):
        self.run_calls.append(workflow_id)
        if workflow_id in self.fail_runs:
            raise sched.ApiError("HTTP 502 Bad Gateway")
        for w in self.workflows:
            if w["id"] == workflow_id:
                return sorted(w.get("runs", []),
                              key=lambda r: r["run_started_at"], reverse=True)
        return []


def load_fixture(name: str):
    payload = json.loads((FIXTURES / f"{name}.json").read_text())
    workflows = payload["workflows"]
    watched = [(w["path"], w["crons"]) for w in workflows]
    return utc(payload["now"]), watched, FakeAPI(workflows)


def by_path(results) -> dict[str, object]:
    return {r.path: r for r in results}


# ---------------------------------------------------------------------------
# Cron fields
# ---------------------------------------------------------------------------
def test_cron_fields_cover_the_forms_github_accepts():
    assert sched.parse_field("*", 0, 6) == frozenset(range(7))
    assert sched.parse_field("5", 0, 59) == frozenset({5})
    assert sched.parse_field("5,7,11", 0, 59) == frozenset({5, 7, 11})
    assert sched.parse_field("9-12", 0, 23) == frozenset({9, 10, 11, 12})
    assert sched.parse_field("*/6", 0, 23) == frozenset({0, 6, 12, 18})
    assert sched.parse_field("0-30/15", 0, 59) == frozenset({0, 15, 30})
    assert sched.parse_field("20/20", 0, 59) == frozenset({20, 40})
    assert sched.parse_field("mon", 0, 7, sched.DAY_NAMES) == frozenset({1})
    assert sched.parse_field("JAN,jul", 1, 12, sched.MONTH_NAMES) == frozenset({1, 7})


@pytest.mark.parametrize("bad", ["", "61", "-1", "9-5", "*/0", "5-", "abc", "1,,2"])
def test_bad_cron_fields_raise_rather_than_guess(bad):
    with pytest.raises(sched.CronError):
        sched.parse_field(bad, 0, 59)


@pytest.mark.parametrize("bad", ["* * * *", "* * * * * *", "@daily", "60 0 * * *"])
def test_bad_cron_expressions_raise(bad):
    with pytest.raises(sched.CronError):
        sched.parse_cron(bad)


def test_sunday_is_both_zero_and_seven():
    assert sched.parse_cron("0 0 * * 7").days_of_week == frozenset({0})
    assert sched.parse_cron("0 0 * * 0").days_of_week == frozenset({0})


# ---------------------------------------------------------------------------
# Last due slot
# ---------------------------------------------------------------------------
NIGHTLY = ["23 9 * * *", "7 12 * * *", "53 14 * * *"]     # three slots a day
WEEKLY = ["29 7 * * 1"]                                    # Mondays only
MIDNIGHT = ["0 0 * * *"]


def slots(expressions, moment: str, limit: int = 1, earliest: str | None = None):
    schedule = sched.Schedule.from_expressions(expressions)
    return schedule.slots_before(utc(moment), limit=limit,
                                 earliest=utc(earliest) if earliest else None)


def test_multi_slot_daily_walks_back_through_the_day_and_the_one_before():
    assert slots(NIGHTLY, "2026-09-03T13:00:00Z", limit=5) == [
        utc("2026-09-03T12:07:00Z"),
        utc("2026-09-03T09:23:00Z"),
        utc("2026-09-02T14:53:00Z"),
        utc("2026-09-02T12:07:00Z"),
        utc("2026-09-02T09:23:00Z"),
    ]


def test_a_slot_exactly_now_counts_as_past():
    assert slots(NIGHTLY, "2026-09-03T09:23:00Z") == [utc("2026-09-03T09:23:00Z")]
    # ...and one minute earlier it is yesterday's last slot.
    assert slots(NIGHTLY, "2026-09-03T09:22:00Z") == [utc("2026-09-02T14:53:00Z")]


def test_weekly_cron_walks_back_a_whole_week():
    # 2026-09-11 is a Friday; the live slot is the Monday before it.
    assert slots(WEEKLY, "2026-09-11T15:43:00Z", limit=2) == [
        utc("2026-09-07T07:29:00Z"),
        utc("2026-08-31T07:29:00Z"),
    ]


def test_weekly_cron_on_its_own_day_before_and_after_the_slot():
    assert slots(WEEKLY, "2026-09-07T07:28:00Z") == [utc("2026-08-31T07:29:00Z")]
    assert slots(WEEKLY, "2026-09-07T07:29:00Z") == [utc("2026-09-07T07:29:00Z")]


def test_midnight_boundary():
    assert slots(MIDNIGHT, "2026-09-03T00:00:00Z") == [utc("2026-09-03T00:00:00Z")]
    assert slots(MIDNIGHT, "2026-09-02T23:59:00Z") == [utc("2026-09-02T00:00:00Z")]
    # The market schedule's last slot is 23:11: just after midnight the last
    # due slot belongs to *yesterday*, which is where an off-by-one day would
    # invent a missed run every single night.
    assert slots(["41 10 * * *", "37 16 * * *", "11 23 * * *"],
                 "2026-09-03T00:30:00Z", limit=2) == [
        utc("2026-09-02T23:11:00Z"),
        utc("2026-09-02T16:37:00Z"),
    ]


def test_month_end_and_leap_day_do_not_derail_the_walk():
    assert slots(MIDNIGHT, "2026-03-01T00:30:00Z", limit=2) == [
        utc("2026-03-01T00:00:00Z"),
        utc("2026-02-28T00:00:00Z"),
    ]
    # A leap-day-only cron from the following March: the walk skips three
    # Februaries without one. (It gives up after MAX_LOOKBACK_DAYS, which is
    # why this asks from 2025 rather than from 2026.)
    assert slots(["0 0 29 2 *"], "2025-03-01T00:30:00Z", limit=1) == [
        utc("2024-02-29T00:00:00Z"),
    ]
    assert slots(["0 0 29 2 *"], "2026-03-01T00:30:00Z", limit=1) == []


def test_slots_stop_at_the_moment_the_workflow_went_live():
    # Created 09-01 18:51, so the 09-01 slots were never going to fire.
    assert slots(NIGHTLY, "2026-09-02T13:00:00Z", limit=9,
                 earliest="2026-09-01T18:51:04Z") == [
        utc("2026-09-02T12:07:00Z"),
        utc("2026-09-02T09:23:00Z"),
    ]
    assert slots(WEEKLY, "2026-09-03T13:00:00Z", limit=5,
                 earliest="2026-09-02T19:43:52Z") == []


def test_day_of_month_and_day_of_week_together_use_the_or_rule():
    # Vixie cron's quirk, which GitHub inherits: the 1st *or* any Monday.
    both = sched.parse_cron("0 0 1 * 1")
    assert both.matches_date(datetime(2026, 9, 1).date())    # a Tuesday, but the 1st
    assert both.matches_date(datetime(2026, 9, 7).date())    # a Monday, not the 1st
    assert not both.matches_date(datetime(2026, 9, 2).date())


# ---------------------------------------------------------------------------
# Reading schedules out of workflow YAML
# ---------------------------------------------------------------------------
def test_cron_scanner_reads_quotes_comments_and_unquoted_values():
    text = """\
name: Example
# - cron: "0 0 * * *"   <- a commented-out slot must not count
on:
  schedule:
    # Odd minutes on purpose.
    - cron: "23 9 * * *"     # 09:23 UTC
    - cron: '7 12 * * *'
    - cron: 53 14 * * *
  workflow_dispatch:
    inputs:
      cron:
        description: not a schedule
        default: "0 0 * * *"

jobs:
  build:
    steps:
      - run: echo "- cron: 0 0 * * *"
"""
    assert sched.parse_workflow_crons(text) == ["23 9 * * *", "7 12 * * *",
                                                "53 14 * * *"]


def test_cron_scanner_ignores_workflows_without_a_schedule():
    text = "name: CI\non:\n  pull_request:\n  push:\n    branches: [main]\n"
    assert sched.parse_workflow_crons(text) == []


def test_discovery_covers_every_scheduled_workflow_in_this_checkout():
    """The list is discovered, never configured — including the two the

    age-based freshness check is blind to, because they write to R2, Modal and
    W&B rather than to the repo.
    """
    found = dict(sched.discover_workflows(ROOT))
    assert ".github/workflows/statcast-ingest.yml" in found
    assert ".github/workflows/modal-refit.yml" in found
    assert ".github/workflows/nightly-odds.yml" in found
    assert ".github/workflows/market-snapshot.yml" in found
    # The other alarm is watched too; this one's own workflow is not, because
    # the run doing the asking is itself a run since the last slot.
    assert ".github/workflows/freshness-check.yml" in found
    assert f".github/workflows/{sched.SELF_WORKFLOW}" not in found
    assert ".github/workflows/ci.yml" not in found        # no schedule block


def test_discovered_crons_are_the_ones_in_the_files_and_all_parse():
    for path, crons in sched.discover_workflows(ROOT):
        text = (ROOT / path).read_text()
        for expression in crons:
            assert expression in text
            sched.parse_cron(expression)   # raises if the file drifts


def test_every_watched_slot_is_on_an_odd_non_round_minute():
    """docs/automation.md's mitigation, enforced.

    GitHub's own advice is that the top of the hour is the worst window for
    schedule delivery, and every slot this repo lost was on :00/:15/:30/:45.
    """
    for path, crons in sched.discover_workflows(ROOT, skip=()):
        for expression in crons:
            for minute in sched.parse_cron(expression).minutes:
                assert minute % 5 != 0, f"{path}: {expression} is on a round minute"


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------
WORKFLOW = {
    "id": 1,
    "name": "Nightly playoff odds",
    "path": ".github/workflows/nightly-odds.yml",
    "state": "active",
    "created_at": "2026-08-01T00:00:00Z",
}


def check(runs, now: str, workflow: dict | None = WORKFLOW, crons=None, **kw):
    return sched.check_workflow(
        WORKFLOW["path"], crons or NIGHTLY, workflow, runs, utc(now), **kw)


def run(started: str, conclusion: str | None = "success", number: int = 1,
        event: str = "schedule", status: str = "completed") -> dict:
    return {"id": 1000 + number, "run_number": number, "event": event,
            "status": status, "conclusion": conclusion,
            "created_at": started, "run_started_at": started,
            "html_url": "https://github.com/owner/repo/actions/runs/1"}


def test_a_run_after_the_last_due_slot_is_ok():
    result = check([run("2026-09-03T09:25:00Z")], "2026-09-03T15:43:00Z")
    assert result.status == "OK" and not result.failed
    assert result.last_slot == utc("2026-09-03T09:23:00Z")
    assert result.late_hours == pytest.approx(2 / 60)


def test_nothing_since_the_last_due_slot_is_a_drop():
    result = check([run("2026-09-02T09:25:00Z")], "2026-09-03T15:43:00Z")
    assert result.status == "DROPPED" and result.failed
    assert result.last_slot == utc("2026-09-03T09:23:00Z")
    assert "nothing has run since" in result.detail
    assert "2026-09-02T09:25:00+00:00" in result.detail


def test_a_workflow_that_has_never_run_at_all_says_so():
    result = check([], "2026-09-03T15:43:00Z")
    assert result.status == "DROPPED" and "never run" in result.detail


def test_grace_is_what_keeps_a_late_run_from_paging():
    """The worst delay this repo has measured, replayed at two grace periods.

    Same instant, same data: 2h of grace pages, 6h stays quiet, and the run
    turns up at 13:48 proving the 6h call right. This is the calibration.
    """
    yesterday = [run("2026-09-02T14:55:00Z", number=1)]
    assert check(yesterday, "2026-09-03T11:43:00Z",
                 grace_hours=2).status == "DROPPED"
    assert check(yesterday, "2026-09-03T11:43:00Z").status == "OK"

    arrived = yesterday + [run("2026-09-03T13:48:00Z", number=2)]
    late = check(arrived, "2026-09-03T15:43:00Z")
    assert late.status == "LATE" and not late.failed
    assert late.late_hours == pytest.approx(4 + 25 / 60)


def test_runs_that_start_after_now_are_invisible():
    """A replay must only see what was knowable at the instant it replays."""
    future = [run("2026-09-03T13:48:00Z")]
    assert check(future, "2026-09-03T11:43:00Z", grace_hours=2).status == "DROPPED"


def test_lateness_is_reported_but_does_not_fail():
    result = check([run("2026-09-03T10:30:00Z")], "2026-09-03T15:43:00Z")
    assert result.status == "LATE" and not result.failed
    assert "1h07m after the slot" in result.detail
    assert result.run_url.endswith("/runs/1")


def test_a_run_the_moment_its_slot_opens_counts():
    result = check([run("2026-09-03T09:23:00Z")], "2026-09-03T15:43:00Z")
    assert result.status == "OK" and result.late_hours == 0


def test_a_manual_rescue_run_clears_the_alarm_and_is_labelled():
    result = check([run("2026-09-03T14:00:00Z", event="workflow_dispatch")],
                   "2026-09-03T15:43:00Z")
    assert result.status == "LATE" and not result.failed
    assert "event=workflow_dispatch" in result.detail


def test_a_failed_run_is_reported_even_though_it_started():
    result = check([run("2026-09-03T09:25:00Z", conclusion="failure")],
                   "2026-09-03T15:43:00Z")
    assert result.status == "FAILED" and result.failed
    assert "conclusion=failure" in result.detail


def test_a_cancelled_run_counts_as_failed():
    # The data-commits concurrency group can cancel a pending run; the job
    # never happened, so the data never landed.
    result = check([run("2026-09-03T09:25:00Z", conclusion="cancelled")],
                   "2026-09-03T15:43:00Z")
    assert result.status == "FAILED" and result.failed


def test_a_run_still_in_progress_is_not_a_failure():
    result = check([run("2026-09-03T09:25:00Z", conclusion=None,
                        status="in_progress")], "2026-09-03T15:43:00Z")
    assert result.status == "OK" and not result.failed
    assert "still in_progress" in result.detail


def test_no_slot_due_yet_is_pending_not_dropped():
    workflow = {**WORKFLOW, "created_at": "2026-09-02T19:43:52Z"}
    result = check([], "2026-09-03T08:00:00Z", workflow=workflow, crons=WEEKLY)
    assert result.status == "PENDING"
    assert not result.failed
    assert "not a missed run" in result.detail
    assert result.last_slot is None


def test_a_disabled_workflow_is_an_alarm():
    workflow = {**WORKFLOW, "state": "disabled_inactivity"}
    result = check([run("2026-09-03T09:25:00Z")], "2026-09-03T15:43:00Z",
                   workflow=workflow)
    assert result.status == "DISABLED" and result.failed
    assert "60 days" in result.detail


def test_a_workflow_the_api_does_not_know_is_an_alarm():
    result = check([], "2026-09-03T15:43:00Z", workflow=None)
    assert result.status == "ABSENT" and result.failed


def test_an_unparsable_cron_is_an_error_not_a_silent_pass():
    result = check([], "2026-09-03T15:43:00Z", crons=["every friday"])
    assert result.status == "ERROR" and result.failed


def test_per_slot_history_counts_the_misses():
    # Two runs in six slots: 09-02 12:07 and 09-03 09:23.
    runs = [run("2026-09-02T12:10:00Z", number=1),
            run("2026-09-03T09:25:00Z", number=2)]
    result = check(runs, "2026-09-03T15:43:00Z", history=6)
    assert [s.slot.strftime("%m-%d %H:%M") for s in result.slots] == [
        "09-03 09:23", "09-02 14:53", "09-02 12:07",
        "09-02 09:23", "09-01 14:53", "09-01 12:07"]
    assert [s.run is not None for s in result.slots] == [
        True, False, True, False, False, False]
    assert result.missed_slots == 4
    assert result.status == "OK"      # the alarm is the newest slot only


def test_a_run_that_limps_past_the_next_slot_is_attributed_to_that_later_slot():
    """The documented ambiguity, pinned down.

    A run 4h25m after 09:23 starts after the 12:07 slot, so it is counted for
    12:07 and 09:23 reads as missed. The per-slot column is a lower bound on
    delivery; only the newest slot decides the alarm.
    """
    result = check([run("2026-09-03T13:48:00Z")], "2026-09-03T21:43:00Z",
                   history=3)
    outcomes = {s.slot.strftime("%H:%M"): s.run is not None for s in result.slots}
    assert outcomes == {"14:53": False, "12:07": True, "09:23": False}
    assert result.status == "DROPPED"   # nothing since 14:53


# ---------------------------------------------------------------------------
# check_all, against the API fake
# ---------------------------------------------------------------------------
def test_the_outage_of_2026_09_03_would_have_been_caught():
    now, watched, api = load_fixture("outage_2026_09_03")
    results = check_all_default(watched, api, now)
    nightly = by_path(results)[".github/workflows/nightly-odds.yml"]

    assert nightly.status == "DROPPED"
    assert nightly.failed
    assert nightly.last_slot == utc("2026-09-03T09:15:00Z")
    # The one run it ever had, and it was the day before.
    assert "2026-09-02T13:40:56+00:00" in nightly.detail
    # The API answers `created_at` in the owner's offset, not in Z. Reading
    # 14:51:11-04:00 as UTC would move "live since" four hours and change which
    # slots were ever live.
    assert nightly.created_at == utc("2026-09-01T18:51:11Z")
    # Two of the three slots that had come due were never delivered.
    assert nightly.missed_slots == 2

    assert any(r.failed for r in results)
    report = sched.format_report(results, now, sched.DEFAULT_GRACE_HOURS)
    assert "DROPPED Nightly playoff odds" in report
    assert "MISSED" in report


def test_statcast_and_modal_refit_are_pending_not_dropped():
    """The mistake we nearly made by hand on the morning of 2026-09-03."""
    now, watched, api = load_fixture("outage_2026_09_03")
    results = by_path(check_all_default(watched, api, now))
    for path in (".github/workflows/statcast-ingest.yml",
                 ".github/workflows/modal-refit.yml"):
        result = results[path]
        assert result.status == "PENDING", path
        assert not result.failed, path
        assert result.last_slot is None
        assert "no slot has come due yet" in result.detail


def test_merely_late_does_not_fire():
    now, watched, api = load_fixture("late_but_fine")
    results = check_all_default(watched, api, now)
    statuses = {r.name: r.status for r in results}
    assert statuses == {"Nightly playoff odds": "LATE",
                        "Market snapshot": "OK",
                        "Modal refit": "OK"}
    assert not any(r.failed for r in results)
    nightly = by_path(results)[".github/workflows/nightly-odds.yml"]
    assert nightly.late_hours == pytest.approx(4 + 25 / 60)


def test_the_late_fixture_is_what_a_tighter_grace_would_have_paged_on():
    """Calibration, made explicit: 2h of grace turns a good day into an alarm.

    Replayed at 11:43, four hours before the day's run lands, a 2h grace calls
    the 09:23 slot dropped and opens an issue that the 13:48 run then closes.
    The default sits out and is right to.
    """
    _, watched, api = load_fixture("late_but_fine")
    before_the_run = utc("2026-09-11T11:43:00Z")

    tight = sched.check_all(watched, api, before_the_run, grace_hours=2.0)
    assert by_path(tight)[".github/workflows/nightly-odds.yml"].status == "DROPPED"

    generous = sched.check_all(watched, api, before_the_run)
    assert by_path(generous)[".github/workflows/nightly-odds.yml"].status == "OK"
    assert not any(r.failed for r in generous)


def check_all_default(watched, api, now):
    return sched.check_all(watched, api, now)


def test_api_failure_is_an_alarm_not_a_pass():
    now, watched, _ = load_fixture("late_but_fine")
    payload = json.loads((FIXTURES / "late_but_fine.json").read_text())

    dead = FakeAPI(payload["workflows"], fail_list=True)
    results = sched.check_all(watched, dead, now)
    assert all(r.status == "ERROR" and r.failed for r in results)
    assert "cannot list workflows" in results[0].detail

    partial = FakeAPI(payload["workflows"], fail_runs=(101,))
    results = by_path(sched.check_all(watched, partial, now))
    assert results[".github/workflows/nightly-odds.yml"].status == "ERROR"
    assert results[".github/workflows/market-snapshot.yml"].status == "OK"


def test_only_one_runs_call_per_workflow():
    """The watchdog must stay cheap enough to run every couple of hours."""
    now, watched, api = load_fixture("late_but_fine")
    sched.check_all(watched, api, now)
    assert sorted(api.run_calls) == [101, 102, 104]


def test_json_output_is_machine_readable():
    now, watched, api = load_fixture("outage_2026_09_03")
    payload = sched.as_json(sched.check_all(watched, api, now), now,
                            sched.DEFAULT_GRACE_HOURS)
    assert payload["ok"] is False
    assert payload["grace_hours"] == sched.DEFAULT_GRACE_HOURS
    nightly = next(w for w in payload["workflows"]
                   if w["path"] == ".github/workflows/nightly-odds.yml")
    assert nightly["status"] == "DROPPED"
    assert nightly["last_due_slot"] == "2026-09-03T09:15:00+00:00"
    assert nightly["crons"] == ["15 9 * * *", "45 11 * * *"]
    assert nightly["missed_slots"] == 2
    assert [s["missed"] for s in nightly["slots"]] == [True, False, True]
    statcast = next(w for w in payload["workflows"]
                    if w["path"] == ".github/workflows/statcast-ingest.yml")
    assert statcast["status"] == "PENDING" and statcast["failed"] is False


# ---------------------------------------------------------------------------
# End to end through main(): YAML on disk -> cron -> API -> exit code
# ---------------------------------------------------------------------------
def write_workflows(root: Path, watched) -> None:
    directory = root / ".github/workflows"
    directory.mkdir(parents=True, exist_ok=True)
    for path, crons in watched:
        lines = ["name: Example", "on:", "  schedule:"]
        lines += [f'    - cron: "{c}"   # a comment' for c in crons]
        lines += ["  workflow_dispatch:", "", "jobs:", "  go:",
                  "    runs-on: ubuntu-latest", "    steps:",
                  "      - run: echo hi", ""]
        (root / path).write_text("\n".join(lines))


def test_main_exits_1_on_the_outage_and_names_the_workflow(tmp_path, capsys,
                                                           monkeypatch):
    now, watched, api = load_fixture("outage_2026_09_03")
    write_workflows(tmp_path, watched)
    monkeypatch.setattr(sched, "ActionsAPI", lambda *a, **k: api)

    code = sched.main(["--root", str(tmp_path), "--repo", "owner/repo",
                       "--token", "x", "--now", now.isoformat()])
    out = capsys.readouterr().out
    assert code == 1
    assert "DROPPED" in out and "nightly-odds.yml" in out
    assert "PENDING" in out and "statcast-ingest.yml" in out
    assert "2 of 4 scheduled workflows are not delivering" in out


def test_main_exits_0_when_everything_is_merely_late(tmp_path, capsys,
                                                     monkeypatch):
    now, watched, api = load_fixture("late_but_fine")
    write_workflows(tmp_path, watched)
    monkeypatch.setattr(sched, "ActionsAPI", lambda *a, **k: api)

    code = sched.main(["--root", str(tmp_path), "--repo", "owner/repo",
                       "--token", "x", "--now", now.isoformat(), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["ok"] is True
    assert {w["status"] for w in payload["workflows"]} == {"OK", "LATE"}


def test_main_refuses_to_run_without_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        sched.main(["--root", str(tmp_path), "--repo", "owner/repo"])
    assert exc.value.code == 2
