"""Watchdog: ask the Actions API whether each scheduled workflow actually ran.

Companion to `scripts/check_freshness.py`, and deliberately not a replacement
for it. The freshness check infers a dropped run from the *age of the output*,
which is slow by construction: on the 2026-09-03 outage a 36h budget would not
have fired until 09-04 01:41Z, about sixteen hours after the board went stale,
and the budget cannot be tightened because the nightly's three slots span
09:23–14:53 so the worst *legitimate* gap between good runs is ~29h30m.

This script stops inferring. `GITHUB_TOKEN` inside Actions can read the Actions
API, so we can ask the question directly, per workflow:

    given this workflow's own cron slots, has a run actually started since the
    last slot came due?

That answers in hours instead of a day, and it needs no staleness budget to
guess at. It also sees the two workflows the age check is blind to —
`statcast-ingest.yml` writes to R2 and the Modal volume, `modal-refit.yml` to
Modal and W&B, and neither leaves anything in the repo.

Design notes, in the order they matter:

- **The schedules are parsed out of the workflow YAML, never restated here.**
  A second copy of the cron slots would drift the first time someone edits a
  schedule, and a watchdog watching last month's schedule is worse than none.
  Every workflow under `.github/workflows` that has a `schedule:` block is
  watched; the list is discovered, not configured.

- **"Never due" is not "dropped".** A workflow that reached `main` after its
  last slot has nothing to answer for. On 2026-09-03 `run_number: 1` on all
  four workflows looked like total failure until we counted which slots were
  actually live — two of the four had simply never had one come due. That
  distinction is `PENDING` below, and it does not fail the check.

- **Grace before a slot is "due".** GitHub's `schedule` event is best effort;
  the two runs we have were 1h58m and 4h25m late. A slot only counts as due
  once `slot + GRACE` has passed — see `DEFAULT_GRACE_HOURS` for the value and
  the argument for it.

- **A run is a run.** The drop test asks whether a run *started*, not whether
  it succeeded, because a failed run is already loud: it turns the workflow
  red and mails the owner. Silence is the failure mode this exists for. The
  conclusion is reported anyway (status `FAILED`), since for the R2/Modal jobs
  nothing else in the repo would ever mention it.

- **Disabled workflows.** GitHub disables scheduled workflows in public repos
  after 60 days without activity, and disables them silently. A disabled
  workflow reports `DISABLED` and fails — the age check could only ever see
  the consequence, months later.

Stdlib only, like the freshness check, and for the same reason: the watchdog
workflow installs nothing, so there is no dependency resolution step between a
dropped run and a human finding out. That includes the YAML — PyYAML is not in
`requirements-ci.txt`, and it would parse the `on:` key as the boolean `True`
anyway. The scanner in `parse_workflow_crons` reads exactly the one construct
we need.

Usage:
    python scripts/check_schedules.py                     # inside Actions
    python scripts/check_schedules.py --repo owner/name --json
    python scripts/check_schedules.py --now 2026-09-03T13:23:00+00:00
    python scripts/check_schedules.py --grace-hours 3

Exit codes: 0 every schedule is delivering, 1 something dropped, is disabled,
failed, or could not be checked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXIT_OK = 0
EXIT_ALARM = 1

# ---------------------------------------------------------------------------
# The grace period, which is the only tuned number in this file.
#
# A slot counts as "due" once GRACE has passed since it. Too small and the
# alarm fires on a schedule that is merely late — which is what killed the last
# alarm we had, not by being wrong but by being ignored. Too large and it just
# takes longer to notice a real drop.
#
# Floor: the delivery delays we have actually measured are 1h58m and 4h25m
# (docs/automation.md). Anything at or under 4h25m would have paged on a run
# that did arrive, on the very sample we are calibrating against.
#
# Ceiling: the cost of extra grace is detection latency and nothing else. The
# redundancy in the schedules absorbs most of it — the drop test asks whether
# *anything* has run since the last due slot, so a later slot firing clears an
# earlier one. For the nightly (09:23 / 12:07 / 14:53) a whole lost day is
# caught at 09:23 + GRACE, i.e. the same afternoon, versus ~16h late for the
# age-based check.
#
# 6h: ~35% headroom over the worst delay observed, and still same-day detection
# for every workflow here. It is generous on purpose. Override with
# --grace-hours to see what a tighter alarm would have said.
DEFAULT_GRACE_HOURS = 6.0

# A run that started more than this after its slot is reported LATE. Purely
# informational — lateness is not a failure, it is the thing GRACE forgives —
# but it is the leading indicator that delivery is degrading, and it is how we
# will know whether the odd-minute cron slots helped.
DEFAULT_LATE_HOURS = 1.0

# How many past slots to show per workflow. Enough to see a pattern of drops in
# the report without turning it into a log.
DEFAULT_HISTORY = 6

# Cheap guard on the day-walk in Schedule.slots_before: no cron in this repo is
# rarer than weekly, and walking a year back is still microseconds.
MAX_LOOKBACK_DAYS = 400

# This file's own watchdog, excluded because watching itself is vacuous: the run
# doing the asking is itself a run since the last slot, so the row would always
# be green. freshness-check.yml *is* watched from here like any other scheduled
# workflow. Nothing watches this one, which is why its schedule has twelve slots
# a day — losing every one of them is the failure mode it cannot report. See
# docs/automation.md.
SELF_WORKFLOW = "schedule-watchdog.yml"

# Conclusions that mean the run happened but the work did not.
BAD_CONCLUSIONS = frozenset({"failure", "cancelled", "timed_out", "startup_failure"})

MONTH_NAMES = {name: i for i, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
DAY_NAMES = {name: i for i, name in enumerate(
    ["sun", "mon", "tue", "wed", "thu", "fri", "sat"])}


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------
class CronError(ValueError):
    """A cron expression we refuse to guess at."""


def parse_field(text: str, low: int, high: int,
                names: dict[str, int] | None = None) -> frozenset[int]:
    """One cron field to the set of values it matches.

    Supports `*`, `a`, `a-b`, `*/n`, `a-b/n`, `a/n` and comma-separated lists
    of those, plus three-letter month and day names. That is everything GitHub
    Actions accepts; `@daily`-style nicknames and the non-standard `L`/`W`/`#`
    are not accepted here, and raise rather than being silently mis-read.
    """
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise CronError(f"empty term in {text!r}")
        step = 1
        if "/" in part:
            part, _, step_text = part.partition("/")
            if not step_text.isdigit() or int(step_text) == 0:
                raise CronError(f"bad step {step_text!r} in {text!r}")
            step = int(step_text)
        part = part.strip()
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_text, _, end_text = part.partition("-")
            start = _field_number(start_text, names, text)
            end = _field_number(end_text, names, text)
        else:
            start = _field_number(part, names, text)
            # `5/2` means 5, 7, 9 ... to the top of the range; a bare `5` is
            # just itself.
            end = high if step > 1 else start
        if start > end or start < low or end > high:
            raise CronError(f"{part!r} out of range {low}-{high} in {text!r}")
        values.update(range(start, end + 1, step))
    if not values:
        raise CronError(f"no values in {text!r}")
    return frozenset(values)


def _field_number(text: str, names: dict[str, int] | None, whole: str) -> int:
    text = text.strip()
    if names is not None and text.lower() in names:
        return names[text.lower()]
    if not text.isdigit():
        raise CronError(f"bad value {text!r} in {whole!r}")
    return int(text)


@dataclass(frozen=True)
class Cron:
    """One five-field cron expression, evaluated in UTC (GitHub's timezone)."""
    expression: str
    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool

    def matches_date(self, day: date) -> bool:
        if day.month not in self.months:
            return False
        dom_hit = day.day in self.days_of_month
        # date.weekday() is Monday=0; cron is Sunday=0.
        dow_hit = ((day.weekday() + 1) % 7) in self.days_of_week
        if self.dom_restricted and self.dow_restricted:
            # Vixie cron's OR rule, which GitHub inherits: when both fields are
            # restricted a day matches if *either* does. Nothing in this repo
            # uses both, but a wrong guess here would silently invent slots.
            return dom_hit or dow_hit
        return dom_hit and dow_hit

    def times_on(self, day: date) -> list[tuple[int, int]]:
        if not self.matches_date(day):
            return []
        return [(h, m) for h in sorted(self.hours) for m in sorted(self.minutes)]


def parse_cron(expression: str) -> Cron:
    fields = expression.split()
    if len(fields) != 5:
        raise CronError(
            f"expected 5 fields, got {len(fields)} in {expression!r}")
    minute, hour, dom, month, dow = fields
    days_of_week = parse_field(dow, 0, 7, DAY_NAMES)
    if 7 in days_of_week:  # both 0 and 7 mean Sunday
        days_of_week = frozenset({0} | (days_of_week - {7}))
    return Cron(
        expression=expression.strip(),
        minutes=parse_field(minute, 0, 59),
        hours=parse_field(hour, 0, 23),
        days_of_month=parse_field(dom, 1, 31),
        months=parse_field(month, 1, 12, MONTH_NAMES),
        days_of_week=days_of_week,
        dom_restricted=dom.strip() != "*",
        dow_restricted=dow.strip() != "*",
    )


@dataclass(frozen=True)
class Schedule:
    """Every cron slot a workflow has, merged."""
    crons: tuple[Cron, ...]

    @classmethod
    def from_expressions(cls, expressions) -> "Schedule":
        return cls(tuple(parse_cron(e) for e in expressions))

    def slots_before(self, moment: datetime, limit: int = 1,
                     earliest: datetime | None = None) -> list[datetime]:
        """The `limit` most recent slots at or before `moment`, newest first.

        `earliest` cuts the walk off at the moment the workflow came into
        existence: a slot from before the file was on `main` was never going to
        fire, and counting it is exactly the mistake that made every workflow
        look broken on 2026-09-03.
        """
        found: list[datetime] = []
        day = moment.date()
        for _ in range(MAX_LOOKBACK_DAYS):
            times = sorted({t for cron in self.crons for t in cron.times_on(day)},
                           reverse=True)
            for hour, minute in times:
                slot = datetime(day.year, day.month, day.day, hour, minute,
                                tzinfo=timezone.utc)
                if slot > moment:
                    continue
                if earliest is not None and slot < earliest:
                    return found
                found.append(slot)
                if len(found) >= limit:
                    return found
            day -= timedelta(days=1)
            if earliest is not None and day < earliest.date() - timedelta(days=1):
                break
        return found


# ---------------------------------------------------------------------------
# Reading the schedules out of the workflow files
# ---------------------------------------------------------------------------
_CRON_ITEM = re.compile(r"""^\s*-\s*cron\s*:\s*(?:(['"])(?P<q>.*?)\1|(?P<b>[^#]*?))\s*(?:#.*)?$""")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_workflow_crons(text: str) -> list[str]:
    """Cron expressions from a workflow's `on: schedule:` block, in file order.

    A hand-rolled scan rather than a YAML parser: this script installs nothing
    (see the module docstring), and the construct is a fixed shape — a
    `schedule:` key whose value is a list of `- cron: "..."` items. We enter on
    `schedule:` and leave at the first non-blank, non-comment line indented no
    deeper than it, so a `- cron:` written inside some other block, or quoted in
    a comment, is not mistaken for a schedule.
    """
    crons: list[str] = []
    schedule_indent: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if schedule_indent is not None and _indent(line) <= schedule_indent:
            schedule_indent = None
        if schedule_indent is None:
            if re.match(r"^\s*schedule\s*:\s*(#.*)?$", line):
                schedule_indent = _indent(line)
            continue
        match = _CRON_ITEM.match(line)
        if match:
            value = match.group("q")
            if value is None:
                value = match.group("b") or ""
            value = value.strip()
            if value:
                crons.append(value)
    return crons


def discover_workflows(root: Path, skip: tuple[str, ...] = (SELF_WORKFLOW,)
                       ) -> list[tuple[str, list[str]]]:
    """Every scheduled workflow in the checkout: (repo-relative path, crons).

    Sorted by path so the report is stable, and skipping this script's own
    workflow because a watchdog cannot usefully watch itself.
    """
    directory = root / ".github" / "workflows"
    found: list[tuple[str, list[str]]] = []
    for entry in sorted(directory.glob("*.y*ml")):
        if entry.name in skip:
            continue
        crons = parse_workflow_crons(entry.read_text())
        if crons:
            found.append((f".github/workflows/{entry.name}", crons))
    return found


# ---------------------------------------------------------------------------
# The Actions API
# ---------------------------------------------------------------------------
class ApiError(RuntimeError):
    """The API would not answer. Treated as an alarm, not as a pass."""


class ActionsAPI:
    """The three reads this needs, over urllib. No third-party HTTP client."""

    def __init__(self, repo: str, token: str,
                 api_url: str = "https://api.github.com",
                 attempts: int = 3, sleep=time.sleep):
        self.repo = repo
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.attempts = attempts
        self.sleep = sleep

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.api_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "baseball-projections-schedule-watchdog",
        })
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = exc
                # 5xx and secondary rate limits are worth another try; 401/403
                # on a missing token and 404 on a missing workflow are not.
                if exc.code not in (429, 500, 502, 503, 504):
                    raise ApiError(f"GET {path} -> HTTP {exc.code} {exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            if attempt + 1 < self.attempts:
                self.sleep(2 ** attempt)
        raise ApiError(f"GET {path} failed after {self.attempts} attempts: {last}")

    def list_workflows(self) -> list[dict]:
        workflows: list[dict] = []
        page = 1
        while True:
            payload = self._get(f"/repos/{self.repo}/actions/workflows",
                                {"per_page": 100, "page": page})
            batch = payload.get("workflows", [])
            workflows.extend(batch)
            if len(batch) < 100 or len(workflows) >= payload.get("total_count", 0):
                return workflows
            page += 1

    def list_runs(self, workflow_id: int, per_page: int = 50) -> list[dict]:
        """Most recent runs of one workflow, newest first (the API's order)."""
        payload = self._get(
            f"/repos/{self.repo}/actions/workflows/{workflow_id}/runs",
            {"per_page": per_page})
        return payload.get("workflow_runs", [])


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------
def parse_time(text: str) -> datetime:
    """ISO 8601 to an aware UTC datetime. A naive timestamp is read as UTC."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    stamp = datetime.fromisoformat(cleaned)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def run_start(run: dict) -> datetime | None:
    """When a run began.

    `run_started_at` is the real start; `created_at` is when GitHub created the
    run object. They were equal on the 4h25m-late run of 2026-09-03, which is
    how we know the schedule *event* was delivered late rather than the job
    sitting in a queue — so both are kept, and this returns the start.
    """
    for key in ("run_started_at", "created_at"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return parse_time(value)
    return None


@dataclass
class SlotOutcome:
    """One due slot and the first run attributed to it.

    A run belongs to the most recent slot at or before its start, so a run that
    limps in after the *next* slot is counted for that later slot. The per-slot
    column is therefore a lower bound on delivery; only the newest slot decides
    the alarm, and that test is the unambiguous one — did anything at all run
    since the last slot came due.
    """
    slot: datetime
    run: dict | None = None
    started: datetime | None = None

    @property
    def late_hours(self) -> float | None:
        if self.started is None:
            return None
        return (self.started - self.slot).total_seconds() / 3600.0


@dataclass
class Result:
    path: str
    name: str
    crons: list[str]
    status: str                        # OK | LATE | DROPPED | PENDING |
                                       # DISABLED | FAILED | ABSENT | ERROR
    last_slot: datetime | None = None
    started: datetime | None = None
    late_hours: float | None = None
    created_at: datetime | None = None
    detail: str = ""
    run_url: str | None = None
    slots: list[SlotOutcome] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status in {"DROPPED", "DISABLED", "FAILED", "ABSENT", "ERROR"}

    @property
    def missed_slots(self) -> int:
        return sum(1 for s in self.slots if s.run is None)


def check_workflow(path: str, crons: list[str], workflow: dict | None,
                   runs: list[dict], now: datetime,
                   grace_hours: float = DEFAULT_GRACE_HOURS,
                   late_hours: float = DEFAULT_LATE_HOURS,
                   history: int = DEFAULT_HISTORY) -> Result:
    """One workflow: are its slots being delivered?"""
    name = (workflow or {}).get("name") or Path(path).name
    result = Result(path=path, name=name, crons=list(crons), status="OK")

    if workflow is None:
        result.status = "ABSENT"
        result.detail = ("the API does not know this workflow — it is not on "
                         "the default branch, or the file was just renamed")
        return result

    try:
        schedule = Schedule.from_expressions(crons)
    except CronError as exc:
        result.status = "ERROR"
        result.detail = str(exc)
        return result

    # Careful: the API answers a workflow's `created_at` in the repo owner's
    # local offset ("2026-09-01T14:51:11-04:00"), not in Z like the run
    # timestamps. `parse_time` normalises; reading it as UTC would move every
    # "live since" by hours and invent or hide a due slot.
    created = workflow.get("created_at")
    result.created_at = parse_time(created) if created else None

    state = workflow.get("state", "active")
    if state != "active":
        result.status = "DISABLED"
        result.detail = (f"workflow state is {state!r} — GitHub disables "
                         "scheduled workflows in public repos after 60 days "
                         "without repository activity, and does it quietly")
        return result

    # A slot is only due once the grace period has passed since it.
    cutoff = now - timedelta(hours=grace_hours)
    slots = schedule.slots_before(cutoff, limit=max(history, 1),
                                  earliest=result.created_at)
    if not slots:
        result.status = "PENDING"
        first = "" if result.created_at is None else \
            f" (live since {result.created_at.isoformat()})"
        result.detail = (f"no slot has come due yet{first}: nothing to answer "
                         f"for, not a missed run")
        return result

    # Runs that started after `now` only exist when replaying history with
    # --now; dropping them keeps a backtest honest about what was knowable at
    # that instant, which is the whole point of the fixtures in the tests.
    starts = [(run, run_start(run)) for run in runs]
    starts = [(run, start) for run, start in starts
              if start is not None and start <= now]

    outcomes: list[SlotOutcome] = []
    for index, slot in enumerate(slots):
        upper = slots[index - 1] if index else None  # the next slot, or open
        attributed = [(run, start) for run, start in starts
                      if start >= slot and (upper is None or start < upper)]
        if attributed:
            run, start = min(attributed, key=lambda pair: pair[1])
            outcomes.append(SlotOutcome(slot=slot, run=run, started=start))
        else:
            outcomes.append(SlotOutcome(slot=slot))
    result.slots = outcomes
    result.last_slot = slots[0]

    newest = outcomes[0]
    if newest.run is None:
        result.status = "DROPPED"
        overdue = (now - slots[0]).total_seconds() / 3600.0
        last_ever = max((s for _, s in starts), default=None)
        seen = ("never run" if last_ever is None
                else f"last run started {last_ever.isoformat()}")
        result.detail = (f"slot {slots[0].isoformat()} came due "
                         f"{format_hours(overdue)} ago and nothing has run "
                         f"since; {seen}")
        return result

    result.started = newest.started
    result.late_hours = newest.late_hours
    result.run_url = newest.run.get("html_url")
    conclusion = newest.run.get("conclusion")
    status_field = newest.run.get("status")
    where = (f"run #{newest.run.get('run_number', '?')} started "
             f"{newest.started.isoformat()} "
             f"({format_hours(newest.late_hours)} after the slot, "
             f"event={newest.run.get('event', '?')})")

    if conclusion in BAD_CONCLUSIONS:
        result.status = "FAILED"
        result.detail = f"{where}, conclusion={conclusion}"
    elif conclusion is None:
        result.detail = f"{where}, still {status_field or 'running'}"
    elif newest.late_hours is not None and newest.late_hours > late_hours:
        result.status = "LATE"
        result.detail = f"{where}, conclusion={conclusion}"
    else:
        result.detail = f"{where}, conclusion={conclusion}"
    return result


def check_all(workflows: list[tuple[str, list[str]]], api, now: datetime,
              grace_hours: float = DEFAULT_GRACE_HOURS,
              late_hours: float = DEFAULT_LATE_HOURS,
              history: int = DEFAULT_HISTORY) -> list[Result]:
    """Check every discovered workflow. One API listing plus one call each."""
    try:
        listed = api.list_workflows()
    except ApiError as exc:
        return [Result(path=path, name=Path(path).name, crons=list(crons),
                       status="ERROR", detail=f"cannot list workflows: {exc}")
                for path, crons in workflows]

    by_path = {w.get("path"): w for w in listed}
    results: list[Result] = []
    for path, crons in workflows:
        workflow = by_path.get(path)
        runs: list[dict] = []
        if workflow is not None:
            try:
                runs = api.list_runs(workflow["id"])
            except ApiError as exc:
                results.append(Result(path=path, name=workflow.get("name", path),
                                      crons=list(crons), status="ERROR",
                                      detail=f"cannot list runs: {exc}"))
                continue
        results.append(check_workflow(path, crons, workflow, runs, now,
                                      grace_hours, late_hours, history))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def format_hours(hours: float | None) -> str:
    if hours is None:
        return "-"
    sign = "-" if hours < 0 else ""
    minutes = int(abs(hours) * 60 + 0.5)
    return f"{sign}{minutes // 60}h{minutes % 60:02d}m"


def format_slot(slot: datetime | None) -> str:
    return "-" if slot is None else slot.strftime("%m-%d %H:%MZ")


def format_report(results: list[Result], now: datetime,
                  grace_hours: float) -> str:
    name_w = max([len(r.name) for r in results] + [len("workflow")])
    lines = [
        f"Scheduled workflow delivery at {now.isoformat()} "
        f"(grace {format_hours(grace_hours)})",
        "",
        f"{'workflow':<{name_w}}  {'last due slot':>13}  {'ran':>8}  status",
        f"{'-' * name_w}  {'-' * 13}  {'-' * 8}  {'-' * 6}",
    ]
    for r in results:
        ran = format_hours(r.late_hours) if r.started else "-"
        lines.append(f"{r.name:<{name_w}}  {format_slot(r.last_slot):>13}  "
                     f"{ran:>8}  {r.status}")
    lines.append("")
    for r in results:
        if r.status != "OK":
            lines.append(f"{r.status} {r.name} ({r.path}): {r.detail}")
        if r.slots and r.missed_slots:
            history = "  ".join(
                f"{format_slot(s.slot)}"
                f"{'  MISSED' if s.run is None else '+' + format_hours(s.late_hours)}"
                for s in reversed(r.slots))
            lines.append(f"     recent slots: {history}")
    failures = [r for r in results if r.failed]
    if failures:
        lines.append("")
        lines.append(
            f"{len(failures)} of {len(results)} scheduled workflows are not "
            "delivering. GitHub's schedule event is best effort and drops runs "
            "without a trace; re-run the workflow by hand to recover the data, "
            "then check whether the slot has been dropping repeatedly above.")
    else:
        lines.append(f"All {len(results)} scheduled workflows have run since "
                     "their last due slot.")
    return "\n".join(lines)


def as_json(results: list[Result], now: datetime, grace_hours: float) -> dict:
    return {
        "now": now.isoformat(),
        "grace_hours": grace_hours,
        "ok": not any(r.failed for r in results),
        "workflows": [
            {
                "path": r.path,
                "name": r.name,
                "crons": r.crons,
                "status": r.status,
                "failed": r.failed,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "last_due_slot": r.last_slot.isoformat() if r.last_slot else None,
                "run_started_at": r.started.isoformat() if r.started else None,
                "late_hours": (round(r.late_hours, 3)
                               if r.late_hours is not None else None),
                "run_url": r.run_url,
                "missed_slots": r.missed_slots,
                "slots": [
                    {
                        "slot": s.slot.isoformat(),
                        "run_started_at": s.started.isoformat() if s.started else None,
                        "late_hours": (round(s.late_hours, 3)
                                       if s.late_hours is not None else None),
                        "missed": s.run is None,
                    }
                    for s in r.slots
                ],
                "detail": r.detail,
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="checkout whose workflow files hold the schedules")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"),
                    help="owner/name (default: $GITHUB_REPOSITORY)")
    ap.add_argument("--token", default=(os.environ.get("GITHUB_TOKEN")
                                        or os.environ.get("GH_TOKEN")),
                    help="API token (default: $GITHUB_TOKEN)")
    ap.add_argument("--api-url", default=os.environ.get("GITHUB_API_URL",
                                                        "https://api.github.com"))
    ap.add_argument("--now", type=parse_time, default=None,
                    help="ISO 8601 instant to evaluate at (default: now, UTC)")
    ap.add_argument("--grace-hours", type=float, default=DEFAULT_GRACE_HOURS,
                    help=f"delay tolerated before a slot is due "
                         f"(default {DEFAULT_GRACE_HOURS})")
    ap.add_argument("--late-hours", type=float, default=DEFAULT_LATE_HOURS,
                    help="report a run this far past its slot as LATE")
    ap.add_argument("--history", type=int, default=DEFAULT_HISTORY,
                    help="how many past slots to account for per workflow")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output instead of the table")
    args = ap.parse_args(argv)

    if not args.repo:
        ap.error("no repository: pass --repo owner/name or set GITHUB_REPOSITORY")
    if not args.token:
        ap.error("no token: pass --token or set GITHUB_TOKEN (inside Actions "
                 "the workflow's own GITHUB_TOKEN can read the Actions API)")

    now = args.now or datetime.now(timezone.utc)
    workflows = discover_workflows(args.root)
    if not workflows:
        print(f"No scheduled workflows found under {args.root}/.github/workflows",
              file=sys.stderr)
        return EXIT_ALARM

    api = ActionsAPI(args.repo, args.token, args.api_url)
    results = check_all(workflows, api, now, args.grace_hours, args.late_hours,
                        args.history)

    if args.json:
        print(json.dumps(as_json(results, now, args.grace_hours), indent=2))
    else:
        print(format_report(results, now, args.grace_hours))

    return EXIT_ALARM if any(r.failed for r in results) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
