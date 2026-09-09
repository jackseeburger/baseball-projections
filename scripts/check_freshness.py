"""Watchdog: fail when a Layer 1 production artifact is older than its budget.

docs/automation.md promises a staleness alarm for the nightly jobs. This is it.
The jobs themselves are GitHub Actions cron, and GitHub cron is best-effort —
it drops scheduled runs silently and starts the ones it keeps late. When that
happens nothing in the repo changes, no workflow turns red, and the site keeps
serving yesterday's board. So this script checks the *age of the output* on a
separate schedule and shouts.

It has a faster sibling. `scripts/check_schedules.py` asks the Actions API
whether each workflow has run since its last cron slot, which catches a dropped
run the same afternoon instead of a day later. Both ship, because they answer
different questions: the API knows whether the job RAN, and this file knows
whether it PRODUCED. A job that runs green and writes nothing — a bad input, a
silent exception in a `try`, a builder falling back to yesterday's data — is
invisible to the API check and obvious here. Aim the two at their strengths:
this table is the correctness backstop, not the outage alarm, and its budgets
are loose on purpose.

Design notes, in the order they matter:

- **Never file mtime.** A fresh `git clone` stamps every file with the checkout
  time, so on a CI runner mtime says "seconds old" for data written a week ago.
  Every artifact below is aged from a timestamp the producing job wrote *into*
  the data: a JSON field, or the timestamp in an immutable filename. Nothing in
  this file calls `stat()`.

- **Budgets come from cadence, not taste.** Each budget is the worst gap the
  schedule can legitimately produce, plus room for a late start. See the table.

- **Absent vs missing.** If an artifact's directory does not exist at all, the
  product is not standing up yet: report `ABSENT` and — for an artifact marked
  `required=False` — do not fail. Every artifact in the table today is
  `required=True`, because every one of them exists in the repo now and its
  workflow has been shipping; an absent directory for one of those is a real
  regression (someone deleted a tree, or checkout is broken), so it fails. The
  `required=False` path exists for artifacts whose job has not landed yet, and
  for the day the snapshots move from git to R2. A directory that exists but
  holds no matching file is always a failure (`MISSING`) — the job that fills
  it ran and produced nothing.

- **Known-blocked staleness.** An artifact whose refresh is blocked by
  something recorded and open carries `blocked_by`. It is still measured and
  still reported as `STALE` in the table and in the summary; it just does not
  fail the job, because a permanently red alarm is one nobody reads, and the
  other five artifacts still need this check to mean something. `blocked_by`
  must name the issue, and it does not excuse `MISSING` or `ERROR`: not being
  able to rebuild a file is a different thing from the file disappearing.

- **Only repo artifacts.** `statcast-ingest.yml` writes to R2 and the Modal
  volume; `modal-refit.yml` writes to Modal and W&B. Neither leaves anything in
  the repo, so this script cannot see them, and reaching into R2 would give the
  alarm credentials and a network dependency of its own — the alarm must be the
  most reliable thing in the system, not the least. They stay uncovered here.
  `scripts/check_schedules.py` covers both of them for *did it run*; whether
  what they wrote is any good still wants a check against R2 object timestamps.

Stdlib only, on purpose: the watchdog workflow installs nothing, so there is no
dependency resolution step between a stale board and a human finding out.

Usage:
    python scripts/check_freshness.py
    python scripts/check_freshness.py --json
    python scripts/check_freshness.py --now 2026-09-03T12:00:00+00:00
    python scripts/check_freshness.py --root /path/to/checkout

Exit codes: 0 everything within budget, 1 something is stale or missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXIT_OK = 0
EXIT_STALE = 1


@dataclass(frozen=True)
class Artifact:
    """One production output and how old it is allowed to get.

    `kind` says where the timestamp lives:
      json_field      — parse `path` as JSON, read ISO 8601 out of `field`
      newest_filename — newest file in directory `path` matching `pattern`,
                        timestamp parsed from the name up to the first dot
                        with `ts_format`
    """
    name: str
    path: str
    kind: str
    budget_hours: float
    required: bool
    workflow: str
    field: str | None = None
    pattern: str | None = None
    ts_format: str | None = None
    # A known, recorded reason this artifact cannot currently be refreshed —
    # an issue reference and one line of why. Set it and a STALE reading is
    # reported but does not fail the job; leave it None and STALE fails, which
    # is the default and should stay the default. It does NOT excuse MISSING
    # or ERROR: "we cannot rebuild it" is not "it may vanish".
    blocked_by: str | None = None


# ---------------------------------------------------------------------------
# The budget table. One place, deliberately.
#
# Each budget is (worst legitimate gap between successful runs) + (slack for a
# late start), and is kept under 24h wherever the job is more frequent than
# daily so that a whole missed day is always caught.
#
# nightly-odds.yml  — 09:23, 12:07 and 14:53 UTC daily. Worst legitimate gap
#   between good runs is 29h30m: the day's first slot (09:23) followed by only
#   the next day's last (14:53). Budget 36h leaves 6h30m on top of that, and a
#   whole missed day is at least 42h30m, so every missed day trips it while a
#   4h25m-late run — which has already happened once — does not. All three
#   files below come from that one workflow and each re-stamps `generated_at`
#   every run, including the paths where the builder falls back to the last
#   committed data and marks it stale. That is the right signal here: this
#   script asks "did the job run", not "is the data any good".
#
# market-snapshot.yml — 10:41, 16:37 and 23:11 UTC. Worst legitimate gap is
#   11h30m (23:11 to 10:41); one dropped slot stretches that to 18h04m. Budget
#   22h: tolerates a single dropped slot plus a ~4h late start, trips on two
#   consecutive dropped slots (24h) and on a whole missed day (35h30m). The
#   20h this table shipped with was computed against the old round-minute slots
#   and left only 1h56m of slack over a single drop — less than the 1h58m delay
#   we have actually observed, i.e. tuned to cry wolf.
#
# Both budgets stay deliberately loose, because since 2026-09 they are no
# longer the thing that catches a dropped run: scripts/check_schedules.py asks
# the Actions API directly and answers the same afternoon. What is left for
# this table is the question the API cannot answer — did the job that ran
# actually produce output — so it is allowed to be slow, and it is not allowed
# to cry wolf.
#
# paper ledger — written by the same market-snapshot.yml run, immediately after
#   the snapshot it emits tickets from, so it inherits that workflow's cadence
#   exactly and gets the same 22h budget for the same reasons. It is not given
#   a looser one on the grounds that a paper ledger is not production: a ledger
#   that silently stops appending is a forward test that has quietly stopped,
#   and the whole value of docs/bankroll.md's Stage 0 is that its history
#   cannot be edited or interrupted without anyone noticing.
#
# career-war.yml — Mondays 19:11 UTC only, one slot, no redundancy. This is
#   not an oversight: the input is season-aggregate posteriors
#   (data/projections/*_projections_2026.parquet) that a weekly Modal refit
#   might change and nothing else does, so a second same-day slot would just
#   re-commit an identical generated_at. Worst legitimate gap between good
#   runs is 7 days (168h) if every Monday fires; one dropped Monday stretches
#   that to 14 days (336h). Budget 216h (9 days) leaves 48h of slack for a
#   late start — GitHub's worst measured delay elsewhere in this repo is
#   4h25m, so two full days is already generous — while still tripping on a
#   fully dropped week (336h) well before the next one is due. Issue #75.
# ---------------------------------------------------------------------------
ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name="playoff odds board",
        path="public/data/playoff_odds/latest.json",
        kind="json_field",
        field="generated_at",
        budget_hours=36,
        required=True,
        workflow="nightly-odds.yml",
    ),
    Artifact(
        name="rest-of-season projections",
        path="public/data/projections/latest.json",
        kind="json_field",
        field="generated_at",
        budget_hours=36,
        required=True,
        workflow="nightly-odds.yml",
    ),
    Artifact(
        name="accuracy page data",
        path="public/data/accuracy/latest.json",
        kind="json_field",
        field="generated_at",
        budget_hours=36,
        required=True,
        workflow="nightly-odds.yml",
    ),
    # BAS-72: the current season's rows are rebuilt every night, before
    # build_ros_projections.py, by
    # `scripts/build_contact_quality.py --update-season <current>` — same
    # nightly-odds.yml cadence and slots as the three artifacts above, so the
    # same 36h budget (worst legitimate gap 29h30m + 6h30m slack) applies.
    # The timestamp lives in a JSON sidecar next to the parquet
    # (`contact_quality_monthly.meta.json`, `built_at`), not in the parquet
    # itself — this script is stdlib-only on purpose (see the module
    # docstring), and parsing parquet needs pandas/pyarrow. `build_year`'s
    # caller stamps the sidecar only when the season it just rebuilt is the
    # newest one on record, so a rebuild of an old season for a backfill does
    # not make tonight's freshness read as current.
    Artifact(
        name="contact-quality monthly (current season)",
        path="data/features/contact_quality_monthly.meta.json",
        kind="json_field",
        field="built_at",
        budget_hours=36,
        required=True,
        workflow="nightly-odds.yml",
    ),
    Artifact(
        name="market snapshot archive",
        path="data/market/snapshots",
        kind="newest_filename",
        pattern="*.jsonl.gz",
        ts_format="%Y-%m-%dT%H%MZ",
        budget_hours=22,
        required=True,
        workflow="market-snapshot.yml",
    ),
    Artifact(
        name="market latest.json",
        path="public/data/market/latest.json",
        kind="json_field",
        field="as_of",
        budget_hours=22,
        required=True,
        workflow="market-snapshot.yml",
    ),
    Artifact(
        name="paper ledger",
        path="public/data/market/paper_ledger.json",
        kind="json_field",
        field="generated_at",
        budget_hours=22,
        required=True,
        workflow="market-snapshot.yml",
    ),
    Artifact(
        name="career WAR (ungated Bayesian)",
        path="public/data/career_war.json",
        kind="json_field",
        field="generated_at",
        budget_hours=216,
        required=True,
        workflow="career-war.yml",
        # This artifact is ~150 days stale on arrival and cannot be rebuilt
        # today: `build_career_war.py` needs `data/hitter_seasons.parquet`,
        # which has no automated source in this repo. Holding the whole check
        # red on a blocker with no available fix is how an alarm stops being
        # read, and then it cannot catch the artifacts it still can. So it
        # reports every run and does not fail the job. Delete this line the
        # moment that input has a source — the 216h budget above is already
        # the right one for the weekly cadence.
        blocked_by="#87 — hitter_seasons.parquet has no automated source",
    ),
)


@dataclass
class Result:
    artifact: Artifact
    status: str                      # OK | STALE | ABSENT | MISSING | ERROR
    timestamp: datetime | None = None
    age_hours: float | None = None
    detail: str = ""

    @property
    def failed(self) -> bool:
        if self.status == "ABSENT":
            return self.artifact.required
        if self.status == "STALE" and self.artifact.blocked_by:
            return False
        return self.status != "OK"


def parse_time(text: str) -> datetime:
    """ISO 8601 to an aware UTC datetime. A naive timestamp is read as UTC."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    stamp = datetime.fromisoformat(cleaned)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def read_json_field(path: Path, field: str) -> datetime:
    with path.open() as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or field not in doc:
        raise ValueError(f"no {field!r} field")
    value = doc[field]
    if not isinstance(value, str):
        raise ValueError(f"{field!r} is {type(value).__name__}, not a string")
    return parse_time(value)


def newest_by_filename(directory: Path, pattern: str, ts_format: str
                       ) -> tuple[datetime, str]:
    """Newest timestamp encoded in a filename, and the filename it came from.

    The name is cut at the first dot, so `2026-09-03T0058Z.jsonl.gz` is parsed
    as `2026-09-03T0058Z`. Names that do not parse are skipped rather than
    fatal — the directory is allowed to hold a README.
    """
    found: list[tuple[datetime, str]] = []
    unparsed = 0
    for entry in sorted(directory.glob(pattern)):
        stem = entry.name.split(".", 1)[0]
        try:
            stamp = datetime.strptime(stem, ts_format).replace(tzinfo=timezone.utc)
        except ValueError:
            unparsed += 1
            continue
        found.append((stamp, entry.name))
    if not found:
        raise FileNotFoundError(
            f"no file matching {pattern} with a {ts_format} name"
            + (f" ({unparsed} unparsable name(s) skipped)" if unparsed else ""))
    return max(found)


def check_artifact(artifact: Artifact, root: Path, now: datetime) -> Result:
    target = root / artifact.path

    if artifact.kind == "newest_filename":
        if not target.is_dir():
            return Result(artifact, "ABSENT", detail="directory does not exist")
        try:
            stamp, name = newest_by_filename(
                target, artifact.pattern, artifact.ts_format)
        except (FileNotFoundError, ValueError) as exc:
            return Result(artifact, "MISSING", detail=str(exc))
        detail = f"newest {name}"
    elif artifact.kind == "json_field":
        if not target.parent.is_dir():
            return Result(artifact, "ABSENT", detail="directory does not exist")
        if not target.is_file():
            return Result(artifact, "MISSING", detail="file does not exist")
        try:
            stamp = read_json_field(target, artifact.field)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            return Result(artifact, "ERROR", detail=str(exc))
        detail = f"{artifact.field}={stamp.isoformat()}"
    else:  # pragma: no cover - guards a typo in the table above
        raise ValueError(f"unknown artifact kind {artifact.kind!r}")

    age_hours = (now - stamp).total_seconds() / 3600.0
    status = "STALE" if age_hours > artifact.budget_hours else "OK"
    return Result(artifact, status, timestamp=stamp, age_hours=age_hours,
                  detail=detail)


def check_all(root: Path, now: datetime,
              artifacts: tuple[Artifact, ...] = ARTIFACTS) -> list[Result]:
    return [check_artifact(a, root, now) for a in artifacts]


def format_age(hours: float | None) -> str:
    if hours is None:
        return "-"
    sign = "-" if hours < 0 else ""
    total_minutes = int(abs(hours) * 60 + 0.5)
    return f"{sign}{total_minutes // 60}h{total_minutes % 60:02d}m"


def format_report(results: list[Result], now: datetime) -> str:
    name_w = max([len(r.artifact.name) for r in results] + [len("artifact")])
    lines = [
        f"Production data freshness at {now.isoformat()}",
        "",
        f"{'artifact':<{name_w}}  {'age':>8}  {'budget':>7}  status",
        f"{'-' * name_w}  {'-' * 8}  {'-' * 7}  {'-' * 6}",
    ]
    for r in results:
        lines.append(
            f"{r.artifact.name:<{name_w}}  {format_age(r.age_hours):>8}  "
            f"{format_age(r.artifact.budget_hours):>7}  {r.status}")
    lines.append("")
    for r in results:
        if r.failed or r.status != "OK":
            lines.append(
                f"{r.status} {r.artifact.name} ({r.artifact.path}, "
                f"{r.artifact.workflow}): {r.detail}")
    failures = [r for r in results if r.failed]
    # An artifact that is stale but `required=False` still has to appear in the
    # summary. Printing "All N artifacts within budget" directly under a STALE
    # line would be the same kind of quiet lie this script exists to catch.
    noted = [r for r in results if r.status != "OK" and not r.failed]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} of {len(results)} artifacts out of budget. "
                     "Check the workflow's run history: GitHub drops scheduled "
                     "runs without a trace.")
    elif noted:
        lines.append(f"{len(results) - len(noted)} of {len(results)} artifacts "
                     "within budget; none that can be refreshed is out of it.")
    else:
        lines.append(f"All {len(results)} artifacts within budget.")
    for r in noted:
        lines.append(f"  {r.artifact.name} is stale and known-blocked "
                     f"({r.artifact.blocked_by}) — reported, not failing.")
    return "\n".join(lines)


def as_json(results: list[Result], now: datetime) -> dict:
    return {
        "now": now.isoformat(),
        "ok": not any(r.failed for r in results),
        "artifacts": [
            {
                "name": r.artifact.name,
                "path": r.artifact.path,
                "workflow": r.artifact.workflow,
                "status": r.status,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "age_hours": round(r.age_hours, 3) if r.age_hours is not None else None,
                "budget_hours": r.artifact.budget_hours,
                "required": r.artifact.required,
                "failed": r.failed,
                "detail": r.detail,
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT,
                    help="repository checkout to inspect (default: this one)")
    ap.add_argument("--now", type=parse_time, default=None,
                    help="ISO 8601 timestamp to age against (default: now, UTC)")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output instead of the table")
    args = ap.parse_args(argv)

    now = args.now or datetime.now(timezone.utc)
    results = check_all(args.root, now)

    if args.json:
        print(json.dumps(as_json(results, now), indent=2))
    else:
        print(format_report(results, now))

    return EXIT_STALE if any(r.failed for r in results) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
