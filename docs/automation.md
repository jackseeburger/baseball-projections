# Cloud Automation — How This Project Runs in the Background

Companion to [roadmap.md](roadmap.md). The roadmap says *what* gets built and in what
order; this doc says *who does the work* when nobody is at a keyboard. It is also the
template for future factories (financial markets etc.) — see the last section.

## Two layers, strictly separated

The core design decision: **deterministic production jobs and agentic development work
are different layers with different tools.** Mixing them is how automated systems become
unreliable and expensive.

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2 — Development factory (agentic, Claude Code cloud)      │
│                                                                 │
│  Scheduled Routine ──► fresh cloud session ──► picks next       │
│  queued task ──► implements on a branch ──► runs backtest       │
│  harness ──► opens PR with before/after metrics ──► drives      │
│  CI green ──► human reviews & merges from phone                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ merges change the code that Layer 1 runs
┌───────────────────────────▼─────────────────────────────────────┐
│ Layer 1 — Production (deterministic, Modal cron, no LLM)        │
│                                                                 │
│  Nightly: pull results/statcast delta ──► refresh standings &   │
│  schedule ──► run season Monte Carlo ──► write dated JSON       │
│  snapshot to R2 ──► deploy static site                          │
│  Weekly: refit Bayesian components, log to W&B                  │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1 — Production (Modal cron)

The nightly job (roadmap 3.2) is plain scheduled code — `modal.Cron` on functions in
`modal_functions/`. No agent involved: it must run at 4am whether or not anyone is
paying attention, cost cents, and fail loudly.

- **Nightly** (`@app.function(schedule=modal.Cron(...))`): incremental Statcast/results
  pull → standings + remaining schedule from `statsapi.mlb.com` → season simulation →
  write `projections/snapshots/YYYY-MM-DD/*.json` to R2 (dated, never overwritten —
  roadmap 3.1) → copy to `latest/` → trigger static deploy.
- **Weekly**: component refits with W&B logging. Refits are the expensive step and do
  not need to be nightly (roadmap 3.2).
- **Staleness alarm**: the job writes a `last_success` timestamp; the site shows it
  prominently, and the Layer-2 watchdog (below) alerts when it goes stale. This is the
  "nightly job fails silently during October" risk from the roadmap.

#### Where this actually runs, and how it failed

The plan above says `modal.Cron`. What shipped is GitHub Actions — `nightly-odds.yml`,
`market-snapshot.yml`, `statcast-ingest.yml` and `modal-refit.yml` — because the
simulator is cheap CPU that needs no credentials, and only the Bayesian refit needs
Modal at all (the refit workflow is a trigger: the runner authenticates and starts the
job, and the sampling happens on Modal's hardware).

On 2026-09-03 we went looking for why the site was serving a board from the previous
day and found that the schedules were not running. The evidence, counting only slots
that came due *after* each workflow reached `main`:

| Workflow | Live since | Slots due | Runs | Lateness |
|---|---|---|---|---|
| Nightly playoff odds | 09-01 18:51Z | 3 | 1 | 4h25m |
| Market snapshot | 09-02 19:27Z | 2 | 1 | 1h58m |
| Statcast ingest | 09-02 20:26Z | 0 | 0 | — |
| Modal refit | 09-02 19:43Z | 0 | 0 | — |

Two runs out of five due slots, and both of those hours late. The last two rows are
not evidence of anything — their first slot had not come due yet — and it is worth
saying so, because `run_number: 1` on every workflow looks like total failure until
you check which slots were actually live.

Two hypotheses were ruled out. The repository is public, so Actions minutes are free
and unlimited — this is not an exhausted quota. And the one delayed run's `created_at`
equals its `run_started_at`, with a `head_sha` committed 25 minutes earlier, so the run
was *created* 4h25m late rather than sitting in a queue from its nominal slot: GitHub
delivered the schedule event late, it did not hold a job.

That points at GitHub's documented behaviour — the `schedule` event is best effort and
is delayed or dropped under load, with the start of an hour being the worst window.
Every slot this repo had was on `:00`, `:15`, `:30` or `:45`, which is exactly where
the load is. **This remains a hypothesis.** It is consistent with the evidence and it
is GitHub's own advice, but we cannot prove it from outside, and five slots is a small
sample. The mitigations are worth taking either way:

1. **Every cron minute is now odd and non-round** (`:07`, `:11`, `:19`, `:23`, `:29`,
   `:37`, `:41`, `:43`, `:47`, `:53`). Costs nothing, and is the documented remedy.
   `tests/test_scripts/test_check_schedules.py` asserts it for every slot in the
   repo, so it cannot quietly regress.
2. **The nightly job has three slots instead of two.** The script never overwrites a
   dated snapshot, so a redundant run only refreshes `latest.json`.
3. **Slots are spaced at least an hour apart across all four workflows.** They share
   the `data-commits` concurrency group, and a group with `cancel-in-progress: false`
   cancels a *pending* run when a third one arrives. Nothing has been lost this way
   yet, but two jobs on adjacent round minutes would have made it possible.
4. **The staleness alarm above finally exists** — `scripts/check_freshness.py` and
   `freshness-check.yml`, on its own odd-minute schedule, outside the `data-commits`
   group. It reads timestamps from *inside* the data rather than file mtimes (a fresh
   clone gives every file the same mtime) and opens a single GitHub issue when
   anything is past its budget.
5. **And it has a faster sibling** — `scripts/check_schedules.py` and
   `schedule-watchdog.yml`, which ask the Actions API directly instead of inferring
   from artifact age. See "Two alarms, two questions" below.

The alarm is the part that matters. The cron minutes are a guess at a cause; the
watchdog is what makes the next silent failure loud, whatever causes it. Nothing
caught this one on its own — it was found by hand, a day late, because the promised
alarm had been written down and never built.

#### Two alarms, two questions

The age check was slow by construction. It infers a dropped run from the *age of
the output*: on the outage it was built for, the last good board was written
09-02 13:41Z, so a 36-hour budget would not have fired until 09-04 01:41Z —
roughly sixteen hours after the board actually went stale on the morning of the
3rd. The budget cannot simply be tightened to close that gap: the nightly's three
slots span 09:23 to 14:53, so the worst *legitimate* gap between successful runs
is about 29h30m, and a budget much under 36h cries wolf on a schedule that is
merely late.

So we stopped inferring. `scripts/check_schedules.py` + `schedule-watchdog.yml`
ask the Actions API the question directly, per workflow: **given this workflow's
own cron slots, has a run actually started since the last one came due?**

- **The slots come out of the workflow YAML**, never restated in the checker. A
  second copy would drift the first time someone edits a schedule, and a watchdog
  watching last month's schedule is worse than none. Every workflow with a
  `schedule:` block is discovered and watched, which is how `statcast-ingest.yml`
  and `modal-refit.yml` — invisible to the age check, because they write to R2,
  Modal and W&B rather than to the repo — are covered for the first time.
- **"Never due" is not "dropped."** A slot only counts if it came due *after* the
  workflow's `created_at`. That is the mistake we nearly made by hand: two of the
  four workflows had never had a slot come due, and their `run_number: 1` was not
  evidence of anything. The check calls that `PENDING` and does not fail.
- **Grace: 6 hours**, before a slot counts as due. The floor is the worst delay we
  have measured — 4h25m — because anything at or under that would have paged on a
  run that did arrive, on the very sample we are calibrating against; 6h is ~35%
  headroom over it. The ceiling is detection latency, and nothing else: the check
  asks whether *anything* has run since the last due slot, so a later slot firing
  clears an earlier one, and the arithmetic lands the alarm the same day anyway.
  A nightly that loses 09:23 and everything after it is reported at 16:43 UTC,
  against 01:41 the *next* day for the age check. Both fixtures in
  `tests/fixtures/actions/` pin this down: the real outage fires, and a 4h25m-late
  run does not. `--grace-hours` replays either at a different setting.
- **It also sees two things the age check never could.** A run that started and
  failed (or was cancelled by the `data-commits` group) is reported instead of
  being left to email; and a *disabled* workflow is reported loudly, which matters
  because GitHub disables scheduled workflows in public repos after 60 days
  without repository activity, silently.

**Both alarms ship, and neither is redundant.** The API check knows whether the
job RAN. The age check knows whether it PRODUCED. A job that runs green and writes
nothing — a bad input, a swallowed exception, the builder falling back to
yesterday's data — is invisible to the API check and obvious to the age check.
They also fail in different ways: the API check needs a token, network and a
correct cron parser, while the age check needs a checkout and nothing else. What
did change is their division of labour: the age check is no longer the thing that
catches a dropped run, so its budgets are deliberately loose (36h nightly, 22h
market) and tuned only to never cry wolf.

The market budget was 20h and is now 22h. The original was computed against the
old round-minute slots; with 10:41/16:37/23:11 a single dropped slot already opens
an 18h04m gap, and 20h left less slack than the 1h58m late start we have actually
observed.

**What neither alarm can see.** Whether the numbers are *right* — both check that
work happened, not that it was correct; that is the backtest harness's job. Nor
does anything yet age the R2 and Modal outputs of `statcast-ingest.yml` and
`modal-refit.yml`: those two are now watched for *did it run*, but a refit that
runs and writes a garbage posterior is still invisible. And the watchdogs are not
symmetric: `schedule-watchdog.yml` watches `freshness-check.yml` like any other
scheduled workflow, so if the age check goes quiet we hear about it — but nothing
watches the schedule watchdog itself, since the run doing the asking is itself a
run since the last slot. That is why it has twelve slots a day rather than one:
losing all of them to dropped `schedule` events is the failure mode it cannot
report, and the only remaining backstop is a human noticing the issue tracker has
gone silent.

### Layer 2 — Development factory (Claude Code cloud)

Agentic sessions do what cron cannot: implement roadmap items, diagnose why last
night's numbers look wrong, fix broken pipelines, and open PRs. The machinery, all of
which exists in Claude Code on the web today:

1. **Work queue = GitHub issues.** Each station task from
   [architecture.md](architecture.md) becomes an issue labeled `factory:ready`,
   naming its station and the baseline it must beat. Issues are the durable state between sessions —
   cloud containers are ephemeral, so nothing lives only in a conversation.
2. **Scheduled Routines** (Claude Code cloud triggers) fire on a schedule and spawn a
   *fresh* session in this environment with a standing prompt. Two routines:
   - **Factory worker** (daily): pick the top `factory:ready` issue, implement it on a
     `claude/...` branch, run the backtest harness, open a PR whose description contains
     the before/after scores vs. baselines, subscribe to the PR and drive it to green.
     One issue per session — small reviewable PRs, not marathons.
   - **Ops watchdog** (daily, mornings): check the freshness of last night's snapshot in
     R2 and the latest W&B runs; if anything is stale or red, diagnose and either fix
     (small, in-scope) or open a `factory:incident` issue with findings.
3. **PR subscriptions** wake sessions on CI failures and review comments, so PRs get
   driven to mergeable without polling.
4. **The human is the merge gate.** Nothing lands on `main` without review. The factory
   produces *reviewed* throughput, not unattended commits — that is what keeps it safe
   to leave running.

### The backtest harness is the keystone

An autonomous worker is only useful if it can verify its own work numerically. Roadmap
0.2 (`src/eval/` — component-wise log loss / MAE / RMSE / calibration vs. Marcel,
previous-season, and league-average baselines) is therefore not just a modeling task:
**it is the factory's objective function.** Until it exists, agent sessions can write
code but cannot know whether a change helped, so Phase 0 is built with a human in the
loop and the factory goes progressively hands-off as the harness lands. Every factory
PR must include harness output; a PR without numbers is an automatic "changes
requested."

Fast checks also gate every PR via GitHub Actions CI: `pytest`, lint, and a smoke
backtest on a small data slice. Full refits stay on Modal — CI runners cannot do MCMC
in reasonable time.

## Environment requirements

Secrets live in the Claude Code cloud **environment settings** (and in Modal secrets
for Layer 1), never in the repo:

| Credential | Status | Needed for |
|---|---|---|
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | ✅ verified | reading/writing Statcast + snapshots. Use these names: the platform injects its own `AWS_*` pair (not usable, not yours) |
| `R2_ENDPOINT_URL`, `R2_BUCKET_NAME` | ✅ verified | boto3/DuckDB against R2 — go through `src/data/r2.py`, which strips the bucket path Cloudflare appends to the endpoint it shows you |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | ⚠️ set, unusable from cloud sessions | Modal's client is gRPC, which the session proxy does not pass. Refits are launched from **GitHub Actions** (add the same two names as Actions secrets) or a laptop, never from a Claude session |
| `WANDB_API_KEY` | ✅ verified | reading run status, logging |
| `ODDS_API_KEY` | ✅ wired | station M sportsbook lines (The Odds API, 500 req/month). Lives in GitHub Actions secrets; `market-snapshot.yml` pulls books on the two daytime slots only (4 requests each, ~240/month) |

Everything in stations D–H and the exchange half of station M runs **without
any of these** (MLB Stats API, Chadwick, Kalshi and Polymarket market data are
all public), which is why the simulator, site, nightly job, and market archive
shipped before any key existed. Both scheduled jobs run on GitHub Actions, not
Modal, for that reason; Modal is reserved for Bayesian refits.

Two GitHub Actions jobs commit data to `main` today, serialized by a shared
`concurrency` group:

| Workflow | Schedule (UTC) | Writes |
|---|---|---|
| `nightly-odds.yml` | 09:23, 12:07, 14:53 | `public/data/playoff_odds/YYYY-MM-DD.json` + `latest.json` |
| `market-snapshot.yml` | 10:41, 16:37, 23:11 | `data/market/snapshots/<ts>.jsonl.gz` (immutable) + `public/data/market/latest.json` |
| `statcast-ingest.yml` | 13:19 | `statcast/statcast_<year>.parquet` and `pa_outcomes/pa_outcomes_<year>.parquet` in R2, then the Modal volume |
| `modal-refit.yml` | Mondays 07:29 | Modal training runs; diagnostics to W&B |

Every minute above is odd and non-round on purpose, and no two production slots
are within an hour of each other; see the failure write-up above. Two watchdogs
run outside the `data-commits` group and commit nothing:

| Workflow | Schedule (UTC) | Asks |
|---|---|---|
| `schedule-watchdog.yml` | every 2h at :43 | did each workflow run since its last due cron slot (Actions API) |
| `freshness-check.yml` | 05:53, 13:23, 21:47 | is any committed artifact older than its budget (timestamps inside the data) |

**Keep the current season flowing.** The Statcast archive in R2 ran 2015–2025
and was uploaded before the 2026 season began, so every refit trained on a
season that had already ended — the models could not know anything about the
year they were projecting. `statcast-ingest.yml` closes that: it pulls the
season from Baseball Savant (public, no key), rebuilds PA outcomes, writes both
to R2, and pushes the PA file to the Modal volume the training functions read.
Run it before a refit, or the refit is stale by construction.

Git is the interim archive because it needs no credentials; once the `R2_*`
keys exist the snapshots move to R2 and the repo keeps only `latest.json`.

**Never commit credentials.** Early scripts hard-coded R2 keys as env-var
fallbacks; those were removed and the key must be treated as leaked and
rotated. Scripts now fail loudly if the `R2_*` variables are missing.

Network policy must allow: `modal.com`, `api.wandb.ai`, the R2 endpoint,
`statsapi.mlb.com`, `baseballsavant.mlb.com`, `github.com`, PyPI.

A `SessionStart` hook (or setup script) should install requirements so every fresh
factory session can run tests immediately.

## Rollout order

1. **Now (Phase 0, human-in-the-loop):** birthdates → backtest harness → re-score →
   binomial aggregation → pitcher random effect → cleanup. Sessions like this one do
   the work; the harness is the deliverable that unlocks autonomy.
2. **Phase 0 done:** seed the issue queue from the roadmap, add CI, turn on the
   factory-worker Routine for Phase 1–2 items.
3. **Phase 3:** stand up the Modal nightly cron + snapshot archive, turn on the ops
   watchdog Routine.
4. **October:** factory shifts to operate mode — watchdog + incident issues, small
   fixes, retrospective logging (roadmap Phase 4).

## The reusable factory template

What generalizes to the next domain (e.g. financial markets) is not the baseball code —
it is the skeleton:

1. **Data pipeline → object store** (partitioned Parquet in R2, DVC-tracked, queried
   with DuckDB) with an incremental daily pull.
2. **An honest eval harness with dumb baselines** — the domain-specific Marcel
   equivalent. This is always built first; nothing is trusted without it.
3. **Serverless training** (Modal) with experiment tracking (W&B), triggered by cron
   for production and by agent sessions for experiments.
4. **A work queue in GitHub issues + scheduled agent sessions** that turn queue items
   into PRs with eval numbers attached, and a human merge gate.
5. **A public scoreboard** (the accuracy page) so the system is accountable to
   out-of-sample reality.

Standing up a new factory = new repo from this skeleton + new data pipeline + new eval
harness. Layers 1 and 2 carry over unchanged.
