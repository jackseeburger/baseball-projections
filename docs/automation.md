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
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | ❌ add | reading/writing Statcast + snapshots. Use these names: the platform injects its own `AWS_*` pair (not usable, not yours) |
| `R2_ENDPOINT_URL`, `R2_BUCKET_NAME` | ❌ add | boto3/DuckDB against R2 |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | ❌ add | sessions launching training/sim runs |
| `WANDB_API_KEY` | ❌ add | reading run status, logging |
| `ODDS_API_KEY` | ❌ add | station M market-line archive (The Odds API, free tier) |

Everything in stations D–H runs **without any of these** (MLB Stats API and
Chadwick are public), which is why the simulator, site, and nightly job shipped
on day one. The nightly sim runs on GitHub Actions, not Modal, for that reason;
Modal is reserved for Bayesian refits.

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
