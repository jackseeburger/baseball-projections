# Command as a level, not a residual

**BAS-87.** Station A, pitcher side. Pre-registered 2026-09-10 09:00 UTC in
the Linear issue, before any run; this file mirrors that text. The
follow-up BAS-76 (`docs/pitching-command.md`) named.

## Why

BAS-76 measured pitching command from location: at the pitch, location is
a third of the reachable signal (CSW log-loss −.06 to −.08 over stuff on
every season 2017–2026). Its pitcher-level aggregate, the *residual* of
CSW over the stuff model, failed the pre-registered persistence floor
(r 0.40 vs 0.45) and stage 2 was never run. The *level* aggregates in the
same artifact persist fine (`cmd_csw` r 0.76, `zone_share` 0.56,
`waste_share` 0.65): it is the differencing that fails, not the location.
The walk rates still have no served measurement (stuff's BB/BF gain was a
level correction; BAS-80). This is the follow-up BAS-76 named, under its
own pre-registration.

## What gets built

Stage 2 only, on the existing `data/features/pitching_command_monthly.parquet`
(no stage-1 rebuild): the level aggregates `cmd_csw`, `zone_share`,
`waste_share` (and `edge_share` if present), month-lagged and summed
strictly before the cutoff exactly as `src/eval/command.py` does, as
covariates on `marcel_pitcher_tuned`, **with the stuff aggregates the
served engine uses entered as explicit controls in the same fit** so the
command coefficients are conditional on stuff. Arms: `command_level`,
`command_level_additive` (baseline pinned at 1), `_recal` (controls only,
no command), permuted command, and `stuff_additive + command_level_additive`
(incremental over the served engine). Cells 2022–2026 × May/Jul/Aug
cutoffs, SE clustered by pitcher; components K/BF, BB/BF, (BB+HBP)/BF,
HR/BF. §6 information-vs-denoising split (exposure terciles × cutoffs).

## Pre-registered predictions

1. **BB/BF clears with a real covariate share:** `command_level_additive`
   −1.5 to −3% of MAE vs `marcel_pitcher_tuned`, |t| > 2.5, and
   covariate-only share (vs the `_recal` control) ≥ 1.0% at |t| > 2.0,
   architecture §3's effect floor. (BB+HBP)/BF likewise.
2. **Information, not denoising, on BB/BF:** the gain survives the August
   cutoff and the high-exposure tercile.
3. **Incremental over the served stuff engine on BB/BF ≥ 1.0% at
   |t| > 2.0**: command adds what stuff's kinematic proxy for location did
   not carry.
4. **K/BF:** incremental over `stuff_additive` < 0.75%, |t| < 2.5; HR/BF
   moves < 1%.
5. Under §3, BB/BF (and (BB+HBP)/BF) would be served on
   `stuff_additive + command_level_additive`; nothing else.

### Vacuity check

Year-over-year r of the pitcher-season `cmd_csw` level (≥ 1,000 pitches
both years) ≥ 0.45. BAS-76 already measured 0.757, so this is recorded as
met before the run rather than a live gate; it is stated so the doc is
complete.

### Failure conditions

- Prediction 1 fails: location's pitcher-level signal is inside stuff's
  kinematic proxy already; say so, and the walk rates stay on stuff and
  Marcel.
- Prediction 3 fails with 1 holding: command replaces stuff's information
  on BB/BF rather than adding to it; the serving decision is between them,
  not both, and is a separate ticket.

## What ships

Nothing in this pass. Serving is its own ticket under architecture.md §3
with a Serving section here first.
