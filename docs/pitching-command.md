# Command — pitching layer 1 for the walk rates

**BAS-76.** Station A, pitcher side. Pre-registered 2026-09-09 17:10 UTC in
the Linear issue, before any run; this file mirrors that text.

## Why

BAS-71's labelled location-inclusive arm beat stuff on every season (whiff
log-loss .4416 vs .4490; CSW .5024 vs .5305 in 2026): roughly a third of
the reachable pitch-level signal is *where* the pitch went, not what it
was. Stuff cleared the gate on K/BF and HR/BF but could not deliver a
covariate-only gain on the walk rates (BAS-79, BAS-80). Command is the
measurement walks lack.

## What gets built (the stuff shape, reused)

- **Stage 1:** gradient-boosted models, walk-forward by season, from
  location features (`plate_x`, `plate_z` relative to the batter's
  `sz_top`/`sz_bot`, count, handedness matchup, pitch type) *with* the
  stuff features already in `src/models/stuff.py` — the `pitching` arm —
  to CSW and to called-strike-given-taken. Controls: stuff-only (BAS-71's
  model) and location-only.
- **Reduction:** per pitcher per month additive buckets of command
  *residuals* — the pitch's predicted CSW under `pitching` minus under
  `stuff` (the location contribution) — plus edge/zone/chase-region shares
  → `data/features/pitching_command_monthly.parquet`, leak-free at
  first-of-month cutoffs.
- **Stage 2:** aggregates as covariates on `marcel_pitcher_tuned`, arms
  `command`, `command_additive`, `_recal`, permuted, and **incremental over
  the served stuff engine** (`stuff_additive` + command vs
  `stuff_additive`), 2022–2026 × May/Jul/Aug, SE clustered by pitcher, for
  K/BF, BB/BF, (BB+HBP)/BF, HR/BF; the §6 split for BB/BF.

## Pre-registered predictions

1. Stage 1 `pitching` beats stuff-only out of sample on every season by
   ≥ 0.02 log-loss on CSW (BAS-71's numbers say ~0.03; this is the
   replication, not the discovery), and location-only beats
   pitch-type-only.
2. **BB/BF clears with a real covariate share**: `command_additive` −2 to
   −4% of MAE, |t| > 3, and covariate-only share ≥ 1.5% at |t| > 2.5 — the
   thing stuff could not do. (BB+HBP)/BF likewise.
3. **Information, not denoising, on BB/BF:** the gain survives the August
   cutoff and the high-exposure tercile.
4. **K/BF:** incremental over the served stuff engine < 0.75%, |t| < 2.5
   (command is the walk measurement; whiffs already have theirs). HR/BF
   moves < 1%.
5. Under architecture.md §3's covariate-share rule, BB/BF would be served
   on command; nothing else.

### Vacuity check

Year-over-year correlation of the pitcher-season command aggregate
(≥ 1,000 pitches both years) must exceed **0.45**; below that the location
field is not measuring a repeatable skill at this reduction. Report and
stop.

## What ships

Nothing in this pass. Serving is its own ticket, under the covariate-share
rule (or its BAS-82 successor), with a pre-registered Serving section here
first.
