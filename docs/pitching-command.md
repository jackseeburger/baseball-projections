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

## Results (2026-09-09, run after the pre-registration above)

**Stopped at the vacuity gate.** Stage 1 ran in full and prediction 1
holds by three to four times its margin; the vacuity check failed, so
stage 2 was not run. Predictions 2–5 are unscored, not failed. Nothing is
served. Evidence: `data/eval/pitching_command_stage1.json`. There is
deliberately no `pitching_command_stage2.json`.

### Vacuity: FAIL

Pooled year-over-year correlation of the pitcher-season command aggregate,
pitchers with ≥ 1,000 pitches in both seasons, every consecutive pair
2015→2026 (1,518 pairs):

| aggregate | r | floor |
| --- | --- | --- |
| `cmd_resid` (CSW residual per pitch) | **0.397** | 0.45 |
| `cs_resid` (called-strike residual per take) | **0.330** | 0.45 |

Recorded as a diagnostic and not a rescue: the within-pair correlations
are 0.42–0.76 (median ≈ 0.62), and with each season's league level removed
the pooled figure is 0.623 (`cs_resid` 0.585). The walk-forward design fits
a different stage-1 model per scored season, so the residual's league mean
moves (+0.015 in 2017, −0.008 in 2026) and pooling pairs whose centres sit
in different places attenuates the pooled r. The check was not redefined
after seeing it; the verdict is the pre-registered number. For contrast the
*levels* carry fine (`cmd_csw` per pitch r = 0.757, `zone_share` 0.561,
`waste_share` 0.650); it is the stuff-differenced residual that does not.

### Stage 1, walk-forward by season (out-of-sample log-loss)

CSW | pitch:

| season | pitching | stuff | location-only | pitch-type only |
| --- | --- | --- | --- | --- |
| 2017 | **.4685** | .5403 | .4747 | .5847 |
| 2018 | **.4579** | .5352 | .4650 | .5886 |
| 2019 | **.4748** | .5341 | .4813 | .5886 |
| 2020 | **.4711** | .5377 | .4773 | .5944 |
| 2021 | **.4689** | .5350 | .4749 | .5904 |
| 2022 | **.4727** | .5352 | .4794 | .5884 |
| 2023 | **.4721** | .5359 | .4807 | .5897 |
| 2024 | **.4698** | .5346 | .4782 | .5877 |
| 2025 | **.4699** | .5331 | .4783 | .5859 |
| 2026 | **.4668** | .5306 | .4776 | .5838 |

Called strike | taken:

| season | pitching | stuff | location-only | pitch-type only |
| --- | --- | --- | --- | --- |
| 2017 | .1719 | .3459 | **.1711** | .6185 |
| 2018 | **.1496** | .2989 | .1554 | .6194 |
| 2019 | **.1493** | .2839 | .1549 | .6152 |
| 2020 | **.1512** | .2823 | .1565 | .6154 |
| 2021 | **.1429** | .2752 | .1502 | .6150 |
| 2022 | **.1409** | .2766 | .1488 | .6156 |
| 2023 | **.1386** | .2766 | .1467 | .6140 |
| 2024 | **.1385** | .2699 | .1459 | .6146 |
| 2025 | **.1381** | .2748 | .1470 | .6108 |
| 2026 | **.1225** | .2721 | .1334 | .6088 |

### Scoring the predictions

1. **Holds.** `pitching` beats stuff-only on every season by .060–.077 of
   log-loss on CSW against a pre-registered 0.02, and by .150–.175 on
   called-strike-given-taken. Location-only beats pitch-type-only
   everywhere, by .105–.123 on CSW and .445–.475 on called strikes.
2.–5. **Unscored.** All four are stage-2 claims and the pre-registration
   makes the vacuity check a gate that fires before stage 2. BAS-76 does
   not know whether BB/BF clears with a real covariate share, whether it
   survives August and the high-exposure tercile, whether K/BF and HR/BF
   stay under their ceilings, or what §3 would say. Nothing is served,
   which is the outcome prediction 5 named for three of the four
   components, reached for a different reason.

### Things that complicate the reading

- **The failure is about the residual, not about location.** Stage 1 says
  location is enormously predictive of the pitch's own outcome; the
  season-to-season correlation says the pitcher-level average of
  location's *marginal* contribution over the stuff model does not carry.
  The level does. Differencing against stuff was the right choice for this
  ticket's question — the served engine already has the stuff score — and
  it is the choice that makes the aggregate fail its own floor. A
  follow-up that pre-registers a *level* aggregate with the stuff
  covariates as explicit controls in the stage-2 fit is a different ticket
  and needs its own pre-registration; it was not reached for here, after
  seeing this table.
- **Stuff is not location-blind.** The BAS-71 feature set carries release
  position, extension and the full velocity/acceleration vector, which
  together determine the trajectory and therefore plate location. The
  "location-free" stuff arm reaches AUC .966 and log-loss .272 on
  called-strike-given-taken in 2026, a target that is almost purely about
  where the pitch crossed. The command residual is therefore a lower bound
  on what location contributes, and some of what `stuff_additive` is paid
  for on the walk rates may be command wearing stuff's name. The same note
  is in `docs/pitching-stuff.md`.
- `pitching_command_monthly.parquet` is 40,420 rows, 1.75 MB, 14 columns;
  it is not refreshed nightly because nothing consumes it.
