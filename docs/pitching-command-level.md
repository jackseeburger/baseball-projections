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

## Results (2026-09-10, run after the pre-registration above)

**BB/BF clears.** The command level block, fitted with the six stuff
aggregates as explicit controls, beats tuned pitcher Marcel on BB/BF by
4.5% of MAE and beats the served `stuff_additive` engine by 1.5% at
t −3.6, which is a covariate-only share above §3's 1.0% floor at
|t| > 2.0. It adds nothing to K/BF or HR/BF, and the permuted control
lands on the stuff-only control. Evidence:
`data/eval/pitching_command_stage2.json`. Nothing is wired in this pass;
the Serving section below is the pre-registration for that.

### Setup

4,954 pitcher-cells, 955 pitchers, 2022–2026 × May 1 / Jul 1 / Aug 1,
`min_trials` 100, SE clustered by pitcher (t) and by season-cutoff cell
(t_cell; with 15 clusters it is downward-biased and runs 1.5–2× t, so read
t(pitcher)). Command block `cmd_csw`, `zone_share`, `waste_share`; the
artifact has no `edge_share` (Statcast's `shadow_share` is that region and
is run as a labelled sensitivity). Hyperparameters chosen on 2019 + 2021
by the rule already in `run_command_backtest.py`: weights (1.0, 0.35, 0.1),
ballast 50, an interior optimum (stuff's is a grid corner). The
`stuff_additive` control reproduces BAS-79/80's committed serving table on
the identical cells to within 0.02 points on every component.

### Vacuity: PASS

Pitcher-season aggregate, ≥ 1,000 pitches both years, 1,518 pairs:
`cmd_csw` r 0.757 (floor 0.45), `zone_share` 0.561, `waste_share` 0.650;
BAS-76's residuals 0.397 / 0.330 reproduce. The level barely moves under
season demeaning (0.742), unlike the residual (0.397 → 0.623): the pooling
artefact BAS-76 documented is a residual problem, not a level one.

### The tables (% of the base arm's own MAE; negative is better)

BB/BF (base MAE .01792):

| arm | vs | Δ MAE | % | t (pitcher) | t (cell) | W |
| --- | --- | --- | --- | --- | --- | --- |
| command_level | marcel_pitcher_tuned | −.000824 | −4.60 | −6.51 | −8.20 | .568 |
| command_level_recal | marcel_pitcher_tuned | −.000551 | −3.08 | −5.70 | −6.93 | .565 |
| command_level | _recal | −.000272 | −1.57 | −3.73 | −7.50 | .514 |
| **command_level_additive** | **marcel_pitcher_tuned** | −.000812 | **−4.54** | **−6.47** | −7.66 | .565 |
| stuff_additive | marcel_pitcher_tuned | −.000548 | −3.06 | −5.83 | −7.31 | .567 |
| **command_level_additive** | **stuff_additive** | −.000264 | **−1.52** | **−3.64** | −6.72 | .522 |
| stuff_then_command_level | stuff_additive | −.000169 | −0.97 | −2.77 | −6.40 | .527 |
| command_level_shuffled | _recal | +.000028 | +0.16 | +2.17 | +1.28 | .492 |

(BB+HBP)/BF has the same shape: −4.10% vs Marcel (t −6.00), −1.46% vs
`stuff_additive` (t −3.54). K/BF: `command_level_additive` vs
`stuff_additive` +0.05% (t 0.54). HR/BF: −0.26% (t −0.90).

Covariate-only share vs `_recal` and the incremental over `stuff_additive`
are the same number with the baseline pinned at 1 (the additive stuff-only
recalibration control is byte-for-byte the served fit, asserted in a test):

| component | share | t | clears 1.0% at |t| > 2.0 | two-stage reading | t |
| --- | --- | --- | --- | --- | --- |
| K/BF | +0.05% | 0.54 | no | −0.09% | −1.60 |
| **BB/BF** | **−1.52%** | **−3.64** | **yes** | −0.97% | −2.77 |
| (BB+HBP)/BF | −1.46% | −3.54 | yes | −0.89% | −2.60 |
| HR/BF | −0.26% | −0.90 | no | −0.02% | −0.06 |

§6 split on BB/BF, `command_level_additive`:

| slice | vs Marcel | t | vs stuff_additive | t |
| --- | --- | --- | --- | --- |
| exposure T1 (low) | −3.22% | −3.93 | −1.98% | −3.15 |
| exposure T2 | −5.56% | −5.69 | −1.41% | −2.34 |
| exposure T3 (high) | −4.67% | −4.14 | −1.15% | −1.90 |
| cutoff May 1 | −5.28% | −6.46 | −1.70% | −3.28 |
| cutoff Jul 1 | −4.06% | −5.19 | −1.53% | −3.22 |
| cutoff Aug 1 | −3.01% | −3.05 | −0.91% | −1.56 |

### Scoring the predictions

1. **FAILS on the band, holds on every other clause.** BB/BF
   `command_level_additive` is −4.54% (t −6.47) against a predicted −1.5
   to −3%: it overshoots the band in the good direction. The covariate-only
   share, −1.52% at t −3.64, clears the floor. (BB+HBP)/BF is the same
   shape. Scored as written the prediction fails; the substance holds.
2. **Holds against the baseline, marginal against the served engine.**
   August −3.01% (t −3.05) and the high-exposure tercile −4.67% (t −4.14)
   both survive vs Marcel. Against `stuff_additive`, August is −0.91%
   (t −1.56) and the high tercile −1.15% (t −1.90), each just under a bar.
   At stuff's pinned hyperparameters the same slices are −1.14% (t −2.24)
   and −1.60% (t −3.15) and clear. The verdict flips on the hyperparameter.
3. **Holds on the pre-registered arm.** Joint fit −1.52% (t −3.64) ≥ 1.0%
   at |t| > 2.0. The two-stage reading (stuff coefficients frozen, the
   served engine's prediction as the base) is −0.97% (t −2.77), 0.03
   points under.
4. **Holds.** K/BF incremental +0.05% (t 0.54); HR/BF incremental −0.26%
   (t −0.90). `stuff_additive` alone already carries HR/BF's −2.75% vs
   Marcel; nothing command-shaped moves it.
5. **Holds.** Under §3, `p_bb_rate` is the only served pitcher column that
   clears; (BB+HBP)/BF clears too but is station E's rate, not a served
   column. Nothing was wired.

### Things that complicate the reading

- **The two blocks are not independent measurements.** The stage-1 model
  that produced `cmd_csw` reads the stuff features; on holdout cells
  `cmd_csw` has R² 0.53 on the six stuff controls. The joint prediction is
  fine; the individual coefficients are not separately interpretable.
- **Predictions 2 and 3 both sit on the 1.0% floor from opposite sides**
  (August share 0.91% tuned vs 1.14% pinned; two-stage incremental 0.97%
  tuned vs 0.67% pinned). The floor BAS-82 adopted so that decisions stop
  flipping on ±0.1 of a t is now the thing they flip on. The pre-registered
  arm is the joint additive fit at its own tuned setting, and that is the
  one scored.
- **Per season the BB/BF share is negative in all five holdout seasons**
  (−0.65% to −2.73%) but individually significant only in 2022 (t −2.72).
  The pooled −1.52% rests on five seasons the way BAS-79's K/BF did.
- **Both hyperparameters are interior**, unlike stuff's corner; a
  slower-moving skill wants a longer window and more shrinkage, but the
  two arms are not on one setting.
- **The free arm buys almost nothing over the additive one** (−4.60% vs
  −4.54%), so unlike BAS-79 there is no baseline-rescaling story: the
  additive stuff control gets −3.06% and command adds −1.52% on top.

## Serving (pre-registered 2026-09-10 09:30 UTC, before any wiring; BAS-88)

**What ships.** On the pitcher side, `p_bb_rate` moves from
`stuff_additive` to `stuff_additive + command_level_additive` (the joint
additive fit above, at its tuned hyperparameters: weights (1.0, 0.35, 0.1),
ballast 50, pinned in code). `p_k_rate`, `p_hr_rate` and `p_babip` are
unchanged. The command monthly artifact joins the nightly path exactly as
stuff's did (`scripts/build_pitching_command.py --update-season 2026`
before the projection step, a JSON sidecar watched by
`check_freshness.py` at 36 h, month-lagged features through the last month
boundary, `command_features_through` stamped in the document beside
`stuff_features_through`), with the same honest fallback: a component
whose command fit cannot be built falls back to `stuff_additive`, logged.

**Predictions for the served arm.**

1. The served fit, trained walk-forward on cells strictly before 2026 and
   scored on 2026's May/Jul/Aug cutoffs, beats `stuff_additive` on BB/BF by
   ≥ 1.0% of MAE with the covariate-only share ≥ 1.0% at |t| > 2.0 (the
   effect floor), replicating the pooled result on the served season.
2. The served document's `p_bb_rate` column changes for ≥ 80% of pitchers
   by less than 0.010 in absolute rate, and the league-average projected
   BB/BF moves by less than 0.002: command is a re-ranking, not a level
   shift.
3. Downstream, the station E run environment's team walk rates and the
   playoff-odds board move by less than 0.5 percentage points of playoff
   probability for every club; the props ledger's K-prop pricing is
   untouched (K/BF is not re-engined).
4. **Vacuity:** if the 2026 served fit's command coefficients straddle zero
   (|t| < 2 on `cmd_csw`), the serving is withheld and this section says so.

### Served results (2026-09-10): withheld

**`p_bb_rate` stays on `stuff_additive`.** Both gate clauses of the
Serving pre-registration say withhold, independently. The engine
(`command_additive`: `fit_live_command`, `command_provider`, the rung in
`pitcher_ros`) is built, tested and dormant; the serving map is unchanged.
Evidence: `data/eval/pitching_command_serving.json`. The 2026 command
artifact was refreshed walk-forward (`--update-season 2026`) and
reproduces BAS-87's cells exactly.

The served fit, walk-forward on 2017–2025 (8,231 cells, 1,334 pitchers,
SE clustered by pitcher): `waste_share` t +7.49, `zone_share` t −2.15,
**`cmd_csw` t −0.55**; the three command coefficients are jointly
significant (χ² 118 on 3 df).

1. **FAILS on significance.** On 2026's May/Jul/Aug cutoffs (824 cells)
   `command_additive` vs `stuff_additive` on BB/BF is −1.11% of MAE at
   t −1.13. Size clears the 1.0% floor; |t| > 2.0 does not. By cutoff:
   May −1.64% (t −1.34), July −0.10% (t −0.09), August −1.13% (t −0.95).
   This reproduces BAS-87's committed 2026 row to the digit, so the
   apparatus is the arm that was scored; what fails is the single served
   season, exactly as BAS-87's note anticipated (negative in all five
   seasons, individually significant only in 2022).
2. **HOLDS**, counterfactually (682 projected pitchers): 98.4% move by
   less than 0.010 in BB/BF (bar 80%), median |Δ| 0.0025, max 0.014;
   BF-weighted league BB/BF +0.00075 (bar 0.002). A re-ranking, as
   predicted.
3. **Not scored as written.** With the serving withheld the odds board
   does not move. Counterfactual team walk rates move at most 0.32
   points of BB/BF (STL), mean 0.09; K-prop pricing untouched by
   construction.
4. **Vacuity clause fires**: `cmd_csw` at |t| 0.55 < 2.

**What complicates the reading.** The clause fires on one coefficient
while the block is not vacuous (`waste_share` alone is t +7.5). The
pre-registration wrote "command coefficients straddle zero" and
operationalised it as `cmd_csw`, which is the one covariate BAS-87 had
already flagged as unreadable on its own (R² 0.53 on the stuff controls).
That may have been the wrong operationalisation, and it is a question for
the next pre-registration, not for this table: prediction 1 fails on its
own, so the serving would be withheld either way. HR/BF picks up an
incidental −1.18% (t −2.11) over `stuff_additive` on 2026 alone, against
BAS-87's pooled −0.26%; a single-season fluctuation on a component the
gate already withheld, recorded and not acted on.

**What is deliberately not wired** until the map changes: the nightly
refresh step for the command artifact, its freshness sidecar alarm, and
the `command_features_through` stamp. A freshness alarm on an artifact
nothing served reads would mean nothing. They land with the one-line map
change when a served season clears.
