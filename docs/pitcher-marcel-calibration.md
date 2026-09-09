# The pitcher Marcel's walk-rate level, and re-scoring stuff against a calibrated baseline

**BAS-80.** Station A, pitcher side. Pre-registered 2026-09-09, before any
run.

## Why

BAS-79 found that of the additive stuff arm's −3.05% on BB/BF, −2.20% is a
fitted intercept and only −0.87% is the covariate; for (BB+HBP)/BF the
split is −2.04% against −0.64%. A constant that a control can fit and a
measurement cannot explain is a calibration defect in the baseline —
`marcel_pitcher_tuned` projects the league walk level wrongly — and it is
being corrected under the stuff engine's name. Fix it where it lives, then
re-score the stuff arm against the corrected baseline so the serving
decision rests on measurement and not on an accident of the baseline.

## What gets built

1. **Calibrate.** `src/eval/marcel_pitcher_params.json` carries the tuned
   ballast, weights, league mode and age curve per pitcher component. Refit
   the **projected league rate** (the `LEAGUE_GRID` axis in
   `src/eval/tuning.py`) and, if the tuning window asks for it, the ballast
   for `p_bb_rate` and `p_bbhbp_rate` on the same 2019–2024 tuning window
   and coordinate search the hitter side used (BAS-52 / BAS-25), scored
   walk-forward on `scripts/run_pitcher_backtest.py`'s five cells. Diagnose
   first: is the miss a level (league rate) or a spread (ballast) problem?
   The intercept the control fitted says level; report which it is.
2. **Re-score stuff.** Run `scripts/run_stuff_backtest.py` against the
   calibrated baseline, all four arms (`stuff`, `stuff_recal`,
   `stuff_additive`, `stuff_additive_recal`), same cells, same clustering.
3. **Apply the serving rule in force now** (architecture.md §3, the
   covariate-share consequence) to the re-scored table, and change the served
   pitcher engines accordingly — including *un*-serving a component if it no
   longer clears. The rule is written down before the run; what it says goes.

## Pre-registered predictions

1. **The miss is a level, not a spread.** Refitting the league mode alone
   recovers ≥ 75% of the intercept the control found; the ballast moves by
   less than one grid step.
2. **Calibrated `marcel_pitcher_tuned` beats the current one on BB/BF by
   1.5–2.5% of MAE** (clustered |t| > 3) and on (BB+HBP)/BF by 1.0–2.0%,
   out of sample on the five cells. K/BF, HR/BF and BABIP move by < 0.3%.
3. **Against the calibrated baseline the additive stuff arm's BB/BF gain
   falls to 0.5–1.2%**, its covariate-only share stays at −0.6 to −1.0% (t
   between −2 and −3), and **BB/BF no longer clears** the covariate-share
   rule: `p_bb_rate` reverts to the calibrated Marcel. (BB+HBP)/BF likewise.
4. **HR/BF is unchanged** (additive −2.5 to −3.0%, covariate share −2.0 to
   −2.5%, |t| > 3) and stays served. **K/BF is unchanged** (additive −2.0 to
   −2.5%, |t| between 2.3 and 2.7) and its serving decision is whatever the
   rule says on the rerun — not chosen after seeing the number.

### Vacuity check

If the calibrated baseline does not beat the current one on BB/BF by at
least 0.5% (|t| > 2), the intercept was not a calibration defect but
something the walk-forward tuning cannot see; report it, leave the served
engines as they are, and stop.

### Failure conditions

- Prediction 3 fails with BB/BF still clearing: the stuff aggregate carries
  walk information the pre-registration did not credit; keep it served and
  say so.
- Prediction 2 holds but 1 fails (spread, not level): the pitcher Marcel's
  ballast was mis-tuned; the doc records that and the serving decision still
  follows the rule.

## Results (2026-09-09)

Evidence: `data/eval/pitcher_marcel_calibration.json` (level reports, the
league-axis inner validation, the refit, the five-cell holdout) and
`data/eval/pitching_stuff_recalibrated.json` (the four arms against the
calibrated baseline). `src/eval/marcel_pitcher_params.json` is unchanged,
byte for byte; no engine map, served document or test moved.

**Prediction 1 holds: the miss is a level.** Across the 2020–2024 tuning
window the walk projections sit +0.00179 (`p_bb_rate`) and +0.00184
(`p_bbhbp_rate`) above the rate that comes back — about 2.2% high, the sign
and size of the intercept the additive control was fitting — while the
ballast is already within one grid step of the window optimum (300 against
the frozen 425, worth 0.06% and 0.10% of MAE). The cause was mechanical:
both walk components failed the original six-axis inner validation, and the
guard reverted the whole point, taking with it the projected league rate
the inner validation had picked, so the walk rates alone regress toward the
*last* training season. Refitting the level restores `weighted3` for both
(and, for `p_bbhbp_rate`, ballast 300 — one grid step, the half of
prediction 1 that is wrong), cutting the mean level error by 17% and 24%
and the window MAE by 0.36% and 0.99%.

**Prediction 2 fails, and with it the vacuity check.** Out of sample on the
five cells, paired per pitcher and clustered by pitcher, the calibrated
baseline beats the incumbent by **0.50% of MAE on BB/BF (t −1.05)** against
a predicted 1.5–2.5% at |t| > 3, and by 0.81% on (BB+HBP)/BF (t −1.50)
against 1.0–2.0%; K/BF, HR/BF and BABIP are byte-identical, as predicted.
The more aggressive point the full-window search prefers reaches −0.93% at
t −1.72, so the miss is not an artefact of the inner-validation guard. By
the pre-registered vacuity clause the calibrated constants are **not
shipped** and the served engines stand.

| Component | 2025 | 2026 | May 1 | Jul 1 | Aug 1 | pooled |
| --- | --- | --- | --- | --- | --- | --- |
| BB/BF | −0.10% | +0.04% | −1.86% | −0.21% | +0.02% | **−0.50% (t −1.05)** |
| (BB+HBP)/BF | −0.37% | −0.62% | −2.24% | +0.49% | −0.21% | **−0.81% (t −1.50)** |
| K/BF, HR/BF, BABIP | 0 | 0 | 0 | 0 | 0 | 0 |

**Prediction 3's mechanism holds and its conclusion fails.** Kept as a
measurement, against the calibrated baseline the additive arm's BB/BF gain
falls from −3.05% to −1.83% and (BB+HBP)/BF from −2.67% to −1.16% — the
intercept really was the baseline's level — but the covariate-only shares
are unmoved (−0.93%, t −2.58 and −0.46%, t −1.54). BB/BF *would still
clear* the covariate-share rule on the calibrated baseline rather than
revert to Marcel: the first failure condition applies, and the stuff
aggregate is credited with walk information the pre-registration did not
expect it to have.

| Component | additive vs baseline — calibrated | (current) | covariate-only — calibrated | (current) |
| --- | --- | --- | --- | --- |
| K/BF | −2.26%, t −2.50 | same | −3.06%, t −3.73 | same |
| BB/BF | **−1.83%, t −4.18** | −3.05%, t −5.86 | **−0.93%, t −2.58** | −0.87%, t −2.41 |
| (BB+HBP)/BF | **−1.16%, t −3.15** | −2.67%, t −5.25 | **−0.46%, t −1.54** | −0.64%, t −1.86 |
| HR/BF | −2.77%, t −3.70 | same | −2.21%, t −3.50 | same |

**Prediction 4 holds exactly.** K/BF and HR/BF are unchanged to the last
digit; K/BF stays withheld on the total gate it misses by 0.001 of a t, and
HR/BF stays served.

**The finding no prediction anticipated.** BB/BF's covariate-only |t| is
2.41 on the current baseline and 2.58 on the calibrated one — straddling
the 2.5 bar — so the serving decision for the one component this ticket set
out to settle is not robust to a baseline change of half a percent. That is
a statement about the bar as much as about the component: a rule that flips
on a t of ±0.1 is measuring sample size, not skill. It gets its own ticket
(BAS-82) rather than a judgement call here; BB/BF stays served under the
rule in force when it was served.

Data note: the box had no PA-outcome parquets for 2017/2018 (the stuff
cells' training seasons); they were rebuilt from R2 Statcast, and the stuff
backtest against the *current* baseline on that cell set reproduces
BAS-79's table to the decimal.
