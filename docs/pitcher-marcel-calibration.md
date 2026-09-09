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
