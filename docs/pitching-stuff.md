# Pitching layer 1: a "stuff" model from public pitch tracking

Tracked as BAS-71. Closes the gap [target-system.md](target-system.md) marks
**NOT STARTED** for pitching at layer 1, and is the first item in
[modelling-roadmap.md §2](modelling-roadmap.md#2-new-information--where-ml-has-something-to-chew-on)
("pitch characteristics → run value ... completely untouched").

**Status: pre-registered, not yet run.** Predictions written into the commit
that added this file, before any model was fitted. Results get appended,
including the ones that go against the predictions.

## Why this is the next layer-1 piece

What front offices do that we do not is mostly **measurement**: models that
read the pitch itself — velocity, movement, release, spin — and score it,
so the talent layer pools a measured skill instead of a noisy strikeout
count. That is the two-stage pattern of
[methods.md §3](methods.md#3-the-two-stage-pattern), and it has cleared the
gate once already on the hitting side: six Statcast contact aggregates beat
`marcel_tuned` on nine of ten components
([contact-quality.md](contact-quality.md)). Pitching has nothing at layer 1.
Public Statcast carries per-pitch `release_speed`, `pfx_x`/`pfx_z`,
`release_pos_x/y/z`, `release_extension`, `release_spin_rate`, `spin_axis`,
`arm_angle`, `vx0…az`, `plate_x/z` and the pitch outcome for every pitch since
2015 — 1.4 GB in R2, twelve seasons. This is the richest thing we own that no
model has read.

## What gets built

Exactly the contact-quality shape, so the harness, the gate and the honesty
checks are all reused rather than reinvented.

**Stage 1 — the measurement model.** A gradient-boosted classifier, walk-
forward by season (fit on pitches from seasons ≤ Y−1, score season Y), from
pitch characteristics to the pitch's outcome:

- **target A: whiff** — swinging strike given a swing;
- **target B: CSW** — called strike or whiff, given the pitch;

from `release_speed`, `pfx_x`, `pfx_z`, `release_pos_x`, `release_pos_z`,
`release_extension`, `release_spin_rate`, `spin_axis`, `arm_angle`, the
velocity/acceleration vector, pitcher handedness, and the pitch's velocity
and movement *relative to the same pitcher's fastball* (the thing "stuff"
models actually key on). **Location is deliberately excluded** — that is the
definition of stuff as opposed to pitching — and a second variant with
`plate_x/z` added is fitted as a labelled comparison, never merged in.

**Stage 1 controls, fitted the same way:** a pitch-type-only model and a
fastball-velocity-only model. A stuff model that cannot beat "what pitch was
it and how hard" out of sample has not measured anything.

**Reduction.** Per pitcher, per calendar month: pitch count, mean predicted
whiff and CSW, the same for fastballs and for non-fastballs, mean velocity,
and the pitch-type mix — committed as
`data/features/pitching_stuff_monthly.parquet`, additive buckets so any
first-of-month cutoff is reconstructed by summing strictly-earlier months
with no leak, exactly as `contact_quality_monthly.parquet` is.

**Stage 2 — the gate.** The monthly aggregates as covariates on
`marcel_pitcher_tuned`, scored walk-forward on the five pitcher cells of
`scripts/run_pitcher_backtest.py` against the live baseline, paired per
pitcher, standard error clustered by pitcher, on the common set.

## Pre-registered predictions

1. **Stage 1 beats both controls out of sample** on every scored season:
   log-loss on whiff-given-swing lower than pitch-type-only and velocity-only
   by at least 0.01. If it does not, stuff is fastball velocity by another
   name and stage 2 is not worth running.
2. **The stuff aggregate clears the gate on pitcher K/BF** — the component
   most directly downstream of whiffs — by **2% to 6% of MAE** against
   `marcel_pitcher_tuned`, clustered \|t\| > 2.5. Reasoning: contact quality
   gave 1.6–4.8% on the hitting side from a coarser measurement.
3. **It is information, not denoising.** Split as contact-quality §6 did:
   the gain does **not** evaporate for high-exposure pitchers or by the
   August cutoff. A pitcher who added velocity or a new pitch is measurably
   different before his strikeout rate catches up, which is the whole reason
   teams build these. If the gain lives only in low-exposure pitchers, this
   prediction fails and the model is a variance reducer.
4. **HR/BF and BB+HBP/BF move less** — under 2% — because stuff is about
   contact avoidance, and a pitch's movement says more about a whiff than
   about a walk.

### Vacuity check

If the walk-forward stage-1 model's out-of-sample AUC on whiff-given-swing
is below **0.62**, the public tracking fields do not separate pitches well
enough for any downstream test to mean anything; report it and stop.

### Failure conditions

- Prediction 2 fails but 1 holds: the measurement is real but K/BF already
  captures it — stuff would then belong at layer 3/5 (usage, matchup) rather
  than as a talent covariate.
- Prediction 3 fails: publish it as denoising, keep the covariate if it
  clears the gate, and do not describe it as measuring skill.

## Results (2026-09-09)

Code: `src/data/pitching_stuff.py`, `src/models/stuff.py`, `src/eval/stuff.py`,
`scripts/build_pitching_stuff.py`, `scripts/run_stuff_backtest.py`. Artifact:
`data/features/pitching_stuff_monthly.parquet` (40,105 rows, 2.4 MB).
Evidence: `data/eval/pitching_stuff_stage1.json`, `pitching_stuff_stage2.json`.

**Vacuity check: passed.** Walk-forward out-of-sample AUC on whiff-given-swing
runs .728 (2017) to .762 (2026) against the .62 floor.

### Stage 1, walk-forward by season

Whiff | swing, log-loss (AUC). `pitching` is the location-augmented arm,
labelled and never merged into stuff:

| season | stuff | pitch-type only | fastball-velo only | pitching (+location) |
| --- | --- | --- | --- | --- |
| 2017 | .4860 (.728) | .5150 (.629) | .5351 (.517) | .4465 (.758) |
| 2020 | .4742 (.746) | .5458 (.611) | .5602 (.531) | .4653 (.761) |
| 2023 | .4594 (.756) | .5342 (.607) | .5472 (.529) | .4504 (.770) |
| 2026 | .4490 (.762) | .5232 (.611) | .5378 (.530) | .4416 (.773) |

(Every season 2017–2026 has the same ordering; the full table is in the
stage-1 JSON.) CSW | pitch shows the same shape: stuff .530–.540 vs
pitch-type .584–.594 vs velocity .585–.596.

**Prediction 1 holds** on every scored season, on both targets, by 0.029–0.074
of log-loss against pitch-type-only and 0.048–0.086 against velocity-only —
three to eight times the pre-registered 0.01.

Swing mapping: swings are `hit_into_play, foul, foul_tip, swinging_strike,
swinging_strike_blocked`; whiffs the two `swinging_strike*`; CSW adds
`called_strike`. Bunts, pitchouts, intentional balls and position-player
pitch types are dropped. Fastball = `FF`/`SI`/`FT` (cutters excluded),
minimum 25 per pitcher-season, below which every relative feature is NaN.
Fits are capped at 1.5M rows (`random_state=20260909`), LightGBM 4.6.0.

### Stage 2 — the gate (2022–2026 × May/Jul/Aug, 4,881 pitcher-cells, 948 pitchers, SE clustered by pitcher)

| Component | Δ MAE vs `marcel_pitcher_tuned` | % | t | vs `stuff_recal` | % | t |
| --- | --- | --- | --- | --- | --- | --- |
| K/BF | −.001009 | **−3.20%** | **−3.37** | −.001231 | −3.87% | −4.57 |
| BB/BF | −.000585 | −3.26% | −5.98 | −.000138 | −0.79% | −2.37 |
| (BB+HBP)/BF | −.000544 | −2.81% | −5.34 | −.000098 | −0.52% | −1.64 |
| HR/BF | −.000255 | −2.51% | −3.56 | −.000217 | −2.15% | −3.58 |

`stuff_recal` (the baseline refit with no covariate) alone: K/BF **+0.70%**
(worse), BB/BF −2.49%, (BB+HBP)/BF −2.30%, HR/BF −0.37%. The permuted
control lands on `stuff_recal` everywhere (|t| ≤ 1.7). Verdict:
`SERVE: p_k_rate, p_bb_rate, p_bbhbp_rate, p_hr_rate`.

**Prediction 2 holds**: K/BF −3.20%, inside the 2–6% band, |t| = 3.37 > 2.5.
More than all of it is the covariate — the recalibration control by itself
is *worse* than the baseline — which is the reverse of what contact quality
found for pitcher K% (+0.5%, t +0.86). That is the pre-registration's central
claim landing.

**Prediction 3 is mixed, leaning information.** K/BF, as % of each slice's
own base MAE:

| axis | low | mid | high | May 1 | Jul 1 | Aug 1 |
| --- | --- | --- | --- | --- | --- | --- |
| pre-cutoff BF tercile | −2.50% | −4.16% | −2.84% | −3.76% (t −3.38) | −3.68% (t −3.17) | **−0.41% (t −0.28)** |

The exposure axis supports information — the gain is larger in the
high-exposure tercile than the low, and HR/BF is textbook (−0.11% low,
−3.34% high). The cutoff axis contradicts it for K/BF: the gain collapses
from −3.76% in May to −0.41% by August. K/BF is not pure denoising (it
survives at high exposure and recalibration buys nothing there) but it does
not survive to the August cutoff. Per the failure conditions: keep the
covariate, and do not describe it as *measuring* skill until the August
collapse is understood.

**Prediction 4 fails, in both directions at once.** BB/BF (−3.26%),
(BB+HBP)/BF (−2.81%) and HR/BF (−2.51%) all move more than the 2% ceiling.
But the control shows almost all of the walk-rate movement is a fitted
rescaling of the pitcher Marcel, not stuff: the covariate's own share is
−0.79% for BB/BF and −0.52% (t −1.64, not significant) for (BB+HBP)/BF —
the trap contact-quality.md §4 documents. Read against the control, the
prediction's *reasoning* is right and its number is wrong. HR/BF is the
opposite: −2.51%, and essentially all of it (−2.15%) is the covariate; the
reasoning under-rated how much movement says about a home run.

### Things that complicate the reading

- Both stage-2 hyperparameters tuned to a grid corner (current season only,
  smallest ballast) on the tuning window. Pinned in `src/eval/stuff.py`; the
  recency corner is a 3.5%-of-MAE effect and is itself evidence for
  prediction 3's mechanism.
- The stage-1 model is not one fixed model across seasons: `spin_axis`
  arrives in 2017 and `arm_angle` around 2021, so the AUC rise from .728 to
  .762 is partly features arriving. Per-season log-losses are not strictly
  comparable.
- 2015–2016 are scored in sample (no two prior seasons); they only enter the
  2017/2018 three-season windows, and the 2022–2026 holdout is strictly
  walk-forward.
- Per-season K/BF is noisy: 2023 goes the wrong way (+0.89%, t +0.41), only
  2025 is individually significant (−6.87%). The pooled result rests on
  five seasons.
- `p_babip` was not scored: stuff has no mechanism for balls in play.
  `p_bbhbp_rate` was added so the control could speak to the rate station E
  consumes.
- The location arm beats stuff on every season (whiff .4416 vs .4490 in
  2026): roughly a third of the reachable signal is command (BAS-76).

### What ships

Nothing in this pass. The gate is cleared on four pitcher components; the
serving path is BAS-79, the pitcher mirror of BAS-72.

## Serving — pre-registered before BAS-79 runs (2026-09-09)

The arm to serve is **additive**: baseline coefficient pinned at 1, the
stuff aggregate added as a correction to `marcel_pitcher_tuned`. Same call
as contact-quality §8, for the same reason, and here the control makes it
sharper — the free fit's walk-rate gains are mostly a rescaling of the
pitcher Marcel, which is a claim about the pitcher Marcel's ballasts and
belongs in its own ticket.

Predictions, additive arm vs `marcel_pitcher_tuned`, same cells, same
clustering:

1. **K/BF clears** at 1.5–3.0% of MAE (60–90% of the free fit's −3.20%,
   the share the hitter side kept), clustered |t| > 2.5.
2. **HR/BF clears** at 1.5–2.5% (it was all covariate in the free fit, so
   pinning the baseline should cost little).
3. **BB/BF and (BB+HBP)/BF do not clear** on the additive arm alone: under
   1% and |t| < 2 for (BB+HBP)/BF. They are **withheld** whatever the
   number says if |t| < 2.5; served only if they clear.
4. Features are read as of the last month boundary on or before the as-of
   date and the fit uses seasons strictly before the predict year; a
   component whose fit cannot be built falls back to `marcel_pitcher_tuned`
   for that build alone.

Vacuity: if the additive K/BF gain is under 1% the serving arm is not worth
the moving parts; report it and leave the pitcher side on Marcel.

### Served (2026-09-09) — scoring the serving pre-registration

The additive arm was scored on the same cells before anything was wired:
4,954 pitcher-cells over 2022–2026 × May/Jul/Aug, 955 pitchers, SE
clustered by pitcher (BAS-71's committed cells in parentheses; the
difference is a PA rebuild picking up games since that run). Evidence:
`data/eval/pitching_stuff_serving.json`.

| Component | Δ MAE vs `marcel_pitcher_tuned` | % | t | covariate-only share (vs `stuff_additive_recal`) |
| --- | --- | --- | --- | --- |
| K/BF | −.000715 | −2.26% (−2.18%) | **−2.50** (−2.38) | −3.06% (t −3.73) |
| BB/BF | −.000547 | −3.05% (−3.24%) | −5.86 | −0.87% (t −2.41) |
| (BB+HBP)/BF | −.000516 | −2.67% (−2.78%) | −5.25 | −0.64% (t −1.86) |
| HR/BF | −.000280 | −2.77% (−2.93%) | −3.70 | −2.21% (t −3.50) |

By the rule written down above, **`p_bb_rate` and `p_hr_rate` are served
as `stuff_additive` and `p_k_rate` is withheld** — it misses the |t| > 2.5
bar, at 2.50 (2.4989) here and 2.38 on the cells the Results table is
stated for. `p_babip` was never scored. `(BB+HBP)/BF` clears but is
station E's rate, not a served pitcher column.

**Prediction 1 fails, on significance only.** K/BF lands inside the
1.5–3.0% band and keeps 69% of the free fit's −3.27%, inside the predicted
60–90% share; pinning the baseline does not shrink the standard error
along with the coefficient.

**Prediction 2 holds and overshoots its own ceiling.** HR/BF is −2.77%
against a predicted 1.5–2.5%, and the additive arm is *better* than the
free fit (−2.37%): pinning the baseline cost nothing there.

**Prediction 3 fails.** Both walk rates were predicted not to clear; both
clear at |t| > 5, so under the rule they are served. A new control,
`stuff_additive_recal` — the same shape with no covariate, baseline plus a
fitted intercept — says what to make of that: of BB/BF's −3.05%, −2.20% is
the intercept and only −0.87% (t −2.41) is stuff; for (BB+HBP)/BF it is
−2.04% against −0.64% (t −1.86, not significant). What the walk rates ship
is mostly a level correction to the pitcher Marcel wearing the stuff
engine's name — contact-quality §4's trap one step further in. K/BF is the
exact opposite: its recalibration control is *worse* than the baseline
(+0.83%) and all of its −3.06% is covariate. That is the uncomfortable
shape of this result: the component where the measurement does the work is
the one held back, and the components served are the ones where it mostly
does not. The rule is applied as written; changing it after seeing this
table is exactly what the gate forbids. Two follow-ups are filed instead
(BAS-80): a pitcher-Marcel calibration ticket for the intercept the walk
rates want, and a pre-registered *covariate-share* condition for future
serving decisions.

**Prediction 4 holds as written.** Features are read at the last month
boundary on or before the as-of date (`pitcher_ros.stuff_cutoff`; the
served document stamps `stuff_features_through`, 2026-08-31 on the
2026-09-09 build), the fit uses seasons strictly before the predict year,
and a component whose fit cannot be built falls back to
`marcel_pitcher_tuned` for that build alone — the document records the
engine that ran, not the one intended.

Vacuity: the additive K/BF gain is 2.26%, above the 1% floor; it is the
significance, not the size, that withholds it.
