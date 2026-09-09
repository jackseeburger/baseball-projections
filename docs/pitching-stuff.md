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
