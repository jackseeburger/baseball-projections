# Swing decisions — hitting layer 1 for K% and BB%

**BAS-75.** Station A. Pre-registered 2026-09-09, before any run.

## Why

Contact quality (BAS-58, served by BAS-72) is the layer-1 measurement for
what happens *after* the bat meets the ball: ISO, BABIP, HR/PA. K% and BB%
have no measurement of their own; the served covariate reaches them only
through contact-quality's denoising. What predicts them at the pitch grain
is the decision *whether* to swing and *whether* the swing finds the ball:
chase rate outside the zone, swing rate inside it, zone-contact rate, whiff
rate. Public Statcast carries `description`, `zone`, `plate_x`/`plate_z`
and the batter's `sz_top`/`sz_bot` for every pitch since 2015; `bat_speed`
and `swing_length` from 2024, too thin to lean on and excluded here.

## What gets built (the contact-quality shape, reused)

- **Features, per batter per month**, additive counts so buckets sum
  leak-free at first-of-month cutoffs: pitches seen in zone / out of zone
  (Statcast `zone` 1–9 vs 11–14, with a `plate_x`/`plate_z` fallback against
  the batter's own `sz_top`/`sz_bot` when `zone` is null); swings at each;
  contacts (`hit_into_play`, `foul`, `foul_tip`) and whiffs
  (`swinging_strike*`) at each; called strikes taken in zone. The same
  swing mapping BAS-71 pinned. Rates are formed at read time from the
  summed counts, never stored. → `data/features/swing_decisions_monthly.parquet`.
- **Derived aggregates** (six, mirroring contact quality's six): chase
  rate, in-zone swing rate, zone-contact rate, out-of-zone contact rate,
  whiff-per-swing, called-strike-taken rate in zone. Each relative to the
  league month.
- **Stage 2:** the aggregates as covariates on `marcel_tuned`, the free fit
  and the **additive** arm (baseline pinned at 1), with the recalibration
  control (`_recal`) and the permuted control, on the dense harness's cells
  (2022–2026 × May/Jul/Aug, hitters), paired per batter, SE clustered by
  batter. Also the **incremental** arm: swing decisions *added to* the
  served `contact_additive` engine, so the question "does this add to what
  is already served" is answered directly.
- No stage-1 model: these are rates, not a classifier's scores. There is
  therefore no learned component to walk forward; the only fitted things
  are stage 2's `(a, b, g)` on prior seasons.

## Pre-registered predictions

1. **BB%: information.** The additive arm beats `marcel_tuned` on BB% by
   **2–4% of MAE**, clustered |t| > 3, and the gain **survives** at the
   August cutoff and in the high-exposure tercile (contact-quality §6's
   split). Approach changes precede walk-rate changes; that is why this is
   the measurement K% and BB% lack.
2. **K%: denoising.** The additive arm beats `marcel_tuned` on K% by
   **1–3%**, |t| > 2.5, and the gain **evaporates** by August (under 0.5%
   at the August cutoff) — whiff rate is the strikeout rate seen sooner,
   not something beyond it.
3. **Incremental over contact quality.** Added to the served
   `contact_additive`, swing decisions improve BB% by ≥ 1.5% (|t| > 2.5)
   and K% by ≥ 0.5%; contact quality and swing decisions are different
   measurements and the gains are close to additive on BB%.
4. **Covariate share.** Under architecture.md §3's covariate-share rule the
   covariate-only share clears on BB% (|t| > 2.5) and does *not* on K%
   after August is included — consistent with 1 and 2.
5. **Nothing on the other three.** HR/PA, ISO and BABIP move by < 1% and
   are not served from this measurement.

### Vacuity check

Year-over-year correlation of a batter's chase rate (≥ 500 pitches both
years) must exceed **0.55**, and of zone-contact rate **0.50**. Below that
the public zone field does not measure a stable skill and stage 2 cannot
mean anything; report and stop.

### Failure conditions

- Prediction 1 fails but 2 holds: swing decisions are a variance reducer
  for both; keep what clears, do not describe it as measuring approach.
- Prediction 3 fails: the two measurements overlap; serve whichever clears
  the covariate-share rule alone, and say so.

## What ships

Nothing in this pass. Serving is its own ticket, under the covariate-share
rule, with a pre-registered Serving section here first.
