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

## Results (2026-09-09)

Code: `src/data/swing_decisions.py`, `scripts/build_swing_decisions.py`,
`src/eval/swing.py`, `scripts/run_swing_backtest.py`. Artifact:
`data/features/swing_decisions_monthly.parquet` (42,795 batter-months,
2015–2026, nine integer counts). Evidence:
`data/eval/swing_decisions_stage2.json`.

Data note: on the competitive regular-season set Statcast's `zone` is
populated for 100% of pitches in every season, so the documented
`plate_x`/`plate_z` fallback never fires on the archive — it is a guard
exercised only by tests.

**Vacuity check passes with room to spare.** Year over year, batters with
≥ 500 pitches in both seasons (n = 3,029 pairs, 2016–2026): chase 0.807,
zone-swing 0.766, zone-contact 0.767, out-of-zone contact 0.791, whiff
0.829 — against thresholds of 0.55 and 0.50. Called strikes taken in zone
is the exception: stable at 0.39–0.58 through 2025 and **0.149 for
2025→2026** — an umpiring change, not the batter; it stayed in as one of
six covariates and was not dropped after the fact.

### Stage 2 — `swing_additive` vs `marcel_tuned`

2022–2026 × May/Jul/Aug, hitters, 5,004 cells / 767 batters, paired per
cell, SE clustered by batter. Weights (1.0, 0.35, 0.1), ballast 10
pitches, chosen on cells ≤ 2021 only.

| Component | Δ MAE | % | t | covariate-only (t) | share |
| --- | --- | --- | --- | --- | --- |
| K% | −.000473 | **−1.57%** | **−3.47** | −.000314 (**−2.96**) | 66% |
| BB% | −.000176 | −0.94% | −2.40 | −.000145 (−2.10) | 82% |
| HR/PA | −.000141 | −1.32% | −1.88 | −.000181 (−3.92) | — |
| BABIP | −.000154 | −0.54% | −1.16 | −.000104 (−1.36) | 68% |
| ISO | −.000063 | −0.16% | −0.23 | −.000567 (−3.50) | — |

The free fit is roughly twice as large on both targets (K% −2.63%, BB%
−2.19%) by rescaling Marcel, as contact quality's was. The permuted control
lands on the recalibration arm on every component (|t| ≤ 1.6, every sign
the wrong way). Fitted coefficients read sensibly: whiff +0.0147 on K%,
chase −0.0039 on BB%.

### Incremental over what is served

`contact_swing_additive` = the served `contact_additive` with the six
swing covariates alongside.

| Component | contact_additive vs Marcel | both vs Marcel | **both vs contact_additive** | t |
| --- | --- | --- | --- | --- |
| K% | −1.93% (t −4.30) | −1.95% | **−0.02%** | −0.09 |
| BB% | −1.56% (t −3.18) | −2.31% | **−0.77%** | **−2.27** |
| HR/PA | −3.18% (t −3.40) | −3.55% | −0.39% | −2.47 |
| BABIP | −1.54% | −1.56% | −0.02% | −0.09 |
| ISO | −3.08% | −3.17% | −0.09% | −0.68 |

### §6 split (`swing_additive` vs `marcel_tuned`, % of the slice's own base MAE)

K%:

| exposure | May 1 | Jul 1 | Aug 1 | all |
| --- | --- | --- | --- | --- |
| low | −2.92% (t −3.47) | −0.96% | −1.26% | −2.12% (t −3.16) |
| mid | −2.32% (t −2.40) | −0.38% | +0.04% | −1.37% |
| high | −1.02% | −0.83% | **−1.79% (t −2.09)** | −1.16% |
| all | −2.24% (t −3.85) | −0.73% | −1.11% (t −2.04) | −1.57% (t −3.47) |

BB%:

| exposure | May 1 | Jul 1 | Aug 1 | all |
| --- | --- | --- | --- | --- |
| low | −2.29% (t −2.92) | −2.04% (t −2.31) | −1.87% | **−2.15% (t −3.34)** |
| mid | −0.51% | −0.53% | −1.56% | −0.69% |
| high | −0.96% | +0.69% | +0.04% | **−0.10% (t −0.18)** |
| all | −1.28% (t −2.60) | −0.45% | −0.85% (t −2.01) | −0.94% (t −2.40) |

### Scoring the predictions

Of the five, one holds in full, two hold in half, and two fail — and the
failures are the interesting part.

**Prediction 1 fails.** BB% moves by 0.94% at t −2.40, well under 2–4% at
|t| > 3, and the shape is the mirror of what was predicted: −2.15% in the
low-exposure tercile and −0.10% (t −0.18) in the high one — the signature
of variance reduction, not of approach change leading walk rate. It does
survive at August (−0.85%, t −2.01), so the failure is one of magnitude
and of exposure, not of persistence.

**Prediction 2 is half right.** K% clears — −1.57% at t −3.47, inside the
1–3% band — but the gain does *not* evaporate by August: −2.24% at May 1,
still −1.11% (t −2.04) at August 1, and in the high-exposure tercile it is
*largest* at the August cutoff (−1.79%, t −2.09). Whiff rate is not simply
the strikeout rate seen sooner.

**Prediction 3 fails on both halves.** Added to the served
`contact_additive`, swing decisions are worth −0.77% on BB% (t −2.27) —
real, half the pre-registered 1.5%, short of |t| > 2.5 — and **nothing on
K%** (−0.02%, t −0.09) against ≥ 0.5%. That is the finding of the pass:
contact quality's unexplained hitter-K% gain, flagged in
contact-quality.md §5 as "a signature of his swing", is *the same
measurement*. Swing decisions explain what contact quality was already
reading on K% and add nothing on top; on BB% the two are closer to
independent, and roughly half of what swing decisions know is new.

**Prediction 4 fails, reversed.** Under the covariate-share rule it is K%
that clears (t −2.96, two thirds of the gain) and BB% that does not
(t −2.10).

**Prediction 5 mostly holds.** ISO −0.16% and BABIP −0.54%, under the bar
and not significant; HR/PA −1.32% (t −1.88) is over the bar and not
distinguishable from zero — recorded as a miss, not a result. Its
covariate-only share is the largest in the table (t −3.92): the
swing-shape channel contact-quality.md §5 described, showing up again.

### What ships

Nothing. Under the failure conditions: prediction 1 failing while 2 holds
means swing decisions are a variance reducer on BB% and something more on
K%, and are not described as measuring approach; prediction 3 failing
means the two measurements overlap on K%, where contact quality already
clears alone. The one live question is whether the −0.77% on BB% on top
of the served engine is worth serving under the covariate-share rule — it
is not, at t −2.27 — and it stays a measurement.
