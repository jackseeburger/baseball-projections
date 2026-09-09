# Teaching the Bayesian arm what tuning bought Marcel

Tracked as [BAS-69](https://linear.app/sigils/issue/BAS-69).

**Status: complete. All three predictions scored below against the full sweep.** Predictions below were written into the
commit that added the model options, before any variant was fitted. Results get
appended to this file — including the ones that go against the predictions,
which is the only reason writing them down first is worth anything.

## Why this, and why now

[densified-intraseason-backtest.md](densified-intraseason-backtest.md) replaced
a one-cutoff `n = 126` "dead heat" with 164 season-cutoff pairs and 36 Bayesian
fits, and the answer changed. Two numbers from it set up everything here:

| Comparison | Pooled Δ MAE (bayes − baseline) | Clustered t |
| --- | --- | --- |
| `bayes` vs `marcel_tuned` | **+0.00110** (SE 0.00035) | **3.11** |
| `bayes` vs `marcel` (stock) | ≈ 0 at every cutoff | \|t\| ≤ 1.5, mostly < 1 |

The deficit is not "Bayes loses to Marcel". It is *specifically* what tuning
bought Marcel — fitted ballast, fitted recency weights, a projected league rate
and a constrained age curve — that the hierarchical model does not have.

That framing narrows the work. Of those four, the hierarchical model already
has two in a better form:

- **Ballast.** `marcel_tuned` regresses toward the league with a fitted
  per-component constant. The hierarchical model estimates `sigma_ability` from
  the data and shrinks each batter by his own trial count. This is the thing
  partial pooling is *for*, and the flat arm tying stock Marcel says it works.
- **Projected league rate.** The league random walk's last node is the current
  partial season, which is what `marcel_tuned`'s league term collapses to at an
  intra-season cutoff (horizon zero). Same convention, reached from opposite
  directions — already documented in `prepare_model_data`'s docstring.

The other two, it does not have at all:

- **Recency.** `player_ability[batter]` is *one time-invariant number*. A 2019
  plate appearance and a 2026 plate appearance enter the likelihood with equal
  weight. Stock Marcel weights them 0/5. This is the largest untreated
  structural difference between the arms, and it is the one that survives the
  observation that flat-Bayes ties *stock* Marcel: stock Marcel already has
  recency, so pooling is buying back roughly what the missing recency costs.
- **A constrained age curve.** The model fits a free quadratic in centered age.
  Its vertex can land anywhere, including outside the observed age range, where
  "quadratic" has stopped meaning "aging curve". `marcel_tuned` fits a peak
  constrained to 25–31 with slopes of opposite signs.

## What gets built

Two independent flags on `src.models.pa_k_rate.ModelOptions`, each gated on its
own, because a variant that moves two things at once cannot say which one paid.

### `ability_walk` — recency the model learns

Replace the time-invariant `player_ability[batter]` with a Gaussian random walk
in season:

```
ability[b, 0]  = mu_ability + sigma_ability * z0[b]
ability[b, s]  = ability[b, s-1] + sigma_step * z[b, s]      s >= 1
```

non-centered throughout, with `sigma_step` a single population-level HalfNormal.
Projection reads the **last** node (plus one extrapolated step when the horizon
is non-zero), so an old season informs the current estimate only through the
walk, damped by how far back it is.

This is recency the model *estimates* rather than one we fix by hand: the data
choose `sigma_step`, and with `sigma_step → 0` the variant collapses back to
the flat model exactly. It is the plate-appearance-level analogue of the
[PyMC Labs Bayesian Marcel](https://www.pymc-labs.com/blog-posts/bayesian-marcel)'s
Dirichlet season weights — that construction reweights season *aggregates*,
which is not available here because this model's likelihood is over cells, and
weighting a likelihood is not a generative statement about anything.

### `constrained_age` — a peak that has to be a peak

Replace `beta_age * age_c + beta_age2 * age_c^2` with

```
peak         ~ 25 + 6 * Beta(2, 2)              # AGE_PEAK_WINDOW
slope_young  ~ HalfNormal, applied for age < peak
slope_old    ~ HalfNormal, applied for age > peak
```

with the two slopes carrying the signs `scripts/tune_marcel.py` enforces for
K% (rises with age past peak). Same shape as
`src.eval.baselines.tuned_age_adjustment`, on the logit scale.

## Pre-registered predictions

Written before any variant was fitted. The measurement is the dense harness at
its existing biweekly grid, clustered by player, against `marcel_tuned` on the
common set.

1. **`ability_walk` closes most of the gap.** The remaining `bayes_walk` −
   `marcel_tuned` deficit is **≤ +0.0005** pooled, and no longer significant at
   \|t\| < 2 clustered by player. *Rationale: recency is the largest untreated
   structural difference, and flat-Bayes-ties-stock-Marcel is what you would see
   if pooling and recency were roughly trading off.*
2. **`constrained_age` alone moves essentially nothing** — \|Δ\| < 0.0002,
   \|t\| < 1.5. *Rationale: the age term is one global curve estimated on
   millions of plate appearances. It is the best-determined parameter in the
   model, and constraining a well-determined parameter can only hurt or do
   nothing.*
3. **The combination does not beat `ability_walk` alone.** Follows from 2.

### Vacuity check, run before reading any of the above

If the posterior for `sigma_step` concentrates near zero — say a posterior mean
below **0.02** on the logit scale — then the walk has collapsed to the flat
model and prediction 1 is untestable rather than false. That result is worth
publishing on its own: it would say the data prefer time-invariant talent at
this window length, which is a direct answer to whether Marcel's 5/4/3 is doing
real work or is a convenience.

### Failure conditions

- **`ability_walk` closes less than half the gap** (remaining deficit >
  +0.00055, still significant). Then recency is not the explanation, and the
  next place to look is shrinkage *calibration* — whether the model's posterior
  spread on ability is too wide relative to the realized spread — rather than
  anything about time.
- **`ability_walk` makes it worse.** Then the walk is fitting season-to-season
  noise as talent change, and the fix is a tighter `sigma_step` prior or a
  shrunk AR(1) rather than a free walk.

## Result 1: the vacuity check, and it passes

*Run before any comparison against a baseline, exactly as the pre-registration
says. This section was written from the numbers it produced.*

The question was whether `sigma_step` collapses toward zero, which would mean
the walk had degenerated into the flat model and prediction 1 was untestable
rather than false. It does not collapse. It is not close.

| | posterior |
| --- | --- |
| `sigma_step` mean | **0.1216** |
| 90% interval | [0.1095, 0.1345] |
| sd | 0.0074 |
| ESS / R-hat | 477 / 0.999 |
| threshold for "collapsed" | 0.02 |

Six times the threshold, with an interval nowhere near it and a well-mixed
posterior for that parameter specifically. On the rate scale, 0.1216 on the
logit at league K% (p ≈ .22, so dp/d(logit) = p(1-p) ≈ .172) is about **2.1
percentage points of K% of true-talent drift per season** — the size of thing
a projection system has to have an opinion about.

So the model says a hitter's K% talent is not a fixed number that seasons
merely measure with error. That is the mechanism by which Marcel's fixed 5/4/3
earns its keep, stated by a model that estimated the weighting rather than
being handed it. And it makes prediction 1 a real test.

**Scale and caveats.** 200 batters, five seasons (2021-2025), cutoff
2025-07-01, 2 chains x 400 draws, no opposing-pitcher term — the same
`include_pitcher=False` the dense sweep itself uses, so this is the sweep's
arm at reduced batter count, not a different model. Zero divergences in both
arms. Neither fit is "healthy" by the diagnostic's own standard at 400 draws:
the flat arm's worst is `beta_age` (R-hat 1.062, ESS 42) and the walk's is
`z_step` (R-hat 1.030, ESS 239). That is a reason to distrust *those*
parameters at this draw count, not `sigma_step`, which is the only number
this section rests on.

Two alternative readings the number cannot rule out, and neither is a reason
to discount it for projection purposes:

- The walk may be absorbing **individual aging** that one global quadratic
  cannot express. That is drift a projection should track either way.
- It may be absorbing **role and context change** — a hitter moving parks,
  moving in the order, facing a different mix of arms. Also real, also
  something a projection wants, but it would mean "true talent drifts" is the
  wrong words for it.

## What this cost, which changes the sweep's design

| arm | wall time | cells |
| --- | --- | --- |
| flat | 38.0s | 1,185 |
| `ability_walk` | 66.8s | 1,185 |

The walk costs **1.76x** the flat model, not the 3-4x its parameter count
would suggest — `pt.cumsum` over a non-centered walk is cheap, and the cell
count is unchanged because the walk adds parameters, not data.

The genuinely expensive knob turned out to be one this comparison does not
use. With `include_pitcher=True` the cell key gains ~1,000 pitcher levels and
almost every plate appearance becomes its own cell: 93,201 cells instead of
1,185, a 79x increase, and a fit that does not finish in an hour on this
hardware. `scripts/run_intraseason_backtest_dense.py` already defaults to
`include_pitcher=False`, so the published sweep and these timings are the same
arm — but it is worth writing down, because an unlucky default here is the
difference between a five-hour sweep and one that never lands.

## Result 2: the verdict

48 (season, cutoff) pairs — biweekly across 2022, 2024, 2025, 2026 — 192 MCMC
fits, 171,108 scored rows, 14,259 per arm on the common set. Every number
below is clustered by player; the unclustered t is reported alongside because
it runs **2.4-2.9x** larger here, in line with the 3.71x
[the previous sweep](densified-intraseason-backtest.md) measured.

| arm | vs | Δ MAE | t (player) | t (cell) | t (unclustered, **wrong**) | cells W-L |
| --- | --- | --- | --- | --- | --- | --- |
| `bayes_flat` | `marcel_tuned` | +0.00121 | **4.08** | 14.00 | 11.19 | 1-47 |
| `bayes_walk` | `marcel_tuned` | **+0.00033** | **1.40** | 4.98 | 3.42 | 8-40 |
| `bayes_age` | `marcel_tuned` | +0.00118 | 4.00 | 12.42 | 10.95 | 1-47 |
| `bayes_walk_age` | `marcel_tuned` | +0.00031 | 1.28 | 4.53 | 3.13 | 9-39 |
| `bayes_walk` | `bayes_flat` | **−0.00087** | **−3.45** | −7.63 | −8.86 | **43-5** |
| `bayes_age` | `bayes_flat` | −0.00002 | −0.54 | −0.98 | −1.38 | 28-20 |
| `bayes_walk` | `marcel` (stock) | −0.00042 | −1.20 | −4.02 | −3.18 | 32-16 |

The flat arm reproduces the previous sweep closely — +0.00121 here against
+0.00110 there, on a disjoint set of cutoffs — which is the check that says
the harness is measuring the same thing.

### Prediction 1: **holds**

> `ability_walk` closes most of the gap. Remaining deficit ≤ +0.0005, \|t\| < 2.

+0.00033 with t = 1.40. The gap to `marcel_tuned` closes by **73%**, and what
is left is no longer distinguishable from zero on the primary clustering.
Against the flat arm the walk wins **43 of 48 cutoffs** at t = −3.45.

Recency was the missing ingredient, and giving the model a way to *estimate*
it recovers most of what hand-tuning bought Marcel.

### Prediction 2: **holds**

> `constrained_age` alone moves essentially nothing. \|Δ\| < 0.0002, \|t\| < 1.5.

−0.00002 against the flat arm, t = −0.54. Predicted for the stated reason: the
age term is one global curve estimated on hundreds of thousands of plate
appearances, so it is already the best-determined thing in the model and
constraining it can only fail to help.

### Prediction 3: **not supported**

> The combination does not beat `ability_walk` alone.

Scored directly rather than by comparing two columns against a third:
`bayes_walk_age` − `bayes_walk` = **−0.0000289**, t(player) −1.47, better at
**31 of 48** cutoffs. The point estimate goes the other way from the
prediction. It is not significant on the primary clustering (\|t\| < 2), and
the improvement is about **3% of the walk's own effect** — but "not
significant" is not what was predicted, and the honest score is that the
prediction failed rather than that it survived on a technicality.

### The robustness check that mattered most

`marcel_tuned`'s constants were fitted walk-forward on 2020-2024. Two of the
four seasons here fall inside that window and two do not, so the split says
whether any of this is the baseline being flattered by its own training data.

| | `bayes_flat` | `bayes_walk` | `bayes_walk_age` |
| --- | --- | --- | --- |
| inside 2020-2024 (24 cells) | +0.00113, t 2.53 | +0.00028, t 0.84 | +0.00025, t 0.75 |
| clean holdout 2025-2026 (24 cells) | +0.00130, t 3.92 | +0.00040, t 1.23 | +0.00037, t 1.15 |

The flat arm's deficit is *larger* in the clean holdout, not smaller, so it
was never a tuning-window artifact — and the walk closes the same share of it
on both halves. Nothing here depends on which seasons Marcel was tuned on.

### What the model learned, in its own parameters

Across all 48 fits of each variant:

| parameter | mean | range | what it says |
| --- | --- | --- | --- |
| `sigma_step` | **0.1336** | 0.111 - 0.169 | ~2.3 points of K% of talent drift per season |
| `peak_age` | 28.4 | 27.2 - 29.9 | where K% bottoms out |
| `slope_young` | 0.0254 | 0.017 - 0.038 | K% *falls* steeply toward the peak |
| `slope_old` | 0.0080 | 0.005 - 0.014 | and rises gently after it |

`sigma_step` never approaches the 0.02 collapse threshold at any cutoff in any
season — the tightest fit puts it at 0.111, five times the threshold. Whatever
else is true, the data are emphatic that K% talent moves between seasons.

**An independent corroboration worth recording.** `scripts/tune_marcel.py`
fits `marcel_tuned`'s K% age curve by coordinate search on season aggregates,
and lands on `age_slope_old = 0.008`. The hierarchical model, at the plate
appearance, with a Beta-scaled peak and HalfNormal slopes, lands on **0.0080**.
Two estimators sharing no code, no likelihood and no data representation
agreeing to two significant figures on how fast a hitter's strikeout rate
rises after his peak. They disagree on the young side — Marcel's grid picks a
flat 0.0 before a peak at 30, the model picks 0.0254 before a peak at 28.4 —
so this is agreement about decline, not about the whole curve.

**Sampling.** Zero divergences in all 192 fits. Worst R-hat 1.039, and the
walk variants sample *better* than the flat model, not worse (lowest ESS 130
and 124 against the flat model's 57) — a random walk over three season nodes
gives the sampler an easier geometry than one level fighting all three
seasons at once. Median 91s per walk fit against 64s flat, so the walk costs
1.4x, not the 3-4x its parameter count suggests.

### What ships, and what does not

`ability_walk` clears its gate against `bayes_flat`: −0.00087, t = −3.45, 43
of 48 cutoffs, on the common set, out of sample, holding on the clean
holdout. That is a real win and it is the first time anything on the Bayesian
track has cleared one.

It does **not** clear a gate against `marcel_tuned`, which is the bar that
decides what the site serves. +0.00033 at t = 1.40 is a tie, not a win, and
§3 says the incumbent keeps its place on a tie. **`marcel_tuned` stays the
live rest-of-season engine.**

The honest summary is that the hierarchical model has gone from *losing* to
tuned Marcel to *drawing* with it, by learning the one thing tuning had that
it did not. Drawing is not winning. What it buys is a model that now matches
the baseline while also carrying a posterior, which is the thing Marcel
structurally cannot do — so the next question is whether that posterior is
worth anything on a decision, not whether the point estimate can be squeezed
further.

`constrained_age` does not ship in either direction: it neither helps nor
hurts, and an unused parameterisation with a tighter prior is not free
complexity to carry. Its value was diagnostic — it says the free quadratic was
not the problem, and it produced the `slope_old` agreement above.

