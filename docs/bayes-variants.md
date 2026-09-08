# Teaching the Bayesian arm what tuning bought Marcel

Tracked as [BAS-69](https://linear.app/sigils/issue/BAS-69).

**Status: pre-registered; the vacuity check has been run, the sweep has not.** Predictions below were written into the
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

