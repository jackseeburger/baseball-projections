# Spending the posterior on player props

Tracked as BAS-70. Follow-on from [bayes-variants.md](bayes-variants.md).

**Status: pre-registered, not yet run.** Predictions below were written into
the commit that added them, before any arm was priced. Results get appended,
including the ones that go against the predictions.

## Why this, and why now

BAS-69 left the hierarchical K% arm level with tuned Marcel on point accuracy.
There is no obvious accuracy left to win at layer 2, and the asset the
Bayesian track has that Marcel structurally cannot offer — a distribution
rather than a number — has never been spent on anything.
[modelling-roadmap.md §1](modelling-roadmap.md#1-the-posterior--things-that-need-a-distribution-and-are-handed-a-number)
names three consumers that are handed a number and need a distribution. The
simulator was tested first and it failed — there was no over-confidence to
fix. The two that remain are where money is the exam: prop pricing and stake
sizing.

## A correction before anything is built

The roadmap says:

> Kelly under parameter uncertainty is provably not Kelly at the mean; the
> correct stake shades down. We are systematically overbetting by an amount
> nobody has measured.

**For a one-shot binary contract this is false.** A contract bought at cost
`c` pays 1, so with stake fraction `f` and true win probability `p`,

    E[log W] = p · log(1 + f·(1−c)/c) + (1 − p) · log(1 − f)

which is **linear in p**. Taking the expectation over any posterior `q(p)` —
however wide — gives the same expression evaluated at `E_q[p]`. So the stake
that maximises expected log growth under the posterior is exactly Kelly at
the posterior mean. Checked numerically before this was written: for Beta
posteriors with mean 0.60 and standard deviations from 0.015 to 0.148, the
posterior-optimal fraction and the plug-in fraction agree to five decimals
(`tests/test_market/test_posterior_kelly.py` pins this).

Where "shade down under uncertainty" *is* right is a different problem —
repeated bets against one persistent unknown `p`, where the policy should
learn, or a payoff nonlinear in `p`. Neither is a prop. What practitioners
actually experience as "Kelly overbets" on a prop is **selection**: the bets
we take are the ones where the estimated edge is large, and an estimate that
is large is more likely to be large *because of* noise. That is a winner's
curse on the edge, and no amount of shading the stake at the mean fixes it,
because the mean is what is biased.

So the posterior is not for the stake. It is for two other things.

## What gets built

The hitter rates behind every prop are Marcel-with-the-partial-season
(`src/sim/lineups.marcel_rates`): weighted successes plus a ballast times the
league rate, over weighted trials plus the ballast. That *is* the mean of a
Beta posterior with pseudo-counts

    Beta( num_w + ballast·lg ,  (den_w − num_w) + ballast·(1 − lg) )

so the distribution has been sitting inside the estimator all along, and
[methods.md §6](methods.md#6-techniques-worth-reusing-and-where-they-came-from)
already derives the correspondence. Nothing new has to be fitted; the pseudo-
counts have to be exposed instead of collapsed.

**Why Marcel's Beta and not the hierarchical model's posterior.** The K% model
is one component of five. A hits, HR or total-bases prop needs all five, and
the other four do not exist yet at the plate-appearance level. The consumer
built here takes posterior draws per component, and it does not care where
they came from — the day the remaining components exist, the same pricing and
selection code takes their draws. What Marcel's Beta *misses* is named rather
than hidden: it carries sampling uncertainty in the estimate and nothing else,
so it has no between-season drift (`sigma_step`, which BAS-69 measured at
0.13 on the logit) and no population uncertainty. It is a lower bound on how
wide the posterior should be.

### 1. Beta-binomial pricing

Today `P(over)` is a Binomial on a point rate. With the rate itself drawn from
the Beta above, `P(over)` is the **marginal** over the rate — a beta-binomial
on hits and home runs, and a rate-mixture on total bases. That is a genuinely
different number, because `P(count ≥ line)` is convex in the rate at the
lines that matter: a hitter at 2+ HR or 3+ hits is priced too low by a point
estimate, and the gap grows as the rate is less certain (a rookie, a
part-timer, April). `props.py`'s own docstring says its Poisson understates
the tail; this is the tail it understates.

### 2. Selection on the probability the edge is real

Today a bet is taken when `|p_model − price| > 2 pts`. With a distribution
over `p_model`, take it when **`P(edge > 0) > τ`** — the posterior probability
that our side is the right side. A 3-point edge on a rate the Beta pins to
±1 point and a 3-point edge on a rate it puts at ±6 points are the same bet
to the current rule and very different bets to this one. `τ` is chosen
walk-forward on the first half of the archive by date and scored on the
second, exactly as the matchup weight was.

### Not built, deliberately

**Posterior Kelly.** See the correction above. It would be plug-in Kelly with
extra steps, and building it would suggest the roadmap's claim survived.

## Pre-registered predictions

The measurement is `scripts/props_exam.py` on the committed archive
(`data/market/prop_closes_2026.parquet`, 2026-07-31 → 09-02, ~62k settled
contracts), Brier paired per contract with the standard error clustered by
game, money at edge ≥ 2 pts on Kalshi as quoted and with the fee waived.

1. **Beta-binomial pricing improves Brier**, paired per contract against the
   current point price, and the improvement is **concentrated at the
   low-probability lines** (over rate below .15: 2+ HR, 3+ hits, 4+ TB) where
   the point price is farthest below the marginal. Pooled effect small —
   between 0.0002 and 0.0010 of Brier — with clustered \|t\| > 2. Reasoning:
   the mixture only moves prices where the tail is fat and the rate is
   uncertain, which is a minority of contracts.
2. **Selecting on `P(edge > 0)` at the walk-forward τ beats selecting on
   \|edge\| > 2 pts at a matched number of bets**, on the second half, on
   ROI with the fee waived. Predicted size: 1 to 3 points of ROI. Reasoning:
   the current rule's losses are the bets where a large mean edge sat on a
   wide posterior; this rule declines exactly those.
3. **Neither turns the props strategy profitable after the Kalshi taker
   fee.** The fee was 5.1 of the 5.7 points lost, and nothing here touches
   the fee. If prediction 2 holds, the fee-waived line moves from −0.6% toward
   +1 to +2%; the as-quoted line stays negative.

### Vacuity check, run before reading any of the above

If the posterior standard deviation of `P(over)` has median below **0.005**
across priced contracts, the Beta is too tight for either prediction to be
testable — the rates are so well-determined that marginalising over them is
a rounding error. That is itself a finding: it would say Marcel's ballast
leaves too little uncertainty for a posterior to be worth anything, and the
hierarchical model's wider posterior (drift, population) is the one to test.

### Failure conditions

- **Prediction 1 fails with the sign reversed** — beta-binomial prices are
  *worse* at the tail. Then the point price was already too fat, the Poisson
  docstring was wrong about the direction, and the fix is the opposite one.
- **Prediction 2 fails at matched bet count.** Then the edge's noise is not
  what the Beta measures — the losses come from *model* error, not
  *estimation* error, and a wider posterior (the hierarchical one) is the
  next test, not a different selection rule.
