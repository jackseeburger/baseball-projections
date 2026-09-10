# Spending the posterior on player props

Tracked as BAS-70. Follow-on from [bayes-variants.md](bayes-variants.md).

**Status: complete. Scored below; two of three predictions fail.** Predictions below were written into
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

## Results

`scripts/props_exam.py --matchup on --posterior`, the served configuration,
on the committed archive: **61,738 settled contracts, 455 games,
2026-07-31 → 09-02**. Brier paired per contract with the standard error
clustered by game; money on the second half by date (30,423 contracts) with
τ chosen on the first.

### Vacuity check: passes

Median posterior sd of `P(over)` is **0.0149** on the full archive (hits
0.014, HR 0.017, TB 0.013, strikeouts 0.039), three times the 0.005 line;
19% of contracts fall below it. So what follows are real results, not
artifacts of a posterior too tight to matter.

### Prediction 1: sign holds, size and location fail

> Beta-binomial improves paired Brier by 0.0002–0.0010, clustered \|t\| > 2,
> concentrated at lines with over-rate below .15.

| | n | Brier(marginal) − Brier(point) | t (game) | mean price shift |
| --- | --- | --- | --- | --- |
| all | 61,738 | **−0.00003** | −6.16 | +0.00002 |
| tail lines (< .15) | 26,492 | −0.00001 | −2.91 | **+0.00026** |
| other lines | 35,246 | −0.00005 | −5.69 | −0.00015 |

The marginal price is better, and it is better with a t of six — but by
**0.00003 of Brier, an order of magnitude below the predicted range**, and the
improvement is *larger off the tail than on it*. The convexity mechanism is
real: at tail lines the marginal sits 0.00026 *above* the point price, as
predicted, and the point price was indeed too thin there. It is just tiny.
Off the tail the marginal sits below the point price, and that correction —
the point price being slightly too *high* where the count distribution is
concave in the rate — turns out to be worth more. Per stat the gain lives
almost entirely in total bases (t −7.1) and strikeouts (t −3.7); hits and
home runs are indistinguishable (t −1.6, −0.4).

Scored as written: the direction survives, the magnitude and the
localisation do not. A price that is better by 0.00003 is not a different
price.

### Prediction 2: fails

> Selecting on `P(edge > 0)` beats the threshold rule by 1–3 points of
> fee-waived ROI at a matched number of bets.

| rule, second half | bets | ROI fee-waived | ROI as quoted |
| --- | --- | --- | --- |
| threshold @ 2 pts (current) | 16,756 | +0.7% (−2.8, +4.0) | −4.0% |
| **posterior @ τ = 0.65** | 23,912 | **+2.5%** (−1.0, +6.2) | −2.7% |
| threshold @ matched count (0.42 pts) | 23,912 | **+2.3%** (−1.2, +5.8) | −2.8% |

Against the current rule the posterior rule looks like a 1.8-point gain.
Against a threshold rule *forced to take the same number of bets*, the gain
is **0.2 points**, inside any interval. The posterior rule at τ = 0.65 is,
almost exactly, "bet whenever the mean edge exceeds 0.4 points" — the
matched threshold that reproduces its bet count. The τ grid says the same
thing from the other side: fee-waived ROI is flat within ±0.2 points from
τ = 0.55 to 0.80 and only falls beyond, so there is no interior optimum,
only a looser filter.

Why: the vacuity check asked whether the Beta was *wide enough*, and it was.
It did not ask whether the width **varies across contracts enough to
reorder them**, and it does not. With `p_over_sd` sitting in a band of
roughly 0.008–0.023 for three of four stats, `P(edge > 0)` is a monotone
function of the mean edge to a very good approximation, and a monotone
re-labelling of the same ranking selects the same bets. The one stat where
the width does vary — strikeouts, whose Beta is the approximation flagged
above — is also the one where every rule loses 10–15%.

This is the failure condition the pre-registration named: *the edge's noise
is not what the Beta measures*. Marcel's sampling uncertainty is nearly the
same for every regular, so it cannot tell a real 3-point edge from a
spurious one. Whatever separates them is model error, and a wider, more
player-specific posterior — the hierarchical model's, with its drift term
and population uncertainty — is the next thing to test here, not a
different selection rule on the same Beta.

### Prediction 3: holds

> Neither turns the props strategy profitable after the Kalshi taker fee.

As quoted, every pooled rule is negative: −4.0%, −2.7%, −2.8%. The fee
remains the whole loss.

### Not pre-registered, and reported as such

**Hits.** Every rule, both halves, fee-waived: +9.5% at the 2-point
threshold, +12.1% under the posterior rule, +11.4% matched — intervals
that exclude zero on the second half (+5.8% to +19.1% for the posterior
rule). As quoted: +4.3% to +5.5%, intervals that do not exclude zero. A
stat-level split of a pooled pre-registered test is exploratory, and the
hits number was not predicted, so it is a lead and not a result. It is the
first time anything on the props exam has been positive after the fee at
the point estimate, and it is worth its own pre-registration on the June
and July contracts the archive has not yet fetched.

*Followed up (BAS-93, `docs/props-replication.md`): the +9.5% / +12.1% is
the second half of one month; on the whole August window the same rule is
+3.7% / +7.1% fee-waived and negative as quoted, and on the July
contracts, fetched and scored with every constant frozen, hits is +0.6%
fee-waived and −4.7% as quoted. The lead did not replicate.*

**Strikeouts** lose 10–16% under every rule, which is the same finding as
the original props exam and is consistent with the pitcher-K Beta being an
approximation stacked on a rate the market prices well.

## What ships, and what does not

Nothing. The marginal price is better by an amount that changes no decision;
the posterior selection rule is a looser threshold wearing a distribution.
`marcel_partial + matchup` at the 2-point threshold remains the props
model, and it remains a model that loses after the fee.

What the exercise bought is sharper than a shipped feature. The roadmap's
Kelly claim is retracted by proof; the posterior's first consumer is
scored and the reason it failed — width that does not vary — points
directly at the next test; and the hits line is the first positive
after-fee number on the board, flagged as exploratory so nobody quotes it
as more.

