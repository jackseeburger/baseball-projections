# The hierarchical posterior's width in the props price

**BAS-92.** Layer 7. Follow-up to BAS-70 (`docs/posterior-props.md`).
Pre-registered 2026-09-10 13:10 UTC in the Linear issue, before any run;
this file mirrors that text.

## Why

BAS-70 spent Marcel's Beta posterior on the props price and found the
marginal price better by .00003 Brier (nothing) and the `P(edge > 0)`
selection rule worth 0.2 points of ROI at a matched bet count (nothing).
Its diagnosis: the Beta's width barely varies across contracts
(`p_over_sd` in a band of .008–.023 for three of four stats), so
`P(edge > 0)` is a monotone relabel of the mean edge and selects the same
bets. Its named next test: "a wider, more player-specific posterior, the
hierarchical model's, with its drift term and population uncertainty".

That posterior now exists on three components: `bayes_walk` on K%, BB%
and HR/PA (BAS-73, `src/models/pa_rate.py`), draws with tuned Marcel on
the mean, and carries per-player posterior sd (`<c>_std`) that Marcel
structurally lacks: wide for a rookie or a player whose walk has drifted,
narrow for a veteran. This is the one exam where the hierarchical model
can win without beating Marcel's point estimate: the mean stays the
served price, only the width changes.

## What gets built

- **Fits.** `bayes_walk` (numpyro, the single-component ability-walk arm
  as in BAS-73) on K%, BB%, HR/PA for 2026 at the dense harness's
  biweekly cutoffs 2026-07-15, 08-01, 08-15, 09-01, persisting the
  per-player posterior mean and sd (seen and unseen batters; unseen get
  the population projection's sd). Training seasons and cutoff rules
  exactly as the dense harness (PA strictly before the cutoff).
- **Arm `bayes_width`.** In `src/market/props.price`, for each batter and
  each of K, BB, HR the Beta's total pseudo-count (α+β) is replaced by
  the method-of-moments total implied by the Bayes posterior sd at the
  most recent cutoff ≤ the game date, recentred on the served daily mean
  (`rate_<c>` from `batter_rates`): the price sold is unchanged, the width
  is the hierarchical model's. BABIP and ISO keep the Beta (no
  hierarchical model exists for them). Pitcher strikeout props keep the
  Beta (no pitcher-rate hierarchical model). The Bayes sd is therefore up
  to 16 days stale relative to the daily mean; recorded, not corrected.
- **Scoring.** `scripts/props_exam.py --matchup on --posterior`, the
  served configuration, on the committed archive (61,738 settled
  contracts, 455 games, 2026-07-31 → 09-02), first/second half split on
  the median date (08-16), τ chosen on the first half, money on the
  second, exactly as BAS-70. Primary set: **HR contracts** (11,896), where
  the changed width is the HR/PA posterior alone. Secondary: hits and TB
  (their width mixes Bayes K/BB/HR with the Beta on BABIP/ISO).
  Strikeouts are unchanged and serve as the negative control (any
  movement there is a bug).

## Pre-registered predictions

1. **(Vacuity, run first) The Bayes width is different information.** On
   HR contracts, the Spearman correlation between the Beta `p_over_sd`
   and the `bayes_width` `p_over_sd` is < 0.9, and at least 25% of HR
   contracts change `p_over_sd` by more than 30%. Below either, the
   single-component posterior's width is as uniform as the Beta's, the
   test cannot resolve anything, and it says so.
2. **(Selection) `P(edge > 0)` at τ chosen on the first half beats the
   threshold rule at a matched bet count on the second half by ≥ 1.0
   point of fee-waived ROI** on HR contracts, and ≥ 0.5 pooled over HR +
   hits + TB. (BAS-70's number was +0.2 pooled with the Beta.)
3. **(Winner's curse) The width separates real from spurious edges.**
   Among second-half contracts with mean edge ≥ 2 points, fee-waived flat
   return in the top tercile of `P(edge > 0)` exceeds the bottom tercile
   by ≥ 2.0 points, on HR and pooled.
4. **(Null) The marginal price does not move.** Paired Brier of
   `p_over_bb` under `bayes_width` vs the Beta is within ±.0001 on every
   stat: the mean carries the price, the width carries the selection.
5. **(Negative control)** Strikeout contracts reproduce BAS-70's numbers
   exactly.

### Failure conditions

If 1 fails, the doc says the single-component posterior width does not
vary enough either, and the thread waits for the joint (BAS-84) and
measurement (BAS-85) posteriors, whose widths carry more structure. If 2
and 3 fail with 1 passing, the edge's noise is model error the posterior
does not measure, and `P(edge > 0)` is retired as a selection rule for
the props ledger. Hits after fee stays exploratory whatever happens; it
is reported, not tested.

## What ships

Only if 2 holds with the second-half interval excluding zero on HR
**and** 3 holds: the paper ledger's selection rule (Stage 0.1) changes to
`P(edge > 0)` with the Bayes width, under a Stage 0 amendment with its
own pre-registered thresholds, and the nightly refresh gains the three
Bayes fits. Otherwise nothing ships; the arm stays behind a flag.

## Risks recorded up front

The Bayes sd is stale by up to 16 days. Method-of-moments Beta from a
logit-normal posterior is an approximation (recorded; the alternative,
pricing from draws, is the follow-up if the width matters). Half-split on
date makes the second half 30,474 contracts, 17 game-days; intervals are
wide. Multiple stats: the primary is HR, named here; the pooled numbers
are secondary. The archive is the same 455 games BAS-70 scored, so this
is a re-analysis of the same contracts with one changed input, not new
data.
