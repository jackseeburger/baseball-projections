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

## Results (2026-09-10)

Evidence `data/eval/bas92/posterior_width.json` (fits, both checkpoint
checks, the width audit); fits `scripts/run_bayes_width.py`; analysis
`scripts/analyze_bas92.py`; the arm `props.BayesWidth`,
`props.price(..., bayes_width=)`, `props_exam.py --bayes-width DIR`;
tests `tests/test_market/test_props.py`,
`tests/test_scripts/test_bas92_width.py`.

**Verdict: vacuous by its own rule; nothing ships.** The hierarchical
width is wider than the Beta's (median 1.21× on HR contracts) but in
almost the same order across contracts, so it cannot reorder the bets
the Beta already ranks. The pre-registration's "if 1 fails" branch
applies: the posterior-width thread waits for the joint (BAS-84) and
measurement (BAS-85) posteriors, whose widths carry structure a single
component's cannot. The arm stays behind a flag, default off.

### Fits and reproduction

Twelve `bayes_walk` fits (numpyro, 2 chains × 500 draws, training
seasons 2024–2026 as the dense harness), 2026 at the four cutoffs: zero
divergences everywhere, max r-hat 1.03, `sigma_step` in BAS-73's order
(BB > HR > K) and magnitudes. Per cutoff 723–751 seen batters plus
208–232 projected from the population. Median sd, seen vs unseen: K
.026 vs .061; BB .014 vs .022; HR .007 vs .011.

Two things about the cutoffs. First, the pre-registration called them
"the dense harness's biweekly cutoffs"; only 2026-08-01 is on the
harness's grid (`BIWEEKLY_MMDD` ends at 08-05), so the checkpoint
comparison rests on that date alone. Second, at 2026-08-01 the served
fits differ from the checkpoint by up to .012 on K% because the
checkpoint was written before `data/features/park_factors.parquet`
existed (BAS-86 landed between them) and `load_park_factors` now finds
it; with park factors neutralised the new fits reproduce the checkpoint
to 0.0 on all three components. Both runs are committed. The park block
is immaterial here (sd Spearman between the two runs .987–.998).

The Beta baseline re-price is bit-for-bit BAS-70's archive (61,845
rows, max abs diff 0.0 on every price column) and every published BAS-70
number reproduces (t on the marginal −6.41 against the doc's −6.16, per
stat identical).

### The arm

`total = m(1−m)/s² − 1`, recentred on the served daily mean, K, BB, HR
only. Applied 78,096 times; the pseudo-count floor bound 0 times; every
one of the 55,597 batter contracts and 477 batters got a hierarchical
width, 3.4% of contracts from the population projection for at least one
component. Staleness max 16 days, median 7.

### Predictions

1. **Vacuity: fails on one leg.** HR contracts (11,896): Spearman
   between the Beta and the Bayes `p_over_sd` **.963**, needed < .9; share
   changing by > 30% **28.8%**, needed ≥ 25% (passes). Median ratio 1.21×
   on HR (.0166 → .0190), 1.16× hits, 1.10× TB, 1.00× K. Hits ρ .966,
   TB .991, K 1.000. Wider, same order.
2. **Selection: fails, wrong direction.** τ chosen on the first half
   pooled (Beta .65, `bayes_width` .75). Second half, fee-waived: HR
   `P(edge > 0)` −3.3% vs matched threshold −0.7% (gain **−2.6 pts**,
   needed ≥ +1.0; 3,091 bets each); pooled +2.5% vs +3.7% (**−1.2 pts**,
   needed ≥ +0.5). The Beta's own gains are −1.2 HR / +0.2 pooled. Every
   interval spans zero.
3. **Winner's curse: fails with the sign reversed, under both
   posteriors.** Among second-half contracts the 2-point rule bets, top
   tercile of `P(edge > 0)` minus bottom, fee-waived: HR **−47.7 pts**
   (`bayes_width`) and −45.6 (Beta); pooled **−11.6** and −11.7. The top
   tercile is where `P(edge > 0)` saturates at 1.000 (heavy favourites
   bought near cost, pooled hit rate .75) and it is the worst bucket. The
   Beta and the Bayes width agree, so this is a property of `P(edge > 0)`
   as a ranking, not of the width behind it.
4. **Null: passes.** Paired Brier of `p_over_bb`, `bayes_width` − Beta,
   clustered by game: worst |Δ| 8.6e−6 (TB), K exactly 0. The mean
   carries the price.
5. **Negative control: passes exactly.** 6,248 strikeout contracts
   bit-identical on `p_over_bb` and `p_over_sd`, once the RNG was
   isolated (below).

Exploratory, hits after fee (as quoted, second half): Beta +4.3 / +5.4 /
+5.0%; `bayes_width` +4.3 / +3.7 / +5.4% (threshold / posterior /
matched). Intervals include zero on all six, as in BAS-70.

### Two bugs found on the way

- **The pricing RNG was shared across players.** `props.price` drew every
  player's component Betas from one generator, and `Generator.beta` is a
  rejection sampler that advances the stream by a parameter-dependent
  amount, so changing one batter's Beta moved the Monte Carlo draws of
  every player priced after him, pitcher strikeouts included (max
  |Δ p_over_bb| .0083 with no price change). The negative control caught
  it. `props.DRAW_STREAMS` now has `"shared"` (default, bit-for-bit the
  committed archive) and `"isolated"` (seeded per kind, date and player);
  the scored pair uses `isolated`. Verdicts on predictions 1–4 are
  identical under both.
- **Every Bayes fit on main raised before sampling.** `src/eval/bayes_arm.py`
  on main is BAS-85's version and forwards `extra_quantiles=` to
  `pa_rate.generate_projections`, which on main takes no such argument;
  the dense harness's `bayes_walk` arm requests coverage quantiles, so it
  was broken too. The kwarg is now forwarded only when non-empty. The
  measurement arm (`pa_measurement`) is still BAS-85's to land.

### What this means

- **For the ledger:** nothing changes. The threshold rule stays.
- **For `P(edge > 0)`:** the pre-registration retires it only if the
  vacuity check passes, which it did not, so it is not retired by this
  ticket. Prediction 3's reversal is nonetheless the same number under
  both posteriors and is evidence about the rule itself: ranking by
  `P(edge > 0)` puts saturated favourites on top, and they are the worst
  bets. Any future use of it needs the saturation handled.
- **For the structural track:** a single-component posterior's width is
  Marcel's width with a learned ballast, the same arithmetic that makes
  its mean Marcel's mean. Width that reorders contracts has to come from
  structure: the joint model's correlated components or the measurement
  model's channels. That test waits for BAS-85.
- **The BB denominator**: the model's `bb_rate` is BB/PA while the served
  component is (BB+HBP)/PA; the BB width rides on a slightly larger
  denominator's mean. Recorded, not corrected.
- The four priced parquets are not committed (4.2 MB each); the doc
  names the command that regenerates them in ~3 min each.
