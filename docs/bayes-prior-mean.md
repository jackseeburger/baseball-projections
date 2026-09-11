# Shrink toward similar players: the prior's mean as a function of the profile

**BAS-94.** Station A, layer 2. Pre-registered 2026-09-11 13:20 UTC in the
Linear issue, before any run; this file mirrors that text. Follow-up to
BAS-83 (`docs/bayes-covariates.md`) and BAS-85
(`docs/bayes-measurement.md`).

## Why

Every hierarchical arm so far shrinks each player toward one league mean.
That is why the single-component model draws with tuned Marcel (it is
Marcel with a learned ballast, BAS-69/73), and it is the structure the
two failed covariate tickets never changed: BAS-83 put the Statcast
aggregates in the *likelihood* as per-cell regressors and lost to errors
in variables on the thin current window; BAS-85 made them *observation
channels* of a latent and lost to the same mechanism plus a mis-specified
whiff channel. Neither touched the prior.

A team's model shrinks a thin sample toward what players who hit the ball
like this usually do. That is a prior whose mean is a function of the
player's profile: `ability_i ~ Normal(mu + gamma · x_i, sigma_ability)`,
with `x_i` measured on **prior seasons**, where the profile is
full-season and low-noise, so the current season's thin window enters
only through the outcome likelihood and cannot leak noise into the
shrinkage target. This is the last hierarchical structure on the list
that public data can feed, and it is the structure the served two-stage
engine (`contact_additive`, a linear correction on tuned Marcel)
approximates by hand.

## What gets built

`src/models/pa_rate.py` gains `ModelOptions.prior_mean_covariates`: the
batter's ability prior mean becomes `mu_ability + gamma · x[batter,
season]`, `gamma ~ Normal(0, 0.5)` per feature, `x` the standardised
contact-quality aggregates (`src.eval.contact.FEATURES`: EV mean, EV90,
barrel, hard-hit, sweet-spot, launch-angle mean, plus whiff share for K%)
computed on the **previous two full seasons** of Statcast through the
`src.models.pa_covariates` machinery, exposure-weighted, standardised per
season on the training population, zero (league mean) for a batter with
no prior-season profile. Under the ability walk, the prior mean applies
to each season's level (the season-0 level and the walk's innovations
are unchanged), so a player's shrinkage target moves with his measured
profile year to year. `None` is the old model, bit for bit (test).
`BayesArmConfig` and the dense sweep expose it as
`--bayes-prior-mean contact`.

**Arms.** A (primary): prior mean from prior-season features only. B
(secondary): prior mean from features through the last month boundary ≤
cutoff, with the current season's contribution exposure-weighted against
the prior-season profile (a ballasted feature, so a thin window moves the
target little). Comparators: `bayes_walk` (the same model with the prior
mean off), `contact_additive` (served), `marcel_tuned`. Components: HR/PA
and K%. Cells: the dense harness's biweekly cutoffs on 2022, 2024, 2025,
2026 (the BAS-85 grid), numpyro (valid for this graph, BAS-92), 2 chains
× 500 draws, paired per batter, SE clustered by batter and by cell,
walk-forward with features strictly before the cutoff
(`assert_window_clean`).

## Pre-registered predictions

1. **(Vacuity) The prior moves.** `gamma` on barrel or EV excludes zero
   for HR/PA and on whiff for K% at ≥ 90% of cutoffs, and the
   between-player sd of the prior mean is ≥ 30% of `sigma_ability` (the
   shrinkage target is player-specific, not a relabelled league mean).
2. **(Early season, the point of the structure.)** At the May cutoffs,
   arm A beats `bayes_walk` on HR/PA by ≥ 3% of MAE (clustered |t| > 2.5)
   and on K% by ≥ 2%.
3. **(Pooled.)** Arm A vs `bayes_walk` pooled over all cutoffs ≤ −1.5% on
   HR/PA (t < −2) and ≤ −1% on K%. Arm A vs `contact_additive` pooled
   within ±1.5% on HR/PA: a draw with the served engine is the honest
   expectation, and the number is written down so a win or a loss is
   read as one.
4. **(Mechanism.)** The gain over `bayes_walk` shrinks from May to August
   (August gap ≤ half the May gap): shrinkage matters less as outcomes
   accumulate.
5. **(Calibration.)** The 80% posterior interval covers the realised
   rest-of-season rate 75–85% of the time; arm A's interval is narrower
   than `bayes_walk`'s at May.

### Failure conditions

If 1 fails, the profile carries no information the league mean does not,
and the doc says the structural track is closed on public data at these
sample sizes. If 2 fails with 1 holding, shrinking to similar players
does not help where it should; nothing ships and the doc says why. Arm B
is reported beside A and never substitutes for it.

## What ships

Nothing in this pass. If arm A beats `contact_additive` pooled at t < −2
on HR/PA or K%, a serving ticket follows under `docs/serving-rules.md`
(gate on the served season plus the effect floor and joint vacuity test).
Otherwise the arm stays behind the flag.

## Risks recorded up front

The profile is stale by up to a season for arm A (recorded, it is the
design); rookies get the league mean (as today); the contact aggregates
were chosen for the two-stage engine and may not be the best prior-mean
features (the feature set is frozen here, not searched); numpyro on this
graph is validated by BAS-92 on the same model family but the prior-mean
block is new, so r-hat and divergences are reported per fit and any fit
with r-hat > 1.2 is named and excluded in a sensitivity, never silently.
