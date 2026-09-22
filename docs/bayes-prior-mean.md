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

## Results (grid completed 2026-09-11, written up 2026-09-22)

Evidence `data/eval/bas94/` (`RESULT.md`, `analysis_bas94.json`,
`cells_bayes.parquet`, `cells_scored.parquet`, `bayes_fits.json`,
`sampler_audit.json`, the per-season interims); model
`src/models/pa_prior_mean.py` and `ModelOptions.prior_mean_covariates`;
harness `--bayes-prior-mean contact` / `contact_cur`; scorer
`scripts/analyze_bas94.py`.

**Verdict: the prior moves, and shrinking toward similar players does not
help where it should. Nothing ships.** Prediction 1 holds and 2 fails,
which is the second failure condition. Arm A does not beat
`contact_additive` on either component, so no serving ticket follows and
the arm stays behind the flag.

### Fits

192 numpyro fits (2 arms × 2 components × 48 cutoffs across 2022, 2024,
2025, 2026), 2 chains × 500 draws, 9.0 fit-hours. **No fit exceeded r-hat
1.2 and none diverged** (max r-hat 1.06 for arm A, 1.10 for arm B), so
the converged-only sensitivity is identical to the headline. Comparators
`bayes_walk`, `contact_additive` and `marcel_tuned` are the BAS-85 grid's
own rows on the same 96 cells, spliced rather than refit; this run's
independently computed `marcel_tuned` matches BAS-85's at max |diff| 0.0
over 28,934 rows.

### How the features were built

`x[batter, season]` is the standardised contact profile built through
the served engine's own pipeline (`contact.window_counts`,
`contact_metrics`, `standardize`, `pa_covariates.season_cutoff`'s month
lag): `contact.FEATURES` for every component plus whiff share per swing
for K%, shrunk toward the window's league rate by 15 swings. Arm A reads
the previous two full seasons only (a test asserts the design matrix is
bit-identical after deleting the whole current season, and identical
between April and August). Arm B adds the cutoff season through the last
month boundary as raw counts against the prior profile's own exposure, so
a thin window moves the target in proportion to how much has been played.
Both weights and the ballast are the served engine's values; nothing was
tuned. 85.9% of (batter, season) pairs carry a profile at 2024; the rest
get the league mean. `assert_month_boundary` and `assert_window_clean` run
on both windows.

Model: `ability[b,s] = mu + gamma · x[b,s] + sigma_ability · z[b] +
walk`, `gamma ~ Normal(0, 0.5)`. A graph-level test recovers the walk's
innovations unchanged and reproduces the plain walk exactly at a zero
design matrix; `prior_mean_covariates=None` is the old graph bit for bit.

### Predictions

| # | pre-registered | result | verdict |
|---|---|---|---|
| 1 | gamma on barrel or EV excludes zero ≥ 90% of cutoffs (whiff for K%); prior-mean sd ≥ 30% of `sigma_ability` | HR/PA 48/48 (EV90 alone 48/48, mean +.20); K% whiff 48/48 (+.16); sd ratio HR/PA 1.82, K% 0.77, 48/48 above the floor | pass |
| 2 | May: arm A vs `bayes_walk` ≤ −3% on HR/PA (t > 2.5), ≤ −2% on K% | HR/PA **+0.66%** (t +0.73); K% **+6.93%** (t +7.19, 0–12) | fail |
| 3 | pooled: ≤ −1.5% HR/PA (t < −2), ≤ −1% K%; vs `contact_additive` within ±1.5% on HR/PA | HR/PA +1.14% (t +1.49, 18–30); K% +5.69% (t +7.39, 0–48); vs `contact_additive` **+3.83%** (t +3.73, 7–41) | fail |
| 4 | May gain shrinks by August | no gain: HR/PA +0.66% → +2.59%, K% +6.93% → +2.48% (a loss that shrinks) | fail |
| 5 | 80% coverage 75–85%; narrower than `bayes_walk` at May | HR/PA 77.5% and narrower (.0171 vs .0176); K% 76.3% but wider (.0695 vs .0651) | fail |

### Pooled, 48 cutoffs (n = 14,467 paired rows)

| component | arm | vs | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---|---|---|---|---|---|
| HR/PA | A | `bayes_walk` | +.00012 | +1.14 | +1.49 | +3.49 | 18–30 |
| HR/PA | A | `contact_additive` | +.00038 | +3.83 | +3.73 | +8.91 | 7–41 |
| HR/PA | B | `bayes_walk` | −.00001 | −0.06 | −0.07 | −0.14 | 20–28 |
| HR/PA | B | `contact_additive` | +.00026 | +2.60 | +2.43 | +3.94 | 12–36 |
| K% | A | `bayes_walk` | +.00165 | +5.69 | +7.39 | +9.89 | 0–48 |
| K% | A | `contact_additive` | +.00258 | +9.18 | +7.68 | +10.96 | 0–48 |
| K% | B | `bayes_walk` | +.00064 | +2.19 | +2.72 | +6.20 | 15–33 |
| K% | B | `contact_additive` | +.00156 | +5.57 | +5.67 | +10.47 | 5–43 |

Arm B beats arm A on both components (HR/PA −1.18%, 36–12; K% −3.30%,
44–4) and in every season except 2026 HR/PA. Per season, arm A vs
`bayes_walk` on HR/PA: 2022 +3.8%, 2024 −0.6%, 2025 +0.9%, 2026 +0.5%;
on K%: +8.6%, +5.5%, +3.0%, +5.4%, 0–12 each. `contact_additive` remains
the best arm on both components (−2.44% vs `marcel_tuned` on HR/PA,
35–13; −2.07% on K%, 39–9).

### What complicates the reading

- **The borrowed comparator's sampler is worth up to 2.9 points of K%
  MAE.** BAS-85's `bayes_walk` rows were fit under pymc; these arms ran
  under numpyro. Refitting `bayes_walk` under numpyro on 2024 (24 fits,
  `sampler_audit.json`) shows the numpyro walk is +0.87% worse on HR/PA
  and +2.84% worse on K% than the pymc one. Against a same-sampler
  comparator, arm A's 2024 K% deficit halves (+5.5% → +2.6%) and its
  HR/PA figure improves (−0.6% → −1.5%; May −1.5%, t −0.9). No verdict
  moves: arm A still loses on K% and still misses the −3% May floor on
  HR/PA. The headline K% magnitudes are part sampler, and the audit
  covers 2024 only.
- **The prior mean is not weak; it is too strong.** `sigma_ability`
  collapses under it (HR/PA 0.20 against a prior-mean spread of 0.36):
  the model hands the profile most of the between-player variance and
  then shrinks the outcome hard toward it. That is why K% is 0–48: the
  whiff feature sets a target the outcomes cannot pull back from. Arm B,
  which lets the current season into the target, is uniformly better,
  the opposite of the BAS-83 ordering.
- **The algebra.** Under a Gaussian ability, a covariate in the prior
  mean and a covariate in the likelihood are the same model; the whole
  content of this ticket was the window (two finished seasons instead of
  the thin current one), and the window did not rescue it.
- Park factors were present for every arm here (BAS-85's published
  numbers were computed at a neutral park), so absolute MAEs are not
  comparable across the two docs; the spliced comparators share this
  run's footing.

### What this means

- **For serving:** nothing changes. `contact_additive` on all five hitter
  components; `bayes_walk` the reference hierarchical arm.
- **For the structural track:** closed on public data at these sample
  sizes. Every hierarchical structure the roadmap named has now been
  pre-registered and scored on the same four seasons: single-component
  random effects (draw), correlated components (a third of a percent),
  covariates in the likelihood (worse), measurement channels (worse),
  covariates in the prior (worse). What wins, every time, is a
  measurement layer fitted separately on Statcast and added as a linear
  correction to tuned Marcel. The hierarchical machinery stays built and
  tested; its remaining unsat exam is width with structure from the joint
  posterior at layer 7.
- **The one thread worth keeping:** arm B's ordering says the current
  season's profile, ballasted, is information the outcome alone does not
  carry, and the served engine already uses it that way.
