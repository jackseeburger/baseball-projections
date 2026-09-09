# The joint multi-component hierarchical model

**BAS-84.** Station A, layer 2. Pre-registered 2026-09-09 20:15 UTC in the
Linear issue, before any run; this file mirrors that text.

## Why

The outcomes-only hierarchical model draws with tuned Marcel on K%, BB% and
HR/PA (BAS-69, BAS-73). That is the expected result: a single-component
random-effects model on the logit *is* Marcel with a learned ballast, and
tuned Marcel already sits at that family's optimum. The Bayes model can
only win by carrying structure Marcel cannot express. The first such
structure is that the five components share one hitter: today each is fit
alone, so a hitter's power tells the walk model nothing and his strikeouts
tell the home-run model nothing. Teams fit these jointly.

## What gets built

`src/models/pa_joint.py`: one PyMC model over the same cells `pa_rate.py`
uses, with a per-batter *vector* of abilities across components (K%, BB%,
HR/PA in this pass) drawn from a multivariate normal with an LKJ(2)
correlation prior and per-component scales; the ability walk (`sigma_step`
per component) and the age curve stay per component; league trend,
handedness and park per component as today. Each component keeps its own
binomial likelihood on its own trials. `BayesArmConfig` gains `joint=True`;
the dense sweep gets `--variants joint_walk`; the checkpoint keys on
(component, season, cutoff) as before so the joint fit fills three
components per cell.

Scope: 2022/2024/2025/2026 × 12 biweekly cutoffs, 2 chains × 500 draws,
cores=1, no pitcher effect, one fit per cell yielding all three
components. Comparators on the common set, SE clustered by player:
`bayes_walk` (the single-component model, same cells), `marcel_tuned`, and
the served engine `contact_additive`.

## Pre-registered predictions

1. **The correlations are real.** Posterior correlation between the K% and
   HR/PA abilities is positive and its 95% interval excludes zero at every
   cutoff (power hitters strike out more); between BB% and K% it is
   positive; if any interval straddles zero at more than a quarter of
   cutoffs, the joint structure adds nothing and 2–4 are moot.
2. **HR/PA gains from borrowing strength.** `joint_walk` beats `bayes_walk`
   on HR/PA by ≥ 1.5% of MAE, clustered |t| > 2.5; the rare event is where
   a second channel of information about the player is worth most.
3. **K% and BB% draw** with `bayes_walk` (|Δ| ≤ 0.0003, |t| < 2): the
   high-count components already saturate.
4. **Against the served engine:** `joint_walk` vs `contact_additive` on
   HR/PA is Δ < 0 with t < −2 *only if* prediction 2's gain exceeds contact
   quality's own (≈ 3% on HR/PA); the honest expectation is that it does
   not, and the joint model with the contact covariates (BAS-83's block)
   is the arm that would.
5. **Vacuity:** every `sigma_step` stays above 0.02 and no LKJ correlation
   posterior is pinned at ±1.

### Failure conditions

- Prediction 1 fails: report, stop, do not tune the prior.
- Prediction 2 fails with 1 holding: the components are correlated but the
  correlation carries no predictive information beyond the player's own
  history; say so in those words.

## What ships

Nothing in this pass. Serving is its own ticket under architecture.md §3.
The joint model becomes the base the covariates (BAS-83) and the
measurement channel (BAS-85, `docs/bayes-measurement.md`) build on,
whatever the verdict.
