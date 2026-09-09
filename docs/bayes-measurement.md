# The measurement model: one latent talent, several observation channels

**BAS-85.** Station A, layer 2. Pre-registered 2026-09-09 20:15 UTC in the
Linear issue, before any run; this file mirrors that text.

## Why

Contact quality is served as a *covariate on Marcel* (BAS-72) and is being
tested as a covariate inside the hierarchical model (BAS-83). Both treat
the measurement as a fixed regressor. A team's model treats it as a second
observation of the same latent talent: the hitter's true power produces
both his home-run count (binomial, noisy at 600 PA) and his exit-velocity
and launch-angle distribution (measured every batted ball, far less noisy).
A measurement model with two observation channels and one latent state is
how a 200-PA sample can carry a full-season posterior. This is the
structure that lets the model *beat* the outcome-only ballast rather than
tie it.

## What gets built

`src/models/pa_measurement.py`: per batter-season (or batter-month under
the walk) one latent power state `theta_hr` and one latent contact state
`theta_k`; HR/PA ~ Binomial(logistic(f(theta_hr) + …)) as today; barrel
rate per batted ball ~ Binomial(logistic(g(theta_hr))) and mean exit
velocity ~ Normal(h(theta_hr), sigma_ev/√BBE) as second and third
channels; whiff share as the K% channel. Channel loadings are global
parameters learned across players and seasons. Builds on the joint model
(BAS-84) if it has landed, otherwise on `pa_rate.py`. Walk-forward exactly
as the harness: at a cutoff, only batted balls strictly before it are
observed. Cells, seasons, draws, comparators and clustering as in BAS-84;
components K% and HR/PA.

## Pre-registered predictions

1. **The channels load.** Posterior loadings of barrel rate and exit
   velocity on `theta_hr`, and of whiff share on `theta_k`, exclude zero at
   every cutoff.
2. **Early-season is where it wins.** At the May cutoffs, `measurement`
   beats `contact_additive` on HR/PA by ≥ 3% of MAE (|t| > 2.5); at the
   August cutoffs the gap is ≤ 1.5%. Information, not denoising, and most
   of it when outcomes are thin.
3. **Pooled over cutoffs:** `measurement` vs `contact_additive` on HR/PA is
   Δ < 0, clustered t < −2 — the first arm that would replace the served
   engine — and on K% is a draw within ±0.0005.
4. **The posterior width is right:** the 80% posterior interval on the
   rest-of-season rate covers the realised rate 75–85% of the time on the
   common set; `contact_additive` has no interval and `bayes_walk`'s covers
   < 75% at May.
5. **Vacuity:** the latent-state scale stays above 0.02 and the loadings
   are not degenerate (|corr| between any two channel loadings < 0.95).

### Failure conditions

- Prediction 3 fails with 1 and 2 holding: early-season gains exist but
  wash out over a season; the model earns a place in the props engine
  (short horizons), not the season projection, and the roadmap says so.
- Prediction 1 fails: the channel definitions are wrong, fix before
  reading anything else.

## What ships

Nothing in this pass. Serving is its own ticket under architecture.md §3.
