# The fair fight: the hierarchical PA model with the layer-1 measurements as covariates

**BAS-83.** Station A, layer 2. Pre-registered 2026-09-09 17:20 UTC in the
Linear issue, before any run; this file mirrors that text.

## Why

Teams do not run hierarchical Bayes on outcomes alone: the measurements go
*into* the model and the partial pooling does the rest. We built contact
quality (BAS-58) and stuff (BAS-71) first and served them as corrections on
Marcel because that cleared the gate fastest. The K% hierarchical model
draws with tuned Marcel on outcomes alone (BAS-69), and HR/PA's flat arm
beat Marcel at 11 of 12 cutoffs in smoke (BAS-73). The fight that decides
which is the layer-2 engine has never been run: **Bayes + measurement vs
Marcel + measurement**.

## What gets built

`src/models/pa_rate.py` gains an optional per-batter-season covariate
block: the contact-quality aggregates summed strictly before the cutoff
(month-lagged exactly as `ros.contact_cutoff` does), standardised per
season, entering the batter's logit-rate as `beta_cov · x` with
`beta_cov ~ Normal(0, 0.5)` per aggregate, on top of the ability walk.
`ModelOptions.covariates` names them (`None` is the old model, bit for
bit); `BayesArmConfig` exposes it; the dense sweep gets
`--bayes-covariates contact`. Batters with no contact history before the
cutoff get the league mean (x = 0 after standardisation). The comparator
is the served engine, `contact_additive` (Marcel + the same aggregates,
baseline pinned at 1), on the same cells (2022–2026, biweekly), paired per
batter, SE clustered by batter, for K% and HR/PA (BB% if the grid has
finished).

## Pre-registered predictions

1. **The covariate transfers.** `bayes_walk+contact` beats `bayes_walk` on
   K% by ≥ 1.0% of MAE (|t| > 2.5) and on HR/PA by ≥ 2.0% — the
   measurement helps the Bayes model at least as much as it helped Marcel.
2. **K%: a draw with the served engine.** `bayes_walk+contact` vs
   `contact_additive`: |Δ| ≤ 0.0005, |t| < 2. Same shape as BAS-69's draw;
   the covariate does not break the tie.
3. **HR/PA: the Bayes model wins.** `bayes_flat+contact` (flat, because the
   smoke said the walk gives HR/PA's gain back) beats `contact_additive`
   with Δ < 0, clustered t < −2 — the first component where the
   hierarchical arm beats the served engine and would replace it.
4. **`sigma_step` shrinks** by ≥ 15% on K% when the covariate is present:
   the walk was absorbing what the measurement now explains.
5. **Vacuity:** the posterior on `beta_cov` for barrel rate excludes zero
   on HR/PA and for average EV on K%; if it does not, the covariate did
   not enter the model and 1–4 are untestable.

### Failure conditions

- Prediction 3 fails: the Bayes model stays a research arm on HR/PA too,
  and the roadmap says so in those words.
- Prediction 1 fails: the covariate block is wrong, not the model — fix
  the block before reading anything else.

## What ships

Nothing in this pass. Serving is its own ticket under architecture.md §3
(the effect-floor rule), with a pre-registered Serving section here first.
