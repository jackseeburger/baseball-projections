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

## Results (2026-09-09, smoke plus one full season; the full grid was not run)

Scope actually run: 2024, 12 biweekly cutoffs, K% and HR/PA, arms
`bayes_walk`, `bayes_flat`, each with and without the contact block, and
`contact_additive` fit walk-forward on the same cells; 96 MCMC fits at
2 chains × 500 draws, one at a time (2.5 h, median 86 s a fit); n = 3,785
per arm per component; SE clustered by player. Evidence:
`data/eval/bas83/` (`analysis_bas83.json`, every fit's diagnostics and
posteriors, the cell parquet). The full 2022–2026 grid was not run: at the
measured rate it is 12–16 hours, and with prediction 1 failing by a factor
of ten on HR/PA the pre-registration's own failure condition says the block
is what to fix first.

### Vacuity (prediction 5): PASS

| coefficient | component | posterior mean | 95% | excludes zero |
| --- | --- | --- | --- | --- |
| `beta_cov_ev_mean` | K% | −0.080 | [−0.085, −0.069] | 12 / 12 fits |
| `beta_cov_barrel` | HR/PA | +0.288 | [+0.260, +0.312] | 12 / 12 fits |

Both signs are the physical ones (harder average contact, fewer
strikeouts; more barrels, more home runs). The block is in the model.

### The tables

K% (`marcel_tuned` MAE .02819):

| arm | vs | Δ MAE | % | t (player) | W–L |
| --- | --- | --- | --- | --- | --- |
| bayes_walk+contact | bayes_walk | +0.00065 | +2.3% | 2.00 | 3–9 |
| bayes_flat+contact | bayes_flat | +0.00064 | +2.2% | 1.61 | 6–6 |
| bayes_walk+contact | contact_additive | +0.00132 | +4.7% | 2.59 | 2–10 |
| bayes_flat+contact | contact_additive | +0.00158 | +5.7% | 2.75 | 3–9 |
| bayes_walk | marcel_tuned | +0.00049 | +1.7% | 1.31 | 0–12 |
| contact_additive | marcel_tuned | −0.00019 | −0.7% | −0.65 | 7–5 |

HR/PA (`marcel_tuned` MAE .01044):

| arm | vs | Δ MAE | % | t (player) | W–L |
| --- | --- | --- | --- | --- | --- |
| bayes_walk+contact | bayes_walk | +0.00200 | +19.2% | 6.10 | 0–12 |
| bayes_flat+contact | bayes_flat | +0.00210 | +20.3% | 5.88 | 0–12 |
| bayes_flat+contact | contact_additive | +0.00254 | +25.6% | 6.77 | 0–12 |
| bayes_flat | marcel_tuned | −0.00008 | −0.7% | −0.81 | 11–1 |
| contact_additive | marcel_tuned | −0.00051 | −4.9% | −2.51 | 11–1 |

`sigma_step`, with and without the block:

| component | bayes_walk | bayes_walk+contact | change |
| --- | --- | --- | --- |
| K% | 0.125 [.123–.127] | 0.124 [.119–.130] | −0.4% |
| HR/PA | 0.143 [.125–.160] | **0.023** [.018–.039] | **−84%** |

### Scoring the predictions

1. **FAILED, with the opposite sign.** The block makes `bayes_walk` worse
   by 2.3% of K% MAE (t 2.00, 3 of 12 cutoffs) and by 19.2% of HR/PA MAE
   (t 6.10, 0 of 12). On HR/PA that is not "did not help"; it is a
   different model.
2. **FAILED.** K% is not a draw with the served engine: +0.00132, t 2.59.
3. **FAILED.** HR/PA does not go the other way: `bayes_flat+contact` is
   worse than `contact_additive` by +0.00254, t 6.77, at every cutoff.
4. **FAILED on K%** (`sigma_step` moves 0.4%, not 15%). On HR/PA the
   mechanism fires and then some: an 84% collapse, on the arm that is the
   worst of the six. That is over-explaining in sample, not a covariate
   earning its keep.
5. **PASS**, without qualification.

### What the contradiction is

The block is in the graph and it hurts, and the per-cutoff split says
where: the damage is largest where the current season's contact window is
thinnest and falls about five-fold by midseason (HR/PA +0.0052 at April 15,
+0.0006 by June 10; K% +0.0032 to +0.0002). One coefficient is pooled
across a full prior season, with roughly 400 batted balls behind each
aggregate, and the partial current one, with a few dozen; the projection
reads the current season's node. So a coefficient fitted mostly on the
reliable measurement is applied to the unreliable one: errors in
variables, imported at the full window's coefficient. `contact_additive`
never has this problem, because its coefficient is fitted on cells whose
features were built at the same cutoff shape it serves, so the attenuation
is already priced in. HR/PA's `sigma_step` collapsing while the arm gets
worse is the same story from the other side: the covariate absorbs the
movement the walk used to carry.

This is exactly the failure condition the pre-registration wrote for
prediction 1: **the block is wrong, not the model**, and the fix belongs to
the block. Two fixes are on the table and neither is a re-tune of anything
scored here: an exposure-dependent coefficient (the loading shrinks with
the batted balls behind the aggregate), or the principled version, which is
the measurement model (BAS-85, `docs/bayes-measurement.md`), where each
batted ball is its own observation and a thin window carries little weight
by construction rather than by a correction. BAS-85 is the follow-up; the
covariate-as-regressor block stays in the code as the negative control it
now is.

Until then the layer-2 engine on both components is `contact_additive`,
which beats `marcel_tuned` on HR/PA by 4.9% (t −2.51, 11 of 12) and draws
on K%, replicating BAS-72's serving result on one more season.

### Things that complicate the reading

- One season. The pre-registration's measurement is four; the direction
  and size on HR/PA (t 6.1, 0 of 12) leave no reading where three more
  seasons flip it, but the K% numbers (t 2.0, 1.6) are one-season numbers.
- The worktree lacked `data/parquet/pa_outcomes/` for 2017–2025 and the
  dense-harness checkpoint, so the no-covariate twins were refit here
  rather than reused from BAS-73's grid; their numbers agree with that
  grid's 2024 rows to the third decimal. `birthdates.parquet` was also
  absent on the first launch, which silently ran on placeholder ages; that
  run was discarded and everything above is from the restart. Park factors
  were absent throughout, neutral for every arm.
- The merge that followed brought a rebuilt `contact_quality_monthly.parquet`;
  these numbers were computed against the pre-merge artifact.
