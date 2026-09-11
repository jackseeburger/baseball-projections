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

## Results (2026-09-11)

Evidence `data/eval/bas85/` (`analysis_bas85.json`, `RESULT.md`,
`analysis_console.txt`, `NUMPYRO_NON_IDENTIFIABILITY.md`, the per-season
interims, both sampler smokes); model `src/models/pa_measurement.py`;
harness `--variants measurement_walk` with the coverage quantiles the
dense sweep now carries; scorer `scripts/analyze_bas85.py`, which prints
every table twice (the pre-registered all-cutoff analysis and a
converged-only sensitivity) and names the fits it excludes.

Grid as pre-registered: 48 cutoffs across 2022, 2024, 2025 and 2026, K%
and HR/PA, pymc 2 chains × 500 draws, 192 fit records, 27 fit-hours.
Two deviations, recorded: the sampler is pymc, not the sweep's numpyro
default, because numpyro produces an invalid posterior on this graph in
both parameterisations (r-hat 2.2, the two power loadings collapsed to
zero, documented in the evidence directory); and every arm ran at a
neutral park, because the worktree predates `data/features/park_factors`.

**Verdict: every pre-registered prediction fails; nothing ships. The
channels are real, the model is not.**

### Predictions

| # | pre-registered | result | verdict |
|---|---|---|---|
| 1 | loadings exclude zero at every cutoff | 45/48 barrel, 45/48 EV, 44/48 whiff | fail |
| 2 | May ≥ 3% better than `contact_additive` on HR/PA (t > 2.5); August gap ≤ 1.5% | May **+8.8% worse** (t +5.0); August +1.3% | fail |
| 3 | pooled HR/PA Δ < 0 at t < −2 vs `contact_additive`; K% within ±.0005 | HR/PA **+.00054** (t +4.0); K% **+.00271** | fail |
| 4 | 80% interval covers 75–85%; `bayes_walk` < 75% at May | HR/PA 75.1% but `bayes_walk` 76.9% at May; K% 72.2% | fail |
| 5 | latent scale > .02; loading correlations < .95 | scales fine (0 of 96); 6 of 96 loading pairs ≥ .95 | fail |

### Pooled, 48 cutoffs (converged 43 in brackets)

| component | vs | Δ MAE | % of base | t(player) | t(cell) | W–L |
|---|---|---|---|---|---|---|
| HR/PA | `contact_additive` | +.00054 (+.00052) | +5.5 (+5.2) | +4.0 | +8.4 | 7–41 (7–36) |
| HR/PA | `bayes_walk` | +.00028 (+.00024) | +2.7 (+2.4) | +2.5 | +5.8 | 10–38 (10–33) |
| HR/PA | `marcel_tuned` | +.00029 (+.00027) | +2.9 (+2.7) | +2.5 | +6.6 | 9–39 (9–34) |
| K% | `contact_additive` | +.00271 (+.00235) | +9.7 (+8.4) | +6.6 | +11.3 | 0–48 (0–43) |
| K% | `bayes_walk` | +.00178 (+.00139) | +6.2 (+4.8) | +4.5 | +7.5 | 0–48 (0–43) |
| K% | `marcel_tuned` | +.00212 | +7.4 | +5.0 | +9.1 | 0–48 |

By season, HR/PA vs `contact_additive`: 2022 +1.9% (5–7), 2024 +6.1%
(0–12), 2025 +6.4% (0–12), 2026 +8.0% (2–10). Vs `bayes_walk`: 2022
+4.0%, 2024 +0.7% (a draw once the three non-converged fits are removed:
−0.4%, 5–4), 2025 +3.4%, 2026 +2.9%. K% loses to everything in every
season, 0–12 each. The controls behave: `contact_additive` beats
`marcel_tuned` 35–13 on HR/PA and 39–9 on K%.

### Non-converged fits

Five of 48 measurement fits: 2022-05-13, 2022-06-24, 2024-06-24,
2024-07-08, 2024-07-22, r-hat 1.84–1.86, ESS 3, three with the two power
loadings correlated at .999. This is the scale-and-sign ridge on the HR
latent, not a benign mirror: `sigma_ability[hr]` halves in the collapsed
chain and the projections lose half their spread, so those cells are
degraded, not relabelled. They are scored in the headline (the
pre-registered analysis) and excluded in the sensitivity; the verdict is
the same either way.

### What complicates the reading

- **The channels load.** Over the 43 converged cutoffs prediction 1 holds
  43 of 43 on all three loadings, and prediction 5 holds outright. Barrel
  loads +.55 to +.69 per sd of latent talent, exit velocity +2.0 to +2.2
  mph per sd, whiff +.34, stable across four seasons. Prediction 1's
  failure is the sampler, and the measurement carries real signal. It
  just does not improve the projection.
- **The K% failure is a channel-definition error.** The whiff channel
  carries about 789k swings against about 65k partial-season PA, twelve
  times the outcome channel's exposure, so the K latent is determined by
  whiff-per-swing, which misses called strikes and fouls. The latent
  measures the wrong thing and the K logit inherits it; 0–48 is what that
  looks like.
- **The HR/PA failure is the BAS-83 mechanism in a different coat.** The
  served engine fits a linear correction on tuned Marcel from cells of
  seasons with full-season contact; the joint model reads the thin
  current-season channel through a latent and shrinks the outcome toward
  it. At these sample sizes the two-stage fit is the better use of the
  same measurement.
- numpyro, the sweep's default, would have failed prediction 1 spuriously
  and 3.6× faster; the evidence directory carries the diagnosis.

### What this means

- **For serving:** nothing changes. `contact_additive` stays on all five
  hitter components; `bayes_walk` stays the reference hierarchical arm.
- **For the structural track:** this closes the three-ticket arc
  (BAS-83 covariates in the likelihood, BAS-84 joint components, BAS-85
  measurement channels). Correlated components buy a third of a percent on
  K% and BB%; the other two structures lose. What the hierarchical models
  have not lost is the exams they were built for and have not yet sat:
  posterior width with structure (the joint posterior's, now that the
  single-component width is known to be a rescale of the Beta's, BAS-92),
  and time and age as a process.
- **The follow-up worth registering, not opened:** (a) the contact
  channel as CSW per pitch, so the K latent measures what K% measures;
  (b) pin `lambda_barrel > 0` and anchor `sigma_ability[hr]` to its
  single-component posterior, since HR/PA at ~3% cannot identify that
  scale alone and sign-pinning by itself leaves the ridge; (c) an
  explicit exposure cap per channel so a channel with twelve times the
  outcome's trials cannot dominate the latent it is meant to inform. The
  honest prediction to write into it: CSW turns K% into a draw or better,
  and HR/PA still loses to `contact_additive`, because a fitted linear
  correction on Marcel is a better use of contact quality than a
  latent-variable model at these sample sizes.
