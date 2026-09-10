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

## Results (2026-09-10, the full grid)

Scope, exactly the pre-registration's: 2022/2024/2025/2026 × 12 biweekly
cutoffs, one `joint_walk` fit per cell filling K%, BB% and HR/PA (48 joint
fits, ~6,500 free parameters each, 2 chains × 500 draws, numpyro, cores=1,
no pitcher effect), with `bayes_walk` (144 single-component fits) and
`contact_additive` fitted **in the same cell** so every comparison is
paired on one common-player set; n = 14,467 rows, SE clustered by player.
Eight hours of sampler time, interrupted twice by container restarts and
resumed from the checkpoint; 144/144 cells present, no duplicates.
Evidence: `data/eval/bas84/` (cells, every fit's diagnostics and
correlation posteriors, `analysis_bas84.json` with pooled and per-season
tables). Nothing is served.

### Scoring the predictions

1. **The correlations are real — FAILS as written, informatively.** Pooled
   over 48 cutoffs: BB%–HR/PA +0.40 (excludes zero 48/48), K%–HR/PA +0.24
   (36/48), K%–BB% +0.07 (41/48, sign flips). The kill switch was "any
   interval straddles zero at more than a quarter of cutoffs"; K%–HR/PA
   straddles at exactly 12/48. The pooled number averages two worlds:

   | season | K%–HR/PA | K%–BB% | BB%–HR/PA |
   | --- | --- | --- | --- |
   | 2022 | −0.06, 0/12 exclude zero | −0.11, 12/12 | +0.54, 12/12 |
   | 2024 | +0.35, 12/12 | +0.09, 5/12 | +0.39, 12/12 |
   | 2025 | +0.34, 12/12 | +0.14, 12/12 | +0.35, 12/12 |
   | 2026 | +0.33, 12/12 | +0.14, 12/12 | +0.33, 12/12 |

   Within a season the correlation barely moves across twelve cutoffs;
   across seasons 2022 flips the sign of two of the three pairs. "Power
   hitters strike out more" holds in three seasons of four and is absent
   in the fourth. Only BB%–HR/PA holds everywhere.
2. **HR/PA gains from borrowing strength — FAILS.** `joint_walk` vs
   `bayes_walk`: −0.15% of MAE, t(player) −0.78, cells 29–19, against a
   bar of ≥ 1.5% at |t| > 2.5. Per season −0.27 / −0.60 / +0.23 / +0.10%.
   This is the pre-registration's second failure condition, in its own
   words: the components are correlated but the correlation carries no
   predictive information beyond the player's own history.
3. **K% and BB% draw — FAILS in the joint model's favour, on size.** Both
   are small, reliable wins rather than draws: K% −0.27% (t −3.39, cells
   41–7), BB% −0.47% (t −3.25, 35–13). |Δ| sits inside the ±0.0003 band;
   |t| does not sit inside 2. This is the clearest real finding on the
   grid: the joint model's only consistent gain is on the two high-count
   components, about a third of a percent, too small to serve.
4. **Against the served engine — FAILS, as expected.** `joint_walk` vs
   `contact_additive` on HR/PA +2.51% (t 2.42, 14–34); on K% +3.03%
   (t 3.91); on BB% +2.55% (t 3.05). Contact quality remains the
   information that pays; joint structure is not a substitute for it.
5. **Vacuity — PASSES.** Every `sigma_step` above 0.02 in all 48 fits
   (K% 0.134, BB% 0.128, HR/PA 0.168 on average); no correlation pinned
   (max |mean| 0.56).

### The pooled tables

| component | comparison | Δ MAE | % base | t (player) | t (cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- |
| K% | joint vs bayes_walk | −0.00008 | −0.27% | −3.39 | −6.60 | 41–7 |
| K% | joint vs marcel_tuned | +0.00026 | +0.90% | 1.08 | 4.15 | 10–38 |
| K% | joint vs contact_additive | +0.00085 | +3.03% | 3.91 | 9.81 | 4–44 |
| BB% | joint vs bayes_walk | −0.00009 | −0.47% | −3.25 | −6.29 | 35–13 |
| BB% | joint vs marcel_tuned | +0.00024 | +1.31% | 1.51 | 2.30 | 25–23 |
| BB% | joint vs contact_additive | +0.00046 | +2.55% | 3.05 | 3.84 | 16–32 |
| HR/PA | joint vs bayes_walk | −0.00002 | −0.15% | −0.78 | −1.88 | 29–19 |
| HR/PA | joint vs marcel_tuned | +0.00000 | +0.02% | 0.03 | 0.07 | 25–23 |
| HR/PA | joint vs contact_additive | +0.00025 | +2.51% | 2.42 | 3.88 | 14–34 |

Joint-vs-walk on K% strengthens season by season (−0.09, −0.31, −0.33,
−0.36%); on HR/PA it never does.

### What this means

The structure is real and it is not where the money is. A hitter's
strikeouts, walks and power do share information, and the joint model
extracts it, but what it extracts moves the high-count estimates by a
third of a percent and the rare-event estimate not at all, even in the
three seasons where the K%–HR/PA correlation is strong and tight. The
inversion of the pre-registration (wins where a draw was predicted, a draw
where a win was predicted) is the result. Contact quality still beats
every Bayes arm on every component: the measurement channel, not the
correlation channel, is the one that carries information a ballast lacks.
The joint model stays in the code as the base the measurement model
(BAS-85) is built on, which is where the two channels meet.

### Things that complicate the reading

- **Nothing on this grid clears the r-hat gate.** At 2 × 500 draws with
  ~6,500 parameters every joint fit sits at r-hat 1.02–1.06 (max 1.058,
  min ESS 49, zero divergences), worst on the walk innovations and
  step-size hyperparameters, which is exactly what prediction 5 rests on.
  The comparator `bayes_walk` is worse mixed (max r-hat 1.129, min ESS
  13), so the joint model is the better-sampled arm of the pair. The
  predictive comparisons average over 48 fits and are robust to this; the
  `sigma_step` table is the number least worth defending from these chains.
- **The correlation is a per-season fact.** Pooling across seasons is how
  prediction 1 was written, and it averages one −0.06 and three +0.34s
  into a +0.24 that describes no season. Any serving decision built on the
  correlation would need to know which regime it is in.
- **No park factors exist**, so every arm ran at a neutral park (BAS-86).
  Uniform across arms, so the comparisons are fair, but HR/PA is the
  component where park matters most.
- **2026 is a partial season** (PA data through August) with the shortest
  horizons and smallest population (n = 3,262 vs 3,808 in 2022), and it is
  where `contact_additive` beats the joint arm most heavily.
- **t(cell) is 2–4× t(player)** throughout. Within a cell every row shares
  one MCMC fit, so cell clustering is arguably the conservative grouping
  for a Bayes-vs-Bayes comparison and the optimistic one here; the
  player-clustered number is reported as primary per the harness
  convention, and the two disagree about how much evidence 48 cells carry.
- `bayes_walk` here is a fresh refit paired in-cell, not BAS-73's
  published rows, so its numbers do not match that grid cell for cell.
