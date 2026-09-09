# Rolling the K% pattern to BB% and HR/PA

Tracked as BAS-73. Follow-on from [bayes-variants.md](bayes-variants.md).

**Status: pre-registered, not yet run.**

The hierarchical model exists for one of five hitting components. BB/PA and
HR/PA are per-plate-appearance binomials exactly like K/PA — same cell
structure, same likelihood, a different numerator column and a different
league-level prior — so they are the two that roll out with no new design.
ISO (per AB, not binomial) and BABIP (per ball in play) need their own
denominators and come after.

## Pre-registered predictions

Measured on the dense harness (biweekly cutoffs, 2022/2024/2025/2026),
`bayes_walk` against `marcel_tuned`, clustered by player.

1. **BB%: a draw, like K%.** Remaining deficit ≤ +0.0005, \|t\| < 2. Walks
   are a stable, high-count skill; there is nothing for pooling to add that
   a tuned ballast does not.
2. **HR/PA: the walk beats tuned Marcel.** Δ MAE < 0, clustered t < −2. Home
   runs are the rare event — a .03 rate on a few hundred trials — and rare
   events are where partial pooling earns more than a fixed ballast, because
   the right amount of shrinkage varies with the player's exposure in a way
   one constant cannot track. This is the first component where the
   hierarchical arm is predicted to *win*, not draw.
3. `sigma_step` is **smaller for BB% than for K%** (walks drift less) and
   **larger for HR/PA** (power changes faster and is noisier), on the logit
   scale.

### Vacuity check

If either component's `sigma_step` posterior mean is below 0.02, the walk
collapsed for that component and prediction 1/2 is untestable for it.

### Smoke, one season (2024)

Scope: 2024, biweekly, 12 cutoffs per component, variants flat and
ability_walk, `include_pitcher=False`, 2 chains × 500 draws (tune 500),
numpyro, 48 fits, n = 3,785 per arm on the common set. Outputs at
`data/eval/bas73_smoke` (bb_rate) and `data/eval/bas73_smoke_hr` (hr_rate).

Wall time per fit, on a box also running a gradient-boosting job:

| component | flat | ability_walk | ratio |
| --- | --- | --- | --- |
| bb_rate | 86.7s [59–122] | 122.5s [87–173] | 1.41× |
| hr_rate | 68.5s [45–94] | 98.7s [76–139] | 1.44× |

The same 1.4× the K% sweep measured. Zero divergences in all 48 fits.
Nothing is "healthy" by the diagnostic's own standard at 500 draws, the
same caveat the K% sweep carries: worst r-hat 1.0366 (bb flat), 1.0279
(bb walk), 1.0299 (hr flat), and 1.1288 for the hr walk, whose min ESS of
13 says that arm specifically needs more draws before its point estimate
is worth much.

The vacuity check passes for both, and prediction 3 holds:

| component | sigma_step | range |
| --- | --- | --- |
| bb_rate | 0.1183 | 0.1020 – 0.1261 |
| k_rate | 0.1336 | (the published sweep, for reference) |
| hr_rate | 0.1431 | 0.1253 – 0.1598 |

Smaller for BB% than K%, larger for HR/PA, exactly as pre-registered, and
neither within six times the 0.02 collapse threshold.

Paired, clustered by player, on the common set (n = 3,785 per arm):

| component | arm | vs | Δ MAE | t (player) | cells W–L |
| --- | --- | --- | --- | --- | --- |
| bb_rate | bayes_walk | marcel_tuned | +0.00025 | 1.22 | 6–6 |
| bb_rate | bayes_flat | marcel_tuned | +0.00029 | 1.62 | 6–6 |
| bb_rate | bayes_walk | bayes_flat | −0.00004 | −0.22 | 8–4 |
| hr_rate | bayes_walk | marcel_tuned | +0.00001 | 0.14 | 7–5 |
| hr_rate | bayes_flat | marcel_tuned | −0.00008 | −0.81 | 11–1 |
| hr_rate | bayes_walk | bayes_flat | +0.00009 | 1.35 | 1–11 |

One season is not the pre-registration's measurement and none of this
scores it. What it does say is where to look. BB% is a draw at one season,
consistent with prediction 1. HR/PA's point estimate goes the predicted
way, but the arm carrying it is the *flat* one — bayes_flat beats
marcel_tuned at 11 of 12 cutoffs — while the walk gives that back and
loses to flat at 11 of 12. If the full grid holds that shape, prediction
2's direction survives and its mechanism does not: the rare event rewards
partial pooling, and letting the pooled level drift is what costs it.

For the record: for bb_rate, `marcel_tuned` is bit-identical to stock
Marcel at every cutoff (max abs diff 0.0) — tuning found nothing to buy
for walks, so those are one baseline column, not two. hr_rate differs
normally (max abs diff 0.0114).

## Results — the full grid (2026-09-09)

Scope, exactly the pre-registration's: 2022/2024/2025/2026 × 12 biweekly
cutoffs × {flat, ability_walk} × {bb_rate, hr_rate}, 192 fits, 2 chains ×
500 draws (tune 500), numpyro, no pitcher effect, SE clustered by player on
the common set (n = 14,467 player-cutoffs per arm; 256/244/233/152 players
per season). Two cells died mid-sample under load and were refit from the
checkpoint; every (component, season, cutoff) is present. Evidence:
`data/eval/bayes_components_grid.json` (the variant tables and every fit's
diagnostics; the cell-level parquet stays in the gitignored dense harness).

Paired Δ MAE, arm minus base, on the common set. Negative favours the arm.

| component | arm | vs | Δ MAE | t (player) | t (cell) | cells W–L |
| --- | --- | --- | --- | --- | --- | --- |
| bb_rate | bayes_walk | marcel_tuned | +0.00033 | **2.10** | 3.22 | 22–26 |
| bb_rate | bayes_flat | marcel_tuned | +0.00026 | 1.67 | 2.61 | 17–31 |
| bb_rate | bayes_walk | bayes_flat | +0.00007 | 0.62 | 2.02 | 23–25 |
| hr_rate | bayes_walk | marcel_tuned | +0.00002 | 0.30 | 0.81 | 26–22 |
| hr_rate | bayes_flat | marcel_tuned | +0.00008 | 1.17 | 3.81 | 17–31 |
| hr_rate | bayes_walk | bayes_flat | −0.00006 | −1.20 | −2.54 | 26–22 |

For bb_rate `marcel_tuned` is still bit-identical to stock Marcel, so the
`marcel` rows are the same rows.

### Vacuity: PASS

| component | sigma_step (mean of 48 fits) | range | by season 2022/24/25/26 |
| --- | --- | --- | --- |
| bb_rate | 0.130 | 0.098 – 0.161 | 0.142 / 0.118 / 0.109 / 0.151 |
| k_rate | 0.134 | (BAS-69 sweep, for reference) | |
| hr_rate | 0.169 | 0.125 – 0.244 | 0.223 / 0.143 / 0.170 / 0.142 |

Neither is near the 0.02 collapse threshold; predictions 1 and 2 are
testable.

### Scoring the predictions

1. **BB%: a draw — FAILED on its own terms, narrowly.** The magnitude clause
   holds (deficit +0.00033 against a ≤ +0.0005 bound) and the significance
   clause does not (|t| 2.10 against < 2). The walk is worse than tuned
   Marcel by a third of a walk per thousand PA, and with 14,467 paired
   player-cutoffs that small a loss is just resolvable. This is the K%
   result again in shape (walk ≈ Marcel, t 1.40 there) with a slightly
   larger deficit; walks have nothing for pooling to add, and the pooled
   arm pays a small price for its prior. Read: draw in substance, loss by
   the pre-registered letter, and the letter is what counts.
2. **HR/PA: the walk beats tuned Marcel — FAILED.** Δ MAE +0.00002, t 0.30,
   cells 26–22. The point estimate is the wrong sign and the interval is
   centred on zero: the hierarchical arm neither beats nor loses to the
   tuned ballast on the rare event. The smoke's shape did not survive the
   grid either — there the *flat* arm led at 11 of 12 cutoffs and the walk
   gave it back; over four seasons flat is worse than Marcel (t 1.17,
   17–31) and the walk beats flat (−0.00006, t −1.20, 26–22). So the
   mechanism the pre-registration named — varying shrinkage earns more on
   a rare event — is not visible at this sample, and the smoke's opposite
   reading was one season's noise.
3. **sigma_step ordering — HOLDS.** BB% 0.130 < K% 0.134 < HR/PA 0.169 on
   the logit scale, the direction pre-registered, though the BB%–K% gap is
   inside the season-to-season spread (BB% ranges 0.109–0.151 by season)
   and only the HR/PA side of the ordering is clear.

### What this means

Three components, one verdict: **the outcomes-only hierarchical model
draws with tuned Marcel** — K% (BAS-69), BB% and HR/PA (this grid). It
does not beat it anywhere, and it loses by a hair on walks. Nothing here
argues for serving a Bayes arm on any hitter component, and nothing here
argues for throwing the model away: a two-stage Marcel with a per-component
tuned ballast is a very good shrinkage estimator on outcomes, and the
hierarchical model recovers the same thing from a prior. The place the
hierarchical model can still win is the one this grid does not test — with
the layer-1 measurements *inside* it, where pooling and covariates share
one likelihood. That is BAS-83 (`docs/bayes-covariates.md`), and it is the
fair fight that decides whether layer 2 is Bayes or Marcel-plus-covariates.

### Things that complicate the reading

- The HR/PA walk fits are the worst-sampled in the grid: min ESS 12.9 and
  max r-hat 1.129 across the 48 fits (bb walk 73 / 1.036; both flats > 74 /
  ≤ 1.037). Zero divergences anywhere. The HR/PA walk's *point* estimates
  are therefore noisier than the table implies, in both directions; more
  draws would sharpen the +0.00002, not move a t of 0.30 past 2.
- 500 draws is the K% sweep's budget, chosen for wall time (75–91 s per
  walk fit here, 1.4× flat), not for convergence. Nothing in the grid is
  "healthy" by the diagnostic's own standard, the same caveat BAS-69 carries.
- The 2026 common set is 152 players at cutoffs through late August; the
  season's weight in the pooled tables is a little over half the others'.
- The pre-registration's bar for prediction 1 was |t| < 2 and the result
  is 2.10. A bar that fails by 0.1 of a t is the case §3's effect-floor
  rule was written for on the serving side; on the scoring side there is
  no such rule and the prediction is marked failed. The practical reading
  (a draw) is stated above and is not the verdict.
