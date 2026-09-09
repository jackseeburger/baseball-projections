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
