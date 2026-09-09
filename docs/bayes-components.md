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
