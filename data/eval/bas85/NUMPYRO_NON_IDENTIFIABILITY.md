# NumPyro cannot sample the measurement model's HR latent (BAS-85)

**One line:** the sweep's *default* sampler produces an invalid posterior on
this graph, and the way it fails looks exactly like a clean failure of
pre-registered prediction 1 — the two power loadings collapse onto zero with
prior-wide intervals — so the BAS-85 grid was run on PyMC instead, at roughly
3.6x the wall time.

Everything below is one cell: 2024, cutoff 2024-06-01, `measurement_walk`,
2 chains x 300 draws, `cores=1`, seed 59, neutral park. Raw records are in
`smoke_pymc/` and `smoke_numpyro/` beside this file.

## What was seen

| | wall | max R-hat | min ESS-bulk | divergences |
|---|---|---|---|---|
| PyMC | 1,015 s | 1.038 | 103 | 0 |
| NumPyro | 285 s | **2.232** (`z_ability`) | **3** | 54 |

Loadings, per sd of ability:

| | NumPyro | PyMC |
|---|---|---|
| `lambda_barrel` | +0.005 **[-0.613, +0.611]** | +0.583 [+0.544, +0.623] |
| `lambda_ev` | -0.005 **[-0.158, +0.145]** | +0.138 [+0.128, +0.149] |
| `lambda_whiff` | +0.348 [+0.325, +0.370] | +0.343 [+0.324, +0.362] |
| `sigma_ability[hr_rate]` | 0.237 | 0.476 |

## It is not the step size, and it is not the funnel

Two fixes were tried and neither worked, which is what identifies the cause.

**1. `target_accept` 0.95, then 0.99.** At 0.99 every divergence disappears
(54 -> 0) and R-hat stays at 2.24. The chains were never taking bad steps;
they were in different places and could not mix.

**2. Reparameterising away the division.** The channels originally read
`(theta - mu) / sigma_ability`, a parameter in a denominator, which is the
textbook funnel. That was removed — channels now read the centred latent
`theta - mu` and the loading carries the scale (`src/models/pa_measurement.py`,
and `test_sigma_ability_is_not_in_a_denominator_anywhere` pins it at the graph
level so it cannot come back). NumPyro's divergences went to zero and R-hat
only fell to 1.86, ESS still 3.

PyMC on the reparameterised graph reproduces the original run exactly, which
is what makes the change safe to keep — same posterior, better geometry,
and 15% faster into the bargain:

| per sd of ability | PyMC, divided | PyMC, reparameterised |
|---|---|---|
| `lambda_barrel` | +0.583 [+0.544, +0.623] | +0.580 [+0.542, +0.619] |
| `lambda_ev` | +0.138 (1.99 mph/sd) | +0.137 (1.98 mph/sd) |
| `lambda_whiff` | +0.343 [+0.324, +0.362] | +0.342 [+0.321, +0.363] |
| `sigma_ability[hr_rate]` | 0.476 | 0.475 |
| max abs loading corr | 0.699 | 0.563 |
| wall / R-hat / ESS / div | 1,015 s / 1.038 / 103 / 0 | 860 s / 1.059 / 121 / 0 |

The grid therefore runs the reparameterised graph on PyMC.

## What it actually is: a scale ridge in the HR latent

The reparameterised run reports **`max_abs_loading_corr` = 0.999** between
`lambda_barrel` and `lambda_ev`, with `sigma_ability[hr_rate]` at 0.246
against PyMC's 0.476, and a per-logit `lambda_barrel` of -11.60 against a
per-sd of +0.02 (two chains in opposite-sign basins, averaging to nothing).

Both power channels read `theta_hr`. Shrink `theta_hr` by *c* and both
loadings grow by 1/*c*, leaving every channel likelihood unchanged. That is a
ridge, not a funnel, which is why removing the division could not help.

What is supposed to pin the scale is the outcome loading fixed at 1 — but
HR/PA is about 3%, so at a May cutoff a hitter has ~150 PA and ~4 home runs
and the HR binomial constrains the latent's scale very weakly, while 288,739
batted balls constrain the *product* very tightly. The identification is real
but faint. PyMC's initialisation lands in the right basin; NumPyro's does not.

`lambda_whiff` is untouched in every run and reproduces PyMC to three decimals.
It reads `theta_k`, whose scale is pinned, because K% happens ~22% of the time
and its binomial is informative. The failure is localised exactly where the
theory says it should be.

## Why this belongs in the write-up

Pre-registered **prediction 5** ("the loadings are not degenerate: |corr|
between any two channel loadings < 0.95") fires on precisely this fit, at
0.999. The vacuity check catches the broken run and stops it being read as a
substantive negative on prediction 1. That is the check doing its job, and it
is the reason the grid's headline is trustworthy — but it also means any
future run of this model on a gradient backend with a different initialisation
has to be read with prediction 5 checked *first*.

Two consequences for anyone reading BAS-85's numbers:

* Every fit record carries `sampler`, `target_accept` and `parameterisation`.
  `scripts/analyze_bas85.py` refuses to pool quietly: a checkpoint whose
  measurement fits came from more than one of them prints a warning and lists
  them under `provenance`.
* The faint scale identification is a property of the *model*, not of NumPyro.
  A future version that wants a fast backend should pin `theta_hr`'s scale
  harder — the obvious candidate is fixing `sigma_ability[hr_rate]` to its
  single-component posterior rather than letting the channels bid for it.
