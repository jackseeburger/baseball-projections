# BAS-85: the measurement model — result

Grid: seasons [2022, 2024, 2025, 2026], 48 cutoffs, components ['hr_rate', 'k_rate'], 192 fit records, 27.1 fit-hours (median 300 s).

Provenance: [{'sampler': 'pymc', 'parameterisation': 'centred-latent/loading-carries-scale'}]

Every arm ran at a **neutral park** — `park_factors.parquet` does not exist in this repo.


## Verdict: all five pre-registered predictions FAIL

| prediction | result |
|---|---|
| prediction_1_loadings — the channels load at every cutoff | **FAILS** |
| prediction_2_early_season — May gain ≥3%, August gap ≤1.5% | **FAILS** |
| prediction_3_pooled — pooled HR/PA Δ<0 t<−2, K% a draw | **FAILS** |
| prediction_5_vacuity — latent scale >0.02, |loading corr| <0.95 | **FAILS** |
| prediction_4_coverage[hr_rate] | **FAILS** |
| prediction_4_coverage[k_rate] | **FAILS** |

## Non-converged fits

| cutoff | why |
|---|---|
| 2022-05-13 | R-hat 1.840 > 1.2 |
| 2022-06-24 | R-hat 1.853 > 1.2 |
| 2024-06-24 | |loading corr| 0.999 >= 0.95; R-hat 1.856 > 1.2 |
| 2024-07-08 | |loading corr| 0.999 >= 0.95; R-hat 1.836 > 1.2 |
| 2024-07-22 | |loading corr| 0.999 >= 0.95; R-hat 1.855 > 1.2 |

43/48 cutoffs converged. Every table below is given over all cutoffs (the pre-registered analysis) and over the converged ones (sensitivity).


## hr_rate — pooled, all 48 cutoffs

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 14467 | +0.00054 | 5.45 | 4.04 | 8.42 | 7–41 |
| `bayes_measurement_walk` | `bayes_walk` | 14467 | +0.00028 | 2.72 | 2.53 | 5.82 | 10–38 |
| `bayes_measurement_walk` | `marcel_tuned` | 14467 | +0.00029 | 2.88 | 2.46 | 6.62 | 9–39 |
| `bayes_walk` | `contact_additive` | 14467 | +0.00026 | 2.66 | 2.59 | 4.18 | 13–35 |
| `contact_additive` | `marcel_tuned` | 14467 | -0.00025 | -2.44 | -2.22 | -4.17 | 35–13 |

### hr_rate — converged 43 cutoffs only

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 12890 | +0.00052 | 5.23 | 3.94 | 7.52 | 7–36 |
| `bayes_measurement_walk` | `bayes_walk` | 12890 | +0.00024 | 2.37 | 2.15 | 4.88 | 10–33 |
| `bayes_measurement_walk` | `marcel_tuned` | 12890 | +0.00027 | 2.66 | 2.19 | 5.67 | 9–34 |
| `bayes_walk` | `contact_additive` | 12890 | +0.00028 | 2.79 | 2.75 | 4.13 | 12–31 |
| `contact_additive` | `marcel_tuned` | 12890 | -0.00025 | -2.43 | -2.23 | -3.87 | 31–12 |

### hr_rate — by season (all cutoffs)


**2022**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3808 | +0.00018 | 1.88 | 0.63 | 1.18 | 5–7 |
| `bayes_measurement_walk` | `bayes_walk` | 3808 | +0.00037 | 3.96 | 1.74 | 4.44 | 2–10 |
| `bayes_measurement_walk` | `marcel_tuned` | 3808 | +0.00036 | 3.87 | 1.48 | 5.64 | 1–11 |

**2024**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3785 | +0.00060 | 6.09 | 2.28 | 10.67 | 0–12 |
| `bayes_measurement_walk` | `bayes_walk` | 3785 | +0.00008 | 0.73 | 0.35 | 0.82 | 5–7 |
| `bayes_measurement_walk` | `marcel_tuned` | 3785 | +0.00009 | 0.87 | 0.42 | 1.17 | 4–8 |

**2025**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3612 | +0.00064 | 6.41 | 2.71 | 7.07 | 0–12 |
| `bayes_measurement_walk` | `bayes_walk` | 3612 | +0.00035 | 3.41 | 1.59 | 3.75 | 1–11 |
| `bayes_measurement_walk` | `marcel_tuned` | 3612 | +0.00025 | 2.39 | 1.01 | 2.67 | 4–8 |

**2026**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3262 | +0.00081 | 7.97 | 3.00 | 11.68 | 2–10 |
| `bayes_measurement_walk` | `bayes_walk` | 3262 | +0.00031 | 2.90 | 1.34 | 6.14 | 2–10 |
| `bayes_measurement_walk` | `marcel_tuned` | 3262 | +0.00051 | 4.89 | 2.14 | 10.40 | 0–12 |

## k_rate — pooled, all 48 cutoffs

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 14467 | +0.00271 | 9.66 | 6.59 | 11.33 | 0–48 |
| `bayes_measurement_walk` | `bayes_walk` | 14467 | +0.00178 | 6.15 | 4.48 | 7.50 | 0–48 |
| `bayes_measurement_walk` | `marcel_tuned` | 14467 | +0.00212 | 7.39 | 4.99 | 9.10 | 0–48 |
| `bayes_walk` | `contact_additive` | 14467 | +0.00093 | 3.31 | 4.21 | 10.83 | 3–45 |
| `contact_additive` | `marcel_tuned` | 14467 | -0.00059 | -2.07 | -3.95 | -7.26 | 39–9 |

### k_rate — converged 43 cutoffs only

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 12890 | +0.00235 | 8.40 | 5.42 | 14.27 | 0–43 |
| `bayes_measurement_walk` | `bayes_walk` | 12890 | +0.00139 | 4.81 | 3.40 | 9.28 | 0–43 |
| `bayes_measurement_walk` | `marcel_tuned` | 12890 | +0.00174 | 6.10 | 3.80 | 10.92 | 0–43 |
| `bayes_walk` | `contact_additive` | 12890 | +0.00096 | 3.43 | 4.38 | 10.40 | 3–40 |
| `contact_additive` | `marcel_tuned` | 12890 | -0.00061 | -2.12 | -4.05 | -7.22 | 36–7 |

### k_rate — by season (all cutoffs)


**2022**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3808 | +0.00366 | 13.20 | 4.98 | 6.62 | 0–12 |
| `bayes_measurement_walk` | `bayes_walk` | 3808 | +0.00245 | 8.48 | 3.38 | 4.04 | 0–12 |
| `bayes_measurement_walk` | `marcel_tuned` | 3808 | +0.00252 | 8.75 | 3.41 | 4.38 | 0–12 |

**2024**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3785 | +0.00318 | 11.35 | 4.26 | 7.49 | 0–12 |
| `bayes_measurement_walk` | `bayes_walk` | 3785 | +0.00249 | 8.66 | 3.33 | 6.12 | 0–12 |
| `bayes_measurement_walk` | `marcel_tuned` | 3785 | +0.00299 | 10.62 | 4.05 | 7.51 | 0–12 |

**2025**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3612 | +0.00141 | 4.90 | 1.83 | 13.18 | 0–12 |
| `bayes_measurement_walk` | `bayes_walk` | 3612 | +0.00087 | 2.96 | 1.17 | 5.51 | 0–12 |
| `bayes_measurement_walk` | `marcel_tuned` | 3612 | +0.00098 | 3.35 | 1.21 | 6.64 | 0–12 |

**2026**

| arm | base | n | Δ MAE | % base | t(player) | t(cell) | W–L |
|---|---|---:|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | `contact_additive` | 3262 | +0.00246 | 8.89 | 2.96 | 21.26 | 0–12 |
| `bayes_measurement_walk` | `bayes_walk` | 3262 | +0.00112 | 3.84 | 1.47 | 9.80 | 0–12 |
| `bayes_measurement_walk` | `marcel_tuned` | 3262 | +0.00187 | 6.62 | 2.22 | 20.39 | 0–12 |

## Prediction 2 — the May/August split (HR/PA vs contact_additive)

| regime | % of base MAE | t(player) | pre-registered | holds |
|---|---:|---:|---|---|
| may | +8.78 | +5.03 | ≤ −3.0% and |t| > 2.5 | **no** |
| august | +1.28 | +0.73 | |gap| ≤ 1.5% | yes |

Converged-only: may +8.73%, august +1.28%.


## Prediction 3 — pooled

| component | Δ MAE | t(player) | pre-registered | holds |
|---|---:|---:|---|---|
| hr_rate | +0.00054 | +4.04 | Δ<0 and t<−2 | **no** |
| k_rate | +0.00271 | — | \|Δ\| ≤ 0.0005 | **no** |

## Prediction 4 — coverage of the 80% interval


**hr_rate** (scored on the `predictive` reading; target 75%–85%)

| arm | predictive | rate-only | n | miss low | miss high |
|---|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | 75.1% | 37.6% | 14467 | 8.0% | 16.8% |
| `bayes_measurement_walk_may` | 73.5% | 41.4% | 4055 | 7.5% | 19.0% |
| `bayes_walk_may` | 76.9% | 50.9% | 4055 | 6.0% | 17.1% |
| `contact_additive` | (no interval) | (no interval) | – | – | – |

**k_rate** (scored on the `predictive` reading; target 75%–85%)

| arm | predictive | rate-only | n | miss low | miss high |
|---|---:|---:|---:|---:|---:|
| `bayes_measurement_walk` | 72.2% | 46.4% | 14467 | 13.4% | 14.4% |
| `bayes_measurement_walk_may` | 71.2% | 51.2% | 4055 | 14.1% | 14.7% |
| `bayes_walk_may` | 77.0% | 59.8% | 4055 | 10.9% | 12.1% |
| `contact_additive` | (no interval) | (no interval) | – | – | – |

## Prediction 1 — the channel loadings

| loading | cutoffs excluding zero | mean of means | min | max | holds |
|---|---:|---:|---:|---:|---|
| `lambda_barrel` | 45/48 | +0.560 | +0.019 | +0.690 | **no** |
| `lambda_ev` | 45/48 | +0.135 | -0.003 | +0.158 | **no** |
| `lambda_whiff` | 44/48 | +0.320 | +0.005 | +0.377 | **no** |

**Over the 43 converged fits only:** **HOLDS** — `lambda_barrel` 43/43, `lambda_ev` 43/43, `lambda_whiff` 43/43. The cutoffs whose interval spans zero are *exactly* the non-converged ones, so prediction 1's failure is a sampler artefact, not a statement about the channels.


## Prediction 5 — vacuity

| statistic | value | pre-registered | holds |
|---|---|---|---|
| `sigma_ability[hr_rate]` | mean 0.461, range [0.242, 0.545] | > 0.02 | yes |
| `sigma_ability[k_rate]` | mean 0.342, range [0.177, 0.409] | > 0.02 | yes |
| fits with \|loading corr\| ≥ 0.95 | 6 of 96 | 0 | **no** |

**Over the 43 converged fits only:** **HOLDS** — no fit left with |loading corr| >= 0.95.

