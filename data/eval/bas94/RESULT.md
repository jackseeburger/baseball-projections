# BAS-94 — the prior's mean as a function of the profile

Scope: seasons [2022, 2024, 2025, 2026], 48 cutoffs, components ['hr_rate', 'k_rate'], arms under test `bayes_walk+prior_contact` (A) and `bayes_walk+prior_contact_cur` (B); comparators `bayes_walk`, `contact_additive`, `marcel_tuned`.

**Verdict: prediction 1 holds and prediction 2 fails: the prior moves, and shrinking toward similar players does not help where it should. Nothing ships.**

## Predictions

| # | pre-registered | verdict |
| --- | --- | --- |
| 1 | the prior moves (gamma excludes zero at ≥90% of cutoffs; sd(prior mean) ≥ 30% of sigma_ability) | PASS |
| 2 | May: arm A beats bayes_walk by ≥3% on HR/PA (|t|>2.5) and ≥2% on K% | FAIL |
| 3 | pooled: ≤−1.5% HR/PA (t<−2), ≤−1% K% vs bayes_walk; within ±1.5% of contact_additive on HR/PA | FAIL |
| 4 | August gap ≤ half the May gap | FAIL |
| 5 | 80% interval covers 75–85%; arm A narrower than bayes_walk at May | hr_rate: PASS, k_rate: FAIL |

## hr_rate — pooled over every cutoff

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 14467 | +0.00012 | +1.14% | 1.49 | 3.49 | 18–30 |
| `bayes_walk+prior_contact` | `contact_additive` | 14467 | +0.00038 | +3.83% | 3.73 | 8.91 | 7–41 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 14467 | +0.00013 | +1.30% | 1.31 | 3.69 | 16–32 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 14467 | -0.00001 | -0.06% | -0.07 | -0.14 | 20–28 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 14467 | +0.00026 | +2.60% | 2.43 | 3.94 | 12–36 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 14467 | -0.00012 | -1.18% | -1.76 | -3.41 | 36–12 |
| `bayes_walk` | `contact_additive` | 14467 | +0.00026 | +2.66% | 2.59 | 4.18 | 13–35 |
| `contact_additive` | `marcel_tuned` | 14467 | -0.00025 | -2.44% | -2.22 | -4.17 | 35–13 |


### hr_rate — by season

**2022**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3808 | +0.00036 | +3.80% | 2.15 | 9.17 | 0–12 |
| `bayes_walk+prior_contact` | `contact_additive` | 3808 | +0.00017 | +1.72% | 0.70 | 1.68 | 5–7 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3808 | +0.00035 | +3.71% | 1.70 | 5.58 | 0–12 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3808 | +0.00010 | +1.09% | 0.55 | 2.17 | 1–11 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3808 | -0.00009 | -0.93% | -0.35 | -0.62 | 8–4 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3808 | +0.00009 | +1.00% | 0.47 | 3.18 | 2–10 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3808 | -0.00025 | -2.61% | -1.57 | -3.39 | 12–0 |


**2024**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3785 | -0.00007 | -0.63% | -0.40 | -2.03 | 8–4 |
| `bayes_walk+prior_contact` | `contact_additive` | 3785 | +0.00046 | +4.66% | 2.32 | 10.83 | 0–12 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3785 | -0.00005 | -0.49% | -0.27 | -1.80 | 9–3 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3785 | -0.00027 | -2.62% | -1.53 | -3.65 | 9–3 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3785 | +0.00025 | +2.56% | 1.19 | 6.21 | 0–12 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3785 | -0.00026 | -2.49% | -1.30 | -3.62 | 10–2 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3785 | -0.00021 | -2.01% | -1.51 | -3.63 | 11–1 |


**2025**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3612 | +0.00010 | +0.93% | 0.57 | 1.89 | 3–9 |
| `bayes_walk+prior_contact` | `contact_additive` | 3612 | +0.00039 | +3.85% | 1.94 | 13.62 | 0–12 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3612 | -0.00001 | -0.07% | -0.03 | -0.20 | 6–6 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3612 | +0.00003 | +0.31% | 0.19 | 0.84 | 3–9 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3612 | +0.00032 | +3.22% | 1.61 | 6.11 | 1–11 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3612 | -0.00007 | -0.68% | -0.34 | -1.52 | 7–5 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3612 | -0.00006 | -0.61% | -0.52 | -1.79 | 9–3 |


**2026**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3262 | +0.00005 | +0.49% | 0.36 | 0.95 | 7–5 |
| `bayes_walk+prior_contact` | `contact_additive` | 3262 | +0.00056 | +5.44% | 2.92 | 5.71 | 2–10 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3262 | +0.00026 | +2.43% | 1.49 | 5.19 | 1–11 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3262 | +0.00015 | +1.39% | 0.94 | 1.79 | 7–5 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3262 | +0.00065 | +6.39% | 3.20 | 5.15 | 3–9 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3262 | +0.00035 | +3.35% | 1.93 | 4.81 | 2–10 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3262 | +0.00010 | +0.89% | 0.64 | 2.67 | 4–8 |


## k_rate — pooled over every cutoff

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 14467 | +0.00165 | +5.69% | 7.39 | 9.89 | 0–48 |
| `bayes_walk+prior_contact` | `contact_additive` | 14467 | +0.00258 | +9.18% | 7.68 | 10.96 | 0–48 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 14467 | +0.00198 | +6.92% | 5.43 | 11.27 | 2–46 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 14467 | +0.00064 | +2.19% | 2.72 | 6.20 | 15–33 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 14467 | +0.00156 | +5.57% | 5.67 | 10.47 | 5–43 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 14467 | -0.00101 | -3.30% | -4.45 | -8.15 | 44–4 |
| `bayes_walk` | `contact_additive` | 14467 | +0.00093 | +3.31% | 4.21 | 10.83 | 3–45 |
| `contact_additive` | `marcel_tuned` | 14467 | -0.00059 | -2.07% | -3.95 | -7.26 | 39–9 |


### k_rate — by season

**2022**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3808 | +0.00247 | +8.55% | 4.77 | 7.21 | 0–12 |
| `bayes_walk+prior_contact` | `contact_additive` | 3808 | +0.00368 | +13.27% | 4.78 | 7.76 | 0–12 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3808 | +0.00255 | +8.82% | 3.11 | 7.30 | 0–12 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3808 | +0.00128 | +4.41% | 2.35 | 10.65 | 0–12 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3808 | +0.00248 | +8.96% | 3.85 | 9.86 | 0–12 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3808 | +0.00135 | +4.68% | 2.00 | 9.43 | 0–12 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3808 | -0.00120 | -3.81% | -2.47 | -5.07 | 12–0 |


**2024**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3785 | +0.00159 | +5.54% | 3.33 | 6.58 | 0–12 |
| `bayes_walk+prior_contact` | `contact_additive` | 3785 | +0.00228 | +8.15% | 3.27 | 6.79 | 0–12 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3785 | +0.00210 | +7.44% | 2.87 | 7.64 | 0–12 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3785 | +0.00064 | +2.24% | 1.59 | 3.37 | 4–8 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3785 | +0.00133 | +4.77% | 2.48 | 4.74 | 2–10 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3785 | +0.00115 | +4.08% | 2.03 | 5.33 | 2–10 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3785 | -0.00095 | -3.13% | -2.29 | -7.63 | 12–0 |


**2025**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3612 | +0.00089 | +3.03% | 2.30 | 4.40 | 0–12 |
| `bayes_walk+prior_contact` | `contact_additive` | 3612 | +0.00143 | +4.98% | 2.20 | 4.59 | 0–12 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3612 | +0.00100 | +3.43% | 1.44 | 4.82 | 2–10 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3612 | +0.00043 | +1.47% | 0.99 | 4.15 | 3–9 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3612 | +0.00097 | +3.38% | 1.96 | 6.06 | 3–9 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3612 | +0.00054 | +1.86% | 0.98 | 4.67 | 1–11 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3612 | -0.00046 | -1.52% | -1.00 | -1.80 | 8–4 |


**2026**

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk+prior_contact` | `bayes_walk` | 3262 | +0.00157 | +5.40% | 3.90 | 6.85 | 0–12 |
| `bayes_walk+prior_contact` | `contact_additive` | 3262 | +0.00292 | +10.52% | 4.18 | 8.76 | 0–12 |
| `bayes_walk+prior_contact` | `marcel_tuned` | 3262 | +0.00233 | +8.21% | 3.14 | 7.73 | 0–12 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3262 | +0.00002 | +0.07% | 0.05 | 0.14 | 8–4 |
| `bayes_walk+prior_contact_cur` | `contact_additive` | 3262 | +0.00137 | +4.93% | 2.47 | 8.44 | 0–12 |
| `bayes_walk+prior_contact_cur` | `marcel_tuned` | 3262 | +0.00077 | +2.74% | 1.33 | 6.26 | 0–12 |
| `bayes_walk+prior_contact_cur` | `bayes_walk+prior_contact` | 3262 | -0.00155 | -5.06% | -3.88 | -6.89 | 12–0 |


## The borrowed comparator's sampler

`bayes_walk` is the BAS-85 grid's own rows, drawn under **pymc**; the prior-mean arms here run under **numpyro**. One season ([2024]) of `bayes_walk` was refit under numpyro to say what that costs. It is not nothing:

- **hr_rate**: arm A's gap against the walk moves -0.85 points of MAE when the comparator is refit under numpyro.
- **k_rate**: arm A's gap against the walk moves -2.92 points of MAE when the comparator is refit under numpyro.

| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk_numpyro` | `bayes_walk` | 3785 | +0.00009 | +0.87% | 0.96 | 2.76 | 2–10 |
| `bayes_walk+prior_contact` | `bayes_walk` | 3785 | -0.00007 | -0.63% | -0.40 | -2.03 | 8–4 |
| `bayes_walk+prior_contact` | `bayes_walk_numpyro` | 3785 | -0.00016 | -1.48% | -1.02 | -7.85 | 11–1 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3785 | -0.00027 | -2.62% | -1.53 | -3.65 | 9–3 |
| `bayes_walk+prior_contact_cur` | `bayes_walk_numpyro` | 3785 | -0.00036 | -3.46% | -1.98 | -6.18 | 12–0 |


| arm | base | n | Δ MAE | % of base | t(player) | t(cell) | W–L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bayes_walk_numpyro` | `bayes_walk` | 3785 | +0.00082 | +2.84% | 3.19 | 17.09 | 0–12 |
| `bayes_walk+prior_contact` | `bayes_walk` | 3785 | +0.00159 | +5.54% | 3.33 | 6.58 | 0–12 |
| `bayes_walk+prior_contact` | `bayes_walk_numpyro` | 3785 | +0.00077 | +2.62% | 1.93 | 3.00 | 4–8 |
| `bayes_walk+prior_contact_cur` | `bayes_walk` | 3785 | +0.00064 | +2.24% | 1.59 | 3.37 | 4–8 |
| `bayes_walk+prior_contact_cur` | `bayes_walk_numpyro` | 3785 | -0.00017 | -0.59% | -0.41 | -0.90 | 8–4 |


## Fit diagnostics

```
{
 "bayes_walk+prior_contact": {
  "n_fits": 96,
  "max_rhat": {
   "median": 1.0253760823769011,
   "max": 1.05713885936231,
   "n_above_ceiling": 0
  },
  "min_ess_bulk": {
   "median": 238.2544918302413,
   "min": 71.44116324856981
  },
  "divergences": {
   "total": 0,
   "max": 0,
   "n_fits_with_any": 0
  },
  "elapsed_s": {
   "median": 159.89999999999998,
   "min": 96.0,
   "max": 473.5,
   "total_hours": 4.55
  },
  "samplers": [
   "numpyro"
  ]
 },
 "bayes_walk+prior_contact_cur": {
  "n_fits": 96,
  "max_rhat": {
   "median": 1.0264686506726348,
   "max": 1.097451742136189,
   "n_above_ceiling": 0
  },
  "min_ess_bulk": {
   "median": 155.92630519245478,
   "min": 15.987645564155999
  },
  "divergences": {
   "total": 0,
   "max": 0,
   "n_fits_with_any": 0
  },
  "elapsed_s": {
   "median": 158.2,
   "min": 92.0,
   "max": 466.0,
   "total_hours": 4.4447222222222225
  },
  "samplers": [
   "numpyro"
  ]
 }
}
```

No fit exceeded the R-hat ceiling of 1.2.
