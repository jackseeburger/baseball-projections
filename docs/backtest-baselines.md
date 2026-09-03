# Baseline Backtest Scores (roadmap 0.3, baseline half)

Four parts: **season-level** splits (below), the **intra-season
walk-forward** at date cutoffs ([jump](#intra-season-walk-forward--rest-of-2026-rates)),
which is the one that judges rest-of-season projections, **tuning
Marcel's own constants**
([jump](#tuning-marcel--fitted-constants-beat-tangos-defaults)), which moves
this bar up, and the **constrained refit of the age curve**
([jump](#the-age-curve-was-not-aging--a-constrained-refit-and-a-projected-league-rate)),
which is the fit currently frozen in `src/eval/marcel_params.json`.

**Run:** Sept 1, 2026 · **Data:** MLB Stats API season hitting totals 2015–2026
(`data/parquet/hitter_seasons_api.parquet`, MLBAM-keyed, rebuildable via
`src/data/mlb_stats_api.py`) · **Method:** `scripts/run_backtest.py --sweep`,
six splits (train ≤ 2019→2020 … train ≤ 2024→2025), scored on players with
≥ 100 trials in the predict year, trials-weighted.

**These are the numbers every Bayesian component must beat.** A model change
that doesn't improve on Marcel here is a regression regardless of how
principled it looks. Log loss is per-trial binomial NLL; MAE/RMSE are on rates.

## Mean across the six splits

| Component | Model | Log loss | MAE | RMSE |
|---|---|---|---|---|
| **k_rate** | marcel | **0.5219** | **0.0305** | **0.0389** |
| | previous_season | 0.5261 | 0.0339 | 0.0465 |
| | league_average | 0.5283 | 0.0494 | 0.0615 |
| **bb_rate** | marcel | **0.2918** | **0.0174** | **0.0223** |
| | previous_season | 0.2999 | 0.0210 | 0.0284 |
| | league_average | 0.2950 | 0.0248 | 0.0314 |
| **hr_rate** | marcel | **0.1428** | **0.0105** | **0.0132** |
| | previous_season | 0.1554 | 0.0125 | 0.0164 |
| | league_average | 0.1443 | 0.0130 | 0.0163 |
| **iso** | marcel | — | **0.0379** | **0.0474** |
| | previous_season | — | 0.0458 | 0.0597 |
| | league_average | — | 0.0462 | 0.0581 |
| **babip** | marcel | **0.6067** | **0.0273** | **0.0347** |
| | league_average | 0.6069 | 0.0289 | 0.0365 |
| | previous_season | 0.6373 | 0.0375 | 0.0539 |

## Sanity checks that came out right

- **Marcel wins every component on every metric** — the 5/4/3 + regression +
  age structure earns its keep.
- **BABIP is barely skill:** league average nearly ties Marcel and
  previous-season is far worse — the classic "BABIP is mostly luck" result.
  A Bayesian BABIP model has very little room to add value; K% has the most.
- **Predictability ordering** (player-specific edge over league average):
  K% > BB% > ISO > HR rate > BABIP — matches published year-to-year
  reliability orderings.
- Marcel's decile calibration tracks the diagonal with |gap| < 0.016 on K%.

## Caveats

- Season totals from the Stats API, not the Statcast PA-level slice the
  Bayesian components train on — when re-scoring components (0.3 proper),
  regenerate realized outcomes from the same PA universe for a fair fight.
- 2020 (60 games) thins the ≥100-trials pool for the 2019→2020 split.
- Ages here are the Stats API's seasonal age; components use Chadwick
  birthdates (roadmap 0.1) — identical convention (June 30).

---

# Intra-season walk-forward — rest-of-2026 rates

**Run:** Sept 2, 2026 · **Data:** PA-level 2026 outcomes
(`pa_outcomes/pa_outcomes_2026.parquet` in R2, 157,749 PA through Sept 1)
aggregated to the season schema either side of each cutoff, plus 2015–2025
season totals from `data/parquet/hitter_seasons_api.parquet` ·
**Method:** `python scripts/run_intraseason_backtest.py`, three cutoffs,
scored on players with ≥ 100 realized trials *after* the cutoff,
trials-weighted, common players across all six arms.

The season-level table above answers "given everything through 2025, how good
is 2026?". The product is a **rest-of-season** projection, so the question that
matters is "given everything through July 1, how good is the rest of 2026?"
Training is the prior full seasons plus the current season **through the
cutoff**; the realized side is every PA **on or after** it. The leakage guard
(`assert_split_clean`) rejects any training PA dated on/after the cutoff and
any realized PA before it.

## The arms

| Arm | Sees 2026? | What it is |
|---|---|---|
| `marcel` | **yes** | 5/4/3 with the partial 2026 as the most recent year. Marcel weights by trials, so a 300-PA partial season scales itself down automatically. |
| `marcel_preseason` | no | The same Marcel with the partial season withheld — the control. `marcel` − `marcel_preseason` **is** the value of in-season information, with the model held fixed. |
| `season_to_date` | **yes** | The player's own 2026 rate regressed to league with the component's stabilization-point ballast (K% 60 PA, BB% 120, HR 170, ISO 160 AB, BABIP 820 BIP). The "just use this year" arm. |
| `bayes_preseason` | no | Our Bayesian components (`data/projections/*_projections_2026.parquet`), fit through 2025 and generated Apr 10 — the same file at every cutoff. |
| `previous_season` | no | 2025 rate, unregressed. |
| `league_average` | **yes** | League rate through the cutoff. |

## MAE by component and cutoff (bold = best arm at that cutoff)

| Component | Arm | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|---|
| **k_rate** | marcel | **.0278** | **.0296** | **.0343** |
| | marcel_preseason | .0293 | .0310 | .0365 |
| | season_to_date | .0355 | .0340 | .0371 |
| | bayes_preseason | .0296 | .0325 | .0386 |
| | previous_season | .0319 | .0342 | .0363 |
| | league_average | .0507 | .0518 | .0588 |
| **bb_rate** | marcel | **.0172** | **.0206** | **.0268** |
| | marcel_preseason | .0179 | .0218 | .0275 |
| | season_to_date | .0224 | .0225 | .0280 |
| | bayes_preseason | .0179 | .0222 | .0279 |
| | previous_season | .0210 | .0246 | .0286 |
| | league_average | .0254 | .0263 | .0321 |
| **hr_rate** | marcel | **.0098** | **.0115** | **.0152** |
| | marcel_preseason | .0098 | .0118 | .0155 |
| | season_to_date | .0115 | .0123 | .0156 |
| | bayes_preseason | .0099 | .0117 | .0153 |
| | previous_season | .0121 | .0141 | .0167 |
| | league_average | .0128 | .0135 | .0161 |
| **iso** | marcel | **.0345** | **.0413** | .0567 |
| | marcel_preseason | .0352 | .0428 | .0558 |
| | season_to_date | .0410 | .0435 | .0594 |
| | bayes_preseason | .0358 | .0436 | **.0551** |
| | previous_season | .0418 | .0493 | .0584 |
| | league_average | .0448 | .0472 | .0609 |
| **babip** | marcel | **.0269** | .0350 | .0424 |
| | marcel_preseason | .0270 | .0349 | **.0393** |
| | season_to_date | .0286 | .0352 | .0461 |
| | bayes_preseason | .0271 | **.0345** | .0417 |
| | previous_season | .0344 | .0389 | .0411 |
| | league_average | .0288 | .0352 | .0427 |

Scored players (common across arms): K%/BB%/HR 315 · 231 · 126;
ISO 308 · 213 · 80; BABIP 267 · 157 · **4**.

Log loss orders the arms the same way wherever it applies; the spreads are
tiny (K% at Jul 1: marcel .50704, marcel_preseason .50750, bayes .50784,
league average .51587). Marcel-with-partial stays calibrated — decile gaps on
K% at Jul 1 are within ±.018 and unsigned.

## The in-season increment: `marcel` vs `marcel_preseason`

Change in MAE from adding the partial season to Marcel (negative = better):

| Component | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|
| k_rate | **−5.1%** | **−4.6%** | **−6.1%** |
| bb_rate | **−3.9%** | **−5.6%** | **−2.7%** |
| hr_rate | −0.0% | **−2.3%** | **−2.3%** |
| iso | **−2.0%** | **−3.5%** | +1.6% |
| babip | −0.4% | +0.4% | +7.8% *(n=4)* |

## Reading it

**In-season data helps, it helps most where skill is most real, and it is
worth more than our model is.** Folding the current season into Marcel cuts
rest-of-season K% error by 4.6–6.1% and BB% error by 2.7–5.6% at every
horizon; ISO gains 2.0–3.5% through July, HR/PA gains 2.3% from July onward
and nothing in April; BABIP gains nothing at any cutoff, the same "BABIP is mostly
luck" result the season-level table shows, now confirmed within a season. The
ordering of the increment — K% > BB% > ISO ≈ HR > BABIP — is the reliability
ordering, as it should be: in-season PA only carry information about the
components that carry information at all. Two things follow. First, **the
increment is not small in context**: 5% of K% MAE is about half the entire
preseason spread between Marcel and Depth Charts, the best public system
([accuracy-2026.md](accuracy-2026.md)), and we get it for free by not
throwing away April. Second, **it is bigger than the gap this table reads as our model's edge**:
`marcel` beats `bayes_preseason` on K% by 6% at May 1 and 11% at Aug 1 — but
`bayes_preseason` is a fixed preseason file and `marcel` is not, so that gap is
mostly the information difference and not a model difference. **Refitting the
Bayesian model at the cutoff removes essentially all of it**: see
[the fair fight](#the-fair-fight--the-bayesian-arm-refit-at-the-cutoff-bas-59)
(Sept 3), where the refit arm gains the same 6–11% over its own withheld self
and lands statistically level with tuned Marcel. The route to a better
rest-of-season number still runs through *ingesting the current season* rather
than through a fancier prior — that conclusion survives — but the evidence for
it is the in-season increment itself, not a Bayes-versus-Marcel gap that was
never measured on equal footing. It also matters **how** you ingest it:
`season_to_date` — this year's rate regressed with the right ballast — loses
to Marcel-with-partial at every cutoff and every component, though its
disadvantage on K% shrinks from +21% at May 1 to +1.5% at Aug 1 as the sample
stabilizes. The value is in *adding* the partial season to the multi-year
prior, not in replacing it.

## Caveats

- **MAE rises across cutoffs for every arm.** That is the realized window
  shrinking (a shorter rest-of-season is a noisier target), not the models
  degrading. Only the within-cutoff comparison between arms is meaningful.
- **BABIP at Aug 1 is 4 players.** `min_trials=100` on BIP is brutal at a
  one-month horizon; treat that column as absent, and the Jul 1 BABIP column
  (n=157) as thin.
- **The 2026 season here ends Sept 1**, the last date in the PA parquet, so
  "rest of season" at the Aug 1 cutoff is one month, not two.
- **`bayes_preseason` is a fixed file, not a refit.** It is our Apr 10
  projection for 2026 (`projection_year == 2026`), scored unchanged at all
  three cutoffs — deliberately, since it is the no-in-season-information arm.
  It is *not* an answer to "how good would our model be if refit on July 1".
  That question was finally answered on Sept 3 —
  [the fair fight](#the-fair-fight--the-bayesian-arm-refit-at-the-cutoff-bas-59)
  — and the answer moves the arm from 2.6–2.7 standard errors behind the live
  one to inside noise at every cutoff.
- **PA-derived seasons are not identical to the Stats API totals.** The
  Statcast universe runs ~0.7% more PA per player (mean +1.8 PA over 2026;
  AB, K, BB, HR all within a few counts). Training mixes the two — prior
  seasons from the API table, the partial current season from PA data — so a
  hair of the in-season increment could be universe drift rather than signal.
  Rebuilding prior seasons from `pa_outcomes_<year>.parquet` would close it.
  Only 2026 exists in R2, but 2024 and 2025 can now be rebuilt locally from the
  Statcast parquets — see
  [the fair fight](#the-fair-fight--the-bayesian-arm-refit-at-the-cutoff-bas-59),
  which does exactly that so both arms read the same three seasons.
- Components are derived from the PA outcome flags with the standard
  identities (AB = PA − BB − HBP − SF − SH − interference; BIP = AB − K − HR
  + SF; xb_points = 2B + 2·3B + 3·HR). All five components are fully
  derivable; nothing was dropped.

---

# Tuning Marcel — fitted constants beat Tango's defaults

> **Superseded in part by [the Sept 3 section below](#the-age-curve-was-not-aging--a-constrained-refit-and-a-projected-league-rate).**
> Everything about the method still stands; the constants in "The chosen
> constants" and every number scored against them are the *previous* frozen
> fit, kept here for the record. `src/eval/marcel_params.json` now holds the
> constrained refit.

**Run:** Sept 2, 2026 · **Code:** `src/eval/baselines.marcel_tuned`,
`src/eval/tuning.py`, `scripts/tune_marcel.py` · **Params:**
`src/eval/marcel_params.json` (committed) · **Data:** the same two sources as
above — `data/parquet/hitter_seasons_api.parquet` for seasons and
`pa_outcomes_2026.parquet` for the dated cutoffs.

Marcel-with-partial is the production rest-of-season projection and the bar
every station A model has to clear, and its constants have never been fit to
anything: 5/4/3 recency, one 200-trial ballast for all five components, and
one age curve that does not know whether a component stabilizes in 60 PA or
820 BIP. The pitcher work found that per-component ballasts near the
stabilization points matter a great deal. This is that discipline applied to
the hitter baseline.

`marcel_tuned` is the *same estimator* with three constants pulled out per
component — the ballast (in the component's trials unit), the recency weights
(only their ratios matter), and a three-number age curve (a peak age and a
signed per-year slope either side). At stock values it reproduces `marcel`
bit for bit on every component and every split, which is the test that makes
everything below a measurement of parameters rather than of a refactor.

## Method

- **Fit:** walk-forward on **2020–2024 only**. For each predict year Y, train
  on seasons ≤ Y−1 and score Y; the objective is the **mean trials-weighted
  MAE** across the five years (log loss is carried alongside and orders the
  arms the same way). Coordinate search over five axes — ballast, the two
  weight ratios jointly, peak age, and the two slopes — 89 grid points per
  pass, up to three passes, ties keeping the incumbent so a component where
  tuning does nothing comes back holding stock's numbers.
- **Grid:** ballast 25–2600 in 13 steps; weight ratios w2/w1 and w3/w1 each
  ∈ {0, .2, .4, .6, .8, 1} (stock's 5/4/3 is exactly (0.8, 0.6)); peak age
  23–32; slopes ±0.012 in 15 steps. Deliberately coarse — the objective is
  flat, and a finer grid buys fractions of a percent *in sample*, which is
  exactly the kind of gain that does not survive a holdout.
- **Guard:** an **inner validation** inside the tuning window — fit on
  2020–2022, score 2023–2024 — decides whether a component's fit generalises
  at all. One that does not beat stock there keeps stock's constants. BB% is
  the only component that trips it. This is a rule about the fit, not about
  the holdout: the guard was added after the holdout showed BB% regressing,
  and the point is that the inner validation had *already* said so (+0.13%),
  so the decision is reproducible without spending holdout data.
- **Freeze, then score:** the params are written to
  `src/eval/marcel_params.json` and scored, unchanged, on data the search
  never saw — season-level **2025** (train ≤ 2024) and **2026** (train ≤
  2025), and the three **2026 intra-season cutoffs** through the same
  `cutoff_date` path the arms above use (`marcel_tuned` reads a partial
  season exactly as `marcel` does: it is the most recent season, and the
  estimator's trials weighting scales it down on its own).
- **Ages are Chadwick** (`src/data/birthdates.py`, age as of June 30 from
  real birthdates), not the `debut − 23` proxy. The register covers
  **100.0%** of the 12,069 batter-seasons in the table, and floor(Chadwick
  age) equals the Stats API's seasonal age on **100.0%** of them — the two
  sources agree exactly under the June 30 convention, so nothing in this
  section turns on which one is used. Both `marcel` and `marcel_tuned` floor
  the age before applying the curve, so the comparison isolates parameters.

## The chosen constants

| Component | Ballast (trials) | Weights (w1, w2, w3) | Peak age | Slope young | Slope old |
|---|---|---|---|---|---|
| k_rate | **100** PA | 1 / **0.4** / **0.2** | 31 | +0.006 | +0.012 |
| bb_rate | 200 PA *(stock)* | 5 / 4 / 3 *(stock)* | 27 | −0.001 | −0.003 |
| hr_rate | 200 PA | 1 / **0.6** / **0.6** | 23 | −0.012 | −0.012 |
| babip | **600** BIP | 1 / 0.8 / **1.0** | 26 | −0.012 | −0.003 |
| iso | **300** AB | 1 / 0.8 / 0.6 | 23 | −0.012 | −0.012 |

Stock is ballast 200, weights 5/4/3 ≡ 1 / 0.8 / 0.6, peak 27, and slopes
(+0.002, +0.005) for K%, (−0.001, −0.003) for BB%/ISO/BABIP and (0, 0) for
HR/PA — stock Marcel ages HR/PA not at all. The ballast is quoted at the
average year weight; normalize the weights to mean 1 and it is plain trials.

Two of these are real content and read the way the reliability literature
says they should. **K% wants half stock's ballast and much sharper recency**
(1 / 0.4 / 0.2 against 1 / 0.8 / 0.6) — it is the most reliable component, so
a player's own recent sample deserves more of the weight and league average
less of it. **BABIP wants three times stock's ballast and essentially flat
recency** (1 / 0.8 / 1.0) — it is the least reliable, so pull hard toward
league and treat all three years alike, because recency in a mostly-luck
signal is just noise. That is the same ordering the season-level table above
found, now expressed in the constants instead of in the scores.

## In sample (2020–2024, the tuning window)

| Component | Stock MAE | Tuned MAE | Gain | Gain from ballast+weights alone | Inner validation (2023–24) |
|---|---|---|---|---|---|
| k_rate | .030657 | **.029708** | **−3.10%** | −1.79% | −4.10% |
| bb_rate | .017569 | .017476 | −0.53% | −0.09% | **+0.13% → kept stock** |
| hr_rate | .010679 | **.010288** | **−3.66%** | −0.33% | −1.39% |
| babip | .027762 | **.026851** | **−3.28%** | −3.08% | −4.17% |
| iso | .038630 | **.037167** | **−3.79%** | −0.63% | −0.22% |

The BB% row is the fit the guard threw away: the search does find a −0.53%
in-sample improvement there, and the inner validation says it will not travel,
so the frozen file carries stock's constants for BB% and the holdout confirms
the call (the unguarded BB% fit is +0.86%, t +2.5, i.e. reliably *worse* out
of sample).

Log loss moves the same way where it applies (K% −0.03%, HR −0.05%, BABIP
−0.03%), which is a weak but consistent confirmation that the MAE gain is not
an artifact of the loss function.

The fourth column is the tell. For BABIP the whole gain is the ballast and
the weights; for HR/PA and ISO almost none of it is, and the rest comes from
the age curve — which the search pushes to a **peak age at an end of its
grid with both slopes equal**, i.e. a straight line in age. That is only
half an aging curve. The other half is a *level* correction: Marcel regresses
to the last training season's league rate while a player's own weighted
history spans three, so a league trending up or down leaves a bias that a
linear age term can absorb. The holdout is where that distinction gets
settled.

## Out of sample — MAE by arm and cell

Never seen by the search. Season-level 2025 and 2026 (train ≤ Y−1); the three
2026 cutoffs scored on PA on or after the cutoff, ≥ 100 realized trials,
common players across all arms at each cell. `marcel_tuned_noage` is the
ballast-and-weights-only fit, carried as a diagnostic.

| Component | Arm | 2025 | 2026 | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|---|---|---|
| **k_rate** | marcel_tuned | **.0292** | **.0269** | **.0267** | **.0292** | .0346 |
| | marcel_tuned_noage | .0300 | .0272 | .0269 | .0292 | **.0343** |
| | marcel | .0299 | .0278 | .0278 | .0296 | .0343 |
| | marcel_preseason | — | — | .0293 | .0310 | .0365 |
| | season_to_date | — | — | .0355 | .0340 | .0371 |
| | bayes_preseason | — | .0284 | .0296 | .0325 | .0386 |
| | previous_season | .0351 | .0307 | .0319 | .0342 | .0363 |
| | league_average | .0479 | .0496 | .0507 | .0518 | .0588 |
| **bb_rate** | marcel_tuned *(= marcel)* | **.0163** | .0182 | **.0172** | .0206 | **.0268** |
| | marcel_tuned_noage | .0164 | .0183 | .0172 | **.0205** | .0269 |
| | marcel_preseason | — | — | .0179 | .0218 | .0275 |
| | season_to_date | — | — | .0224 | .0225 | .0280 |
| | bayes_preseason | — | **.0179** | .0179 | .0222 | .0279 |
| | previous_season | .0202 | .0220 | .0210 | .0246 | .0286 |
| | league_average | .0244 | .0251 | .0254 | .0263 | .0321 |
| **hr_rate** | marcel_tuned | .0098 | **.0089** | .0101 | **.0114** | **.0148** |
| | marcel_tuned_noage | **.0097** | .0091 | **.0098** | .0115 | .0150 |
| | marcel | .0098 | .0091 | .0098 | .0115 | .0152 |
| | marcel_preseason | — | — | .0098 | .0118 | .0155 |
| | season_to_date | — | — | .0115 | .0123 | .0156 |
| | bayes_preseason | — | .0092 | .0099 | .0117 | .0153 |
| | previous_season | .0114 | .0113 | .0121 | .0141 | .0167 |
| | league_average | .0122 | .0117 | .0128 | .0135 | .0161 |
| **babip** | marcel_tuned | **.0235** | **.0235** | .0267 | **.0340** | .0415 |
| | marcel_tuned_noage | .0236 | .0235 | **.0267** | .0343 | .0426 |
| | marcel | .0249 | .0242 | .0269 | .0350 | .0424 |
| | marcel_preseason | — | — | .0270 | .0349 | **.0393** |
| | season_to_date | — | — | .0286 | .0352 | .0461 |
| | bayes_preseason | — | .0243 | .0271 | .0345 | .0417 |
| | previous_season | .0337 | .0329 | .0344 | .0389 | .0411 |
| | league_average | .0253 | .0254 | .0288 | .0352 | .0427 |
| **iso** | marcel_tuned | .0342 | **.0321** | .0361 | .0412 | **.0550** |
| | marcel_tuned_noage | **.0335** | .0326 | .0347 | **.0409** | .0563 |
| | marcel | .0339 | .0333 | **.0345** | .0413 | .0567 |
| | marcel_preseason | — | — | .0352 | .0428 | .0558 |
| | season_to_date | — | — | .0410 | .0435 | .0594 |
| | bayes_preseason | — | .0341 | .0358 | .0436 | .0551 |
| | previous_season | .0405 | .0404 | .0418 | .0493 | .0584 |
| | league_average | .0430 | .0406 | .0448 | .0472 | .0609 |

Scored players: 407 (2025) · 362 (2026) · 315 / 231 / 126 at the three
cutoffs for K%/BB%/HR; 390 · 356 · 308 / 213 / 80 for ISO; 347 · 319 ·
267 / 157 / **4** for BABIP. `marcel_preseason` is only defined where a
partial season exists, and `bayes_preseason` is the fixed Apr 10 2026 file,
so both are blank in the 2025 column.

## The paired test

Per-player difference in absolute error, `marcel_tuned` − `marcel`, on the
players both arms cover — a within-player comparison, so its SE is not
inflated by the spread of player skill. Trials-weighted; **negative = tuned
better**. BB% is stock, hence identically zero.

| Component | 2025 | 2026 | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|---|---|
| k_rate | **−.00075 ± .00057** | **−.00087 ± .00053** | **−.00107 ± .00050** | **−.00042 ± .00060** | +.00029 ± .00077 |
| bb_rate | .00000 | .00000 | .00000 | .00000 | .00000 |
| hr_rate | +.00001 ± .00016 | **−.00021 ± .00015** | +.00027 ± .00017 | **−.00008 ± .00019** | **−.00034 ± .00027** |
| babip | **−.00142 ± .00043** | **−.00076 ± .00044** | **−.00018 ± .00046** | **−.00107 ± .00053** | **−.00090 ± .00210** |
| iso | +.00029 ± .00071 | **−.00126 ± .00073** | +.00159 ± .00076 | **−.00005 ± .00088** | **−.00167 ± .00136** |

Pooled within each component across its five cells:

| Component | Pooled difference | SE | t | % of stock MAE | Cells won |
|---|---|---|---|---|---|
| babip | −.000875 | .000230 | **−3.80** | **−3.27%** | 5/5 |
| k_rate | −.000706 | .000263 | **−2.68** | **−2.41%** | 4/5 |
| hr_rate | −.000031 | .000079 | −0.40 | −0.30% | 3/5 |
| iso | +.000008 | .000367 | +0.02 | +0.02% | 3/5 |
| bb_rate | 0 | — | — | 0 | 0/5 *(identical)* |

**Overall: −1.10% of stock Marcel's MAE (SE 0.36, t −3.06)**, pooling every
cell's difference as a fraction of that cell's own stock MAE so ISO's scale
does not swamp K%'s.

## Verdict — the gate clears

> **Gate:** beat stock Marcel-with-partial out of sample on the majority of
> component × cutoff cells with the pooled paired difference below zero.

`marcel_tuned` wins **15 of 25** cells and the pooled difference is
**−1.10% ± 0.36**, so the gate clears — but read where the gain lives before
believing more of it than it says. **It is BABIP and K%, and nothing else.**
BABIP improves 3.3% and wins all five cells, K% improves 2.4% and wins four;
HR/PA and ISO are inside a tenth of a standard error of zero, and BB% is
stock by construction. Both winners are exactly the components whose fitted
constants had a story attached — the ballast moving toward the stabilization
point in both directions at once, down for the reliable component and up for
the unreliable one.

The result also survives its obvious robustness checks. Refitting with **2020
excluded** (the 60-game season, and the one most likely to be driving a
ballast estimate somewhere strange) moves the constants barely — K% ballast
75 instead of 100 with the same 1 / 0.4 / 0.2 weights, BABIP's ballast
unchanged at 600 — the guard still fires on BB% and nothing else, and the
holdout verdict is the same to two decimal places: 15/25 cells, pooled
**−1.11% ± 0.38**. The
ballast-and-weights-only arm, which cannot exploit the level-correction
degeneracy at all, clears the gate too and does so more evenly: **18 of 25
cells, −1.01% ± 0.23 (t −4.45)**, better than stock on four components and
never significantly worse on any.

**Consequence for production.** `src/projections/ros.py` should switch
engines. It is one line: in `marcel_rates`, the arm tuple

```python
for arm, provider in (("marcel", baselines.marcel),
                      ("marcel_preseason", baselines.marcel_preseason)):
```

becomes `baselines.marcel_tuned` (and, if the control arm should move with
it, `baselines.marcel_tuned_preseason`). Nothing else changes: same provider
signature, same partial-season handling, params read from
`src/eval/marcel_params.json` with a stock fallback if the file is ever
missing. `ros.py` is not edited here — it is in a PR in flight.

## Caveats

- **The age curve is the weak part of the fit, and probably not aging.** Its
  in-sample gains on HR/PA (−3.66% vs −0.33% without it) and ISO (−3.79% vs
  −0.63%) are the largest in the table and they are the ones that vanish on
  the holdout, where both components land on zero. The search puts the peak
  at a grid end with equal slopes on both sides, which is a straight line in
  age — part aging, part level correction for regressing to one season's
  league rate with three seasons of history. Fixing that properly means
  projecting the league rate forward rather than dressing it as aging, and
  then refitting the age term against the corrected baseline. Until then,
  treat the age numbers in `marcel_params.json` as a fitted nuisance
  parameter, not as an aging curve anyone should quote.
- **The grid is coarse and the optimum is flat.** Ballast in 13 steps, two
  weight ratios in 6 each, slopes in 15. Several parameters sit on a grid
  edge (peak age 23 and 31, slopes at ±0.012), which is a sign the age axes
  are being used for something they are not shaped for, not that a wider grid
  would find more.
- **Tuning did nothing for BB%**, on both the inner validation and the
  holdout, and it was the one component where the search's in-sample gain was
  smallest to begin with (−0.53%). BB% keeps Tango's constants.
- **The five holdout cells are not independent.** The three cutoffs are
  nested windows of the same 2026 season and the 2026 season cell contains
  all of them, so the pooled SEs are optimistic; the per-cell SEs are the
  honest ones. The genuinely independent replication is 2025 vs 2026, and
  BABIP (−.00142 and −.00076) and K% (−.00075 and −.00087) both replicate
  across it.
- **BABIP at Aug 1 is still 4 players** and ISO at Aug 1 is 80; those two
  cells carry almost no weight in the pooled numbers and should not be read
  on their own.
- **A tuned Marcel is a better baseline, not a model.** It moves the bar the
  Bayesian components have to clear *up* by 2–3% on the two components where
  they were closest, which makes station A's problem harder, not easier.

# The age curve was not aging — a constrained refit, and a projected league rate

**Run:** Sept 3, 2026 · **Code:** `src/eval/baselines.projected_league_rate`,
`src/eval/tuning.py` (`AGE_PEAK_WINDOW`, `constrain`, `age_curve_ok`),
`scripts/tune_marcel.py` · **Params:** `src/eval/marcel_params.json`
(refrozen) · **Data:** unchanged — `data/parquet/hitter_seasons_api.parquet`
and `pa_outcomes_2026.parquet`, same holdout, same paired machinery.

The section above ended with a caveat it could not act on: the fitted age
curve was not an aging curve. The search put its peak at an end of the grid
(23 or 31) with **equal slopes on both sides**, which is a straight line in
age — and half of a straight line in age is a *level* correction, because
Marcel regresses three seasons of a player's history toward **one** season's
league rate. Its in-sample gains on HR/PA (−3.66%) and ISO (−3.79%) were the
largest in the table and were exactly the two that vanished out of sample. On
the live board the same thing showed as a level: mean projected K% .2133
against a 2026 league rate of .2207, where stock Marcel sat on league.

Two changes, meant to separate aging from bookkeeping.

## 1. Regress toward a projected league rate

`marcel_tuned` no longer hard-codes "the last training season's rate".
`MarcelParams.league_mode` picks one of three, per component:

| Option | What it regresses toward |
|---|---|
| `last` | the last training season's rate — stock Marcel, and still the default |
| `weighted3` | the same three seasons under the component's **own recency weights**, so the thing being regressed toward is measured over the same window as the thing being regressed |
| `drift` | `r_last + damp · (predict_year − last) · (r_last − r_prev)`, the one-season change extrapolated and damped by a constant (`league_damp` ∈ {.25, .5, .75, 1}) |

The horizon scaling matters at a cutoff: there the most recent *training*
season is the target season, so the horizon is zero and `drift` collapses
back to `last` — the partial season's own rate is already a same-season
measurement and there is nothing to extrapolate.

**The choice is made on the inner validation, never the holdout.** For each
component and each option, the same coordinate search runs on 2020–2022 with
the option pinned (so the other five axes adapt to it) and the fitted params
are scored on 2023–2024. Percent of stock's validation MAE, negative better:

| Component | `last` | `weighted3` | `drift@.25` | `drift@.5` | `drift@.75` | `drift@1` |
|---|---|---|---|---|---|---|
| k_rate | **−2.47** | −2.44 | −2.43 | −2.39 | −2.34 | −2.29 |
| bb_rate | **+0.19** | +0.37 | +0.27 | +0.36 | +0.47 | +0.60 |
| hr_rate | **−1.25** | −1.15 | −0.80 | −0.31 | +0.22 | +0.80 |
| babip | **−3.78** | −3.25 | −3.74 | −3.66 | −3.56 | −3.45 |
| iso | −0.22 | **−0.95** | +0.43 | +1.14 | +1.94 | +2.77 |
| **mean** | **−1.51** | −1.48 | −1.25 | −0.97 | −0.65 | −0.31 |

**`last` wins.** It is best on four of five components and on the mean; only
ISO takes `weighted3`, and it takes it by a real margin (−0.95% against
−0.22%). Every drift option is worse than doing nothing, monotonically worse
the harder it is pushed. So the frozen file carries `last` for K%, BB%, HR/PA
and BABIP, and `weighted3` for ISO.

Note that the in-sample optimum disagrees: search the league axis on the full
2020–2024 window and it picks `weighted3` for four of five components. That
is the signature of an axis a search cannot pick honestly for itself, which
is why `tune` pins it from the inner validation instead of sweeping it.

### The level, which is what the option was for

Trials-weighted mean projection minus the realized rate of the same scored
players, averaged over the five tuning years, with stock's other constants
held fixed (`--league-modes` prints all of it per season):

| Component | `last` | `weighted3` | `drift@.25` | `drift@.5` | `drift@.75` | `drift@1` |
|---|---|---|---|---|---|---|
| k_rate | +.00219 | +.00183 | +.00224 | +.00228 | +.00232 | +.00236 |
| bb_rate | +.00045 | +.00026 | +.00047 | +.00049 | +.00051 | +.00053 |
| hr_rate | +.00201 | +.00200 | +.00201 | +.00201 | +.00200 | +.00200 |
| babip | +.00133 | +.00157 | +.00132 | +.00130 | +.00129 | +.00128 |
| iso | +.00838 | +.00854 | +.00834 | +.00829 | +.00825 | +.00821 |

Mean projected K% by season, all six options, against the season's realized
league K%:

| Predict year | `last` | `weighted3` | `drift@.5` | `drift@1` | Realized league |
|---|---|---|---|---|---|
| 2020 | .21868 | .21763 | .21933 | .21999 | .23435 |
| 2021 | .22694 | .22523 | .22760 | .22827 | .23180 |
| 2022 | .22559 | .22552 | .22529 | .22498 | .22418 |
| 2023 | .22281 | .22382 | .22187 | .22093 | .22728 |
| 2024 | .22575 | .22577 | .22610 | .22644 | .22580 |

**There was no lag to correct.** The options differ in the fourth decimal;
against the scored population Marcel already sits within a couple of tenths
of a point of league, and the year it misses badly (2020) is the 60-game
season, which no amount of drift damping reaches. The live board's K% gap
was never the league rate — **it was the age curve**, which is the second
change.

## 2. Constrain the age term so it cannot act as a level

Two rules, in `src/eval/tuning.py`:

1. the peak lives in **25–31**, a window an aging curve can plausibly peak in;
2. the two slopes take **opposite signs**, so the multiplier turns over at the
   peak instead of running monotonically across the whole age range.

Rule 2 is signed per component by which way *performance* runs
(`AGE_DIRECTION`). For the four components where a bigger number is a better
hitter the multiplier rises to the peak and falls after it; **K% is
mirrored** — a trough at the peak age, rising after — because there a bigger
number is worse. This deliberately excludes stock Marcel's own age term,
which is monotone in age with a kink at 27, since a monotone line is exactly
the shape that doubles as a level. `constrain()` projects any start point
into the family (peak clipped, wrong-signed slope zeroed), so a constrained
search cannot land outside it, from any start — that is a test.

## The chosen constants

| Component | Ballast (trials) | Weights (w1, w2, w3) | Peak age | Slope young | Slope old | League rate |
|---|---|---|---|---|---|---|
| k_rate | **100** PA | 1 / **0.4** / **0.4** | **30** | 0 | **+0.008** | last |
| bb_rate | 200 PA *(stock)* | 5 / 4 / 3 *(stock)* | 27 | −0.001 | −0.003 | last |
| hr_rate | **300** PA | 1 / **0.6** / **0.4** | **25** | **+0.012** | **−0.012** | last |
| babip | **600** BIP | 1 / 0.8 / **1.0** | **26** | 0 | −0.003 | last |
| iso | **300** AB | 1 / **0.6** / **0.4** | **25** | 0 | **−0.012** | **weighted3** |

The previous fit, for comparison: k_rate 100 / 1-0.4-0.2 / peak **31** /
+0.006 / +0.012; hr_rate 200 / 1-0.6-0.6 / peak **23** / **−0.012 / −0.012**;
babip 600 / 1-0.8-1.0 / peak 26 / −0.012 / −0.003; iso 300 / 1-0.8-0.6 / peak
**23** / **−0.012 / −0.012**. The two components whose curves were flat lines
across the whole age range (HR/PA and ISO, peak at the grid edge with equal
slopes) are the two that changed most.

The ballasts and weights barely moved, which is the reassuring part: the
content of the original fit — **K% wants half stock's ballast and sharp
recency, BABIP wants three times stock's ballast and flat recency** — is
untouched by any of this. What moved is the age term.

And the constrained curves are readable as aging for the first time. **K%:
flat to 30, then +0.8% a year** — hitters' strikeout rates hold through the
prime and climb late. **BABIP: flat to 26, then −0.3% a year** — a slow
decline as legs go. **ISO: −1.2% a year after 25**, and **HR/PA: +1.2% a year
to 25, −1.2% after**. The two power components peak young and at the window
edge, which is the caveat below; the K% and BABIP shapes are the ones worth
quoting.

## In sample (2020–2024, the tuning window)

| Component | Stock MAE | Tuned MAE | Gain | Gain without the age term | Inner validation (2023–24) |
|---|---|---|---|---|---|
| k_rate | .030657 | **.030135** | **−1.70%** | −1.79% | −2.47% |
| bb_rate | .017569 | .017476 | −0.53% | −0.09% | **+0.19% → kept stock** |
| hr_rate | .010679 | **.010369** | **−2.90%** | −0.33% | −1.25% |
| babip | .027762 | **.026922** | **−3.03%** | −3.08% | −3.78% |
| iso | .038630 | **.037398** | **−3.19%** | −0.64% | −0.95% |

BB% trips the guard for the third time and keeps Tango's constants; the
holdout again confirms the call. Compare the in-sample column with the
previous fit's — K% −3.10% → −1.70%, HR/PA −3.66% → −2.90%, ISO −3.79% →
−3.19%, BABIP −3.28% → −3.03%. **Every in-sample gain got smaller**, which is
what taking away a degree of freedom is supposed to do. The question is
whether the gains that remain are the real ones.

## Out of sample — MAE by arm and cell

Same holdout, never seen by the search: season-level 2025 (train ≤ 2024) and
2026 (train ≤ 2025), and the three 2026 intra-season cutoffs.

| Component | Arm | 2025 | 2026 | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|---|---|---|
| **k_rate** | marcel_tuned | **.0298** | **.0272** | **.0269** | .0293 | **.0341** |
| | marcel_tuned_noage | .0300 | .0272 | .0269 | **.0292** | .0343 |
| | marcel | .0299 | .0278 | .0278 | .0296 | .0343 |
| | marcel_preseason | — | — | .0293 | .0310 | .0365 |
| | season_to_date | — | — | .0355 | .0340 | .0371 |
| | bayes_preseason | — | .0284 | .0296 | .0325 | .0386 |
| | previous_season | .0351 | .0307 | .0319 | .0342 | .0363 |
| | league_average | .0479 | .0496 | .0507 | .0518 | .0588 |
| **bb_rate** | marcel_tuned *(= marcel)* | **.0163** | .0182 | **.0172** | .0206 | **.0268** |
| | marcel_tuned_noage | .0164 | .0183 | .0172 | **.0205** | .0269 |
| | bayes_preseason | — | **.0179** | .0179 | .0222 | .0279 |
| | previous_season | .0202 | .0220 | .0210 | .0246 | .0286 |
| | league_average | .0244 | .0251 | .0254 | .0263 | .0321 |
| **hr_rate** | marcel_tuned | .0097 | **.0089** | .0100 | **.0114** | **.0148** |
| | marcel_tuned_noage | **.0097** | .0091 | **.0098** | .0115 | .0150 |
| | marcel | .0098 | .0091 | .0098 | .0115 | .0152 |
| | marcel_preseason | — | — | .0098 | .0118 | .0155 |
| | bayes_preseason | — | .0092 | .0099 | .0117 | .0153 |
| | previous_season | .0114 | .0113 | .0121 | .0141 | .0167 |
| | league_average | .0122 | .0117 | .0128 | .0135 | .0161 |
| **babip** | marcel_tuned | .0237 | **.0235** | **.0266** | .0344 | .0425 |
| | marcel_tuned_noage | **.0236** | .0235 | .0267 | **.0343** | .0426 |
| | marcel | .0249 | .0242 | .0269 | .0350 | .0424 |
| | marcel_preseason | — | — | .0270 | .0349 | **.0393** |
| | bayes_preseason | — | .0243 | .0271 | .0345 | .0417 |
| | previous_season | .0337 | .0329 | .0344 | .0389 | .0411 |
| | league_average | .0253 | .0254 | .0288 | .0352 | .0427 |
| **iso** | marcel_tuned | **.0336** | **.0320** | .0351 | .0411 | .0557 |
| | marcel_tuned_noage | .0337 | .0325 | .0346 | **.0410** | .0566 |
| | marcel | .0339 | .0333 | **.0345** | .0413 | .0567 |
| | marcel_preseason | — | — | .0352 | .0428 | .0558 |
| | bayes_preseason | — | .0341 | .0358 | .0436 | **.0551** |
| | previous_season | .0405 | .0404 | .0418 | .0493 | .0584 |
| | league_average | .0430 | .0406 | .0448 | .0472 | .0609 |

Scored players are unchanged from the section above: 407 (2025) · 362 (2026)
· 315 / 231 / 126 at the three cutoffs for K%/BB%/HR; 390 · 356 · 308 / 213 /
80 for ISO; 347 · 319 · 267 / 157 / 4 for BABIP.

## The paired test — new params vs stock Marcel

Per-player difference in absolute error, trials-weighted, **negative = tuned
better**. BB% is stock, hence identically zero.

| Component | 2025 | 2026 | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|---|---|
| k_rate | **−.00006 ± .00046** | **−.00059 ± .00039** | **−.00084 ± .00034** | **−.00031 ± .00041** | **−.00019 ± .00050** |
| bb_rate | .00000 | .00000 | .00000 | .00000 | .00000 |
| hr_rate | **−.00003 ± .00014** | **−.00015 ± .00014** | +.00022 ± .00016 | **−.00009 ± .00017** | **−.00041 ± .00024** |
| babip | **−.00127 ± .00040** | **−.00075 ± .00041** | **−.00030 ± .00043** | **−.00068 ± .00049** | +.00004 ± .00288 |
| iso | **−.00032 ± .00055** | **−.00131 ± .00056** | +.00062 ± .00059 | **−.00018 ± .00070** | **−.00099 ± .00106** |

Pooled within each component across its five cells, **new fit beside the
previous frozen one**:

| Component | New: pooled diff | SE | t | % of stock MAE | Cells won | Previous fit: % of stock MAE (t) |
|---|---|---|---|---|---|---|
| babip | −.000792 | .000215 | **−3.68** | **−2.96%** | 4/5 | −3.27% (−3.80) |
| k_rate | −.000415 | .000195 | **−2.12** | **−1.42%** | 5/5 | −2.41% (−2.68) |
| iso | −.000385 | .000286 | −1.35 | **−1.06%** | 4/5 | +0.02% (+0.02) |
| hr_rate | −.000048 | .000073 | −0.66 | −0.47% | 4/5 | −0.30% (−0.40) |
| bb_rate | 0 | — | — | 0 | 0/5 | 0 |

**Overall: −1.10% of stock Marcel's MAE (SE 0.30, t −3.71), 17 of 25 cells**,
against the previous fit's −1.10% ± 0.36 (t −3.06) on 15 of 25. Same point
estimate, tighter, more cells, and — the part that matters — **no component
is worse than stock any more**, where the previous fit left ISO at +0.02%.

## Head to head: new params vs the previous frozen params

Scored as a third arm on the same cells, paired per player. (Run with only
`marcel`, the new params and the old ones as arms, so the common-player set
is slightly larger than in the eight-arm table above and the percentages move
a hair; the comparison between the two columns is what this table is for.)

| Component | new − frozen, pooled | SE | t | % of stock MAE |
|---|---|---|---|---|
| iso | −.000313 | .000138 | **−2.27** | **−0.86%** |
| hr_rate | −.000028 | .000031 | −0.90 | −0.27% |
| bb_rate | 0 | — | — | 0 |
| babip | +.000037 | .000126 | +0.29 | +0.14% |
| k_rate | +.000247 | .000179 | +1.38 | +0.83% |
| **overall** | | **0.18** | **−0.22** | **−0.04%** |

Against stock in that same three-arm framing: **new −1.19% ± 0.29 (t −4.17),
17/25 cells; previous −1.15% ± 0.34 (t −3.36), 16/25.**

So: the new fit is **not worse pooled** and **not significantly worse on any
component** — K% gives back 0.83% at t +1.38 and BABIP 0.14% at t +0.29,
neither close to significant, while ISO gains 0.86% at t −2.27. That clears
the bar for refreezing, and `src/eval/marcel_params.json` now holds the
constrained fit.

## Do the HR/ISO in-sample gains survive now?

This was the whole question, and the answer is **yes, partly, and for the
first time.**

| Component | In sample (old fit) | Holdout (old fit) | In sample (new fit) | Holdout (new fit) |
|---|---|---|---|---|
| hr_rate | −3.66% | −0.30% (t −0.40) | −2.90% | −0.47% (t −0.66) |
| iso | −3.79% | +0.02% (t +0.02) | −3.19% | −1.06% (t −1.35) |

Neither is significant on its own — HR/PA's five cells are small and ISO's
t is −1.35 — but both now point the right way, ISO by a full percent where it
used to be exactly zero, and ISO wins its head-to-head against the old
params at t −2.27. The pattern is the one a real effect makes and the old fit
did not: a *smaller* in-sample gain that *survives*.

K% and BABIP give a little back (−2.41% → −1.42%, −3.27% → −2.96%), which is
the same story from the other side: part of what they were earning was the
level correction, and they no longer get it.

## The level, before and after

Trials-weighted mean projection minus the realized rate of the scored
players, averaged over the tuning window:

| Component | Stock | Previous fit | New fit |
|---|---|---|---|
| k_rate | +.00219 | **−.00227** | +.00184 |
| bb_rate | +.00045 | +.00045 | +.00045 |
| hr_rate | +.00201 | −.00027 | +.00032 |
| babip | +.00133 | +.00046 | −.00103 |
| iso | +.00838 | **−.00259** | +.00121 |

The previous fit's age curve was pushing K% **down** by 0.45 points relative
to stock and ISO **down** by 1.1 — a level correction wearing an aging
curve's clothes. Note that on ISO it was correcting a real stock bias
(+.0084) and overshooting it; the constrained curve corrects the same bias
and lands nearer zero (+.0012).

**On the live board** (`scripts/build_ros_projections.py --as-of 2026-09-02`,
420 hitters, `stale: false`), mean projected K% weighted by projected
rest-of-season PA goes **.2133 → .2185** against a 2026 league K% of **.2207**
through Sept 1: the gap closes from −7.4 to −2.2 tenths of a point.
Unweighted across the 420 projected hitters it goes .2201 → .2252, which now
sits *above* league, as a bench-heavy population should.

## Caveats

- **The two power components still peak at the window edge.** HR/PA and ISO
  both land on peak age 25 with a −0.012 old-side slope, both at the edge of
  their grids. The constraint stops them being straight lines but does not
  stop them wanting to be as young-peaking and as steep as the window allows.
  Read them as "power declines from the mid-twenties, at about the fastest
  rate the grid offers", and treat the exact peak as unidentified.
- **A multiplier fixed at 1.0 at the peak still shifts the level.** Inside
  the constrained family the age term is ≤ 1 everywhere for the four
  "bigger is better" components and ≥ 1 everywhere for K%, so any non-flat
  curve moves the population mean. The constraint bounds that shift; it does
  not remove it. Renormalizing the age multiplier to be mean-1 over the
  projected population would, and is the obvious next thing to try.
- **The league-rate choice and the guard share a validation split.** Both use
  fit 2020–2022 / score 2023–2024. The option is chosen on that split and
  then the fitted component is guarded on it, which is a mild reuse. It is
  all inside the tuning window and the holdout is untouched, but the ISO
  `weighted3` pick in particular rests on two years.
- **`drift` is untested where it would matter most.** It collapses to `last`
  at every intra-season cutoff by construction, so three of the five holdout
  cells cannot distinguish it, and the two that can are the two season-level
  ones. A league genuinely trending mid-season is not something this holdout
  contains.
- **BB% has now kept stock's constants three times running** — the search's
  in-sample gain is small (−0.53%), the inner validation says +0.19%, and the
  holdout is identically zero because the frozen params *are* stock. Whatever
  is left in BB% is not reachable by these six knobs.
- **The holdout has now been looked at twice.** The constants were chosen on
  the tuning window both times and the guard is an inner-validation rule, but
  this is the second fit scored on the same 2025/2026 cells, and the honest
  reading of a −0.04% head-to-head is "the two fits are indistinguishable out
  of sample; the case for the new one is that it is the one whose in-sample
  gains are the same kind of thing as its out-of-sample gains."
- Everything the previous section says about **cell independence, the 4-player
  BABIP cell at Aug 1, and the coarse flat grid** still applies unchanged.

## Reproducing

```
python scripts/tune_marcel.py --league-modes   # the option table + levels
python scripts/tune_marcel.py --markdown       # refit, refreeze, score, tables
python scripts/tune_marcel.py --skip-tune      # score the committed params
python scripts/build_ros_projections.py --as-of 2026-09-02
```

---

# The pitcher side of station A — Sept 3, 2026

**Run:** Sept 3, 2026 · **Data:** 2015–2026 pitcher season totals from the
Stats API (`data/parquet/pitcher_seasons_api.parquet`, 9,811 pitcher-seasons,
Chadwick ages on 100% of them) plus the PA-level 2026 outcomes
(`pa_outcomes/pa_outcomes_2026.parquet`, 157,749 plate appearances through
Sept 1) **aggregated by pitcher instead of by batter** · **Method:**
`python scripts/run_pitcher_backtest.py` for the scores and the gate,
`python scripts/tune_marcel_pitchers.py` for the constants.

Station A has always had a pitcher shaped hole in it. Station E's starter term
computes Marcel-weighted K, BB+HBP and HR rates per batter faced and has done
since it shipped, but those rates were never scored against a baseline and
never reached the site — they went straight into a FIP and out into a game
price. This section scores them, tunes them, and puts four of them on the
player pages.

Nothing here is a new estimator. The pitcher components register into the same
`COMPONENTS` registry the hitter ones live in, under a `p_` prefix, and
`marcel_pitcher` *is* `marcel_tuned` carrying pitcher constants. The only
thing the harness needed was to stop assuming its id column was called
`batter`.

## The components, and where the definitions do not overlap

| Component | Numerator / denominator | Same as station E? |
|---|---|---|
| `p_k_rate` | K / BF | **yes**, bit for bit |
| `p_bbhbp_rate` | (BB + HBP) / BF | **yes**, bit for bit |
| `p_hr_rate` | HR / BF | **yes**, bit for bit |
| `p_bb_rate` | BB / BF | **no** — station E has only the pair |
| `p_babip` | (H − HR) / (AB − K − HR + SF) | **no** — FIP is defined to ignore balls in play |

On the three that overlap, `pitcher_rates` with `PITCHER_STOCK_PARAMS`
reproduces `starters.marcel_rates` to 1e-16 on the real 2024–2026 counts frame,
and `tests/test_sim/test_starters.py` pins the two together on a fixed
pitcher-season, on a whole rate table, and on the ballast unit conversion.
`src/sim/starters.py` is now a *caller* of the provider rather than a second
implementation of it, so station A and station E cannot drift.

Two definitional gaps remain and are not reproducible, because station E never
had them. **BB%** on the site means walks; station E's FIP term folds hit
batsmen in because FIP treats them identically, and the published
stabilization point it regresses with is the pair's. Both are projected, both
are scored, and only the walks-only one is a site column. **BABIP against**
does not exist in station E at all — FIP is *defined* to ignore balls in play —
so it is projected here because the site wants it and because "does
BABIP-against carry any signal" is now a question the harness can answer.

Stock constants are station E's: 5/4/3 recency, ballast = 2× the published
stabilization point (K 70 BF, BB and BB+HBP 170, HR 1300, BABIP 2000 BIP), and
**no age term at all** — station E never aged a pitcher, so every age effect in
the tuned arm is something the search found rather than something assumed.

## The arms

| Arm | Sees 2026? | What it is |
|---|---|---|
| `marcel_pitcher_tuned` | **yes** | the served arm: fitted ballast, recency weights and constrained age curve, frozen in `src/eval/marcel_pitcher_params.json` |
| `marcel_pitcher` | **yes** | stock, i.e. what station E runs |
| `marcel_pitcher_tuned_preseason` | no | the same tuned arm with the partial season withheld — the control that isolates in-season information |
| `season_to_date` | **yes** | this year's rate regressed to league with the component's stabilization point |
| `previous_season` | no | 2025 rate, unregressed |
| `league_average` | **yes** | league rate through the cutoff |

Five cells: season-level 2025 and 2026 (train ≤ Y−1) and the three 2026
cutoffs (train on everything strictly before the date, score every batter faced
on or after it). Scored on pitchers with ≥ 100 realized trials after the
cutoff, trials-weighted, on the same pitchers across every arm.

## MAE by arm and cell (bold = best arm in that cell)

| Component | Arm | 2025 | 2026 | 2026-05-01 | 2026-07-01 | 2026-08-01 |
|---|---|---|---|---|---|---|
| **K%** | marcel_pitcher_tuned | **0.0288** | **0.0314** | **0.0308** | **0.0354** | 0.0383 |
|  | marcel_pitcher | 0.0305 | 0.0324 | 0.0318 | 0.0357 | 0.0397 |
|  | season_to_date | — | — | 0.0368 | 0.0367 | **0.0379** |
|  | marcel_pitcher_tuned_preseason | — | — | 0.0331 | 0.0377 | 0.0433 |
|  | marcel_pitcher_preseason | — | — | 0.0342 | 0.0389 | 0.0447 |
|  | previous_season | 0.0373 | 0.0389 | 0.0387 | 0.0411 | 0.0495 |
|  | league_average | 0.0402 | 0.0416 | 0.0445 | 0.0472 | 0.0499 |
| **BB%** | marcel_pitcher_tuned | **0.0142** | **0.0160** | 0.0168 | **0.0202** | **0.0231** |
|  | marcel_pitcher | **0.0142** | **0.0160** | 0.0168 | **0.0202** | **0.0231** |
|  | season_to_date | — | — | 0.0214 | 0.0204 | 0.0234 |
|  | marcel_pitcher_tuned_preseason | — | — | **0.0166** | 0.0212 | 0.0241 |
|  | previous_season | 0.0211 | 0.0202 | 0.0199 | 0.0248 | 0.0263 |
|  | league_average | 0.0182 | 0.0201 | 0.0245 | 0.0245 | 0.0275 |
| **(BB+HBP)%** | marcel_pitcher_tuned | **0.0148** | **0.0174** | 0.0184 | **0.0211** | 0.0236 |
|  | marcel_pitcher | **0.0148** | **0.0174** | 0.0184 | **0.0211** | 0.0236 |
|  | season_to_date | — | — | 0.0231 | 0.0215 | **0.0235** |
|  | marcel_pitcher_tuned_preseason | — | — | **0.0182** | 0.0220 | 0.0245 |
|  | previous_season | 0.0216 | 0.0218 | 0.0217 | 0.0265 | 0.0273 |
|  | league_average | 0.0196 | 0.0220 | 0.0265 | 0.0260 | 0.0279 |
| **HR/BF** | marcel_pitcher_tuned | **0.0087** | **0.0089** | **0.0095** | 0.0123 | **0.0124** |
|  | marcel_pitcher | **0.0087** | **0.0089** | **0.0095** | 0.0123 | **0.0124** |
|  | season_to_date | — | — | 0.0098 | 0.0126 | 0.0126 |
|  | marcel_pitcher_tuned_preseason | — | — | 0.0096 | **0.0122** | 0.0125 |
|  | previous_season | 0.0120 | 0.0117 | 0.0123 | 0.0130 | 0.0137 |
|  | league_average | 0.0090 | 0.0094 | 0.0100 | 0.0129 | 0.0130 |
| **BABIP** | marcel_pitcher_tuned | **0.0257** | 0.0270 | **0.0263** | 0.0316 | 0.0403 |
|  | marcel_pitcher | 0.0259 | **0.0269** | 0.0263 | 0.0315 | 0.0403 |
|  | season_to_date | — | — | 0.0264 | **0.0315** | 0.0400 |
|  | marcel_pitcher_tuned_preseason | — | — | 0.0264 | 0.0320 | **0.0386** |
|  | previous_season | 0.0395 | 0.0391 | 0.0382 | 0.0431 | 0.0439 |
|  | league_average | 0.0264 | 0.0272 | 0.0267 | 0.0317 | 0.0405 |

Pitchers scored (common across arms): K% / BB% / (BB+HBP)% / HR — 427 · 413 ·
326 · 189 · 104; BABIP — 366 · 333 · 240 · 114 · **12**.

`marcel_pitcher_preseason` — the *stock* preseason control — is listed only
under K%. For the three components the guard sent back to stock it is the same
arm as `marcel_pitcher_tuned_preseason` to the digit; for BABIP it differs by
at most .0012 (Aug 1: .0398 against .0386) and neither is the best cell.
`scripts/run_pitcher_backtest.py` prints every arm in every cell.

## The paired test against each dumb baseline

Trials-weighted paired difference in absolute error, served arm minus
baseline, per pitcher on the same cell. Negative means the served arm is
better. These per-cell standard errors are the honest ones.

**vs `league_average`**

| Component | 2025 | 2026 | 2026-05-01 | 2026-07-01 | 2026-08-01 |
|---|---|---|---|---|---|
| K% | −.01142 ± .00161 (t −7.1) | −.01027 ± .00160 (t −6.4) | −.01370 ± .00157 (t −8.7) | −.01180 ± .00206 (t −5.7) | −.01158 ± .00281 (t −4.1) |
| BB% | −.00392 ± .00061 (t −6.4) | −.00407 ± .00059 (t −6.9) | −.00760 ± .00079 (t −9.6) | −.00427 ± .00099 (t −4.3) | −.00439 ± .00125 (t −3.5) |
| (BB+HBP)% | −.00479 ± .00066 (t −7.3) | −.00458 ± .00064 (t −7.1) | −.00809 ± .00087 (t −9.3) | −.00483 ± .00102 (t −4.7) | −.00431 ± .00125 (t −3.4) |
| HR/BF | −.00031 ± .00010 (t −3.1) | −.00054 ± .00010 (t −5.3) | −.00052 ± .00009 (t −5.7) | −.00060 ± .00014 (t −4.2) | −.00059 ± .00020 (t −3.0) |
| BABIP | −.00070 ± .00030 (t −2.3) | −.00026 ± .00034 (t −0.8) | −.00040 ± .00026 (t −1.6) | −.00010 ± .00046 (t −0.2) | −.00017 ± .00123 (t −0.1) |

**vs `previous_season`**

| Component | 2025 | 2026 | 2026-05-01 | 2026-07-01 | 2026-08-01 |
|---|---|---|---|---|---|
| K% | −.00853 ± .00133 (t −6.4) | −.00754 ± .00113 (t −6.7) | −.00784 ± .00162 (t −4.8) | −.00579 ± .00229 (t −2.5) | −.01123 ± .00294 (t −3.8) |
| BB% | −.00684 ± .00110 (t −6.2) | −.00418 ± .00077 (t −5.4) | −.00304 ± .00086 (t −3.5) | −.00454 ± .00107 (t −4.2) | −.00316 ± .00121 (t −2.6) |
| (BB+HBP)% | −.00679 ± .00113 (t −6.0) | −.00437 ± .00086 (t −5.1) | −.00322 ± .00090 (t −3.6) | −.00534 ± .00126 (t −4.2) | −.00377 ± .00141 (t −2.7) |
| HR/BF | −.00328 ± .00060 (t −5.4) | −.00278 ± .00048 (t −5.7) | −.00287 ± .00056 (t −5.1) | −.00069 ± .00065 (t −1.1) | −.00127 ± .00086 (t −1.5) |
| BABIP | −.01378 ± .00203 (t −6.8) | −.01212 ± .00183 (t −6.6) | −.01192 ± .00207 (t −5.8) | −.01143 ± .00281 (t −4.1) | −.00354 ± .00771 (t −0.5) |

**vs `season_to_date`** (only defined at a cutoff)

| Component | 2026-05-01 | 2026-07-01 | 2026-08-01 |
|---|---|---|---|
| K% | −.00596 ± .00119 (t −5.0) | −.00138 ± .00113 (t −1.2) | **+.00036** ± .00116 (t +0.3) |
| BB% | −.00453 ± .00060 (t −7.5) | −.00018 ± .00056 (t −0.3) | −.00031 ± .00067 (t −0.5) |
| (BB+HBP)% | −.00470 ± .00067 (t −7.1) | −.00040 ± .00060 (t −0.7) | **+.00009** ± .00069 (t +0.1) |
| HR/BF | −.00031 ± .00009 (t −3.6) | −.00031 ± .00010 (t −3.1) | −.00017 ± .00012 (t −1.4) |
| BABIP | −.00019 ± .00017 (t −1.1) | **+.00018** ± .00026 (t +0.7) | **+.00033** ± .00072 (t +0.4) |

## The gate

The rule for this task: **the served arm must beat every dumb baseline out of
sample on every component it serves.** Pooled n-weighted within a component
across its cells — the cells share pitchers and the three cutoffs are nested
windows of one season, so this pooled SE is optimistic and the per-cell tables
above are the honest ones.

| Component | vs league average | vs previous season | vs season to date | Clears? |
|---|---|---|---|---|
| **K%** | −.01166 (t −14.3) | −.00793 (t −11.0) | −.00350 (t −4.7) | **yes** |
| **BB%** | −.00487 (t −14.3) | −.00468 (t −10.1) | −.00249 (t −6.6) | **yes** |
| **(BB+HBP)%** | −.00544 (t −14.9) | −.00490 (t −9.9) | −.00258 (t −6.3) | **yes** |
| **HR/BF** | −.00048 (t −9.3) | −.00257 (t −9.3) | −.00029 (t −4.9) | **yes** |
| **BABIP** | −.00042 (t −2.5) | −.01248 (t −11.7) | **−.00006 (t −0.4)** | **yes, by nothing** |

All five clear, so all five are served — the four the site shows, plus the
walks-plus-hit-batsmen rate station E consumes. **Read BABIP's row honestly.**
Its win over season-to-date is six hundred-thousandths of a rate with a t of
−0.4. Its win over league average is 1.5% of league average's MAE, but only
the 2025 cell reaches a t past 2 — the other four sit between −1.6 and −0.1.
What the table actually says about BABIP against is DIPS: a pitcher's own
balls in play carry so little signal that regressing them 1,387 balls deep to
league is nearly indistinguishable from just using league. It clears the letter of the gate. It
is not a model win, and the accuracy page says so in a note.

The one place the served arm loses a *cell* outright is K% and (BB+HBP)% at
Aug 1 against season-to-date, on 104 pitchers with a month left, at t +0.3 and
+0.1. That is the same shape the hitter table has at Aug 1 and it is noise;
pooling is what the gate is stated on.

## The in-season increment

`marcel_pitcher_tuned` minus `marcel_pitcher_tuned_preseason`, as a fraction of
the preseason arm's MAE (negative = folding in the current season helps):

| Component | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|
| K% | −6.9% | −6.1% | −11.5% |
| BB% | +1.2% | −4.7% | −4.1% |
| (BB+HBP)% | +1.1% | −4.1% | −3.7% |
| HR/BF | −1.0% | +0.8% | −0.8% |
| BABIP | −0.4% | −1.3% | +4.4% *(n=12)* |

Same ordering the hitters show and for the same reason: in-season batters
faced carry information about the components that carry information at all.
Pitcher K% gains more from the current season than hitter K% does (−6 to −11%
against −5 to −6%), which is what you would expect from a rate that stabilizes
in 70 batters faced against one that takes 60 plate appearances *and* is
partly the pitcher's doing. BB% at May 1 is the one cell where the partial
season actively hurts, and it is a month of data against a 340-batter ballast.

## Tuning the constants

Same procedure, same module (`src/eval/tuning.py`), same script layer —
`scripts/tune_marcel_pitchers.py` imports the search, the guard and the league
choice from `scripts/tune_marcel.py` so the two runs cannot diverge in method.
Coordinate search on predict years 2020–2024 (train ≤ Y−1), objective = mean
trials-weighted MAE, the age term constrained to a peak inside 25–31 with
slopes of opposite signs, and an inner-validation guard (fit 2020–2022, score
2023–2024) that sends a component back to stock if the fit does not beat stock
there.

**The aging direction is the hitter table's, flipped on strikeouts and only on
strikeouts.** A high K% is a good pitcher and a bad hitter; walks, home runs
and hits on balls in play are bad for the pitcher either way. So `p_k_rate`
peaks at the peak age and the other four trough there.

### The inner validation, which is what the guard reads

| Component | stock MAE | tuned MAE | tuned vs stock | ballast+weights only | generalises |
|---|---|---|---|---|---|
| p_k_rate | .029668 | .028763 | **−3.05%** | −0.32% | **yes** |
| p_bb_rate | .016560 | .016650 | +0.54% | +0.54% | no |
| p_bbhbp_rate | .018013 | .018084 | +0.39% | +0.39% | no |
| p_hr_rate | .008530 | .008573 | +0.50% | +0.50% | no |
| p_babip | .025362 | .025204 | **−0.62%** | −0.62% | **yes** |

Three of five do not generalise inside the tuning window and are frozen
holding stock's constants. That is the guard doing exactly what it is for, and
it is why those components' two Marcel rows in the MAE table are identical.

### The chosen constants

| Component | ballast (real trials) | weights | league rate | peak age | slope young / old |
|---|---|---|---|---|---|
| **p_k_rate** | **107 BF** (was 140) | **1 / 0.4 / 0.2** | **weighted3** | **26** | **+0.012 / −0.006** |
| p_bb_rate | 340 BF | 5 / 4 / 3 | last | — | flat *(stock, by the guard)* |
| p_bbhbp_rate | 340 BF | 5 / 4 / 3 | last | — | flat *(stock, by the guard)* |
| p_hr_rate | 2600 BF | 5 / 4 / 3 | last | — | flat *(stock, by the guard)* |
| **p_babip** | **1387 BIP** (was 4000) | **1 / 0.2 / 0.4** | **weighted3** | — | flat |

Ballasts are quoted in real trials at the most recent season's weight, which
is the unit the published stabilization points are in. (`MarcelParams.ballast`
stores them at the *average* year weight, a factor of `w0/mean(w)` away; the
conversion is `starters.marcel_params` and it has its own test.) Note that the
conversion uses each component's *own* fitted weights, so K%'s stored 200 is
107 real batters faced and BABIP's stored 2600 is 1,387 real balls in play —
both regress *less* than stock does, not more, because the shorter memory has
already thrown away most of the sample.

Two things in that table are worth reading twice. **K% wants a much shorter
memory than 5/4/3** — the fitted weights put a fifth of the current season's
weight on the year before last, against three fifths for Tango's defaults —
which is the opposite of what the hitter fit wanted and is consistent with
pitcher strikeout rate being a fast-moving thing (velocity, a new pitch, an
injury) rather than a stable skill. And **K% got a real aging curve**: it
peaks at 26 and falls after, which is the shape a pitcher's strikeout rate
actually has, and the search was constrained to curves that turn over so it
could not have produced a straight line instead.

### Out of sample, tuned vs stock

Paired per-pitcher difference in absolute error, `marcel_pitcher_tuned` minus
`marcel_pitcher`:

| Component | 2025 | 2026 | 2026-05-01 | 2026-07-01 | 2026-08-01 | pooled % of stock MAE |
|---|---|---|---|---|---|---|
| p_k_rate | −.00169 ± .00052 | −.00107 ± .00049 | −.00091 ± .00060 | −.00036 ± .00077 | −.00139 ± .00101 | **−3.52%** (t −4.2) |
| p_babip | −.00026 ± .00017 | +.00005 ± .00020 | −.00009 ± .00013 | +.00009 ± .00025 | +.00005 ± .00075 | −0.32% (t −0.9) |
| p_bb_rate, p_bbhbp_rate, p_hr_rate | 0 | 0 | 0 | 0 | 0 | 0 *(stock, by the guard)* |

Pooled over all 25 component × cell cells the frozen file is **−0.81% ± 0.19
of stock's MAE (t −4.3)**, with no component worse than stock. Almost all of
it is K%.

### The arm that was not frozen, and why it is worth naming

The unguarded ballast-and-weights-only fit (`marcel_pitcher_tuned_noage`) does
*better* out of sample than the frozen file — **−1.16% ± 0.22** pooled,
negative on 5/5 components, 20 of 25 cells — because it keeps its ballast and
weight fits for the three components the guard sent back to stock. Their
inner-validation numbers said those fits were noise (+0.39% to +0.54%); the
holdout says they were worth about a percent each. The two disagree, the
sample sizes are similar, and the guard is a rule set before the holdout was
looked at. The frozen file keeps the guard's answer. What that costs is
recorded here rather than quietly re-fit away: **a third of a percent of MAE,
if the holdout is right and the inner validation is wrong.**

## Caveats

- **The gate is stated on pooled differences, and pooling is generous.** The
  five cells share pitchers, and the three cutoffs are nested windows of one
  2026. The per-cell tables above are the honest ones; nothing that clears
  pooled but loses per-cell at every horizon should be trusted, and nothing
  here does — except BABIP, which does not really win anywhere.
- **BABIP against at Aug 1 is 12 pitchers.** A 100-BIP floor at a one-month
  horizon is brutal. Treat that column as absent, and the Jul 1 column
  (n=114) as thin.
- **2026 appears in the holdout twice.** The season-level 2026 cell and the
  three cutoffs are the same season cut four ways. They are not four
  independent pieces of evidence, and the 2025 cell is the only fully
  independent one.
- **Prior seasons come from the Stats API table, the partial season from
  Statcast PA data.** The same universe-drift caveat the hitter side carries:
  the 2026 API pitcher table has 158,289 batters faced against the PA
  parquet's 157,749, a gap of 0.34%. Rebuilding prior seasons from
  `pa_outcomes_<year>.parquet` would close it; only 2026 exists in R2 today.
- **2020 is in the tuning window.** A 60-game season is a sixth of the
  training weight for the 2021 fit and it is not a normal season for pitcher
  workloads. The hitter fit has the same problem and it was left alone here
  for comparability rather than because it is harmless.
- **The tuned K% weights and the tuned K% age curve were fitted together.**
  A much shorter memory and a peak at 26 are two ways of saying "recent is
  what matters", and the coordinate search cannot tell which of them is
  carrying the −3.5%. The no-age arm gets −1.2% on K%, which suggests roughly
  a third of it is the weights, but that arm also has different ballasts.
- **Nothing in the odds chain moved.** Station E keeps stock's constants
  deliberately: a refit of station A must not change a game price without the
  game price being re-scored. `scripts/backtest_game_odds.py --season 2026
  --min-games 20 --market data/parquet/market_closes_2026.parquet` still reads
  `pythag_C_sp_bpa_ip` **0.24388** on the 756 common games after the rate table
  became a call into the provider.

## Reproducing

```
python scripts/build_pitcher_seasons.py              # the season table
python scripts/tune_marcel_pitchers.py --inner-validation
python scripts/tune_marcel_pitchers.py --markdown    # refit, refreeze, score
python scripts/tune_marcel_pitchers.py --skip-tune   # score the committed params
python scripts/run_pitcher_backtest.py --markdown    # the tables and the gate
python scripts/build_ros_projections.py              # the site's pitcher block
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
       --market data/parquet/market_closes_2026.parquet
```

---

# The fair fight — the Bayesian arm refit at the cutoff (BAS-59)

**Run:** Sept 3, 2026 · **Data:** PA-level 2024, 2025 and partial-2026
outcomes · **Method:** the Bayesian workflow this repo expects
([methods.md §5](methods.md#5-workflow-by-family)) — prior predictive,
convergence, posterior predictive, LOO for structure, walk-forward for the
gate · **Scale: every table below is labelled with the fit that produced it,
and none of it is the full Modal refit.**

Every published comparison between our Bayesian components and Marcel had been
rigged — not on purpose, but structurally. `bayes_preseason` is a fixed April
10 file that has never seen a 2026 plate appearance; `marcel` had seen every
one before the cutoff. And `src/models/` had no dated cutoff at all: it knew
only `cutoff_year`, which filters *active batters*, not plate appearances, so
there was no way to refit the Bayesian model on the partial season the
baselines were being fed. The section above prices that information at 4.6–6.1%
of K% MAE — the same order as the entire deficit the Bayesian arm was being
charged with. **The comparison had never been run.**

`src/models/cutoff.py` and `src/eval/bayes_arm.py` run it. `apply_cutoff` keeps
exactly `game_date < cutoff`, the same strict inequality
`intraseason.split_at_cutoff` uses, so a game played *on* the cutoff date is
withheld from both arms; `assert_no_post_cutoff` is the model-side twin of
`assert_split_clean` and fires twice on the way into a fit.

## Giving both arms the same three seasons

The caveat above used to read "only 2026 exists in R2 today". It no longer
holds: `src/data/pa_outcomes_pipeline.py` rebuilt **2024 and 2025** from the
Statcast parquets in R2, so the Bayesian arm reads 2024 + 2025 + partial 2026
while `marcel` reads its 5/4/3 window over the same three seasons. **Same
seasons, same cutoff, same plate appearances** — that is what makes it a fair
fight rather than another handicap. Ages come from the Chadwick register
(`scripts/build_birthdates.py`; 649 batters, 100% matched) instead of the
`first_year − 23` fallback.

## K% MAE by arm and cutoff (bold = best arm at that cutoff)

**Scale: 4 chains × 1500 draws (tune 1500), PyMC's own NUTS, opposing-pitcher
term off, full hitter coverage.** See
[the pitcher term](#the-opposing-pitcher-term--loo-says-yes-and-loo-is-not-a-gate)
for why it is off here — and for the one cutoff scored with it on, where it
turns out to make the projection *worse* despite LOO preferring it by 11 dSE.

| Arm | Sees 2026? | MAE May 1 | MAE Jul 1 | MAE Aug 1 |
|---|---|---|---|---|
| `marcel_tuned` (live) | **yes** | **.0269** | **.0293** | **.0341** |
| **`bayes` — refit at the cutoff** | **yes** | .0278 | .0302 | .0341 |
| `marcel` (stock, + partial) | **yes** | .0278 | .0296 | .0343 |
| `marcel_tuned_preseason` | no | .0287 | .0308 | .0357 |
| `marcel_preseason` | no | .0293 | .0310 | .0365 |
| `bayes_preseason` | no | .0296 | .0325 | .0386 |
| `previous_season` | no | .0319 | .0342 | .0363 |
| `season_to_date` | **yes** | .0355 | .0340 | .0371 |
| `league_average` | **yes** | .0507 | .0518 | .0588 |

Two cells are closer than four decimals show: at May 1 stock `marcel` is
.027755 against the refit arm's .027767, and at Aug 1 `marcel_tuned` is .034120
against .034148. Tuned Marcel is still the best arm in every column; at Aug 1 it
is ahead by three parts in a hundred thousand.

Hitters scored: 315 at May 1, 231 at Jul 1, 126 at Aug 1 — **identical to the
run without the Bayesian arm**, so adding it did not shrink the common player
set. Batters the fit never saw (a call-up, or anyone under the 50-PA career
floor) are projected from the fitted population rather than dropped, which is
what keeps that number stable.

## The paired test — within-hitter, against the live arm

Trials-weighted difference in absolute error against `marcel_tuned`, paired on
the hitter. Positive means the live arm is better.

| Arm | Cutoff | n | diff | SE | t |
|---|---|---|---|---|---|
| **`bayes` refit** | May 1 | 315 | +.00086 | .00055 | **1.57** |
| **`bayes` refit** | Jul 1 | 231 | +.00092 | .00058 | **1.58** |
| **`bayes` refit** | Aug 1 | 126 | +.00003 | .00072 | **0.04** |
| `bayes_preseason` | May 1 | 315 | +.00270 | .00099 | 2.73 |
| `bayes_preseason` | Jul 1 | 231 | +.00320 | .00122 | 2.62 |
| `bayes_preseason` | Aug 1 | 126 | +.00444 | .00163 | 2.72 |
| `marcel` (stock, + partial) | May 1 | 315 | +.00084 | .00034 | 2.47 |
| `marcel` (stock, + partial) | Jul 1 | 231 | +.00031 | .00041 | 0.75 |
| `marcel` (stock, + partial) | Aug 1 | 126 | +.00019 | .00050 | 0.38 |

## Reading it

**The deficit was the withheld season — essentially all of it.** The same
estimator, refit on the same information, moves from losing to `marcel_tuned`
by .0027–.0044 at t = 2.6–2.7 to losing by .0000–.0009 at t ≤ 1.6: not
separable from noise at any cutoff, and a dead heat at Aug 1.

**The control that isolates the season.** `bayes_preseason` cannot answer "what
is the current season worth to this model", because it is the legacy April 10
projection file — a different code path, no opposing-pitcher term, fit under the
old `cutoff_year` semantics — so refit-minus-`bayes_preseason` mixes the
withheld season together with every change since April. The clean control is
*this* estimator and *this* code fitted on 2024 + 2025 only, scored at the same
three cutoffs (`--bayes-seasons 2024 2025`; the fit does not depend on the
cutoff, so it is one 1,471-cell fit over 662 batters scored three times, r-hat
1.0077, 0 divergences):

| Cutoff | `bayes` refit | `bayes_withheld` (same code, no 2026) | `bayes_preseason` (Apr 10 file) | the season is worth | for comparison: Marcel's own increment |
|---|---|---|---|---|---|
| May 1 | .02777 | .02867 | .02961 | **3.2%** | 6.4% |
| Jul 1 | .03020 | .03212 | .03248 | **6.0%** | 5.0% |
| Aug 1 | .03415 | .03633 | .03856 | **6.0%** | 4.5% |

So the season is worth **3–6%** to this model — the same order as the 4.5–6.4%
it is worth to tuned Marcel, which is what you would expect of two estimators
reading the same plate appearances. The remaining 1.1–5.8% between
`bayes_withheld` and `bayes_preseason` is the model and code moving on since
April, not information. Quoting the full 6–11% as an information gain would
have been the same kind of conflation this section exists to undo, in our own
favour.

**And the control still loses.** Paired against `marcel_tuned`,
`bayes_withheld` is +.00176 / +.00284 / +.00221 at t = 2.37 / 2.94 / 1.61 —
significantly behind at two of three cutoffs. The refit arm, same code, same
model, is +.00086 / +.00092 / +.00003 at t = 1.57 / 1.58 / 0.04. **Withholding
the season is the difference between losing significantly and not losing at
all.** The old sentence — "the Bayesian components buy nothing over Marcel on
the same information" — was reading a handicap as a model result. On the same
information the refit arm is level with **stock** Marcel-with-partial (.00001
behind at May 1, .0006 behind at Jul 1, .0002 ahead at Aug 1) and
statistically indistinguishable from the tuned one.

**It still does not clear the gate, and that matters.** Not losing
significantly is not winning. `marcel_tuned` is better on the point estimate at
every cutoff; the refit arm costs an MCMC fit per cutoff against a closed form;
and [the gate rule](architecture.md#3-the-gate-rule) asks a challenger to
*beat* its baseline out of sample. It does not. What changed is that the reason
it does not is now an honest one, and the honest gap is roughly a tenth of the
one we had been publishing.

## What each refit actually was

| Cutoff | Cells | PA | Batters | max r-hat | min ESS | divergences | BFMI |
|---|---|---|---|---|---|---|---|
| May 1 | 1,970 | 398,229 | 679 | 1.0045 | 851 | 0 | .66–.72 |
| Jul 1 | 2,116 | 459,421 | 719 | 1.0101 | 769 | 0 | .61–.67 |
| Aug 1 | 2,156 | 487,422 | 732 | 1.0124 | 486 | 0 | .68–.73 |

Zero divergences everywhere and BFMI comfortably above the 0.3 alarm line. Two
of the three worst r-hats sit a hair over the 1.01 convention (1.0101 and
1.0124), and in all three fits both the worst r-hat and the worst ESS land on
the age terms `beta_age`/`beta_age2` — never on a player, league or park term.
An independent 2-chain × 1000-draw run of the same three fits put the arm's K%
MAE at .027764 / .030246 / .034167 against the .027767 / .030204 / .034148
above: agreement to within 4×10⁻⁵, an order of magnitude smaller than the gap
being reported, so the result is not a sampling artefact.

## Is the fit trustworthy? (`scripts/validate_pa_k_rate.py`)

Run at the Jul 1 cutoff with the pitcher term **on**, at the largest scale that
completed here: **2 chains × 300 draws (tune 500), 60 busiest batters, 97,001
PA in 43,374 cells, 1,176 pitchers, 12 minutes.**

- **Prior predictive.** A hitter's implied strikeout rate has median .242 and a
  5–95 band of .107–.482, with 2.7% of mass outside .05–.55. Loose, but not on
  impossible rates — which is the whole question this check exists to answer,
  and it costs seconds (`--prior-only`) against hours of sampling.
- **Convergence.** 0 divergences; BFMI .94 / .91. Every scalar has r-hat ≤ 1.01
  (`sigma_pitcher` 1.00 at ESS 386). Worst r-hat over the whole trace is 1.044
  on an individual `z_ability` coordinate at ESS 68 — the per-batter terms are
  the under-sampled part of this reduced fit, and the reason it is reported as
  a reduced fit.
- **Posterior predictive**, rolled up to a hitter's rate at his real exposure
  (a cell holds ~2 PA once the pitcher is in the key, so a per-cell rate is
  almost always 0, ½ or 1 and checks nothing):

  | | observed | replicated (mean) | replicated 5–95 |
  |---|---|---|---|
  | league K rate | .2062 | .2064 | .2037–.2093 |
  | SD across hitters | .0549 | .0550 | .0522–.0577 |

  The model reproduces both the level and the spread of hitters. That is the
  check that says the partial pooling is not over- or under-shrinking.

## The opposing-pitcher term — LOO says yes, and LOO is not a gate

The term is partially pooled, non-centered, with the mean fixed at zero (a free
mean is exactly confounded with `league_init`). Its posterior scale is
**`sigma_pitcher` = .249, 94% HDI [.225, .273]**, r-hat 1.00 at ESS 386 —
tightly identified, and close to the .23 the prior was constructed to put its
mass near. On the logit scale .25 is about the gap between a league-average arm
and a good one, so the term is picking up something real rather than absorbing
noise.

PSIS-LOO on the **same cells** (both models built on the pitcher-keyed
partition, so the pointwise log-likelihoods are over identical observations):

| Model | elpd_loo | SE | p_loo | elpd_diff | dSE | weight |
|---|---|---|---|---|---|---|
| with pitcher | −33,210.8 | 136.9 | 463.6 | — | — | .98 |
| no pitcher | −33,491.2 | 138.4 | 67.8 | 280.4 | 24.7 | .02 |

**280 nats at a differential SE of 24.7 — 11.3 dSE.** The term earns its ~396
effective parameters by a wide margin.

**Two things that difference is not.** First, `az.compare` raised the Pareto-k
warning on the with-pitcher model: with ~2 PA per cell and a random effect that
varies within the cell, some observations are highly influential and the
importance sampling behind PSIS is doing badly on them. Second, and more
important, **LOO chooses a specification; only walk-forward scoring clears a
gate** ([methods.md §5](methods.md#5-workflow-by-family)). Leaving out a cell
is not leaving out the rest of a season. The K% table above was run with the
term *off*, because the term joins the cell key — it is the one effect that
varies inside the old (batter, season, team, stand) cell — and that takes the
Aug 1 problem from 2,156 cells to 248,195. Hitter coverage is the one the
walk-forward comparison cannot do without: subsampling batters cuts cells
roughly linearly but barely touches the parameter count (the pitchers stay —
1,152 of them at 40 batters against 1,236 at all 719), and a fit restricted to
the 60 busiest batters can cover at most 60 of the 315 / 231 / 126 hitters the
three cutoffs score — 19% / 26% / 48% in the best case, and measured against
the busiest-batter subsample it is 15% / 19% / 31%. Everyone else falls back to
a population-level projection, so a subsampled arm's MAE would be measuring
that fallback rather than the model.

### One walk-forward cell, and it disagrees with LOO

One cutoff was affordable with the term on at full hitter coverage: **Aug 1,
248,195 cells, 1,260 pitchers, 2,031 parameters, 4 chains × 600 draws (tune
600), 2 hours 44 minutes** — r-hat 1.0160, min ESS 221, 0 divergences, BFMI
.62–.75.

| Aug 1 arm | K% MAE | paired vs `marcel_tuned` | t |
|---|---|---|---|
| `marcel_tuned` (live) | .03412 | — | — |
| `bayes`, pitcher term **off** | .03415 | +.00003 | 0.04 |
| `bayes`, pitcher term **on** | .03615 | +.00203 | 1.28 |

**The term makes the rest-of-season projection worse** — .0020 of MAE, 5.9%,
against run-to-run sampling noise of 4×10⁻⁵. It moves the arm from a dead heat
with the live engine to visibly behind it. The two fits are not matched on
sampling effort (600 draws against 1,500, because the with-pitcher problem is
115× the cells), so treat the exact size as soft; the direction is not soft,
and it is the opposite of what LOO said by 11.3 dSE.

That is worth stating plainly, because it is the concrete version of the rule
[methods.md §5](methods.md#5-workflow-by-family) states in the abstract.
Leaving out a *cell* — two plate appearances against a pitcher whose effect is
estimated from the surrounding cells — is an easy prediction, and the pitcher
term is very good at it. Projecting the rest of a season is a different
question: the projection is made at a neutral pitcher, so everything the term
learned is deliberately discarded at exactly the moment it would be used, while
the extra 1,260 partially pooled parameters take estimation noise out of the
batter terms and put it nowhere useful. **LOO measured the thing the term is
good at and the gate measured the thing we need.**

**So the honest statement is:** the opposing-pitcher term is decisively
favoured by LOO for predicting held-out plate appearances inside the fitted
window, and the one rest-of-season cell we could afford to score walk-forward
says it costs 5.9% of K% MAE. Those are different claims about different
quantities; only the second is a gate, and on the gate the term currently
fails. It should not go into a served projection on the LOO evidence alone.
A full Modal refit should score it at all three cutoffs before the question is
called either way.

## Caveats

- **Every number here is a reduced local fit.** No JAX and no NumPyro in this
  sandbox, so all of it ran on PyMC's own NUTS on 4 cores. The scale is stated
  on every table above. A reduced fit is evidence about a reduced fit.
- **Park factors were neutral.** `data/parquet/park_factors.parquet` was not
  present, so `log_pf_k` entered as 1.0 for every team-year. Marcel has no park
  term either, so this does not favour one arm, but the Bayesian model was
  running without a feature it normally has.
- **K% only.** `src/eval/bayes_arm.py` wraps `src/models/pa_k_rate.py`; the
  other four components are separate models in `modal_functions/app.py` with
  their own denominators, and the provider raises rather than quietly serving
  the K% number under another name. BB%, HR/PA and ISO in the accuracy table
  still have no refit arm.
- **The prior-season universe still differs slightly between the arms.** Marcel
  reads prior seasons from the Stats API table; the Bayesian arm reads them
  from the Statcast-derived PA parquets, which run ~0.7% more PA per player.
  The current-season slice is bit-identical. Rebuilding 2024–25 closed the
  larger version of this gap; this is what is left of it.
- **The 2026 season here ends Sept 1**, so the Aug 1 "rest of season" is one
  month and n = 126.
- **`min_pa = 50` career floor.** Batters under it are projected from the
  fitted population rather than dropped; at these cutoffs that is 2–10% of the
  scored set.

## Reproducing

```
# prior-season PA data (needs R2 credentials; ~290 MB of Statcast in, 8 MB out).
# The pitch-level parquets are only needed once and can be deleted afterwards.
python -c "
from src.data.r2 import get_s3_client, bucket
s3, b = get_s3_client(), bucket()
for y in (2024, 2025):
    s3.download_file(b, f'statcast/statcast_{y}.parquet', f'data/raw/statcast_{y}.parquet')
"
python -c "
from src.data.pa_outcomes_pipeline import build_pa_dataset
build_pa_dataset(years=[2024, 2025])
"
cp data/parquet/pa_outcomes_2026.parquet data/parquet/pa_outcomes/   # the R2 copy
python scripts/build_birthdates.py \
       --pa-parquet data/parquet/pa_outcomes/pa_outcomes_2026.parquet

# the prior predictive alone — seconds, and it is the check worth doing first
python scripts/validate_pa_k_rate.py --cutoff 2026-07-01 \
       --seasons 2024 2025 2026 --max-batters 60 --prior-only --sampler pymc

# convergence + posterior predictive + the LOO ablation (~20 min)
python scripts/validate_pa_k_rate.py --cutoff 2026-07-01 \
       --seasons 2024 2025 2026 --max-batters 60 \
       --draws 300 --tune 500 --chains 2 --sampler pymc --ablation

# the fair fight (~25 min)
python scripts/run_intraseason_backtest.py \
       --components k_rate bb_rate hr_rate iso \
       --bayes --bayes-no-pitcher --bayes-seasons 2024 2025 2026 \
       --bayes-sampler pymc --bayes-draws 1500 --bayes-tune 1500 --bayes-chains 4

# the same-code control: this estimator with 2026 withheld entirely (~10 min)
python scripts/run_intraseason_backtest.py --components k_rate \
       --bayes --bayes-no-pitcher --bayes-seasons 2024 2025 \
       --bayes-sampler pymc --bayes-draws 1500 --bayes-tune 1500 --bayes-chains 4

# the one with-pitcher walk-forward cell (2h44m on 4 cores — Modal for the rest)
python scripts/run_intraseason_backtest.py --components k_rate \
       --cutoffs 2026-08-01 --bayes --bayes-seasons 2024 2025 2026 \
       --bayes-sampler pymc --bayes-draws 600 --bayes-tune 600 --bayes-chains 4

# the accuracy page, from that run
python scripts/build_accuracy_json.py --ros-json <the --json-out from above>
```

Drop `--bayes-sampler pymc` wherever JAX and NumPyro are installed; the
defaults are NumPyro and the full `SAMPLER_KWARGS` in `src/models/pa_k_rate.py`.
