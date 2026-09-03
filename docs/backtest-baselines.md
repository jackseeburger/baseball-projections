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
throwing away April. Second, **it is bigger than our model's edge**:
`marcel` beats `bayes_preseason` on K% by 6% at May 1 and 11% at Aug 1, and
`bayes_preseason` never beats `marcel_preseason` on K% either (a tie at May 1,
5% behind at Jul 1 and Aug 1) — so the Bayesian components buy nothing over
Marcel on the same information, while the *same* information advantage that
in-season data provides is larger than the entire gap between them. The route to a
better rest-of-season number runs through *ingesting the current season*, not
through a fancier prior. It also matters **how** you ingest it:
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
  It is *not* an answer to "how good would our model be if refit on July 1";
  that is what this harness exists to judge once the Modal refits land.
- **PA-derived seasons are not identical to the Stats API totals.** The
  Statcast universe runs ~0.7% more PA per player (mean +1.8 PA over 2026;
  AB, K, BB, HR all within a few counts). Training mixes the two — prior
  seasons from the API table, the partial current season from PA data — so a
  hair of the in-season increment could be universe drift rather than signal.
  Rebuilding prior seasons from `pa_outcomes_<year>.parquet` would close it;
  only 2026 exists in R2 today.
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
