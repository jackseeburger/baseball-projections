# Baseline Backtest Scores (roadmap 0.3, baseline half)

Two halves: **season-level** splits (below) and the **intra-season
walk-forward** at date cutoffs ([jump](#intra-season-walk-forward--rest-of-2026-rates)),
which is the one that judges rest-of-season projections.

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
