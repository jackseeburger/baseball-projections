# Baseline Backtest Scores (roadmap 0.3, baseline half)

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
