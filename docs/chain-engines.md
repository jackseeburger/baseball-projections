# The layer-2 engines into the per-game chain

**BAS-90.** Station E (and C). Pre-registered 2026-09-10 10:30 UTC in the
Linear issue, before any run; this file mirrors that text.

## Why

The per-game chain (station E: starter, lineup, bullpen terms; station C:
the bottom-up run environment) prices every game from **stock Marcel** on
both sides: `src/sim/starters.marcel_rates` passes stock constants (5/4/3,
2× stabilization ballasts, no age curve) and `src/sim/lineups.marcel_rates`
is an in-module stock Marcel. Neither `marcel_tuned` /
`marcel_pitcher_tuned` nor the served engines (`contact_additive` on all
five hitter components, `stuff_additive` on BB/BF and HR/BF) reach the
chain, and `run_playoff_odds.py` does not read the served projection
document. The chain that sits the money exam runs a generation behind the
player pages. `docs/journey.md` names this the largest unclosed gap to the
market and the untested lever.

Where it stands (`docs/market-benchmark-2026.md`, 756-game common set,
2026-07-04 → 09-02, Brier): Kalshi close .24156, served chain
`pythag_C_sp_bpa_ip_lvl` .24358, gate baseline .24388, `pythag_60`
.24619. Market residual over the served chain .00163 (se ≈ .0013).

## What gets built

Three nested rungs on `ChainConfig`, each reproducing the rung below at
its off setting (stock params reproduce today's column bit for bit):

1. `tuned`: pitcher table on `marcel_pitcher_tuned` params
   (`src/eval/marcel_pitcher_params.json`, ages from
   `data/parquet/birthdates.parquet`); hitter table on `marcel_tuned`
   params.
2. `+stuff`: `stuff_additive` correction on the served pitcher components
   (BB/BF, HR/BF), fitted on cell seasons strictly before the scored
   season, features through the last month boundary before `as_of`
   (`assert_month_boundary`, never rounded forward).
3. `+contact`: `contact_additive` on the five hitter components, same
   cutoff rule.

One switch in `starters.marcel_rates` / `lineups.marcel_rates`, shared by
`run_playoff_odds.py` and `scripts/backtest_game_odds.py` through
`game_model`.

**Cells.** 2025 (all games; constants from ≤ 2024) and 2026 (all games,
and the market set frozen at the pre-registration date: the 737-game
Kalshi re-pull 07-07 → 09-02). Constants never chosen on 2026. Compare on
`_lvl`, not `_lvl_lu` (the nightly runs before cards post).

**Comparators.** The served chain `pythag_C_sp_bpa_ip_lvl`; Kalshi and
Polymarket closes; `pythag_60`; and a recalibration control (rung 1 with
tuned ballasts but age slopes zeroed) to separate ballast change from
covariate.

## Pre-registered predictions

Paired per-game Brier, t on the paired difference.

1. Rung 3 vs the served chain: **≤ −.00020 on the 737 market games**, and
   the same sign on all 2026 games and all 2025 games.
2. The pitcher side (rung 2 − rung 1) carries at least half the gain: the
   starter term is the largest term and stuff is served on HR/BF, which
   FIP weights 13×.
3. The correlation of our deviation from `pythag_60` with the market's
   rises from .778 to ≥ .79, and the market residual falls below .00140.
4. The top bucket (P(home) > 0.65) does not get more overconfident:
   realised − predicted no worse than today's.
5. (Null) Rung 1 alone moves the 737-game Brier by < .00010: tuned
   constants are a 1% MAE gain at player grain, tiny at game grain.

### Vacuity check

Mean |ΔP(home)| between rung 3 and the served chain must exceed .003 on
the 737 games (the park term moved .0005 and was declared inert); below
that the test cannot resolve anything and says so.

### Failure conditions

Prediction 1 fails, or the sign disagrees across the three sets, or the
recalibration control matches rung 3 within .00005 (the gain was the
ballasts, not the engines): nothing ships, the engines stay behind the
chain, and this file says why.

## What ships

Only if prediction 1 holds and the sign replicates on 2025 and all of
2026: the default params in `starters.py` / `lineups.py` flip to the
served engines, the chain-agreement test pins nightly = harness, and the
served document's provenance names the engines the chain used. The blend
weight between the bottom-up and top-down halves is re-swept on 2025 only
and reported, never on 2026.

## Risks recorded up front

Closing-line leakage (compare against the close, never fit on it); lineup
posting times (`_lvl`, not `_lu`); month-boundary features; the 2026
market set is two months (se ≈ .0013); archive drift (freeze the game
set); double-counting of stuff and contact already inside the top-down
half via actual runs.
