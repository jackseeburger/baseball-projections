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

## Results (2026-09-10)

Evidence `data/eval/chain_engines_stage1.json`; runner
`scripts/run_chain_engines.py`; the switch is `src/sim/engines.py`,
`ChainConfig.engines`, `engine=` on `starters.marcel_rates` /
`lineups.marcel_rates`, `--engine-rung` on `scripts/backtest_game_odds.py`.

**Verdict: nothing ships.** The engines make the game price worse on every
set. Prediction 1 fails with the wrong sign, four times the bar in the
wrong direction; predictions 2–5 fail with it; the vacuity check passes
(the rungs move the price, .013 mean |ΔP(home)|, and move it the wrong
way); the recalibration control does not match rung 3, so the failure is
not "the ballasts". The rungs stay off; the switch stays in the code for
whichever ticket earns it. No default changed.

### The market set

The pre-registration named a 737-game Kalshi re-pull; that re-pull is not
on disk (`data/parquet/` is gitignored) and pulling again would make a
third set, not that one. The set scored is the archive the
pre-registration's own reference numbers come from: 756 games priced by
both exchanges, 2026-07-04 → 09-02, reproduced to five decimals
(.24358 / .24388 / .24619 / .24156). The same archive cut to the
pre-registered 07-07 → 09-02 window is 722 games and gives the same
verdict, slightly stronger (rung 3 +.00089, t +1.51; rung 1 +.00087,
t +2.21). Both id sets are in the evidence.

### Per-rung Brier

756 market games, Δ vs the served chain, paired, t on the difference:

| rung | Brier | log loss | Δ | se | t |
|---|---|---|---|---|---|
| served (0) | .24358 | .68008 | — | — | — |
| tuned (1) | .24435 | .68166 | +.00077 | .00039 | +1.99 |
| +stuff (2) | .24454 | .68200 | +.00096 | .00058 | +1.66 |
| +contact (3) | .24435 | .68160 | +.00076 | .00058 | +1.31 |
| recalibration control | .24381 | .68055 | +.00023 | .00024 | +0.96 |
| `pythag_60` | .24619 | .68554 | +.00261 | .00172 | +1.51 |
| Kalshi close | .24156 | .67589 | −.00202 | .00128 | −1.58 |
| Polymarket close | .24165 | .67611 | −.00193 | .00122 | −1.58 |

All 2026 (1,883 games): served .24543, tuned +.00020 (t +0.83), +stuff
+.00012, +contact +.00026 (t +0.73), recalibration +.00003.
All 2025 (2,105 games): served .24343, tuned −.00012 (t −0.57), +stuff
+.00012, +contact +.00025 (t +0.74), recalibration −.00002.

### Predictions

1. **Fails.** Rung 3 vs served +.00076 on the market games against
   ≤ −.00020 required. Same wrong sign on all 2026 (+.00026) and all 2025
   (+.00025).
2. **Fails.** There is no gain to split. Pitcher side (rung 2 − rung 1)
   +.00019, hitter side (rung 3 − rung 2) −.00020; the two corrections
   cancel and rung 3 ≈ rung 1.
3. **Fails both clauses.** Correlation with the market's deviation from
   `pythag_60` falls, .795 → .781 (tuned .778, stuff .779); the market
   residual rises, .00202 → .00279 against < .00140 required. The one arm
   that improves it is the recalibration control, .804.
4. **Fails as written, uninformative.** The top bucket is under-confident
   in both arms (served 32 games, .680 predicted / .813 realised; rung 3
   40 games, .680 / .775). Rung 3 is less under-confident, which the
   clause scores as "more overconfident". At 32–40 games it is not a
   finding either way.
5. **Fails.** Rung 1 alone moves the market Brier +.00077 (t +1.99)
   against < .00010 predicted. Tuned constants are not tiny at game grain;
   they are the largest single engine effect on 2026.

Recalibration control (tuned ballasts and weights, age slopes zeroed):
+.00023 where rung 1 is +.00077, so about 70% of rung 1's cost on 2026 is
the **age curve**, not the re-ballasting. On 2025 both are ≈ 0.

### Blend sweep (2025 only, as pre-registered)

Bottom-up vs top-down weight w ∈ {0, .25, .5, .75, 1}: served chain
.24418 / .24367 / **.24343** / .24345 / .24376; rung 3 .24441 / .24391 /
**.24368** / .24372 / .24403. Argmin .5 in both arms, rung 3 worse at
every weight. The w = .5 cell reproduces the main 2025 served run bit for
bit.

### Why it gets worse

- **Spread.** The engines widen both rate tables: on 2026-08-15 the
  pitcher FIP RA/9 sd goes .273 → .323 (+18%) and the hitter
  runs-above-average-per-PA sd .0127 → .0174 (+37%). Every ballast, blend
  weight and lineup weight in the chain was chosen against the narrower
  tables and nothing was re-tuned; the chain's logistic recalibration
  slope moves with it on all three sets (market 1.054 → .944, all-2026
  .930 → .882, all-2025 .881 → .849). `docs/contact-quality.md` §8 already
  measured `marcel_tuned`'s spread on ISO and HR/PA at roughly twice what
  it should be (free-fit baseline coefficients .55 and .49) and
  `contact_additive` deliberately does not correct it. This is the
  mechanism.
- **Level leak, real but not the mechanism.** The tuned age curve moves the
  priced population's mean runs-above-average from −.0031 to −.0091 per PA
  (recalibration: −.0040), and `run_environment.team_rs9` reads the table
  as a level, so its "cannot move the league" invariant only holds while
  the table is centred. Stuff's intercept lifts mean FIP by .072 runs/9
  the same way. Mean P(home) shifts ≤ .0004 across every rung and set.
- **Population mismatch.** Station A's gates cleared these engines on
  players with ≥ 100 realised trials at May/July/August cutoffs; the chain
  applies them to every batter on a club and every arm in the pen from
  opening day. The ladder in the evidence puts rung 1's cost at +.00011 on
  `pythag_60_sp` (announced starters) and +.00064 on `pythag_C` (the whole
  roster) on the market games, same shape on all-2026, not replicated on
  2025.
- **Double counting.** One corrected pitcher rate enters three places
  (announced starter, rotation slot, pen), one corrected hitter rate two
  (posted card, station C runs scored), and the top-down half of station C
  is actual runs, which already contains whatever the covariates measure.
  Recorded up front; still unquantified.

### Things that qualify the reading

- **BB vs BB+HBP.** Station E's FIP consumes (BB+HBP)/BF; the site serves
  `stuff_additive` on BB/BF. The stuff correction was fitted on
  `p_bbhbp_rate`, the rate the chain consumes; the two fits are
  near-identical (intercepts −.00317 vs −.00304, every coefficient within
  15%). The covariate-only share of that arm is −0.64% at t −1.86, below
  the serving floor, so most of what the chain is handed there is
  recalibration of the pitcher Marcel, and the chain has no way to absorb
  a level.
- **`league_mode` never fires.** Station E hands in its own league rate,
  so the tuned params' `weighted3` mode has no effect; "tuned" here means
  ballast + recency weights + age curve. Identical across rungs.
- **Feature lag.** Covariates are read at the last month boundary ≤
  `as_of` (`assert_month_boundary`, `assert_window_clean`) and lag by up
  to 30 days while the Marcel rates on the same row are current to the
  previous day. Fits are on cell seasons strictly before the scored
  season (2026: 24 cells over 2017–2025; 2025: 21 over 2017–2024). Tuned
  params were fitted on 2020–2024.
- **Bit-for-bit reproduction.** `tests/test_sim/test_engines.py` asserts
  the default slate's `sp_ra9` and `runs_lookup` are exactly the
  pre-switch values; a rung-0 walk-forward of 2026 reproduces the
  reference Briers to five decimals. The tuned path at stock constants
  reproduces to ~1e-15 relative (harness accumulates `Σ wᵢnᵢ` against
  `ballast·mean(w)`, the chain `Σ(wᵢ/w₀)nᵢ` against `ballast`).
- **A scorer bug fixed on the way.** `backtest_game_odds.walk_forward`
  emits a few repeated `game_pk`s (1 in 2026, 4 in 2025; suspended games
  on the schedule twice), and joining rungs on `game_pk` inflated the sets
  to 1,913 / 2,225. The scorer aligns rungs by row position with keys
  asserted equal.
- `IP_LEVEL_RUNS = 0.9` was fitted on 2015–2025, which includes the 2025
  cell; pre-existing, identical across rungs, cancels in every paired
  difference. All-2026 is 1,883 games through 09-09.

### What this means for the odds board

Nothing changes on the site: the nightly odds keep pricing from stock
Marcel, which is still the best chain we have against the close. The
untested lever turned out to be a lever in the wrong direction until the
chain's downstream constants are re-tuned to the engines' spread. The
next step is not another engine; it is a **spread-aware chain**: re-fit
the ballasts, blend and lineup weights against the engine tables on 2025
only, or shrink the engine tables toward the stock spread with a single
pre-registered factor, and score rung 3 again. Either goes under its own
pre-registration.
