# Level- and spread-matched engine tables in the chain

**BAS-91.** Station E (and C). Follow-up to BAS-90 (`docs/chain-engines.md`).
Pre-registered 2026-09-10 12:25 UTC in the Linear issue, before any run;
this file mirrors that text.

## Why

BAS-90 put the served layer-2 engines into the per-game chain as three
nested rungs and lost on every set: rung 3 (tuned + stuff + contact)
+.00076 Brier vs the served stock-Marcel chain on the 756-game market set,
+.00026 on all 2026, +.00025 on all 2025. The evidence
(`data/eval/chain_engines_stage1.json`) points at two separable
mechanisms, neither of which is "the engines are wrong at player grain":

- **M1, the age curve on an ungated population.** Rung 1 (tuned
  constants) alone costs +.00077; the recalibration control (same
  ballasts and weights, age slopes zeroed) costs +.00023. Station A gated
  the tuned age curve on players with ≥ 100 realised trials; the chain
  applies it to every batter on a club and every arm in the pen, where a
  heavily-ballasted league-mean rate times an age factor moves
  thin-history players off the league mean in a direction the truth does
  not support.
- **M2, spread and level.** Stuff widens the pitcher FIP RA/9 table (sd
  .273 → .323, +18%) and lifts its mean (+.072 runs/9); tuned + contact
  widen the hitter runs-per-PA table (sd .0127 → .0148/.0174) and move its
  mean (−.0031 → −.0091), which `run_environment.team_rs9` reads as a
  level. Every ballast, blend and lineup weight in the chain was chosen
  against the stock tables; the chain's logistic slope on the market set
  goes 1.054 (served) → .927 (stuff) → .944 (contact).

Neither mechanism was tested in isolation. This ticket does that with one
deterministic transform and one switch, no new fitted parameter.

## What gets built

**Matching.** A `match: bool` on `ChainEngines`. When on, after a rung's
table is built at `as_of`, each rate column is centred to the stock
table's weighted mean at the same `as_of` (weights `pa_weighted` for
hitters, weighted BF for pitchers) and its deviations rescaled so the
weighted sd equals the stock table's. Rate clip applied after.
Parameter-free: the stock table is data through `as_of` − 1 on the
identical population, so there is no fit and no leakage.
Missing-in-stock players (none expected: same count frame) keep their
rung value.

**Arms**, all on the 2026 constants and fits BAS-90 used (cell seasons
strictly before the scored season):

| arm | rung | age curve | matched |
|---|---|---|---|
| served | 0 | — | — |
| recal (exists) | 1 | off | no |
| R1-match | 1 | on | yes |
| R3 (exists) | 3 | on | no |
| R3-noage | 3 | off | no |
| R3-match | 3 | on | yes |
| **R3-noage-match** | 3 | off | yes |

R3-noage-match is the pre-named candidate. Scored with
`scripts/run_chain_engines.py`, same three sets: the 756-game market set
(ids frozen in the BAS-90 evidence), all 2026, all 2025; paired Brier, t
on the paired difference, logistic slope, market correlation and residual
as in BAS-90.

## Pre-registered predictions

1. **(M2) Matching restores calibration.** The logistic slope of every
   matched arm is within .03 of served's on each of the three sets
   (market: served 1.054).
2. **(M1) The age curve is a cost the matching does not remove.** At
   matched tables, R3-match − R3-noage-match ≥ +.00030 on the market set,
   same sign on all 2025 and all 2026.
3. **(Ship test) R3-noage-match vs served** ≤ −.00020 on pooled all-2025 +
   all-2026 (≈ 3,990 games) and the same sign on the market set. Power,
   stated up front: the pooled paired se is ≈ .00025, so −.00020 is
   t ≈ −0.8; this test can confirm harm or a large gain, and a null inside
   ±.00025 means the game-grain exam cannot resolve a player-grain gain of
   this size. That null is a finding and gets written as one.
4. **(Null) Matching alone does not rescue rung 1.** R1-match vs served
   ≥ +.00040 on the market set: at rung 1 the cost is the age curve, not
   the spread (the rung 1 pitcher table is narrower than stock, .249 vs
   .273).

### Vacuity check

Mean |ΔP(home)| between R3-noage-match and served > .003 on the market
set.

### Failure conditions

If 1 fails, matching is not the fix and the spread story is wrong;
nothing more is built on it. If 3 fails on sign, nothing ships. If 4
fails (R1-match within +.00010 of served), the doc records that spread
was the mechanism at rung 1 too and M1 is dropped.

## What ships

Only if prediction 3 holds **and** t ≤ −2.0 on the pooled set: the chain
default flips to R3-noage-match, the chain-agreement test pins nightly =
harness, and the served document's provenance names it. Otherwise nothing
ships and the arms stay behind the switch. No blend re-sweep unless
something ships (then 2025 only, as in BAS-90).

## Risks recorded up front

Matching is a linear rescale of a table the chain reads non-linearly
(FIP → RA/9 → Pythagorean), so slope restoration is expected, not
guaranteed. Matching to the stock sd throws away any *correct* extra
spread the engines carry; a gain at player grain that is real but small
will be invisible here (prediction 3's power statement). Closing-line
leakage: compare against the close, never fit on it. Same market-set
caveat as BAS-90 (756 archive set, not the 737 re-pull).

## Results (2026-09-10)

Evidence `data/eval/chain_engines_matched.json`; `match` and `age_slopes`
on `ChainEngines` / `build_engines`; `--engine-match` / `--engine-no-age`
on `scripts/backtest_game_odds.py`; the seven arms, predictions and
verdict in `scripts/run_chain_engines.py`; tests in
`tests/test_sim/test_engines.py` and
`tests/test_scripts/test_run_chain_engines.py`.

**Verdict: nothing ships. A real null.** The matching does what it says
(every matched arm's weighted mean and sd equal stock's on all eight rate
columns to five decimals) and removes essentially all of the engines'
cost, but the candidate arm's gain over the served chain is −.00008 ±
.00015 Brier on the pooled seasons. The game-grain exam cannot resolve a
player-grain gain of this size. No default flipped; the arms stay behind
the switch.

### Reproduction

BAS-90's prediction frames were not on disk, so served, recal and R3 were
re-walked; they reproduce BAS-90 to five decimals on every set (market
.24358 / .24381 / .24435; all-2026 +.00003 / +.00026; all-2025 −.00002 /
+.00025; the 722-game 07-07 cut R3 +.00089, t +1.51). Market set frozen
by id from the BAS-90 evidence, all 756 present.

### Per-arm Brier, Δ vs served, paired

| arm | market (756) | t | all 2026 (1,883) | all 2025 (2,105) | pooled (3,988) | t |
|---|---|---|---|---|---|---|
| served | .24358 | — | .24543 | .24343 | .24438 | — |
| recal | +.00023 | +0.96 | +.00003 | −.00002 | +.00000 | — |
| R1-match | +.00071 | +2.02 | +.00018 | −.00006 | +.00006 | — |
| R3 | +.00076 | +1.31 | +.00026 | +.00025 | +.00026 | +1.04 |
| R3-noage | +.00027 | +0.51 | +.00014 | +.00037 | +.00026 | +1.14 |
| R3-match | +.00064 | +1.45 | +.00002 | −.00005 | −.00002 | −0.09 |
| **R3-noage-match** | +.00017 | +0.50 | −.00013 | −.00003 | **−.00008** | **−0.51** |

Kalshi close .24156, Polymarket .24165, `pythag_60` .24619, unchanged.
On the market set R3-noage-match has the highest correlation with the
market's deviation from `pythag_60` of any arm, .808 (served .795), and a
slope of 1.026 (served 1.054).

Mechanism split of R3's cost: pooled +.00026 → matching alone −.00002,
age off alone +.00026, both −.00008. Matching is what removes the cost on
the full seasons; on the market set it is the age curve (+.00076 →
+.00064 / +.00027 / +.00017).

### Rate-table spread at 2026-08-15 (derived tables)

| arm | FIP RA/9 mean | sd | runs/PA mean | sd |
|---|---|---|---|---|
| served | 4.549 | .273 | −.0031 | .0127 |
| R3 | 4.667 | .323 | −.0045 | .0174 |
| R3-noage | 4.617 | .324 | +.0006 | .0172 |
| R3-match | 4.538 | .272 | −.0037 | .0132 |
| R3-noage-match | 4.538 | .272 | −.0034 | .0131 |

The match is exact on the rate columns; the derived FIP and runs tables
come back near stock but not on it (FIP and the runs map are non-linear
in the rates), the risk recorded up front. R1-match's RA/9 sd is .256,
not .273: rung 1's narrower K spread propagates differently through FIP.

### Predictions

1. **Fails as written.** Slope gaps vs served on the market set: R1-match
   −.067, R3-match −.053, R3-noage-match −.029; the worst exceeds .03.
   The reading is not "the transform failed": the matched arms sit at
   .987–1.026 while served sits at 1.054, so they are closer to 1.0 than
   served is, and the clause measures distance to served. On both full
   seasons every matched arm is within .029 of served, and R3-noage-match
   is within .009 on all three sets.
2. **Fails on the sign clause.** R3-match − R3-noage-match: market
   +.00046 (se .00022, t +2.11), clearing the +.00030 bar; all-2026
   +.00015 (t +1.09); all-2025 −.00002 (t −0.18). Same shape as BAS-90's
   control: the age curve costs on 2026 and nothing on 2025.
3. **Fails.** Pooled R3-noage-match − served −.000076 (se .000149,
   t −0.51) against ≤ −.00020 required, and the market set has the wrong
   sign (+.00017). The realised pooled se is better than the
   pre-registered ≈ .00025, so the test could have resolved ≈ −.0003 at
   t −2; it resolved nothing. That is the pre-registered null, and it is
   a real null, not a dead arm.
4. **Passes.** R1-match − served +.00071 (se .00035, t +2.02) ≥ +.00040
   and not within +.00010 of served. At rung 1 the cost is the age
   curve, not the spread: recal (age off, unmatched) +.00023, R1-match
   (age on, matched) +.00071. M1 stands.

**Vacuity passes.** Mean |ΔP(home)| between R3-noage-match and served
.0078 > .003. The 07-07 → 09-02 cut (722 games) agrees in sign
everywhere (R3-noage-match +.00019; R1-match +.00078, t +2.19; prediction
2 +.00053, t +2.38).

### Failure conditions as fired

Prediction 1 failed, so per the text "matching is not the fix"; read with
the caveat above (the tables match exactly and the matched arms are the
best-calibrated in absolute terms, but not within .03 of served on the
market set). Prediction 3 failed on threshold and sign: nothing ships.
Prediction 4 passed: M1 is not dropped. No blend re-sweep (nothing
shipped).

### What this means

- **For the odds board:** nothing changes. Stock Marcel stays in the
  chain.
- **For the engines:** they are not wrong at game grain; with level and
  spread matched and the age curve off they price games as well as stock
  Marcel on every set, and no better. The 1% MAE gains at player grain are
  below what the per-game Brier can see, on 3,988 games, at se .00015.
- **For the age curve:** applying station A's tuned age curve to the
  chain's whole-roster population costs about +.0005 Brier on 2026 and
  nothing on 2025, in both BAS-90 and here. If a future chain ever runs
  the tuned estimator, it runs it with the age slopes zeroed, or with a
  curve gated on the chain's population, under its own pre-registration.
- **For the exam:** two tickets have now shown the per-game chain cannot
  discriminate between layer-2 engines that differ by 1% of player MAE.
  The lever that moves the game price is not a better rate table; it is
  the terms the market carries that the chain does not (the .002 residual
  to the close), which `docs/market-benchmark-2026.md` and the ledger
  are the place to chase. The engine switch stays in the code, off.
