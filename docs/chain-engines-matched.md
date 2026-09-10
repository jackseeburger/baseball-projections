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
