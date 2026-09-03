# Station B-pitchers — Projected batters faced (and innings)

**Scored Sept 3, 2026.** Reproduce with
`python3 scripts/build_pitcher_workload.py --fetch --all` (the inputs; they
are committed under `data/workload/`, so this is only needed to refresh
them), then
`python3 scripts/run_pitcher_workload_backtest.py` (the batters-faced table),
`--unit outs` (innings), `--seasons 2024 2025 2026` (the holdout on its own),
`--sweep` and `--calibrate --method structural` (the two selection runs, both
on 2022-2023 only). Code: `src/projections/pitcher_workload.py` (the math,
pure functions over DataFrames), `src/projections/pitcher_ros.py` (the served
projection, called by the harness rather than copied),
`src/projections/il_returns.py` (station B's return-time distribution, reused
unchanged), `scripts/build_pitcher_workload.py` (assembly).

## 0. Why this exists

The pitcher line on the site is a rate times a workload:

```
rest-of-season K = projected K/BF  x  projected batters faced
```

The left half went through the [gate](architecture.md#3-the-gate-rule) on
Sept 3, 2026 — five components, five walk-forward cells, three baselines
each. The right half never had. It was stamped `structural` in the served
document, which was an honest label for "nobody has scored this against
anything", and it made **every counting stat the site publishes for a
pitcher ungated**: the multiplicand was measured and the multiplier was
asserted.

This document is the missing score. It is also a test of whether station B's
one large finding ports. Station B's biggest single gain — larger than any
modelling sophistication on the rate side — came from *not* zeroing an
injured hitter but projecting him at his pre-injury share times an expected
return fraction: worth 6.4 PA a hitter at two months, t −5.4
([playing-time.md §5](playing-time.md#5-expected-returns-instead-of-a-hard-zero)).
Pitchers have all of that plus a rotation slot, a turn every fifth day, an
innings limit and a more common injury pattern. If workload modelling pays
anywhere, it should pay here.

**It does not, and the reason is interesting.** The served projection wins,
every candidate built to beat it loses, and the *one* term inside it that
carries station B's information — knowing who is unavailable this morning —
is by a wide margin the largest thing in the model. What does not port is
station B's actual fix: for pitchers, the injured are better projected near
zero than at their pre-injury usage discounted by a return date.

## 1. The design

**Unit.** The harness projects a column, and runs twice: `bf` (batters faced,
what the site multiplies the rates by) and `outs` (whose third is innings).
Innings are therefore scored as their own model rather than read off batters
faced through a league constant, which is what the served document does.

**Cutoffs.** The 1st and the 15th of each month from May to September —
biweekly as-of dates — in 2022, 2023, 2024, 2025 and 2026. A cutoff with
fewer than 14 days of horizon is dropped, which is what trims 2026 to eight
(its data ends 2026-09-02). **44 cutoffs**, horizons from 25 to 143 club
games remaining. 2020 is excluded everywhere: a 60-game season has no
rest-of-season horizon worth projecting and its injured-list spells censor at
a length no other year has.

**Scoring.** Each method projects at a cutoff from rows strictly before it,
and is scored on what each pitcher actually did from the cutoff through the
last day of that season's data. Every method at a cutoff is scored on the
**same pitcher universe** — the union of everyone who faced a batter in the
window and everyone any method projects above zero — so nobody is rewarded
for declining to project someone.

**The gate.** Every constant the candidate methods carry was chosen on **2022
and 2023** and frozen; **2024, 2025 and 2026** are the score. Both halves are
printed below, because a candidate that wins its own fitting seasons and
loses the holdout is a finding too.

**Standard errors.** Nine cutoffs inside a season score the same pitcher nine
times, so a pitcher-level standard error would be far too small. The paired
tables below take each **cutoff** as the unit and cluster across **seasons**:
the mean of the per-cutoff mean paired differences, with the SE taken across
the season means. That is the same convention
[station G's backtest](team-projection-backtest.md) uses, and it is
conservative — three seasons on the holdout means two degrees of freedom.

## 2. The methods

Six baselines and four candidates. The served projection is deliberately
listed among the candidates rather than treated as an incumbent that only has
to be defended.

| Name | What it is | Whose information |
|---|---|---|
| `zero` | nobody pitches again | none |
| `last_season` | last season's total x (games left / 162) | prior season only |
| `season_rate` | season-to-date workload per club game x games left | this season, no role, no regression, no roster |
| `recent_rate` | the same over the trailing 30 days | this season, recent only |
| `structural` | **the projection the site serves**, called through `pitcher_ros.projected_batters_faced` | usage + role + the 40-man |
| `structural_nogate` | the same with station B's expected-return fractions removed | usage + role |
| `blend` | role from `gamesStarted` rather than a batters-faced threshold; appearance rate blended across the trailing window and the season with a fitted horizon weight; both halves regressed toward the **league's own role averages at the cutoff** instead of frozen constants; the unavailable zeroed | usage + role + the 40-man |
| `blend_il` | **station B's fix, ported**: an unavailable pitcher weighed as he was the day he went out, scaled by the fraction of the horizon he is expected back for | + the transaction feed |
| `blend_il_share` | **station B's normalization, ported**: a club's staff must between them face the club's opponents, so every staff is scaled to the club's own projected total | + the club total |
| `structural_cal` | the served projection times one constant per role, fitted on 2022-2023 | usage + role + the 40-man |

`structural_cal` deserves a sentence. Rest-of-season workload has a long left
tail — a pitcher who tears something in August faces nobody — and MAE is
minimized at a conditional *median*, so a projection built as an expectation
sits above the number the metric wants. One constant per role is the cheapest
possible correction and the first thing to try before anything structural.
On 2022-2023 the MAE-minimizing constants are **0.90 for starters and 0.87
for relievers**, which says the served projection is about a tenth too high
for the metric it is scored on.

**The served projection is called, not copied.** `structural` runs
`pitcher_ros.projected_batters_faced`, the same function the nightly builder
runs, so the harness cannot score a model the site does not serve. The one
substitution is the input: the live builder aggregates the PA-level parquet
and the harness aggregates the Stats API pitching game log. On 2026-08-01
those two agree on **772 pitchers, 125,041 batters faced league-wide to the
digit**, with 17 pitchers differing by one or two batters (a pitching change
in the middle of a plate appearance) and games played never differing by more
than one.

## 3. The score — batters faced

Pooled over the 26 holdout cutoffs (2024, 2025, 2026), on **22,807
pitcher-projections**. MAE and RMSE in batters faced per pitcher; `wMAE`
weights by realized workload (the `src/eval/metrics` convention — a rotation
starter's miss counts for more than a September call-up's); `bias` is mean
signed error, negative meaning the method projects less work than happens;
`top-5` is the share of realized club workload taken by the five pitchers
each method ranked highest for that club. Lower is better except `top-5`.

<!-- TABLE-HOLDOUT-BF -->

<!-- TABLE-PAIRED-BF -->

## 4. Innings

<!-- SECTION-INNINGS -->

## 5. Does station B's insight port?

<!-- SECTION-PORT -->

## 6. Starters and relievers

<!-- SECTION-ROLE -->

## 7. What this changes

<!-- SECTION-CHANGES -->

## 8. Leakage guards

Every method takes the **whole** season's appearance log and filters it on the
cutoff itself, which means a method that forgets to filter still runs and
quietly scores brilliantly. `tests/test_projections/test_pitcher_workload.py`
closes that with a synthetic two-club season in which every post-cutoff
appearance is rewritten to 9,999 batters faced, a pitcher who does not exist
before the cutoff is added after it, and every one of the ten methods is
required to be bit-identical to the same method run on the pre-cutoff rows
alone — on both units. A companion test asserts the poison would have been
visible: the unfiltered season totals reach 40,000 batters faced and no
projection exceeds 1,000.

Three other guards are pinned by their own tests:

- `window_totals` excludes a game played *on* the cutoff. A game that starts
  the evening of the day the projection is made has not finished.
- `realized` includes both ends of the scored window, so the cutoff day
  itself is scored rather than falling between the two.
- `blend_il` reads an unavailable pitcher's usage as of the day he went out,
  which is the only place in the model where a window ends somewhere other
  than the cutoff. The spell date is **clamped at the cutoff**, so a
  transaction the feed dates in the future cannot walk that window forward
  into games that have not been played. The test proves it by handing the
  poisoned season a spell start at the end of the season and requiring the
  projection not to move.

The return-time distribution has the same walk-forward discipline station B
gives it: fitted on the three seasons *before* the one being projected (minus
2020), and read against spells this season's transaction feed opened strictly
before the cutoff.

## 9. What would improve it, in order

<!-- SECTION-NEXT -->
