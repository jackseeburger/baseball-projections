# Station B-P — Projected batters faced (and innings) for pitchers

**Scored Sept 3, 2026.** Reproduce with
`python3 scripts/build_pitcher_workload.py --fetch --all` (the inputs — they
are committed under `data/workload/`, so this is only needed to refresh them),
then `python3 scripts/run_pitcher_workload_backtest.py` (the batters-faced
tables), `--unit outs` (innings), `--seasons 2024 2025 2026` (the holdout on
its own), and `--sweep`, `--calibrate --method structural`,
`--calibrate-hazard` (the three selection runs, all on 2022–2023 only). Code:
`src/projections/pitcher_workload.py` (the math, pure functions over
DataFrames), `src/projections/pitcher_ros.py` (the served projection, *called*
by the harness rather than copied), `src/projections/il_returns.py` (station
B's return-time distribution, reused unchanged),
`scripts/build_pitcher_workload.py` (assembly).

## 0. Why this exists, and the answer in one paragraph

The pitcher line on the site is a rate times a workload. The rates went
through the [gate](architecture.md#3-the-gate-rule) on Sept 3, 2026. The
workload never had: it was stamped `structural` in the served document, which
was an honest label for "nobody has scored this against anything", and it made
every counting stat the site publishes for a pitcher ungated — the
multiplicand measured, the multiplier asserted.

**It is scored now, and it wins.** Over 26 walk-forward as-of dates in
2024–2026, on 22,807 pitcher-projections, the arithmetic already in production
beats a season-to-date rate extrapolation by **5.6 batters faced a pitcher
(t −16.7)**, a trailing-30-day one by 4.9 (−12.1), last season prorated by
22.1 (−19.8) and projecting nobody at all by 47.7 (−19.9). Three models
built specifically to beat it lose. The largest single term inside it is station B's
injured-list machinery, worth 4.4 batters faced a pitcher (t −37.8) and 9.2
for a starter. What does **not** carry over is station B's own headline
finding — projecting the injured at their pre-injury usage times an expected
return fraction, which was worth −6.4 PA a hitter, costs **+2.9 batters faced
a pitcher (t +8.6)** against the same model without it, and costs most at
exactly the horizons where it paid most for hitters.
The one candidate that beats the served model, an attrition haircut on the
*healthy*, is reported in §7 and is not wired, for a reason given there with
its numbers.

The `structural` stamp is gone. `batters_faced_method` in
`public/data/projections/latest.json` reads `recent_usage`, and the sentence
the site prints under the pitcher table carries the margin instead of the
disclaimer.

## 1. The design

**Unit.** The harness projects a column and runs twice: `bf` (batters faced,
what the site multiplies the rates by) and `outs` (whose third is innings).
Innings are scored as their own model rather than read off batters faced
through a league constant, which is what the served document does.

**Cutoffs.** The 1st and the 15th of each month from May to September —
biweekly as-of dates — in 2022, 2023, 2024, 2025 and 2026. A cutoff with fewer
than 14 days of horizon is dropped, which is what trims 2026 to eight (its
data ends 2026-09-02). **44 cutoffs**, horizons from 25 to 143 club games
remaining. 2020 is excluded everywhere: a 60-game season has no
rest-of-season horizon worth projecting and its injured-list spells censor at
a length no other year has.

**Scoring.** Each method projects at a cutoff from rows strictly before it and
is scored on what each pitcher actually did from the cutoff through the last
day of that season's data. Every method at a cutoff is scored on the **same
pitcher universe** — the union of everyone who faced a batter in the window
and everyone any method projects above zero — so nobody is rewarded for
declining to project someone.

**The gate.** Every constant the candidate methods carry was chosen on **2022
and 2023** and frozen; **2024, 2025 and 2026** are the score. Both halves are
printed below.

The served model has no *fitted* constants — nothing in it was chosen by a
selection run, and it is the same arithmetic it was before this document
existed — but it is not innocent of 2026 either: its role averages are
commented "from 2026 league usage" and its ballasts and recency weight were
picked by hand while looking at a 2026 board. **2024 and 2025 are therefore
the cleanest read on it**, and they say the same thing as 2026 does (MAE 49.19
and 49.79 against `season_rate`'s 54.53 and 56.00), so the conclusion does not
rest on the one season where the served model had a hand on the scale.

**Standard errors.** Nine cutoffs inside a season score the same pitcher nine
times, so a pitcher-level standard error would be far too small. The paired
tables take each **cutoff** as the unit and cluster across **seasons**: the
mean of the per-cutoff mean paired differences, with the SE taken across the
season means. That is the convention
[station G's backtest](team-projection-backtest.md) uses, and it is
conservative — three seasons on the holdout is two degrees of freedom.

## 2. The methods

Six baselines and five candidates. The served projection is listed among the
candidates rather than treated as an incumbent that only has to be defended.

| Name | What it is | Whose information |
|---|---|---|
| `zero` | nobody pitches again | none |
| `last_season` | last season's total × (games left / 162) | the prior season only |
| `season_rate` | season-to-date workload per club game × games left | this season; no role, no regression, no roster |
| `recent_rate` | the same over the trailing 30 days | this season, recent only |
| **`structural`** | **the projection the site serves**, through `pitcher_ros.projected_batters_faced` | usage + role + the 40-man |
| `structural_nogate` | the same with station B's expected-return fractions removed | usage + role |
| `blend` | role from `gamesStarted` rather than a batters-faced threshold; appearance rate blended across the trailing window and the season with a fitted horizon weight; both halves regressed toward the **league's own role averages at the cutoff** instead of frozen constants; the unavailable zeroed | usage + role + the 40-man |
| `blend_il` | **station B's fix, ported**: an unavailable pitcher weighed as he was the day he went out, scaled by the fraction of the horizon he is expected back for | + the transaction feed |
| `blend_il_share` | **station B's normalization, ported**: a club's staff must between them face the club's opponents, so every staff is scaled to the club's own projected total | + the club total |
| `structural_cal` | the served projection × one constant per role, fitted on 2022–2023 | usage + role + the 40-man |
| `structural_hazard` | the served projection × a per-role constant hazard of a *healthy* pitcher losing the rest of his season, fitted on 2022–2023 | usage + role + the 40-man |

The last two exist because rest-of-season workload has a long left tail — a
pitcher who tears something in August faces nobody — so MAE is minimized at a
conditional *median* and a projection built as an expectation sits above the
number the metric wants. The flat multiplier is the crudest possible
correction; the hazard is the same idea with the right shape (§7). On
2022–2023 the MAE-minimizing multipliers are **0.90 for starters and 0.87 for
relievers**, and the MAE-minimizing hazards **0.0020 and 0.0030 per club
game**.

**The served projection is called, not copied.** `structural` runs
`pitcher_ros.projected_batters_faced`, the same function the nightly builder
runs, so the harness cannot score a model the site does not serve. The one
substitution is the input: the live builder aggregates the PA-level parquet
and the harness aggregates the Stats API pitching game log. On 2026-08-01
those two agree on **772 pitchers and 125,041 batters faced league-wide to the
digit**, with 17 pitchers differing by one or two (a pitching change in the
middle of a plate appearance) and appearances never differing by more than
one. Running the served function on *outs* additionally requires putting outs
on the batters-faced scale its constants are written in — `STARTER_MIN_BF` is
twelve **batters** — which the harness does by dividing by the league's outs
per batter faced and multiplying back.

## 3. The score — batters faced

MAE and RMSE in batters faced per pitcher. `wMAE` weights by realized workload
(the `src/eval/metrics` convention: a rotation starter's miss counts for more
than a September call-up's). `top-5` is the share of realized club workload
taken by the five pitchers each method ranked highest for that club — "did you
pick the rotation?" separated from "did you get the counts right?".
`proj/actual` is the league total projected over the league total realized,
pooled. Lower is better except `top-5`, and `proj/actual` wants 1.

#### The holdout: 2024, 2025, 2026 — 26 cutoffs, n = 22,807

| Method | MAE | RMSE | wMAE | SP MAE | RP MAE | top-5 | proj/actual |
|---|---|---|---|---|---|---|---|
| `zero` | 93.27 | 136.74 | 200.49 | 205.67 | 59.99 | .259 | 0.000 |
| `last_season` | 67.62 | 99.01 | 94.84 | 118.89 | 52.14 | .383 | 0.867 |
| `season_rate` | 51.14 | 78.53 | 70.58 | 92.25 | 38.80 | .457 | 0.973 |
| `recent_rate` | 50.44 | 78.96 | 71.26 | 91.61 | 38.18 | .473 | 0.983 |
| `structural_nogate` | 49.96 | 75.28 | 61.94 | 88.10 | 38.55 | .472 | 1.060 |
| `blend_il` | 49.75 | 71.10 | **60.45** | 82.19 | 40.12 | .481 | 1.072 |
| `blend_il_share` | 48.14 | 68.44 | 61.50 | 79.02 | 38.88 | .481 | **1.011** |
| `blend` | 46.56 | 75.13 | 71.41 | 82.62 | 35.89 | **.485** | 0.829 |
| **`structural`** (served) | 45.57 | 70.91 | 64.88 | 78.85 | 35.67 | .484 | 0.943 |
| `structural_cal` | 44.22 | 68.37 | 66.95 | 75.72 | 34.79 | .484 | 0.837 |
| `structural_hazard` | **44.05** | **68.26** | 66.09 | **75.36** | **34.70** | .484 | 0.848 |

#### The fitting seasons: 2022, 2023 — 18 cutoffs, n = 15,889

| Method | MAE | RMSE | wMAE | SP MAE | RP MAE | proj/actual |
|---|---|---|---|---|---|---|
| `zero` | 105.60 | 154.56 | 226.24 | 234.10 | 67.73 | 0.000 |
| `last_season` | 72.44 | 107.40 | 103.74 | 130.39 | 54.94 | 0.837 |
| `recent_rate` | 58.49 | 91.95 | 86.84 | 106.00 | 44.24 | 0.973 |
| `blend_il` | 57.83 | 84.81 | 76.41 | 97.67 | 46.03 | 1.027 |
| `season_rate` | 57.83 | 89.76 | 83.01 | 103.77 | 44.05 | 0.961 |
| `blend_il_share` | 57.80 | 84.26 | 78.84 | 97.67 | 45.77 | 1.011 |
| `structural_nogate` | 57.33 | 87.46 | 75.21 | 100.87 | 44.39 | 1.049 |
| `blend` | 55.35 | 90.11 | 88.40 | 100.34 | 42.00 | 0.804 |
| **`structural`** (served) | 53.89 | 85.34 | 81.08 | 95.09 | 41.65 | 0.919 |
| `structural_cal` | 52.10 | 82.59 | 83.00 | 91.60 | 40.29 | 0.816 |
| `structural_hazard` | 51.96 | 82.39 | 82.70 | 91.27 | 40.21 | 0.817 |

The ordering is the same in both halves, and in every individual season:

| Season | `zero` | `last_season` | `recent_rate` | `season_rate` | `structural_nogate` | **`structural`** | `structural_hazard` | `structural_cal` | `blend` | `blend_il` | `blend_il_share` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 106.90 | 70.37 | 56.78 | 55.55 | 55.90 | **53.29** | 51.58 | 51.75 | 55.07 | 57.34 | 57.81 |
| 2023 | 104.30 | 74.50 | 60.20 | 60.10 | 58.76 | **54.48** | 52.33 | 52.46 | 55.63 | 58.31 | 57.80 |
| 2024 | 99.65 | 72.55 | 54.61 | 54.53 | 53.61 | **49.19** | 47.43 | 47.52 | 50.30 | 54.48 | 52.36 |
| 2025 | 99.11 | 72.56 | 54.81 | 56.00 | 54.35 | **49.79** | 48.05 | 48.18 | 50.84 | 53.34 | 51.96 |
| 2026 | 79.52 | 56.53 | 40.83 | 41.87 | 40.93 | **36.76** | 35.76 | 36.03 | 37.56 | 40.41 | 39.10 |

(2026 is lower everywhere because its horizons are shorter: its data ends
2026-09-02, so the last two cutoffs are dropped and the rest are scored over
less season.)

### The gate, paired

Every method sees the same pitchers at the same cutoff, so the difference in
MAE is a paired quantity. Negative means the first method is better.

| Comparison | Role | n | mean diff | SE | t |
|---|---|---|---|---|---|
| `structural` − `zero` | all | 22,807 | **−47.70** | 2.40 | **−19.9** |
| `structural` − `zero` | SP | 5,519 | −126.81 | 3.71 | −34.2 |
| `structural` − `zero` | RP | 10,798 | −41.48 | 2.88 | −14.4 |
| `structural` − `last_season` | all | 22,807 | **−22.05** | 1.11 | **−19.8** |
| `structural` − `last_season` | SP | 5,519 | −40.03 | 1.44 | −27.8 |
| `structural` − `last_season` | RP | 10,798 | −15.25 | 0.54 | −28.0 |
| `structural` − `recent_rate` | all | 22,807 | **−4.87** | 0.40 | **−12.1** |
| `structural` − `recent_rate` | SP | 5,519 | −12.75 | 1.55 | −8.2 |
| `structural` − `recent_rate` | RP | 10,798 | −4.18 | 0.05 | −80.2 |
| `structural` − `season_rate` | all | 22,807 | **−5.57** | 0.33 | **−16.7** |
| `structural` − `season_rate` | SP | 5,519 | −13.39 | 1.96 | −6.9 |
| `structural` − `season_rate` | RP | 10,798 | −5.10 | 0.33 | −15.3 |
| `structural` − `structural_nogate` | all | 22,807 | **−4.39** | 0.12 | **−37.8** |
| `structural` − `structural_nogate` | SP | 5,519 | −9.24 | 1.06 | −8.7 |
| `structural` − `structural_nogate` | RP | 10,798 | −4.73 | 0.31 | −15.5 |

And the two candidates built to beat it, on the same 26 cutoffs (positive
means the candidate is **worse**):

| Comparison | Role | n | mean diff | SE | t |
|---|---|---|---|---|---|
| `blend` − `structural` | all | 22,807 | **+0.99** | 0.09 | +10.6 |
| `blend` − `structural` | SP | 5,519 | +3.77 | 0.70 | +5.4 |
| `blend` − `structural` | RP | 10,798 | +0.59 | 0.17 | +3.5 |
| `blend_il` − `structural` | all | 22,807 | **+4.18** | 0.56 | +7.4 |
| `blend_il` − `structural` | SP | 5,519 | +3.34 | 0.90 | +3.7 |
| `blend_il` − `structural` | RP | 10,798 | +1.66 | 0.08 | +21.4 |
| `blend_il_share` − `structural` | all | 22,807 | **+2.57** | 0.31 | +8.3 |
| `blend_il_share` − `structural` | SP | 5,519 | +0.17 | 0.46 | +0.4 |
| `blend_il_share` − `structural` | RP | 10,798 | +0.17 | 0.16 | +1.0 |

The gain over the extrapolations is flat in the horizon — it is not a
long-range effect that washes out when there is little season left:

| As-of | Club games left | n | `structural` − `season_rate` | SE | t |
|---|---|---|---|---|---|
| May 1 | 125 | 2,796 | −6.09 | 0.67 | −9.0 |
| May 15 | 112 | 2,768 | −4.94 | 0.31 | −15.7 |
| Jun 1 | 97 | 2,724 | −5.14 | 0.30 | −17.4 |
| Jun 15 | 85 | 2,687 | −5.99 | 0.95 | −6.3 |
| Jul 1 | 70 | 2,635 | −6.16 | 0.59 | −10.5 |
| Jul 15 | 59 | 2,602 | −5.64 | 0.88 | −6.4 |
| Aug 1 | 46 | 2,521 | −5.63 | 0.64 | −8.7 |
| Aug 15 | 33 | 2,467 | −5.11 | 0.49 | −10.5 |
| Sep 1 | 26 | 1,607 | −5.39 | 0.01 | −729.8 |

(The September SE is an artefact of only two seasons reaching that cutoff and
agreeing almost exactly; read it as "the same as every other row".)

## 4. Innings

The same eleven methods over `outs`, holdout, same 26 cutoffs and the same
22,807 pitcher-projections. MAE in **outs** per pitcher; divide by three for
innings.

| Method | MAE (outs) | MAE (IP) | RMSE | wMAE | SP MAE | RP MAE | proj/actual |
|---|---|---|---|---|---|---|---|
| `zero` | 65.90 | 21.97 | 97.66 | 144.74 | 146.21 | 42.12 | 0.000 |
| `last_season` | 48.43 | 16.14 | 71.00 | 69.41 | 86.18 | 37.05 | 0.872 |
| `season_rate` | 36.45 | 12.15 | 56.14 | 51.87 | 67.13 | 27.24 | 0.976 |
| `recent_rate` | 36.00 | 12.00 | 56.52 | 52.62 | 66.87 | 26.81 | 0.986 |
| `blend_il` | 35.59 | 11.86 | 50.97 | **44.45** | 59.96 | 28.35 | 1.075 |
| `structural_nogate` | 35.57 | 11.86 | 53.69 | 45.43 | 63.84 | 27.10 | 1.060 |
| `blend_il_share` | 34.38 | 11.46 | 49.03 | 45.14 | 57.42 | 27.47 | **1.010** |
| `blend` | 33.31 | 11.10 | 53.89 | 52.16 | 60.31 | 25.32 | 0.837 |
| **`structural`** (served) | 32.59 | 10.86 | 50.89 | 47.55 | 57.52 | 25.17 | 0.946 |
| `structural_cal` | 31.61 | 10.54 | 49.10 | 49.09 | 55.12 | 24.57 | 0.840 |
| `structural_hazard` | **31.50** | **10.50** | **49.03** | 48.47 | **54.90** | 24.51 | 0.851 |

**The ordering is identical to batters faced, method for method.** Paired,
the served model beats `season_rate` by −3.86 outs (**−1.29 innings**, t
−15.6), `recent_rate` by −3.41 (−1.14 IP, t −11.2), `structural_nogate` by
−2.98 (−0.99 IP, t −33.2), `last_season` by −15.84 (−5.28 IP, t −21.5) and
`zero` by −33.31 (−11.10 IP, t −19.0). `structural_hazard` beats it by −1.09
outs (t −5.8), `structural_cal` by −0.98 (t −4.6); `blend`, `blend_il` and
`blend_il_share` lose by the same margins and in the same order as on batters
faced.

The practical consequence is that the site loses nothing by deriving innings
from batters faced through the league's batters faced per inning, which is
what `ros_pitching_line` does: an innings model estimated directly ranks the
methods the same way and does not beat the rescale.

## 5. Does station B's insight port?

Station B's finding was not "handle the injured list". It was a specific
substitution: **replace the zero with an expected share** — what a hitter
would have taken healthy, times the fraction of the horizon the return-time
distribution says he will be back for. That was worth 6.4 PA a hitter at two
months, t −5.4, more than anything on the rate side.

For pitchers the finding splits in two, and only the first half ports.

### The half that ports, and pays more than anything else here

`structural` differs from `structural_nogate` only in that it applies station
B's expected-return fractions. It is worth **−4.39 batters faced a pitcher on
the holdout (t −37.8)** and −4.00 over all five seasons — *larger than the gap
to the best baseline*, larger than every difference between the candidate
methods, and stable at every horizon:

| As-of | Club games left | n | `structural` − `structural_nogate` | SE | t |
|---|---|---|---|---|---|
| May 1 | 125 | 2,796 | −5.35 | 0.29 | −18.6 |
| Jun 1 | 97 | 2,724 | −4.09 | 0.09 | −46.2 |
| Jul 1 | 70 | 2,635 | −4.92 | 0.04 | −111.0 |
| Aug 1 | 46 | 2,521 | −4.16 | 0.33 | −12.7 |
| Sep 1 | 26 | 1,607 | −3.97 | 0.26 | −15.4 |

For a starter it is worth **−9.24 (t −8.7)**. Nothing else in the model is
close. Whatever else this document says, *knowing who is unavailable this
morning is the single most valuable input to a pitcher workload projection*,
and that is station B's insight arriving intact.

### The half that does not port

`blend_il` is `blend` with exactly station B's substitution. On hitters that
change was worth −6.4 PA. On pitchers it **costs 2.90 batters faced a pitcher
across all five seasons (t +8.6)**, and the damage is concentrated exactly
where station B's gain was — at long horizons:

| As-of | Club games left | n | `blend_il` − `blend` | SE | t | station B, hitters |
|---|---|---|---|---|---|---|
| May 1 | 131 | 4,681 | **+9.48** | 0.87 | +10.9 | — |
| May 15 | 118 | 4,638 | **+6.69** | 0.59 | +11.3 | — |
| Jun 1 | 103 | 4,566 | +3.54 | 0.66 | +5.3 | — |
| Jun 15 | 90 | 4,502 | +1.96 | 0.42 | +4.7 | **−13.51** (t −6.9) |
| Jul 1 | 76 | 4,410 | +1.26 | 0.43 | +3.0 | **−9.39** (−6.3) |
| Jul 15 | 64 | 4,335 | +0.45 | 0.39 | +1.2 | **−7.08** (−5.9) |
| Aug 1 | 51 | 4,217 | +0.97 | 0.18 | +5.3 | **−4.09** (−4.1) |
| Aug 15 | 38 | 4,134 | +0.53 | 0.22 | +2.4 | **−2.18** (−3.2) |
| Sep 1 | 28 | 3,213 | +0.78 | 0.17 | +4.6 | **−0.70** (−2.2) |

(The hitter column is [playing-time.md §5.4](playing-time.md#54-2025-across-seven-horizons),
in plate appearances, at 2025 cutoffs with 92 down to 25 games left.)

**Same shape, opposite sign.** The two stations are not disagreeing about the
return-time distribution — they share it, fitted from the same spells, in the
same module. They are disagreeing about what a returning player is worth, and
the roster arithmetic says why:

| At a holdout cutoff, per club-day | pitchers | station B's hitters (2026-09-02 build) |
|---|---|---|
| on 40-man rosters | 742 | 602 |
| active | 388 (52%) | 420 (70%) |
| on the injured list | 173 (23%) | 85 (14%) |
| optioned or otherwise unavailable | 181 (24%) | 97 (16%) |
| share of the rest-of-season workload the unavailable actually take | **18.2%** | 5.7% |
| of those on the injured list, share who appear again at all | **41.9%** | — |
| of those optioned, share who appear again at all | **61.8%** | — |

Half of a pitching staff's 40-man is unavailable on any given morning, against
30% of a club's hitters, and **fewer than half the pitchers on the injured
list at a cutoff throw another major-league pitch that season**. The expected
active fraction answers "how much of the horizon will he be back for"; for
hitters that is nearly the whole question, because a hitter who is back is
back in the lineup. For pitchers it is not: he comes back on a rehab
assignment, then a pitch count, then a shortened outing, and if he was a
starter somebody else now has the slot. Station B's arithmetic gives him one
discount. The served model's cruder treatment gives him two — his trailing
window is *already* empty because he is hurt, and the active fraction is
applied on top of that — and double-discounting turns out to be closer to
right.

There is one place `blend_il`'s treatment does win, and it is worth recording:
**realized-workload-weighted MAE**, where it is the best method in the table
(60.45 against the served model's 64.88, and 44.45 against 47.55 on outs).
Weighting by realized work concentrates the metric on the pitchers who came
back and pitched a lot — precisely the population station B's substitution is
right about. It is wrong about the far larger population of optioned arms who
never appear, and unweighted MAE counts those.

### The normalization half, which nearly ports

`blend_il_share` scales each club's staff so that between them they face the
club's own projected total — the direct analogue of station B normalizing a
club's hitters to the club's plate appearances. Overall it is +2.57 batters
faced behind the served model (t +8.3), but **for starters it is +0.17
(t +0.4), a tie**, it wins RMSE outright (68.44 against 70.91), and its league
total is the best-conserved of any method (**1.011** against 0.943). The
constraint is real. What it cannot fix is that the pitchers it redistributes
toward are relievers whose individual workloads it does not know any better
than before.

## 6. Starters and relievers

They are different processes and the numbers say so at every level. The served
model's holdout MAE is **78.9 batters faced for a starter and 35.7 for a
reliever**; every method's ordering is the same in both, but the sizes are
not, and neither is the calibration.

| | starters | relievers |
|---|---|---|
| served MAE, holdout | 78.85 | 35.67 |
| gain over `season_rate` | −13.39 (t −6.9) | −5.10 (t −15.3) |
| gain over `recent_rate` | −12.75 (t −8.2) | −4.18 (t −80.2) |
| worth of the injured-list gate | −9.24 (t −8.7) | −4.73 (t −15.5) |
| projected total / actual total | **1.084** | **0.799** |

The last row is the one to look at. The served model **over-projects starters
by 8% and under-projects relievers by 20%**, and the two nearly cancel in the
league total (0.943). The starter excess is a modelling failure and §7 is
about it. The reliever shortfall is mostly not one: 7.9% of all realized
workload is taken by arms nobody's 40-man carried at the cutoff, and every
method — including every baseline — projects those at zero. `season_rate`
(1.142), `recent_rate` (1.151) and `structural_nogate` (1.208) over-project
starters by more than the served model does; only the shrunk candidates and
`blend` land nearer 1.

Two structural questions about role were tested and neither changed anything.
**Reading role off `gamesStarted` rather than off a twelve-batter threshold**
— which correctly calls an opener a starter, and which the served model gets
wrong on purpose — is inside `blend`, and `blend` is +0.99 behind overall and
+3.77 behind on starters. **Regressing toward the league's own role averages
computed at the cutoff**, rather than toward the frozen
`ROLE_BF_PER_APPEARANCE = {SP: 22.0, RP: 4.3}`, is in the same comparison and
also does not pay: the frozen constants are close enough to the league's
actual averages that there is nothing there.

The **horizon blend** that station B fitted so carefully is worth almost
nothing here either. Sweeping a constant weight on the trailing window across
the 2022–2023 cutoffs traces a nearly flat curve — MAE 66.84 at w = 0, 65.87
at w = 0.5, 66.78 at w = 1 — a 1.5% spread across the whole range. Fitting the
two anchors gives w(30 games) = 0.76 and w(90) = 0.43, and that fitted blend
is what `blend` runs; it does not rescue it.

## 7. The one candidate that beats the served model, and why it is not wired

The residual has a shape. A flat per-role multiplier fitted on 2022–2023
(0.90 for starters, 0.87 for relievers) says the served projection is about a
tenth too high — but the by-horizon table says the excess is four batters
faced a pitcher at a May cutoff and *nothing at all* from August:

| As-of | Club games left | n | `structural_cal` − `structural` | SE | t | `structural_hazard` − `structural` | SE | t |
|---|---|---|---|---|---|---|---|---|
| May 1 | 125 | 2,796 | −4.28 | 0.25 | −17.4 | **−4.52** | 0.32 | −14.0 |
| May 15 | 112 | 2,768 | −2.98 | 0.61 | −4.9 | −2.96 | 0.62 | −4.8 |
| Jun 1 | 97 | 2,724 | −1.96 | 0.49 | −4.0 | −1.97 | 0.46 | −4.3 |
| Jun 15 | 85 | 2,687 | −1.65 | 0.53 | −3.1 | −1.70 | 0.48 | −3.5 |
| Jul 1 | 70 | 2,635 | −0.71 | 0.42 | −1.7 | −0.87 | 0.34 | −2.5 |
| Jul 15 | 59 | 2,602 | −0.44 | 0.44 | −1.0 | −0.66 | 0.30 | −2.2 |
| Aug 1 | 46 | 2,521 | **+0.13** | 0.33 | +0.4 | −0.26 | 0.19 | −1.3 |
| Aug 15 | 33 | 2,467 | **+0.04** | 0.20 | +0.2 | −0.19 | 0.13 | −1.5 |
| Sep 1 | 26 | 1,607 | **+0.17** | 0.08 | +2.2 | −0.04 | 0.05 | −0.8 |

That is not a level error. It is a **survival** error: the model gives a
healthy pitcher his turn every fifth day from today until October, and some
fraction of healthy pitchers lose the season in July. Station B's own
[§8.6](playing-time.md#8-what-would-improve-it-in-order) names exactly this —
"a hazard model for in-horizon injuries… nothing yet handles the healthy
regular who gets hurt in week three" — as the thing neither station has.

`structural_hazard` is that model, in its smallest form: a constant per-role
hazard λ of losing the rest of the season per club game, so the expected share
of a horizon of `h` club games a healthy pitcher is available for is
`(1 − e^{−λh}) / (λh)`. Two parameters, fitted on 2022–2023 by grid search:
**λ = 0.0020 for starters and 0.0030 for relievers**, which is a 12% haircut
for a starter at a 130-game horizon and 3% at 30.

It beats the served model, out of sample, on every horizon and in every
season:

| Comparison | Role | n | mean diff | SE | t |
|---|---|---|---|---|---|
| `structural_hazard` − `structural` | all | 22,807 | **−1.52** | 0.25 | **−6.1** |
| `structural_hazard` − `structural` | SP | 5,519 | **−3.50** | 0.48 | **−7.3** |
| `structural_hazard` − `structural` | RP | 10,798 | **−1.87** | 0.38 | **−5.0** |
| `structural_hazard` − `structural_cal` | all | 22,807 | −0.16 | 0.05 | −3.0 |

and it beats the flat multiplier it generalizes, which is the evidence that
the shape is right and not just the level. On innings it is −1.09 outs
(t −5.8). It fixes the starters' calibration almost exactly: **projected over
actual for starters goes from 1.084 to 0.991**.

**It is not wired, and here is the whole reason.** It buys 1.52 batters faced
of MAE — 3.3% — and pays for it in two places:

- **Realized-workload-weighted MAE gets worse**, 66.09 against 64.88 (and
  48.47 against 47.55 on outs). The metric that says a rotation starter's miss
  counts for more than a September call-up's prefers the served model.
- **The league total drops from 94.3% of the batters actually faced to 84.8%.**
  The site multiplies this number by rates to publish counting stats; a
  leaderboard whose projected strikeouts are 15% light in aggregate is a
  visibly wrong product, and the metric that rewards the shrink — unweighted
  MAE over a universe half of whose members are optioned minor-leaguers who
  will never appear — is not the loss function the page has.
- The role split makes the case sharper and also makes it un-shippable *as
  fitted*: the starter half of the hazard is clearly right (conservation
  0.991 against 1.084) and the reliever half is clearly wrong (0.702 against
  0.799, in the wrong direction on an already-low number). Keeping λ_SP and
  zeroing λ_RP would be choosing a parameter on the holdout, which is the one
  thing the gate rule forbids.

So: **measured, published, available in `METHODS`, scored on every run, and
not served.** It is the first thing to revisit when there is a season of
2027 to select on, and it should be selected on a metric the product actually
has — weighted MAE, or MAE on the pitchers a page would show — rather than on
the flat one. `blend` sits in station B's method list on exactly this footing.

## 8. Leakage guards

Every method takes the **whole** season's appearance log and filters it on the
cutoff itself, which means a method that forgets to filter still runs and
quietly scores brilliantly. `tests/test_projections/test_pitcher_workload.py`
closes that with a synthetic two-club season in which every post-cutoff
appearance is rewritten to 9,999 batters faced and a pitcher who does not
exist before the cutoff is added after it, and **every one of the eleven
methods, on both units, is required to be bit-identical** to the same method
run on the pre-cutoff rows alone. A companion test asserts the poison would
have been visible: unfiltered, the season totals reach 40,000 batters faced,
and no projection exceeds 1,000.

Three other guards have their own tests:

- `window_totals` excludes a game played *on* the cutoff. A game that starts
  the evening of the morning the projection is made has not finished.
- `realized` includes both ends of the scored window, so the cutoff day is
  scored rather than falling between the projection and the score.
- `blend_il` reads an unavailable pitcher's usage as of the day he went out,
  the only place in the model where a window ends somewhere other than the
  cutoff. That date is **clamped at the cutoff**, so a transaction the feed
  dates in the future cannot walk the window forward into games that have not
  been played. The test hands the poisoned season a spell start at the end of
  the season and requires nothing to move. It found a real bug: before the
  clamp, a future spell date did move the windows.

The return-time distribution keeps station B's discipline: fitted on the three
seasons *before* the one being projected (minus 2020), read against spells
this season's transaction feed opened strictly before the cutoff.

The **constants** are the other half of walk-forward honesty. Everything
fitted here — the blend's two anchor weights, the two calibration constants,
the two hazards — was chosen on 2022 and 2023 by the selection runs in
`run_pitcher_workload_backtest.py` and written into
`src/projections/pitcher_workload.py` before 2024, 2025 and 2026 were scored.
The served model has no fitted constants at all.

## 9. What would improve it, in order

1. **~~Score it.~~** Done, and the answer is that the arithmetic already in
   production beats every simple alternative — including two built to beat it
   — by 5.6 batters faced a pitcher against the nearest baseline.
2. **The attrition hazard (§7), selected on a metric the product has.** It is
   the one candidate that beats the served model out of sample and it does so
   with a survival curve whose shape matches the residual rather than a fudge
   factor. What it needs is a fresh season to choose on and a loss function
   that penalises the aggregate shortfall it creates.
3. **A rotation slot, from the schedule rather than from a rate.** A starter's
   appearances are projected as `rate × games left` with the rate regressed
   toward 1/5.3. A rotation is a queue: knowing a starter pitched Tuesday and
   his club has 130 games in 141 days says how many turns he has, off-days and
   doubleheaders included. The residual variance of turns taken around
   `games/5.3` is where the starters' 78.9 MAE mostly lives, and the schedule
   is already fetched and committed under `data/workload/`.
4. **Injury severity, not just list type.** Shared with
   [station B §8.2](playing-time.md#8-what-would-improve-it-in-order), and
   worth more here: 58% of pitchers on the injured list at a cutoff never
   appear again that season, against a far smaller share of hitters, and the
   transaction description carries the injury in English.
5. **A reliever's leverage tier.** A club's late-inning arms appear far more
   often than its long men, and the model treats both as `RP` regressed toward
   0.40 appearances a club game. The appearance log's inherited-runner and
   inning-entered fields distinguish them and are unused.
6. **Pitchers outside the 40-man.** 7.9% of the workload the holdout scores
   was taken by arms nobody's 40-man carried at the cutoff, and every method —
   every baseline included — projects them at zero. That is a floor on the
   reliever MAE no amount of modelling of the projected pitchers can reach.

## 10. Today's board (as-of 2026-09-03)

676 pitchers projected across the 30 clubs over the 21–24 games each has left:
**249 used as starters for 13,248 projected batters faced (3,081 innings) and
427 as relievers for 8,582 (1,996 innings)**, 21,830 in total. The largest
individual projection is Sandy Alcántara at 115 batters faced. Those are the
same numbers the block carried the day before this document existed — nothing
about the model changed, only what is known about it.
