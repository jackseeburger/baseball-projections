# Scoring the Team Projection (station G)

**The hole this fills.** The site has published playoff, division, pennant and
World Series odds every night since Sept 1, 2026, and until now no
rest-of-season team projection had ever been scored.
`scripts/run_playoff_odds.py` contains no scoring code.
[playoff-odds-validation.md](playoff-odds-validation.md) tests two other
things: whether the board *discriminates* (the coin-flip control — it barely
does in September) and whether it *agrees* with FanGraphs. Neither is
accuracy. `public/data/playoff_odds/` holds a handful of dated snapshots, so
our own published history cannot answer it either. The player side has had a
dated walk-forward harness since the intra-season work
([ros-projections.md](ros-projections.md)); the team side had nothing. This
was the largest unmeasured claim on the site.

Reproduce:

```
# ~16,000 cached Stats API game logs, about twenty minutes at eight workers
python scripts/run_team_backtest.py --stage fetch   --seasons 2015-2026 --workers 8
# one process per season, four at a time, one BLAS thread each
python scripts/run_team_backtest.py --stage project --seasons 2015-2026 --sims 2000
python scripts/run_team_backtest.py --stage score   --seasons 2015-2026 --markdown

# the starter-window sensitivity of §10
python scripts/run_team_backtest.py --stage project --seasons 2022-2025 \
    --sims 2000 --window-days 0 --tag _w0
```

Every table below is printed by the third command; nothing here is typed by
hand. The projections themselves checkpoint to
`data/parquet/team_backtest/projections_<season>.parquet` (gitignored — 48,780
rows), and the scored summary is committed at
`public/data/team_backtest/2015-2025.json`.

---

## 1. The harness

`src/eval/team_season.py` is the team analogue of `src/eval/intraseason.py`.
It cuts a season at a date, hands the whole chain everything strictly before
it, simulates the remainder and returns, per club, projected final wins and
the four probabilities the site publishes.

**The projection is the production one.** `project_chain` calls
`scripts/run_playoff_odds.chain_terms` with the fetch injected, so the team
strength every unannounced game is drawn with, the per-game probability on the
announced ones and the postseason rotations are built by the same function the
nightly job runs. The Monte Carlo is `sim.odds.run_playoff_odds`, unwrapped.
Nothing about the model is re-implemented in the harness; what the harness
owns is the cut, the baselines and the outcomes.

**The cut is applied twice on purpose.** `chain_inputs_before` truncates the
game logs at the cutoff before `ChainInputs` is built, and
`game_model.build_slate` cuts every frame again on its way to a slate. Either
alone would do; both together mean a bug in one fails the guard instead of
quietly improving the score.

**`assert_team_split_clean` is the guard**, and it is stricter than the player
harness's because a season has more places to hide a leak. It asserts six
things:

1. no game in the "played" frame — and so none in the standings summed from
   it — falls on or after the cutoff;
2. no game in the remaining schedule falls before it;
3. the standings **reconcile game for game** with the played frame.
   `fetch_standings(2016)` serves the *final* 2016 table; handing that to a
   projection made on 2016-05-01 is the single most damaging thing that can go
   wrong here and it leaves no trace in the output, so the guard refuses to
   take the standings on trust;
4. every dated frame inside `ChainInputs` — pitching appearances, relief
   outings, pitch counts, starts, start innings, hitting lines, plate
   appearances — ends strictly before the cutoff;
5. `ChainInputs`' prior-season frames carry no row from the season being
   projected;
6. the probables feed reaches no further ahead than the starter window the
   live job sees. For a date already in the past the Stats API serves the
   pitcher who *actually* started, so an untruncated feed would hand the chain
   every remaining game's starter — the one leak here that would look like
   skill.

**One bug the guard's third clause found.** The Stats API lists a game
suspended on one night and finished on another under a single `game_pk` on
*both* dates, each marked Final and each carrying the final score — four of
them in 2025. Counted once each, four clubs finish the season with 163 to 165
games, the as-of standings double-count those results and the Monte Carlo
draws four remaining games twice. `regular_season_games` now keeps the later
row, which is the date the result became known, and `season_outcomes`
reconciles the wins counted off the schedule against the wins the API's own
final table reports. All 300 club-seasons of 2015–2025 agree exactly; before
the fix, four or five clubs a season did not.

The guard is unit-tested by construction rather than by inspection
(`tests/test_eval/test_team_season.py`, 44 tests). The central one builds the
same synthetic 30-club season twice — identical up to the cutoff, and 20-0
blow-outs for the *other* side after it — and asserts the two projections are
equal column for column, Monte Carlo output included. A second fixture makes
every post-cutoff start a 30-strikeout shutout and every hitter a five-homer
night, and asserts the slate's rate tables, run environment and talent win%
do not move.

## 2. What is scored, and against what

| Quantity | Metric |
|---|---|
| Projected final wins | MAE, RMSE against actual final wins |
| Made the playoffs | Brier, log loss against the binary outcome |
| Won the division / pennant / World Series | Brier |
| P(playoffs) | calibration deciles pooled across seasons, with the Murphy decomposition |

Projecting *final* wins and projecting *rest-of-season* wins are the same
problem measured the same way — the banked record is common to the projection
and the outcome and cancels out of the error exactly, and there is a test
asserting it. What does not cancel is the shrinking schedule, which drags
every arm's win MAE toward zero as September arrives whether or not it knows
anything. `rest_wpct_mae`, the same error divided by the club's own remaining
games, is the version that does not shrink for free, and the through-season
curve is read off it and off the Brier scores.

**The four baselines**, each computed at the same as-of dates:

| Arm | What it is |
|---|---|
| `chain` | the production projection |
| `record_500` | the club's current record, .500 the rest of the way. In the Monte Carlo this is every club at .500, which is exactly the coin-flip control [playoff-odds-validation.md](playoff-odds-validation.md) has scored against since the first run |
| `record_wpct` | the current record extrapolated at the club's own season-to-date win rate, capped at .250/.750 so a 2-1 start is not a 121-win club |
| `preseason` / `preseason_light` | a projection made before the season and never updated: last season's run rates regressed half way to the league (162 games of ballast) or a third of the way (81). Same numbers at every cutoff, by construction. Two shrinkages because which is the stronger preseason baseline is an empirical question, and the table below reports the one that beats us harder — a choice made in the baseline's favour, declared rather than quietly optimised |
| `coin_flip` | no information at all: half of each club's schedule won, and the league's own base rates for the four probabilities (12/30 in October since 2022, 10/30 before it; 6/30, 2/30, 1/30) |

Paired differences are computed on the same club, the same date and the same
season, with **standard errors clustered by season**. Clubs inside a season
share a schedule and a pennant race, and the same club appears at every
cutoff; one cluster per season is what keeps 8,000 rows from being counted as
8,000 independent pieces of evidence. The estimator is the usual sandwich for
a sample mean with the finite-cluster correction G/(G−1).

## 3. What is excluded, and why

**2020 is excluded by name.** A 60-game season with an eight-club-per-league
bracket seeded by division place is not the season this projection projects,
and folding it in would average two different questions.
`bracket.format_for_season(2020)` raises rather than guessing, and
`run_team_backtest.parse_seasons` drops it from any range it is asked for.

**The postseason field changed in 2022** — five clubs a league before, six
after — and scoring a 2016 projection against a six-club field would be
scoring a different outcome, not a worse model. `bracket.PlayoffFormat`
carries the field size and the first round's length; `seed_league` takes the
wild-card count; `play_postseason` branches on it. The live board is
unchanged: the default is the modern bracket.

**2026 is projected but not scored.** The season is still running, so there
are no final wins and no October to score against. Its rows are in the
checkpoint parquet and excluded from every table below.

## 4. Known optimism, stated up front

Three places where this harness is kinder to the chain than a live run would
be. All three are small; none is zero.

1. **The starter window.** The nightly job reprices every remaining game whose
   probables are both posted inside seven days, which on a live September
   slate is about 29 of 340 games. For a season already played the Stats API
   serves the pitcher who actually started, so the same seven-day rule reaches
   about 110 of the remaining games — and knowing six days out who will start
   includes injury information nobody had. `--window-days 0` is the
   sensitivity check: it lets the term reach only the games on the as-of date
   itself, which is strictly less than a live run sees. §10 runs it.
2. **The remaining schedule is the one that was actually played**, makeups
   included. A live run draws the schedule as it stands, before the rainouts.
3. **The lineup term is off** (`use_lineups=False`), which is *not* optimism —
   it is what the nightly job does in practice, since it runs at 09:15 UTC and
   no club has posted a card. Feeding the backtest cards for games that had
   not been played would be a leak of a different kind.

---

## 5. What was actually run

**10 seasons — 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025 —
at 249 weekly as-of dates, 30 clubs each: 7,470 scored projections per arm,
44,820 in all.** 2,000 simulations per arm per date, the season's own
postseason format, the whole chain rebuilt at every date from games strictly
before it. 2020 is excluded by name (§3). 2026 was projected too — 22 dates,
3,960 rows — and is **not** scored, because the season has not finished.

Per season the walk runs 24 to 26 dates, starting two weeks after opening day
and stopping when fewer than 30 games remain league-wide.

## 6. The headline

| Arm | n | Final wins MAE | RMSE | Rest-of-season win% MAE | Brier playoffs | Log loss | Brier division | Brier pennant | Brier WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **chain** | 7,470 | **4.50** | **6.16** | **.0675** | **.1034** | **.3169** | **.0814** | **.0517** | .0295 |
| record_500 | 7,470 | 5.80 | 7.83 | .0848 | .1119 | .3430 | .0883 | .0555 | .0301 |
| record_wpct | 7,470 | 6.12 | 9.22 | .0805 | .1236 | .4893 | .1049 | .0569 | **.0293** |
| preseason | 7,470 | 8.47 | 10.54 | .2271 | .2032 | .5992 | .1402 | .0626 | .0320 |
| preseason_light | 7,470 | 8.56 | 10.56 | .2293 | .2157 | .6674 | .1485 | .0646 | .0327 |
| coin_flip | 7,470 | 10.40 | 12.90 | .2796 | .2294 | .6513 | .1600 | .0622 | .0322 |

Paired on the same club, the same date and the same season, with standard
errors clustered by season (10 clusters). Negative favours the chain.

| Baseline | Wins MAE Δ | se | t | Brier playoffs Δ | se | t | Log loss Δ | se | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| record_500 | **−1.307** | 0.160 | −8.17 | **−.0085** | .0038 | −2.27 | **−.0261** | .0109 | −2.39 |
| record_wpct | **−1.622** | 0.234 | −6.94 | **−.0202** | .0037 | −5.42 | **−.1724** | .0288 | −5.99 |
| preseason | **−3.975** | 0.379 | −10.49 | **−.0998** | .0172 | −5.79 | **−.2823** | .0487 | −5.80 |
| preseason_light | **−4.060** | 0.468 | −8.67 | **−.1124** | .0188 | −5.96 | **−.3504** | .0613 | −5.72 |
| coin_flip | **−5.907** | 0.379 | −15.57 | **−.1260** | .0078 | −16.10 | **−.3344** | .0219 | −15.24 |

The projection beats every baseline on both headline metrics, pooled over the
whole season, and it is not one season carrying the rest: **the chain has the
lowest projected-wins MAE in every one of the ten seasons individually**, from
3.28 in 2016 to 5.80 in 2021. On playoff Brier it wins **eight of ten** —
it loses to `record_500` in 2023 (.1076 vs .1068) and 2025 (.1601 vs .1445),
and to `record_wpct` in 2023 (.1076 vs .1044). Both of those are recent
seasons, which is worth watching rather than explaining away.

### The three tail probabilities, with the small-n caveat said plainly

| Outcome | chain | record_500 | Δ | se | t |
|---|---:|---:|---:|---:|---:|
| Division | .0814 | .0883 | −.0069 | .0046 | −1.50 |
| Pennant | .0517 | .0555 | −.0037 | .0024 | −1.55 |
| World Series | .0295 | .0301 | −.0006 | .0010 | −0.58 |

**None of the three separates from the .500-extrapolation baseline.** Say why
rather than round it up: ten seasons produce sixty division winners, twenty
pennant winners and ten champions. A Brier score on 7,470 rows carrying ten
ones is a Brier score on ten events, and the standard error clustered by
season knows it even though the row count does not. Against the weaker arms
the chain does separate on pennant (−.0105 vs the coin flip, t = −3.30;
−.0109 vs preseason, t = −3.16), and on the World Series only barely
(−.0027 vs the coin flip, t = −2.03). `record_wpct` is nominally the *best*
arm on World Series Brier (.0293 against our .0295, t = +0.12), which is a
coin toss on ten events and is reported because it is what the number says.

## 7. Calibration

Deciles of the chain's projected P(playoffs), pooled over the ten seasons.

| Decile | n | Predicted range | Mean predicted | Realized | Gap |
|---:|---:|---|---:|---:|---:|
| 1 | 747 | .000–.000 | .000 | .000 | +.000 |
| 2 | 747 | .000–.001 | .000 | .000 | −.000 |
| 3 | 747 | .001–.026 | .010 | .017 | +.008 |
| 4 | 747 | .026–.101 | .060 | .091 | +.031 |
| 5 | 747 | .101–.218 | .156 | .175 | +.020 |
| 6 | 747 | .218–.388 | .296 | .307 | +.011 |
| 7 | 747 | .388–.595 | .489 | .499 | +.011 |
| 8 | 747 | .595–.801 | .697 | .640 | **−.057** |
| 9 | 747 | .802–.975 | .901 | .877 | −.024 |
| 10 | 747 | .975–1.000 | .995 | .997 | +.002 |

The reliability numbers, from the Murphy decomposition on those deciles
(`brier ≈ reliability − resolution + uncertainty`, base rate .3604,
uncertainty .2305):

| Arm | Brier | Reliability ↓ | Resolution ↑ | Skill score |
|---|---:|---:|---:|---:|
| **chain** | .1034 | **.00055** | **.1257** | **.551** |
| record_500 | .1119 | .00149 | .1188 | .514 |
| record_wpct | .1236 | .00733 | .1141 | .464 |
| preseason | .2032 | .01348 | .0386 | .118 |
| preseason_light | .2157 | .02497 | .0402 | .064 |
| coin_flip | .2294 | .0000014 | .0011 | .005 |

The board is **well calibrated**: reliability of .00055 is a mean squared
miss of 2.3 percentage points across the deciles, and it is three times
better than the .500-extrapolation control and thirteen times better than
extrapolating a club's own rate. Almost the whole Brier advantage over the
controls is **resolution** — the chain separates the field more sharply — and
almost none of it is calibration, which is what a well-plumbed simulator on
top of a merely-adequate strength model should look like.

The one visible defect is decile 8: clubs given a 60–80% chance make it 64%
of the time, so the board is a little too confident about the tier just below
the locks. The coin flip's reliability of 0.0000014 is not a compliment — it
predicts the base rate for everyone and is therefore perfectly calibrated and
perfectly useless, which is why resolution is in the table beside it.

## 8. The curve: what the model is worth, week by week

Buckets are the fraction of the schedule already played. On a 162-game
season, 15% is late April, 30% is late May, 45% is late June, 60% is late
July, 75% is late August and 90% is mid-September.

| Season played | chain wins MAE | record_500 | record_wpct | preseason | coin_flip | | chain Brier | record_500 | record_wpct |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 0–15% | **7.53** | 9.54 | 15.02 | 8.31 | 10.35 | | **.1580** | .1921 | .2491 |
| 15–30% | **6.55** | 8.57 | 9.26 | 8.53 | 10.54 | | **.1500** | .1639 | .1813 |
| 30–45% | **5.32** | 7.05 | 6.60 | 8.43 | 10.32 | | **.1207** | .1290 | .1369 |
| 45–60% | **4.52** | 5.80 | 5.05 | 8.51 | 10.34 | | **.1054** | .1117 | .1171 |
| 60–75% | **3.39** | 4.49 | 3.36 | 8.52 | 10.51 | | .0834 | .0853 | **.0836** |
| 75–90% | **2.21** | 2.76 | 2.25 | 8.48 | 10.41 | | .0574 | **.0558** | **.0558** |
| 90–100% | **1.15** | 1.29 | 1.15 | 8.51 | 10.38 | | .0276 | **.0252** | .0260 |

The same thing as an ASCII plot of playoff Brier — left is better, and the
horizontal axis runs from .0252 to .2491:

```
    0-15% |                           C      FP  L  X   W|
   15-30% |                         C  F  W   P  L  X    |
   30-45% |                   C FW             P  L X    |
   45-60% |                CFW                 P  L X    |
   60-75% |            *                       P L  X    |
   75-90% |      *                             P L  X    |
  90-100% |*                                   P  L X    |

  C = chain   F = record_500   W = record_wpct
  P = preseason   L = preseason_light   X = coin flip   * = two or more
```

And the same curve as paired differences, which is the version with error
bars on it (negative favours the chain, clustered by season):

| Season played | vs record_500: wins | se | t | vs record_500: Brier | se | t | vs record_wpct: wins | se | t | vs record_wpct: Brier | se | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–15% | −2.003 | 0.254 | −7.87 | −.0340 | .0074 | −4.57 | −7.484 | 0.653 | −11.47 | −.0911 | .0117 | −7.80 |
| 15–30% | −2.015 | 0.255 | −7.91 | −.0139 | .0061 | −2.27 | −2.709 | 0.480 | −5.64 | −.0313 | .0087 | −3.59 |
| 30–45% | −1.730 | 0.257 | −6.73 | −.0083 | .0073 | −1.13 | −1.288 | 0.296 | −4.35 | −.0162 | .0054 | −2.99 |
| 45–60% | −1.282 | 0.179 | −7.15 | −.0063 | .0036 | −1.74 | −0.530 | 0.191 | −2.78 | −.0117 | .0051 | −2.30 |
| 60–75% | −1.091 | 0.134 | −8.17 | −.0019 | .0025 | −0.77 | **+0.035** | 0.133 | +0.26 | −.0002 | .0025 | −0.10 |
| 75–90% | −0.550 | 0.073 | −7.58 | **+.0016** | .0021 | +0.77 | −0.043 | 0.053 | −0.82 | **+.0016** | .0016 | +1.01 |
| 90–100% | −0.142 | 0.027 | −5.24 | **+.0024** | .0014 | +1.77 | −0.008 | 0.015 | −0.53 | **+.0017** | .0009 | +1.87 |

### Reading the curve — the answer to "how much is it worth in July"

Three separate crossings, and they are not at the same place.

1. **On projected wins the model never loses to `record_500`.** Even in the
   last tenth of the season it is 0.14 wins better (t = −5.2). The margin
   decays smoothly — 2.0 wins in April, 1.3 at the start of July, 1.1 in
   early August, 0.55 in late August, 0.14 in the final fortnight — but the
   sign never turns. Projecting *how many* games a club will win is a problem
   the model keeps helping with all the way to the end.
2. **On projected wins against extrapolating a club's own rate, the crossing
   is at about 60% of the season — the last week of July.** Before it we win
   by 7.5 wins in April, 1.3 by late June, 0.53 in mid-July. At 60–75% the
   difference is **+0.035 ± 0.133**, i.e. nothing, and it stays nothing to
   the end. Past the trade deadline a club's own season-to-date rate is as
   good a projection of its remaining games as the whole chain.
3. **On playoff probability the model stops beating "current record, .500 the
   rest of the way" at about 60% of the season, and after 75% it is nominally
   behind.** −.034 in April (t −4.6), −.014 in May (t −2.3), −.008 by late
   June (t −1.1, already not significant), −.006 at the start of July
   (t −1.7), −.002 in August (t −0.8), **+.0016 in late August** and
   **+.0024 in September** (t +1.8). The sign turns and the model is on the
   wrong side of it, though never by a significant margin.

**So: in July our projection is worth about 1.1 to 1.3 wins of MAE against a
.500 extrapolation and essentially nothing on the playoff odds themselves.**
The site's playoff odds page is at its most informative in April and May and
is, by August, publishing a number that a pocket calculator on the standings
would produce about as well.

### When does the season stop being able to tell models apart?

Not "any model" — a projection that ignores the standings stays terrible
forever: `preseason` sits at 8.5 wins MAE and .20 Brier in every bucket, and
the coin flip at 10.4 and .23, from April to the last week. What stops
separating is *the arms that read the standings*. From 60% of the season on,
`chain`, `record_500` and `record_wpct` are inside .0025 of Brier of one
another at every bucket, with paired differences smaller than their own
standard errors; by 90–100% all three sit between .0252 and .0276 and the
ordering is inside sampling noise. The **in-season information is the whole
game after August 1, and it is available to anybody who can read a
newspaper.**

This is the same lesson `playoff-odds-validation.md`'s coin-flip control
found on a single September day, now measured across ten seasons and 249
dates, with the crossing point located: **late July.**

## 9. What changes because of this

* Architecture §2's station G row now has a score, and §4's claim that
  September playoff odds are "not edge" is measured rather than asserted.
* The claim the site is entitled to make is **"our projection of final wins
  beats the naive extrapolations all season, by 1.3 wins on average and by
  2.0 in April"** — not "our playoff odds are better", which is only true
  before August.
* The station E and C terms that were wired on Sept 3 are *inside* this
  chain, so this is also the first end-to-end score of that swap at the
  season level. It says the chain is a better season projection than the
  standings arithmetic; it does not say the individual terms above the
  starter are, and it could not — those are worth < .0006 of per-game Brier
  and are invisible at this altitude.

## 10. The sensitivity check on the starter window

§4's first item named the one place this harness could flatter the chain: for a season
already played the Stats API serves the pitcher who *actually* started, so the
nightly job's seven-day probables window reaches about 110 of the remaining
games in a backtest where a live run sees about 30 — and knowing six days out
who will start is partly injury news nobody had.

So the same four seasons were re-projected with `--window-days 0`, which lets
the starter term reach only the games on the as-of date itself. That is
*strictly less* than a live run sees, so the truth is bracketed between the
two. 2022–2025, 103 as-of dates, 3,030 club-projections each.

| Arm | Wins MAE | RMSE | Brier playoffs | Brier division | Brier pennant | Brier WS |
|---|---:|---:|---:|---:|---:|---:|
| chain, 7-day window (as scored above) | 4.3403 | 5.9149 | .12179 | .09943 | .05560 | .02832 |
| chain, same-day window only | 4.3571 | 5.9329 | **.12150** | **.09911** | **.05542** | **.02824** |

| Metric | Δ (7-day − same-day) | se | t |
|---|---:|---:|---:|
| Wins abs err | −0.0167 | 0.0046 | −3.65 |
| Rest-of-season win% abs err | −0.00031 | 0.00004 | −8.01 |
| Brier playoffs | **+0.00029** | 0.00010 | +2.90 |
| Brier division | +0.00031 | 0.00047 | +0.67 |
| Brier pennant | +0.00017 | 0.00006 | +2.86 |
| Brier World Series | +0.00008 | 0.00003 | +2.93 |

**The generous window is worth 0.017 of a win and nothing at all on the
probabilities — where it is in fact very slightly *worse*.** Against an
advantage of 1.31 wins over the .500 extrapolation, the leak this harness was
most at risk from accounts for 1.3% of the margin, and it moves the playoff
Brier the wrong way. Every conclusion in §6–§8 survives it unchanged.

That is not surprising once the arithmetic is done: repricing 110 of ~800
remaining games by a couple of points of win probability moves a club's
expected wins by about a tenth, which is the same order the live job's own
closed-form check reports on 29 games
([playoff-odds-validation.md](playoff-odds-validation.md)). The per-game chain
is worth a great deal *per game* and almost nothing to a season projection;
what the chain buys at this altitude is the **team strength** underneath it,
which reaches every remaining game.

Reproduce:

```
python scripts/run_team_backtest.py --stage project --seasons 2022-2025 \
    --sims 2000 --window-days 0 --tag _w0
```

## 10b. One explanation of §8's crossover has been tested and ruled out

The obvious reading of "our playoff odds stop beating the standings in August,
and almost all our advantage is resolution rather than calibration" is that
the board is over-confident — that it treats one point estimate of team
strength as if it were certain, which it does. That was tested directly by
giving the season Monte Carlo a *distribution* over team strength and
re-running this whole harness, same 249 dates, same arms, same scoring:
[parameter-uncertainty.md](parameter-uncertainty.md).

**It is not the explanation.** The width the model's own 60-game ballast
implies (.0505 of talent win% in April, .0313 in the last fortnight) leaves
playoff Brier at +.00015 (t +0.11) and costs .0104 of projected-wins MAE
(t +4.18), and **the crossover stays at 70–75% of the season, to the week**.
The reason is in §7 of this document already: reliability .00055 is a board
that is *not* over-confident. The best-fitting linear shrinkage of these
playoff probabilities is 0.968, and of the pennant probabilities **1.031** —
the tails want to be *sharper*, not blunter. The one real defect, decile 8,
is a single bin, and a width applied to every club at every date closes it
only by opening deciles 4 and 5.

So §8's crossover is what it looked like: from August on, the remaining
schedule is short enough that a talent estimate adds very little to the
banked record. That is a shortage of *signal*, not a mis-shaped distribution.

## 11. What this does not settle

1. **Ten seasons is ten clusters.** Every standard error here has 9 degrees
   of freedom. A difference at t = 2 is worth roughly what one at t = 2 with
   nine observations is worth, which is "probably real, not certainly".
2. **The tail probabilities are unmeasured, not measured-as-zero.** Ten World
   Series in the sample. Scoring the bracket properly needs either many more
   seasons or a per-series market to score against, which is what
   `KXMLBSERIES` will be in October.
3. **The preseason arm is a stand-in.** Last season's run rates regressed, at
   the better of two shrinkages. A roster-based preseason system (Depth
   Charts, ZiPS) would be a much stronger arm, and this repository has no
   archive of one for 2015. The margin over `preseason` here is therefore an
   upper bound on what a real preseason projection would concede.
4. **The starter window is generous** (§4, item 1) — §10 measures it and
   finds it worth 1.3% of the margin, in the wrong direction on the
   probabilities.
5. **`--sims 2000`**. The Monte Carlo adds about 1.2 × 10⁻⁴ to every arm's
   Brier and cancels out of every paired difference, but it is not zero and
   the runs are not reproducible to the last bit at a different sim count.

