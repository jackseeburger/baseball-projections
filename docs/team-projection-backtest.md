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
python scripts/run_team_backtest.py --stage fetch   --seasons 2015-2025 --workers 8
python scripts/run_team_backtest.py --stage project --seasons 2015-2025 --sims 4000
python scripts/run_team_backtest.py --stage score   --seasons 2015-2025 --markdown
```

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
   itself, which is strictly less than a live run sees.
2. **The remaining schedule is the one that was actually played**, makeups
   included. A live run draws the schedule as it stands, before the rainouts.
3. **The lineup term is off** (`use_lineups=False`), which is *not* optimism —
   it is what the nightly job does in practice, since it runs at 09:15 UTC and
   no club has posted a card. Feeding the backtest cards for games that had
   not been played would be a leak of a different kind.

<!-- RESULTS -->
