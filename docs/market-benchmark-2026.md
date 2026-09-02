# Market Benchmark — 2026 per-game P(win) vs. the exchanges

Station E scored against station M (docs/architecture.md §0: *the market is
the bar*). Produced by:

```
python scripts/backfill_market_closes.py --season 2026
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet
```

## What "market close" means here

For an exchange there is no bookmaker close; the closing line is the last
price before first pitch. Kalshi exposes hourly candlesticks per market, so
the close is the last candle ending at or before the scheduled first pitch —
median **15 minutes** before the game. That price already knows the starting
pitchers and lineups.

Coverage: Kalshi's `KXMLBGAME` series has settled markets from **2026-06-22**;
the first game with a pre-pitch candle is 06-26. Polymarket's closed-event
listing reaches back to early July. Neither venue can give us April–June.

## Scoreboard — 756 games priced by both venues, 2026-07-04 → 09-02

| Model | Brier | Log loss | Mean P(home) |
|---|---|---|---|
| **Kalshi close** | **0.24156** | **0.6759** | 0.537 |
| **Polymarket close** | **0.24165** | **0.6761** | 0.531 |
| **pythag_60_sp_lu_bp** (+ lineups + bullpen, new) | **0.24454** | **0.6821** | 0.532 |
| pythag_60_sp_lu (+ lineups, new) | 0.24479 | 0.6826 | 0.532 |
| **pythag_60_sp** (starters) | 0.24483 | 0.6827 | 0.533 |
| pythag_100 | 0.24616 | 0.6854 | 0.533 |
| pythag_60 (production) | 0.24619 | 0.6855 | 0.533 |
| pythag_160 | 0.24635 | 0.6858 | 0.533 |
| pythag_30 | 0.24651 | 0.6863 | 0.533 |
| win_pct_log5 | 0.24669 | 0.6867 | 0.533 |
| pythag_0 (no regression) | 0.24755 | 0.6886 | 0.533 |
| home_constant | 0.24888 | 0.6909 | 0.533 |

Five decimals because the two new terms are worth less than one unit of the
fourth. Realized home win rate on these games: 0.534. On the wider Kalshi-only
set (876 games from 06-26) the numbers are the same to the third decimal:
Kalshi 0.2415, pythag_60 0.2464.

All three challengers are the same model with more of the nine innings and the
nine hitters priced individually, each as a **delta** on the club's own
regressed run rates:

```
rs9 = team_RS9 + 0.5 · (posted lineup R9 − this club's recent lineups)
ra9 = team_RA9 + (5.5/9) · (starter FIP RA9   − league RA9)
                + (3.5/9) · (available pen RA9 − league pen RA9)
```

`pythag_60_sp` is the middle line alone (`src/sim/starters.py`),
`pythag_60_sp_lu` adds the first (`src/sim/lineups.py`), `pythag_60_sp_lu_bp`
adds the third (`src/sim/bullpen.py`). All of it is the same walk-forward
harness: team rates from games before the date, pitcher and batter rates from
appearances before the date, posted lineups and relief usage from games before
the date, and the same log5 + HFA conversion. On the full 1,776-game 2026 set
(not just the 756 the exchanges priced) they score 0.2466 / 0.2465 / 0.2463
against pythag_60's 0.2479, and on 2025 0.24468 / 0.24464 / 0.24452 — the
ordering survives both wider sets, so it is not an artifact of the market
subset.

**Coverage is essentially total.** Across the 1,776 games only 2 fell back to
`pythag_60` for a missing probable and 2 to `pythag_60_sp` for a missing
lineup (0 and 0 of the 756); 22 of 3,552 starter slots and 67 of 31,968 lineup
slots had no prior history and were scored at league average.

The two exchanges agree with each other closely — mean |Δ| = 0.008,
correlation 0.991 — so the benchmark is not an artifact of one venue.

## Reading it

- **The starting pitcher closes about 30% of the gap to the market; everything
  since has closed another 6%.** pythag_60 → pythag_60_sp is 0.24619 →
  0.24483, which is 0.0014 of the 0.0046 that separated us from Kalshi. Adding
  lineups and the bullpen takes it to 0.24454, another 0.0003. The market still
  wins by 0.0030 (t = 2.0 on the paired per-game difference, so *its* remaining
  edge is real). Our own gain is directionally consistent — 0.0016 on the 756
  market games, 0.0016 on all 1,776 games of 2026, 0.0002 on 2025 — but on any
  one of those sets it is inside one standard error (t = −1.25 on the 756).
  Call it a real term of modest and not-yet-precisely-measured size, not a
  solved station. It clears the station E gate (§3) on the common game set; a
  second season of exchange history is what would make the size of the win
  certain.
- **Lineups are worth almost nothing; the bullpen is worth a little.**
  pythag_60_sp → pythag_60_sp_lu is 0.24483 → 0.24479, a paired difference of
  −0.00004 (se 0.00025, t = −0.18, n = 756) — indistinguishable from zero.
  Adding the pen takes it to 0.24454: −0.00025 on top of the lineup term
  (se 0.00026, t = −0.95) and −0.00029 against `pythag_60_sp`
  (se 0.00038, t = −0.77). Against the production model the whole station-E
  stack is now −0.00165 (se 0.00132, t = −1.25). Every one of those is inside
  one standard error on 756 games; the ordering is the same on all 1,776 games
  of 2026 and on 2025, which is the only reason to believe the sign.
- **The size of a term is set by how far it can move a run rate, and these two
  cannot move one far.** Measured on the 756 games, the starter's FIP sits
  0.373 runs per nine from league average (one standard deviation); a posted
  lineup sits 0.103 from its club's own recent cards; a bullpen sits 0.130
  from the league's relievers. And the lineup number is then multiplied by
  0.5 and the pen number by 3.5/9. There is no version of these terms that was
  ever going to be worth what the starter is worth.
- **Bullpen *availability* — the thing the term was built for — is worth
  nothing at all.** Losing one arm out of eight barely moves a
  workload-weighted pen: with the three-straight-days rule the available pen
  differs from the whole pen on 4 of 756 club-games and by 0.001 runs per nine;
  even the aggressive "threw yesterday at all" reading only reaches 0.098 and
  scores *worse* (0.24450 vs 0.24454 on 2026, 0.24474 vs 0.24452 on 2025).
  What pays is pen **quality** — a component-rate estimate of the relief
  innings beats charging them to a 60-game-ballast team RA/9 — and that is why
  the pen is measured against the league's relievers rather than against the
  club's own pen, the opposite of the choice the lineup term makes.
- **The remaining 0.0030 is a better pitcher model, park, weather and rest.**
  The starter term correlates 0.68 with the market's own deviation from
  pythag_60 at a regression slope of 1.21 — the market moves *further* on the
  same games we move on, in the same direction, so we were under-reacting
  rather than over-reacting. Adding the two new terms raises that correlation
  to **0.72** and brings the slope down to **1.13**: we are under-reacting
  less. Taken on their own against the market's deviation from
  `pythag_60_sp`, the lineup term correlates 0.18 (slope 1.12) and the bullpen
  term 0.34 (slope **1.97**) — the market moves twice as far on bullpens as we
  do, which says the pen term is real and still too timid, not that it is
  wrong. The market's residual over us is now 0.00297 (se 0.00147, t = 2.02),
  still a real edge.
- **Team strength does carry real information.** pythag_60 beats the
  home-constant baseline by 0.0027, and the market beats pythag_60 by 0.0046.
  So roughly 37% of the distance from "know nothing" to "the market" is
  covered by regressed run differential alone, and another ~19% by the starter.
- **The starter term is what put probabilities in the tails, and the tails are
  still where we are wrong.** pythag_60 put 6 of the 756 games below 0.40 and
  26 above 0.65; `pythag_60_sp_lu_bp` puts 36 and 47 there. Calibration of the
  full stack:

  | bucket | n | predicted | realized |
  |---|---|---|---|
  | ≤ 0.40 | 36 | 0.373 | 0.333 |
  | 0.40–0.45 | 80 | 0.431 | 0.475 |
  | 0.45–0.50 | 155 | 0.477 | 0.490 |
  | 0.50–0.55 | 162 | 0.526 | 0.537 |
  | 0.55–0.60 | 183 | 0.576 | 0.585 |
  | 0.60–0.65 | 93 | 0.620 | 0.591 |
  | > 0.65 | 47 | 0.685 | 0.617 |

  The top bucket is unchanged as the soft spot — 47 games predicted 0.685,
  realized 0.617, where `pythag_60_sp` had 42 at 0.683 / 0.619. Neither new
  term touches it: both are small and roughly symmetric, so they move games
  across bucket boundaries without fixing the overconfidence at the top. Our
  biggest home favorites are still too confident, and at 47 games that gap is
  itself about one standard error, so it is a lead rather than a finding.
- **Ballast barely matters** (30–160 games all within 0.0003). The signal we
  were missing was not in how we regress team strength — it was that we had no
  pitcher at all.

### How the pitcher term avoids fitting the test set

The pitcher rates are Marcel-standard: K, BB+HBP and HR per batter faced over
the current season plus the two before it at 5/4/3 recency weights, regressed
toward league average and pushed through the standard FIP coefficients
(13/3/−2) with the constant set so a league-average arm returns league RA/9.
Each component is regressed on its own published rate-stabilization point —
70 batters faced for strikeouts, 170 for walks, 1300 for home runs — so home
runs get regressed nearly twenty times harder than strikeouts, which is what
keeps FIP's 13× home-run coefficient from turning noise into a forecast. The
single free knob is how much harder than *reliability* a **projection** has to
regress, since the next start also has to absorb real talent change. That
multiplier was chosen walk-forward on **2025 only**, where the curve is flat
for anything from 2× to 6× (all within 0.00003 Brier); 2× was taken, giving
ballasts of 140 / 340 / 2600 batters faced. No constant was chosen by looking
at a 2026 score.

One deliberate departure from the obvious construction: the starter enters as a
*delta* from league average applied to the team's own runs-allowed rate, not as
`5.5/9 · FIP + 3.5/9 · team_RA`. FIP is park- and defense-neutral and team RA is
not, so the absolute-level blend quietly regresses 61% of every team's run
prevention toward the league mean — a Coors staff and a Petco staff both told
they allow league-average runs for 5.5 innings. Scored, that version came in at
0.2466, *worse* than pythag_60, and its correlation with the market's deviation
was only 0.48 against the delta form's 0.68. The team-regression it smuggled in
cost more than the pitcher information it added.

### How the lineup and bullpen terms avoid fitting the test set

The lineup term reads the **posted** batting order out of the live feed's
boxscore — the per-player `battingOrder` codes ending in "00", not the
team-level `battingOrder` array, which holds the *last* occupant of each slot
and so is a pinch-hitting decision, i.e. a fact about how the game went. That
distinction is not cosmetic: on 2025 the ending lineup's distance from a club's
own norm correlates −0.06 with the runs that club scored, the wrong sign, while
the posted lineup does not.

Batter rates are Marcel-standard: K, BB+HBP and HR per plate appearance, ISO
per at-bat and BABIP per ball in play over the current season plus the two
before it at 5/4/3 recency weights, each regressed on its own published
stabilization point (60 PA / 120 PA / 170 PA / 160 AB / 820 BIP) times the same
2× projection multiplier `starters.py` uses. The five rates are decomposed into
plate-appearance outcome probabilities and priced with standard linear weights,
centred so a league-average batter is worth exactly zero — the estimator has no
fitted constant. Nine batters are weighted by the plate appearances their slot
actually gets ((T−i)/9 + 1, a 1.21:1 spread from leadoff to ninth at the
league's measured 38.2 PA per team-game).

Bullpen rates are the same Marcel/FIP machinery the rotation is priced with,
run over every pitcher rather than only the announced starters, and pooled
across a club's relief appearances in a trailing window with trailing batters
faced as the weight — so the arms a manager leans on count for more, and when
someone is unavailable his innings fall to whoever is left.

The free knobs are four, all chosen walk-forward on **2025 only**: how much of
the lineup delta to apply (0.5), what to measure it against (the club's own
last 15 posted cards, unregressed), how long a window makes a bullpen (21
days), and what counts as used up (worked each of the last three calendar
days). Each surface is flat — the whole {5,15,30,60,season} × {0,5,20} lineup
grid spans 0.0008 Brier on 2025 and the chosen cell wins it by 0.00004, and
every availability rule on the bullpen lands within 0.0004 of every other. The
one choice that is *not* flat is the lineup baseline: measuring a lineup
against **league average** rather than against the club's own cards costs
0.0037 Brier on 2025, because a good offence is already inside `team_RS9` and
charging its distance from the league a second time counts it twice. The
bullpen goes the other way for the same reason inverted — the availability news
alone is too small to be worth anything, so the pen is measured against the
league and the double-counting is the price of getting a component-rate
estimate of the relief innings at all. Both of those calls were made on 2025.
No constant was chosen by looking at a 2026 score.

## What this does not yet show

- Sportsbook closes (Pinnacle) — the archive started 2026-09-02, so a
  book benchmark exists only from here forward.
- April–June, where no exchange history survives.
- Anything about *money*: this is truth scoring against the market's
  probability, not simulated P&L. CLV and fill-aware ROI come after a
  model that is at least at market on Brier.
- **A morning-of forecast.** `probablePitcher` for a past date returns the
  pitcher who actually started, which is what the exchanges' closes knew — the
  median Kalshi close is 15 minutes before first pitch — so the comparison is
  fair. It is not a simulation of predicting at 9am, where late scratches would
  cost a little.
- **Park, weather, rest and travel** — none of them are in the model, and the
  0.0030 the market still holds is where what is left of it lives. Lineups and
  bullpen state *are* now in, and between them they were worth 0.0003 of the
  0.0033: most of the gap was never in either of them.
- **A morning-of bullpen or lineup.** Same caveat as the probable pitcher: the
  boxscore's posted order equals the card the club filed except for late
  scratches, which the backfill silently absorbs, and relief usage is read from
  finished games. The exchanges' closes (median 15 minutes before first pitch)
  knew about the scratches too, so the comparison is fair; it is not a
  simulation of forecasting at 9am.
