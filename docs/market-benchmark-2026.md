# Market Benchmark — 2026 per-game P(win) vs. the exchanges

Stations E and C scored against station M (docs/architecture.md §0: *the
market is the bar*). Produced by:

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

## Station C — the run environment rebuilt from the players

Everything above starts from the same top-down number: the club's
season-to-date runs scored and allowed, regressed with a 60-game ballast.
That is a record of what happened, and it does not know the roster changed.
Station C (`src/sim/run_environment.py`) builds the other half of the
estimate from the players who will actually play, and blends the two:

```
RS_bottom_up = lg_RS9 + PA/game · Σ_i  (hitter i's share of the club's PA)
                                     · (his runs above average per PA)
RA_bottom_up = (5.5/9) · rotation FIP RA/9   (top-5 by starts, weighted by starts)
             + (3.5/9) · bullpen  FIP RA/9   (workload-weighted)

RS_C = w · RS_bottom_up + (1 − w) · RS_pythag60      (same for RA)
```

Nothing new is estimated: the hitter rates are the same Marcel-regressed
component rates and linear weights the lineup term uses, the PA shares are
station B's trailing-share machinery with its one-lineup-slot cap, and the
pitcher rates are the same Marcel/FIP table the announced starter and the pen
are priced off. C is the *assembly*. `pythag_C_sp` then applies exactly the
starter delta `pythag_60_sp` applies, so the pair is directly comparable.

Produced by:

```
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet
```

### Scoreboard — same 756 games

| Model | Brier | Log loss | Mean P(home) |
|---|---|---|---|
| **Kalshi close** | **0.24156** | **0.6759** | 0.537 |
| **Polymarket close** | **0.24165** | **0.6761** | 0.531 |
| **pythag_C_sp** (station C + starter, new) | **0.24428** | **0.6816** | 0.532 |
| pythag_60_sp_lu_bp (+ lineups + bullpen) | 0.24454 | 0.6821 | 0.532 |
| pythag_60_sp_lu (+ lineups) | 0.24479 | 0.6826 | 0.532 |
| **pythag_60_sp** (starters — the gate) | 0.24483 | 0.6827 | 0.533 |
| **pythag_C** (station C alone, no starter) | **0.24565** | **0.6844** | 0.532 |
| pythag_60 (production) | 0.24619 | 0.6855 | 0.533 |

**`pythag_C_sp` beats `pythag_60_sp` by 0.00055** — paired Brier −0.00055
(se 0.00086, t = −0.64, n = 756). It clears the gate, and it does so on the
wider sets too: 0.24603 vs 0.24657 on all 1,776 games of 2026 (−0.00054,
se 0.00051, t = −1.06) and 0.24401 vs 0.24468 on all 2,105 of 2025
(−0.00067, se 0.00049, t = −1.36). Without the starter, `pythag_C` beats
`pythag_60` by the same amount (0.24565 vs 0.24619, −0.00054, se 0.00087).
Every one of those is inside one standard error on its own set; the sign is
the same on all three, which is the only reason to believe it — the same
standard the starter, lineup and bullpen terms are held to.

For scale: C moves a game's probability by 0.020 on average (sd 0.024)
against `pythag_60_sp`, which is about half of what the starter term moves
and roughly five times what the lineup and bullpen terms move.

### It is not shrinkage

The bottom-up estimate is *less spread out* than season-to-date run
differential — FIP and linear weights on heavily regressed component rates
cannot produce a .700 club — so blending the two compresses the league. C
puts 9 of the 756 games below 0.40 and 19 above 0.65 where `pythag_60_sp`
puts 24 and 42. And `pythag_60` is known to be overconfident exactly there
(its top bucket predicts 0.685 and realizes 0.617). So a gain from plain
shrinkage would look identical to a gain from knowing the roster.

`--c-control league` settles it. It replaces the bottom-up half with league
average, making `pythag_C` exactly *"`pythag_60` shrunk half the way to the
league"* with no player information in it at all:

| | 756 market games | 2025, 2105 games |
|---|---|---|
| `pythag_C_sp` − `pythag_60_sp`, real C | **−0.00055** (se 0.00086) | **−0.00067** (se 0.00049) |
| `pythag_C_sp` − `pythag_60_sp`, shrinkage control | −0.00000 (se 0.00104) | −0.00022 (se 0.00073) |
| `pythag_C` − `pythag_60`, real C | −0.00054 (se 0.00087) | −0.00052 (se 0.00050) |
| `pythag_C` − `pythag_60`, shrinkage control | **+0.00044** (se 0.00105) | **+0.00052** (se 0.00075) |

Shrinking the production model halfway to the league is worth nothing with a
starter on top and is actively *worse* without one. The gain is the players.

### Where it comes from

Split by month on 2026 (all games, `pythag_C_sp` − `pythag_60_sp`):

| slice | n | pythag_60_sp | pythag_C_sp | paired |
|---|---|---|---|---|
| July (up to and around the deadline) | 335 | 0.24733 | 0.24710 | −0.00023 (se 0.00132) |
| **August (post-deadline)** | 417 | 0.24303 | **0.24217** | **−0.00086** (se 0.00113) |
| September | 19 | 0.25123 | 0.24894 | −0.00229 (se 0.00553) |

The post-deadline month is where C is worth the most and July is where it is
worth the least, which is the direction the station was built for — but at
these sample sizes the difference between the two months is itself well
inside noise, so it is a lead, not a finding. **The September call-up slice
cannot be measured on 2026 at all**: the season is 19 games old on the last
scored date (2026-09-02). On 2025's full September (374 games) C is worth
−0.00068 (se 0.00112), indistinguishable from its August value there
(−0.00052, se 0.00109). Nothing in the data yet separates "C helps after the
deadline" from "C helps everywhere by about the same amount".

### Against the market

C is the first term that moves as far as the market does. Our deviation from
`pythag_60` correlates **0.75** with the market's deviation from `pythag_60`
at a slope of **1.06** — the starter term alone was 0.68 at slope 1.21 (we
under-reacted by a fifth) and the full station-E stack 0.72 at 1.13. Taken
on its own against the market's deviation from `pythag_60_sp`, the C term
correlates 0.46 at slope **0.83**: the first term we move *further* on than
the market does, which is a reason to prefer the smaller of the two blend
weights 2025 could not separate. The market's residual edge over us is now
**0.00271** (se 0.00138, t = 1.97), down from 0.00326 over `pythag_60_sp`
and 0.00297 over the full E stack — still real, and still where park,
weather, rest and a better pitcher model live.

### How station C avoids fitting the test set

Three free knobs, all chosen walk-forward on **2025 only**, and `--c-weight 0`
reproduces `pythag_60` and `pythag_60_sp` to the last bit (paired difference
+0.00000 with a standard error of 0.00000 on both seasons), so the sweep is a
clean nesting and any gain is the roster information and only that.

**The blend weight**, `{0, .25, .5, .75, 1}` against `pythag_60_sp` on 2025:

| w | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| paired Brier | +0.00000 | −0.00047 | **−0.00067** | −0.00059 | −0.00022 |

An inverted U with an interior minimum at 0.5 and a flat floor from 0.25 to
0.75 — the two halves of the estimate are worth about the same, and a pure
bottom-up model (w = 1) throws away the park, defense, baserunning and
sequencing information that only the top-down half has and gives most of the
gain back. That is the same lesson `starters.py` records from its
absolute-level blend, in a milder form.

**The two windows**, `{15, 30, 60, season}` days of plate appearances by
`{30, 45, season}` days of starts, at w = 0.5 on 2025 — 12 cells spanning
0.00027 Brier, won by season-to-date shares with a 30-day rotation
(−0.00067) over 30-day shares with a 30-day rotation (−0.00052). Longer wins
for the shares here and shorter wins for station B's own forecasts because
the two answer different questions: B forecasts one hitter's next month, C
wants the club's *average* batter, and averaging over more plate appearances
buys more than reacting a week sooner. The rotation goes the other way — 30
days is about six turns, enough to identify five men, and short enough that a
starter traded in July is out of the August rotation.

Nothing else in C has a free constant. The 5.5/3.5 innings split, the top-5
rotation, the one-lineup-slot cap on a share and the linear weights are all
taken unchanged from the terms already scored, and both halves are centred on
the league by construction: a club of league-average hitters returns exactly
league RS/G and a league-average staff exactly league RA/9, so the bottom-up
estimate can only redistribute runs across clubs, never move the league.

**Coverage is total.** Across all 1,776 games of 2026, 0 of 3,552 club-games
failed to get a bottom-up estimate for either half (0 of 1,512 on the 756),
and 2 games fell back to `pythag_C` for a missing probable. The batter
universe — every hitter who has appeared in a posted lineup this season, 637
men in 2026 and 661 in 2025 — covers **99.98%** of the plate appearances the
clubs' own team logs record for 2026 and 99.99% for 2025 (worst club 99.75%);
what it misses is the pinch hitter who never started a game all year.

**Park is deliberately not applied.** `src/data/park_factors.py` computes
team-season factors from a league-wide hitter-season table with a documented
approximation (no home/road splits), keyed by team abbreviation and year
rather than team id and date, and nothing the simulator reads is wired to it.
More to the point, park is already inside the top-down half of the blend,
where it was measured; a park factor would only be needed if C were swapped in
whole rather than blended.

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
  0.0027 the market still holds is where what is left of it lives. Lineups and
  bullpen state *are* now in, and between them they were worth 0.0003 of the
  0.0033: most of the gap was never in either of them. Station C — the run
  environment rebuilt from the roster — took another 0.0003, leaving 0.0027
  (se 0.0014, t = 1.97).
- **Which half of station C is doing the work.** The blend applies one weight
  to runs scored and runs allowed together, so the hitters-and-playing-time
  half and the rotation-and-pen half were never separated. A two-weight sweep
  on 2025 would say whether C is really a pitching-staff term wearing a run
  environment's clothes.
- **A morning-of bullpen or lineup.** Same caveat as the probable pitcher: the
  boxscore's posted order equals the card the club filed except for late
  scratches, which the backfill silently absorbs, and relief usage is read from
  finished games. The exchanges' closes (median 15 minutes before first pitch)
  knew about the scratches too, so the comparison is fair; it is not a
  simulation of forecasting at 9am.
