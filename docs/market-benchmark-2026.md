# Market Benchmark — 2026 per-game P(win) vs. the exchanges

Stations E and C scored against station M (docs/architecture.md §0: *the
market is the bar*). Produced by:

```
python scripts/backfill_market_closes.py --season 2026
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet
python scripts/backtest_game_odds.py --season 2025 --min-games 20
python scripts/sweep_reliever_usage.py --season 2025
python scripts/build_game_features.py --seasons 2015-2026 --workers 8
python scripts/train_game_learned.py --score-season 2026 \
    --market data/parquet/market_closes_2026.parquet
python scripts/train_game_learned.py --score-season 2025
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
wider sets too: 0.24606 vs 0.24661 on all 1,777 games of 2026 through
2026-09-02 (−0.00055, se 0.00056, t = −0.98) and 0.24401 vs 0.24468 on all
2,105 of 2025 (−0.00067, se 0.00049, t = −1.36). Without the starter,
`pythag_C` beats `pythag_60` by the same amount (0.24565 vs 0.24619,
−0.00054, se 0.00087 on the 756; −0.00067, se 0.00056 on all 1,777).
The 2026 season is still running, so the wider set grows by a game or two a
day; the 756 the exchanges priced is fixed.
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
| July (up to and around the deadline) | 371 | 0.24689 | 0.24666 | −0.00023 (se 0.00125) |
| **August (post-deadline)** | 417 | 0.24303 | **0.24217** | **−0.00086** (se 0.00113) |
| September | 20 | 0.25468 | 0.25189 | −0.00279 (se 0.00527) |

The post-deadline month is where C is worth the most and July is where it is
worth the least, which is the direction the station was built for — but at
these sample sizes the difference between the two months is itself well
inside noise, so it is a lead, not a finding. **The September call-up slice
cannot be measured on 2026 at all**: September is 20 games old on the last
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

**Coverage is total.** Across all 1,777 games of 2026, 0 of 3,554 club-games
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

## Station E — the pen that is actually available, and how deep the starter goes

Two terms, added one at a time on top of `pythag_C_sp`, both aimed at the same
soft spot: the model divides every game 5.5 / 3.5 between a starter and an
undifferentiated pen, and knows nothing about who in that pen threw 38 pitches
last night.

```
ra9 = C's runs allowed
      + (ip/9)     · (starter FIP RA9        − league RA9)     ← starters.py
      + ((9-ip)/9) · (available pen FIP RA9  − league pen RA9) ← reliever_usage.py

available pen RA9 = Σ_i (trailing BF_i · availability_i · RA9_i) / Σ_i (…)
ip                = this starter's own innings per start, clipped to [3, 9]
                    (5.5 for a man with no start on file)
```

`pythag_C_sp_bpa` is the pen line with `ip` held at the flat 5.5;
`pythag_C_sp_bpa_ip` also moves the split. Produced by:

```
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet
python scripts/backtest_game_odds.py --season 2025 --min-games 20
python scripts/sweep_reliever_usage.py --season 2025      # the constants
```

### Where the workload comes from

The Stats API's pitching game log carries **`numberOfPitches`** on every split,
which is the workload number a manager actually counts. It is present on
**17,777 of 17,779** appearances in 2026 and **20,868 of 20,870** in 2025
(99.99%); the four that lack one fall back to batters faced times the league's
measured pitches per batter faced (3.886 in 2026, 3.882 in 2025). Starts count
as work as well as relief outings — an opener's arm is as tired as a
reliever's — while membership in the pen and the workload weights still come
from relief appearances only, as they did before.

For scale: the median relief outing is **17 pitches** and the 90th percentile
**33** (2026; 16 and 31 in 2025).

### Scoreboard — same 756 games, 2026-07-04 → 09-02

| Model | Brier | Log loss | Mean P(home) |
|---|---|---|---|
| **Kalshi close** | **0.24156** | **0.6759** | 0.537 |
| **Polymarket close** | **0.24165** | **0.6761** | 0.531 |
| **pythag_C_sp_bpa_ip** (+ available pen + start length, new) | **0.24388** | **0.6807** | 0.532 |
| **pythag_C_sp_bpa** (+ available pen, new) | **0.24400** | **0.6810** | 0.532 |
| **pythag_C_sp** (station C + starter — the gate) | 0.24428 | 0.6816 | 0.532 |
| pythag_60_sp_lu_bpa (E chain, availability weight) | 0.24452 | 0.6821 | 0.532 |
| pythag_60_sp_lu_bp (E chain, binary rest rule) | 0.24454 | 0.6821 | 0.532 |
| pythag_60_sp_lu (+ lineups) | 0.24479 | 0.6826 | 0.532 |
| pythag_60_sp (starters) | 0.24483 | 0.6827 | 0.533 |
| pythag_C (station C alone) | 0.24565 | 0.6844 | 0.532 |
| pythag_60 (production) | 0.24619 | 0.6855 | 0.533 |

**Both terms clear the gate**, and the ordering is the same on all three sets:

| | 756 market games | all 1,778 of 2026 | all 2,105 of 2025 |
|---|---|---|---|
| `pythag_C_sp` (baseline) | 0.24428 | 0.24610 | 0.24401 |
| `pythag_C_sp_bpa` | **0.24400** | **0.24597** | **0.24386** |
| paired vs baseline | −0.00028 (se 0.00026, t −1.05) | −0.00013 (se 0.00017, t −0.77) | −0.00015 (se 0.00017, t −0.89) |
| `pythag_C_sp_bpa_ip` | **0.24388** | **0.24592** | **0.24360** |
| paired vs `_bpa` | −0.00012 (se 0.00014, t −0.90) | −0.00005 (se 0.00010, t −0.54) | **−0.00026 (se 0.00009, t −2.92)** |
| paired vs baseline | −0.00040 (se 0.00031, t −1.28) | −0.00019 (se 0.00021, t −0.88) | **−0.00042 (se 0.00020, t −2.08)** |

Swapped into the older E chain in place of the binary rest rule — the same
model with the pen read one way or the other — the two are indistinguishable:
`pythag_60_sp_lu_bpa` 0.24452 against `pythag_60_sp_lu_bp` 0.24454, a paired
−0.00002 (se 0.00004, t = −0.42) on the 756, −0.00003 (t = −1.18) on all of
2026 and −0.00001 (t = −0.25) on 2025.

### Reading it

- **The dial fires where the switch never did, and it still is not
  availability that pays.** The binary "worked three calendar days running"
  rule leaves a club a man short on **12 of 1,512** club-games and moves the
  pen by 0.001 runs per nine. The pitch-count weight is below full
  availability on **1,485 of 1,512** and moves it by **0.011** — an order of
  magnitude more, on a hundred times as many games. And the gain is almost
  exactly the gain the old term already had. That is the tell: what pays is
  the pen being priced off component rates against the league at all, not the
  news about who is unavailable tonight.
- **The 2025 sweep says so outright.** Measured against the club's **own**
  whole pen — availability news with every trace of quality removed — the term
  is *worse than not having it* in all 32 cells of the grid, by +0.00001 to
  +0.00007 Brier. Measured against the league it is better, by 0.00009 to
  0.00015. The whole difference between those two columns is the double-count
  of pen quality, which is the same trade `bullpen.py` recorded a station ago
  and which C's runs-allowed half already contains half of.
- **The availability surface has no interior optimum.** The declared grid
  {30, 40, 50} pitches yesterday × {45, 65, 85} over two days × {50, 75, 100,
  150} of taper spans **0.00006 Brier in total** on 2025 and is won at its
  gentle corner (50 / 85 / 150, paired −0.00015), with the gain rising
  monotonically along all three axes as the rule does less. Pushed past the
  plausible range it keeps rising: disabling the rule outright — every
  reliever fully available, the pen a plain workload-weighted average — scores
  **−0.00017**, better than any cell that actually weights by availability. So
  the constants that ship are the gentlest plausible ones (a 50-pitch relief
  outing is past the 90th percentile of 31, and at a taper of 150 a routine
  16-pitch inning costs 11% of an arm), and the honest summary is that the
  availability weighting is worth nothing and the pen-quality estimate around
  it is worth 0.0003.
- **How deep the starter goes is a different story — it is the first term here
  whose sign is significant.** The innings split is worth −0.00026 on 2025
  at **t = −2.92**, and every ballast in {0, 5, 10, 20, 40, 80} agrees on the
  sign at t between −2.2 and −2.9. On 2026 it lands at −0.00012 (t = −0.90) on
  the 756 and −0.00005 (t = −0.54) on all 1,778, so the *size* is
  season-dependent and the 2025 number is the optimistic one. But the direction
  never flips, and the mechanism is not subtle: the mean expected start is
  **5.21 innings** with a standard deviation of **0.50**, so a flat 5.5 charges
  an opener with three innings he will not throw and hands them to the wrong
  staff.
- **For scale.** The pen term moves a game's probability by 0.0061 on average
  (sd 0.0043, max 0.019) against `pythag_C_sp`; adding the innings split takes
  that to 0.0071 (sd 0.0052, max 0.027). Station C moves 0.020 and the starter
  term about 0.04, so these are the two smallest terms on the board — which is
  what a 3.5-inning delta on a workload-weighted average of eight arms was
  always going to be.
- **Against the market.** Our deviation from `pythag_60` now correlates
  **0.770** with the market's at a slope of **1.04** (`pythag_C_sp` was 0.738 at
  1.06, the starter term alone 0.676 at 1.21), and 0.778 at 1.05 with the
  innings split on top. Taken alone against the market's deviation from
  `pythag_C_sp`, the pen term correlates 0.379 at a slope of **2.05** — the
  market moves twice as far on bullpens as we do, exactly what the old bullpen
  term measured (0.34 at 1.97) and a reason to think the term is still too
  timid rather than wrong. The market's residual edge over us is now
  **0.00232** (se 0.00132, t = 1.75), down from 0.00272 over `pythag_C_sp`.
- **Calibration is unchanged where it was already wrong.**

  | bucket | n | predicted | realized |
  |---|---|---|---|
  | ≤ 0.40 | 17 | 0.378 | 0.353 |
  | 0.40–0.45 | 66 | 0.428 | 0.424 |
  | 0.45–0.50 | 147 | 0.477 | 0.463 |
  | 0.50–0.55 | 217 | 0.524 | 0.535 |
  | 0.55–0.60 | 209 | 0.573 | 0.574 |
  | 0.60–0.65 | 73 | 0.622 | 0.616 |
  | > 0.65 | 27 | 0.676 | 0.778 |

  The top bucket has flipped sign since `pythag_60_sp` (42 games at 0.683
  predicted / 0.619 realized, over-confident) to 27 at 0.676 / 0.778,
  under-confident — which is not a finding at 27 games, it is the same
  standard error wearing the other hat. What is real is that station C pulled
  the tails in and these two terms did not push them back out.

### How these terms avoid fitting the test set

Four free constants, all chosen walk-forward on **2025 only**, and every one
of them has a setting that reproduces the model without the term exactly (a
taper and thresholds large enough to disable the weighting; a ballast large
enough to return the flat 5.5), so the sweeps are clean nestings.

**The availability rule**, `scripts/sweep_reliever_usage.py --season 2025`,
paired Brier against `pythag_C_sp` on 2,105 games, league baseline:

| taper → | 50 | 75 | 100 | 150 |
|---|---|---|---|---|
| 30 / 45 | −0.00009 | −0.00011 | −0.00011 | −0.00012 |
| 40 / 65 | −0.00011 | −0.00013 | −0.00014 | −0.00014 |
| **50 / 85** | −0.00011 | −0.00014 | −0.00015 | **−0.00015** |

with every cell's standard error 0.00017. The same grid on the **team**
baseline runs +0.00001 to +0.00007 — worse than no term at all — which is why
the pen is measured against the league, the same call `bullpen.py` made and
the opposite of the one `lineups.py` made.

**The innings ballast**, `--sp-ip-ballast` on 2025, paired against
`pythag_C_sp_bpa` (n = 2,105):

| starts of ballast | 0 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| paired Brier | **−0.00026** | −0.00016 | −0.00012 | −0.00008 | −0.00005 | −0.00003 |
| t | −2.92 | −2.78 | −2.63 | −2.47 | −2.34 | −2.24 |
| mean expected start | 5.28 | 5.40 | 5.43 | 5.45 | 5.47 | 5.48 |

Monotone to the natural boundary of the parameter, so no ballast ships and the
only guard is the clip to 3–9 innings. That is deliberate and it is what the
season says: innings per start is mostly a fact about a pitcher's **role**,
not his luck — an opener is an opener — and regressing the opener's two
innings toward five and a half only throws the role away. A pitcher with no
start on file this season keeps the flat 5.5.

**Leakage.** Both terms read the same kind of thing they predict, so both are
guarded and both guards are unit-tested on synthetic logs
(`tests/test_sim/test_reliever_usage.py`, `tests/test_sim/test_starters.py`).
Availability is computed from pitches thrown on the calendar days *strictly
before* the game — appending the game's own outing, or tomorrow's, leaves every
weight identical — and the expected-innings table is built from starts strictly
before the date, because how long tonight's start lasts is the outcome. The
same cut catches the second game of a doubleheader reading the first.

**Coverage is total.** No game on any of the three sets fell back for want of
pitch counts or a pen, and the two appearances a season that carry no
`numberOfPitches` are estimated from batters faced.

## Sept 3, 2026 — park and team defence: both measured, neither one clears

The chain the nightly serves prices runs two ways and neither of them knows
what ballpark a game is in. Station D's half is a record of what happened *in
the parks the club has played in*; station C's half is FIP and linear weights,
which are park- and defence-blind by construction. The validation doc's table
is what that looks like from outside: Atlanta allows 4.02 runs a game against a
bottom-up 4.38, Milwaukee scores 4.84 against a component 4.61, Los Angeles
bats project 5.19 against 4.78 scored
([playoff-odds-validation.md](playoff-odds-validation.md#sept-3-2026--the-full-chain-is-wired)),
and the w = 0.5 blend absorbs every one of those gaps as one unattributed lump.

Two terms were built to take that lump apart. **Neither clears its gate**, and
the two failures have different and instructive shapes: the park factors are
right and the model cannot use them, while the defence estimate is usable and
the model already had it.

```
park     factor(v) = runs per game at v ÷ runs per game in the road games of
                     the clubs who host there, pooled over the two *prior*
                     seasons, regressed toward 1 with `park_ballast` games and
                     renormalised so the league mean factor is exactly 1
                     → the top-down half is divided by each club's own
                       games-weighted exposure *before* it is regressed, and
                       both clubs' rates are multiplied by tonight's venue
                       factor before Pythagenpat

defence  ΔRA/9 = (club BABIP allowed on the road − league BABIP) · BIP per 9
                 · 0.75 runs a hit, with the club's BABIP regressed toward the
                 league with `def_ballast` balls in play
                 → added to station C's *bottom-up* runs allowed, which is
                   where FIP's blind spot is
```

Produced by:

```
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet \
    --park-ballast 200 --def-ballast 4000
python scripts/backtest_game_odds.py --season 2025 --min-games 20 \
    --park-ballast 200 --def-ballast 4000
python scripts/attribute_run_environment.py --season 2026 --as-of 2026-09-03
```

`--park-ballast inf` and `--def-ballast inf` switch the respective term off and
reproduce `pythag_C_sp_bpa_ip` to the last bit, so both sweeps are clean
nestings and the gate comparison is exact.

### The park factors themselves are fine

Before asking whether the term helps, ask whether the numbers are right. The
2026 factors are built from 2024 and 2025 alone; measured against what actually
happened in 2026 they correlate **0.44** across the 30 parks with 30+ games, at
a regression slope of **1.23** — the regressed factors are, if anything, a
little timid. Coors comes out at 1.126 (raw 1.28) and the flattest park at
0.910; the whole spread is 0.91 to 1.13 after regression.

So the ballpark is being measured. What follows is not a failure of the
estimate.
## Station E — a model that chooses its own form (Sept 3, 2026)

Every term in the chain above is something a person wrote down: log5 on
Pythagenpat, the starter as a delta over the innings he covers, the pen over
the rest, the card at half weight against the club's own norm. Each was worth
one to four ten-thousandths. Nothing has ever been allowed to pick its own
functional form, and until this section there was no machine-learning library
in the repository at all.

So: build the same information as a table, hand it to a gradient-boosted model
with no shape imposed, and score it on the same games.

```
python scripts/build_game_features.py --seasons 2015-2026 --workers 8
python scripts/train_game_learned.py --score-season 2026 \
    --market data/parquet/market_closes_2026.parquet \
    --save-model data/models/game_learned_2026.json
python scripts/train_game_learned.py --score-season 2025
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet \
    --learned data/models/game_learned_2026.json
```

**The answer is that it rediscovers the chain.** It is a little better on a
whole season, a little worse on the market's own games, never outside one
standard error of either, and 88–94% correlated with the number the chain
already serves. It does not clear the gate; the chain runs; the model lives
behind `--learned`.

### The table

23,193 games, one row each, 2015 through 2026:

| season | games | first scored date | club-games with a posted card | starter slots with prior history |
|---|---|---|---|---|
| 2015 | 2,105 | 05-01 | 100.0% | 99.0% |
| 2016 | 2,088 | 04-30 | 100.0% | 99.3% |
| 2017 | 2,104 | 04-28 | 100.0% | 99.2% |
| 2018 | 2,072 | 04-27 | 100.0% | 99.0% |
| 2019 | 2,100 | 04-23 | 100.0% | 98.9% |
| 2020 | 455 | 08-27 | 100.0% | 98.5% |
| 2021 | 2,078 | 04-29 | 100.0% | 98.8% |
| 2022 | 2,099 | 05-02 | 100.0% | 98.9% |
| 2023 | 2,106 | 04-24 | 100.0% | 99.0% |
| 2024 | 2,102 | 04-22 | 100.0% | 99.0% |
| 2025 | 2,105 | 04-21 | 100.0% | 99.0% |
| 2026 | 1,779 | 04-19 | 100.0% | 99.3% |

The first scored date is the harness's own cut — no date is priced until every
club has twenty games — which costs about three weeks of each April and most
of 2020's sixty-game season. 2015 is the floor because Marcel wants the two
completed seasons before the one being built and the repository's season-level
hitting table starts there; 2013 and 2014 were fetched into a file of their own
rather than by refreshing the shared one, which is what makes 2015 and 2016
possible without moving any other station's numbers.

**Where the leakage guarantee comes from.** Every row is read off a
`game_model.build_slate` slate — the one function that applies the
strictly-before cut for both the nightly and the backtest — rather than
rebuilt beside it. Nothing is re-derived, so nothing can drift. The proof that
the reading is faithful is that the `chain_p` column the builder computes
scores **.24388** on exactly the 756 market-priced games and **.24619** for
`pythag_60`: the published numbers to the fifth decimal, out of a different
assembly of the same slate. The guard itself is unit-tested by computing a
synthetic season's features twice — once with the logs truncated mid-season,
once with the whole season on file — and asserting every feature and both
chain columns are identical on the games before the cut
(`tests/test_sim/test_game_features.py`).

The 46 features are each side's top-down regressed rates, station C's
bottom-up half and the blend the chain prices with; the announced starter's
FIP rate, his expected innings, whether he has history; the
availability-weighted pen as a delta from the league's relievers; the posted
card as a delta from the club's own recent cards; rest for the club and for
the pitcher; how far into the season it is; the league's run environment that
day; the observed home-field edge; and month, weekday, venue and day/night.
The chain's own probability is **not** among them — a model handed
`pythag_C_sp_bpa_ip` would be a residual-learner, and the question was what
the raw inputs are worth on their own. The blend below is where the two are
allowed to meet.

### The protocol

Train on seasons strictly before the scored one (≤2025 → score 2026; ≤2024 →
score 2025). Hyperparameters and the tree count come from an inner split of
the training seasons — the last one — and never from the scored set; the grid
is five points over learning rate, leaves and minimum leaf size. Calibration
is isotonic, fitted on **season-blocked out-of-fold** predictions over the
training seasons, where each training season is predicted by a model fitted on
the others; the same out-of-fold frame fits the blend weights, so the blend is
out of sample too. And the whole protocol runs a second time on permuted
training labels as the control.

The 2026 model: learning rate 0.02, 15 leaves, depth 4, 200 minimum samples
per leaf, 387 trees, on 21,414 games. The 2025 model: 0.02, 7 leaves, depth 3,
300 minimum, 338 trees, on 19,309 games.

### Scoreboard — the same 756 games, 2026-07-04 → 09-02

| Model | Brier | Log loss | Mean P(home) |
|---|---|---|---|
| **Kalshi close** | **0.24156** | **0.6759** | 0.537 |
| **Polymarket close** | **0.24165** | **0.6761** | 0.531 |
| `pythag_C_sp_bpa_ip_pk` (+ park) | **0.24382** | 0.68062 | 0.532 |
| **`pythag_C_sp_bpa_ip`** (the gate — what the nightly serves) | **0.24388** | 0.68074 | 0.532 |
| `pythag_C_sp_bpa_ip_pk_def` (+ park + defence) | 0.24389 | 0.68075 | 0.532 |
| `pythag_C_sp_bpa_ip_def` (+ defence) | 0.24395 | 0.68088 | 0.532 |

Paired against the gate on the three sets, park at a 200-game ballast and
defence at a 4,000-BIP ballast:

| | 756 market games | all 1,781 of 2026 | all 2,105 of 2025 |
|---|---|---|---|
| `_pk` − gate | **−0.00006** (se 0.00003, t −2.32) | **−0.00004** (se 0.00002, t −1.98) | **+0.00002** (se 0.00002, t +0.75) |
| `_def` − gate | +0.00007 (se 0.00019, t +0.34) | +0.00010 (se 0.00011, t +0.98) | +0.00003 (se 0.00009, t +0.36) |
| `_pk_def` − gate | +0.00001 (se 0.00020, t +0.03) | +0.00007 (se 0.00011, t +0.63) | +0.00005 (se 0.00009, t +0.51) |

The park term is the only one that is better anywhere, and it is better by
0.00006 of Brier — a fifth of what the smallest term already on the board is
worth. Its t of −2.3 on the market set is real and it is also beside the point,
for two reasons.

**First, the walk-forward constant is "no term".** A term is scored at the
setting its own sweep chose on a prior season, and the 2025 sweep's answer at
every ballast is to use less of the park; extrapolated to the boundary the grid
runs to, that is a park factor of exactly 1.0 for every venue, which is the
gated model. The −0.00006 above is what the term is worth at a constant 2025
rejects. Reading the 2026 grid to pick the ballast is the thing the whole
protocol exists to prevent.

**Second, the sign does not replicate.** The standard every term above the
starter has been held to is that the sign is the same on all three sets,
because none of them is more than a standard error from zero on any one of
them. Park is negative on both 2026 sets (t −2.3 on the 756, −2.0 on all
1,781) and positive on 2025; defence is positive — worse than not having it —
on all three. Neither meets the standard. They do not ship.

### The grids — and the two seasons disagree about the sign

Both ballasts were swept walk-forward on **2025 only**, which is where the
constants have to be chosen. The same grids on the 756 market-priced 2026 games
are shown beside them; those were run after the fact and chose nothing.

**Park**, paired Brier against the gate:

| ballast (games) | 0 | 100 | 200 | 400 | 800 | inf |
|---|---|---|---|---|---|---|
| 2025 (n = 2,105) | +0.00004 | +0.00002 | +0.00002 | +0.00001 | +0.00001 | 0.00000 |
| t | +0.80 | +0.78 | +0.75 | +0.72 | +0.70 | — |
| 2026 (n = 756) | **−0.00013** | −0.00008 | −0.00006 | −0.00004 | −0.00002 | 0.00000 |
| t | −2.34 | −2.31 | −2.32 | −2.33 | −2.34 | — |

**Defence**, the same:

| ballast (BIP) | 0 | 1,000 | 2,000 | 4,000 | 8,000 | inf |
|---|---|---|---|---|---|---|
| 2025 (n = 2,105) | +0.00056 | +0.00013 | +0.00007 | +0.00003 | +0.00001 | 0.00000 |
| t | +1.04 | +0.60 | +0.47 | +0.36 | +0.28 | — |
| 2026 (n = 756) | +0.00060 | +0.00023 | +0.00013 | +0.00007 | +0.00003 | 0.00000 |
| t | +0.80 | +0.53 | +0.43 | +0.34 | +0.27 | — |

Read the park rows together and the verdict writes itself: **each season's grid
is monotone, and they point in opposite directions.** 2025 says use none of the
term and 2026 says use all of it, at every ballast, with the same |t| ≈ 2.3 on
2026 at every point — which is what happens when a term nudges every game in a
consistent direction by an amount too small to be worth anything. Even at its
own best point on 2026, the unregressed factors, the term is worth 0.00013 of
Brier against a market gap of 0.0023. The defence grids at least agree
with each other on the sign, and the sign is that the term is worse than not
having it.

Neither surface has an interior optimum. That is the same shape the
bullpen-availability sweep produced a station ago, and it means the same thing:
the sweep's own answer to "how much of this should we use" is "none". The 200
and 4,000 the scoreboard above uses are the middle of each grid — where the
term is most visible, not where either season would put it.

### Why park cannot pay, even though the factors are right

**A park is symmetric and a win probability is a ratio.** A park factor
multiplies both clubs' runs scored *and* runs allowed by the same number.
Pythagenpat reads `rs^x / (rs^x + ra^x)` with `x = (rs + ra)^0.287`: scaling
both by k leaves the ratio untouched and moves only the exponent. So the whole
park term reaches the answer through a second-order channel, and it shows up in
the size of the move — against the gate the park term shifts a game's
probability by a mean of **0.00048** (sd 0.00059, max 0.0061). The pen term,
the smallest thing on the board before today, moves 0.0061; the starter moves
about 0.04. Park is an order of magnitude below the smallest term we have.

The same symmetry is visible in the attribution table below: for every club the
park moves runs scored and runs allowed by almost exactly the same amount
(Colorado −0.188 and −0.229, Seattle +0.134 and +0.149), so it barely touches
the run *differential* that talent win% is made of. And that is the honest
answer to the question the residual posed: **park is a two-sided correction and
Atlanta's residual is one-sided.** Atlanta's runs scored agree between the two
halves to 0.06 of a run and its runs allowed disagree by 0.35; no symmetric
multiplier can explain a gap of that shape. Park was never the candidate.

### Why defence does not pay, even though it moves plenty

The defence term is not small: it shifts a game's probability by a mean of
**0.00442** (sd 0.0032, max 0.015), comparable to the pen term, and it closes
**16%** of the mean absolute top-down/bottom-up gap on the runs-allowed side.
It scores worse anyway, and the mechanism is the one `lineups.py` recorded a
station ago: **the top-down half already contains the club's defence**, because
it is the club's actual runs allowed. Adding the residual to the bottom-up half
takes the blend from carrying defence at half weight to carrying it at close to
full weight, and the sweep's verdict — the more of it you use the worse it gets
— says half was already the right amount. What the term supplies is not new
information; it is a second copy of information the blend was deliberately
holding at arm's length.

That also explains why the ballast grid is monotone rather than U-shaped. A
term that carried something new would be worth *something* at some weight.

### The blend weight does not move

The blend weight `w` was chosen walk-forward on 2025 when station C shipped —
an inverted U with an interior minimum at 0.5 — and the obvious worry about
adding terms to the bottom-up half is that the half is now better and deserves
more weight. Swept again on 2025 with both terms in (n = 2,105, Brier):

| w | 0 | 0.25 | **0.5** | 0.75 | 1.0 |
|---|---|---|---|---|---|
| the chain as served | 0.24433 | 0.24383 | **0.24360** | 0.24363 | 0.24395 |
| + park + defence | 0.24435 | 0.24388 | **0.24364** | 0.24365 | 0.24391 |

**The optimum does not move.** It is 0.5 with the terms and 0.5 without, with
the same flat floor out to 0.75, and at every weight below 1 the terms cost a
little.

The one cell where they pay is `w = 1`: a *pure* bottom-up run environment is
better with park and defence in it (0.24391) than without (0.24395). That is
the whole finding in one number. The two terms really do carry information the
components lack — they are worth something exactly where nothing else is
supplying it — and the moment the top-down half is mixed back in at any weight,
that information is already in the room.

### Attribution — every club's gap, before and after

`scripts/attribute_run_environment.py --season 2026 --as-of 2026-09-03`, the
day the chain was wired, 2,094 games played. `gap` is bottom-up minus top-down
runs per game; `park` is what neutralising the top-down half moves it; `defence`
is what the BABIP residual moves the bottom-up half; `left` is what neither
term explains.

| Club | RA top-down | RA bottom-up | RA gap | park | defence | left | RS gap | park | left |
|---|---|---|---|---|---|---|---|---|---|
| **MIL** | 4.00 | 4.37 | **+0.374** | +0.040 | −0.100 | **+0.234** | −0.236 | +0.053 | −0.289 |
| **ATL** | 4.03 | 4.39 | **+0.353** | +0.022 | −0.084 | **+0.247** | +0.058 | +0.026 | +0.032 |
| DET | 4.17 | 4.58 | +0.405 | −0.016 | −0.064 | +0.357 | +0.218 | −0.017 | +0.236 |
| NYY | 3.95 | 4.28 | +0.322 | +0.033 | −0.109 | +0.181 | +0.283 | +0.040 | +0.244 |
| MIA | 4.31 | 4.56 | +0.251 | −0.043 | −0.034 | +0.260 | +0.174 | −0.044 | +0.219 |
| SD | 4.26 | 4.51 | +0.249 | +0.031 | +0.077 | +0.295 | +0.331 | +0.031 | +0.300 |
| LAD | 4.03 | 4.11 | +0.073 | −0.023 | −0.148 | −0.052 | +0.408 | −0.030 | +0.438 |
| CWS | 4.46 | 4.56 | +0.094 | +0.033 | +0.039 | +0.100 | −0.034 | +0.035 | −0.070 |
| SEA | 4.44 | 4.26 | −0.188 | +0.149 | +0.134 | −0.203 | +0.559 | +0.134 | +0.426 |
| PHI | 4.32 | 4.10 | −0.214 | −0.051 | +0.080 | −0.084 | +0.267 | −0.054 | +0.321 |
| COL | 5.32 | 4.74 | −0.576 | −0.229 | +0.162 | −0.185 | −0.094 | −0.188 | +0.095 |
| ATH | 5.37 | 4.62 | −0.747 | −0.048 | +0.026 | −0.673 | +0.225 | −0.036 | +0.261 |

(Twelve of the thirty; the full table is what the script prints.) Colorado is
the one club park moves by more than a tenth of a run, and it moves its runs
scored by −0.188 and its runs allowed by −0.229 — the same correction twice,
which is exactly why it does not help a win probability.

Across the league the two terms close **16% of the mean |gap| on runs
allowed** (0.213 → 0.178 runs a game) and **nothing at all on runs scored**
(0.243 → 0.251 — park's neutralisation moves the runs-scored gap the wrong way
about as often as the right way, which is what a symmetric correction on a
one-sided residual does).

### What ships

Nothing, in the model. The two modules (`src/sim/park.py`,
`src/sim/defence.py`) ship, are unit-tested, are scored in the harness as three
new columns, and are wired through the same `game_model.home_win_probability`
both callers use, so the day one of them clears there is one line to change —
`NOT_SERVED` in `scripts/run_playoff_odds.py`. The gate rule
([architecture.md](architecture.md#3-the-gate-rule)) says the baseline runs
until a model beats it out of sample, and the nightly therefore asks the chain
for both terms switched off, which the agreement test pins bit for bit.

### Against the market

The other diagnostic every term on this board gets: regress the market's
deviation from the gate model on ours, over the 756 games.

| term | correlation | slope |
|---|---|---|
| park | 0.14 | **7.05** |
| defence | −0.02 | −0.13 |

The park term points where the market points and is **a seventh of the size** —
the largest under-reaction any term here has recorded (the pen term, the
previous record holder, was a factor of two). That is not evidence that a
bigger version of *this* term would work: a symmetric multiplier is bounded by
what Pythagenpat's exponent will do with it, and scaling it up is scaling up
the wrong channel. It is evidence that the market prices parks through
something our multiplier does not have — which is the component-factor
argument below.

The defence term has no relationship with the market's deviation at all
(correlation −0.02), which is what a term carries when the information in it is
already priced on both sides.

### What this measurement does not cover

- **Component park factors — home runs and balls in play — were not built.**
  The run factor comes from the schedule, which is one request per prior season
  and carries the score of every game; a home-run or BABIP factor needs every
  pitcher's game log for two completed seasons, ~1,700 player-season fetches
  that nothing else in the repository wants. That would have been worth paying
  if the run factor had shown the channel was live, and it did not — but the
  case for it is *not* dead, because a component factor enters differently: a
  home-run park makes a fly-ball pitcher worse and a ground-ball pitcher fine,
  which is asymmetric between the two clubs in a way a run multiplier never is.
  That asymmetry is the only version of park with a mechanism left.
- **Defence was only ever *added* to the bottom-up half.** The mirror
  construction — take the club's measured BABIP luck *out* of the top-down half
  and put the regressed estimate back in both — is the one the sweeps argue for
  (the model behaves as though it is carrying too much of this signal, not too
  little) and it is untested. It is a bigger change than this ticket's, because
  it edits the half of the blend that is a record of what happened.
- **Weather, travel and rest.** Still not in the model, and still inside the
  0.0023 the market holds.
- **The defence estimate is road-only by construction**, which halves the
  sample to keep the park out of it. A park-adjusted all-games BABIP would be
  the sharper estimator and needs the component factors above.

### Leakage

Stronger than the rest of the chain's, on the park side. A season's park
factors are built from the *completed prior seasons only* — no game of the
season being scored is in the pool at all, and
`tests/test_sim/test_park.py::TestLeakage` asserts the fetch asks for exactly
those years. A club's park exposure and its balls-in-play counts are cut
strictly before the date being predicted, the same cut every other frame gets,
with synthetic cases pinning that a game on the date itself, a game tomorrow,
and the first game of a doubleheader all leave the estimate identical
(`tests/test_sim/test_defence.py::TestLeakage`).
| **Kalshi close** | **0.24156** | **0.67589** | 0.537 |
| **Polymarket close** | **0.24165** | **0.67611** | 0.531 |
| `pythag_C_sp_bpa_ip_lu` (chain + posted card) | 0.24382 | 0.68062 | 0.532 |
| **`pythag_C_sp_bpa_ip` (the chain, the gate)** | **0.24388** | **0.68074** | 0.532 |
| blend (logistic stack of chain and learned) | 0.24392 | 0.68085 | 0.531 |
| logistic regression on the same features | 0.24452 | 0.68211 | 0.535 |
| **learned (gradient-boosted)** | **0.24461** | **0.68226** | 0.531 |
| `pythag_60` (production baseline) | 0.24619 | 0.68554 | 0.533 |
| **permuted-label control** | **0.24885** | **0.69085** | 0.537 |

Paired per-game Brier differences on those 756:

| comparison | difference | se | t |
|---|---|---|---|
| learned − chain | **+0.00073** | 0.00123 | +0.59 |
| blend − chain | +0.00004 | 0.00065 | +0.07 |
| blend − learned | −0.00069 | 0.00061 | −1.13 |
| logistic − chain | +0.00064 | 0.00105 | +0.61 |
| learned − `pythag_60` | −0.00158 | 0.00203 | −0.78 |
| chain − Kalshi | +0.00232 | 0.00132 | +1.75 |
| learned − Kalshi | +0.00305 | 0.00161 | +1.89 |

**The gate is not cleared.** The learned model is 0.00073 worse than the chain
on the game set the gate names, and the blend is a rounding error worse. Under
docs/architecture.md §3 the baseline runs.

### Scoreboard — the wider sets

All 1,779 scored games of 2026 (the same models, the same fit):

| Model | Brier | Log loss |
|---|---|---|
| learned | **0.24536** | 0.68384 |
| blend | 0.24537 | 0.68383 |
| `pythag_C_sp_bpa_ip_lu` | 0.24582 | 0.68473 |
| `pythag_C_sp_bpa_ip` (chain) | 0.24592 | 0.68490 |
| logistic | 0.24601 | 0.68514 |
| `pythag_60` | 0.24799 | 0.68916 |
| permuted control | 0.24972 | 0.69259 |

learned − chain **−0.00055** (se 0.00079, t = −0.70, n = 1,779); blend − chain
−0.00054 (se 0.00042, t = −1.31); learned − `pythag_60` −0.00263 (se 0.00134,
t = −1.97).

2025, trained on ≤2024, all 2,105 scored games — the replication:

| Model | Brier | Log loss |
|---|---|---|
| blend | **0.24334** | 0.67951 |
| learned | 0.24339 | 0.67971 |
| `pythag_C_sp_bpa_ip_lu` | 0.24356 | 0.67987 |
| `pythag_C_sp_bpa_ip` (chain) | 0.24360 | 0.67996 |
| logistic | 0.24376 | 0.68026 |
| `pythag_60` | 0.24509 | 0.68310 |
| permuted control | 0.24902 | 0.69118 |

learned − chain **−0.00021** (se 0.00065, t = −0.32, n = 2,105); blend − chain
−0.00026 (se 0.00038, t = −0.69).

So the sign flips with the game set — better on a whole season, worse on the
market's two months — and on none of the three sets is |t| above 1.4. That is
the shape of two models that are the same model.

**The permuted control lands where it must**: 0.24885 on the 756, 0.24972 on
all of 2026, 0.24902 on 2025, all within a few ten-thousandths of the 0.25 a
model with no information can reach, and 0.0035 worse than the real fit. The
protocol is not leaking.

### It reproduces the chain, term for term

Two readings of the same thing.

**Agreement.** The learned probability correlates **0.878** with the chain's on
2026 and **0.941** on 2025; regressing one log-odds on the other gives a slope
of 0.925 and 0.991, with almost the same spread (sd of the log-odds 0.285
against the chain's 0.271 on 2026). The mean absolute difference between the
two probabilities is 2.7 points on 2026 and 2.4 on 2025 — real disagreement
game to game, no disagreement on average, and no edge in either direction.

**Importance.** Gain, top ten of the 2026 model:

| feature | gain | what it is |
|---|---|---|
| `td_diff` | 10.5% | station D's regressed run differential, home − away |
| `sp_ra9_diff` | 9.3% | the two starters' FIP runs allowed per nine |
| `bu_diff` | 8.8% | station C's bottom-up run differential |
| `ra9_diff` | 5.7% | the blended runs-allowed rates |
| `pen_diff` | 5.3% | the available pens, as deltas from the league |
| `rs9_diff` | 3.6% | the blended runs-scored rates |
| `home_card_delta` | 3.1% | the home club's posted card against its own norm |
| `away_sp_ra9` | 2.9% | the visiting starter's own level |
| `sp_ip_diff` | 2.8% | the two starters' expected innings |
| `away_sp_ip` | 2.8% | the visiting starter's expected innings |

That is the chain's own ingredient list, in the chain's own order: team
strength first, the starter second, the pen third, the card fourth.

**The linear probe.** Regress each model's log-odds on the six chain terms,
standardised, and compare coefficients (2026 / 2025):

| term | chain | learned |
|---|---|---|
| `td_diff` | +0.116 / +0.151 | +0.085 / +0.131 |
| `bu_diff` | +0.084 / +0.104 | +0.071 / +0.123 |
| `sp_ra9_diff` | −0.132 / −0.138 | −0.112 / −0.123 |
| `pen_diff` | −0.033 / −0.040 | −0.059 / −0.044 |
| `card_delta` (vs `..._lu`) | +0.029 / +0.029 | +0.033 / +0.037 |
| `sp_ip_diff` | +0.002 / +0.004 | +0.041 / +0.037 |

A *linear* function of the chain's own six terms explains **81%** of the
learned model's log-odds on 2026 and **90%** on 2025. Same signs, nearly the
same magnitudes, slightly softer throughout — which is what a model fitted on
eleven seasons does to a form that was calibrated on one.

**What it contradicts.** Exactly two things, and only one of them replicates:

- **The starter's expected innings carry a level, not just a split.** The
  chain uses `sp_ip` only as the weight that divides the game between starter
  and pen, so its own probability barely responds to it (+0.002 / +0.004). The
  learned model puts ten to twenty times that on it (+0.041 / +0.037), the same
  sign in both seasons, and ranks it in its top ten. The reading: how deep a
  manager expects a starter to go is information about the club that the chain
  currently spends entirely on arithmetic. That is a lead — a term, not a
  ballast.
- **The pen may still be too timid.** The learned model puts 1.8× the chain's
  weight on `pen_diff` in 2026 — which agrees with what the market's own
  deviation already said (slope 1.97 against ours, §"Reading it") — but only
  1.1× in 2025. One season out of two is a hint, not a finding.

**What it ignores.** `rest_diff`, `home_sp_known`, `home_has_card` and
`away_has_card` get exactly zero gain; club rest, month, day/night and the
league's own runs allowed all get under half a percent. So on eleven seasons
of data a model free to use them found nothing in rest or in the calendar —
which is the same answer the pen-availability experiments gave, from the other
direction. Venue is the interesting middle: 1.7% of gain, 24th of 46 — a park
effect the chain does not model, small and real, and not enough to move a
score.

### Calibration

Isotonic, fitted on out-of-fold training-season games. On 2026's 1,779 games,
before and after:

| bucket | n (raw) | predicted | realized | n (calibrated) | predicted | realized |
|---|---|---|---|---|---|---|
| ≤ 0.40 | 66 | 0.372 | 0.303 | 105 | 0.385 | 0.305 |
| 0.40–0.45 | 153 | 0.431 | 0.425 | 27 | 0.427 | 0.556 |
| 0.45–0.50 | 399 | 0.478 | 0.506 | 615 | 0.489 | 0.488 |
| 0.50–0.55 | 486 | 0.525 | 0.512 | 459 | 0.528 | 0.536 |
| 0.55–0.60 | 375 | 0.574 | 0.573 | 282 | 0.584 | 0.582 |
| 0.60–0.65 | 196 | 0.624 | 0.582 | 210 | 0.628 | 0.576 |
| > 0.65 | 104 | 0.683 | 0.625 | 81 | 0.685 | 0.642 |

The one-number summary is the slope of a logistic recalibration, where 1.0 is
perfect and below 1.0 is overconfident: **0.857 → 0.928** on 2026 (intercept
−0.017 → −0.029) and **0.839 → 0.856** on 2025 (+0.022 → +0.018). The chain
itself sits at 0.915 on 2026 and 0.889 on 2025, so all three models are
overconfident by about the same amount, and the calibrator closes roughly half
of the learned model's excess. Brier moves accordingly and slightly: the
calibration is worth about a ten-thousandth, not a thousandth.

The soft spot is the same one every station-E model has had. Both the raw and
the calibrated model predict 0.68 on their biggest home favorites and realize
0.63–0.64, exactly where `pythag_C_sp_bpa_ip` predicts 0.68 and realizes 0.62.
A model with a free functional form, eleven seasons of training data and an
isotonic calibrator fitted on held-out games **did not fix the top bucket**,
which is fairly strong evidence that the overconfidence is not a shape problem
in the model — it is missing information about lopsided matchups.

### The blend

A logistic stack on the two log-odds, weights fitted on the same out-of-fold
training-season frame:

    logit(p) = −0.007 + 0.520 · logit(chain) + 0.536 · logit(learned)      [2026]
    logit(p) = −0.004 + 0.464 · logit(chain) + 0.581 · logit(learned)      [2025]

Two nearly equal weights summing to about 1.05 is the stack saying *these are
two noisy readings of one signal, average them and sharpen slightly*. Out of
sample it does what averaging two correlated estimates does: it beats the
worse one and ties the better one. On the 756 the blend is 0.24392 against the
chain's 0.24388 (+0.00004, t = +0.07) and the learned model's 0.24461
(−0.00069, t = −1.13). On all of 2026 it is 0.24537 against 0.24592 and
0.24536. On 2025, 0.24334 against 0.24360 and 0.24339 — its best showing, and
still 0.7 standard errors.

**The blend does not beat either alone anywhere it matters.** It is never
worse than the better of the two by more than a rounding error, which is the
honest thing to say for it, and it is never better by enough to be a model.

### What this is worth knowing

The chain was built one term at a time by a person choosing forms, and a
gradient-boosted model with eleven seasons, forty-six features and no
constraints on shape lands within a couple of ten-thousandths of it — better
on a full season, worse on the market's own subset, inside one standard error
everywhere. Three consequences:

1. **The remaining 0.0023 to the exchanges is not a functional-form problem.**
   If it were, this is the experiment that would have found it. Whatever the
   market knows that we do not is *information* — park, weather, travel,
   injuries, a better pitcher model, the specific arms available tonight — and
   no amount of flexibility over the inputs we already have will produce it.
2. **The hand-built terms are approximately right.** A free model given the
   raw pieces re-derives log5-on-Pythagenpat's ordering, the starter's weight,
   the pen's, and the card's, and softens all of them slightly. That is a
   validation of five stations' worth of choices that no amount of internal
   consistency-checking could have provided.
3. **The two disagreements are the leads.** The starter's expected innings as
   a level term is the concrete one, consistent across both seasons and
   currently worth nothing to the chain by construction. A pen delta at closer
   to twice its present weight is the other, and the market's own deviation
   said the same thing first.

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
- **Weather, rest and travel** — none of them are in the model, and the
  0.0023 the market still holds is where what is left of it lives. **Park is no
  longer on that list**: it was built and scored on Sept 3 and it is worth
  0.00006 of Brier on the 756 with the wrong sign on 2025, because a park
  multiplies both clubs' runs and a win probability is a ratio (the section
  above). Team defence, measured the same day, is worth less than nothing. Lineups and
  bullpen state *are* now in, and between them they were worth 0.0003 of the
  0.0033: most of the gap was never in either of them. Station C — the run
  environment rebuilt from the roster — took another 0.0003, the
  pitch-count-weighted available pen 0.0003 and the starter's own expected
  innings 0.0001, leaving 0.0023 (se 0.0013, t = 1.75). Rest is now *half* in:
  the reliever's rest is priced and worth nothing, the starter's is not priced
  at all.
- **What a reliever's rest is worth if it is not availability.** Two readings
  of "this arm is used up" — a binary three-days-running rule and a
  pitch-count weight that fires on a hundred times as many club-games — score
  the same to within 0.00002 Brier. Either the information is not in the
  workload, or the 3.5-inning delta on a workload-weighted average of eight
  arms is too blunt an instrument to carry it. Distinguishing those would need
  the *specific* arms a manager would have used tonight, which is a bullpen
  usage model rather than a workload rule.
- **A starter who replaces his rotation slot instead of adding to it.**
  `pythag_C_sp` adds the announced starter as the same delta from league
  average every other model adds him as, on top of a runs-allowed rate that
  already carries half a rotation term over the same 5.5 innings — a mild
  double-count, and precisely the one `pythag_60_sp` already makes against a
  team RA/9 containing the club's whole rotation. Keeping it identical is what
  makes the gate comparison exact. Swapping tonight's starter *into* C's
  rotation slot is the cleaner construction and has not been scored.
- **Which half of station C is doing the work.** The blend applies one weight
  to runs scored and runs allowed together, so the hitters-and-playing-time
  half and the rotation-and-pen half were never separated. A two-weight sweep
  on 2025 would say whether C is really a pitching-staff term wearing a run
  environment's clothes.
- **The starter's expected innings as a level term.** The learned challenger
  reads `sp_ip` at ten to twenty times the weight the chain's own probability
  responds to it with, the same sign on both seasons, because the chain spends
  that quantity entirely on dividing the game between starter and pen. Pricing
  how deep a manager expects a start to go as information about the club — not
  only as the split point — is the one concrete lead the experiment produced,
  and it is unscored.
- **Whether flexibility helps anywhere the inputs are richer.** The learned
  model was given the same information as the chain, and the finding is that
  the form was never the binding constraint. It says nothing about what a
  model would do with park, weather, travel or a pitch-level starter model,
  because none of those are in the table.
- **A morning-of bullpen or lineup.** Same caveat as the probable pitcher: the
  boxscore's posted order equals the card the club filed except for late
  scratches, which the backfill silently absorbs, and relief usage is read from
  finished games. The exchanges' closes (median 15 minutes before first pitch)
  knew about the scratches too, so the comparison is fair; it is not a
  simulation of forecasting at 9am.
