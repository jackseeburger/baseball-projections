# Playoff Odds vs. Public Systems (roadmap 2.6 acceptance)

**First run:** Sept 1, 2026 · 20,000 sims · 2,070 games played, 361 remaining ·
HFA 0.534 (season-to-date .5275 shrunk toward .540) · v0 team strength =
Pythagenpat on run rates regressed 60 games toward league average.
Reproduce: `python scripts/compare_public_odds.py --date 2026-09-01`.

**Acceptance criterion:** odds within a few points of FanGraphs / Baseball
Prospectus for most teams. **Met.**

| Metric | Mean abs. gap vs FanGraphs |
|---|---|
| P(playoffs) | 1.5 pts |
| P(division) | 2.0 pts |
| P(bye) | 2.4 pts |
| P(pennant) | 2.3 pts |
| P(World Series) | 1.5 pts |
| Expected wins | 0.65 wins |

24 of 30 teams are within 2 points on P(playoffs); 27 of 30 within 5.

## Where we diverge, and why it's the strength model

| Team | Ours | FG | Gap | Cause |
|---|---|---|---|---|
| CWS P(division) | 83.9 | 68.7 | +15.2 | AL Central: we rate CWS .521 / CLE .499 from *runs*; FG's roster projections think Cleveland's talent exceeds its run differential. |
| CLE P(division) | 14.9 | 27.8 | −12.9 | Same race, other side. |
| MIL P(WS) | 19.8 | 11.0 | +8.8 | Run-differential strength rewards Milwaukee's season; FG's roster-based (and playoff-rotation-aware) strength favors the Dodgers. |
| LAD P(WS) | 17.2 | 24.7 | −7.6 | |

Every material gap traces to **team strength v0 being outcome-based rather
than roster-based**. That is the roadmap 1.5 upgrade (aggregate player
projections → projected runs scored/allowed) and it now has a measured
target: close the AL Central and MIL/LAD gaps without hurting the 24 teams
that already agree. ~~The swap point is
`src/sim/strength.from_run_environment`.~~ **Made on Sept 3, 2026** — station
C's blend is the strength the board is served off, and both named gaps close
from our side; see [the last section](#sept-3-2026--the-full-chain-is-wired-and-one-function-serves-it).

Secondary candidates, in order of expected payoff:
1. ~~Starting-pitcher adjustment when probables are posted~~ — **wired in Sept 2, 2026**, see below.
2. Tune `regress_games` (60) against historical final standings via the backtest harness.
3. Extend tiebreakers past the coin-flip floor (rarely binding).

---

## Sept 2, 2026 — starting-pitcher term wired into the nightly job (station E)

`pythag_60_sp` cleared the station E gate (Brier **.2448** vs the production
model's **.2462** on the 756 market-priced 2026 games —
[market-benchmark-2026.md](market-benchmark-2026.md)), so per the gate rule
(architecture §3) it now runs in production. For every remaining regular-season
game whose probable starters the Stats API has posted, the season Monte Carlo
uses the starter-adjusted P(home) instead of log5 on team strength. Games
without both probables keep team strength, which is the correct
rotation-average expectation for a game whose starters nobody has announced.
`--no-starters` reverts to the old behaviour.

Reproduce: `python3 scripts/run_playoff_odds.py --sims 5000 --dry-run`.

**Run:** 2,087 games played, 344 remaining, HFA 0.5336, 5,000 sims, seed 245.
**27 of the 344 remaining games** had both probables posted inside the 7-day
window (0 starter slots had no history). Both tables below come from the same
seed, so the draw is identical on the other 317 games.

| Team | P(playoffs) w/o | with | Δ | P(WS) w/o | with | Δ | Δ mean wins |
|---|---|---|---|---|---|---|---|
| MIL | 100.00 | 100.00 | 0.00 | 19.96 | 20.16 | +0.20 | +0.11 |
| LAD | 100.00 | 100.00 | 0.00 | 14.96 | 15.24 | +0.28 | +0.09 |
| ATL | 100.00 | 100.00 | 0.00 | 11.38 | 10.48 | −0.90 | −0.11 |
| TB | 100.00 | 100.00 | 0.00 | 10.52 | 10.32 | −0.20 | +0.08 |
| NYY | 100.00 | 100.00 | 0.00 | 10.16 | 10.10 | −0.06 | +0.02 |
| CWS | 95.76 | 95.54 | −0.22 | 9.02 | 9.08 | +0.06 | −0.03 |
| CHC | 99.48 | 99.36 | −0.12 | 7.96 | 7.60 | −0.36 | −0.11 |
| BOS | 98.52 | 98.52 | 0.00 | 5.16 | 5.56 | +0.40 | −0.01 |
| PHI | 99.48 | 99.52 | +0.04 | 2.88 | 2.90 | +0.02 | +0.10 |
| CLE | 56.42 | 55.88 | −0.54 | 2.40 | 2.34 | −0.06 | −0.01 |

**max abs Δ P(playoffs) = 0.54 pts (CLE) · max abs Δ P(WS) = 0.90 pts (ATL) ·
max abs Δ expected wins = 0.11 (MIL).**

### Those deltas are below the simulation's own noise floor

Re-running the *unchanged* model at three different seeds moves the same
numbers as much or more:

| Seeds compared | max abs Δ P(playoffs) | max abs Δ P(WS) | max abs Δ mean wins |
|---|---|---|---|
| 245 vs 246 | 1.18 pts | 0.94 pts | 0.10 |
| 245 vs 247 | 1.96 pts | 1.32 pts | 0.14 |
| 246 vs 247 | 0.88 pts | 1.48 pts | 0.10 |

Re-running the same comparison at the nightly job's 20,000 sims shrinks the
biggest P(WS) gap from 0.90 pts to **0.41 pts** while the closed-form numbers
below do not move at all — which is what a quantity made of sampling noise
does, and what a real effect does not. So at 5,000 sims the odds columns
cannot resolve this term at all, and the table above should be read as
"nothing moved", not as a measurement. This is
the same lesson the coin-flip control taught in the first run (architecture
§3: *September playoff odds can't distinguish models*) — it is a property of
the question, not a defect in the term.

### The term's actual footprint, in closed form

Expected wins are additive in the per-game probabilities, so the term's effect
can be computed exactly, with no sampling. The script prints this:

```
27 games repriced: mean |Δ P(home)| = 0.0279, max = 0.0869
largest expected-win shifts over those games:
  CHC -0.113, MIL +0.113, ATL -0.112, PHI +0.095, STL -0.083, LAD +0.083
```

Knowing who is pitching moves a single game's win probability by **2.8 points
on average and up to 8.7**, which is a large effect per game — it is exactly
the effect that took Brier from .2462 to .2448. It does not move September
playoff odds because it reaches only 27 of 344 remaining games and each team
plays about two of them, so the biggest expected-win shift on the whole board
is a ninth of a win. (Those closed-form shifts agree with the simulated
`Δ mean wins` column to two decimals, which is the cross-check that the
override really is reaching the Monte Carlo.)

### Why September is the term's worst case, and where it will matter

The standings are mostly banked: 2,087 of 2,431 games are played, so 86% of
every team's record is fixed and no per-game model can move it. The term
matters where the remaining schedule is short relative to the decision:

1. **The last week**, when a division race turns on three or four games and
   the fraction of remaining games with posted probables goes to ~1.
2. **The postseason bracket**, where a seven-game series is decided by four
   announced starters and rotation quality is most of the edge. ~~The bracket
   still runs on team strength today.~~ **Wired the same day** — see
   [the next section](#sept-2-2026--the-starter-term-reaches-the-postseason-bracket),
   where it turns out to be the first version of this term big enough to
   clear the simulation's own noise floor.
3. **Per-game odds and prices**, which is where the term already earns its
   keep and where it is scored (station E, not station G).

Being invisible here is the expected result, and it is why the gate for this
change was the per-game Brier score and not this table.

---

## Sept 2, 2026 — the starter term reaches the postseason bracket

The section above ends by naming the two places the starting-pitcher term was
expected to matter and did not yet reach. This is the second of them: **the
bracket now prices every game of every series off the arm whose turn it is.**

Each club carries an ordered rotation — `(pitcher_id, RA/9 delta from league
average)`, ace first — and `bracket.play_series` walks it game by game,
wrapping after `--rotation-size` (default 4), so a best-of-7 starts the same
pitcher in games 1 and 5. A club with no rotation is priced on team strength
for its games; because the rotation enters as a *delta*, only the side that
has one moves, exactly as in `starters.blend_starter_team`.

The bracket is handed talent win%, not run rates, so `strength_with_starter`
inverts Pythagenpat at the club's run environment, moves runs allowed by
`5.5/9 · delta`, and converts back. The inversion is exact, so a
league-average starter returns the club's strength unchanged to the last bit
and `rotations=None` reproduces the pre-rotation bracket draw for draw.

**Where the rotation comes from.** One rate table, the same one the
regular-season overrides use (`run_playoff_odds.starter_terms` builds it
once): the six pitchers with the most starts for the club this season, ranked
by regressed FIP, best four kept. A postseason series already inside the
7-day probables window is priced off the announced starters by the
regular-season path; the bracket rotation is what fills in every series that
has not been announced yet, which in early September is all of them.

Reproduce:
`python3 scripts/run_playoff_odds.py --sims 5000 --dry-run --cached-pitchers --rotation-compare`.

**Run:** 2,089 games played, 342 remaining, HFA 0.5336, 5,000 sims, seed 245.
27 of the 342 remaining games had both probables posted; all 30 clubs got a
4-man rotation. Both runs below share the seed *and* the 27 regular-season
overrides, so the only thing that differs is how the bracket prices a series.

| | max abs Δ | team |
|---|---|---|
| **P(pennant)** | **4.68 pts** | CHC |
| **P(World Series)** | **3.98 pts** | MIL |
| P(playoffs) | 0.02 pts | BAL |

P(playoffs) is untouched by construction — a rotation cannot change who
reaches October, only who survives it — and the 0.02 is a single simulation
whose wild-card tiebreaker landed differently.

### Top 5 movers, with the arms that moved them

| Team | Δ P(pennant) | Δ P(WS) | Game 1 | Δ RA/9 | Game 2 | Δ RA/9 |
|---|---|---|---|---|---|---|
| CHC | −4.68 | −2.94 | Shota Imanaga | −0.084 | Matthew Boyd | −0.035 |
| CWS | −4.58 | −2.94 | Bryan Hudson | −0.129 | Sean Burke | −0.034 |
| MIL | +2.98 | +3.98 | Jacob Misiorowski | −1.153 | Logan Henderson | −0.702 |
| TB | +2.90 | +0.58 | Griffin Jax | −0.757 | Drew Rasmussen | −0.537 |
| NYY | +2.20 | +0.94 | Cam Schlittler | −0.736 | Max Fried | −0.492 |

The sign is the whole story. Milwaukee and Tampa Bay have the two arms at the
top of the board and gain; the Cubs and the White Sox have staffs that
regressed FIP puts within a tenth of a run of league average and lose, because
in a short series they now face somebody else's ace twice with nothing to
answer it. Team strength cannot see this at all: it prices a series off two
season-long run differentials, which is the *rotation average*, and a
rotation average is the wrong statistic for a seven-game series.

### Unlike the regular-season term, this one clears the noise floor

The lesson of the first starter section was that September playoff odds
cannot resolve a term that reaches 27 of 344 games. This one is different,
and the same test says so. Re-running the *unchanged* model at three seeds:

| Compared | max abs Δ P(playoffs) | max abs Δ P(pennant) | max abs Δ P(WS) |
|---|---|---|---|
| seeds 245 vs 246 | 1.78 | 2.22 | 1.46 |
| seeds 245 vs 247 | 0.94 | 1.64 | 0.64 |
| seeds 246 vs 247 | 0.84 | 1.04 | 1.30 |
| **rotations on/off, seed 245** | **0.02** | **4.68** | **3.98** |
| **rotations on/off, seed 246** | **0.00** | **6.64** | **5.46** |

Two things separate a real effect from sampling noise here, and both hold.
The rotation swing is two to three times the seed-to-seed spread, and it
**replicates**: run the on/off comparison again at seed 246 and the same
clubs move the same way (CWS −6.64, MIL +5.78, CHC −5.18, TB +4.12,
NYY +3.24). Noise does not name the same five teams twice.

That it moves P(pennant) more than P(WS) is the mechanism showing through:
the term compounds over three or four rounds, and by the World Series the
biggest movers are meeting each other.

### Against the market

**Kalshi lists no `KXMLBSERIES` markets right now — none open, and none
unopened, closed or settled either.** The series exists on the exchange but
individual postseason series are not listed until the matchups are set, so
there is nothing yet to score a series probability against. That comparison
has to wait for October, and it is the one that matters for this term:
`KXMLBSERIES` prices exactly the quantity the bracket now models.

What *is* open and liquid is the season-long futures board — `KXMLB` (World
Series, LAD 28.6/29.7¢, ~75k contracts in 24h), `KXMLBAL` and `KXMLBNL`
(pennants) — 30 and 15 markets with one- and two-cent spreads. Comparing
our P(pennant) and P(WS) to the yes-side mid (not de-vigged; the AL and NL
pennant books sum to 1.025 and 1.030, the WS book to 0.995):

| Mean abs gap vs Kalshi mid | all 30 | top 10 by WS price |
|---|---|---|
| P(pennant), bracket on team strength | 2.44 | 5.70 |
| P(pennant), bracket on rotations | **2.27** | **5.14** |
| P(WS), bracket on team strength | **1.48** | 3.91 |
| P(WS), bracket on rotations | 1.51 | **4.01** |

Rotations move us slightly *toward* the market on the pennant and slightly
away on the World Series, and every one of those moves is smaller than the
gap that was already there. Read it as "no contradiction", not as evidence
either way: the board is dominated by two disagreements that predate this
change and that the first section already diagnosed — we are 12 points under
Kalshi on the Dodgers and 12–17 over on the Brewers, both of which trace to
**team strength v0 being outcome-based rather than roster-based**, and
rotations make the Milwaukee gap wider because Milwaukee also has the best
arm on the board. This is a futures snapshot, not a score; the scoreable
version is per-series prices in October.

### What this first pass does not know

The rotation is picked by a rule, not by a manager, and the rule is crude on
purpose. Every one of these is visible in the September 2 pools:

1. **Six-by-starts, then four-by-FIP.** The White Sox lead with Bryan Hudson,
   who has 7 starts, ahead of Sean Burke's 26 — and a better arm with 5
   starts fell outside the six-deep pool entirely. The pool guards against
   ranking a reliever's 40 innings first; it does not guard against a
   swingman with a dozen.
2. **Openers and bullpen games** are counted as starts, so an opener can be
   handed Game 1, and a club that plans to cover Game 4 with the bullpen is
   modelled as starting somebody.
3. **Order is by FIP, not by the manager.** The Yankees line up Schlittler
   ahead of Cole and Fried. Regressed FIP may even be right; it is still not
   what the lineup card will say.
4. **Mid-season trades split a pitcher across both clubs**, so a July
   acquisition looks shallower on his new team than he is.
5. **Health, roster eligibility and rest are all invisible.** A club that
   clinches early and lines its ace up on extra rest gets no credit; an
   injured starter is still in the rotation.
6. **Home-field advantage is not re-estimated per series**, and the delta is
   applied at a league-average run environment where a club's own is unknown.

The fix for 1-3 is the same fix: use the announced probables once the series
is scheduled, which the code already prefers where they exist. Until then
this is a rotation-shaped prior, and the honest claim for it is the one the
numbers support — it is a *large* change to the postseason answer, and it is
not yet scored against anything. `KXMLBSERIES` in October is the exam.

---

## Sept 3, 2026 — the full chain is wired, and one function serves it

The two sections above wired the starting-pitcher term into the regular season
and then into the bracket. Four more terms had cleared the station E gate and
were still sitting outside production: station C's bottom-up run environment,
the posted lineup, the pen that is actually available, and the starter's own
expected innings. Together they are `pythag_C_sp_bpa_ip`, **Brier .24388 on
the 756 market-priced games against the production model's .24619**
([market-benchmark-2026.md](market-benchmark-2026.md)). The gate rule
(architecture §3) says a model that beats its baseline out of sample runs in
production, so it runs.

Three things changed.

1. **One function prices a game.** `src/sim/game_model.py` holds the whole
   chain — `ChainInputs` (the season-long frames), `build_slate` (one date,
   every frame cut to games strictly before it) and `home_win_probability`
   (one game). `scripts/backtest_game_odds.py` computes its
   `pythag_C_sp_bpa_ip` column through it and `scripts/run_playoff_odds.py`
   prices tonight's games through it, so the number on the scoreboard and the
   number on the site are one call, not two implementations that agree today.
2. **Team strength is station C.** Every remaining game the simulator draws
   without a named starter — and every postseason series game the bracket's
   rotation deltas bend — starts from the blended run environment
   (`src/sim/run_environment.py`, w = 0.5) instead of the standings' regressed
   run differential.
3. **The announced games get the whole stack.** For a remaining game with both
   probables posted: the starter's regressed FIP over *his own* expected
   innings, the availability-weighted pen over the rest, and the club's posted
   card when one is up. With no card up the lineup term is exactly zero,
   because a club's own recent cards are what its card is measured against —
   which is why the same function covers every remaining game of the season
   and why the horizon is a boundary in what is *known*, not in how it is
   priced.

`--legacy-chain` reproduces the previous pricing (station D strength, the
starting-pitcher term alone) and `--no-starters` still reverts to team
strength everywhere.

Reproduce (same seed, same sims, same day → the same tables):

```
python scripts/run_playoff_odds.py --sims 20000 --dry-run --legacy-chain
python scripts/run_playoff_odds.py --sims 20000 --dry-run
python scripts/run_playoff_odds.py --sims 20000 --coin-flip
```

### The agreement test

`tests/test_sim/test_chain_agreement.py` builds a whole synthetic season — 4
clubs, 45 dates, every pitching appearance, every hitting line, every posted
card — and serves it to *both* scripts through their own fetch functions. The
backtest sees the last date as played and scores it walk-forward; the odds job
sees the same date as scheduled and prices it from the same history, with
standings built from exactly the games the backtest sums. The two P(home)
values agree to **1e-12**, with and without a posted card, and a third test
pins the property the horizon rests on: the chain asked about a game with
nobody announced returns exactly the log5 probability the Monte Carlo draws
that game with.

On the real season the refactor is a no-op, which is the other half of the
claim: re-running the harness after it, every model's Brier is unchanged to
five decimals (`pythag_60` .24619, `pythag_60_sp` .24483,
`pythag_60_sp_lu_bp` .24454, `pythag_C_sp` .24428, `pythag_C_sp_bpa` .24400,
`pythag_C_sp_bpa_ip` .24388) and the per-game frames are **identical to the
last bit** on all 15 columns.

The card branch is scored too, as a new column: `pythag_C_sp_bpa_ip_lu` gets
**.24382** on the 756, a paired −0.00006 against `pythag_C_sp_bpa_ip`
(se 0.00025, t = −0.25, n = 756) — the same "inside one standard error"
verdict every term above the starter gets. It fires on a live slate only when
a club has already published its card, which at the nightly job's hour is no
game at all.

### What it costs

The chain needs the whole population on both sides of the ball — the pen
window, the rotation window, last night's pitch counts and each club's plate
appearances are all questions about players nobody announced — so the nightly
now pulls every pitcher's and every hitter's game log for the season (1,494
player-seasons) instead of the 185 pitchers tonight's probables and the 30
rotation pools happen to name. Eight at a time (`--chain-workers`, cached
under `data/cache/statsapi/` as always):

| | wall | CPU | fetched |
|---|---|---|---|
| `--legacy-chain` (the old job) | **1m 24.7s** | 23.7s | probables + 185 pitcher logs |
| the full chain (what runs now) | **1m 33.9s** | 1m 38.5s | probables + 844 pitcher logs + 650 hitter logs + 29 live feeds |

Nine seconds of wall clock, on a workflow whose only other steps are the
projection and accuracy builds. The extra CPU is the rate tables now covering
1,300 pitchers and 1,062 hitters rather than the handful on tonight's mound.

### Before and after, every club

**Run:** 2026-09-03, 2,091 games played, 340 remaining, HFA 0.5336, 20,000
sims, seed 246 (day of year). 29 of the 340 remaining games had both probables
posted inside the 7-day window (0 starter slots with no history); 9 of those
29 also had a card up, because the job was run while those games were being
played. All three columns share the seed and the state, so every difference is
the model.

| Club | Div | W-L | playoffs old | new | Δ | division old | new | Δ | pennant old | new | Δ | WS old | new | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LAD | NL West | 82-56 | 100.0 | 100.0 | -0.0 | 100.0 | 100.0 | +0.0 | 26.4 | 34.0 | +7.6 | 16.8 | 22.6 | +5.8 |
| MIL | NL Central | 86-53 | 100.0 | 100.0 | +0.0 | 99.1 | 98.4 | -0.6 | 37.7 | 28.9 | -8.7 | 25.5 | 17.7 | -7.7 |
| TB | AL East | 83-55 | 100.0 | 100.0 | +0.0 | 83.4 | 84.3 | +0.9 | 27.5 | 27.2 | -0.3 | 11.2 | 12.0 | +0.7 |
| NYY | AL East | 79-60 | 100.0 | 100.0 | +0.0 | 16.3 | 15.4 | -1.0 | 24.0 | 23.8 | -0.2 | 11.2 | 11.3 | +0.1 |
| ATL | NL East | 83-58 | 100.0 | 100.0 | +0.0 | 91.8 | 84.6 | -7.2 | 17.2 | 13.9 | -3.3 | 10.0 | 7.4 | -2.6 |
| PHI | NL East | 79-61 | 99.2 | 99.7 | +0.4 | 8.2 | 15.4 | +7.2 | 7.9 | 12.6 | +4.6 | 4.2 | 6.9 | +2.7 |
| BOS | AL East | 75-65 | 97.9 | 97.2 | -0.7 | 0.3 | 0.3 | +0.0 | 13.1 | 12.5 | -0.6 | 5.4 | 5.2 | -0.1 |
| CWS | AL Central | 73-65 | 96.0 | 94.3 | -1.8 | 84.4 | 81.3 | -3.1 | 17.9 | 15.1 | -2.8 | 5.4 | 4.7 | -0.7 |
| CHC | NL Central | 78-61 | 99.2 | 99.2 | +0.1 | 0.9 | 1.6 | +0.6 | 8.3 | 7.6 | -0.7 | 4.5 | 4.0 | -0.6 |
| CLE | AL Central | 70-68 | 59.6 | 57.5 | -2.1 | 14.7 | 17.2 | +2.6 | 7.0 | 7.0 | +0.0 | 2.2 | 2.4 | +0.2 |
| HOU | AL West | 70-69 | 66.3 | 67.5 | +1.2 | 63.9 | 64.8 | +0.9 | 4.5 | 6.7 | +2.2 | 1.1 | 1.9 | +0.9 |
| TEX | AL West | 69-71 | 31.1 | 29.8 | -1.3 | 25.4 | 23.5 | -1.9 | 2.5 | 3.0 | +0.5 | 0.5 | 0.9 | +0.4 |
| AZ | NL West | 74-67 | 49.9 | 50.6 | +0.7 | 0.0 | 0.0 | -0.0 | 1.2 | 1.5 | +0.3 | 0.5 | 0.6 | +0.1 |
| SEA | AL West | 66-74 | 11.8 | 13.0 | +1.2 | 10.7 | 11.7 | +1.0 | 0.9 | 1.5 | +0.6 | 0.3 | 0.5 | +0.3 |
| TOR | AL East | 68-71 | 13.6 | 15.6 | +1.9 | 0.0 | 0.0 | +0.0 | 0.7 | 1.4 | +0.6 | 0.1 | 0.5 | +0.4 |
| SD | NL West | 73-67 | 40.4 | 39.6 | -0.8 | 0.0 | 0.0 | -0.0 | 1.0 | 1.2 | +0.2 | 0.3 | 0.5 | +0.1 |
| BAL | AL East | 69-71 | 16.0 | 16.1 | +0.1 | 0.0 | 0.0 | +0.0 | 1.1 | 1.1 | +0.0 | 0.3 | 0.4 | +0.1 |
| MIN | AL Central | 67-72 | 7.0 | 8.7 | +1.8 | 0.9 | 1.4 | +0.5 | 0.5 | 0.7 | +0.1 | 0.1 | 0.2 | +0.1 |
| MIA | NL East | 70-69 | 7.9 | 7.2 | -0.7 | 0.0 | 0.0 | +0.0 | 0.2 | 0.2 | +0.0 | 0.1 | 0.1 | -0.0 |
| PIT | NL Central | 68-71 | 1.4 | 1.9 | +0.5 | 0.0 | 0.0 | +0.0 | 0.1 | 0.1 | +0.0 | 0.0 | 0.1 | +0.0 |
| DET | AL Central | 63-75 | 0.7 | 0.4 | -0.3 | 0.1 | 0.0 | -0.0 | 0.1 | 0.1 | -0.1 | 0.1 | 0.0 | -0.0 |
| STL | NL Central | 69-70 | 1.9 | 1.8 | -0.1 | 0.0 | 0.0 | +0.0 | 0.1 | 0.1 | -0.0 | 0.0 | 0.0 | +0.0 |
| WSH | NL East | 67-75 | 0.1 | 0.1 | -0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| CIN | NL Central | 67-73 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| KC | AL Central | 62-77 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | +0.0 |
| NYM | NL East | 62-77 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| COL | NL West | 54-86 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| LAA | AL West | 53-86 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| SF | NL West | 58-82 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| ATH | AL West | 54-86 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |

The same board against the control this document has scored against since the
first run — **no strength model at all, every club a .500 coin flip**
(`--coin-flip`):

| Club | playoffs chain | coin flip | Δ | division chain | coin flip | Δ | pennant chain | coin flip | Δ | WS chain | coin flip | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LAD | 100.0 | 100.0 | +0.0 | 100.0 | 99.6 | +0.3 | 34.0 | 19.4 | +14.6 | 22.6 | 10.0 | +12.5 |
| MIL | 100.0 | 100.0 | +0.0 | 98.4 | 97.8 | +0.6 | 28.9 | 25.2 | +3.7 | 17.7 | 13.0 | +4.7 |
| TB | 100.0 | 100.0 | +0.0 | 84.3 | 88.9 | -4.6 | 27.2 | 25.0 | +2.2 | 12.0 | 12.3 | -0.3 |
| NYY | 100.0 | 100.0 | +0.0 | 15.4 | 10.7 | +4.7 | 23.8 | 14.4 | +9.4 | 11.3 | 7.0 | +4.3 |
| ATL | 100.0 | 100.0 | +0.0 | 84.6 | 86.1 | -1.5 | 13.9 | 19.8 | -5.9 | 7.4 | 10.1 | -2.6 |
| PHI | 99.7 | 99.4 | +0.3 | 15.4 | 13.9 | +1.5 | 12.6 | 12.4 | +0.2 | 6.9 | 6.3 | +0.6 |
| BOS | 97.2 | 95.2 | +1.9 | 0.3 | 0.4 | -0.0 | 12.5 | 10.8 | +1.7 | 5.2 | 5.4 | -0.2 |
| CWS | 94.3 | 92.6 | +1.7 | 81.3 | 79.9 | +1.4 | 15.1 | 19.7 | -4.6 | 4.7 | 9.5 | -4.8 |
| CHC | 99.2 | 98.1 | +1.1 | 1.6 | 2.2 | -0.6 | 7.6 | 12.3 | -4.8 | 4.0 | 6.3 | -2.3 |
| CLE | 57.5 | 53.7 | +3.8 | 17.2 | 18.1 | -0.9 | 7.0 | 7.9 | -0.9 | 2.4 | 3.8 | -1.4 |
| HOU | 67.5 | 68.2 | -0.7 | 64.8 | 65.1 | -0.3 | 6.7 | 10.8 | -4.1 | 1.9 | 5.2 | -3.2 |
| TEX | 29.8 | 36.8 | -7.0 | 23.5 | 28.7 | -5.2 | 3.0 | 5.2 | -2.2 | 0.9 | 2.7 | -1.8 |
| AZ | 50.6 | 45.6 | +4.9 | 0.0 | 0.3 | -0.2 | 1.5 | 5.0 | -3.5 | 0.6 | 2.5 | -1.8 |
| SEA | 13.0 | 7.1 | +5.9 | 11.7 | 6.3 | +5.4 | 1.5 | 1.0 | +0.5 | 0.5 | 0.4 | +0.1 |
| TOR | 15.6 | 14.2 | +1.3 | 0.0 | 0.0 | +0.0 | 1.4 | 1.6 | -0.3 | 0.5 | 0.8 | -0.3 |
| SD | 39.6 | 41.5 | -1.9 | 0.0 | 0.1 | -0.1 | 1.2 | 4.3 | -3.1 | 0.5 | 2.2 | -1.8 |
| BAL | 16.1 | 22.4 | -6.3 | 0.0 | 0.0 | +0.0 | 1.1 | 2.4 | -1.3 | 0.4 | 1.2 | -0.8 |
| MIN | 8.7 | 9.4 | -0.7 | 1.4 | 1.9 | -0.6 | 0.7 | 1.2 | -0.6 | 0.2 | 0.6 | -0.4 |
| MIA | 7.2 | 11.1 | -3.9 | 0.0 | 0.0 | +0.0 | 0.2 | 1.2 | -1.0 | 0.1 | 0.6 | -0.5 |
| PIT | 1.9 | 1.4 | +0.5 | 0.0 | 0.0 | +0.0 | 0.1 | 0.1 | -0.0 | 0.1 | 0.1 | -0.0 |
| DET | 0.4 | 0.3 | +0.1 | 0.0 | 0.0 | +0.0 | 0.1 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| STL | 1.8 | 2.7 | -1.0 | 0.0 | 0.0 | +0.0 | 0.1 | 0.2 | -0.2 | 0.0 | 0.1 | -0.1 |
| WSH | 0.1 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| CIN | 0.0 | 0.1 | -0.1 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | +0.0 |
| KC | 0.0 | 0.1 | -0.1 | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | -0.0 |
| NYM | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| COL | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| LAA | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| SF | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |
| ATH | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 | 0.0 | 0.0 | +0.0 |

### Where the moves come from

Almost all of it is the strength swap, not the per-game terms: the chain
reprices 29 of 340 remaining games, while station C's run environment reaches
every one of them and every postseason series. The per-club run environment
(top-down = the regressed season-to-date rates, bottom-up = the hitters and
staff on hand, blend = half and half):

| Club | RS/G top-down → bottom-up | RA/G top-down → bottom-up | rotation FIP | pen FIP | strength | Δ odds |
|---|---|---|---|---|---|---|
| **MIL** | 4.84 → 4.61 | 4.00 → 4.25 | 4.09 | 4.51 | .589 → .563 | pennant −8.7, WS −7.7 |
| **LAD** | 4.78 → 5.19 | 4.03 → 4.11 | 4.03 | 4.23 | .579 → .594 | pennant +7.6, WS +5.8 |
| **PHI** | 4.50 → 4.76 | 4.32 → 4.10 | 4.04 | 4.19 | .519 → .544 | division +7.2, pennant +4.6 |
| **ATL** | 4.61 → 4.65 | 4.02 → 4.38 | 4.43 | 4.31 | .563 → .545 | division −7.2, pennant −3.3 |
| **CWS** | 4.69 → 4.65 | 4.46 → 4.56 | 4.57 | 4.53 | .523 → .517 | division −3.1, pennant −2.8 |
| **SEA** | 4.12 → 4.68 | 4.44 → 4.25 | 4.17 | 4.39 | .465 → .505 | playoffs +1.2, division +1.0 |

Reading the six:

- **Milwaukee (−8.7 pennant, −7.7 WS) is the club whose record its roster
  least supports.** It has scored 4.84 and allowed 4.00 per game; the hitters
  actually taking its plate appearances project 4.61, and its staff — the
  third-best rotation by regressed FIP at 4.09, in front of a pen at 4.51,
  a shade worse than the league's 4.48 — projects 4.25. Half of that gap comes
  off its talent win%, .589 → .563, and
  a club that led the World Series board on run differential now sits second.
  The bracket keeps liking Milwaukee: its game 1 and 2 arms are −1.10 and
  −0.70 runs per nine, the best pair on the board. What it loses is the
  regular-season strength that carried it through three rounds.
- **The Dodgers (+7.6 pennant, +5.8 WS) are the mirror image.** Their nine
  project **5.19** runs a game against 4.78 scored, the best projected offence
  in baseball, in front of the best rotation (4.03). Both halves move the same
  way, .579 → .594, and Los Angeles takes over the top of the board. This is
  the disagreement the first section of this document diagnosed — we were 12
  points under Kalshi on the Dodgers on run differential alone — closing from
  our side.
- **Philadelphia (+7.2 division, +4.6 pennant) and Atlanta (−7.2, −3.3) are
  one race.** Philadelphia's staff projects 4.10 against 4.32 allowed (a 4.04
  rotation, a 4.19 pen) and its bats 4.76 against 4.50 scored; Atlanta's
  projects 4.38 against 4.02 allowed. Atlanta has prevented runs better all
  season than its staff's component rates say it should — defence, park and
  sequencing, everything FIP is blind to — and blending halfway to the
  bottom-up estimate costs it the NL East, which Philadelphia picks up almost
  exactly.
- **Chicago (−3.1 division) and Cleveland (+2.6) are the AL Central version**
  of the same thing, and the same one the first run flagged as our largest
  disagreement with FanGraphs: their roster estimates are a tenth of a run
  apart where their run differentials were further.
- **Seattle is the biggest riser still alive** (+0.040 talent win%, the second
  largest move on the board): 4.68 projected runs against 4.12 scored and 4.25
  allowed against 4.44. It buys 1.2 points of P(playoffs) at 66-74, which is
  what a strength change is worth to a club that needs the other results too.
- The two largest strength moves of all, **Oakland +0.046 and Cincinnati
  +0.029**, move no odds at all: they are 54-86 and 67-73. A strength model
  can only matter where the standings have not already decided.

The per-game terms are the smaller half and they are visible in one place: on
the 29 repriced games the chain moves P(home) by a mean **0.024** and up to
**0.053** against the same games priced by the old starter-only override, and
by a mean 0.034 (max 0.103) against the strength the sim draws the other 311
with. The expected-win shifts those imply are LAD +0.170, STL −0.170,
ATL −0.137, PHI +0.137, MIL +0.126, CHC −0.126 — a sixth of a win at most,
the same order as the starter term on its own.

### Sanity

- **The probabilities add up.** In all three runs P(playoffs) sums to exactly
  6.00 per league, P(division) to 1.00 in each of the six divisions,
  P(pennant) to 1.00 per league, and the two leagues' P(WS) to 1.00 (AL .401,
  NL .599 on the chain; .379/.621 old; .489/.511 on the coin flip).
- **The moves clear the simulation's own noise floor.** Re-running the served
  chain at three seeds at 20,000 sims moves the same numbers by at most 0.93
  points (246 vs 247: P(playoffs) 0.62 CLE, P(division) 0.71 TEX, P(pennant)
  0.87 MIL, P(WS) 0.93 MIL; the other two pairings are smaller). The four
  largest model moves — MIL −8.7, LAD +7.6, PHI +7.2, ATL −7.2 — are eight to
  ten times that.
- **No club moves more than its terms can pay for.** The largest strength
  change on the board is 0.046 of talent win% (Oakland), and over its 22
  remaining games it buys 0.44 of an expected win — the arithmetic ceiling on
  what a strength change can do in September. Every odds move above traces to
  a named run-environment gap of 0.2–0.4 runs a game; St. Louis, whose two
  halves agree to 0.005 of a run, moves 0.1 points of P(playoffs) and nothing
  else.

### What it still does not know

1. **One season, inside one standard error.** Every term above the starter is
   worth < 0.0006 Brier and none is more than one standard error from zero on
   756 games. The ordering holds on all 1,778 games of 2026 and all 2,105 of
   2025, which is the only reason to believe the sign. A second season of
   exchange history is what would size them.
2. **Two rotations in one job.** The bracket still picks its four arms by the
   BAS-42 rule — the club's most-used starters this season, ranked by
   regressed FIP — while station C's runs-allowed half uses the top five by
   starts in the last 30 days. What changed here is the strength those deltas
   bend, not who is in the rotation, and the first-pass limits listed in the
   previous section (openers counted as starts, trades splitting a pitcher,
   health and roster eligibility invisible) are all still there.
3. **The card term is nearly always asleep.** Lineups go up two to four hours
   before first pitch and the nightly job runs at 09:15 UTC, so in production
   `n_games_with_lineups` will read 0 and the served model is exactly the
   gated `pythag_C_sp_bpa_ip`. The 9 games in the table above had cards
   because this run was made while they were being played.
4. **The bottom-up half is park-, defence- and baserunning-neutral.** That is
   why it is blended rather than swapped in, and Atlanta is what it looks like
   when the top-down half is carrying real information the components cannot
   see. `--c-weight` is a knob on that trade, chosen walk-forward on 2025.
