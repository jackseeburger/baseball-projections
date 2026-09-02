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
that already agree. The swap point is `src/sim/strength.from_run_environment`.

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
