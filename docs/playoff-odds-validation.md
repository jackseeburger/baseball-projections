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
   announced starters and rotation quality is most of the edge. The bracket
   still runs on team strength today — nobody has announced a Game 1 starter
   in early September — and wiring starters into `bracket.play_series` is the
   natural follow-on once postseason rotations are set.
3. **Per-game odds and prices**, which is where the term already earns its
   keep and where it is scored (station E, not station G).

Being invisible here is the expected result, and it is why the gate for this
change was the per-game Brier score and not this table.
