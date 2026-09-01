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
1. Starting-pitcher adjustment when probables are posted (roadmap 2.2 optional) — mostly a postseason-odds effect.
2. Tune `regress_games` (60) against historical final standings via the backtest harness.
3. Extend tiebreakers past the coin-flip floor (rarely binding).
