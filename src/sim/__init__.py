"""Season simulator and playoff odds (roadmap Phase 2).

Pipeline: MLB Stats API standings + schedule → regressed Pythagenpat team
strength → log5 + home field per game → Monte Carlo over the remaining
schedule with MLB tiebreakers → postseason bracket → per-team odds.

Team strength is the swap point for the Bayesian layer: once Phase 1.5
produces roster-based projected runs scored/allowed, feed those to
`strength.from_run_environment` and everything downstream is unchanged.
"""
