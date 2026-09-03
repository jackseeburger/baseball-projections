# The Target System

What this is being built toward, in one picture, with every layer marked
**model** or **arithmetic** and every box marked with where it stands today.

Written because "we should do more Bayes and ML" kept failing to answer the
actual question — *what does the finished thing look like?*

## 1. How everyone else assembles it

Worth establishing first, because it removes a false worry.

FanGraphs' public pipeline: Depth Charts are a
[50/50 blend of Steamer and ZiPS scaled to playing-time estimates](https://library.fangraphs.com/features/playoff-odds/);
players are aggregated by team; **BaseRuns** converts batters and pitchers into
projected runs scored and allowed; **Pythagorean** win expectancy turns those
into a projected win percentage; the remaining season is simulated
[**20,000 times**](https://www.fangraphs.com/standings/playoff-odds/about),
adjusted for the identity of announced starters; and the division, pennant and
World Series probabilities are the *frequencies across those simulations*.

That is our architecture, almost box for box. We use Pythagenpat rather than
Pythagorean and a bottom-up run environment rather than BaseRuns, and our
playing time comes from survival curves rather than hand curation. But the
spine is the same, and 20,000 simulations is the same number by coincidence of
both parties picking a round one.

**The spine is identical because it is accounting, not modelling.** Nobody
replaces rates → playing time → runs → wins → simulate with a neural network,
because there is nothing there to learn: it is the definition of how baseball
scores work. Front offices differ from FanGraphs, and we differ from both, in
exactly two places — **how good the leaf estimates are**, and **how much
uncertainty survives the trip through the spine.**

So the answer to "are we doing this the way real systems do it" is yes,
structurally, already. The work is not rearchitecting. It is replacing the
leaves and carrying the variance.

## 2. The target, layer by layer

```
┌─ LAYER 1 · MEASUREMENT ────────────────────────────── MODEL (ML) ─┐
│  Statcast pitch and batted-ball data                              │
│    (exit velo, launch angle) ──► contact quality surface          │
│    (velo, movement, release) ──► pitch run value / "stuff"        │
│  Millions of rows, unknown surface, no entity to pool.            │
│  STATUS: contact quality in progress. Stuff not started.          │
└───────────────────────────────────────────────────────────────────┘
                              │ enters as a covariate
┌─ LAYER 2 · TRUE TALENT ───────────────── MODEL (hierarchical Bayes) ─┐
│  One PA-level model per component: K% BB% HR/PA BABIP ISO            │
│    · partially pooled player ability — shrinkage ESTIMATED           │
│    · Layer 1 output as a covariate                                   │
│    · context: park, platoon, opposing pitcher, count                 │
│    · within-season random walk on skill                              │
│    · hierarchical aging curve                                        │
│  OUTPUT: a posterior per player per rate — not a number.             │
│  STATUS: K% exists and loses to Marcel. Four components not started. │
└──────────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 3 · PLAYING TIME ──────────── MODEL (actuarial, + Bayes later) ─┐
│  IL and option hazards, return curves ──► posterior over PA / BF       │
│  STATUS: LIVE and gated for hitters — the biggest win in the repo.     │
│          Pitcher side ungated (in progress). Point estimate, not yet   │
│          a posterior.                                                  │
└────────────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 4 · ASSEMBLY ─────────────────────────────── ARITHMETIC ─┐
│  BB = bb_rate × PA   AB = PA − BB − HBP − SF   HR = hr_rate × PA │
│  ──► slash line ──► wOBA ──► wRC+ ──► runs ──► WAR               │
│  Identities. Never a model. Unchanged in the target.             │
│  What CHANGES: it runs on posterior DRAWS instead of point        │
│  estimates, so uncertainty survives to everything below.          │
│  STATUS: live on point estimates. Career WAR already does the     │
│          draw version — see issue #75.                            │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 5 · TEAM AND GAME ─────────────────────────── ARITHMETIC ─┐
│  team runs ──► Pythagenpat ──► team strength                      │
│  per game: log5 + HFA + starter FIP over his expected innings     │
│            + availability-weighted pen + posted lineup            │
│  STATUS: live and gated. Best .24388 vs market .24156.            │
└───────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 6 · SIMULATION ──────────────── NOT A MODEL. COUNTING. ─┐
│  20,000 seasons, real tiebreakers, real bracket                 │
│  ──► standings, playoff, pennant, World Series                  │
│  What CHANGES: draw a PARAMETER SET per simulated season, not    │
│  just game outcomes, so the intervals are honest.                │
│  STATUS: live on a fixed strength vector. Under test now.        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 7 · DECISION ─────────────── WHAT THE POSTERIOR UNLOCKS ─┐
│  posterior rate ──► prop prices with real tails                  │
│  posterior edge ──► Kelly that shades for uncertainty            │
│  posterior WAR  ──► $/WAR ──► contract surplus value             │
│  STATUS: props and Kelly live on POINT ESTIMATES — both wrong in  │
│          a known direction. Contract valuation does not exist.    │
└──────────────────────────────────────────────────────────────────┘
```

## 2b. Three tracks, not one — hitting, pitching, defence

The diagram above reads as if layers 1–4 were a single pipeline. They are not.
They are **three parallel tracks that converge at layer 5**, and writing them as
one hid the fact that two of them are much emptier than the first.

| | **Hitting** | **Pitching** | **Defence / catching** |
|---|---|---|---|
| **L1 Measurement** | batted ball → contact quality — **gated**, 7 of 8 components | pitch characteristics → "stuff" / run value — **not started** | fielding location → out probability — **not started** |
| **L2 True talent** | K% BB% HR/PA BABIP ISO — tuned Marcel live; Bayesian arm inside noise | K% BB% HR/BF BABIP-against, WHIP rate — **gated**, all five clear | framing runs, fielder runs — **in progress** (framing); no Marcel equivalent exists |
| **L3 Playing time** | PA — **gated**, the biggest win in the repo | batters faced — **gated** (B-P) | innings by position — **does not exist** |
| **L4 Assembly** | rates × PA → wOBA → wRC+ → oWAR | rates × BF → FIP / RA9 → pWAR | runs saved → dWAR |
| ↓ | | | |
| **L5** | → team runs scored | → team runs allowed | → team runs allowed |

Three things fall out of laying it side by side.

**Defence is nearly an empty column.** Layer 3 does not exist for it at all — we
have no projection of who plays where, for how many innings. That is the same
kind of gap that projected batters faced was until it was scored, and it blocks
a defensive layer 4 entirely: runs saved per inning is useless without innings.

**Pitching has no layer 1.** Every pitcher rate we project comes from box-score
outcomes. The pitch-level data that would tell us *why* a pitcher gets those
outcomes — and would let us separate a real change in stuff from a hot month —
sits unused in R2. This is the largest single untouched asset in the system, and
it feeds the term that already buys us the most per game.

**The defence column is where the "no Marcel equivalent" argument bites
hardest.** There is no simple estimator for a catcher's framing runs or a
shortstop's range: those quantities only exist once a model has adjusted for
everything around them. That is why the framing work is the first entry in a
column that is otherwise blank, rather than a detour.

One correction that belongs here: we tested team defence in the run environment
and it failed. That does **not** settle this column. It was a top-down team-level
blend that duplicated information the run-environment blend already carried — a
different construction, against a different baseline, from a per-fielder
hierarchical spatial model. Do not cite it as evidence that defence is worthless
to model.

## 3. So: are all the rates moving to Bayes?

**Yes.** All five components, at layer 2, replacing Marcel as the engine.

With three conditions that are not negotiable:

1. **One component at a time, each gated.** K% first, because it has the most
   signal and stabilizes fastest, so it is the fairest test. A component swaps
   only when it beats tuned Marcel out of sample.
2. **Marcel never leaves.** It becomes the permanent baseline, not the engine.
   The day nothing is compared against it is the day we stop knowing whether
   the Bayesian stack is earning its complexity.
3. **In the order in [modelling-roadmap.md](modelling-roadmap.md)** — context
   and Statcast before estimated pooling. Doing the pooling first is the
   tempting move because it is the most obviously Bayesian, and it would buy
   about a percent, produce no new information, and read as a failure.

**Why it has not happened yet**, plainly: the Bayesian K% model currently loses
to Marcel because it has never been given the current season, Statcast, park,
platoon, or within-season drift. Every one of those is a rung on the staircase,
and none has been climbed. Its published loss is not evidence about Bayes; it
is evidence about a model running with most of its inputs missing.

**The blocker is compute, not conviction.** Every layer-2 experiment needs
sampling that will not fit in a session container, and the Modal refit path has
never run end to end. That is the single highest-leverage thing to unblock.

## 4. What stays arithmetic forever

Layers 4, 5 and 6. The identities, the Pythagenpat, the log5, the simulator.

This is not a compromise or a stopgap. A flexible model asked to rediscover
log5 from data spends its capacity relearning something we can simply assert,
and does it imperfectly — and we measured exactly that: a gradient-boosted model
over 23,193 games, given the chain's own inputs, reproduced the chain's
structure (0.88–0.94 correlated, importances in the chain's order) and lost by
.00073.

**Encode what you know; learn what you don't.** The models belong at the
leaves. The spine is accounting, and FanGraphs and every front office treat it
the same way.
