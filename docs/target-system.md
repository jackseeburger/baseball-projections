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
│  STATUS: contact quality GATED (7 of 8 components, t clustered by │
│  player) but NOT WIRED — the artifact is monthly and the board     │
│  projects on an arbitrary date, so it needs a partial-month        │
│  top-up. The HSGP surface LOST to six plain aggregates. Stuff not  │
│  started (BAS-67).                                                 │
└───────────────────────────────────────────────────────────────────┘
                              │ enters as a covariate
┌─ LAYER 2 · TRUE TALENT ───────────────── MODEL (hierarchical Bayes) ─┐
│  One PA-level model per component: K% BB% HR/PA BABIP ISO            │
│    · partially pooled player ability — shrinkage ESTIMATED           │
│    · Layer 1 output as a covariate                                   │
│    · context: park, platoon, opposing pitcher, count                 │
│    · random walk on skill — DONE between seasons (BAS-69), not yet   │
│      within one                                                      │
│    · hierarchical aging curve                                        │
│  OUTPUT: a posterior per player per rate — not a number.             │
│  STATUS: K% DRAWS with tuned Marcel; four components not started.    │
│  BAS-69 gave the model a season random walk on player ability —      │
│  recency it estimates rather than one we fix — and that closed 73%   │
│  of the deficit: +0.00033 MAE at t 1.40 over 48 cutoffs, from        │
│  +0.00121 at t 4.08. It beats its own flat self at 43 of 48 (t       │
│  -3.45), which is a real gate cleared; it does not beat Marcel, so   │
│  Marcel keeps serving. A draw with a posterior attached is worth     │
│  more than a draw without one — see docs/bayes-variants.md.          │
│  Statcast covariates clear the gate ON TOP of tuned Marcel, so the   │
│  layer-1 → layer-2 path is proven even though layer 2's own          │
│  Bayesian engine is not yet the one serving.                         │
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
│          draw version — see issue #75.                           │
└──────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 5 · TEAM AND GAME ─────────────────────────── ARITHMETIC ─┐
│  team runs ──► Pythagenpat ──► team strength                      │
│  per game: log5 + HFA + starter FIP over his expected innings     │
│            + availability-weighted pen + posted lineup           │
│  STATUS: live and gated. Best .24388 vs market .24156.            │
└───────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 6 · SIMULATION ──────────────── NOT A MODEL. COUNTING. ─┐
│  20,000 seasons, real tiebreakers, real bracket                 │
│  ──► standings, playoff, pennant, World Series                  │
│  Drawing a PARAMETER SET per simulated season was built, scored   │
│  and FAILED: playoff Brier +.00015 (t +0.11), projected wins      │
│  worse, crossover unmoved. The board was never over-confident —   │
│  best-fitting shrinkage 0.968 playoffs / 1.031 pennants, above    │
│  1.0 walk-forward every season, i.e. it wanted SHARPENING.        │
│  STATUS: live on a fixed strength vector, on purpose.             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─ LAYER 7 · DECISION ─────────────── WHAT THE POSTERIOR UNLOCKS ─┐
│  posterior rate ──► prop prices with real tails                  │
│  posterior edge ──► Kelly that shades for uncertainty            │
│  posterior WAR  ──► $/WAR ──► contract surplus value             │
│  STATUS: Beta width SPENT (BAS-70) and the single-component      │
│          Bayes width SPENT (BAS-92): both a relabel of the       │
│          mean edge; Kelly at the mean is right by proof. Width   │
│          with structure waits on BAS-84/85. No valuation.        │
└──────────────────────────────────────────────────────────────────┘
```

## 2b. Three tracks, not one — hitting, pitching, defence

The diagram above reads as if layers 1–4 were a single pipeline. They are not.
They are **three parallel tracks that converge at layer 5**, and writing them as
one hid the fact that two of them are much emptier than the first.

| | **Hitting** | **Pitching** | **Defence / catching** |
|---|---|---|---|
| **L1 Measurement** | batted ball → contact quality — **gated**, 7 of 8, not yet wired; HSGP version lost | pitch characteristics → "stuff" / run value — **not started** | fielding location → out probability — **not started** |
| **L2 True talent** | K% BB% HR/PA BABIP ISO — tuned Marcel live; Bayesian arm DRAWS with it on K%, BB% and HR/PA (BAS-69, BAS-73); structural track: covariates as a regressor in the likelihood MEASURED and worse (BAS-83, errors-in-variables on the thin current window; the block is the negative control now), joint multi-component MEASURED (BAS-84: correlations real in 3 of 4 seasons, a reliable third of a percent on K%/BB% over the single-component arm, nothing on HR/PA, loses to contact quality everywhere; not served), measurement model MEASURED and worse (BAS-85, `docs/bayes-measurement.md`: four seasons, every prediction fails, HR/PA +5% vs contact_additive and +3% vs bayes_walk, K% 0 of 48; the channels load with stable physical loadings, the joint fit wastes them; follow-up named, not opened: CSW channel, pinned sign, anchored scale, exposure cap); the prior's mean as a function of the prior-season profile is IN FLIGHT (BAS-94, `docs/bayes-prior-mean.md`); park factors BUILT (BAS-86, `docs/park-factors.md`: real and persistent; multiplied onto tuned Marcel they double-count and lose on every component; committed as the offset the Bayes arms always expected, to be scored inside them next) | K% BB% HR/BF BABIP-against, WHIP rate — **gated**, all five clear | framing runs, fielder runs — **in progress** (framing); no Marcel equivalent exists |
| **L2 → chain** | the per-game chain (stations C/E) still runs stock Marcel on both sides, and that is MEASURED as the right call for now (BAS-90, `docs/chain-engines.md`: tuned / +stuff / +contact rungs each price games worse against the close, +.00076 Brier on the market set, wrong sign on all three sets, vacuity passed; the engines widen the rate tables 18–37% and the chain's constants were tuned to the narrow ones, the age curve is 70% of the cost); switch built and off; the level- and spread-matched follow-up MEASURED as a real null (BAS-91, `docs/chain-engines-matched.md`: matching removes the cost, pooled +.00026 → −.00002, candidate arm −.00008 ± .00015 on 3,988 games; the age curve on the whole roster costs +.0005 on 2026 and nothing on 2025; the per-game exam cannot resolve a 1% player-grain gain) | same | — |
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

## 2c. The model inventory, component by component

The three-track table above says where each *layer* stands. This one says
where each *model* stands, so nothing falls between the layers. Every
unstarted cell has a Linear issue; the status here is updated when one moves.

**Hitting**

| Component | L2 hierarchical model | L1 measurement feeding it |
| --- | --- | --- |
| K% (K/PA) | LIVE research arm: PA-level, season random walk, draws with `marcel_tuned` (BAS-69); + contact covariates as a regressor inside the Bayes model: WORSE by 19% of MAE (BAS-83, `docs/bayes-covariates.md`) (`docs/bayes-covariates.md`) | contact quality — **SERVED** (`contact_additive`, BAS-72) |
| BB% (BB/PA) | MEASURED — Bayes walk loses to tuned Marcel by +0.00033, t 2.10 (BAS-73 grid, `docs/bayes-components.md`); a draw in substance, a loss by the pre-registered letter; not served | contact quality — **SERVED** (BAS-72); swing decisions — measured, not served: same measurement as contact quality on K%, variance reducer on BB% (BAS-75/81, `docs/swing-decisions.md`) |
| HR/PA | MEASURED — Bayes walk draws with tuned Marcel (+0.00002, t 0.30; BAS-73 grid); the predicted win did not appear; not served. + contact covariates as a regressor inside the Bayes model: WORSE by 19% of MAE (BAS-83, `docs/bayes-covariates.md`) | contact quality — **SERVED** (BAS-72), *information* |
| ISO (per AB) | recovered HSGP model, never scored in the harness (#86) — after BAS-73 | contact quality — **SERVED** (BAS-72), *information* |
| BABIP (per BIP) | recovered HSGP model, never scored (#86) — after BAS-73 | contact quality — **SERVED** (BAS-72), denoising; sprint speed — NOT STARTED |

**Pitching**

| Component | L2 hierarchical model | L1 measurement feeding it |
| --- | --- | --- |
| K/BF | NOT STARTED — BAS-74 | stuff — gated (−3.2%, t −3.4; BAS-71); additive arm **withheld** at t 2.50 vs the 2.5 bar (BAS-79) |
| BB/BF, (BB+HBP)/BF | NOT STARTED — BAS-74 | stuff — **SERVED** on BB/BF (`stuff_additive`, BAS-79), mostly a level correction; calibration measured and not shipped, vacuity failed (BAS-80, `docs/pitcher-marcel-calibration.md`); command / location — MEASURED, NOT SHIPPED: real at the pitch (CSW log-loss −.06 to −.08 over stuff) but the stuff-differenced pitcher aggregate fails its pre-registered persistence floor (r 0.40 vs 0.45), stage 2 not run (BAS-76, `docs/pitching-command.md`); command as a LEVEL with stuff as an explicit control — **GATED on BB/BF** (BAS-87, `docs/pitching-command-level.md`: −4.5% vs tuned Marcel, covariate-only share −1.5% at t −3.6 over the served stuff engine, nothing on K/BF or HR/BF); serving tried and WITHHELD (BAS-88: on the 2026 season alone −1.1% at t −1.1, and the vacuity clause fires on `cmd_csw`; engine built and dormant, `p_bb_rate` stays on stuff) |
| HR/BF | NOT STARTED — BAS-74 | stuff — **SERVED** (`stuff_additive`, −2.8%, all covariate; BAS-79); contact-quality-allowed — gated, *information* |
| BABIP against | NOT STARTED — BAS-74 | contact-quality-allowed — gated |

`marcel_tuned` + `contact_additive` serves the five hitter components;
`marcel_pitcher_tuned` serves K/BF and BABIP against, and
`marcel_pitcher_tuned` + `stuff_additive` serves BB/BF and HR/BF. Both
Marcels stay the baseline every model above must beat.

**Defence** — framing cancelled; public pitch data carries no fielder
tracking. This track stays thin until a public source exists.

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
