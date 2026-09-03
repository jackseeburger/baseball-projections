# Replacing Marcel

Companion to [methods.md](methods.md). That doc says how to *pick* a tool for a
question already on the table. This one says *which questions to put on the
table next*, and in what order, given what the repo has actually measured.

The goal is not to supplement Marcel. It is to replace it — with a model that
works at the plate appearance rather than the season aggregate, conditions on
context Marcel cannot represent, and returns a posterior. Marcel stays
permanently as the baseline every replacement must beat, which is the job it is
actually good at.

It exists because "we should do more hierarchical Bayes and more ML" is not a
plan. Both sentences are true and neither tells you what to build on Monday.

For the finished picture — every layer marked model or arithmetic, and where
each one stands today — see [target-system.md](target-system.md). That doc
answers "what does this look like when it is done"; this one answers "what do
we build next, and in what order".

## A calibration, before anything else

The public evidence is worth knowing, because it sets expectations that are
otherwise easy to get wrong in both directions.

Hierarchical Bayes and ML are what serious systems use — that is not in
question. PyMC Labs has published a
[Bayesian Marcel](https://www.pymc-labs.com/blog-posts/bayesian-marcel) that
replaces hard-coded regression with a beta-binomial hierarchical model (pooling
by sample size instead of a fixed ballast) and Dirichlet-distributed season
weights estimated from data instead of Marcel's fixed 5/4/3. There is a
substantial academic literature doing the same for
[hitting](https://arxiv.org/pdf/0902.1360),
[fielding](https://arxiv.org/pdf/0802.4317) and
[team strength](https://arxiv.org/pdf/1712.05879).

**And on the tasks Marcel actually performs, the frontier is closer to it than
its reputation suggests.** FanGraphs' 2024 projection review put Marcel's RMSE
on batter plate appearances at **142.2, ahead of THE BAT X (156.2), ZiPS (155.0)
and Steamer (169.5)** —
[Marcel won that test outright](https://fantasy.fangraphs.com/2024-projection-review-batter-playing-time/).
Among the modern systems on adjusted RMSE the spread was 146.7 / 149.1 / 150.8,
a few points apart. The system that has separated itself — THE BAT X, most
accurate for four straight years — did it by
[layering Statcast on an already solid foundation](https://www.fantasypros.com/2024/01/most-accurate-fantasy-baseball-projections-2023/),
not by changing estimator family.

That playing-time result is worth pausing on, because it matches our own: the
largest gain in this repo also came from playing time, and it came from a
survival curve rather than a projection system. Marcel beating three
industrial systems on PA is a strong hint that the *question* was mis-posed for
all four of them.

Two conclusions follow, and they pull in opposite directions on purpose:

1. **Do not expect the replacement to win on aggregate MAE by much.** Nobody
   does. If we judge the Bayesian track by component MAE against Marcel, we
   will conclude it failed even if it is working exactly as the best systems in
   the world work.
2. **Which means MAE is the wrong scoreboard for this decision.** The reason to
   replace Marcel is the set of things it structurally cannot do at all —
   condition on context, carry uncertainty, model a player whose skill changed
   in June. Those are worth building for their own sake, and the aggregate MAE
   tie is the expected result, not the disappointing one.

## 0. What the evidence has already ruled out

Three results constrain this roadmap more than any preference does.

**The functional form is not the binding constraint for per-game win
probability.** A gradient-boosted model over 23,193 games, handed the chain's
own inputs, rediscovered the chain — importances in the chain's order, 0.88–0.94
correlated, a linear function of six chain terms explaining 81–90% of its
log-odds — and lost by .00073 with no |t| above 1.4. *More flexible models over
the same inputs are not the move.* A new model needs new information.

**The Bayesian arm is no longer losing on component accuracy.** Given the
current season and an opposing-pitcher term, the hierarchical model moves from
losing significantly (t 2.4–2.9) to inside noise (t 1.57 / 1.58 / 0.04). That
is one handicap removed, and it moved the result most of the way. There are
more handicaps: it has no Statcast input, no within-season skill drift, and no
context beyond the opposing pitcher.

A tie on component MAE is what the calibration above predicts, and it is not a
verdict on the method. **The gate decides what we serve, not what we
research.** A model that fails the gate is an iteration, not a dead end — the
mistake would be to read "did not clear the gate" as "this approach does not
work" and stop.

**The largest gains came from data and framing, not method.** Injury/option
return share: 6.4 PA per hitter at two months (t −5.4). Feeding Marcel the
current season: 3–6%. Tuning Marcel's constants: 1.1%. The biggest one was a
survival curve on a transaction feed nobody had read.

So the expansion is not "the same questions with fancier machinery." It is:
**where does a non-Marcel model buy something Marcel structurally cannot?**
There are exactly three such places.

## The bigger target: the problems that have no Marcel

The first draft of this doc framed everything as "replace Marcel," and that
turned out to be aiming at the smaller half of the opportunity.

Look at what the hierarchical Bayesian literature in baseball is actually
about: [pitch framing](https://arxiv.org/abs/1704.00823),
[fielding](https://arxiv.org/pdf/0802.4317),
[plate discipline](https://arxiv.org/pdf/2305.05752),
[hitting with shrinkage across time and players](https://projecteuclid.org/journals/bayesian-analysis/volume-4/issue-4/Hierarchical-Bayesian-modeling-of-hitting-performance-in-baseball/10.1214/09-BA424.pdf).
Three of those four are problems **Marcel does not attempt and could not
attempt.** There is no hand-computable baseline for a catcher's framing runs,
because the quantity only exists once you have adjusted for pitch location,
count, umpire, pitcher and batter simultaneously — which is precisely a
hierarchical model with partially pooled random effects.

That is the shape of the real opportunity. Front offices are not mainly running
hierarchical Bayes to beat Marcel at projecting next year's home run total.
They are running it on **valuation problems that have no simple estimator at
all**, where many entities have small and wildly unequal exposure and the
effect of interest is tangled with confounders that only a model can separate.

Our repo has nothing in this category. Everything built so far — every station —
competes with a baseline on a task the baseline already performs. That is good
discipline and it is why the gate rule works, but it has quietly restricted us
to the set of questions a simple method can already answer.

Candidates, roughly by value to what we actually do:

| Problem | Why it needs a hierarchical model | Feeds |
|---|---|---|
| **Catcher framing** | Catcher effect is confounded with pitcher, umpire and location; only joint partial pooling separates them | Run environment → station C/E |
| **Pitcher "stuff"** from pitch characteristics | Velocity/movement/release → run value is an unknown surface; pitcher true talent needs pooling on top of it | The starter term, our single largest per-game lever |
| **Fielding, per fielder and per position** | Many fielders, unequal chances, spatial structure | Run environment |
| **Plate discipline** | Swing decisions by location and count, pooled across hitters | Hitter true talent |

Note what the defence row does *not* say. We tested team defence and it failed
— but that was a top-down team-level blend that duplicated information the
existing run-environment blend already held. A per-fielder hierarchical spatial
model is a different construction against a different baseline, and the earlier
negative result does not settle it.

**These do not replace the staircase below; they run alongside it.** The
staircase is how the projection stack stops being Marcel. This section is how
the repo starts answering questions Marcel was never in the running for, and it
is where the published evidence for hierarchical Bayes is strongest.

## The replacement path, component by component

The three directions below (§1 posterior, §2 new information, §3 actuarial) are
the *ingredients*. This is the order they get assembled in, and the unit of
progress is one component at a time — not one grand model that either works or
does not.

`src/models/pa_k_rate.py` is the beachhead. It is already a real hierarchical
model — PA-level binomial cells, a random walk on the league trend, non-centered
partially pooled player ability, handedness, zero-sum park effects, a quadratic
age curve — and K-rate is the component with the most signal and the fastest
stabilization, so it is the fairest place for the approach to prove itself.

The staircase, each step gated against tuned Marcel on the same component:

1. **Context Marcel cannot represent.** Opposing pitcher (shipped), then park,
   platoon, and count state. Marcel is a season aggregate and structurally
   cannot condition on any of these; a PA-level model gets them nearly free.
   This is the clearest structural advantage and it should be exhausted first.
2. **Statcast as covariates**, then as an HSGP surface — the two-stage pattern of
   [methods.md §3](methods.md#3-the-two-are-not-a-wall). This is what separated
   THE BAT X from the field, so it is the step with the best outside evidence.
3. **Within-season skill drift.** A state-space/random-walk on player ability so
   a hitter who changed in June is not modelled as one flat season. Marcel
   cannot express this at all; neither can our current hierarchical model.
4. **Estimated pooling instead of a fixed ballast**, and season weights inferred
   rather than hard-coded 5/4/3 — the Bayesian Marcel construction. Expect this
   to be worth about a percent on its own, because the closed form in
   [methods.md §6](methods.md#6-techniques-worth-reusing-and-where-they-came-from)
   shows Marcel's ballast is already a close approximation. Do it for the
   principle and the posterior, not for the MAE.
5. **Roll out to the remaining components** once the pattern holds on K-rate,
   then retire Marcel from production and keep it as the baseline.

Steps 1–3 are where the accuracy is. Step 4 is where the posterior is. Neither
is optional, and doing 4 first — which is the tempting order, because it is the
most obviously "Bayesian" — would produce a tie on MAE, no new information, and
a false read that the approach does not work.

## 1. The posterior — things that need a distribution and are handed a number

This is the highest-value and cheapest direction, because the hierarchical model
*already produces* posteriors that nothing downstream consumes. We are paying
the cost of Bayesian machinery and throwing away the only thing it uniquely
gives us.

Three consumers, verified in the code:

| Consumer | What it gets now | Why that is wrong |
|---|---|---|
| `src/market/props.py` | Poisson on a **point estimate** of a rate | Its own docstring admits the Poisson understates the tail. Rate uncertainty is discarded on top of that, so prop prices are overconfident in a way no rate accuracy fixes. |
| `src/market/pnl.py` | `kelly_stake(p_win, ...)` — a **scalar** | Kelly under parameter uncertainty is provably not Kelly at the mean; the correct stake shades down. We are systematically overbetting by an amount nobody has measured. |
| `src/sim/season.py` | `simulate_remaining(state, strength: pd.Series, ...)` — **one number per team** | The Monte Carlo samples game outcomes but not parameter uncertainty. Every published playoff probability is conditional on the point estimate being exactly right. |

The third **has now been built and scored, and it failed.** It is left in the
table because the gap is real — the simulator still takes one number per team —
but the reasoning that put it first was wrong, and the correction matters more
than the original argument did.

The prediction was that understated parameter uncertainty explained station G's
shape: playoff probabilities that stop beating record extrapolation in the last
week of July, with almost all our advantage in resolution rather than
calibration. Drawing a fresh strength vector per simulated season leaves playoff
Brier at **+.00015 (t +0.11)**, makes projected-wins MAE **worse** by .0104
(t +4.18), and does not move the crossover at all — both arms turn positive in
the same 70–75% bucket.

**There was no over-confidence to fix.** Reliability was already .00055, the
best-fitting shrinkage is 0.968 on playoffs and **1.031 on pennants**, and
walk-forward it is above 1.0 in every season: the board wanted to be *sharper*,
not blurrier. August is short of **signal in the remaining schedule**, not of a
better-shaped distribution over the mean. See
[parameter-uncertainty.md](parameter-uncertainty.md).

So this ordering was wrong, and §5's first failure condition has fired. What
survives is the *principle* — a Bayesian change with a pre-registered question
attached is still the only kind worth making first — and the lesson that
"carry the uncertainty" is a claim to be tested per consumer, not a general
improvement. The remaining two consumers are untested and are now the ones that
matter.

The first two are where money is the exam. If a posterior changes P&L, that is
the strongest possible evidence for the Bayesian track — far stronger than a
component MAE tie.

**Sequence:** posterior into the simulator (testable against a known anomaly) →
posterior into Kelly sizing (measurable in the money exam) → posterior into prop
pricing (hardest, needs a count distribution, not just a rate interval).

## 2. New information — where ML has something to chew on

ML earns its keep where the input is not in a season-level box score. Ranked by
size of the untouched asset:

**Pitch characteristics → run value.** Velocity, movement, release point, and
the count state. Millions of rows, no entity that needs pooling at the pitch
level, a genuinely complicated surface nobody can write down. This is the
textbook case from methods.md §2 and it is completely untouched.

**Batted ball → expected outcome.** Exit velocity, launch angle, spray angle,
sprint speed → P(hit), P(XBH). In progress as the first bite at Statcast. The
right shape is the two-stage pattern of methods.md §3: the flexible model learns
the measurement, the hierarchical model pools it and returns the posterior.

**Catcher framing, umpire, and count-state effects.** Many groups, wildly
unequal exposure, a natural zero-sum constraint. This is hierarchical Bayes with
a `ZeroSumNormal`, not ML — listed here because it is new *information* even
though the method is old.

**The market as a feature, not only a benchmark.** We have Kalshi and Polymarket
candles archived. Every model so far predicts the game; none predicts *our
disagreement with the market*. Modelling the residual — where and when our edge
is real versus where we are simply wrong — is a different question with a
different target, and it is the question that actually pays. Note the trap:
this is trivially leaky and will look spectacular if built carelessly.

## 3. Neither — the category that has paid best

Do not let the interesting tools crowd this out. It has the best track record in
the repo.

**Pitcher injury and workload hazard.** The mirror of the hitter return-share
win. Starters have structure hitters lack: a rotation slot, a turn every fifth
day, an innings limit, an IL pattern that is both more common and more
predictive. Survival analysis on a transaction feed, again.

**Identities stay asserted.** Log5, pythagenpat, run expectancy. A flexible
model forced to rediscover log5 spends capacity on something we can simply state
— and the GBM result is the direct evidence, since that is precisely what it
spent its capacity doing.

## 4. The blocker that gates all of §1

**The Bayesian track cannot iterate without a working Modal refit path.** Every
hierarchical experiment above needs sampling that does not fit in a session
container, and the refit workflow has never run. Its prerequisites are mapped:
`train_pa_k_rate` needs a `cutoff_date` and a pitcher term, its model code is
inlined because Modal forbids cross-module imports, the 2024/2025 PA parquets
must be on the volume, and the timeout must clear 2h44m.

Until that path is proven end to end, "stand up more Bayesian models" is a
statement of intent, not a plan. **Unblocking the refit is worth more than any
single model on this page**, because it is the difference between one
hierarchical experiment a week and one an hour.

## 5. What would make this roadmap wrong

Stated in advance, so we notice:

- ~~If the posterior lands in the simulator and playoff-probability calibration
  does not improve, then parameter uncertainty was not the story behind the
  station G result, and §1's ordering is wrong.~~ **This fired (Sept 3).**
  Calibration did not improve (+.00015, t +0.11), projected wins got worse, and
  the crossover did not move. Parameter uncertainty is not what the station G
  shape is made of, and §1 has been rewritten to say so. The sharper finding:
  the board was never over-confident — best-fitting shrinkage is 0.968 on
  playoffs and 1.031 on pennants, above 1.0 walk-forward in every season, so it
  wanted sharpening. Recorded rather than quietly dropped, because a prediction
  written down in advance is only worth anything if it is scored when it comes
  due.
- If Statcast contact quality turns out to only restate the realized outcome
  rate with less noise, then the "new information" premise of §2 is weaker than
  claimed and the pitch-level work should be re-argued before it is built.
- If a posterior-aware Kelly does not change P&L measurably, then the *money*
  argument for the posterior is weaker than claimed and §1's ordering should be
  rethought. Note what this does not imply: the accuracy case for a PA-level
  contextual model in steps 1–3 above stands on its own and is unaffected.

None of these is a reason to stop. Each is a reason to reorder — and the whole
point of writing them down now is that a tie on MAE, which the calibration
section says to expect, is the single most likely thing to be misread as
failure.

The gate rule ([architecture.md §3](architecture.md#3-the-gate-rule)) decides
every one of these, and a negative result on any of them gets published exactly
like a positive one.
