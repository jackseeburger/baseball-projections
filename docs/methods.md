# Choosing a Method — What to Model How, and Why

Companion to [architecture.md](architecture.md). The architecture doc says *what*
each station produces and the bar it must clear. This doc says *how to pick the
tool* — hierarchical Bayes, machine learning, or neither — and it exists because
the question kept getting answered implicitly, one ticket at a time, by whoever
happened to be writing the code.

It is a default, not a law. The [gate rule](architecture.md#3-the-gate-rule)
outranks it: a method that beats its baseline out of sample ships regardless of
which box it came from, and a method that doesn't, doesn't.

For *where to point these methods next* — which questions to open, in what
order, and what would prove the ordering wrong — see
[modelling-roadmap.md](modelling-roadmap.md). This doc chooses a tool for a
question already on the table; that one chooses the questions.

## 0. The result that should frame the whole discussion

Sizes of real, gated gains measured in this repo, largest first:

| Change | What it bought |
|---|---|
| Projecting injured and optioned hitters at their expected return share | **6.4 PA per hitter** at a two-month horizon (t −5.4) — station B's first win on every metric |
| Feeding Marcel the current season instead of a preseason projection | **6–11%** of component MAE |
| Tuning Marcel's ballasts, recency and age curve on 2020–24 | **1.1%** of component MAE (t −3.7) |
| Each per-game term: starter, lineup, pen, run environment, start length | **.0001 – .0004** of Brier apiece |

The largest single improvement came from noticing that injured players come
back. It is not a model class at all — it is a survival curve fitted to a
transaction feed, and it beat every sophistication we had layered on top of the
same station. The second largest came from giving an existing model data it had
not been given.

**The lesson is not "don't build models."** It is that *what the model is fed*
and *what question it is asked* dominate *which family it belongs to*, by an
order of magnitude, at this stage of the project. Method selection matters. It
matters less than the two things above it, and anyone reaching for a more
powerful tool should first check whether the current one has been given
everything it could use.

## 1. Hierarchical Bayes

**Use it when the unit of analysis is an entity with a small and unequal
sample, or when something downstream needs a posterior rather than a number.**

Both halves of that sentence carry weight, and the second is the one we keep
forgetting.

*Small and unequal samples.* Player rates span roughly 20 to 700 plate
appearances in a season. Partial pooling is the correct answer to "what should I
believe about a hitter with 40 PA", and it is correct for a principled reason
rather than a tuned one: the population distribution is estimated from the data,
so the amount of shrinkage is inferred instead of chosen. Marcel's fixed ballast
is a hand-set approximation of the same idea, which is why tuning the ballast
bought about 1% — the approximation was already close.

*Group effects with unequal exposure.* Park, catcher framing, team defence: many
groups, wildly different exposure, and a natural zero-sum or sum-to-one
constraint. `pm.ZeroSumNormal` encodes that directly. A fixed-effect estimate
per park from a partial season is mostly noise; a partially pooled one is not.

*Downstream consumers that need a distribution.* This is the argument that
should decide it for us, and it is currently unserved:

- `src/market/props.py` prices counting outcomes with a Poisson on a **point
  estimate** of a rate. The uncertainty in that rate is thrown away, so every
  prop price is overconfident in a way that no amount of rate accuracy fixes.
- Quarter-Kelly sizing in `src/market/pnl.py` needs a distribution over the edge.
  It is currently handed a scalar.
- Station B wants P(active), not a binary.

A boosted model cannot supply any of that. That is the real reason to invest in
the Bayesian track — not the component MAE, where the honest headroom over a
well-tuned Marcel is a few percent.

**Where it stands here.** `src/models/pa_k_rate.py` is a genuine hierarchical
model: PA-level binomial cells, a random walk on the league trend, non-centered
partially pooled player ability, handedness, zero-sum park effects, a quadratic
age curve. It has never been given in-season data and has no opposing-pitcher
term, so its published loss to Marcel is not yet evidence about the method.

## 2. Machine learning

**Use it when the functional form is unknown, the features are many, the rows
are plentiful, and no entity needs pooling.**

*Batted ball to expected outcome.* Exit velocity, launch angle, spray angle,
sprint speed → P(hit), P(extra-base hit). Millions of rows, no entity to pool,
purely predictive. The relationship is a genuinely complicated surface that
nobody can write down. This is the textbook case, and it is our largest
untouched asset: Statcast 2015–2026 sits in R2 and has only ever been used for
descriptive aggregates.

*Pitch characteristics to run value.* Velocity, movement, release point → run
value. Same shape, same argument.

*Per-game win probability.* How starter quality, park, rest, bullpen state and
lineup interact is not something we know. Every term in the hand-built chain
asserts a form — log5, a fixed innings split, a linear delta — and each one has
bought a ten-thousandth or two. A model given the same inputs and allowed to
find its own form is a fair test of whether the form was the binding constraint.

**What it cannot do.** Give a calibrated posterior per entity. Extrapolate
outside its training distribution. Tell you *why*. For anything feeding a
betting decision or a public projection, those matter.

## 3. The two are not a wall

The most useful pattern in the literature is neither pure: **learn the hard
nonlinear measurement with a flexible model, then feed it as a covariate into a
hierarchical model that does the pooling and returns the posterior.**

The reference tutorial ([Developing Hierarchical Models for Sports Analytics](https://github.com/fonnesbeck/hierarchical_models_sports_analytics), [recording](https://www.youtube.com/watch?v=Fa64ApS0qig)) does exactly this — a Hilbert-space Gaussian process over
(exit velocity, launch angle) produces a nonparametric "contact quality"
surface, which then enters the linear predictor of an otherwise standard
hierarchical model. The GP *is* machine learning. It sits inside the Bayesian
model, and the combination gives both the flexible functional form and the
posterior.

This is roughly what serious baseball organizations do: expected-outcome models
derived from tracking data feed shrinkage-based projection systems. Two stages,
each doing what it is good at.

Prefer this over either extreme when the data supports it. See BAS-60.

## 4. Neither — and this is the category people forget

**Simulation is not a model.** `src/sim/season.py` and `src/sim/bracket.py` turn
per-game probabilities into season outcomes. They are a deterministic
aggregator. Keep them simple, transparent and fast; every ounce of modelling
belongs upstream in the per-game probability.

**Identities are structure we already know.** Log5, pythagenpat, run expectancy,
the arithmetic of a lineup turning over. These are not hypotheses to be learned.
A flexible model forced to rediscover log5 from data is spending capacity on
something we can simply assert, and it will do so imperfectly. **Encode what you
know; learn what you don't.** This is the single most common way to waste a
powerful method.

**Some questions are actuarial.** Injury return curves and option/recall hazards
are survival analysis on a transaction feed — Kaplan-Meier, censoring at season
end, a conditional read given time already elapsed. Neither Bayesian machinery
nor gradient boosting; just the right classical tool applied to a feed nobody
had read. It produced the largest gain in the table at the top of this document.

**The market is not a model.** Station M is a benchmark and a data source. The
exchange close is currently the most accurate per-game predictor in the system,
by .0023 of Brier over our best. Treat it as ground truth to be explained, not
a competitor to be ignored.

## 5. Workflow, by family

The method determines the discipline. Neither discipline is optional, and the
gate rule sits above both.

**Bayesian** — adopted from the tutorial's stated loop:

1. Visualize the data.
2. Build a provisional model.
3. **Prior predictive check.** If the prior puts mass on impossible rates, find
   out before spending compute.
4. Fit.
5. **Assess convergence** — r-hat and ESS, but also energy/BFMI, divergence
   counts, and a funnel pair plot of any group mean against its scale. A new
   random effect is the most likely thing to reintroduce a funnel; reparameterize
   non-centered when it does.
6. **Posterior predictive check.**
7. Improve, and go to 4.

Model *structures* are compared with LOO (`pm.compute_log_likelihood` then
`az.compare`), which estimates out-of-sample fit from a single fit via
Pareto-smoothed importance sampling. This is what makes iteration affordable
without a walk-forward refit per variant. **Say which claims rest on LOO and
which on walk-forward scoring — they are not the same evidence, and only the
second one clears a gate.**

Set hyperpriors with `pm.find_constrained_prior` (declare where the mass goes)
rather than hand-picking parameters, so the choice is reviewable.

**Machine learning:**

1. Split walk-forward by season. Never randomly.
2. Choose hyperparameters on an inner split of the *training* seasons only.
3. Calibrate on held-out training data; report reliability before and after.
4. Run a **permuted-label control** trained identically. It must land at chance.
   If it doesn't, there is leakage.
5. Report feature importances and read them: which hand-built terms did the
   model reproduce, and which did it ignore?
6. Score against the hand-built baseline *and* the market, paired.

**Both:** every constant is chosen on data the scored set never saw, every
leakage guard is unit-tested with a synthetic case where post-cutoff rows are
extreme, and a negative result is published exactly like a positive one.

## 6. Techniques worth reusing, and where they came from

Taken from the reference tutorial linked in §3. It is not in this repository —
clone it fresh from
`https://github.com/fonnesbeck/hierarchical_models_sports_analytics` when you
need the working code. Recorded here because a technique that lives only in a
conversation is a technique we will rediscover the hard way.

**Predicting on players the model has never seen.** The hardest bookkeeping
problem in an entity-indexed hierarchical model, and the one our in-season work
runs into constantly: a September call-up has no prior season, so there is no
fitted random effect to look up. The tutorial's answer is to draw fresh effects
for them from the *fitted population distribution* —

```python
epsilon_new = pm.Normal('epsilon_new', mu=0, sigma=sigma, shape=len(new_players))
```

— reusing the estimated `sigma`, then feeding those through the same linear
predictor with new index arrays. The unseen player gets the population's spread,
not a point estimate and not an exclusion. Wrap model inputs in `pm.Data` and
swap them with `pm.set_data` so the same fitted model can score any holdout
without refitting.

The failure here is silent: index arrays built for the fit no longer line up
with the rows being predicted, and the model happily reads the wrong player's
ability. Test the mapping, not just the output.

**Deriving priors from population data.** Fitting a distribution to the observed
population and using it as the prior (empirical Bayes) is not purely Bayesian —
the data inform their own prior — but it is a defensible way to get an
informative prior where a flat one would be silly. Say when you have done it.

**Partial pooling has a closed form.** For a rate,

```
p̂ ≈ [ (n/σ²_p)·p̄_player + (1/σ²)·p̄_population ] / [ (n/σ²_p) + (1/σ²) ]
```

which is a precision-weighted average of the player and the population. You do
not have to be Bayesian to do this — and this is exactly what Marcel's fixed
ballast approximates, which is the quantitative reason tuning that ballast bought
around 1% and not 10%. Reach for the full machinery when you want the pooling
*strength* estimated rather than assumed, or when you need the posterior.

**Model comparison without refitting.** `pm.compute_log_likelihood` then
`az.loo` / `az.compare` estimates out-of-sample fit from a single fit via
Pareto-smoothed importance sampling. This is what makes structural iteration
affordable when a full walk-forward refit is expensive or needs Modal. Its
limits are real: it approximates leave-*one*-out on the rows as fitted, which is
not the same as leaving out a future season, and it degrades when the Pareto
shape parameter runs high. Use it to choose a specification, never to clear a
gate.

## 7. The failure mode to watch

Sophistication that does not pay. It is the easiest mistake to make from here,
because the tools are more interesting than the work.

Three questions before reaching for a more powerful method:

1. **Has the current model been given everything it could use?** The largest
   two gains in this repo came from answering no.
2. **Is the functional form actually the binding constraint,** or is it the
   data? A boosted model over the same inputs answers this cheaply.
3. **Does anything downstream need what the new method gives?** A posterior is
   worth building for if props and Kelly sizing consume it, and not otherwise.

The gate rule is the backstop. It does not care which family a model came from.
