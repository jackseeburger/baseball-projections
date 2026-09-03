# Where to Expand the Non-Marcel Modelling

Companion to [methods.md](methods.md). That doc says how to *pick* a tool for a
question already on the table. This one says *which questions to put on the
table next*, and in what order, given what the repo has actually measured.

It exists because "we should do more hierarchical Bayes and more ML" is not a
plan. Both sentences are true and neither tells you what to build on Monday.

## 0. What the evidence has already ruled out

Three results constrain this roadmap more than any preference does.

**The functional form is not the binding constraint for per-game win
probability.** A gradient-boosted model over 23,193 games, handed the chain's
own inputs, rediscovered the chain — importances in the chain's order, 0.88–0.94
correlated, a linear function of six chain terms explaining 81–90% of its
log-odds — and lost by .00073 with no |t| above 1.4. *More flexible models over
the same inputs are not the move.* A new model needs new information.

**The Bayesian arm is no longer losing on component accuracy, and that is not
the reason to build it.** Given the current season and an opposing-pitcher term,
the hierarchical model moves from losing significantly (t 2.4–2.9) to inside
noise (t 1.57 / 1.58 / 0.04). It still does not clear the gate. Honest headroom
over a well-tuned Marcel on component MAE is a few percent, because Marcel's
fixed ballast is a hand-set approximation of partial pooling and the
approximation is close. **Chasing MAE with Bayes is chasing the few percent.**

**The largest gains came from data and framing, not method.** Injury/option
return share: 6.4 PA per hitter at two months (t −5.4). Feeding Marcel the
current season: 3–6%. Tuning Marcel's constants: 1.1%. The biggest one was a
survival curve on a transaction feed nobody had read.

So the expansion is not "the same questions with fancier machinery." It is:
**where does a non-Marcel model buy something Marcel structurally cannot?**
There are exactly three such places.

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

The third is the most interesting and the cheapest. We already run a Monte Carlo
over 20,000 seasons; feeding it posterior draws of team strength instead of a
fixed `Series` is close to free. And there is a specific unexplained result
waiting for it: station G's backtest found our playoff probabilities stop
beating record extrapolation in the last week of July, and that almost all our
advantage is resolution rather than calibration. Understated parameter
uncertainty is a live candidate explanation for exactly that shape. **This is a
Bayesian change with a pre-registered question attached, which is the only kind
worth making first.**

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

- If the posterior lands in the simulator and playoff-probability calibration
  does not improve, then parameter uncertainty was not the story behind the
  station G result, and §1's ordering is wrong.
- If Statcast contact quality turns out to only restate the realized outcome
  rate with less noise, then the "new information" premise of §2 is weaker than
  claimed and the pitch-level work should be re-argued before it is built.
- If a posterior-aware Kelly does not change P&L measurably, the money argument
  for Bayes collapses and the honest conclusion is that this project should
  stay largely non-Bayesian.

The gate rule ([architecture.md §3](architecture.md#3-the-gate-rule)) decides
every one of these, and a negative result on any of them gets published exactly
like a positive one.
