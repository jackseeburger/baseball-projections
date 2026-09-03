# Parameter Uncertainty in Team Strength (stations D, F, G)

**The hole this fills.** `src/sim/season.simulate_remaining` took
`strength: pd.Series` — *one number per club*. The season Monte Carlo then
drew 20,000 seasons of game outcomes around that one number, so every playoff,
pennant and World Series probability the site has published is **conditional on
our point estimate of team strength being exactly right**. The Monte Carlo
machinery to do better already existed; we simply never fed it a distribution.

This document is the first version of the missing half, and the score it got.

---

## 1. Pre-registration

Written and committed **before** the backtest was run
(`git log` — the doc's §1-§3 land in their own commit ahead of any result).

### 1.1 The question

The station G backtest ([team-projection-backtest.md](team-projection-backtest.md))
found two things over 249 weekly as-of dates and 44,820 scored
club-projections:

* **(a)** our playoff probabilities stop beating a .500 extrapolation of the
  standings at about 60% of the season — the last week of July — and are
  nominally *behind* it after 75%; pennant and World Series odds never
  separate from it at all;
* **(b)** almost the whole Brier advantage over the controls is
  **resolution** (.1257 against the control's .1188), not calibration
  (reliability .00055 against .00149).

Understated parameter uncertainty is a live candidate explanation for exactly
that shape. A model too confident in its point estimate produces probabilities
that are too extreme; being too extreme hurts most in the tails and hurts most
when there is least schedule left to average the error away.

### 1.2 What the hypothesis predicts

If understated parameter uncertainty is what finding (a) is made of, then
adding a distribution over team strength should:

* **H1.** Improve playoff Brier, and improve it *more* late in the season than
  early — the late-July crossover against `record_500` should move later, or
  disappear.
* **H2.** Improve the **tails** by a larger *relative* margin than the playoff
  column: pennant and World Series Brier should gain more than playoff Brier
  does, because that is where over-confidence is most expensive.
* **H3.** Show up as **reliability**, not resolution: reliability should fall
  and resolution should fall less, so the net is positive. Specifically decile
  8 of the calibration table — clubs given 60–80%, who make it 64% of the time
  — should close.
* **H4.** Survive the tuned-shrinkage control (§1.4), because the width is
  *heterogeneous*: it is larger for clubs with fewer games played and larger
  when more schedule remains, which a single global shrinkage constant cannot
  reproduce.
* **H5.** Leave projected final wins essentially unchanged. Expected wins are
  close to linear in strength, so a symmetric widening should move the mean by
  almost nothing — this is the null-effect prediction that says the change is
  doing what it claims and not something else.

### 1.3 What the mechanism predicts, which is not the same thing

Stated up front because it is the honest reading of the arithmetic and it
points the other way from H1:

> The **influence** of team-strength uncertainty on a playoff probability is
> largest in **April** and smallest in **September**, because the uncertainty
> acts only through the games still to be played. In April 140 games are left
> and a .03 error in talent win% is worth four wins; in mid-September 12 are
> left and it is worth a third of a win. By late September a club's playoff
> probability is mostly its banked record, which carries no parameter
> uncertainty at all.

So the mechanism says the *change* to the board is front-loaded, while the
hypothesis needs the *gain* to be back-loaded. Those are compatible only if
the April board is currently over-confident and the September board is not.
The April board is where the model's advantage is largest (−.034 of Brier
against `record_500`), which is not what an over-confident forecast usually
looks like.

**Therefore the pre-registered expectation of this session is that H1 fails**:
the width will move April a lot and September almost not at all, and the
late-July crossover will not move. If that is what happens, the roadmap's §1
ordering — parameter uncertainty first, because it explains the station G
shape — is wrong, and the doc will say so.

### 1.4 The controls a positive result has to beat

Adding parameter uncertainty moves probabilities toward the base rate almost
mechanically, and a Brier gain from shrinking toward the base rate can be had
with no new information at all. Three controls, all reported:

* **`shrink_half`** — the *served* chain's probabilities, shrunk linearly
  toward 0.5 by the amount that best fits: `p' = λp + (1−λ)·0.5`, with λ
  chosen to minimise Brier **on the same rows being scored**. This is an
  oracle: it is allowed to see the answer, which no real model is. If the
  uncertainty arm does not beat it, the uncertainty arm is adding calibration,
  not information.
* **`shrink_base`** — the same thing toward the outcome's own base rate rather
  than toward 0.5, also fitted in-sample. Stronger than `shrink_half` for a
  base rate of .36, and the more honest version of the same objection.
* **`shrink_wf`** — the same shrinkage with λ fitted **walk-forward**, on the
  seasons strictly before the one being scored. This is the version a real
  model would be allowed to use, and it is the one the gate rule cares about.

If the uncertainty arm beats `chain` but not `shrink_half` / `shrink_base`,
the finding is "our probabilities were over-confident and any shrinkage fixes
it", which is a smaller and different claim, and the doc will make it in those
words.

### 1.5 The gate

Station G's baseline is `record_500` (the .500 extrapolation) for the
probabilities and for projected wins. A change to what we *serve* must beat
the current served version — `chain` — out of sample on the common set, paired
per club per date, with standard errors clustered by season. **A negative
result is a complete outcome** and is published with the same rigour.

---

## 2. What was built

### 2.1 The simulator takes a distribution

`simulate_remaining(state, strength, hfa, n_sims, rng, ...)` now accepts either

* a `pd.Series` — one number per club, the original path, unchanged; or
* a `strength.StrengthDistribution` — in which case **each simulated season
  draws its own strength vector** and parameter uncertainty composes with
  game-outcome noise instead of the second being counted alone.

Four properties hold by construction and are unit-tested
(`tests/test_sim/test_strength_uncertainty.py`, 25 tests;
`tests/test_eval/test_team_season.py::TestUncertaintyArmLeakage`, 5 more):

1. **Zero width is the old model, bit for bit.** `scale=0` returns the point
   estimate broadcast and takes *nothing* from the generator, so
   `run_playoff_odds` produces a frame that compares equal column for column
   to the point-estimate run. That is what makes every difference in the score
   attributable to the width and to nothing else.
2. **The two arms are a common-random-numbers comparison.** The strength draw
   comes from a **spawned** stream (`Generator.spawn`), which derives an
   independent sequence without consuming the parent's bits, and the game draw
   consumes exactly `n_sims × n_remaining` uniforms either way. So the point
   arm and the uncertainty arm see the same uniforms on the same games, the
   same tiebreak draws and the same bracket draws; the paired difference in
   the backtest measures the width and not the seed.
3. **One draw prices the season and its October.** `run_playoff_odds` hands
   `play_postseason` the *same* per-sim strength vector the regular season was
   drawn with, so a club that is good in a simulated season is good in that
   season's playoffs. Drawing the two independently would wash out exactly the
   correlation that makes tail probabilities move.
4. **An override keeps its starter information and moves with the draw.** For
   a game whose probables are posted, the chain's per-game probability is
   shifted by the same logit-space deviation the draw put on the matchup,
   rather than being pinned at a point value — otherwise the model would
   re-assert certainty on precisely the games it knows most about.

### 2.2 Where the width comes from — and why it needs no new constant

Not a Bayesian refit: the Modal refit path has never run end to end and
waiting for it would have been the whole session. What is used instead is the
uncertainty **already implicit in the estimate**, read off the shrinkage the
model already applies.

`regressed_run_rates` shrinks a club's runs per game toward the league with
`regress_games = 60` of ballast. That shrinkage *is* a normal prior: adding
`k` pseudo-games at the league rate is the posterior **mean** of a
normal-normal model whose prior variance is the game-level run variance over
`k`. The posterior **standard deviation** of the same model is then

```
sd(regressed rate) = s / sqrt(g + k)
```

for a club with `g` games played and a game-level run standard deviation `s`.
The ballast the model already ships therefore names its own uncertainty; no
constant is introduced that was not already in production, and nothing is
fitted on the evaluation set.

Two sanity checks the derivation has to pass, and does:

| games played | implied SD of talent win% |
|---:|---:|
| 0 | ≈ .057 |
| 30 | ≈ .046 |
| 100 | ≈ .035 |
| 150 | ≈ .030 |

At `g = 0` the width is the full spread of MLB team talent (about .060 of
win%, ~10 wins over 162) — which is exactly what "we know nothing about this
club yet" should mean. By 100 games it is .035, where a normal-normal
posterior on a hundred games belongs.

**The naive bootstrap is a different — and wrong — number.** Resampling the
club's games and re-shrinking gives the sampling standard error of the
*shrunk estimator*, `s·sqrt(g)/(g + k)`. That is the variability of a
statistic, not the posterior spread of the talent it estimates; it is smaller
by `sqrt(g/(g+k))` at every `g`, and, worse, it goes to **zero** as `g` goes
to zero and peaks at `g = k`. It claims we are most certain about a club in
the first week of April. It was implemented anyway and is scored below as
`chain_pu_boot`, because it is the obvious first thing to reach for and the
comparison is the point.

The deviation is applied in **logit space as a shift** on whatever strength
vector the arm serves, so it composes with station C's blend, with a flat .500
board, or with a preseason vector, rather than replacing any of them.

### 2.3 What this is not

* **Not a posterior.** It carries no uncertainty from the *bottom-up* half of
  station C's blend — the player rate models, playing time, the rotation — and
  the blend is half of the served strength. A real posterior would be wider.
* **No in-season drift.** A club's talent is held fixed for the rest of the
  season: no trades, no injuries, no callups. That too is a source of variance
  this omits.
* Both omissions push the same way, so **the width used here is a lower
  bound**, and the `chain_pu_double` arm exists to say what twice it does.

### 2.4 Leakage

The width is built from `split.played` — the games strictly before the cutoff,
the same frame `split.standings` is summed from — so it reads exactly what the
point estimate reads. `assert_team_split_clean` already guards that frame.
Beyond that, `TestUncertaintyArmLeakage` rebuilds the same synthetic season
twice, identical up to the cutoff and 20-0 blowouts for the other side after
it, and asserts the *width* is identical array for array as well as the
projection being identical column for column — a leak into a width would look
like a well-tuned prior rather than a wrong mean, which is the quieter
failure.

---

## 3. What was run

Ten seasons — 2015–2019, 2021–2025 — at weekly as-of dates, 30 clubs each,
2,000 simulations per arm per date, the season's own postseason format, the
whole chain rebuilt at each date from games strictly before it. 2020 is
excluded by name. The as-of dates, the outcomes and the scoring are the
station G harness unchanged; the only new thing is the arm.

Reproduce:

```
python scripts/run_team_backtest.py --stage fetch   --seasons 2015-2025 --workers 12
python scripts/run_team_backtest.py --stage project --seasons 2015-2025 --sims 2000
python scripts/run_team_backtest.py --stage score   --seasons 2015-2025 --markdown
python scripts/analyse_parameter_uncertainty.py
```

<!-- RESULTS BELOW THIS LINE ARE FILLED IN AFTER THE RUN -->
