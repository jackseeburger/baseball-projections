# Parameter Uncertainty in Team Strength (stations D, F, G)

**The hole this fills.** `src/sim/season.simulate_remaining` took
`strength: pd.Series` — *one number per club*. The season Monte Carlo then
drew 20,000 seasons of game outcomes around that one number, so every playoff,
pennant and World Series probability the site has published is **conditional on
our point estimate of team strength being exactly right**. The Monte Carlo
machinery to do better already existed; we simply never fed it a distribution.

This document is the first version of the missing half, and the score it got.

> **The verdict, up front.** The machinery is built, tested and reversible;
> the model it makes possible **fails the gate and is not served**. Over the
> same 10 seasons and 249 weekly as-of dates the station G backtest used,
> drawing a fresh team-strength vector per simulated season leaves playoff
> Brier at **.10354 against the served .10339 (Δ +.00015, se .00135, t +0.11,
> n = 7,470)** and makes projected wins **worse by .0104 of MAE (t +4.18)**.
> The **late-July crossover does not move** — both arms first fall behind the
> .500 extrapolation at 70–75% of the season.
>
> **The pre-registered hypothesis is refuted, and the reason is specific.**
> The board was never over-confident: the linear shrinkage of its playoff
> probabilities that would have minimised Brier is **0.968**, and of its
> pennant probabilities **1.031** — the tails want to be *sharper*. There was
> nothing for parameter uncertainty to fix, so the roadmap's ordering, which
> put this first because it would explain the station G shape, is wrong.

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
(`tests/test_sim/test_strength_uncertainty.py`, 27 tests;
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

# every table in sections 5 to 9, and the committed artefact
python scripts/analyse_parameter_uncertainty.py --markdown \
    --json-out public/data/parameter_uncertainty/2015-2025.json
# section 4
python scripts/report_strength_width.py \
    --json-out public/data/parameter_uncertainty/strength_width.json
```

**The rerun reproduces the published station G baseline exactly.** The `chain`
row below is 4.4971 wins MAE, .10339 playoff Brier, reliability .00055,
resolution .12569, and −1.307 wins (t −8.17) / −.0085 Brier (t −2.27) against
`record_500` — the same numbers, digit for digit, as
[team-projection-backtest.md](team-projection-backtest.md) §6 and §7. That is
the check that the new code path left the point-estimate arm untouched, and it
is why every difference below can be attributed to the width.

Artefacts: `public/data/parameter_uncertainty/2015-2025.json` (every table
here, generated) and `strength_width.json` (§4). The projections themselves
checkpoint to `data/parquet/team_backtest/projections_<season>.parquet`
(gitignored — 74,700 scored club-projections across ten arms).

---

## 4. The width, measured

Before any score: how wide the distribution actually is, at each of the 249
as-of dates, and how many wins that is worth over the schedule each club has
left. `python scripts/report_strength_width.py`.

| Season played | games played | games left | implied talent-win% SD | = wins of spread over the remaining schedule |
|---|---:|---:|---:|---:|
| 0–15% | 15.9 | 146.1 | **.0505** | 7.38 |
| 15–30% | 36.9 | 125.1 | .0464 | 5.82 |
| 30–45% | 60.0 | 102.0 | .0416 | 4.25 |
| 45–60% | 85.9 | 76.1 | .0377 | 2.87 |
| 60–75% | 110.2 | 51.7 | .0350 | 1.82 |
| 75–90% | 134.1 | 27.8 | .0328 | 0.92 |
| 90–100% | 153.0 | 9.0 | **.0313** | 0.28 |

Two things to read off it.

1. **The width is right-sized and monotone.** .0505 in April is close to the
   real spread of MLB team talent, which is what "two weeks in, we know almost
   nothing" should mean; .0313 in the last fortnight is a club known to about
   five wins over a full season. It falls every bucket, because knowledge only
   accumulates.
2. **Its influence is front-loaded, exactly as §1.3 said.** The same SD is
   worth **7.4 wins of spread in April and 0.28 in the last fortnight**, a
   factor of twenty-six, because it acts only through games not yet played.

The naive bootstrap, for contrast, runs **.0225 → .0293 → .0265**: narrowest
in April, peaking at `g = k = 60` games, exactly the wrong shape (§2.2).

---

## 5. The headline

Ten seasons, 249 as-of dates, 7,470 club-projections per arm.

| Arm | Wins MAE | Rest win% MAE | Brier playoffs | Log loss playoffs | Brier division | Brier pennant | Brier WS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **chain** (served) | **4.497** | **.06751** | .10339 | .31693 | .08138 | .05171 | .02952 |
| **chain_pu** (the proposal) | 4.507 | .06762 | .10354 | .31866 | **.08026** | .05162 | .02947 |
| chain_pu_half (½ width) | 4.500 | .06754 | .10300 | .31580 | .08058 | .05168 | .02946 |
| chain_pu_double (2× width) | 4.534 | .06790 | .10858 | .33680 | .08304 | .05194 | .02949 |
| chain_pu_boot (bootstrap width) | 4.502 | .06757 | **.10285** | **.31528** | .08026 | **.05161** | **.02943** |
| shrink_half (oracle λ, toward .5) | 4.497 | .06751 | .10327 | .32004 | .08119 | .05171 | .02951 |
| shrink_base (oracle λ, toward base) | 4.497 | .06751 | .10325 | .32008 | .08099 | .05170 | .02950 |
| shrink_wf (walk-forward λ) | 4.497 | .06751 | .10349 | .31817 | .08145 | .05176 | .02953 |
| record_500 (the baseline) | 5.804 | .08481 | .11191 | .34304 | .08831 | .05546 | .03012 |
| record_wpct | 6.119 | .08046 | .12357 | .48932 | .10489 | .05695 | .02933 |
| preseason | 8.472 | .22711 | .20323 | .59923 | .14018 | .06264 | .03203 |
| coin_flip | 10.405 | .27962 | .22943 | .65132 | .16000 | .06222 | .03222 |

Paired on the same club, the same date and the same season, standard errors
clustered by season (10 clusters), **common random numbers** — the two arms
draw the same uniforms on the same games (§2.1). Negative favours the
uncertainty arm.

| chain_pu minus | Metric | chain_pu | other | Δ | se | t | n |
|---|---|---:|---:|---:|---:|---:|---:|
| **chain** | wins MAE | 4.5075 | 4.4971 | **+.01042** | .00249 | **+4.18** | 7470 |
| **chain** | Brier playoffs | .10354 | .10339 | **+.00015** | .00135 | +0.11 | 7470 |
| **chain** | Brier division | .08026 | .08138 | −.00112 | .00162 | −0.69 | 7470 |
| **chain** | Brier pennant | .05162 | .05171 | −.00009 | .00017 | −0.54 | 7470 |
| **chain** | Brier WS | .02947 | .02952 | −.00005 | .00008 | −0.55 | 7470 |
| **chain** | Log loss playoffs | .31866 | .31693 | +.00173 | .00486 | +0.36 | 7470 |
| shrink_half | Brier playoffs | .10354 | .10327 | +.00027 | .00113 | +0.24 | 7470 |
| shrink_base | Brier playoffs | .10354 | .10325 | +.00029 | .00110 | +0.26 | 7470 |
| shrink_wf | Brier playoffs | .10354 | .10349 | +.00005 | .00148 | +0.03 | 7470 |
| chain_pu_half | Brier playoffs | .10354 | .10300 | +.00054 | .00090 | +0.60 | 7470 |
| chain_pu_double | Brier playoffs | .10354 | .10858 | −.00504 | .00166 | −3.04 | 7470 |
| chain_pu_boot | Brier playoffs | .10354 | .10285 | +.00069 | .00062 | +1.10 | 7470 |
| record_500 | wins MAE | 4.5075 | 5.8041 | −1.2966 | .15753 | −8.23 | 7470 |
| record_500 | Brier playoffs | .10354 | .11191 | −.00838 | .00275 | −3.05 | 7470 |

**The proposal does not clear the gate.** On the metric it was meant to fix —
playoff Brier — it is **+.00015 worse (t +0.11)**: not a loss, not a win,
nothing. On projected wins it is **+.0104 of MAE worse (t +4.18)**, small in
size (0.2% of 4.50) but significant in sign. Its only nominal gains are
division (−.00112, t −0.69), pennant (−.00009, t −0.54) and World Series
(−.00005, t −0.55), none of which separates from zero. Per architecture.md §3
the point estimate keeps running.

Season by season, playoff Brier, `chain_pu` minus `chain`: **+.0032, +.0048,
+.0076, −.0029, −.0010, −.0018, −.0022, +.0016, +.0003, −.0074** for 2015,
2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025. Six of ten nominally
better, four worse, the largest single-season move (−.0074 in 2025) in its
favour. This is noise with the sign flipping, which is what a t of 0.11 on ten
clusters looks like when it is honest.

---

## 6. Did the late-July crossover move? No.

The pre-registered question. Paired against `record_500` on playoff Brier,
bucket by bucket. The crossover is where the row turns positive.

| Season played | chain Δ vs record_500 | se | t | chain_pu Δ | se | t |
|---|---:|---:|---:|---:|---:|---:|
| 0–15% | **−.03405** | .00745 | −4.57 | **−.02879** | .00495 | −5.82 |
| 15–30% | −.01387 | .00611 | −2.27 | −.01565 | .00449 | −3.49 |
| 30–45% | −.00830 | .00735 | −1.13 | −.00919 | .00501 | −1.83 |
| 45–60% | −.00627 | .00361 | −1.74 | −.00678 | .00291 | −2.33 |
| 60–75% | −.00191 | .00249 | −0.77 | −.00191 | .00222 | −0.86 |
| 75–90% | **+.00158** | .00206 | +0.77 | **+.00154** | .00180 | +0.86 |
| 90–100% | **+.00243** | .00138 | +1.77 | **+.00229** | .00133 | +1.72 |

On the twentieths grid the served chain first turns positive at **70–75% of
the season** (+.00026) and so does the uncertainty arm (+.00004). **The
crossover is in the same place, to the week.** H1 fails.

And it fails in the direction §1.3 predicted. The width's effect on the score
is concentrated in April, where the model was already winning by the largest
margin, and it is a *cost* there:

| Season played | chain_pu − chain, Brier playoffs | se | t |
|---|---:|---:|---:|
| 0–15% | **+.00526** | .00318 | +1.65 |
| 15–30% | −.00178 | .00239 | −0.75 |
| 30–45% | −.00089 | .00297 | −0.30 |
| 45–60% | −.00051 | .00147 | −0.34 |
| 60–75% | −.00000 | .00055 | −0.00 |
| 75–90% | −.00004 | .00051 | −0.08 |
| 90–100% | −.00015 | .00017 | −0.88 |

Nine tenths of the effect is in the first fifth of the season, and it is
negative. From 60% on the two arms are the same board: the width has stopped
being able to change anything, because there is almost no schedule left for it
to act through (§4: 0.28 wins of spread in the last fortnight).

**One thing the width does buy, and it is not accuracy.** The uncertainty
arm's advantage over `record_500` is *more consistent across seasons*: the
same −.0085-ish edge on playoff Brier comes with a season-clustered standard
error of .00275 instead of .00376, so t goes from −2.27 to −3.05. Shrinking
toward the base rate cannot make a forecast better in a season where it was
right, but it caps how badly it can lose in a season where it was wrong.
That is variance reduction, not skill, and the gate scores skill.

---

## 7. Is it information or is it shrinkage? Shrinkage — and too much of it

The width, expressed as the single linear shrinkage toward the base rate that
best mimics it, beside the shrinkage that would actually have minimised Brier
on the same rows. 1.0 is no shrinkage; below 1.0 is pulling toward the base
rate; above 1.0 is *sharpening*.

| Outcome / bucket | λ the width applies | λ that would have been right |
|---|---:|---:|
| P(playoffs), all rows | **0.933** | **0.968** |
| P(pennant), all rows | 0.949 | **1.031** |
| P(playoffs), 0–15% | **0.740** | **1.001** |
| P(playoffs), 15–30% | 0.828 | 0.897 |
| P(playoffs), 30–45% | 0.900 | 0.953 |
| P(playoffs), 45–60% | 0.945 | 0.958 |
| P(playoffs), 60–75% | 0.973 | 0.983 |
| P(playoffs), 75–90% | 0.990 | 0.978 |
| P(playoffs), 90–100% | 0.999 | 1.007 |

This is the sharpest statement in the document.

* **The served board is barely over-confident at all.** The best in-sample λ
  on P(playoffs) is **0.968** — three percent of shrinkage, on a fitted oracle
  that is allowed to see the answer. On P(pennant) it is **1.031**: the board
  is *under*-confident there, and the right correction is to make the tail
  probabilities more extreme, not less. There is very little over-confidence
  for parameter uncertainty to remove, which is the same fact finding (b)
  reported from the other direction — reliability .00055 is a mean squared
  calibration miss of 2.3 points.
* **The width applies roughly twice the shrinkage that helps**, 0.933 against
  0.968, and it applies it in the wrong *shape*: 0.740 in April, where the
  right answer is 1.001 — no shrinkage whatsoever — and 0.999 in September,
  where a little would have been fine.
* **And the oracle shrinkage is worth almost nothing anyway.** `shrink_half`
  and `shrink_base` are fitted on the very rows they are scored on and buy
  .00012 and .00014 of Brier. Walk-forward (`shrink_wf`, λ fitted on prior
  seasons only) it is worth **−.00010**: slightly *worse* than not shrinking,
  because the λ prior seasons ask for is above 1.0 in six of the nine seasons
  that have a prior (1.026, 1.051, 1.066, 1.031, 1.016, 1.001, 0.997, 0.999,
  0.990) and the sign is not stable.

So the pre-registered decision rule in §1.4 is answered twice over. The
uncertainty arm does not beat the tuned-shrinkage control (+.00027 against
`shrink_half`, t +0.24; +.00029 against `shrink_base`, t +0.26) — but the
control itself is worth nothing, so the more useful statement is the stronger
one: **there is no over-confidence here for either of them to fix.**

### The calibration table says the same thing in a different alphabet

| Decile | chain: mean predicted | realized | gap | chain_pu: mean predicted | realized | gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .0000 | .0000 | +.0000 | .0000 | .0000 | +.0000 |
| 2 | .0002 | .0000 | −.0002 | .0013 | .0000 | −.0013 |
| 3 | .0099 | .0174 | +.0075 | .0265 | .0295 | +.0029 |
| 4 | .0598 | .0910 | +.0312 | .1027 | .0790 | **−.0237** |
| 5 | .1558 | .1754 | +.0196 | .2054 | .1566 | **−.0488** |
| 6 | .2960 | .3066 | +.0105 | .3237 | .3280 | +.0043 |
| 7 | .4888 | .4993 | +.0106 | .4700 | .4913 | +.0213 |
| 8 | .6972 | .6399 | **−.0573** | .6376 | .6546 | +.0170 |
| 9 | .9011 | .8768 | −.0243 | .8484 | .8675 | +.0191 |
| 10 | .9949 | .9973 | +.0024 | .9882 | .9973 | +.0091 |

H3 half-holds and half-backfires. The width **does** close decile 8 — the one
visible defect of the served board, clubs given 60–80% who make it 64% of the
time — from −.057 to +.017. It pays for that by opening two new gaps of the
opposite sign in deciles 4 and 5 (−.024 and −.049): clubs it now gives 10% and
21% actually make it 8% and 16% of the time, because the width pushed them up
off the floor and they did not belong there. Reliability over all ten deciles
moves .00055 → .00042 and resolution .12569 → .12576, and the Brier score does
not move at all. **The miscalibration was not a global over-confidence; it was
one bin, and a global widening cannot fix one bin without breaking others.**

---

## 8. The other four predictions

| | Prediction | Outcome |
|---|---|---|
| **H1** | Playoff Brier improves, more late than early; the crossover moves | **Fails.** +.00015 (t +0.11) pooled; +.00526 in the first fifth and −.00015 in the last tenth — the effect is front-loaded and the sign is against us where it is largest. The crossover is at 70–75% for both arms. |
| **H2** | The tails gain more, relatively, than the playoff column | **Weakly holds, and it is the only thing that does.** Pennant −.00009 (t −0.54, −0.2% of the score) and World Series −.00005 (t −0.55, −0.2%) against playoffs' +0.1%. Division is the largest nominal gain in the document at −.00112 (t −0.69), −1.4% of the score. None separates from zero; ten seasons carry twenty pennants and ten champions. |
| **H3** | Shows up as reliability, closing decile 8 | **Half.** Decile 8 closes (−.057 → +.017); deciles 4 and 5 open (−.024, −.049); pooled reliability .00055 → .00042 and Brier does not move. |
| **H4** | Beats the tuned-shrinkage control, because the width is heterogeneous | **Fails, and the reason is more interesting than the failure.** It does not beat the control (+.00027, t +0.24) — but the control is worth .00012 on an oracle fit and −.00010 walk-forward, so there was nothing there to beat. The heterogeneity is real (λ 0.740 in April to 0.999 in September) and is heterogeneous in the *wrong direction*: it shrinks hardest exactly where the fitted λ says not to shrink at all. |
| **H5** | Projected wins essentially unchanged | **Fails, by a mechanism worth naming.** +.0104 of MAE (t +4.18). log5 with a home-field odds multiplier is exactly linear in the logit of strength, and `expit` is concave above .500 and convex below it, so a symmetric shift on that logit *lowers* a favourite's expected win probability and *raises* an underdog's. Projected wins are compressed toward 81 by Jensen's inequality: the slope of `chain_pu`'s projected wins on the chain's, about 81, is **0.99724** (SD 10.248 against 10.276). A 0.28% compression of a column whose MAE is 4.50 is +.010 of MAE, which is what was measured. It is a real bias, not sampling noise, and doubling the width doubles it (slope 0.98992, MAE +.026). |

---

## 9. The width's *shape* is worth more than its size — and neither is worth anything

`chain_pu_boot` is the width this document argues is theoretically wrong: the
bootstrap standard error of the shrunk estimator, narrower everywhere and, in
April, nearly zero. It scores **.10285** on playoff Brier against the
posterior width's .10354 and the served chain's .10339 — the best of the
three, by .0005 and .0007, at t = +1.10 against the posterior arm.
`chain_pu_half`, which is just the posterior width halved, scores .10300.

The ordering across the four widths is monotone in how much they shrink:

| Arm | λ applied to P(playoffs) | Brier playoffs |
|---|---:|---:|
| chain (no width) | 1.000 | .10339 |
| chain_pu_half | 0.979 | .10300 |
| chain_pu_boot | 0.968 | .10285 |
| chain_pu (proposal) | 0.933 | .10354 |
| chain_pu_double | 0.831 | .10858 |

which is a shallow U with its floor at λ ≈ 0.97 — precisely the fitted oracle
λ of 0.968 — and a total depth of **.0005 of Brier**, one seventeenth of the
model's own margin over `record_500` and about four times the Monte Carlo
noise floor at 2,000 sims. The whole family is one reparameterisation of "how
much do you shrink toward the base rate", the answer is "a little, and it does
not matter", and **the theoretically correct width lands on the wrong side of
the theoretically incorrect one** because the correct one shrinks more than
helps. Reporting the incorrect width as a win would be fitting the evaluation
set; it is reported as what it is, which is the U-curve's floor.

---

## 10. What this means, plainly

**The pre-registered hypothesis is refuted.** Understated parameter
uncertainty is not what finding (a) is made of, and the roadmap's ordering —
parameter uncertainty first, because it explains the shape of the station G
result — is wrong. Three independent lines say so:

1. **The timing is backwards.** The width's influence falls by a factor of 26
   from April to September (7.4 wins of spread to 0.28) because it acts only
   through unplayed games. The thing to be explained lives in August and
   September. A cause whose effect has almost vanished by the time the effect
   appears is not the cause.
2. **The over-confidence is not there.** The best-fitting shrinkage of the
   served playoff probabilities is 0.968 — and 1.031 on pennants, i.e.
   *sharpening*. Finding (b) already said this: reliability .00055 is a
   forecast that is almost exactly calibrated. A well-calibrated forecast has
   no over-confidence to remove, and removing some anyway costs Brier.
3. **The one real miscalibration is local, not global.** Decile 8 misses by
   −.057; every other decile is inside .031. A width applied to all thirty
   clubs at every date cannot fix one decile without breaking two others, and
   that is exactly what it does.

**What the late-July crossover actually is, then.** Not over-confidence. The
simplest remaining reading is the one
[team-projection-backtest.md](team-projection-backtest.md) §8 already gave:
from August on, a club's playoff probability is mostly a function of its
banked record and the number of games left, both of which a pocket calculator
has, and the marginal information in a talent estimate over 30 remaining games
is small enough to be inside the noise. That is a statement about **how little
signal is left in the remaining schedule**, not about how our probabilities
are shaped. Widening the strength distribution changes the shape; it cannot
add signal, and signal is what is missing.

**The next place to look**, on this evidence, is not the width of the strength
estimate but its *level* late in the season — whether the blend's weight
between the top-down record and the bottom-up rates should move as the season
goes on, which is a question about the mean and can only be answered by
something the standings do not already contain. Roster changes after the trade
deadline are the obvious candidate and are not in station C at all.

---

## 11. What changes in the code

* `simulate_remaining`, `run_playoff_odds` and `team_season.project` take a
  distribution as well as a point estimate. **The served model is unchanged**:
  the point estimate is the default everywhere, and the zero-width equivalence
  test guarantees the board is bit-identical to before.
* `scripts/run_playoff_odds.py --strength-uncertainty SCALE` exists, defaults
  to 0, and its help text says it does not clear the gate.
* `scripts/run_team_backtest.py` carries `chain_pu`, `chain_pu_half`,
  `chain_pu_double` and `chain_pu_boot` as scored arms, so the next width to be
  proposed has a harness and four measured points to beat.
* Nothing on the site moves.

## 12. What this does not settle

1. **This is not a posterior** (§2.3). It carries no uncertainty from the
   bottom-up half of station C's blend, and it holds a club's talent fixed for
   the rest of the season. A real posterior would be *wider*, and wider is the
   direction that scored worse here — `chain_pu_double` is .0052 of Brier
   behind at t −3.04 — so the honest reading is that this result argues
   against wider, not that a better-shaped width is ruled out. A width that
   was **narrower in April and wider in August** — the opposite of every width
   tried here — has not been tested and is the only shape the data leaves room
   for.
2. **The tails are still unmeasured, not measured-as-zero.** Ten seasons carry
   twenty pennants and ten champions. Pennant and World Series Brier moved in
   the arm's favour and at t = −0.5 that is worth nothing either way.
3. **Ten seasons is ten clusters.** Every standard error here has 9 degrees of
   freedom.
4. **2,000 simulations.** The Monte Carlo adds about 1.2 × 10⁻⁴ to every
   arm's Brier. Common random numbers remove most of it from the paired
   differences — the point arm and the uncertainty arm draw the same uniforms
   — but the U-curve of §9 has a total depth of only 5 × 10⁻⁴, so its ordering
   between `chain_pu_half` and `chain_pu_boot` should not be over-read.
5. **The shrinkage controls fit one λ per outcome**, not one per bucket. A
   per-bucket oracle would be stronger still, and would beat the uncertainty
   arm by more; it was not needed, because the global one already ties.
