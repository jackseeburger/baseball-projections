# Statcast Contact Quality Inside the Rate Models

**BAS-58.** Station A. Sept 3, 2026.

Statcast has been in R2 since 2015 and had never been read by a model —
[methods.md](methods.md) §2 calls it "our largest untouched asset". This is
what happens when it is: six exit-velocity and launch-angle aggregates per
player, added as covariates to the component projections, scored walk-forward
against the live baseline; and then the tutorial's Hilbert-space Gaussian
process over the same two coordinates, asked to replace those six.

**The headline.** Stage 1 clears the gate. On five hitter components and four
of five pitcher components the contact arm beats `marcel_tuned` — the model the
site serves — out of sample on the common player set, by **1.6% to 4.8% of
component MAE**, at |t| from 2.6 to 5.9 clustered by player, on 3,100–4,900
paired projections over five seasons and three cutoffs. That is larger than the
gain from tuning Marcel's constants (1.1%) and of the same order as feeding
Marcel the current season (6–11%). Stage 2 does not: the HSGP surface is
**significantly worse** than the six hand-chosen aggregates it was built to
replace, on every component of both sides, and adding it alongside them is
worth nothing. The reason is the interesting part, and it is §7.

---

## 1. What the question actually was

Contact quality is a measurement of what already happened. The projection
harness already sees the outcomes that contact produced — home runs, hits on
balls in play, extra bases — and Marcel regresses those to the league with a
fitted ballast. Exit velocity and launch angle are a *different measurement of
the same events*. So the honest question is not "does Statcast predict
anything", it is:

> Does a contact-quality aggregate carry information about the rest of the
> season **beyond** what the realized outcome rate already carries, or does it
> merely restate it with less noise?

Both answers are worth having and they look different in the numbers. Extra
information shows up as a gain that survives at large samples. Pure variance
reduction shows up as a gain that lives in the players with little exposure and
at the early cutoff, and evaporates by August. §6 splits it three ways and the
answer turns out to be **component-specific**: BABIP and K% are denoising, ISO
and pitcher HR/BF are information.

## 2. The data

`data/features/contact_quality_monthly.parquet` — 81,000 rows, 2.5 MB,
committed. Twelve seasons of pitch-level Statcast (1.4 GB in R2, 1,388,220
tracked batted balls) reduced to sufficient statistics per player, per calendar
month, for hitters and pitchers alike: batted balls, EV sums and sums of
squares, LA sums, barrels, hard-hit, sweet-spot, batted-ball-type mix, and an
exit-velocity histogram in 2.5 mph bins for quantiles.
`scripts/build_contact_quality.py` rebuilds it; the file is committed precisely
so the 1.4 GB download never has to happen twice.

**Why monthly.** The harness cuts a season at a date and may only see data
strictly before it. Monthly buckets are additive, so a cutoff on the first of a
month — every cutoff the intra-season harness uses — is reconstructed *exactly*
by summing the buckets before it, with no filtering of a multi-million-row
table at score time and no way for a later month to leak into an earlier
feature. A cutoff that is not the first of a month is **refused**, not rounded:
rounding a cutoff forward is leakage. This is also the one thing standing
between the result and the live board — see §9.

**Barrels come from the archive.** Statcast's own `launch_speed_angle == 6`,
not one of the mutually disagreeing public re-derivations.

## 3. The estimator, and the two controls it needs

Three arms, all reading the same training frame, all scored by the harness's
own `score()` on the harness's own common player set:

| Arm | What it is |
|---|---|
| `marcel_tuned` | the live baseline, untouched (`marcel_pitcher_tuned` on the pitcher side) |
| `contact_recal` | `a + b·baseline`, with `(a, b)` fitted on **earlier seasons only** |
| `contact` | the same fit plus the six standardized contact covariates |

`contact` − `marcel_tuned` is the gate. `contact` − `contact_recal` is what the
covariate itself is worth: without that control, a fitted recalibration of
Marcel would be credited to Statcast. Two more arms exist for specific
questions — `contact_additive`, which pins the baseline's coefficient at 1 and
adds contact as a pure correction (§8, the deployable shape), and
`contact_shuffled`, the permuted control (§5).

**The covariates**, each shrunk toward the league by `ballast` batted balls and
then standardized batted-ball-weighted across the players at that cutoff:
mean EV, 90th-percentile EV (off the committed histogram), barrel rate,
hard-hit rate, sweet-spot rate, mean launch angle.

**Every constant is walk-forward.** The recency weights over the three-season
window and the shrinkage ballast are chosen by grid search on cells through
2021 — pooled MAE over the contact-dependent components — and every scored
season is later than that. The regression coefficients are refit for each
scored season on cells strictly before it. Chosen: hitters (1.0, 0.35, 0.1) and
5 batted balls of ballast; pitchers (1.0, 0.6, 0.35) and 20. Both surfaces are
shallow — the whole hitter grid spans 0.4% of MAE — so read those as the flat
region, not as sharp optima.

**The scored set.** Cell seasons 2017–2026 excluding 2020 (a 60-game season
that started July 23 has no May 1 cutoff); tuning through 2021; **holdout
2022–2026 at the May 1, Jul 1 and Aug 1 cutoffs**, 15 cells per component per
side, `min_trials = 100` realized trials, exactly as
`scripts/run_intraseason_backtest.py` scores.

**Standard errors are clustered by player.** One hitter appears in three
cutoffs of five seasons, and those fifteen rows are not fifteen independent
observations. `src/eval/tuning.paired_abs_error_diff` now takes a
`cluster_col`; without it the t values below are inflated by about 30%, which
is roughly the square root of the cutoffs — hitter K% reads −6.42 unclustered
against −4.68 clustered, ISO −5.65 against −4.07, BABIP −3.34 against −2.55.
Every t in this document is the clustered one.

## 4. Stage 1 — the result

Pooled over the 2022–2026 holdout. `diff` is the trials-weighted paired
difference in absolute error per player-cell, negative meaning the contact arm
is better; `t` is clustered by player; `n` is player-cells and `clusters` is
distinct players.

### Hitters

| Component | `marcel_tuned` MAE | `contact` MAE | paired diff | % of MAE | t | n | clusters | win rate |
|---|---|---|---|---|---|---|---|---|
| **ISO** | .038820 | **.037108** | −.001712 | −4.4% | **−4.07** | 4663 | 746 | .533 |
| **HR/PA** | .010699 | **.010188** | −.000511 | −4.8% | **−4.48** | 4935 | 763 | .531 |
| **K%** | .030163 | **.029517** | −.000646 | −2.1% | **−4.68** | 4935 | 763 | .546 |
| **BB%** | .018824 | **.018519** | −.000305 | −1.6% | **−3.23** | 4935 | 763 | .516 |
| **BABIP** | .028633 | **.028135** | −.000497 | −1.7% | **−2.55** | 3696 | 658 | .519 |

### Pitchers

| Component | `marcel_pitcher_tuned` MAE | `contact` MAE | paired diff | % of MAE | t | n | clusters | win rate |
|---|---|---|---|---|---|---|---|---|
| **HR/BF** | .010140 | **.009656** | −.000483 | −4.8% | **−5.86** | 4881 | 948 | .557 |
| **BB%** | .017944 | **.017486** | −.000458 | −2.6% | **−5.58** | 4881 | 948 | .566 |
| **(BB+HBP)/BF** | .019390 | **.018935** | −.000455 | −2.3% | **−5.36** | 4881 | 948 | .565 |
| **BABIP against** | .028208 | **.027737** | −.000470 | −1.7% | **−3.00** | 3134 | 792 | .526 |
| **K%** | .031570 | .031718 | +.000148 | +0.5% | +0.86 | 4881 | 948 | .480 |

### How much of that is the covariate rather than the recalibration

`contact` against `contact_recal` — the same fitted rescaling of the baseline,
with the covariates removed:

| Component | vs `contact_recal` | t | Reading |
|---|---|---|---|
| hitter ISO | −.002031 | −7.09 | all of it, and then some: `contact_recal` alone is *worse* than the baseline (+.000318) |
| hitter HR/PA | −.000492 | −6.27 | essentially all of it |
| hitter K% | −.000438 | −4.37 | two thirds of it |
| hitter BABIP | −.000454 | −3.32 | essentially all of it |
| hitter BB% | −.000275 | −2.96 | most of it |
| pitcher HR/BF | −.000445 | −6.14 | essentially all of it |
| pitcher BABIP | −.000454 | −2.88 | essentially all of it |
| pitcher BB% | −.000012 | −0.53 | **none of it** — the whole gain is a level correction |
| pitcher (BB+HBP)/BF | −.000009 | −0.31 | **none of it** |
| pitcher K% | −.000074 | −0.98 | nothing either way |

The two pitcher walk rates are the clearest lesson in the table: they show a
large, highly significant win over the baseline (t −5.6) that has **nothing to
do with contact quality**. Fitted recalibration of the pitcher Marcel is worth
that on its own. Without the `contact_recal` control we would have published
"Statcast improves pitcher walk projections by 2.6%", and it would have been
false.

### Where the baseline's own constants are also out of sample

`src/eval/marcel_params.json` was fitted on 2020–2024, so on the 2022–2024
holdout cells the *baseline* has seen its own future. That biases against the
contact arm, but it is worth isolating the tail where it does not apply:

| Component | 2025–2026 diff (t, n) | 2026 alone diff (t, n) |
|---|---|---|
| hitter ISO | −.002100 (−3.10, 1664) | −.002059 (−2.25, 704) |
| hitter HR/PA | −.000592 (−3.17, 1801) | −.000602 (−2.42, 795) |
| hitter K% | −.000535 (−2.66, 1801) | −.000763 (−3.01, 795) |
| hitter BB% | −.000313 (−2.10, 1801) | −.000337 (−1.54, 795) |
| hitter BABIP | −.000408 (−1.25, 1276) | −.000677 (−1.47, 495) |
| pitcher HR/BF | −.000414 (−3.31, 1764) | −.000300 (−1.83, 751) |
| pitcher BABIP | −.000345 (−1.31, 1094) | −.000038 (−0.11, 429) |
| pitcher K% | −.000022 (−0.09, 1764) | +.000007 (+0.02, 751) |

Nothing changes sign. On 2026 alone — one season, three cutoffs — hitter K%,
HR/PA and ISO stay significant; BB%, BABIP and the pitcher rates do not, which
is what 400–800 player-cells buys.

## 5. Three guards, and what each caught

**Leakage.** Features are summed from monthly buckets strictly before the
cutoff; a cutoff that is not the first of a month is refused rather than
rounded; and the guard re-checks the filtered rows rather than trusting the
filter. `tests/test_eval/test_contact.py` drives it with a synthetic season
whose every post-cutoff month is a thousand batted balls of 120 mph barrels,
and asserts the feature at a May 1 cutoff is *identical* to the feature built
from a frame with those rows deleted. The same guard on batted-ball rows covers
stage 2.

**The permuted control** (methods.md §5.4). `contact_shuffled` permutes the
covariates across players within each cell — every player keeps a real
covariate vector, attached to the wrong player — and refits identically. It
lands exactly on the recalibration control: +.000052 (hitter K%), +.000015
(BB%), +.000005 (HR/PA), −.000001 (BABIP), −.000006 (ISO) against
`contact_recal`. If the pipeline were fitting the split rather than the
covariate, this arm would win too. It does not.

**The pre-registered falsification check.** Contact quality is a measurement of
batted balls, so it should say nothing about strikeouts and walks, where no
batted ball exists. On the pitcher side it behaved: K% is flat, and both walk
rates are pure recalibration. On the **hitter** side it fired — K% is the
*largest* t in the table. That is a genuine covariate effect, not an artifact:
the permuted arm buys nothing, and the effect is stable across seasons. The
explanation is that a hitter's launch-angle and exit-velocity profile is a
signature of his swing, and swing type predicts strikeouts and walks; the
fitted K% coefficients (mean EV −0.0078, EV90 +0.0036, barrel +0.0052) say the
model is reading swing shape, not contact value. We flag it as an unexpected
result rather than a clean one, and it does not change the gate.

## 6. Information, or variance reduction?

The question §1 set. Three cuts of the same paired difference, as % of the
baseline's own MAE on that slice so a shrinking error scale cannot masquerade
as a fading effect.

| Component | low exposure | high exposure | May 1 | Jul 1 | Aug 1 | Verdict |
|---|---|---|---|---|---|---|
| hitter BABIP | −2.5% | −1.0% | −2.7% | −0.9% | −0.2% | **denoising** |
| hitter K% | −2.3% | −1.9% | −3.1% | −1.3% | −0.8% | mostly denoising |
| hitter BB% | −1.9% | −1.3% | −2.1% | −1.4% | −0.9% | mostly denoising |
| hitter HR/PA | −4.8% | −4.7% | −6.5% | −4.4% | −1.2% | mixed |
| hitter ISO | −3.1% | **−5.7%** | −5.4% | −4.1% | −2.3% | **information** |
| pitcher HR/BF | −4.1% | −5.3% | −5.9% | −4.4% | −2.3% | **information** |
| pitcher BABIP | −2.3% | −1.1% | −2.1% | −0.8% | −1.5% | **denoising** |

"Exposure" here is the batted balls behind the covariate itself, split at the
median; the split on current-season plate appearances tells the same story.

So the ticket's suspicion is half right. BABIP on both sides of the ball, and
the hitter walk and strikeout rates, behave exactly as a variance reduction on
a small sample should: the gain is concentrated in the low-exposure half and is
gone by August. **ISO and pitcher HR/BF do not.** Their gain is as large or
larger for the players with the most batted balls behind them and survives at
the August cutoff. Those two are carrying information the realized outcome rate
does not have — which is the textbook story for both of them, because a
hitter's extra-base output and a pitcher's home runs allowed are the two rates
where the outcome is a small, heavily-context-dependent sample of a physically
much better measured process.

That both readings appear in one table, from one estimator, is the finding.
"Statcast is a denoiser" and "Statcast is new information" are both true, of
different components, and a single number for the station would have hidden it.

## 7. Stage 2 — the HSGP surface, and why it loses

Stage 1 paid, so stage 2 was built: `pm.gp.HSGP` over the (EV, LA) plane with
the wOBA of the batted ball as the response — methods.md §3's two-stage
pattern, the reference tutorial's own technique. The plane is binned to
2.5 mph × 5°, each cell's mean value observed with its own known precision
(the cell's standard deviation over its own batted balls, divided by the root
of its count), the GP placed over the whole grid so unpopulated corners come
back as interpolation rather than holes, and the surface refitted at **every
walk-forward fold** on strictly earlier seasons. A player's covariate is that
surface averaged over his own pre-cutoff batted balls, shrunk and standardized
exactly as the six are. One covariate replaces six.

Five fits, 721–771 cells over 493k–987k batted balls, max r-hat 1.01, zero
divergences, min bulk ESS 481, length scales stable at ≈0.46 and ≈0.26 of a
standardized unit across every fold. The surface is sane: on the 2026 fold,
70 mph at 10° is worth .43, 90 mph at 10° .64, 105 mph at 25° 1.68, 110 mph at
25° 2.04 — a home run is 2.0 — and 95 mph at −20°, the same exit velocity
buried into the ground, is worth .18. It is, recognizably, the
expected-wOBA-on-contact surface, and it was learned rather than asserted.

It still loses.

| Component | `contact_hsgp` vs `contact` | t | `contact_both` vs `contact` | t |
|---|---|---|---|---|
| hitter ISO | **+.001767** | +6.75 | +.000010 | +0.68 |
| hitter HR/PA | **+.000432** | +5.90 | +.000002 | +0.41 |
| hitter BABIP | **+.000287** | +2.28 | −.000017 | −0.98 |
| hitter K% | **+.000173** | +2.04 | −.000023 | −1.09 |
| hitter BB% | **+.000122** | +3.19 | +.000003 | +0.35 |
| pitcher HR/BF | **+.000381** | +5.73 | +.000007 | +1.80 |
| pitcher BABIP | **+.000462** | +3.01 | −.000002 | −0.29 |

Positive is worse. The surface is significantly worse than the six hand-chosen
aggregates on every component of both sides, and putting it *alongside* them
adds nothing anywhere (|t| ≤ 1.8). Against the baseline it is not useless — it
still clears the gate on hitter K% (−.000473, t −4.25) and the pitcher walk
rates — but on the two components where stage 1 was carrying real information,
hitter ISO and pitcher HR/BF, it is worth **nothing at all** (+.000055, t 0.18;
−.000103, t −2.39 of which the recalibration control explains all but −.000065).

**Why.** The surface maps a batted ball to *what it was worth*. Averaging the
league's own value function over a player's batted balls returns approximately
what those balls were worth to him — which is approximately what his realized
outcome rate says, and the baseline already reads his realized outcome rate.
The single number collapses back onto the thing it was supposed to add to. The
six aggregates do not, because they keep **how hard** and **at what angle**
separate from **what it was worth**, and the extra information lives in exactly
that separation: two hitters with the same expected wOBA on contact, one
reaching it through 108 mph ground balls and the other through 95 mph fly
balls, are different hitters going forward, and the surface value cannot tell
them apart.

This is the ticket's own warning, arriving from an unexpected direction. It was
issued about contact quality in general; it turns out to apply precisely to the
*most sophisticated* version of it. The flexible model did its job — the
surface is right — and the covariate built from it is the wrong summary.

Two caveats we would want closed before calling this settled. The response is
wOBA on contact, which is the value definition most likely to collapse onto the
outcome; a surface predicting *next* season's outcome from this season's
(EV, LA) would be a different and possibly better object. And the surface is
league-wide with no player term, so it is a measurement device rather than a
hierarchical model — the tutorial's version puts the GP inside the model that
does the pooling, and here it sits in front of a Marcel that does the pooling
with a fixed ballast. Neither caveat changes what was measured; both name the
next experiment.

## 8. What is wired, and what stands in the way

**Nothing is wired.** The result clears the gate in the harness and the change
to the served board is not made in this pass, for one concrete reason and one
judgement call.

The concrete reason: **the live projection is made on an arbitrary date.**
Today is September 3. The committed artifact is monthly, so it can answer a
question asked on the first of a month exactly and refuses any other date
rather than rounding one forward. Serving this needs a partial-month top-up
computed from the current season's Statcast file — which the daily ingest
(`scripts/ingest_statcast.py`) already puts in R2 — and that path has no
walk-forward score of its own yet. It is small and it is the next ticket.

The judgement call: the shape to ship is `contact_additive`, not `contact`.
That arm pins the baseline's coefficient at exactly 1 and adds contact quality
as a correction to the served projection, leaving `marcel_tuned` untouched. It
gives up roughly a third of the gain and keeps all of the sign:

| Component | `contact_additive` vs baseline | t | vs the free fit's −diff |
|---|---|---|---|
| hitter K% | −.000598 | −4.36 | 93% |
| hitter ISO | −.001171 | −3.16 | 68% |
| hitter BB% | −.000294 | −3.17 | 96% |
| hitter BABIP | −.000396 | −2.17 | 80% |
| hitter HR/PA | −.000346 | −3.47 | 68% |
| pitcher HR/BF | −.000493 | −5.93 | 102% |
| pitcher BABIP | −.000522 | −3.54 | 111% |
| pitcher BB% | −.000411 | −5.66 | 90% |
| pitcher K% | +.000192 | +1.27 | — |

The free fit buys the rest by rescaling Marcel — the fitted coefficient on the
baseline is 0.55 for ISO and 0.49 for HR/PA, which is the fit saying Marcel's
spread on those two is roughly twice what it should be. That is a real and
separately interesting claim about the baseline, and it belongs in a ticket
about Marcel's ballasts rather than smuggled in under a Statcast change.

## 9. Reproducing it

```bash
# the archive -> the committed monthly artifact (1.4 GB download, once)
python scripts/build_contact_quality.py --download

# PA-level outcomes for every season, from the same raw files
python -c "from src.data.pa_outcomes_pipeline import build_pa_dataset; \
           build_pa_dataset(data_dir='data/raw')"

# stage 1 + the permuted control + stage 2, both sides
python scripts/run_contact_backtest.py --side hitter  --tune --shuffle-control \
    --hsgp --hsgp-cache data/cache/surfaces --json-out out_hitter.json
python scripts/run_contact_backtest.py --side pitcher --tune --shuffle-control \
    --hsgp --hsgp-cache data/cache/surfaces --json-out out_pitcher.json
```

The two payloads behind every number above are committed at
`data/models/contact_quality_hitter.json` and `..._pitcher.json`. The five
surface fits take about three minutes each; everything else is seconds.

Code: `src/data/contact_quality.py` (the artifact), `src/eval/contact.py`
(stage 1), `src/eval/hsgp_contact.py` (stage 2),
`scripts/run_contact_backtest.py` (the harness driver).
Tests: `tests/test_data/test_contact_quality.py`,
`tests/test_eval/test_contact.py`, `tests/test_eval/test_hsgp_contact.py`.
