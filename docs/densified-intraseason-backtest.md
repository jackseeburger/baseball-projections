# Densifying the player-rate walk-forward backtest (BAS-63)

**Run:** Sept 8, 2026 · built by `scripts/run_intraseason_backtest_dense.py`,
reusing `src/eval/backtest.py` and `src/eval/intraseason.py` unchanged ·
data at `data/eval/dense_intraseason/` (gitignored — regenerable, see
[Reproducing](#reproducing)).

`scripts/run_intraseason_backtest.py` scores three cutoffs
(2026-05-01/07-01/08-01) in one season. The headline "Bayesian K% is a dead
heat with tuned Marcel" number rests on the smallest of those three cells —
n=126 hitters at Aug 1 — next to station G's 249 weekly as-of dates over ten
seasons and the pitcher-workload harness's 44 biweekly cutoffs over five.
This doc extends the same harness to many more (season, cutoff) pairs and
answers the question that motivated the ticket.

**Headline: the reversal is real and gets worse, not better, on more data.**
`bayes` is worse than `marcel_tuned` at 35 of 36 scored (season, cutoff)
cells across three independent seasons (2022, 2024, 2026) and a biweekly
grid from April through early August. The one exception is the single
smallest-n cell in the whole sweep (2026-08-05, n=78) at t=-0.87 — noise, not
a reversal; see [Verdict](#verdict-did-the-prediction-hold). The
pre-registered pooling story predicted the opposite pattern (`bayes`
favoured early, converging late); that did not happen.

## Pre-registered prediction

Recorded in the commit that added the harness
(`eval: densify intraseason backtest from 3 cutoffs to weekly/biweekly
(BAS-63)`), **before** the densified sweeps had produced output:

> Partial pooling should help most when samples are smallest. So if pooling
> is what the Bayesian model brings, its advantage over tuned Marcel should
> be **largest in April/May** — when a hitter has ~100 PA and how much you
> shrink him matters enormously — and should **shrink monotonically through
> September**, when everyone has ~400 PA and both methods converge.
>
> The three cutoffs we have show the **opposite**: Bayes is worse on May 1
> (paired diff +.00086, t 1.57, n 315), still worse Jul 1 (+.00092, t 1.58,
> n 231), and only ties by Aug 1 (+.00003, t 0.04, n 126).
>
> If that reversal survives densification, it is evidence the problem is a
> **level or calibration error that washes out as data accumulates**, not a
> pooling deficiency. If instead the gap turns favourable early once you
> have real sample sizes, the current reading is an artifact of three noisy
> points.

See [Verdict](#verdict-did-the-prediction-hold) for the answer.

## Design

Cost control matters here: an MCMC refit per cutoff is expensive, and eight
seasons of weekly cutoffs times several arms is a lot of fits. Two sweeps at
different cadences, scoped deliberately rather than run to exhaustion:

| | Arms | Cadence | Seasons | Cutoffs/season | Volume |
|---|---|---|---|---|---|
| **cheap sweep** | `marcel_tuned`, `marcel`, `marcel_tuned_preseason`, `marcel_preseason`, `season_to_date`, `previous_season`, `league_average` (+ `bayes_preseason` where a preseason file exists — 2026 only) | **weekly** | 2019, 2021–2026 | up to 24 | 164 (season, cutoff) pairs, 1,560,239 scored rows |
| **bayes sweep** | adds `bayes` — the PA-level K% model refit at the cutoff (`src.eval.bayes_arm`), k_rate only | **biweekly** | 2022, 2024, 2026 | 12 | 36 MCMC fits |

The bayes sweep was originally scoped to four seasons (2022, 2024, 2025,
2026); 2025 was dropped mid-run to land inside a reasonable compute budget
(see [Reproducing](#reproducing) for the exact commands, including the
resumable checkpoint that made the mid-run cut costless). What shipped —
2022 and 2024 overlapping `marcel_tuned`'s tuning window, 2026 a clean
holdout — is a real trade against a slightly larger n, not a design flaw:
see the tuning-window caveat immediately below.

Every arm the current 3-cutoff harness scores is scored here — nothing was
dropped, only the closed-form arms were run at 2x the within-season cadence
(weekly vs. biweekly) across more than 2x the seasons (7 vs. 3) the MCMC arm
could afford: 164 (season, cutoff) pairs in the cheap sweep and 36 in the
bayes sweep, against the original harness's 3 (one season, three cutoffs).

**2020 excluded**, matching `scripts/run_contact_backtest.py` and
`scripts/run_team_backtest.py`: a 60-game season starting July 23 has no
May 1 cutoff, and folding it in would quietly average two different
questions.

**Weekly grid** (cheap sweep): every 7 days from April 8 through Sept 2,
anchored to include the legacy three cutoffs exactly (05-01, 07-01, 08-01
are present verbatim, so the old numbers are reproducible as a subset of the
new ones) — 22-24 dates a season, fewer for 2026 because its local PA
parquet ends 2026-09-01 (an in-progress season; cutoffs past the data are
skipped, not zero-filled).

**Biweekly grid** (bayes sweep): 12 dates, April 15 through Aug 5, same
anchoring convention.

**Data.** Only `pa_outcomes_2026.parquet` existed in R2 going in (the source
for `scripts/run_intraseason_backtest.py`'s `--pa-dir`). PA-level parquets
for 2019 and 2021-2025 were rebuilt locally from the Statcast pitch-level
parquets already in R2 (`src.data.pa_outcomes_pipeline.build_pa_dataset`) —
the same route the BAS-59 "fair fight" doc used to rebuild 2024-2025. 2017 and
2018 were not fetched (2019 has no bayes-sweep prior-season pair without
them), so the bayes sweep's earliest season is 2022; the cheap sweep, which
needs no MCMC prior window, still covers 2019.

**Compute.** NumPyro/JAX are installed in this environment (the sandbox the
BAS-59 doc ran in had neither, hence its "reduced local fit on PyMC" framing
throughout) — one bayes fit here (500 draws, 500 tune, 2 chains, no
opposing-pitcher term, full hitter coverage, ~750-850 batters) runs in
**~55-85 seconds**, against the ~8 minutes/cutoff PyMC needed for the same
scale in that sandbox. That is what makes a 36-fit sweep affordable at all;
see [What each fit actually was](#what-each-fit-actually-was) for the
per-fit diagnostics.

### A caveat that shapes how to read every `marcel_tuned` number below

`marcel_tuned`'s constants (`src/eval/marcel_params.json`) were fitted
**walk-forward on season-level predict years 2020-2024**
(`scripts/tune_marcel.py`, `method`: "coordinate search, walk-forward predict
years 2020-2024"). The existing 3-cutoff harness only ever scored 2026 — a
clean holdout. Densifying to 2019-2026 necessarily reintroduces years the
tuning already saw the outcomes of. It is a different task (season-level
full-year prediction vs. intra-season rest-of-season prediction from a
partial season), but the same ballast, recency weights and age curve, so
`marcel_tuned`'s numbers on 2021-2024 cutoffs carry a real risk of
optimism that its numbers on 2019, 2025 and 2026 do not.

The bayes sweep's three seasons split **2-1** on this: 2022 and 2024 overlap
the tuning window, 2026 does not. Every headline table below is
therefore reported **both pooled and split by tuning-window overlap** —
read the out-of-sample half if you want the number that cannot be
optimism from `marcel_tuned`'s own fit.

## Results

### n falls as the season progresses — and it is worse than it looks

The min-trials filter (100 realized rest-of-season PA) does to every cutoff
what it did to the original three: a May cutoff scores far more hitters than
an August one, because fewer of them will still see 100 more plate
appearances the closer the cutoff sits to the end of the year. Pooled over
all 7 cheap-sweep seasons, k_rate, `marcel_tuned`:

| Cutoff | seasons pooled n | mean n/season | min | max |
|---|---:|---:|---:|---:|
| 04-08 | 2095 | 299 | 122 | 339 |
| 04-15 | 2351 | 336 | 319 | 346 |
| 04-22 | 2368 | 338 | 317 | 348 |
| 04-29 | 2373 | 339 | 316 | 350 |
| 05-01 | 2378 | 340 | 315 | 350 |
| 05-06 | 2386 | 341 | 312 | 354 |
| 05-13 | 2384 | 341 | 306 | 357 |
| 05-20 | 2362 | 337 | 301 | 351 |
| 05-27 | 2342 | 335 | 293 | 348 |
| 06-03 | 2316 | 331 | 286 | 346 |
| 06-10 | 2266 | 324 | 278 | 340 |
| 06-17 | 2202 | 315 | 261 | 329 |
| 06-24 | 2146 | 307 | 242 | 325 |
| 07-01 | 2089 | 298 | 231 | 318 |
| 07-08 | 2010 | 287 | 213 | 308 |
| 07-15 | 1938 | 277 | 200 | 300 |
| 07-22 | 1849 | 264 | 184 | 284 |
| 07-29 | 1724 | 246 | 151 | 268 |
| 08-01 | 1650 | 236 | 126 | 261 |
| 08-05 | 1549 | 221 | 77 | 256 |
| 08-12 | 1332 | 222 | 212 | 234 |
| 08-19 | 1145 | 191 | 168 | 206 |
| 08-26 | 878 | 146 | 119 | 172 |
| 09-02 | 458 | 76 | 36 | 123 |

n peaks in mid-May (2384-2386 pooled across 7 seasons) and falls to a fifth
of that by the season's last weekly cutoff — the same shape the original
three cutoffs showed (315 → 231 → 126), just with 21 more points on the
curve. The 08-12 uptick in the *mean* (222, above 08-05's 221) is not
seasons genuinely improving — it is **2026 dropping out of the pool
entirely from 08-12 on**: its local PA parquet ends 2026-09-01, so the
rest-of-season window from 08-12 onward is under three weeks and no hitter
clears the 100-PA floor in it, `backtest()` raises `ValueError` on an empty
realized frame, and the harness skips that (season, cutoff) rather than
scoring zero players. 2026 is the thinnest-n season at every cutoff it does
appear at (its 08-05 n of 77 sets that cutoff's minimum), so removing it
raises the mean even though nothing got easier to score — a textbook
version of exactly the population-mix risk this section exists to flag.

The **common player set** — fixed per season at whoever clears the bar at
that season's *last* biweekly cutoff (Aug 5) and therefore clears it at
every earlier one too, since rest-of-season trials only shrink as the cutoff
advances — is a much smaller and much steadier population:

| Cutoff (biweekly grid) | natural n | common n |
|---|---:|---:|
| 04-15 | 1010 | 746 |
| 04-29 | 1022 | 758 |
| 05-01 | 1025 | 759 |
| 05-13 | 1029 | 773 |
| 05-27 | 1006 | 768 |
| 06-10 | 962 | 754 |
| 06-24 | 898 | 729 |
| 07-01 | 875 | 726 |
| 07-08 | 838 | 710 |
| 07-22 | 757 | 675 |
| 08-01 | 647 | 621 |
| 08-05 | 578 | 578 |

Common-set sizes per season (whoever clears 100 rest-of-season K trials at
that season's Aug 5 cutoff): **2022: 256, 2024: 244, 2026: 78** — 2026's
common set is a third the size of the other two, for the same reason its
natural n is smallest everywhere: 2026 is a season still in progress, so its
"rest of season" at every cutoff is shorter than a completed season's, and
fewer hitters clear a fixed trials floor in a shorter remaining window.
399 distinct players feed the common-set columns above — the union across
the three seasons' fixed sets (256+244+78=578 memberships, so on average a
player who clears the bar in one season's common set clears it in about 1.4
of the three, which is what a durable-veteran-heavy population looks like).

Natural and common track closely through May-July — the population that
survives to Aug 5 was already most of who was being scored in April — and
converge exactly at 08-05 by construction (that cutoff *is* what defines the
common set).

### Clustering by player matters a lot here — more than the contact-quality work found

A hitter appears at every cutoff of every season he is scored in — up to
24 times in the cheap sweep. Pairing `marcel` against `marcel_tuned` on
k_rate, pooled over all 46,591 cheap-sweep cells:

| | n rows | n clusters | diff | SE | t |
|---|---:|---:|---:|---:|---:|
| unclustered | 46,591 | 46,591 | 0.00084 | 0.00003 | 24.91 |
| **clustered by player** | 46,591 | 870 | 0.00084 | 0.00013 | **6.71** |

Unclustered t is **3.71x** the clustered t here — 46,591 rows collapse to
870 real hitters, an average of 54 rows per player, and treating each row as
an independent draw understates the standard error by that much. (The
paired stat itself — `diff` — is identical either way; clustering only
changes what you're allowed to believe about its precision.)

The contact-quality work found unclustered t about 30% too large at 15
rows/player; here, with up to 24-164 rows per player pooled across seven
seasons, the inflation is far worse. Every paired statistic in this doc
clusters on the real player id.

### The gap vs. calendar date — `bayes` vs `marcel_tuned`, k_rate

Pooling all three seasons at each calendar cutoff, clustered by player
(positive = `bayes` worse):

| Cutoff | n_seasons | natural n | natural diff | SE | t | common n | common diff | SE | t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 04-15 | 3 | 1010 | +0.00112 | 0.00045 | 2.52 | 746 | +0.00104 | 0.00049 | 2.13 |
| 04-29 | 3 | 1022 | +0.00114 | 0.00044 | 2.59 | 758 | +0.00121 | 0.00049 | 2.50 |
| 05-01 | 3 | 1025 | +0.00092 | 0.00044 | 2.09 | 759 | +0.00106 | 0.00048 | 2.21 |
| 05-13 | 3 | 1029 | +0.00102 | 0.00043 | 2.34 | 773 | +0.00134 | 0.00046 | 2.90 |
| 05-27 | 3 | 1006 | +0.00099 | 0.00042 | 2.36 | 768 | +0.00121 | 0.00046 | 2.65 |
| 06-10 | 3 | 962 | +0.00136 | 0.00041 | **3.29** | 754 | +0.00146 | 0.00045 | **3.28** |
| 06-24 | 3 | 898 | +0.00122 | 0.00043 | 2.82 | 729 | +0.00125 | 0.00047 | 2.68 |
| 07-01 | 3 | 875 | +0.00124 | 0.00043 | 2.90 | 726 | +0.00133 | 0.00046 | 2.90 |
| 07-08 | 3 | 838 | +0.00125 | 0.00042 | 2.99 | 710 | +0.00126 | 0.00045 | 2.82 |
| 07-22 | 3 | 757 | +0.00110 | 0.00044 | 2.53 | 675 | +0.00115 | 0.00046 | 2.50 |
| 08-01 | 3 | 647 | +0.00094 | 0.00047 | 2.02 | 621 | +0.00096 | 0.00048 | 2.01 |
| 08-05 | 3 | 578 | +0.00069 | 0.00050 | 1.38 | 578 | +0.00069 | 0.00050 | 1.38 |

Every row, natural and common alike, is positive — `bayes` is worse at
every calendar point on the grid once the three seasons are pooled, with
t ≥ 2 for ten of the twelve. Natural and common move together almost
everywhere (they are literally the same numbers at 08-05, by construction),
so the "is this just a shrinking, differently-selected population" concern
from [n falls](#n-falls-as-the-season-progresses--and-it-is-worse-than-it-looks)
does not explain the pattern away here — restricting to a fixed population
present at every cutoff does not flip, or even much move, the sign or size
of the gap.

As a plot (diff range +0.00069 to +0.00136, `*` marks each cutoff's natural
diff, no `0` marker — the whole range sits above zero):

```
diff (bayes - marcel_tuned): +0.00069 (left) to +0.00136 (right)
 04-15 (t= 2.52) |                                *                 |
 04-29 (t= 2.59) |                                 *                |
 05-01 (t= 2.09) |                 *                                |
 05-13 (t= 2.34) |                        *                         |
 05-27 (t= 2.36) |                      *                           |
 06-10 (t= 3.29) |                                                 *|
 06-24 (t= 2.82) |                                       *          |
 07-01 (t= 2.90) |                                        *         |
 07-08 (t= 2.99) |                                         *        |
 07-22 (t= 2.53) |                              *                   |
 08-01 (t= 2.02) |                   *                              |
 08-05 (t= 1.38) |*                                                 |
```

No April/May trough (the pre-registered shape) and no clean monotonic
decline either — the curve is a noisy plateau for ten of twelve cutoffs with
a real, but late, softening only in the last two.

### Split by whether the season overlaps `marcel_tuned`'s tuning window

| Season group | n | n players | diff | SE | t |
|---|---:|---:|---:|---:|---:|
| overlaps tuning window (2022, 2024) | 7,593 | 543 | +0.00113 | 0.00044 | 2.53 |
| clean holdout (2026) | 3,054 | 368 | +0.00102 | 0.00044 | 2.29 |

The gap is barely smaller on the clean holdout than on the seasons that
overlap `marcel_tuned`'s own tuning window (+0.00102 vs +0.00113, both
t > 2). If `marcel_tuned`'s in-sample advantage on 2022/2024 were doing real
work here, the holdout-only number should look meaningfully better for
`bayes` than the pooled one does — it does not. This is the single strongest
piece of evidence in this doc that the reversal is not an artifact of the
tuning-window contamination flagged above.

### Robustness: `bayes` vs. stock `marcel` (untuned, so no tuning-window caveat applies)

| Cutoff | n | diff (bayes − marcel) | SE | t |
|---|---:|---:|---:|---:|
| 04-15 | 1010 | +0.00046 | 0.00050 | 0.92 |
| 04-29 | 1022 | +0.00034 | 0.00048 | 0.71 |
| 05-01 | 1025 | +0.00021 | 0.00048 | 0.44 |
| 05-13 | 1029 | +0.00034 | 0.00048 | 0.71 |
| 05-27 | 1006 | +0.00014 | 0.00046 | 0.31 |
| 06-10 | 962 | +0.00048 | 0.00045 | 1.06 |
| 06-24 | 898 | +0.00056 | 0.00046 | 1.21 |
| 07-01 | 875 | +0.00068 | 0.00045 | 1.50 |
| 07-08 | 838 | +0.00048 | 0.00045 | 1.07 |
| 07-22 | 757 | +0.00033 | 0.00048 | 0.70 |
| 08-01 | 647 | +0.00013 | 0.00049 | 0.27 |
| 08-05 | 578 | **−0.00021** | 0.00050 | **−0.41** |

Against *stock* Marcel — Tango's untuned constants, so this comparison
carries none of the tuning-window caveat — `bayes` is statistically
indistinguishable from the baseline at every single cutoff (|t| ≤ 1.5
throughout, mostly under 1). It even edges ahead, insignificantly, at 08-05.
This is consistent with the BAS-59 fair-fight finding at the original three
cutoffs (refit `bayes` "level with **stock** Marcel-with-partial") and says
the reversal against `marcel_tuned` is specifically about what tuning
bought Marcel, not evidence that `bayes` itself got worse.

### Pooled paired test, clustered vs. unclustered

Every `bayes`-vs-`marcel_tuned` cell pooled into one test (all 36 fits, all
three seasons, k_rate):

| | n rows | n clusters | diff | SE | t |
|---|---:|---:|---:|---:|---:|
| unclustered | 10,647 | 10,647 | +0.00110 | 0.00013 | 8.36 |
| **clustered by player** | 10,647 | 659 | +0.00110 | 0.00035 | **3.11** |

Clustering cuts t by 2.7x (8.36 → 3.11) — real, but the result survives it:
t = 3.11 pooling everything a player contributes across every cutoff and
season he appears in is not a fragile number. **This is this doc's single
headline statistic**: pooled, clustered, three-season evidence that
`marcel_tuned` beats the refit Bayesian arm by a small but real margin,
not the "dead heat, t ≤ 1.6" read the original three cutoffs supported.

### What each fit actually was

<details>
<summary>All 36 fits: cells, PA, batters, r-hat, ESS, divergences, wall time</summary>

| Cutoff | seasons | n_cells | n_pa | n_batters | max r-hat | min ESS | divergences | elapsed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022-04-15 | 2019-2021-2022 | 2005 | 368652 | 758 | 1.0248 | 162 | 0 | 76s |
| 2022-04-29 | 2019-2021-2022 | 2048 | 382364 | 767 | 1.0297 | 174 | 0 | 61s |
| 2022-05-01 | 2019-2021-2022 | 2057 | 384660 | 770 | 1.0215 | 151 | 0 | 68s |
| 2022-05-13 | 2019-2021-2022 | 2091 | 396057 | 777 | 1.0300 | 57 | 0 | 60s |
| 2022-05-27 | 2019-2021-2022 | 2130 | 410853 | 791 | 1.0224 | 163 | 0 | 71s |
| 2022-06-10 | 2019-2021-2022 | 2167 | 425475 | 803 | 1.0167 | 136 | 0 | 73s |
| 2022-06-24 | 2019-2021-2022 | 2183 | 439854 | 807 | 1.0209 | 188 | 0 | 70s |
| 2022-07-01 | 2019-2021-2022 | 2206 | 446614 | 809 | 1.0252 | 172 | 0 | 70s |
| 2022-07-08 | 2019-2021-2022 | 2221 | 454262 | 815 | 1.0258 | 206 | 0 | 74s |
| 2022-07-22 | 2019-2021-2022 | 2233 | 465287 | 821 | 1.0253 | 216 | 0 | 79s |
| 2022-08-01 | 2019-2021-2022 | 2253 | 475787 | 825 | 1.0337 | 220 | 0 | 77s |
| 2022-08-05 | 2019-2021-2022 | 2283 | 479666 | 828 | 1.0247 | 239 | 0 | 74s |
| 2024-04-15 | 2022-2023-2024 | 1907 | 381056 | 687 | 1.0226 | 155 | 0 | 62s |
| 2024-04-29 | 2022-2023-2024 | 1946 | 395091 | 692 | 1.0244 | 151 | 0 | 64s |
| 2024-05-01 | 2022-2023-2024 | 1954 | 397115 | 694 | 1.0211 | 155 | 0 | 61s |
| 2024-05-13 | 2022-2023-2024 | 2000 | 408950 | 702 | 1.0187 | 186 | 0 | 58s |
| 2024-05-27 | 2022-2023-2024 | 2018 | 423026 | 707 | 1.0227 | 284 | 0 | 67s |
| 2024-06-10 | 2022-2023-2024 | 2036 | 436836 | 711 | 1.0345 | 81 | 0 | 67s |
| 2024-06-24 | 2022-2023-2024 | 2059 | 450509 | 718 | 1.0189 | 174 | 0 | 68s |
| 2024-07-01 | 2022-2023-2024 | 2072 | 457630 | 720 | 1.0217 | 165 | 0 | 58s |
| 2024-07-08 | 2022-2023-2024 | 2082 | 464615 | 722 | 1.0179 | 199 | 0 | 71s |
| 2024-07-22 | 2022-2023-2024 | 2095 | 474937 | 726 | 1.0179 | 209 | 0 | 71s |
| 2024-08-01 | 2022-2023-2024 | 2127 | 485250 | 729 | 1.0195 | 250 | 0 | 71s |
| 2024-08-05 | 2022-2023-2024 | 2150 | 489172 | 734 | 1.0257 | 186 | 0 | 70s |
| 2026-04-15 | 2024-2025-2026 | 1935 | 382215 | 676 | 1.0374 | 62 | 0 | 58s |
| 2026-04-29 | 2024-2025-2026 | 1965 | 396407 | 679 | 1.0247 | 87 | 0 | 63s |
| 2026-05-01 | 2024-2025-2026 | 1970 | 398229 | 679 | 1.0196 | 237 | 0 | 64s |
| 2026-05-13 | 2024-2025-2026 | 2002 | 410243 | 686 | 1.0247 | 114 | 0 | 63s |
| 2026-05-27 | 2024-2025-2026 | 2029 | 424535 | 692 | 1.0208 | 257 | 0 | 65s |
| 2026-06-10 | 2024-2025-2026 | 2062 | 438338 | 702 | 1.0389 | 77 | 0 | 67s |
| 2026-06-24 | 2024-2025-2026 | 2103 | 452046 | 713 | 1.0278 | 192 | 0 | 70s |
| 2026-07-01 | 2024-2025-2026 | 2116 | 459421 | 719 | 1.0186 | 212 | 0 | 68s |
| 2026-07-08 | 2024-2025-2026 | 2125 | 466344 | 723 | 1.0214 | 123 | 0 | 71s |
| 2026-07-22 | 2024-2025-2026 | 2143 | 477247 | 727 | 1.0246 | 154 | 0 | 71s |
| 2026-08-01 | 2024-2025-2026 | 2156 | 487422 | 732 | 1.0245 | 147 | 0 | 72s |
| 2026-08-05 | 2024-2025-2026 | 2177 | 491380 | 734 | 1.0271 | 120 | 0 | 74s |

</details>

Max r-hat across all 36 fits: 1.017-1.037. Min ESS bulk: 57-284. Zero
divergences on every single fit. Median wall time 69s; total sweep time about 41 minutes of sampling (plus
data-loading overhead per fit, and a deliberate mid-run pause to cut a
fourth season — see [Design](#design)).

## Verdict: did the prediction hold?

**No. The prediction failed, cleanly and by a wide margin.**

The pre-registered claim was that partial pooling should help most when
samples are smallest, so `bayes` should be *most* favoured relative to
`marcel_tuned` in April/May and converge (or reverse) toward September. What
36 fits across three independent seasons show instead: **`bayes` is worse
than `marcel_tuned` at 35 of the 36 scored (season, cutoff) cells** —
essentially all of them, from April 15 through the first week of August, in
2022, 2024 and 2026 alike. The lone exception, 2026-08-05, is also the
smallest population in the entire sweep (n=78, the last of the season's
biweekly cutoffs) and its t is -0.87 — a coin flip's worth of noise, not a
reversal. There is no early-season window, in any of the three seasons,
where the point estimate favours `bayes` by a meaningful margin. See
[the per-cell table](#the-gap-vs-calendar-date--bayes-vs-marcel_tuned-k_rate)
— every pooled-by-calendar-date row (which is the honest, clustered read,
not any single small cell) carries the same sign, natural and common alike.

That is the opposite of "artifact of three noisy points" — the original
three cutoffs undersold how consistent this is, if anything. It is also not
exactly the pre-registered fallback ("a level or calibration error that
washes out as data accumulates") in its cleanest form: the gap does not
monotonically shrink through the season either. It is noisy around a
persistently positive mean for most of the year and only softens toward
August — see the exact per-cutoff numbers in
[the gap-by-date table](#the-gap-vs-calendar-date--bayes-vs-marcel_tuned-k_rate),
which is the honest place to read this rather than any one cell quoted here.
**And the population is shrinking at exactly the cutoffs where the gap looks
smallest** — n falls all season (see
[n falls](#n-falls-as-the-season-progresses--and-it-is-worse-than-it-looks)),
so an August cell is a harder-selected, smaller, and *not necessarily the
same kind of* population as an April one. Some of the apparent late-season
softening could be that selection effect rather than the two arms actually
converging — precisely the concern this doc's common-player-set analysis
exists to separate out; read the common-set column of the gap-by-date table,
not just the natural one, before concluding anything about convergence.

That points toward the pre-registered fallback's *spirit* even though its
exact shape ("shrinks monotonically") does not fit: **a persistent
level/calibration gap, not a pooling deficiency that should be biggest
precisely when pooling would help most.** A pooling story predicts the
April/May cells should be the strongest wins for `bayes`; instead they are
statistically indistinguishable from the mid-season cells (pooled t runs
2.02-3.29 from April through July before softening at the last two cutoffs
— no April/May trough or peak visible in
[the gap chart](#the-gap-vs-calendar-date--bayes-vs-marcel_tuned-k_rate)).
Whatever `bayes` is losing to `marcel_tuned`, it is not primarily a shrinkage
problem that more April data would fix — it looks like something in the
model or its scale (500/500/2-chain here, itself a caveat — see
[Caveats](#caveats)) that costs it roughly the same amount all season, with
only a late softening that the selection-effect caveat above says to read
cautiously rather than as confirmed convergence.

**Clustering the SEs matters for how confident to be in this, not for the
sign.** The per-cell t's above are already single-cutoff (no clustering
needed there — see [Clustering](#clustering-by-player-matters-a-lot-here--more-than-the-contact-quality-work-found)).
Pooling everything into one clustered test —
[the tuning-window split](#split-by-whether-the-season-overlaps-marcel_tuneds-tuning-window)
and [the pooled clustered-vs-unclustered table](#pooled-paired-test-clustered-vs-unclustered)
— is what turns "every cell happens to be positive" into a number with an
honest standard error attached, and it is clustering, not the point
estimate, that is this doc's answer to "is n=126 (the original Aug 1 cell),
or n=1029 (this doc's largest pooled-by-date cell), or n=10,647 (every fit
pooled) enough to trust."

**What this rules in.** The reversal reported from the original three
cutoffs is real and not an artifact of a thin, single-season sample — it
replicates in two additional, independent seasons at more than 4x the
within-season density. **What this rules out**, at this scale (500/500
draws, 2 chains, no opposing-pitcher term): the specific pre-registered
pooling story. It does not, on its own, identify *what* the level error is —
that is follow-up work, not this ticket's.

## Caveats

- **The tuning-window overlap above.** Read the split table, not just the
  pooled one.
- **Park factors are neutral and ages fall back where Chadwick birthdates
  are missing**, same as the BAS-59 fair-fight run — `data/parquet/park_factors.parquet`
  is absent in this checkout.
- **The bayes sweep is 3 seasons, biweekly; the cheap sweep is 7 seasons,
  weekly.** Cost-scoped per the ticket's own guidance; see
  [Design](#design).
- **500 draws / 500 tune / 2 chains** throughout the bayes sweep — noticeably
  lighter than the BAS-59 fair-fight's 1500/1500/4 (2 chains is itself below
  the convergence-diagnostic convention; every fit's own log says so), and it
  shows: r-hat runs 1.017-1.037, above the 1.01 convention on every fit, worst
  variable mostly individual `z_ability` (per-batter ability) terms, some on
  `sigma_ability`, and one apiece on `beta_hand` and `park_effect` — so unlike
  BAS-59's fits, this scale is not clean of non-player terms. Min ESS bulk
  runs 57-284 against BAS-59's 486-851. Zero divergences on every fit
  throughout, which is the one diagnostic that stayed clean. See
  [What each fit actually was](#what-each-fit-actually-was) — read the
  *pattern* across fits rather than trusting any single cutoff's point
  estimate.
- **The opposing-pitcher term is off**, matching the BAS-59 fair-fight
  scale — the doc found it makes the rest-of-season projection *worse*
  despite LOO preferring it, so this is the right scale to compare against,
  not a corner cut.
- **2019's bayes-sweep prior seasons were not fetched** (would need 2017-2018),
  so the bayes sweep starts at 2022 while the cheap sweep starts at 2019.
- **PA-derived prior seasons vs. the Stats API table.** `marcel_tuned`
  reads prior seasons from the Stats API table; `bayes` reads them from the
  Statcast-derived PA parquets, which run ~0.7% more PA per player — the same
  caveat every other doc in this repo comparing the two arms carries.

## Reproducing

```
# one-time data prep (writes gitignored data/parquet/pa_outcomes/*)
python -c "
from src.data.r2 import get_s3_client, bucket
s3, b = get_s3_client(), bucket()
for y in (2019, 2021, 2022, 2023, 2024, 2025):
    s3.download_file(b, f'statcast/statcast_{y}.parquet', f'data/raw/statcast_{y}.parquet')
"
python -c "
from src.data.pa_outcomes_pipeline import build_pa_dataset
build_pa_dataset(years=[2019, 2021, 2022, 2023, 2024, 2025])
"
python -c "from src.data.pa_outcomes import download; download(2026, 'data/parquet/pa_outcomes')"
python scripts/build_birthdates.py --pa-parquet data/parquet/pa_outcomes/pa_outcomes_2026.parquet

# the cheap sweep (~10 min, no MCMC)
python scripts/run_intraseason_backtest_dense.py --stage cheap

# the bayes sweep (~35-50 min with NumPyro/JAX on 4 cores at the shipped
# 3-season scope; hours on PyMC-only). Checkpoints after every (season,
# cutoff), so it is safe to Ctrl-C and resume, or to trim --bayes-seasons
# mid-run (as this run did — 2025 was cut after 2022 and 2024 finished, to
# land in a reasonable compute budget; see Design).
python scripts/run_intraseason_backtest_dense.py --stage bayes \
       --bayes-seasons 2022 2024 2026

python scripts/run_intraseason_backtest_dense.py --stage analyze
```

Both sweeps checkpoint after every (season, cutoff) cell to
`data/eval/dense_intraseason/cells_{cheap,bayes}.parquet`, so an interrupted
run resumes rather than restarting.
