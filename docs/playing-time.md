# Station B — Playing time (projected rest-of-season PA)

**Built Sept 2, 2026. Horizon blend added Sept 2, 2026. Expected returns from
the injured list added Sept 2, 2026.** Reproduce with
`python3 scripts/build_playing_time.py --score` (the 2026 table),
`python3 scripts/build_playing_time.py --score --season 2025` (the 2025 one),
`python3 scripts/build_playing_time.py --sweep --season 2025` (the selection
curve the blend's two parameters come from),
`python3 scripts/build_playing_time.py --il-table` (the return-time
distribution) and `python3 scripts/build_playing_time.py --cutoff 2026-09-02`
(the projection). Code: `src/projections/playing_time.py` (the math, pure
functions over DataFrames), `src/projections/il_returns.py` (the return-time
distribution), `src/data/mlb_stats_api.py` (the fetchers),
`scripts/build_playing_time.py` (assembly).

Station B is the multiplier every station above it needs. Station A says how
often a hitter strikes out *per plate appearance*; station C needs runs *per
game*; the bridge is how many plate appearances he actually gets. Until now
there was no such number anywhere in the repo —
`scripts/assemble_and_compare.py` hardcodes `pa = 550` for every hitter alive.

## 1. The method

For each club, one line:

```
projected_pa(hitter) = team_pa_per_game  x  games_remaining  x  pa_share(hitter)
```

| Term | Where it comes from | How much modelling |
|---|---|---|
| `team_pa_per_game` | the club's own hitting game log, season to date before the cutoff | none — an average. Range across the 30 clubs on 2026-09-02 is 37.2 (TOR, NYY) to 39.6 (CHC) |
| `games_remaining` | `fetch_schedule`, regular-season games only | none — counting |
| `pa_share` | **the model** | all of it |

`pa_share` is the hitter's slice of his club's plate appearances:

1. **Eligibility.** The 40-man roster *as of the cutoff date*, non-pitchers
   only. Status `A` is eligible and takes a full share. An injured-list
   (`D7`/`D10`/`D15`/`D60`) or optioned (`RM`) hitter takes the share he would
   have taken healthy — weighed as of the day he was placed — times the
   fraction of the remaining horizon he is expected to be back for (§5). The
   remaining unavailable statuses (`PL` paternity, `BRV` bereavement, `RL`
   restricted, `SU` suspended) still project to **exactly zero**, as does an
   injured or optioned hitter whose spell the transaction feed cannot date.
2. **Weight, twice.** The **recent** weight is plate appearances in the
   trailing **30 days** before the cutoff; a hitter with none falls back to
   his trailing 60 days, then to his season to date, each rescaled onto a
   30-day time base so a first-half regular's 60-day count cannot outweigh a
   current regular's 30-day count. The **long** weight is simply his plate
   appearances season to date.
3. **Call-ups.** An eligible hitter with no plate appearances at all gets a
   league bench default — 3% of his club's total *in whichever window is being
   weighed*. He is not a zero; he is a bench bat.
4. **Normalize** each weight within the club so each set of shares sums to 1.
5. **Blend them by the horizon.**
   `share = w(h) x share_30 + (1 − w(h)) x share_season`, where `h` is the
   club's games remaining and `w` is a decreasing logistic:

   ```
   w(h) = 1 / (1 + exp((h − midpoint) / scale))
   ```

   The two parameters are carried as the weight at two anchor horizons —
   **w(30 games) = 0.83, w(90 games) = 0.75** — because that is the readable
   form of the answer; the midpoint and scale they imply (225 and 123 games)
   are derived and sit far outside any horizon a projection is asked for. Over
   the range that matters this is a shallow slide from about .84 at a fortnight
   to about .63 at a full season, not a switch between two windows. Both
   numbers were chosen walk-forward on **2025 only** (§3) and frozen before the
   2026 table in §2 was scored. Blending *shares* rather than raw counts is
   what makes the weight meaningful — a 30-day count and a season count are not
   commensurate — and since `w` is constant within a club the blend still sums
   to 1.
6. **Cap at one lineup slot.** No hitter may take more than **1/8** of his
   club's plate appearances, with the excess water-filled onto the hitters
   still under the cap. Nine lineup slots, and the top of the order absorbs
   the extra plate appearances of an incomplete final turn, so a leadoff
   hitter on a 37.5-PA-per-game club gets about 4.7 of them. That is lineup
   arithmetic, not a fitted constant, and 2026 agrees: across every 30-day
   window at every cutoff, the largest share any hitter *actually* took was
   0.123–0.125, never more. The cap binds because step 4 renormalizes over
   the survivors — without it the 2026-09-02 Cardinals projected their best
   remaining hitter at a 0.171 share, 6.5 plate appearances a game. Adding
   the cap improved every metric in §2 at both cutoffs.

Output columns: `batter, team_id, cutoff_date, games_remaining, pa_share,
projected_pa_ros`, written to `data/parquet/playing_time_ros.parquet`
(gitignored). `pa_share` now depends on the horizon twice over — through
`w(h)` and through the expected return fractions — so it is no longer
horizon-free at all; see §7.

**Walk-forward, strictly.** Every function that reads a game log takes a
cutoff and keeps only rows with `date < cutoff`. A game played *on* the
cutoff date is future information. Scoring picks up at `date >= cutoff`, so
the two windows have no gap and no overlap; there is a unit test for each
side of that boundary.

## 2. The score

Five methods, two cutoffs, projected from data strictly before the cutoff and
scored against plate appearances actually taken from the cutoff through
2026-09-02. Both cutoffs are scored on the *same* hitter universe for all five
methods, so nobody gets credit for declining to project someone.

- `uniform` — equal share across the active-roster hitters.
- `season_share` — season-to-date PA share. No window, no IL handling, no cap.
- `last_30` — trailing-30-day share, IL zeroed, bench default, capped. The
  model two changes ago.
- `blend` — `w(h)` of `last_30` and the rest of the season share, through the
  same plumbing, capped. The model one change ago, and the baseline the
  expected returns have to beat at both horizons.
- `blend_il` — the model: the same blend with the hard roster gate replaced by
  expected returns from the injured list and the minors (§5).

MAE and RMSE are in plate appearances per hitter. Weighted variants weight by
realized PA (the `src/eval/metrics` convention: a regular's miss counts more
than a bench bat's). `top9_capture` is the share of realized club plate
appearances taken by the nine hitters each method ranked highest for that
club — "did you pick the right lineup?" separated from "did you get the
counts right?". Lower is better except `top9_capture`.

| Cutoff | Horizon | Method | n | MAE | RMSE | wMAE | wRMSE | top-9 capture |
|---|---|---|---|---|---|---|---|---|
| 2026-07-01 | 63 days | uniform | 644 | 54.34 | 72.11 | 59.86 | 71.80 | .592 |
| 2026-07-01 | 63 days | season_share | 644 | 41.52 | 57.99 | 43.63 | 58.96 | .716 |
| 2026-07-01 | 63 days | last_30 | 644 | 44.07 | 63.88 | 53.35 | 71.46 | .723 |
| 2026-07-01 | 63 days | blend | 644 | 43.40 | 62.63 | 52.67 | 70.08 | .724 |
| 2026-07-01 | 63 days | **blend_il** | 644 | **36.98** | **52.12** | **38.79** | **55.42** | **.734** |
| 2026-08-01 | 32 days | uniform | 624 | 27.31 | 37.35 | 30.32 | 36.24 | .648 |
| 2026-08-01 | 32 days | season_share | 624 | 24.70 | 34.84 | 25.36 | 34.71 | .719 |
| 2026-08-01 | 32 days | last_30 | 624 | 21.02 | 32.39 | 25.38 | 35.64 | .765 |
| 2026-08-01 | 32 days | blend | 624 | 20.88 | 31.85 | 25.34 | 35.06 | **.767** |
| 2026-08-01 | 32 days | **blend_il** | 624 | **20.32** | **29.32** | **21.92** | **31.47** | **.767** |

The methods saw the same hitters in the same season, so the difference in MAE
is a *paired* quantity and has a standard error worth quoting — most of the
variance in either MAE is common to both and cancels. Per-hitter MAE
difference, the method minus the one it is compared against, negative meaning
the method is better:

| Cutoff | Horizon | Method | vs | n | mean diff | SE | t |
|---|---|---|---|---|---|---|---|
| 2026-07-01 | 63 days | blend | last_30 | 644 | −0.67 | 0.28 | −2.39 |
| 2026-07-01 | 63 days | blend | season_share | 644 | +1.88 | 1.73 | +1.09 |
| 2026-07-01 | 63 days | blend | uniform | 644 | −10.94 | 1.70 | −6.45 |
| 2026-07-01 | 63 days | **blend_il** | blend | 644 | **−6.42** | 1.19 | **−5.41** |
| 2026-07-01 | 63 days | **blend_il** | last_30 | 644 | **−7.09** | 1.20 | **−5.89** |
| 2026-07-01 | 63 days | **blend_il** | season_share | 644 | **−4.53** | 1.21 | **−3.73** |
| 2026-08-01 | 32 days | blend | last_30 | 624 | −0.14 | 0.13 | −1.07 |
| 2026-08-01 | 32 days | blend | season_share | 624 | −3.82 | 1.02 | −3.73 |
| 2026-08-01 | 32 days | blend | uniform | 624 | −6.42 | 0.97 | −6.62 |
| 2026-08-01 | 32 days | **blend_il** | blend | 624 | **−0.56** | 0.47 | **−1.18** |
| 2026-08-01 | 32 days | **blend_il** | last_30 | 624 | **−0.70** | 0.49 | **−1.42** |
| 2026-08-01 | 32 days | **blend_il** | season_share | 624 | **−4.37** | 0.78 | **−5.58** |

**The gate was: beat `blend` and `season_share` on MAE at both horizons. All
four comparisons clear, and the two-month loss to `season_share` that survived
the last change is gone.**

- **Against `blend`**: −6.42 PA at two months (5.4 SE) and −0.56 at one
  (1.2 SE). The two-month number is an order of magnitude larger than
  anything the window blend moved, which is what §4 predicted it would be.
- **Against `season_share`**: −4.53 PA at two months (3.7 SE) and −4.37 at one
  (5.6 SE). This is the comparison that had never been won at the long
  horizon: `blend` was +1.88 behind and `last_30` +2.55.
- **It also wins RMSE and both realized-PA-weighted metrics at both cutoffs**,
  which no previous version of station B did — `last_30` and `blend` both lost
  the weighted metrics at one month, because zeroing a regular who comes back
  is a large error on a hitter with a large weight. Weighted MAE at the August
  cutoff falls from 25.34 to 21.92, a 13% cut.
- **top-9 capture improves at two months** (.724 → .734) and is unchanged at
  one, so the gain is in the counts, not in a different nine.

**One caveat on the level of these numbers.** The scored universe is the union
of everyone who took a plate appearance and everyone any method projects above
zero, so adding a method that projects the injured *enlarges* it: 616 → 644
hitters at the July cutoff and 595 → 624 at the August one. That is the
conservative direction — the added hitters are ones where `blend_il` can be
wrong and the others are automatically right — but it does move every method's
MAE, which is why the four rows shared with the previous version of this table
read a little lower than they did (`season_share` 41.52 here against 43.38
before). Scored on the old four-method universe instead, the previously
published numbers reproduce (`season_share` 43.40, `blend` 45.37 at July;
25.90 and 21.90 at August) and `blend_il` wins by *more*, 37.92 and 21.04.
Either universe clears the gate; the shared one is the one the harness
prints.

## 3. Choosing `w(h)` on 2025

The two parameters were chosen on 2025 and nothing else, and frozen before the
2026 table above was scored (`--sweep --season 2025`).

Two cutoffs cannot identify a function of the horizon, so the selection grid is
seven cutoffs from 2025-06-15 to 2025-09-15, all scored on realized PA through
2025-09-30 — horizons from 12 to 92 games remaining, which brackets the 2026
pair (30 and 54.5) rather than leaving them off the end of an extrapolation.
At each cutoff the horizon is fixed, so sweeping a *constant* weight traces
`MAE(w)` and its argmin is the best that cutoff could have done:

| Cutoff | Games remaining | Horizon | best `w` | MAE at best `w` | MAE at `w`=1 (`last_30`) | MAE at `w`=0 (season) |
|---|---|---|---|---|---|---|
| 2025-09-15 | 12.0 | 15 days | 0.90 | 7.60 | 7.66 | 9.61 |
| 2025-09-01 | 25.0 | 29 days | 0.80 | 17.41 | 17.65 | 20.76 |
| 2025-08-15 | 41.0 | 46 days | 0.75 | 32.32 | 32.76 | 35.62 |
| 2025-08-01 | 53.0 | 60 days | 0.75 | 44.84 | 45.33 | 48.31 |
| 2025-07-15 | 66.0 | 77 days | 0.75 | 56.94 | 57.40 | 61.43 |
| 2025-07-01 | 78.0 | 91 days | 0.75 | 65.31 | 65.76 | 71.02 |
| 2025-06-15 | 92.5 | 107 days | 0.85 | 79.53 | 79.74 | 84.36 |

**Read that column honestly: 2025 wants a weight near 0.8 at every horizon
from two weeks to three and a half months, and the curve is nearly flat near
its minimum.** The horizon dependence the whole exercise is about is real in
sign — 0.90 at 12 games, 0.75 in the middle — but small, and not even monotone
(the longest cutoff wants 0.85). The blend's entire margin over `last_30` is
the 0.05–0.45 PA gap between the last two columns.

The fit reflects that. Parameters are searched as `w` at 30 and at 90 games
with `w(30) > w(90)`, which is bounded and identified, rather than as a
midpoint and a scale, which are not: the surface is so flat that a midpoint
search runs off to 300 games and beyond. The objective is each cutoff's MAE
divided by the best that cutoff could have done with a constant weight, because
raw MAE scales with the horizon (80 PA at three months, 8 at two weeks) and a
raw-MAE fit is a fit to the longest cutoff alone — which is precisely the
horizon dependence being estimated. The winner, **w(30) = 0.83, w(90) = 0.75**,
gives up 0.03% against the per-cutoff oracle.

For what it is worth as a *post-hoc* check (this played no part in the
selection): the same sweep run on the two 2026 cutoffs wants 0.80 at 30 games
and 0.60 at 54.5 — the same neighbourhood at the short horizon, a steeper
decline at the long one. The frozen 2025 parameters give 0.83 and 0.80 there,
so on 2026 the model is *under*-shrunk at two months, and closing that gap
would have bought 0.26 PA of the 2.0 it is behind `season_share`.

## 4. Where the July error actually comes from

The previous version of this document read the two-month loss as the 30-day
window being noisy over a long horizon, and put "blend the windows" at the top
of the improvement list. **That diagnosis was wrong, and the blend built on it
is worth only what §2 says it is worth.** The earlier split by roster status
compared `last_30` against `season_share`, which differ in three things at
once — the window, the roster gate (IL and optioned zeroed, then the survivors
renormalized), and the lineup-slot cap — and attributed the whole active-hitter
gap to the window. Renormalizing over the survivors *is* the roster gate seen
from the other side: it is exactly what inflates the actives' projections.

Varying the two one at a time, at the same two cutoffs, on a common universe
(MAE, plate appearances per hitter; "gated" means IL and optioned zeroed,
renormalized, and capped):

| Cutoff | Horizon | 30-day, ungated | 30-day, gated | season, ungated | season, gated |
|---|---|---|---|---|---|
| 2026-07-01 | 54.5 games | **39.23** | 44.07 | 42.86 | 46.47 |
| 2026-08-01 | 30.0 games | 23.06 | **21.03** | 25.82 | 22.99 |

Paired per-hitter differences, same population:

| Cutoff | Comparison | mean diff | SE | t |
|---|---|---|---|---|
| 2026-07-01 | 30-day − season, ungated | −3.63 | 1.26 | −2.88 |
| 2026-07-01 | 30-day − season, gated | −2.40 | 1.17 | −2.06 |
| 2026-07-01 | gated − ungated, 30-day window | **+4.84** | 1.42 | +3.40 |
| 2026-07-01 | gated − ungated, season window | **+3.60** | 1.77 | +2.03 |
| 2026-08-01 | 30-day − season, ungated | −2.76 | 0.71 | −3.87 |
| 2026-08-01 | 30-day − season, gated | −1.96 | 0.66 | −2.98 |
| 2026-08-01 | gated − ungated, 30-day window | −2.03 | 0.79 | −2.58 |
| 2026-08-01 | gated − ungated, season window | −2.83 | 1.00 | −2.82 |

**The 30-day window beats the season window at both horizons and in both
gating regimes.** It is not the noisy half at two months; it never loses. What
flips with the horizon is the roster gate: zeroing the injured and optioned is
worth 2.0–2.8 PA a hitter at one month and *costs* 3.6–4.8 at two. The 2025
selection curve says the same thing from the other direction — the best blend
weight there never drops below 0.75, because the season window has nothing to
offer the model that the recent window does not.

2025 corroborates it across seven horizons rather than two. `season_share`
overtakes `last_30` there at almost exactly the same place — somewhere between
53 and 66 games remaining — even though the 30-day window is the better window
at every one of them:

| Games remaining | `uniform` | `season_share` | `last_30` |
|---|---|---|---|
| 12.0 | 10.73 | 11.14 | **7.66** |
| 25.0 | 24.16 | 22.13 | **17.65** |
| 41.0 | 39.93 | 35.45 | **32.76** |
| 53.0 | 53.74 | 46.05 | **45.33** |
| 66.0 | 68.50 | **53.26** | 57.40 |
| 78.0 | 79.53 | **60.76** | 65.75 |
| 92.5 | 94.25 | **70.71** | 79.74 |

That is a fact about a horizon, not about a window: over 63 days most of the
July 1 injured list comes back and plays, and a projection that says zero for
all of them is wrong by everything they go on to take. Over 32 days most of
them do not. `season_share`'s two-month win is bought entirely by declining to
zero anybody — a "model" that is right for the wrong reason, since it also
projects the man with the torn ACL — and the way to take that win off it is
status-and-horizon return probabilities, not a different window. §5 does
exactly that, and the two-month win comes off.

### The horizon caveat — this is the noise floor, not a bug

Realized plate appearances over one to two months contain **injuries,
demotions, trades and September call-ups that no cutoff-date roster can
know**. The 87 hitters on the injured list on 2026-07-01 went on to take 4,463
plate appearances; of the 87 on the list on 2026-08-01, only 11% took more
than 50. Neither number is knowable at the cutoff — though §5 is about
knowing the *distribution* of it, which is a different and answerable
question. A perfect model of *today's depth chart* would still post a large
MAE here, and the gap between the methods (37 vs 43 PA at two months) is much
smaller than the level (all of them ~40 PA on a mean realized ~96). Read the
table as a ranking of methods, not as a claim about achievable accuracy.

Two smaller measurement artifacts, identical across all five methods:

- Games on the scoring end date (2026-09-02) were still in progress when the
  table was built, so realized totals are short by roughly a day of league
  play. Every method's projected league total runs ~2% above realized for
  this reason (62,473 vs 61,109 PA at the July cutoff).
- The scored universe is the union of the 40-man hitters at all three dates
  and everyone with a 2026 plate appearance, 644 and 624 hitters; it captures
  essentially every league plate appearance in the scored windows.

## 5. Expected returns instead of a hard zero

§4 says the roster gate is the binding constraint: zeroing the injured and
optioned is worth 2.0–2.8 plate appearances a hitter over one month and costs
3.6–4.8 over two. Both statements are the same statement — over a long enough
horizon most of the injured come back — so the fix is not to gate or not to
gate but to say *how much of the horizon* each man is expected to be back for.

### 5.1 The feed

The 40-man roster endpoint says a hitter is `D60` today. It cannot say when he
went on the list or which list he went on first, and both are what a return
time has to be conditioned on. `GET /api/v1/transactions` can: it carries
every placement, activation, option and recall with a date.
`fetch_transactions(season)` pulls it month by month into the same
`data/cache/statsapi/` cache as everything else — 48 files and 27 MB for
2023–2026, 13,000–17,000 transactions a season.

The injured list is the awkward part of that feed. Every list move is filed
under the single type code `SC` ("status change") and distinguished only in
English, so `il_returns.classify` reads the three sentences the API writes:

| Sentence | Event |
|---|---|
| "…placed CF X on the 10-day injured list." | placement, `IL10` |
| "…activated CF X from the 10-day injured list." | activation |
| "…transferred RHP Y from the 15-day injured list to the 60-day injured list." | transfer to `IL60` |
| "…activated CF X." (no list named — about a third of activations) | activation |

Options and recalls are typed properly (`OPT`, `CU`, `SE`), and a release,
free agency or outright ends a spell without a return. A rehab assignment
(`ASG`) does **not** end an injured-list stint and is ignored.

### 5.2 The spells and the curve

`build_spells` walks each player's events in date order: an injured-list spell
runs placement → activation, an option spell runs option → recall, and a spell
still open on the last day of the regular season is **censored**, not a
non-return. A stint transferred from the 15- to the 60-day list becomes two
rows — the 15-day part censored at the transfer, the 60-day part dated from
the *original* placement but entering the risk set only on the day of the
transfer (left truncation), so a player is only ever counted against the list
he is actually on, and the elapsed time a projection reads off the roster
matches the elapsed time the curve is indexed by.

`survival_table` is then Kaplan-Meier per type. Censoring is the reason to
bother: at any cutoff the long stints are the ones most likely to be
unfinished, and averaging only the completed ones would say every list is
shorter than it is.

Fitted on **2023–2025** — the three seasons before the one being projected,
and never the season itself (`--il-table`):

| Type | spells | returns | censored | median days |
|---|---|---|---|---|
| IL7 | 53 | 48 | 5 | 9 |
| IL10 | 929 | 719 | 210 | 21 |
| IL15 | 1,235 | 779 | 456 | 23 |
| IL60 | 807 | 657 | 150 | 110 |
| OPT | 4,554 | 3,948 | 606 | 24 |

`S(d)`, the chance a spell is still running after `d` days:

| Type | S(7) | S(10) | S(14) | S(21) | S(30) | S(45) | S(60) | S(90) |
|---|---|---|---|---|---|---|---|---|
| IL7 | .679 | .434 | .358 | .208 | .132 | .132 | .132 | .106 |
| IL10 | .949 | .812 | .698 | .555 | .408 | .236 | .113 | .058 |
| IL15 | .978 | .957 | .849 | .671 | .528 | .340 | .198 | .129 |
| IL60 | .992 | .990 | .982 | .972 | .962 | .944 | .882 | .679 |
| OPT | .858 | .808 | .743 | .586 | .456 | .319 | .242 | .153 |

Read it as a sanity check rather than a finding: the 10-day list returns half
its cases inside three weeks, the 60-day list still holds 96% of its cases at
a month and 68% at three, an option runs a median 24 days, and the 7-day list
(concussions) is the fastest of the lot. Nothing here is fitted in any other
sense — it is five empirical curves, printable in a dozen rows.

### 5.3 The expected share

The projection reads the curve *conditionally*, because the roster already
told it he is still out today:

```
P(back by day e + d | still out at day e) = 1 − S(e + d) / S(e)
```

and averages that over the days left in the horizon, which gives the fraction
of the rest of the season he is expected to be available for. That fraction
multiplies the share he would have taken healthy — his ordinary blended
share, computed **as of the day he was placed**, since after it his trailing
30-day window is empty by construction and weighing him at the cutoff would
turn a regular into a bench bat. Both halves of the blend are weighted that
way; the per-club normalization and the lineup-slot cap are untouched, and
setting every fraction to zero reproduces `blend` row for row.

What that comes to in 2026:

| Cutoff | Horizon | Type | n | median days elapsed | mean expected fraction |
|---|---|---|---|---|---|
| 2026-07-01 | 63 days | IL10 | 52 | 19.5 | .603 |
| 2026-07-01 | 63 days | IL60 | 31 | 57.0 | .197 |
| 2026-07-01 | 63 days | OPT | 114 | 34.0 | .407 |
| 2026-08-01 | 32 days | IL10 | 48 | 19.5 | .387 |
| 2026-08-01 | 32 days | IL60 | 34 | 81.5 | .122 |
| 2026-08-01 | 32 days | OPT | 113 | 52.0 | .244 |

198 of the 202 unavailable hitters on the July rosters are dated this way and
195 of 202 in August; the handful left over (a `D10` whose placement predates
the feed we pulled, a `PL` or `RL` with no curve) keep the old zero. The 63-day
column is the whole argument in one line: a hitter on the 10-day list on July 1
is expected back for 60% of the rest of the season, and calling that zero was
never going to be right.

### 5.4 2025, across seven horizons

The 2026 table is two cutoffs. The 2025 grid is seven, with the distribution
refitted on **2022–2024** so the walk-forward holds there too
(`--score --season 2025`). MAE in plate appearances per hitter:

| Cutoff | Games left | `season_share` | `last_30` | `blend` | `blend_il` |
|---|---|---|---|---|---|
| 2025-06-15 | 92.5 | 68.16 | 76.86 | 76.71 | **63.19** |
| 2025-07-01 | 78.0 | 58.60 | 63.42 | 62.99 | **53.60** |
| 2025-07-15 | 66.0 | 51.34 | 55.33 | 54.89 | **47.82** |
| 2025-08-01 | 53.0 | 44.56 | 43.86 | 43.40 | **39.31** |
| 2025-08-15 | 41.0 | 34.35 | 31.74 | 31.32 | **29.14** |
| 2025-09-01 | 25.0 | 21.46 | 17.12 | 16.89 | **16.19** |
| 2025-09-15 | 12.0 | 10.82 | 7.44 | **7.38** | 7.41 |

Paired per-hitter, `blend_il` minus `blend`: −13.51 (t −6.9), −9.39 (−6.3),
−7.08 (−5.9), −4.09 (−4.1), −2.18 (−3.2), −0.70 (−2.2) and +0.03 (+0.2) at the
seven cutoffs in that order. Against `season_share` it is −4.96, −5.01, −3.52,
−5.25, −5.21, −5.28 and −3.41, all at 2.6 SE or more — **it beats the
season-share baseline at every horizon from a fortnight to three and a half
months**, which neither `last_30` nor `blend` managed past about 55 games
remaining. It also wins RMSE and both weighted metrics at all seven.

The one cell it does not win is the shortest horizon, 12 games left, where it
ties `blend` (7.41 against 7.38, t +0.25). That is the expected shape: with a
fortnight to play almost nobody on any list comes back in time, the expected
fractions are near zero, and the method converges to the hard gate it
replaced. It costs nothing there and pays everywhere else.

### 5.5 The flag

`playing_time.USE_IL_RETURNS` picks which of the two the production build
runs, and the gate decided it: **it is `True`**, so
`scripts/build_playing_time.py --cutoff …` runs `blend_il`. `blend` is still
in `METHODS` and still scored at every cutoff, because it is now the baseline
the next change has to beat. Flipping the flag to `False` reverts production
to the hard roster gate without removing anything.

## 6. Today's build (cutoff 2026-09-02)

602 hitters on 40-man rosters across the 30 clubs, projecting 26,190 plate
appearances over the 21–24 games each club has left:

| | count | projected PA |
|---|---|---|
| Active | 420 (14 per club exactly — September rosters are 28) | 24,710 |
| On the injured list | 85, of which 83 project above zero | 983 |
| Optioned / other unavailable | 97, of which 95 project above zero | 497 |
| Active but with no prior 2026 PA (bench default) | 4 | — |

180 of the 182 unavailable hitters are dated from the transaction feed; the
two that are not, and the pair whose curve says no chance, are the only zeros
left. At a 25-day horizon the mean expected fraction is .33 for the 10-day
list, .08 for the 60-day and .22 for an option, so the injured and optioned
take 5.7% of the league's remaining plate appearances between them where they
used to take none.

Projected club totals run 808 (WSH, 21 games left) to 918 (LAD, 24 games left)
plate appearances; the highest individual projection is 114 PA (about 4.7 a
game). Five hitters across three clubs sit exactly at the lineup-slot cap and
nobody is above it — down from thirty before the expected returns, because the
share that used to be water-filled onto a club's healthy regulars now stays
with the men it belongs to.

## 7. What station C needs from this

Station C (team run environment, roadmap 1.5) reads
`data/parquet/playing_time_ros.parquet` and joins on `batter`:

- **Join key** is `batter` — the MLBAM id, identical to Statcast's `batter`
  and to the id in `hitter_seasons_api.parquet`, so it joins to the rate
  models and the Chadwick birthdates with no crosswalk.
- **Use `projected_pa_ros` as the weight** in the PA-weighted wOBA over each
  club's hitters. It already sums, per club, to `team_pa_per_game x
  games_remaining`, so `sum(pa) x wOBA -> runs` needs no second normalization.
- **Zeros are meaningful, and there are far fewer of them.** A row with
  `projected_pa_ros == 0` is now a hitter the model has no way to project at
  all: unavailable with a status that carries no return curve, or a spell the
  transaction feed cannot date. An injured or optioned hitter with a curve
  gets a small positive number instead of a zero, and it is a real projection
  — the share he would take healthy, discounted by how much of the horizon he
  is expected to miss. Filter zeros out rather than treating them as missing
  values; do **not** filter the small positive numbers out.
- **`pa_share` is no longer horizon-free — rebuild, do not rescale.** To
  re-project over a different number of games (a playoff series, next week),
  multiplying `pa_share` by a new `team_pa_per_game x games` is the right
  arithmetic but the wrong number: the share depends on the horizon through
  `w(h)` (§1 step 5) and much more strongly through the expected return
  fractions (§5.3), which fall towards zero as the horizon shortens. Rescale
  only for a rough number, and rerun the build with the right
  `games_remaining` for a real one.
- **Rate coverage will not be complete.** A September call-up has a bench
  default share and no rate projection; station C needs a replacement-level
  rate for those, not a dropped row, or the club's PA will not sum to the
  club's games.
- **Rebuild it before you use it.** The cutoff is baked into the file
  (`cutoff_date`) and the roster moves daily.

## 8. What would improve it, in order

1. **~~Stop treating the injured list as binary.~~** Done (§5), and it was
   worth what §4 said it would be: −6.4 PA a hitter against `blend` at two
   months and −4.5 against `season_share`, an order of magnitude more than the
   window blend bought, and the first version of station B to beat every
   baseline on every metric at both cutoffs.
2. **Injury severity, not just list type.** The return curve is conditioned on
   the list and the days elapsed and nothing else, so a hamstring strain and a
   torn UCL on the same 10-day list get the same number. The transaction
   description carries the injury in English ("Abdominal strain.") and the
   feed is already pulled; a dozen coarse buckets fitted the same way is the
   obvious next cut, and §5's table is where it would show.
3. **Rehab assignments as a signal.** A hitter sent on a rehab assignment
   (`ASG`) is days from activation, and the feed dates that too. Today it is
   ignored on purpose — it does not end a stint — but conditioning on "on the
   60-day list, 70 days elapsed, *and on a rehab assignment*" is most of the
   difference between .08 and something much larger.
4. **~~Blend the windows instead of choosing one.~~** Done, and it was the
   wrong first move: it beats `last_30` at both horizons on every metric, but
   by 0.15 PA at one month and 0.70 at two, and it did not close the two-month
   gap to `season_share` because the gap was never in the window (§4). Left
   here as the record of a diagnosis that measurement corrected.
5. **Depth-chart position awareness.** Shares are currently position-blind, so
   two catchers on the same roster can both be projected as regulars. Position
   from the roster feed is already in the frame and unused.
6. **A hazard model for in-horizon injuries.** The return-time distribution
   handles the men who are *already* out; nothing yet handles the healthy
   regular who gets hurt in week three. Roadmap 1.3 says no hazard model is
   needed at the ~26-game horizon and the table still agrees; it becomes the
   binding constraint the moment station B is asked for a full season.

## 9. Endpoint notes (Stats API, no key)

- `GET /teams/{id}/roster?rosterType=40Man&date=YYYY-MM-DD` is the right call,
  not `rosterType=active`. The 40-man variant *keeps* the unavailable players
  and labels them, so *who* is on the injured list falls out of the same
  request. The `date` parameter is honored historically: Aaron Judge is `A` on
  2026-05-01 and `D60` on 2026-08-01. What it cannot say is *since when*,
  which is why §5 needs the transactions feed as well.
- `GET /transactions?startDate=&endDate=&sportId=1` returns the whole league's
  moves for a date range in one response — a full season is about 5 MB and
  13,000–17,000 rows, pulled a month at a time so a running season only
  invalidates its current month. Every injured-list move is type code `SC`
  and has to be read out of the English `description` (§5.1); options and
  recalls are properly typed. About a third of activations name no list, so a
  parser that only matches "activated … from the N-day injured list" will
  think a third of all stints never ended.
- That endpoint **returns some players twice on the same date** — an option
  and a recall that both landed that day give an `A` row and an `RM` row
  (2–6 players a day on the dates used here). Left alone the duplicate is
  counted twice in the club's PA share. `fetch_team_roster` dedupes to the
  most-available status.
- `GET /teams/{id}/stats?stats=gameLog&group=hitting` carries **no `gameType`
  field** (the player game logs do), but `season=YYYY` already restricts it to
  the regular season. Player logs are filtered to `gameType == "R"`.
- Season logs are cached per player-season under `data/cache/statsapi/`
  (gitignored). The season is still running, so a cached log is complete only
  to the day it was pulled — pass `--refresh` to re-pull. Rosters are cached
  only for dates strictly in the past, since today's roster is still moving.
- The 2025 selection run (§3) pulls the same three feeds for `season=2025`:
  eight roster dates x 30 clubs, 30 team hitting logs, and hitter game logs
  for a 699-man universe. Every one of those dates is in the past, so the
  whole thing caches and `--sweep --season 2025` re-runs offline in seconds
  after the first pull. 2025 finished, so unlike 2026 those caches never go
  stale.
