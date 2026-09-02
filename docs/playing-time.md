# Station B — Playing time (projected rest-of-season PA)

**Built Sept 2, 2026. Horizon blend added Sept 2, 2026.** Reproduce with
`python3 scripts/build_playing_time.py --score` (the 2026 table),
`python3 scripts/build_playing_time.py --sweep --season 2025` (the selection
curve the blend's two parameters come from) and
`python3 scripts/build_playing_time.py --cutoff 2026-09-02` (the projection).
Code: `src/projections/playing_time.py` (the math, pure functions over
DataFrames), `src/data/mlb_stats_api.py` (the fetchers),
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
   only. Status `A` is eligible. Every injured-list status (`D7`/`D10`/`D15`/
   `D60`) and every other unavailable status (`RM` optioned, `PL` paternity,
   `RL` restricted, `SU` suspended) projects to **exactly zero**.
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
(gitignored). `pa_share` now depends on `games_remaining` through `w(h)`, so
it is no longer strictly horizon-free — see §6.

**Walk-forward, strictly.** Every function that reads a game log takes a
cutoff and keeps only rows with `date < cutoff`. A game played *on* the
cutoff date is future information. Scoring picks up at `date >= cutoff`, so
the two windows have no gap and no overlap; there is a unit test for each
side of that boundary.

## 2. The score

Four methods, two cutoffs, projected from data strictly before the cutoff and
scored against plate appearances actually taken from the cutoff through
2026-09-02. Both cutoffs are scored on the *same* hitter universe for all four
methods, so nobody gets credit for declining to project someone.

- `uniform` — equal share across the active-roster hitters.
- `season_share` — season-to-date PA share. No window, no IL handling, no cap.
- `last_30` — trailing-30-day share, IL zeroed, bench default, capped. The
  model until this change; now the third baseline, because it is what the
  blend has to beat at the short horizon as `season_share` is what it has to
  beat at the long one.
- `blend` — the model: `w(h)` of `last_30` and the rest of the season share,
  through the same plumbing, capped.

MAE and RMSE are in plate appearances per hitter. Weighted variants weight by
realized PA (the `src/eval/metrics` convention: a regular's miss counts more
than a bench bat's). `top9_capture` is the share of realized club plate
appearances taken by the nine hitters each method ranked highest for that
club — "did you pick the right lineup?" separated from "did you get the
counts right?". Lower is better except `top9_capture`.

| Cutoff | Horizon | Method | n | MAE | RMSE | wMAE | wRMSE | top-9 capture |
|---|---|---|---|---|---|---|---|---|
| 2026-07-01 | 63 days | uniform | 616 | 56.64 | 73.59 | 59.58 | 71.55 | .607 |
| 2026-07-01 | 63 days | **season_share** | 616 | **43.38** | **59.22** | **43.55** | **58.76** | .716 |
| 2026-07-01 | 63 days | last_30 | 616 | 46.07 | 65.28 | 53.43 | 71.44 | .723 |
| 2026-07-01 | 63 days | blend | 616 | 45.37 | 64.00 | 52.76 | 70.06 | **.724** |
| 2026-08-01 | 32 days | uniform | 595 | 28.46 | 38.10 | 29.99 | 35.94 | .630 |
| 2026-08-01 | 32 days | season_share | 595 | 25.82 | 35.57 | **25.18** | **34.43** | .719 |
| 2026-08-01 | 32 days | last_30 | 595 | 22.05 | 33.12 | 25.46 | 35.59 | .766 |
| 2026-08-01 | 32 days | **blend** | 595 | **21.91** | **32.56** | 25.42 | 35.01 | **.768** |

The methods saw the same hitters in the same season, so the difference in MAE
is a *paired* quantity and has a standard error worth quoting — most of the
variance in either MAE is common to both and cancels. Per-hitter MAE
difference, blend minus the other method, negative meaning the blend is
better:

| Cutoff | Horizon | vs | n | mean diff | SE | t |
|---|---|---|---|---|---|---|
| 2026-07-01 | 63 days | last_30 | 616 | **−0.70** | 0.29 | −2.38 |
| 2026-07-01 | 63 days | season_share | 616 | **+2.00** | 1.81 | +1.11 |
| 2026-07-01 | 63 days | uniform | 616 | −11.27 | 1.77 | −6.37 |
| 2026-08-01 | 32 days | last_30 | 595 | **−0.15** | 0.14 | −1.07 |
| 2026-08-01 | 32 days | season_share | 595 | **−3.92** | 1.07 | −3.66 |
| 2026-08-01 | 32 days | uniform | 595 | −6.56 | 1.02 | −6.46 |

**The gate was: beat both `last_30` and `season_share` on MAE at both
horizons. Three of the four comparisons clear; the fourth does not.**

- **The blend beats `last_30` at both horizons** — by 0.15 PA at one month
  (inside one SE) and by 0.70 PA at two (2.4 SE). It also beats it on RMSE and
  both weighted metrics at both cutoffs, and on top-9 capture. Strictly
  dominating the window it replaces, at every horizon, on every metric in the
  table, is the part of the gate that is unambiguously met.
- **It beats `season_share` at one month** by 3.9 PA (3.7 SE), the horizon
  station B actually serves.
- **It does not beat `season_share` at two months**: 45.37 against 43.38, +2.0
  PA with an SE of 1.8, so the loss is real in sign and inside 1.1 SE in size.
  The blend narrows the gap `last_30` had (2.7 PA) to 2.0 but does not close
  it. **No setting of the two parameters would have closed it**, which is the
  useful finding here: sweeping a constant weight on the 2026 July cutoff
  itself — the oracle, unavailable to an honest model — bottoms out at 45.12
  at w = 0.6, still 1.7 PA behind `season_share`. The gap is not in the mixing
  weight. §4 shows where it is.

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
status-and-horizon return probabilities (§7 item 1), not a different window.

### The horizon caveat — this is the noise floor, not a bug

Realized plate appearances over one to two months contain **injuries,
demotions, trades and September call-ups that no cutoff-date roster can
know**. The 87 hitters on the injured list on 2026-07-01 went on to take 4,463
plate appearances; of the 87 on the list on 2026-08-01, only 11% took more
than 50. Neither number is knowable at the cutoff. A perfect model of *today's
depth chart* would still post a large MAE here, and the gap between the
methods (43 vs 45 PA at two months) is much smaller than the level (both
~45 PA on a mean realized ~96). Read the table as a ranking of methods, not as
a claim about achievable accuracy.

Two smaller measurement artifacts, identical across all four methods:

- Games on the scoring end date (2026-09-02) were still in progress when the
  table was built, so realized totals are short by roughly a day of league
  play. Every method's projected league total runs ~2% above realized for
  this reason (62,473 vs 61,109 PA at the July cutoff).
- The scored universe is the union of the 40-man hitters at all three dates
  and everyone with a 2026 plate appearance, 616 and 595 hitters; it captures
  61,102 of the 61,109 league plate appearances in the July window, so
  essentially nothing real is left out.

## 5. Today's build (cutoff 2026-09-02)

602 hitters on 40-man rosters across the 30 clubs:

| | count |
|---|---|
| Non-pitchers on a 40-man | 602 |
| Active, projected > 0 PA | 420 (14 per club exactly — September rosters are 28) |
| On the injured list, projected 0 | 85 |
| Optioned / other unavailable, projected 0 | 97 |
| Active but with no prior 2026 PA (bench default) | 4 |

Projected club totals run 808 (WSH, 21 games left) to 918 (LAD, 24 games left)
plate appearances; the highest individual projection is 115 PA (about 4.8 a
game). Thirty hitters across fourteen clubs sit exactly at the lineup-slot
cap, and nobody is above it: at a 21-to-24-game horizon `w(h)` is 0.84, so the
shares are close enough to the trailing-30-day shares that the same clubs bind
that bound before the blend.

## 6. What station C needs from this

Station C (team run environment, roadmap 1.5) reads
`data/parquet/playing_time_ros.parquet` and joins on `batter`:

- **Join key** is `batter` — the MLBAM id, identical to Statcast's `batter`
  and to the id in `hitter_seasons_api.parquet`, so it joins to the rate
  models and the Chadwick birthdates with no crosswalk.
- **Use `projected_pa_ros` as the weight** in the PA-weighted wOBA over each
  club's hitters. It already sums, per club, to `team_pa_per_game x
  games_remaining`, so `sum(pa) x wOBA -> runs` needs no second normalization.
- **Zeros are meaningful.** A row with `projected_pa_ros == 0` is a hitter the
  roster says will not bat (injured, optioned). Filter them out rather than
  treating a zero as a missing value.
- **`pa_share` is *nearly* horizon-free — rebuild rather than rescale.** To
  re-project over a different number of games (a playoff series, next week),
  multiplying `pa_share` by a new `team_pa_per_game x games` is still the
  right arithmetic and is close to right, but it is no longer exact: the share
  itself depends on the horizon through `w(h)` (§1 step 5). The difference is
  small — `w` moves from .84 to .80 between a one-week and a two-month
  horizon — so rescale for a rough number and rerun the build with the right
  `games_remaining` for a real one.
- **Rate coverage will not be complete.** A September call-up has a bench
  default share and no rate projection; station C needs a replacement-level
  rate for those, not a dropped row, or the club's PA will not sum to the
  club's games.
- **Rebuild it before you use it.** The cutoff is baked into the file
  (`cutoff_date`) and the roster moves daily.

## 7. What would improve it, in order

1. **Stop treating the injured list as binary.** A `D10` hitter on July 1 is
   mostly back within the horizon; a `D60` is mostly not. Status-and-horizon
   specific return probabilities would recover most of the 4,463 PA the model
   currently throws away at the two-month cutoff. §4 measures the size of the
   prize directly: the roster gate costs 3.6–4.8 PA a hitter at two months and
   is worth 2.0–2.8 at one, and closing a 4-PA-a-hitter hole is worth an order
   of magnitude more than the 0.15–0.70 the window blend bought. This is now
   the only thing on this list that would change the verdict in §2.
2. **~~Blend the windows instead of choosing one.~~** Done, and it was the
   wrong first move: it beats `last_30` at both horizons on every metric, but
   by 0.15 PA at one month and 0.70 at two, and it does not close the
   two-month gap to `season_share` because the gap was never in the window
   (§4). Left here as the record of a diagnosis that measurement corrected.
3. **Depth-chart position awareness.** Shares are currently position-blind, so
   two catchers on the same roster can both be projected as regulars. Position
   from the roster feed is already in the frame and unused.
4. **A hazard model for in-horizon injuries.** Roadmap 1.3 explicitly says no
   hazard model is needed at the ~26-game horizon, and the table agrees; it
   becomes the binding constraint the moment station B is asked for a full
   season.

## 8. Endpoint notes (Stats API, no key)

- `GET /teams/{id}/roster?rosterType=40Man&date=YYYY-MM-DD` is the right call,
  not `rosterType=active`. The 40-man variant *keeps* the unavailable players
  and labels them, so the injured list falls out of the same request with no
  need for the `/transactions` feed. The `date` parameter is honored
  historically: Aaron Judge is `A` on 2026-05-01 and `D60` on 2026-08-01.
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
