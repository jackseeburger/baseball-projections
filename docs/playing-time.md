# Station B — Playing time (projected rest-of-season PA)

**Built Sept 2, 2026.** Reproduce with
`python3 scripts/build_playing_time.py --score` (the table) and
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
2. **Weight.** Plate appearances in the trailing **30 days** before the
   cutoff. A hitter with none falls back to his trailing 60 days, then to his
   season to date; each fallback is rescaled onto a 30-day time base so a
   first-half regular's 60-day count cannot outweigh a current regular's
   30-day count.
3. **Call-ups.** An eligible hitter with no plate appearances at all gets a
   league bench default — 3% of his club's 30-day total. He is not a zero;
   he is a bench bat.
4. **Normalize** within the club so the shares sum to 1.
5. **Cap at one lineup slot.** No hitter may take more than **1/8** of his
   club's plate appearances, with the excess water-filled onto the hitters
   still under the cap. Nine lineup slots, and the top of the order absorbs
   the extra plate appearances of an incomplete final turn, so a leadoff
   hitter on a 37.5-PA-per-game club gets about 4.7 of them. That is lineup
   arithmetic, not a fitted constant, and 2026 agrees: across every 30-day
   window at every cutoff, the largest share any hitter *actually* took was
   0.123–0.125, never more. The cap binds because step 1 renormalizes over
   the survivors — without it the 2026-09-02 Cardinals projected their best
   remaining hitter at a 0.171 share, 6.5 plate appearances a game. Adding
   the cap improved every metric in §2 at both cutoffs.

Output columns: `batter, team_id, cutoff_date, games_remaining, pa_share,
projected_pa_ros`, written to `data/parquet/playing_time_ros.parquet`
(gitignored).

**Walk-forward, strictly.** Every function that reads a game log takes a
cutoff and keeps only rows with `date < cutoff`. A game played *on* the
cutoff date is future information. Scoring picks up at `date >= cutoff`, so
the two windows have no gap and no overlap; there is a unit test for each
side of that boundary.

## 2. The score

Three methods, two cutoffs, projected from data strictly before the cutoff and
scored against plate appearances actually taken from the cutoff through
2026-09-02. Both cutoffs are scored on the *same* hitter universe for all
three methods, so nobody gets credit for declining to project someone.

- `uniform` — equal share across the active-roster hitters.
- `season_share` — season-to-date PA share. No window, no IL handling, no cap.
- `last_30` — the model above.

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
| 2026-07-01 | 63 days | last_30 | 616 | 46.07 | 65.28 | 53.43 | 71.44 | **.723** |
| 2026-08-01 | 32 days | uniform | 595 | 28.46 | 38.10 | 29.99 | 35.94 | .630 |
| 2026-08-01 | 32 days | season_share | 595 | 25.82 | 35.57 | **25.18** | **34.43** | .719 |
| 2026-08-01 | 32 days | **last_30** | 595 | **22.05** | **33.12** | 25.46 | 35.59 | **.766** |

**The verdict, honestly: the answer depends on the horizon, and the horizon
station B actually serves is the short one.**

- **At one month (the Aug 1 cutoff) the model wins**, by 15% on MAE (22.05 vs
  25.82), 7% on RMSE, and 4.7 points of top-9 capture. It loses the two
  realized-PA-weighted metrics by 1–3%. One month is the operative horizon:
  roadmap 1.3 exists to distribute ~26 remaining games, and today's build has
  21–24 games left per club.
- **At two months (the July 1 cutoff) the model loses** on MAE, RMSE and both
  weighted metrics, and only keeps the top-9 lead. The 30-day window is
  *sharper* about who is playing right now and *noisier* about who will be
  playing in September.
- **Both beat `uniform` comfortably at both cutoffs**, which is the floor the
  station had to clear to be worth anything at all.
- **top-9 capture is the metric the model wins everywhere** (.723 / .766 vs
  .716 / .719), and it is the one station C cares about most: the run
  environment is a PA-weighted average over the lineup, so identifying *which
  nine* bat matters more than the third digit of each one's count.

### Where the July error actually comes from

Splitting the July 1 cutoff's total absolute error by roster status
(`last_30` vs `season_share`, plate appearances):

| Group | n | realized PA | last_30 abs err | season_share abs err |
|---|---|---|---|---|
| Active | 387 | 51,932 | 19,211 | 16,469 |
| On the IL | 87 | 4,463 | 4,463 | 4,742 |
| Optioned / other | 115 | 2,919 | 2,919 | 3,720 |

The IL rule is **not** what costs the model — zeroing the injured is already
better than `season_share`'s naive nonzero projection, even at two months, and
strictly better at one month (1,267 vs 3,314). The loss is entirely among
healthy active hitters: the 30-day window is the noisy part over a long
horizon. That is the thing to fix first (see §5), not the IL handling.

### The horizon caveat — this is the noise floor, not a bug

Realized plate appearances over one to two months contain **injuries,
demotions, trades and September call-ups that no cutoff-date roster can
know**. The 87 hitters on the injured list on 2026-07-01 went on to take 4,463
plate appearances; of the 87 on the list on 2026-08-01, only 11% took more
than 50. Neither number is knowable at the cutoff. A perfect model of *today's
depth chart* would still post a large MAE here, and the gap between the
methods (43 vs 46 PA at two months) is much smaller than the level (both
~45 PA on a mean realized ~96). Read the table as a ranking of methods, not as
a claim about achievable accuracy.

Two smaller measurement artifacts, identical across all three methods:

- Games on the scoring end date (2026-09-02) were still in progress when the
  table was built, so realized totals are short by roughly a day of league
  play. Every method's projected league total runs ~2% above realized for
  this reason (62,473 vs 61,109 PA at the July cutoff).
- The scored universe is the union of the 40-man hitters at all three dates
  and everyone with a 2026 plate appearance, 616 and 595 hitters; it captures
  61,102 of the 61,109 league plate appearances in the July window, so
  essentially nothing real is left out.

## 3. Today's build (cutoff 2026-09-02)

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
game). Nobody exceeds the lineup-slot cap.

## 4. What station C needs from this

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
- **`pa_share` is the horizon-free quantity.** To re-project over a different
  number of games (a playoff series, next week), multiply `pa_share` by a new
  `team_pa_per_game x games` rather than rescaling `projected_pa_ros`.
- **Rate coverage will not be complete.** A September call-up has a bench
  default share and no rate projection; station C needs a replacement-level
  rate for those, not a dropped row, or the club's PA will not sum to the
  club's games.
- **Rebuild it before you use it.** The cutoff is baked into the file
  (`cutoff_date`) and the roster moves daily.

## 5. What would improve it, in order

1. **Blend the windows instead of choosing one.** The 30-day window wins at
   one month and loses at two; a weighted blend of 30-day and season-to-date
   share, with the weight set by the horizon, should beat both at both. This
   is the single change most likely to move the table.
2. **Stop treating the injured list as binary.** A `D10` hitter on July 1 is
   mostly back within the horizon; a `D60` is mostly not. Status-and-horizon
   specific return probabilities would recover most of the 4,463 PA the model
   currently throws away at the two-month cutoff.
3. **Depth-chart position awareness.** Shares are currently position-blind, so
   two catchers on the same roster can both be projected as regulars. Position
   from the roster feed is already in the frame and unused.
4. **A hazard model for in-horizon injuries.** Roadmap 1.3 explicitly says no
   hazard model is needed at the ~26-game horizon, and the table agrees; it
   becomes the binding constraint the moment station B is asked for a full
   season.

## 6. Endpoint notes (Stats API, no key)

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
