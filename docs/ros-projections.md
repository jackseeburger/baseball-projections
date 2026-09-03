# Rest-of-season projections — the number the site serves

**Station A, live.** What a visitor sees on a player page is now
*tuned Marcel fed the partial current season* × station B's projected
rest-of-season plate appearances. The preseason Bayesian components — the
only number this site used to show — are still there, labelled, one column
to the right.

Shipped Sept 2, 2026 · engine switched from stock to tuned Marcel the same
day · **pitchers added Sept 3, 2026** ([below](#pitchers-sept-3-2026)) ·
built by `scripts/build_ros_projections.py` · math in
`src/projections/ros.py` and `src/projections/pitcher_ros.py` · constants in
`src/eval/marcel_params.json` and `src/eval/marcel_pitcher_params.json` ·
data at `public/data/projections/{latest,YYYY-MM-DD}.json`.

## Why the model changed

The [gate rule](architecture.md#3-the-gate-rule) says a station runs the
model that beats its baseline in the harness, and until Sept 2 nothing had
asked the harness the *product's* question. The season-level backtest asks
"given everything through 2025, how good is 2026?" The player page asks
"given everything through this morning, how good is the rest of 2026?" —
and the intra-season walk-forward
([backtest-baselines.md](backtest-baselines.md#intra-season-walk-forward--rest-of-2026-rates))
answered it:

(The `marcel` column here is *stock* Marcel — the arm that won this
comparison and shipped on Sept 2. The engine was tuned later the same day;
the next section has the current numbers.)

**`bayes_preseason` here is a fixed preseason file, not a refit.** It had
never seen a 2026 plate appearance; `marcel` had seen every one before the
cutoff. The `marcel` − `marcel_preseason` column below prices that difference
at 4.6–6.1% of K% MAE, which is the same order as the gap in this table, so
the numbers here are not on their own a verdict on the Bayesian model. The
model-against-model comparison — the same estimator refit at each cutoff on
the same plate appearances — is
[the fair fight](backtest-baselines.md#the-fair-fight--the-bayesian-arm-refit-at-the-cutoff-bas-59),
run Sept 3.

| Component | Cutoff | `marcel` (stock) | `bayes_preseason` (was live) | Gap |
|---|---|---|---|---|
| k_rate | May 1 | **.0278** | .0296 | −6.3% |
| k_rate | Jul 1 | **.0296** | .0325 | −8.9% |
| k_rate | Aug 1 | **.0343** | .0386 | −11.0% |
| bb_rate | Aug 1 | **.0268** | .0279 | −4.1% |
| hr_rate | Aug 1 | **.0152** | .0153 | −1.0% |
| iso | Aug 1 | .0567 | **.0551** | +2.9% |

Marcel with the current season folded in won 11 of the 12 component-cutoff
cells the accuracy page shows, and beat *Marcel with 2026 withheld* on the
same 11 — so the gain is in-season information, not a better prior. The
comparison that does hold the information fixed is `bayes_preseason` against
`marcel_preseason`, and our components never beat it on K% there either (a tie
at May 1, 5% behind at Jul 1 and Aug 1): on the same information they bought
nothing, and they were giving up the information advantage on top of it.
Shipping them as the live number was the one thing the gate rule forbids.

BABIP is the exception in both directions: in-season data adds nothing to
it, and league average roughly ties Marcel on it. It is projected because
the wOBA reconstruction needs it, not because it is skill.

## Why the engine changed again — tuned Marcel

Stock Marcel's constants are Tango's defaults: 5/4/3 recency, one 200-trial
ballast for every component, one age curve. `marcel_tuned`
([backtest-baselines.md](backtest-baselines.md#tuning-marcel--fitted-constants-beat-tangos-defaults))
is the same estimator with those made per-component parameters, fitted
walk-forward on **2020–2024 only** and frozen in `src/eval/marcel_params.json`.
Scored on the holdout — season-level 2025 and 2026 plus the three 2026
cutoffs — it wins 17 of 25 component × cell cells at a pooled **−1.10% ±
0.30** of stock Marcel's MAE with no component worse than stock, so the gate
rule swaps it in. (Those are the Sept 3
[constrained refit](backtest-baselines.md#the-age-curve-was-not-aging--a-constrained-refit-and-a-projected-league-rate);
the first fit was 15/25 at −1.10% ± 0.36 and left ISO at +0.02%.)

Tuned minus stock on the three cutoffs, scored on the page's own arm set
(`scripts/run_intraseason_backtest.py`, so these are the numbers the accuracy
page renders):

| Component | May 1 | Jul 1 | Aug 1 |
|---|---|---|---|
| **k_rate** tuned | **.02691** | **.02928** | **.03412** |
| k_rate stock | .02776 | .02959 | .03431 |
| | −3.0% | −1.0% | −0.6% |
| **iso** tuned | .03510 | **.04109** | **.05569** |
| iso stock | **.03448** | .04127 | .05668 |
| | +1.8% | −0.4% | −1.8% |
| **hr_rate** tuned | .01005 | **.01140** | **.01476** |
| hr_rate stock | **.00983** | .01149 | .01517 |
| | +2.2% | −0.8% | −2.7% |
| **bb_rate** *(stock)* | .01720 | .02057 | .02679 |

BABIP is not on that table (its 100-trial floor leaves four players at Aug 1);
across the full holdout it is the biggest single gain, −3.0%.

K% wants half stock's ballast (100 PA) and sharper recency (1 / 0.4 / 0.4
against 5/4/3); BABIP wants triple it (600 BIP) and flat recency — the
ballast moving toward each component's stabilization point in both
directions at once. **BB% keeps Tango's constants exactly**: an inner
validation inside the tuning window said a BB% fit would not travel, and the
holdout agreed, so the frozen file ships stock for it and the live BB%
column is byte-identical to what it was. HR/PA and ISO used to come out even
and now go the right way, which is what the constrained age curve bought.

The four components the accuracy page scores exclude BABIP (its 100-trial
floor leaves four players at Aug 1), so on *that* table the tuned arm beats
stock on 7 of 12 cells, ties 3 (BB% is stock) and loses 2, both of them in
April on the two power components — the honest headline for the page, and the
reason the framing sentence there is a recomputed count rather than a claim.
Under the previous fit it was 6 of 12; the constrained age curve is what moved
the extra cell. The full holdout, BABIP and the two season-level years
included, is what cleared the gate.

(`public/data/accuracy/latest.json` is regenerated by the nightly job, so the
committed copy carries whichever fit was frozen when it last ran; rerun
`python scripts/build_accuracy_json.py` to refresh it by hand.)

## The model

    rest-of-season projection = marcel_tuned(prior full seasons
                                             + 2026 through as_of − 1 day)
                                x projected_pa_ros

Both halves are code that already existed and was already scored:

* **Rates.** `src/eval/baselines.marcel_tuned`, called on exactly the
  training frame `src/eval/intraseason.build_training_frame` builds at a
  cutoff. Nothing is re-implemented, so the model that ships is bit-for-bit
  the arm the harness scored. Marcel weights by trials, so a 480-PA partial
  season naturally counts more than a 90-PA one — the model scales itself as
  the year goes on with no extra machinery. The engine is named once, in
  `ros.LIVE_ENGINE`; the builder stamps it into the document as `engine` and
  `scripts/build_accuracy_json.py` reads it to decide which arm the accuracy
  page marks live, so the scoreboard cannot end up scoring a model the site
  does not serve.
* **Playing time.** Station B (`src/projections/playing_time.py`): the
  horizon blend of the 30-day and season PA shares, a one-lineup-slot cap,
  and the injured and optioned projected at their pre-injury share times
  their expected return fraction — MAE 20.3 PA at one month and 37.0 at two
  ([playing-time.md](playing-time.md)). The method is named once, in
  `playing_time.PRODUCTION_METHOD`, exactly as the engine is named once in
  `ros.LIVE_ENGINE`; the builder asks for it rather than naming a method, so
  whatever station B's gate last put into production is what the site
  serves, and stamps it into the document as `playing_time_method`.

**The cutoff is exclusive.** A game played *on* the as-of date has not
finished when the morning's projection is made, so the partial season runs
through `as_of − 1 day` — the same convention station B uses, and the same
one the harness enforces with `assert_split_clean`.

**Who gets a projection.** Every hitter with `projected_pa_ros > 0`: 599 of
the 602 hitters on a 40-man roster at 2026-09-03. That used to be 420 — only
the active ones — and the difference is station B's expected returns: a
hitter on the injured list or optioned to the minors now carries the share he
would take healthy, discounted by how much of the horizon he is expected to
miss, so he gets a small line instead of none. 109 of the 599 are under five
plate appearances and 243 are over fifty; the page's leaderboard defaults to
a 50-PA floor and the player card picks its default hitter the same way, so
a man projected for two plate appearances does not head the table.
A hitter with projected PA and no professional record gets the league rate,
which is what Marcel with zero trials *is*; his preseason-Marcel comparison
column stays empty, because there is no completed season to withhold.

## What station B's switch changed (Sept 3, 2026)

The site had been multiplying tuned Marcel by station B's *previous* method —
the trailing-30-day share with a hard injured-list zero — for a week after
station B replaced it. Rebuilt on the same as-of date so nothing but the
playing time moves (2026-09-02, tuned Marcel both sides):

| | before | after |
|---|---|---|
| hitters in the document | 420 | 598 |
| hitters whose `pa_ros` went 0 → positive | — | 178, 1,480 PA between them |
| hitters dropped | — | 0 |
| club total PA | 808 (WSH) – 918 (LAD) | **unchanged**, 808 – 918 |
| hitters per club | 14 | 15–23 |
| largest rate difference, 15 columns × 420 common hitters | — | **0.000** |

Read the last two rows together: **the club totals do not move at all, and no
rate moves at all.** Station B normalizes within a club, so the 1,480 plate
appearances the injured and optioned now take are 1,480 the healthy regulars
no longer take — Ohtani 114.8 → 113.8, Yordan Alvarez 101.3 → 97.3, Corbin
Carroll 98.0 → 93.7, mean −3.5 PA across the 420 who were already there.
Judge, on the 60-day list, goes from absent to 1.7. The counting stats
(`k_ros`, `bb_ros`, `hr_ros`) move with the plate appearances because they are
the rate times the playing time; the rates themselves are bit-identical, which
is the check worth making — a playing-time change that moved a rate would mean
something was wired wrong.

The committed `latest.json` is a day later than that comparison (as-of
2026-09-03, 599 hitters, club totals 769–880 over 21–23 remaining games), and
the rates are identical there too: the local PA parquet ends 2026-09-01, so
the extra day of cutoff adds no data to the rate side.

**The accuracy page is unaffected.** Its rest-of-season section
(`scripts/build_accuracy_json.py`, `section_ros`) scores component *rate* MAE
from `scripts/run_intraseason_backtest.py`, on hitters with a minimum count of
realized trials. Neither script reads playing time or reports a plate
appearance or a counting stat, so there is nothing there to keep in step with
the served method.

## Rates to a line

The five components are rates over different denominators, so the counting
line is rebuilt with the standard identities — using the player's *own*
regressed walk and strikeout rates wherever a denominator depends on them,
not league ones:

    BB  = bb_rate x PA          HBP = hbp_rate x PA     SF = sf_rate x PA
    AB  = PA - BB - HBP - SF    K   = k_rate x PA       HR = hr_rate x PA
    BIP = AB - K - HR + SF      H   = babip x BIP + HR
    2B + 2*3B + 3*HR = iso x AB

`hbp_rate` and `sf_rate` are the league's rates through the cutoff — nobody
projects hit-by-pitch. Sacrifice bunts and interference are folded into AB
rather than carried (together ~0.4% of PA, so AB runs ~0.5% high; the wOBA
effect is under a point). Extra-base points split into doubles and triples
at a fixed triples-per-double ratio of 0.12, matching
`scripts/assemble_and_compare.py` so the preseason and live wOBA on the same
page are computed the same way.

wOBA uses the FanGraphs 2024 linear weights:

| BB | HBP | 1B | 2B | 3B | HR |
|---|---|---|---|---|---|
| .690 | .722 | .883 | 1.244 | 1.569 | 2.015 |

Its denominator AB + BB + HBP + SF is exactly PA under the identity above,
so `woba_ros` is a per-PA rate. Fed the 2026 league line (K% .2207, BB%
.0892, HR/PA .0303, BABIP .2893, ISO .1564) it returns **.3139**, which is
the league wOBA computed from the raw 2026 counts — the sanity test in
`tests/test_projections/test_ros.py`.

## Pitchers (Sept 3, 2026)

The same file now carries a `pitchers` array beside `players`, and the
leaderboard and player pages have a hitters/pitchers toggle. Nothing in the
hitter half of the contract moved: `players`, `n_hitters`, `engine`, `arms`,
`components` and the wOBA block are exactly what they were, which is why the
page's hitter views needed no changes at all.

**The two halves of a pitcher's line are held to different standards, and the
document says which is which in a field rather than only in prose.**

| Half | What it is | Gated? |
|---|---|---|
| the **rates** | `marcel_pitcher_tuned` fed the harness's own training frame at the cutoff — K%, BB%, HR/BF, BABIP against. `pitcher_engine` names it. | **yes** — each beat league average, the previous season *and* season to date out of sample on five cells ([backtest-baselines.md](backtest-baselines.md#the-pitcher-side-of-station-a--sept-3-2026)) |
| the **batters faced** | a projected workload. `batters_faced_method` reads `"structural"`. | **no** — there is no station B for pitchers, and this does not pretend there is |

The rates are the same estimator the hitter side runs, with pitcher constants;
the components register into the harness under a `p_` prefix and the
projection the page shows is bit-for-bit the arm that was scored. Four of the
five are columns. The fifth, the walks-plus-hit-batsmen rate, is what station
E's FIP term consumes: it is scored in the same run and it feeds the odds, but
a column labelled BB% has to mean walks, so it is not on the page.

The workload is arithmetic on the pitcher's own recent usage:

    projected BF = club games remaining
                 x appearance rate per club game   (trailing 30 days and
                   season blended half and half, each regressed toward the
                   pitcher's role's rate with 10 club games of ballast)
                 x batters faced per appearance    (regressed toward the
                   role's average with 5 appearances of ballast)
                 x station B's expected active fraction, for a pitcher who is
                   hurt or optioned

Role is read off the workload itself, not a depth chart: a pitcher averaging
at least 12 batters an outing is being used as a starter this month, whatever
he was in April, and an opener is correctly a reliever. That is crude. It is a
denominator, it is labelled as one, and the gated half does not depend on it.
A pitcher with no 2026 appearances is not projected at all — there is no usage
to extrapolate, and inventing one would be a depth chart rather than a
projection.

`fip_ros` is the same FIP arithmetic station E's starter term runs, on the
same coefficients, re-centred so a league-average line comes back at the
league's own runs allowed per nine. Innings come from the league's batters
faced per inning; projecting a pitcher's own would be another ungated
structural choice for no gain.

The pitcher block also fails on its own. If the pitcher season table or the
Stats API roster call is missing, `pitchers` comes back empty and the hitter
projection is still fresh — the site's established product does not go stale
because a newer block could not be built.

**On the 2026-09-03 board** (`python scripts/build_ros_projections.py`, the
nightly's exact invocation): 599 hitters and **676 pitchers**, 249 used as
starters and 427 as relievers, 21,830 projected batters faced in total.
Skubal 29.8% K / 5.2% BB, Skenes 28.1% / 6.9%, Wheeler 27.6% / 7.2%, Webb
21.2% / 6.3%, Mason Miller 40.3% / 9.5%. The dated snapshot for that day had
already been written that morning by the previous build and is not
overwritten — the archive records what was served — so `2026-09-03.json`
carries no pitcher block and `latest.json` does, exactly as happened with the
engine switch a day earlier.

## The file

`public/data/projections/latest.json` is rewritten every night;
`YYYY-MM-DD.json` is written once and never overwritten, the same archive
discipline as the playoff odds and the accuracy page. Per hitter:

    batter, name, team_id, team_abbrev, as_of, pa_ros,
    {k,bb,hr,babip,iso}_rate_{marcel,marcel_preseason,bayes},
    k_ros, bb_ros, hr_ros, woba_ros

Per pitcher:

    pitcher, name, team_id, team_abbrev, as_of, role, appearances,
    bf_to_date, bf_ros,
    {k,bb,hr,babip}_rate_{marcel,marcel_preseason},
    k_ros, bb_ros, hr_ros, fip_ros

Document level: `as_of`, `through`, `n_hitters`, `engine`,
`playing_time_method`, `method`, `framing`, `stale`, `stale_reason`, the arm
labels the page renders, and the wOBA weights, so the file says which wOBA it
means rather than the page assuming. The pitcher block adds `n_pitchers`,
`pitcher_engine`, `batters_faced_method`, `pitcher_method`, `pitcher_arms`
and `pitcher_components` — all additive, all prefixed, so a reader that has
never heard of pitchers still finds everything it looks for.

**Two models fill this file, so it names both.** `engine` says which Marcel
filled the rate columns and `playing_time_method` says which station B filled
`pa_ros` — `"blend_il"` from Sept 3, 2026 on, and absent in the snapshots
before it, which were built with the trailing-30-day share and a hard
injured-list zero. Both are read from the module that owns them
(`ros.LIVE_ENGINE`, `playing_time.PRODUCTION_METHOD`) rather than typed here,
so the document cannot claim a model the code does not run.

**The arm keys did not change with the engine.** `..._rate_marcel` is a slot
on the page — "the live arm" — not a set of constants, so the switch to
tuned Marcel left the column names and the site's JavaScript alone and moved
the labels ("Live (tuned Marcel + 2026)") and the `method` sentence instead.
`engine` is the field that says which Marcel filled the slot:
`"marcel_tuned"` from Sept 2, 2026 on, and absent in the dated snapshots
written before the switch.

That includes `2026-09-02.json` itself, which was written by stock Marcel
that morning and is **not** rewritten — the archive records what was served,
and the dated snapshot is never overwritten. So for one date the archive and
`latest.json` disagree, and `engine` (null against `"marcel_tuned"`) is what
tells them apart. From Sept 3 the two agree again.

## Failure modes, and what the page shows

| What is missing | What happens |
|---|---|
| `pa_outcomes_2026.parquet` and the `R2_*` credentials | The last committed file is served with `stale: true`, a reason naming the missing variables, and an orange badge on the page. |
| MLB Stats API unreachable | Same — station B cannot be rebuilt, so the previous day's projection carries. |
| `latest.json` absent entirely | The player and leaderboard pages render as before and say the projection has not been built in this checkout. |
| A hitter's preseason Bayesian file has no row for him | That comparison cell is a dash. 434 of 599 hitters had one on 2026-09-03. |
| `pitcher_seasons_api.parquet` absent, or the roster call fails | `pitchers` is empty and `n_pitchers` is 0; the hitter projection is still fresh, and both toggles disable themselves. |

The nightly workflow runs the build before the accuracy page and commits
`public/data/projections/` alongside the other snapshots.

## Caveats

- **The nightly runner has no `R2_*` secrets today.** Without them it cannot
  download the PA parquet, so both this projection and the accuracy page's
  rest-of-season section will render stale on GitHub until the secrets are
  configured. Everything is written to degrade visibly rather than fail.
- **Prior seasons come from the Stats API table, the partial season from
  Statcast PA data.** The two universes differ by ~0.7% of PA, so a hair of
  the in-season increment could be universe drift. Same caveat the harness
  carries; closing it means rebuilding prior seasons from
  `pa_outcomes_<year>.parquet`, and only 2026 exists in R2 today.
- **Names come from committed files.** `comparison_2026.parquet` covers the
  ~590 hitters the rest of the site names; the Chadwick birthdate table
  fills the rest but is gitignored, so a fresh CI checkout may leave some
  September call-ups showing an MLBAM id.
- **The projection is a point estimate.** The Bayesian components publish an
  interval and this does not; calibrated rest-of-season bands are roadmap
  5.7 and are a real thing the old model had that this one does not.
- **The tuned age curve used to be the weak part of the live engine, and it
  showed on the board — fixed Sept 3.** The first fit put K%'s peak age at 31
  with both slopes positive, which is a straight line in age and therefore
  partly a *level* correction rather than aging (Marcel regresses to the last
  season's league rate while the player's history spans three). The visible
  effect on the Sept 2 board was that the PA-weighted mean projected K% moved
  from .2208 (stock) to .2133 against a 2026 league rate of .2207 — stock sat
  exactly on league, tuned sat 0.7 points under it. The refit constrains the
  age term to peak inside 25–31 with slopes of opposite signs so it cannot
  act as a level, and the Sept 2 board rebuilt on the refrozen params comes
  back to **.2185**, 0.2 points under league. The projected-league-rate
  options built at the same time turn out not to be the answer — they lose
  their own inner validation, and only ISO takes one
  ([backtest-baselines.md](backtest-baselines.md#the-age-curve-was-not-aging--a-constrained-refit-and-a-projected-league-rate)).
  What remains is that a multiplier pinned to 1.0 at the peak still shifts the
  population mean; renormalizing it to mean 1 is the next thing to try.
- **A pitcher's projected batters faced is the weakest number on the page.**
  It has never been scored against a baseline, because no baseline for it
  exists in this repository yet. The obvious one — season-to-date appearance
  share, the pitcher twin of what station B is scored against — is a day's
  work and would turn `batters_faced_method` from a label into a gate. Until
  then, read the rate columns and treat the counting columns as scaling.
- **This is not station C.** Nothing here feeds the run environment or the
  playoff odds; it is the site's player-level number only. The pitcher rates
  *are* the same estimator station E's starter term runs, but station E keeps
  stock's constants deliberately, so a refit here cannot move a game price
  without the game price being re-scored.
