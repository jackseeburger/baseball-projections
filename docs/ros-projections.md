# Rest-of-season projections — the number the site serves

**Station A, live.** What a visitor sees on a player page is now
*Marcel fed the partial current season* × station B's projected
rest-of-season plate appearances. The preseason Bayesian components — the
only number this site used to show — are still there, labelled, one column
to the right.

Shipped Sept 2, 2026 · built by `scripts/build_ros_projections.py` ·
math in `src/projections/ros.py` · data at
`public/data/projections/{latest,YYYY-MM-DD}.json`.

## Why the model changed

The [gate rule](architecture.md#3-the-gate-rule) says a station runs the
model that beats its baseline in the harness, and until Sept 2 nothing had
asked the harness the *product's* question. The season-level backtest asks
"given everything through 2025, how good is 2026?" The player page asks
"given everything through this morning, how good is the rest of 2026?" —
and the intra-season walk-forward
([backtest-baselines.md](backtest-baselines.md#intra-season-walk-forward--rest-of-2026-rates))
answered it:

| Component | Cutoff | `marcel` (live) | `bayes_preseason` (was live) | Gap |
|---|---|---|---|---|
| k_rate | May 1 | **.0278** | .0296 | −6.3% |
| k_rate | Jul 1 | **.0296** | .0325 | −8.9% |
| k_rate | Aug 1 | **.0343** | .0386 | −11.0% |
| bb_rate | Aug 1 | **.0268** | .0279 | −4.1% |
| hr_rate | Aug 1 | **.0152** | .0153 | −1.0% |
| iso | Aug 1 | .0567 | **.0551** | +2.9% |

Marcel with the current season folded in wins 11 of the 12
component-cutoff cells the accuracy page shows, and it beats *Marcel with
2026 withheld* on the same 11 — so the gain is in-season information, not a
better prior. Our Bayesian components never beat `marcel_preseason` on K%
either: on the same information they buy nothing, and they were giving up
the information advantage on top of it. Shipping them as the live number
was the one thing the gate rule forbids.

BABIP is the exception in both directions: in-season data adds nothing to
it, and league average roughly ties Marcel on it. It is projected because
the wOBA reconstruction needs it, not because it is skill.

## The model

    rest-of-season projection = marcel(prior full seasons
                                       + 2026 through as_of − 1 day)
                                x projected_pa_ros

Both halves are code that already existed and was already scored:

* **Rates.** `src/eval/baselines.marcel`, called on exactly the training
  frame `src/eval/intraseason.build_training_frame` builds at a cutoff.
  Nothing is re-implemented, so the model that ships is bit-for-bit the arm
  the harness scored. Marcel weights by trials, so a 480-PA partial season
  naturally counts more than a 90-PA one — the model scales itself as the
  year goes on with no extra machinery.
* **Playing time.** Station B (`src/projections/playing_time.py`): 30-day
  PA share, IL zeroed, one-lineup-slot cap, MAE 22.1 PA at the ~26-game
  horizon it serves ([playing-time.md](playing-time.md)).

**The cutoff is exclusive.** A game played *on* the as-of date has not
finished when the morning's projection is made, so the partial season runs
through `as_of − 1 day` — the same convention station B uses, and the same
one the harness enforces with `assert_split_clean`.

**Who gets a projection.** Every hitter with `projected_pa_ros > 0`: 420 of
the 602 hitters on a 40-man roster at 2026-09-02. The rest are on the IL,
optioned, or otherwise unavailable and have no rest-of-season line at all.
A hitter with projected PA and no professional record gets the league rate,
which is what Marcel with zero trials *is*; his preseason-Marcel comparison
column stays empty, because there is no completed season to withhold.

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

## The file

`public/data/projections/latest.json` is rewritten every night;
`YYYY-MM-DD.json` is written once and never overwritten, the same archive
discipline as the playoff odds and the accuracy page. Per hitter:

    batter, name, team_id, team_abbrev, as_of, pa_ros,
    {k,bb,hr,babip,iso}_rate_{marcel,marcel_preseason,bayes},
    k_ros, bb_ros, hr_ros, woba_ros

Document level: `as_of`, `through`, `n_hitters`, `method`, `framing`,
`stale`, `stale_reason`, the arm labels the page renders, and the wOBA
weights, so the file says which wOBA it means rather than the page assuming.

## Failure modes, and what the page shows

| What is missing | What happens |
|---|---|
| `pa_outcomes_2026.parquet` and the `R2_*` credentials | The last committed file is served with `stale: true`, a reason naming the missing variables, and an orange badge on the page. |
| MLB Stats API unreachable | Same — station B cannot be rebuilt, so the previous day's projection carries. |
| `latest.json` absent entirely | The player and leaderboard pages render as before and say the projection has not been built in this checkout. |
| A hitter's preseason Bayesian file has no row for him | That comparison cell is a dash. 323 of 420 hitters had one on 2026-09-02. |

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
- **This is not station C.** Nothing here feeds the run environment or the
  playoff odds; it is the site's player-level number only.
