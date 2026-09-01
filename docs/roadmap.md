# Baseball Projections — Roadmap

**Written:** Sept 1, 2026
**Hard deadline for v1:** Sept 28, 2026 (Wild Card round opens Sept 29)

## Calendar anchors

| Date | Event |
|---|---|
| Sept 27 | Regular season ends |
| Sept 29 – Oct 1 | Wild Card round (best-of-3, all at higher seed) |
| Oct 3 | Division Series begin (best-of-5) |
| Oct 23 – 31 | World Series (best-of-7, 2-3-2) |
| Early Nov | Free agency opens |

## Scoping decisions

Three calls that make the deadline reachable:

1. **Marcel for pitchers in v1.** All five Bayesian components are hitter-side. Building the pitcher stack before Sept 28 eats the month. Marcel pitcher projections are adequate for team run prevention. Swap in December.
2. **Rest-of-season, not next-season.** It's September. Refit including 2026-to-date and project the remaining ~26 games. Shorter horizon is strictly easier than the offseason problem, and playing time becomes near-trivial because current rosters are known.
3. **Team-strength simulation, not lineup-level.** Aggregate player projections to team runs scored / runs allowed per game, then log5 per matchup. A full PA-level game simulator is a v2 feature and is where projects die.

## Explicit cut list for v1

Do not build these before Sept 28:
- Bayesian pitcher component models
- Career trajectory / multi-year walk
- Contract or trade valuation
- Health hazard models
- Bat tracking features
- Stuff+ / pitch quality models
- Lineup-level or PA-level game simulation
- In-game win probability

---

## Phase 0 — Foundation repair (Sept 1 – 8)

Nothing downstream is trustworthy until these land. Roughly a week.

**0.1 Real birthdates.** Replace `birth_year = first_year - 23` in `prepare_model_data`. Pull from the Chadwick Bureau register (`people.csv`, has MLBAM ID) or Lahman `People.csv`. Compute age as of June 30 of the season. Every existing aging output is invalid until this ships.

**0.2 Backtest harness.** New module, `src/eval/`. Interface:

```
backtest(component, train_through_year, predict_year) -> DataFrame
```

Refits on data through year Y, projects Y+1, joins realized outcomes. Score component-wise log loss, MAE, RMSE, and calibration (predicted vs realized in decile buckets). Baselines: Marcel, previous-season, league-average. Run over 2019 → 2020-2025.

*Acceptance: you can answer "does change X help?" with a number.*

**0.3 Re-score the five existing components.** With real ages and a real backtest. Expect the numbers to be worse than you thought. That's the point.

**0.4 Binomial aggregation.** Collapse PA rows to counts within (batter, pitcher, season, park, platoon) cells and switch the likelihood from Bernoulli to Binomial. Identical up to a constant, order of magnitude fewer rows. Drop the `p_k` `Deterministic` over `obs_id` at the same time — it's writing a float per PA per draw into the trace.

*This is what makes the rest of the month affordable. Do it before adding features.*

**0.5 Pitcher random effect.** Add `pitcher_ability[pitcher_idx]` as a second crossed random effect. `pitcher` is already in the Statcast rows. Re-score against 0.3. Biggest single accuracy gain available on the existing structure.

**0.6 Cleanup (timeboxed to one evening).** Collapse `train_pa_k_rate` / `train_iso_model` / `train_babip_model` into one parameterized `train_component()`. Deletes roughly 2,000 lines of the 3,266-line Modal app. Replace `/home/hermes/...` absolute paths with config. Delete root-level `fix_statcast.py`, `fix_volume.py`, `check_data.py`, `build_pa_data.py`.

---

## Phase 1 — Rest-of-season projections (Sept 8 – 15)

**1.1 Ingest 2026-to-date.** Extend the Statcast pipeline through the current date. Set up a daily incremental pull, not a full refetch.

**1.2 Refit components including 2026.** Same five, now with the current season in the training data.

**1.3 Playing time, short horizon.** For ~26 remaining games this is mostly bookkeeping: current active roster from the MLB Stats API, recent 30-day PA share by player, current IL list. Distribute projected team PAs across the roster by recent share. No hazard model needed at this horizon.

**1.4 Fix the assembly layer.** `assemble_and_compare.py` currently hardcodes `pa = 550` for everyone and reconstructs doubles and triples from ISO with magic constants (`triples = 0.12 * non_hr_xb / 1.12`). Replace with real projected PA from 1.3. For hit types, either project the outcome multinomial directly or at minimum project 2B and 3B rates as their own components rather than backing them out of ISO.

**1.5 Team run environment.** Aggregate player projections to team level:
- Runs scored: PA-weighted wOBA across the projected lineup → runs via linear weights
- Runs allowed: IP-weighted projected pitching (Marcel), split rotation vs bullpen
- Apply park factors

*Acceptance: 30 teams with projected runs scored and allowed per game, summing to a sane league total.*

---

## Phase 2 — Simulator and playoff odds (Sept 15 – 22)

The main new build. `src/sim/`.

**2.1 Schedule and standings ingest.** MLB Stats API (`statsapi.mlb.com`) for remaining schedule, current standings, results to date, probable starters. Daily refresh.

**2.2 Per-game win probability.**
- Pythagenpat on each team's projected run environment → team strength
- log5 for the matchup
- Home field advantage (~.540 for the home team, verify against current-season data)
- Optional: starting pitcher adjustment when probables are posted

**2.3 Season Monte Carlo.** Simulate all remaining games, 20,000+ runs. Carry forward actual results to date. Output the full joint distribution of final records, not marginals — you need the joint for seeding.

**2.4 Tiebreakers.** No Game 163 since 2022. Applied in order: head-to-head record, then intradivision record, then intraleague record over the last half of the season. Get this right or your division odds will be visibly wrong in tight races.

**2.5 Bracket.** Per league: 3 division winners + 3 wild cards. Top two division winners get byes. Wild Card best-of-3, all games at the higher seed. DS best-of-5, LCS and WS best-of-7, home field by record.

**2.6 Outputs per team.** P(make playoffs), P(win division), P(bye), P(win pennant), P(win World Series), and the projected final record distribution with percentile bands.

*Acceptance: your playoff odds land within a few points of FanGraphs and Baseball Prospectus for most teams. Large divergence is a bug until proven otherwise.*

---

## Phase 3 — Site and ship (Sept 22 – 28)

Static site, no backend. Nightly Modal job writes JSON, static host serves it. You already have `public/` and `export_to_r2.py`; Vercel is also connected if you'd rather deploy there.

**Pages:**

1. **Playoff odds** (landing page through October) — table of all 30 teams with the Phase 2 outputs, sortable, plus the current bracket
2. **Team projections** — projected final records with bands, runs scored/allowed
3. **Player projections** — rest-of-season and full-season component rates with credible intervals, sortable, searchable
4. **Model accuracy** — the credibility page. Backtest results vs Marcel, Steamer, ZiPS across 2020-2025 plus in-season tracking of your preseason 2026 projections against actuals

**3.1 Archive daily snapshots.** Write dated JSON every night and never overwrite. This is cheap now and impossible to reconstruct later. Odds-movement-over-time charts during the playoffs are the single most compelling thing the site will have.

**3.2 Nightly job.** Modal scheduled function: pull yesterday's results → update standings → re-run simulator → write dated JSON → trigger static deploy. Model refits weekly, not nightly.

**3.3 Ship by Sept 26** to leave two days of buffer before the Wild Card round.

---

## Phase 4 — Live through October (Sept 29 – Oct 31)

Mostly operating rather than building.

- Series-level odds updating after each postseason game
- Odds-movement charts from the archived snapshots
- Retrospective: how did preseason projections do against final standings, scored honestly against Steamer, ZiPS, and DepthCharts
- Log what broke. October is the stress test that tells you what to fix in November.

---

## Phase 5 — Career projection (Nov – Dec)

Timed to free agency. This is the structural upgrade discussed at length.

**5.1 Latent skill becomes dynamic.** Move `player_ability` from `dims="batter"` to `(batter, season)` with a transition:

```
theta[b, 0] ~ Normal(mu_ability, sigma_ability)
theta[b, t] = theta[b, t-1] + drift(age[b,t], component) + sigma_walk * z[b,t]
```

Random walk with age-dependent drift. No mean reversion toward `mu_ability` in the transition — it compounds over a ten-step walk and collapses everyone to average by 33.

Verify against Phase 0's backtest that one-year accuracy did not regress. It's easy to make the model more expressive and less accurate.

**5.2 Component-level aging curves.** Separate drift per component. Speed and defense decline early and steeply; walk rate holds into the mid-30s; power peaks around 26-27. This is what makes two identical 4-WAR 27-year-olds diverge by 35, and it's the main reason a real career model beats a WAR-decay heuristic.

**5.3 Health hazards.** Multi-state process: regular → limited → out of league (absorbing). Fit on career histories from Lahman and Retrosheet, conditioned on age, position, recent WAR, playing time history. At long horizons this term dominates the rate decline.

**5.4 Position transitions.** Defensive component crossing a threshold moves the player down the spectrum (SS → 3B → 1B/DH). Discrete jump in the positional adjustment that no smooth decay produces.

**5.5 Playing time, long horizon.** Endogenous on skill and health. Declining players must get fewer PAs or you manufacture impossible negative-WAR seasons.

**5.6 Career Monte Carlo.** 10,000 paths per player. Output percentile bands per future season — the ZiPS-style table.

**5.7 Calibration check.** Project every player from 2010 forward ten years and verify roughly 10% of realized careers landed below your 10th percentile. Also check that simulating the whole current population forward five years reproduces the real age distribution of MLB regulars.

---

## Phase 6 — Valuation (Dec – Jan)

**6.1 $/WAR model.** Fit on free agent signings, tiered or as a convex function of projected WAR. The 2025-26 market paid roughly $12.8M per projected WAR for 2+ WAR players, $8.5M for 1-2 WAR, $6.7M below that. A single blended rate systematically underpays stars.

**6.2 Surplus value per simulated path.**

```
Surplus_j = Σ_t [WAR_jt × $perWAR × (1+g)^t − Salary_t] / (1+r)^t
```

Discount rate 5-8%, inflation g fit from history. Report the distribution: median, P(negative surplus), 10th and 90th percentiles.

**6.3 Opt-outs.** Evaluate inside the path loop — player leaves when projected remaining market value exceeds remaining guarantee. One line, and it's the only correct way to price them.

**6.4 Arbitration model.** Separate comp-based regression on service class, platform-year counting stats, and prior salary. Not a value model. Train and benchmark against the MLB Trade Rumors projections and actuals.

**6.5 Trade value.** Surplus value with the arb stream substituted for the contract. Backtest against real signings using Cot's Contracts.

**6.6 New site pages.** Career trajectory viewer with percentile fans, contract valuation for actual free agents as they sign, running scoreboard of your valuation vs the actual deal.

---

## Risks

| Risk | Mitigation |
|---|---|
| Phase 0 runs long and eats the month | Birthdates and backtest are the only non-negotiables. Ship v1 with the current models if needed. |
| Simulator odds diverge wildly from public systems | Build 2.3 first and validate final-record distributions before adding bracket logic. |
| Nightly job fails silently during October | Alert on staleness. Site shows last-updated timestamp prominently. |
| Refits too slow to iterate | Phase 0.4 exists for this reason. Do it early. |
| Scope creep into pitcher models before Sept 28 | See cut list. |

## Definition of done for v1

Live site on Sept 26 with playoff odds for all 30 teams, updating nightly, backed by hitter projections you have honestly backtested against Marcel and the public systems, with a visible accuracy page and archived daily snapshots.
