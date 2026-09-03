# North Star Architecture

**The one document every session works against.** The roadmap
([roadmap.md](roadmap.md)) is the dated plan; the factory design
([automation.md](automation.md)) is how work gets done in the background;
this is the *system* — what feeds what, where each piece stands, and the rule
that decides when a piece is allowed into production.

Last updated: Sept 3, 2026.

## 0. The north star

**Truth first; the market is the bar; money is the exam.**

- **Primary objective: accuracy against reality**, measured with proper
  scoring rules (log loss, Brier, calibrated intervals) on outcomes that
  actually happened. Marcel, Depth Charts, and the market price are all
  *baselines* for that one score. Beating Marcel is a unit test; beating the
  market is the exam — but it is the same test, with a harder benchmark.
- **The market is the strongest available baseline** wherever one exists.
  For per-game outcomes, totals, props, and futures, our probability is
  compared to the market's implied probability on the same event, out of
  sample. Public projection systems already live inside that price.
- **Money is a second scoreboard, not a second truth.** It adds three things
  accuracy alone cannot see: a **hurdle** (fees/vig), **selectivity** (we
  need to be right where we *disagree* with the market, not everywhere), and
  **sizing** (fractional Kelly on a calibrated edge; a miscalibrated model
  with good point estimates still loses). Money metrics — closing-line value
  as the leading indicator, simulated ROI with confidence intervals as the
  lagging one — decide *what to bet*. They never decide *what is true*.
- **Why not optimize against the market directly:** Goodhart. A model fit
  to the market's mistakes learns things like "fade day-game favorites" that
  vanish when the market adapts. Player rates don't adapt. Truth-based
  scoring keeps the model about baseball; the gate rule (§3) stays
  truth-based for that reason.

**Two model classes, one harness.**

| Class | Inputs | Scored on | Used for |
|---|---|---|---|
| **Independent** | Our data only — no market prices as features | Truth | The site, the accuracy page, career and contract valuation (no market exists for 5-year WAR) |
| **Market-anchored** | Market price as a feature + our private information (pitcher quality, lineups, rates) | Truth *and* money | Betting. This is how professionals actually do it, and it cannot claim independent skill |

**Venues (station M).** Two kinds, and they play different roles:

| Venue | Examples | How they price | What caps you | Role for us |
|---|---|---|---|---|
| **Sportsbooks** | Pinnacle, DraftKings, FanDuel | Book sets the line, takes vig (2–4.5 pts moneyline, more on props) | **They limit or ban winning accounts** | The sharpest *reference price* (Pinnacle close). Archive it as the accuracy benchmark; do not plan to scale money there. |
| **Prediction markets / exchanges** | Kalshi (CFTC-regulated, US), Polymarket | Order book; you trade against other participants, pay a small maker/taker fee | **Liquidity and position limits, not account bans** — you can only win what counterparties will lose | The **money venue**. No ban risk, you can be a *maker* (earn the spread instead of paying it — worth 7.8 pts of ROI on 2026's book, measured, and still not enough: [money-exam-2026.md](money-exam-2026.md)), and the order-book mechanics are the same as finance: fills, slippage, adverse selection, inventory. |

Prediction-market sports prices are typically less sharp than Pinnacle,
especially props, futures, and mid-liquidity contracts, so edge is more
plausible there — and book-vs-exchange price gaps are a low-model-risk
first trade. Both venues expose **public market-data APIs with no key**
(Kalshi trade-api v2: markets, order book, trades, settled history;
Polymarket Gamma/CLOB/data APIs). Archive both daily; score against both.

**Player props are the shortest path to money.** A K% model that beats the
strikeout-prop price monetizes station A on its own, without the whole
rollup working, and it is the cleanest test of whether the Bayesian
components have edge, because the prop price already embeds Steamer, ZiPS,
and the sharps.

**Ground rules that carry to financial markets unchanged:** the market is
the baseline; walk-forward only — every prediction uses only information
available before the price you compare it to; archive prices daily starting
now because they cannot be reconstructed; score in money terms *and*
probability terms; sizing and execution (fills, not mids) are part of the
model; deep learning earns its place in the same harness against the same
baselines. Sports exchanges are the training ground because they are less
efficient than financial ones, settle nightly, and have the same order-book
structure.

## 1. The rollup

Everything on the site is a rollup of models below it. Each arrow is a
**station**: an input contract, a swap point in code, a baseline, and a score.

```
 Statcast PA data ─┐
 Chadwick ages ────┼─► [A] Player rate models ──► [C] Team run environment ──► [D] Team strength
 Pitchers faced ───┘        K% BB% HR ISO BABIP        RS/G, RA/G per team          talent win%
                                   ▲                          ▲                          │
 Rosters, transactions ───► [B] Playing time ──────────────────┘                          ▼
                                                                                 [E] Per-game P(win)
 Starters, lineups, bullpen, park, weather ──────────────────────────────────────────────┤
                                                                                         ▼
 Schedule + results (Stats API) ───────────────────────────────────────► [F] Season Monte Carlo
                                                                          tiebreakers, bracket
                                                                                         ▼
                                                              [G] Standings, playoff odds, pennant, WS
                                                                                         ▼
                                                                          [H] Site + daily snapshot archive
```

**Since Sept 3 the chain is wired from [C] down** (roadmap 1.5's swap, and
[BAS-55](playoff-odds-validation.md#sept-3-2026--the-full-chain-is-wired)).
Team strength for the live odds is no longer the standings' run differential
on its own: it is [C]'s bottom-up rebuild — A's rate machinery over B's shares
for the hitters, the same FIP table for the rotation and pen — blended half
and half with those regressed rates, and every remaining game whose starters
are announced is priced by [E]'s full stack on top of it. Each of those terms
cleared its gate on the market game set (§2), each by less than .0006 and
inside one standard error, which is the honest risk in the change: the gate
rule (§3) says a model that beats its baseline out of sample runs, and one
season is what "out of sample" currently means here. A second season sizes
them; `--legacy-chain` keeps the previous answer reproducible in the meantime.
One function prices a game for both the harness and the nightly
(`src/sim/game_model.py`), so the scoreboard and the site cannot drift apart.

[A] and [B] *are* both live, but on the player and leaderboard pages rather
than in the odds chain: the rest-of-season projection the site serves is
tuned Marcel times station B's playing time, and both halves are named by the
module that owns them (`ros.LIVE_ENGINE`, `playing_time.PRODUCTION_METHOD`)
and stamped into the document, so a station that clears its gate reaches the
page without either builder being edited. The pitcher line is the same shape
with the other id on the plate appearance, and since Sept 3 **both of its
halves have a score**: the rates (station A, pitcher side) and the projected
batters faced (**station B-P**, `pitcher_ros.BF_METHOD`), which had been
stamped `structural` — the label for a model nobody had scored — for exactly
one day.

## 2. Station status

| Station | What it is | Baseline to beat | Current best (Sept 1) | Status | Swap point |
|---|---|---|---|---|---|
| **A. Player rates** | K%, BB%, HR/PA, ISO, BABIP projections. The site's live number is the **rest-of-season** one | Preseason: Marcel (K% MAE .0261 on 2026). Rest-of-season: `marcel_tuned_preseason` — the same tuned Marcel with the current season withheld | Preseason: Depth Charts .0234 · ours .0271 (last). **Rest-of-season: tuned Marcel fed the partial season, K% MAE .0269/.0293/.0341 at the May 1 / Jul 1 / Aug 1 cutoffs — beats our preseason Bayesian on 10 of 12 component-cutoff cells** ([ros-projections.md](ros-projections.md)). The tuning itself: per-component ballast, recency and age curve fitted walk-forward on 2020–24, frozen in `src/eval/marcel_params.json`, then scored on 2025, 2026 and the three 2026 cutoffs — **17/25 cells, pooled −1.10% ± 0.30 of stock Marcel's MAE**, with no component worse than stock: BABIP (−3.0%), K% (−1.4%), ISO (−1.1%), HR/PA (−0.5%); BB% keeps Tango's constants ([backtest-baselines.md](backtest-baselines.md#tuning-marcel--fitted-constants-beat-tangos-defaults)) | **The Bayesian arm finally got a fair fight (Sept 3)**: given a dated cutoff on PA rows and prior seasons rebuilt from Statcast, the same K% model refit at each cutoff scores .0278/.0302/.0341 against tuned Marcel's .0269/.0293/.0341 — paired gaps of +.0009/+.0009/+.0000 (t 1.6/1.6/0.0, n 315/231/126), so it no longer loses significantly at any cutoff, and it beats its own preseason self by 6–11%, which means the deficit the published table had been charging it with was the withheld season rather than the model ([backtest-baselines.md](backtest-baselines.md#the-fair-fight--the-bayesian-arm-refit-at-the-cutoff-bas-59)); level is still not a win, so Bayesian components **lose to Marcel** → still not wired. The **opposing-pitcher term added alongside it is the cautionary half**: PSIS-LOO prefers it by 280 nats at 11.3 dSE, and the one rest-of-season cutoff affordable with it on scores 5.9% *worse* (K% MAE .03615 against .03415), because the projection is made at a neutral pitcher and discards exactly what the term learned — LOO measured held-out plate appearances, the gate measures the rest of a season, and only the second one counts. **Tuned Marcel-with-partial is wired and live (Sept 2)**: it is what the player and leaderboard pages serve, with the preseason Bayesian numbers beside it as a labelled comparison. The **age curve was refit under a constraint (Sept 3)** — peak in 25–31, slopes of opposite signs, so it cannot double as a level correction — and the HR/PA and ISO gains that used to vanish out of sample now survive, while the live board's mean K% comes back from 0.7 pts under league to 0.2 under; the projected-league-rate options built alongside it lose their own inner validation and only ISO takes one. **The pitcher half of the station went in the same day (Sept 3)**: the same estimator pointed at the other id on the plate appearance, scored on five walk-forward cells against league average, the previous season and season to date, and all five components — K%, BB%, HR/BF, BABIP against and the walks-plus-hit-batsmen rate station E reads — clear the gate against all three, K% by 3.5% of stock's MAE after tuning and BABIP against by nothing at all ([backtest-baselines.md](backtest-baselines.md#the-pitcher-side-of-station-a--sept-3-2026)); `src/sim/starters.rate_table` now *calls* that provider rather than re-implementing it, so station A and station E cannot drift, and the station E scoreboard below is unmoved (`pythag_C_sp_bpa_ip` still .24388) because station E keeps stock's constants until it is re-scored | `src/projections/ros.py` (`LIVE_ENGINE`), `pitcher_ros.py`, `scripts/build_ros_projections.py`, `public/data/projections/`; `src/eval` providers (`pitchers.py`); `src/eval/marcel_params.json`, `marcel_pitcher_params.json`; `data/projections/*.parquet` |
| **B. Playing time** | PA per player, rest of season | Season-to-date PA share (MAE 24.7 PA over one month, 41.5 over two; equal-share floor 27.3) | Horizon blend of the 30-day and season shares + one-lineup-slot cap + **expected returns from the injured list and the minors in place of the hard roster gate** — return-time distribution fitted on 2023–2025 only: **MAE 20.3 at one month and 37.0 at two**, top-9 capture .767/.734. Paired per-hitter against season-to-date, **−4.37 PA (t −5.6) at one month and −4.53 (t −3.7) at two**; against the IL-zeroing blend it replaces, −0.56 and −6.42. It wins RMSE and both realized-PA-weighted metrics at both cutoffs too, which no earlier version did, and 2025 corroborates across seven horizons from 12 to 92 games remaining ([playing-time.md](playing-time.md) §5). 644/624 hitters, walk-forward at 2026-07-01 and 2026-08-01 | **Wired and live (Sept 3, 2026)** — it is what `public/data/projections/latest.json` multiplies the rate model by, stamped there as `playing_time_method`; the site builder asks for `playing_time.PRODUCTION_METHOD` rather than naming a method, so the next thing through this gate ships without editing it. Beats every baseline on every metric at both horizons, the first version that does | `src/projections/playing_time.py`, `src/projections/il_returns.py`; `data/parquet/playing_time_ros.parquet`, `public/data/projections/` |
| **B-P. Pitcher workload** | Batters faced per pitcher, rest of season — the number the site multiplies the gated pitcher rates by | Season-to-date batters faced per club game x games remaining (MAE 51.1 BF over the 26 holdout cutoffs); trailing-30-day version 50.4; last season prorated 67.6; projecting nobody 93.3 | **The projection already in production wins, and now has a score instead of a stamp: MAE 45.6 BF, paired −5.6 against the season-rate extrapolation (t −16.7), −4.9 against the 30-day one (−12.1), −22.1 against last season (−19.8) and −47.7 against no model (−19.9)**, on 22,807 pitcher-projections at 26 walk-forward as-of dates a fortnight apart over 2024–2026, constants for every challenger chosen on 2022–2023 only. Three candidates were built to beat it and all three lose: re-deriving role from `gamesStarted` with a fitted horizon blend and league role priors costs +0.99 (t +10.6), normalizing each club's staff to the club's own projected total costs +2.57 (t +8.3), and **porting station B's own winning change — projecting the injured at their pre-injury usage times an expected return fraction — costs +4.18 (t +7.4) against the served model and +3.19 (t +6.2) against the same candidate without it, with the damage largest at exactly the horizon where it paid most for hitters** (+9.48 BF at a May 1 cutoff against −13.5 PA for station B's hitters). What *does* port is the other half: applying station B's return-time distribution at all is worth −4.39 BF a pitcher (t −37.8), and −9.24 for a starter — the largest single term in the model. The one thing that beats the served model is an attrition haircut on the *healthy* — a constant per-role hazard of losing the rest of the season, station B §8.6's unbuilt idea — which wins MAE and loses the realized-workload-weighted metric that station B's gate also required, so it is measured and not wired ([pitcher-workload.md](pitcher-workload.md)) | **Wired and live, and no longer stamped `structural`** — `batters_faced_method` in `public/data/projections/latest.json` reads `recent_usage` and `pitcher_method` carries the margin, so the page stops implying a gap that is not there. The harness calls `pitcher_ros.projected_batters_faced` rather than copying it, so it cannot score a model the site does not serve | `src/projections/pitcher_workload.py`, `pitcher_ros.py`, `il_returns.py`; `scripts/build_pitcher_workload.py`, `run_pitcher_workload_backtest.py`; `data/workload/` |
| **C. Team run env.** | Projected RS/G, RA/G per team from A×B | Season-to-date runs, regressed (`pythag_60_sp`, Brier .24483) | **`pythag_C_sp` .24428** on the same 756 market games (−.00055, se .00086); .24606 vs .24661 on all 1,777 of 2026 and .24401 vs .24468 on 2025 — [market-benchmark-2026.md](market-benchmark-2026.md) | **Wired, live (Sept 3)** — the blend is the team strength the nightly draws every unannounced game with and the run environment the bracket's rotations bend; it clears the gate on all three sets, inside 1 SE on each. **The two things the bottom-up half is blind to have now been measured and neither one clears** (Sept 3): per-venue run factors from the prior seasons' home/road splits are worth −.00006 on the 756 and +.00002 on 2025, because a park multiplies both clubs' runs and a win probability is a ratio; a club-level BABIP-allowed residual moves a game ten times as far and scores *worse* (+.00007), because the top-down half already carries the club's defence — `src/sim/park.py`, `src/sim/defence.py`, [market-benchmark-2026.md](market-benchmark-2026.md) | `src/sim/run_environment.py`, `src/sim/game_model.py`; `scripts/run_playoff_odds.py --legacy-chain` reverts |
| **D. Team strength** | Talent win% | Every team .500 (coin flip) | Regressed Pythagenpat, 60-game ballast | **Wired, live** | `src/sim/strength.py` |
| **E. Per-game P(win)** | Home win prob for one game | log5 + HFA on team strength (Brier .2462) | **+ starting pitcher: .2448** (wired, live) · + posted lineup + bullpen quality: **.24454** · + station C's run environment: **.24428** · **+ the pen that is actually available, weighted by pitch counts: .24400** · **+ the starter's own expected innings as the workload split: .24388** · Kalshi **.2416** / Polymarket **.2417** (756 games, measured — [market-benchmark-2026.md](market-benchmark-2026.md)) | **Wired, live** (Sept 2: the starter term; **Sept 3: the whole chain**) — one function (`src/sim/game_model.py`) prices a game for both the harness and the nightly, so every remaining game with both probables posted is served as `pythag_C_sp_bpa_ip` — station C's run environment, the starter over his own expected innings, the availability-weighted pen over the rest, and the posted card when a club has published one. The four terms above the starter each clear the gate on the common game set, and each is < .0006 and inside 1 SE there, so one more season is what would size them. **Availability itself is worth nothing** (a pitch-count weight fires on 1,485 of 1,512 club-games where the old binary rest rule fired on 12, and scores the same); what pays is pricing the relief innings off component rates against the league. The start-length term is the only one whose *sign* is significant anywhere (t = −2.9 on 2025). **Park and team defence were added to the same ladder on Sept 3 and did not clear it** — .24382 and .24395 against the served .24388, with the sign flipping between seasons on the first and negative on both on the second — so the served model is unchanged and the harness scores them as `pythag_C_sp_bpa_ip_pk`, `_def` and `_pk_def`. The market still holds .0023, and two of the four things it was assumed to be hiding are now known not to be there **A learned challenger has now been tried and does not clear the gate (Sept 3)**: a gradient-boosted model over 23,193 games of 2015-2026, given the same pre-game inputs and no functional form at all, scores .24461 on the 756 against the chain's .24388 (paired +.00073, se .00123) — better on all of 2026 (.24536 vs .24592) and on 2025 (.24339 vs .24360), worse where the gate is, inside one standard error everywhere, and 88-94% correlated with the chain. It re-derives the chain's own terms in the chain's own order; the two places it disagrees are the starter's expected innings as a *level* and a pen delta at nearly twice the chain's weight. Available behind `backtest_game_odds.py --learned`; the chain runs ([market-benchmark-2026.md](market-benchmark-2026.md#station-e--a-model-that-chooses-its-own-form-sept-3-2026)) | `strength.home_win_prob`, `src/sim/starters.py`, `lineups.py`, `bullpen.py`, `reliever_usage.py`, `run_environment.py` |
| **F. Season sim** | Monte Carlo, MLB tiebreakers, bracket | — (plumbing) | Within 1.6 pts of FanGraphs; coin-flip control within 1.9 | **Wired, validated** | `src/sim/season.py`, `standings.py`, `bracket.py` |
| **G. Odds** | P(playoffs/div/bye/pennant/WS), projected wins | Current record extrapolated at .500 (the coin-flip control) and at the club's own rate; a preseason projection held fixed; no information at all | **Scored at last (Sept 3): projected final wins MAE 4.50 against 5.80 for the .500 extrapolation, 6.12 for own-rate, 8.47 preseason and 10.40 for no model; playoff Brier .1034 against .1119 / .1236 / .2032 / .2294.** Paired and clustered by season, −1.31 wins (se 0.16) and −.0085 of Brier (se .0038) against the .500 extrapolation, on **7,470 club-projections an arm over 2015–2025 (2020 excluded) at 249 weekly as-of dates**. Calibration: reliability .00055, resolution .1257, skill score .551. **The edge is front-loaded**: on playoff probability it is −.034 in April and gone by early August, and nominally negative (+.0024, t 1.8) in the last tenth; on projected wins it survives to the end (−0.14, t −5.2). Division, pennant and World Series never separate from the control (t −1.5, −1.6, −0.6 — sixty, twenty and ten events). The generous starter window is worth 1.3% of the margin ([team-projection-backtest.md](team-projection-backtest.md)) | **Wired, live, and now measured** — since Sept 3 the board is served off the full chain (E on the announced games, C's blend on the rest and in the bracket), with `--legacy-chain` for the previous answer and `--coin-flip` for the no-model control. The gate is cleared on projected wins at every horizon and on playoff probability only before August, and the site's framing now says so | `src/sim/odds.py`, `scripts/run_playoff_odds.py`; `src/eval/team_season.py`, `src/eval/team_backtest.py`, `scripts/run_team_backtest.py` |
| **H. Site + archive** | Landing page, **Model Accuracy page**, dated JSON, nightly job | — | Live; first snapshot 2026-09-01. Accuracy page renders the A/E/G scoreboards from generated JSON only (`public/data/accuracy/`), with a stale badge and a reason wherever a section could not be rebuilt | **Wired**; nightly first run pending | `scripts/run_playoff_odds.py`, `scripts/build_accuracy_json.py`, `nightly-odds.yml` |
| **M. Market** | Daily archive of prices from **exchanges** (Kalshi, Polymarket: bid/ask, last, volume, open interest — public, no key) and **sportsbooks** (~30 books incl. Pinnacle via The Odds API, de-vigged), game markets **and seven player-prop series** (HR, K, hits, TB, RBI, SB, outs, keyed to MLBAM ids); 2026 pre-game closes reconstructed for both exchanges and for every settled prop that traded, plus the **hourly pre-game price path** for both contracts (876 game markets / 20,989 candles; **78,134 prop markets / 516,666 candles**); score E/A/G against all three | The market price itself | Kalshi ≈ Polymarket within ~1 pt on the same games. Fill-aware P&L now runs **on both sides of the book and on two contracts**. Taking moneylines: **every station-E model loses money at every edge threshold on both venues** — the full stack returns **−11.6% ROI** on 405 Kalshi bets at ≥2 pts (CI −22.3% to −1.3%), against −6.2% for a random-edge control; 4.4 pts of that is the fee, 4.0 the spread, 3.2 the model. Making: resting a limit order instead recovers **7.8 of those 8.4 points** and still returns **−3.1%** (CI −21.7% to +15.2%, 142 fills of 377 posted at a 5-pt margin), no better than its own shuffled-edge control — **execution was the whole difference between losing 11 points and losing 3, and there is no edge underneath either** ([money-exam-2026.md](money-exam-2026.md)). **Player props are the softer contract the moneyline exam went looking for, and the fee eats the whole difference**: the Marcel-with-partial rate model returns **−5.7% ROI** on 30,075 Kalshi prop bets at ≥2 pts (CI −8.3% to −3.1%) — it beats its random-edge control (−8.2%) where the moneyline model lost to its own, and the loss net of the fee is 0.7 pts against 7.2 on moneylines — but 5.1 of the 5.7 is the taker fee, and waiving it (a maker fill) gives −0.6% (−3.2%, +2.0%), indistinguishable from zero and from a league-average-rate control. **Both of that finding's two open ends were closed on Sept 3.** The opposing pitcher is now in the hitter's price and the opposing card in the pitcher's, by log5 on the probable starter over his own expected innings and the pen over the rest (`src/market/matchup.py`): it clears the gate on the half of the archive its one constant was not chosen on — **−0.00095 of Brier paired per contract, t −4.4 clustered by game, every stat the right sign** — so it is the default, and it closes a fifth of the market's lead (0.0048 → 0.0038), nearly all of it on strikeouts (−0.0057). And **resting a limit order on props does not rescue the money**: −7.2% at the margin its training half chose, against −3.2% for crossing the same 30,423 contracts, because 27% of the "limit orders" would have crossed the book and the genuinely passive fills are adversely selected (hit rate falls monotonically from .53 to .34 as the margin widens). The maker route the last write-up named as the remaining one is now measured and is worse than taking ([props-exam-2026.md](props-exam-2026.md)) | **Wired** — `market-snapshot.yml` 3×/day now archives ~4,400 open prop markets a run (99.5% resolved to MLBAM ids), closes and candles for both contracts committed under `data/market/` (`prop_closes_2026.parquet`, `kalshi_candles_2026.parquet`, `kalshi_prop_candles_2026.parquet` — the prop path cannot be re-fetched once Kalshi ages the markets out), all four exams in `src/market/pnl.py` (`scripts/money_exam.py [--maker]`, `scripts/props_exam.py [--matchup on] [--maker]`); true CLV — entries scored against a later close — waits on the 3×/day snapshot archive | `src/market/`, `pnl.py`, `props.py`, `matchup.py`, `backfill.py` |

Scoreboards: [accuracy-2026.md](accuracy-2026.md) (stations A, E, G),
[backtest-baselines.md](backtest-baselines.md) (station A baselines
2019–2025), [playing-time.md](playing-time.md) (B, hitters),
[pitcher-workload.md](pitcher-workload.md) (B-P, pitchers — 5 seasons, 44
biweekly as-of dates), [playoff-odds-validation.md](playoff-odds-validation.md) (G, the
plumbing and the controls),
[team-projection-backtest.md](team-projection-backtest.md) (G, the accuracy —
10 seasons, 249 weekly as-of dates).

## 3. The gate rule

> A station's model is wired into production only when it beats the
> station's baseline out of sample in the harness, on the common player or
> game set, and is recorded on the scoreboard. Until then the baseline runs.

Consequences:
- **Bayesian components stay out of the rollup** until they beat Marcel.
  Wiring them in today would make the odds *worse*, not better.
- **Baselines are production code**, not scaffolding. Marcel, regressed
  Pythagenpat, and "home team always" ship until dethroned.
- **A PR that changes a model must include harness output.** No numbers, no
  merge. The factory worker enforces this.
- **Agreement with FanGraphs is not a score.** The coin-flip control proved
  September playoff odds can't distinguish models; use per-game Brier and
  component MAE.

Which *method* a station reaches for — hierarchical Bayes, machine learning,
or neither — is a separate question from the bar it has to clear, and is
answered in [methods.md](methods.md).

## 4. Edge thesis — where we expect to win, and how we'd know

Ordered by (expected payoff × how soon we can test it).

| # | Pocket | Why public systems are beatable there | Test | Station |
|---|---|---|---|---|
| 1 | **Fix what's broken in A** | Our fits used fake ages and no pitcher effect. Refit with real ages (done in code) + pitcher random effect and re-score. If we don't reach Depth Charts we learn the structure isn't the problem. | Component MAE vs Marcel/DC on 2020–2026 | A |
| 2 | **Per-game with starters, lineups, bullpen** | Public projection systems don't publish game odds; the market does. Team strength gets Brier .2462; both exchanges' closes get .2416–.2417 on the same 756 games. A regressed-FIP starter term takes **0.0014 of that 0.0046** (.2448); station C's run environment, the available bullpen and the starter's expected innings take another **0.0009** between them (.24388), and the remaining **0.0023** is weather, rest and a better pitcher model — measured, see market-benchmark-2026.md. Bullpen *availability* has now been tested twice and is worth nothing either time, and **park and team defence have now been tested once each and are worth nothing either**: the park factors predict 2026's park run environments at r = 0.44 and still move a game's probability by 0.0005, because they are symmetric. | Walk-forward Brier vs. market closing lines | E |
| 3 | **Calibrated uncertainty** | Everyone publishes a point estimate. Contract valuation (Phase 6) needs a distribution; a 10th/90th band that actually covers 80% is a product no one sells. | Coverage tests (roadmap 5.7) | A, G |
| 4 | **Statcast-informed rates** | Marcel/Steamer/ZiPS use outcomes; batted-ball and swing data lead outcomes. Our PA-level models are positioned to use them and don't yet. | Same harness, add features | A |
| 5 | **Rookies and low-sample players** | Regression-to-mean systems are weakest exactly where hierarchical pooling (with minor-league / Statcast priors) is strongest. | Score the <200-PA-history cohort separately | A |
| 6 | **Multi-year horizons** | Steamer/ZiPS are one-year systems; long-term public projections are heuristics. Dynamic skill + component aging + health (Phase 5) is a structural difference. | Backtest 2010→2015 careers | A→Phase 5 |

What is **not** edge: September playoff odds, team-level per-game odds, and
any component where the harness says skill ≈ noise (BABIP: league average
ties Marcel). The first of those is now **measured rather than asserted**:
over 2015–2025 our playoff probabilities beat a .500 extrapolation of the
standings by .034 of Brier in April, by nothing at all from the start of
August, and are nominally behind it in the last tenth of the season
([team-projection-backtest.md](team-projection-backtest.md)). What survives
all season is the *projected-wins* column, better by 1.31 wins of MAE pooled
and by 0.14 even in the final fortnight — so the honest claim for station G
is a win total, not an October probability.

## 5. Sequencing from here

Keyless work (runs in any cloud session now):
0. **Station M, archive first** — nightly job snapshots Kalshi and
   Polymarket MLB markets (bid/ask, last, volume, open interest; **no key
   needed**) plus sportsbook lines (Odds API free key, SportsbookReview
   history), writes dated snapshots. Backfill 2026 from Kalshi's settled
   markets + trade history and SportsbookReview so E can be scored against
   both venues immediately, not next year.
1. **Wire the rollup with baselines** — B (playing time from Stats API
   rosters), C (Marcel rest-of-season rates × PA → RS/RA), then flip D to
   `from_run_environment`. Score against the coin-flip and current D on
   per-game Brier. This makes the chain real end-to-end with honest parts,
   so any later A improvement flows to the site automatically.
2. **Station E v1** — probable starters from the Stats API + a pitcher
   Marcel; score on Brier **and against the market** (M). This is edge
   pocket #2 and the postseason odds driver.
2b. **Props pipeline** — map station A's K%/HR projections + station B's
   PA to per-game prop probabilities; score against archived prop lines.
   First direct test of whether A has monetizable edge.
3. ~~**Accuracy page** on the site from the scoreboard docs (roadmap 3, page 4).~~ **Done** — `scripts/build_accuracy_json.py` writes `public/data/accuracy/{latest,YYYY-MM-DD}.json` from `score_2026_projections.py --json-out`, `backtest_game_odds.py --json-out` and the §2b control table; the page hand-types no numbers. Market rows need the `R2_*` secrets on the nightly runner; without them that section renders stale.

Needs R2 / Modal / W&B:
4. **Edge pocket #1** — refit A with real ages + binomial likelihood +
   pitcher effect; re-score 2020–2026 vs Marcel and Depth Charts.
5. Decide, from the number, whether A earns a place in C.

## 6. Definition of "working against it"

Every task in the queue names its station and its baseline. A task is done
when the scoreboard row for that station changes (or the doc records that it
didn't and why). Anything that can't be expressed that way is infrastructure,
and infrastructure is judged by whether it makes the next scoreboard change
cheaper.
