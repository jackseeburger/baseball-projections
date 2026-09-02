# North Star Architecture

**The one document every session works against.** The roadmap
([roadmap.md](roadmap.md)) is the dated plan; the factory design
([automation.md](automation.md)) is how work gets done in the background;
this is the *system* — what feeds what, where each piece stands, and the rule
that decides when a piece is allowed into production.

Last updated: Sept 2, 2026.

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
| **Prediction markets / exchanges** | Kalshi (CFTC-regulated, US), Polymarket | Order book; you trade against other participants, pay a small maker/taker fee | **Liquidity and position limits, not account bans** — you can only win what counterparties will lose | The **money venue**. No ban risk, you can be a *maker* (earn the spread instead of paying it), and the order-book mechanics are the same as finance: fills, slippage, adverse selection, inventory. |

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
 Rosters, IL, PA share ──► [B] Playing time ──────────────────┘                          ▼
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

**Today the chain is wired from [D] down and bypassed above it.** Team
strength comes straight from the standings' run differential, so the live
playoff odds contain no player modeling. That is deliberate for v1 (it ships)
and it must stay that way until [A]–[C] beat their baselines. [C] now does,
by .00055 Brier on the market game set (§2) — but it is a *blend* that reads
A's rate machinery and B's shares directly rather than the published
projections, and it is one season inside one standard error, so it waits for
a second season the same way E's lineup and bullpen terms do.

## 2. Station status

| Station | What it is | Baseline to beat | Current best (Sept 1) | Status | Swap point |
|---|---|---|---|---|---|
| **A. Player rates** | K%, BB%, HR/PA, ISO, BABIP projections. The site's live number is the **rest-of-season** one | Preseason: Marcel (K% MAE .0261 on 2026). Rest-of-season: `marcel_preseason` — the same Marcel with the current season withheld | Preseason: Depth Charts .0234 · ours .0271 (last). **Rest-of-season: Marcel fed the partial season, K% MAE .0278/.0296/.0343 at the May 1 / Jul 1 / Aug 1 cutoffs — beats our preseason Bayesian by 6–11% and wins 11 of 12 component-cutoff cells** ([ros-projections.md](ros-projections.md)) | Bayesian components **lose to Marcel** → still not wired. **Marcel-with-partial is wired and live (Sept 2)**: it is what the player and leaderboard pages serve, with the preseason Bayesian numbers beside it as a labelled comparison | `src/projections/ros.py`, `scripts/build_ros_projections.py`, `public/data/projections/`; `src/eval` providers; `data/projections/*.parquet` |
| **B. Playing time** | PA per player, rest of season | Season-to-date PA share (MAE 25.8 PA over one month; equal-share floor 28.5) | 30-day share + IL zeroed + one-lineup-slot cap: **MAE 22.1**, top-9 capture **.766** at a one-month horizon; **loses at two months** (46.1 vs 43.4) — 616/595 hitters, walk-forward at 2026-07-01 and 2026-08-01, [playing-time.md](playing-time.md) | Built; **beats the baseline at the ~26-game horizon it serves**, not yet wired | `src/projections/playing_time.py`; `data/parquet/playing_time_ros.parquet` |
| **C. Team run env.** | Projected RS/G, RA/G per team from A×B | Season-to-date runs, regressed (`pythag_60_sp`, Brier .24483) | **`pythag_C_sp` .24428** on the same 756 market games (−.00055, se .00086); .24606 vs .24661 on all 1,777 of 2026 and .24401 vs .24468 on 2025 — [market-benchmark-2026.md](market-benchmark-2026.md) | Built; **clears the gate on all three sets, inside 1 SE on each**; not wired | `src/sim/run_environment.py`; `strength.from_run_environment(rs, ra)` |
| **D. Team strength** | Talent win% | Every team .500 (coin flip) | Regressed Pythagenpat, 60-game ballast | **Wired, live** | `src/sim/strength.py` |
| **E. Per-game P(win)** | Home win prob for one game | log5 + HFA on team strength (Brier .2462) | **+ starting pitcher: .2448** (wired, live) · + posted lineup + bullpen quality: **.24454** (clears the gate, inside 1 SE; not wired) · + station C's run environment: **.24428** · Kalshi **.2416** / Polymarket **.2417** (756 games, measured — [market-benchmark-2026.md](market-benchmark-2026.md)) | **Wired, live** (Sept 2) — the starter term prices every remaining game with both probables posted; lineups, bullpen and C are each worth < .0006 and stay out of production until a second season sizes them; the market still holds .0027 | `strength.home_win_prob`, `src/sim/starters.py`, `lineups.py`, `bullpen.py`, `run_environment.py` |
| **F. Season sim** | Monte Carlo, MLB tiebreakers, bracket | — (plumbing) | Within 1.6 pts of FanGraphs; coin-flip control within 1.9 | **Wired, validated** | `src/sim/season.py`, `standings.py`, `bracket.py` |
| **G. Odds** | P(playoffs/div/bye/pennant/WS), win bands | — | Live, 20k sims | **Wired, live** | `src/sim/odds.py` |
| **H. Site + archive** | Landing page, **Model Accuracy page**, dated JSON, nightly job | — | Live; first snapshot 2026-09-01. Accuracy page renders the A/E/G scoreboards from generated JSON only (`public/data/accuracy/`), with a stale badge and a reason wherever a section could not be rebuilt | **Wired**; nightly first run pending | `scripts/run_playoff_odds.py`, `scripts/build_accuracy_json.py`, `nightly-odds.yml` |
| **M. Market** | Daily archive of prices from **exchanges** (Kalshi, Polymarket: bid/ask, last, volume, open interest — public, no key) and **sportsbooks** (~30 books incl. Pinnacle via The Odds API, de-vigged); 2026 pre-game closes reconstructed for both exchanges; score E/A/G against all three | The market price itself | Kalshi ≈ Polymarket within ~1 pt on the same games. Fill-aware P&L now runs: **every station-E model loses money at every edge threshold on both venues** — the full stack returns **−11.6% ROI** on 405 Kalshi bets at ≥2 pts (bootstrap CI −22.3% to −1.3%), against −6.2% for a random-edge control and 0% for doing nothing; 4.4 pts of that is the fee, 4.0 the spread, 3.2 the model ([money-exam-2026.md](money-exam-2026.md)) | **Wired** — `market-snapshot.yml` 3×/day, closes in `market_closes_2026.parquet`, simulated P&L in `src/market/pnl.py` (`scripts/money_exam.py`); true CLV — entries scored against a later close — waits on the 3×/day snapshot archive | `src/market/`, `pnl.py` |

Scoreboards: [accuracy-2026.md](accuracy-2026.md) (stations A, E, G),
[backtest-baselines.md](backtest-baselines.md) (station A baselines
2019–2025), [playoff-odds-validation.md](playoff-odds-validation.md) (G).

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

## 4. Edge thesis — where we expect to win, and how we'd know

Ordered by (expected payoff × how soon we can test it).

| # | Pocket | Why public systems are beatable there | Test | Station |
|---|---|---|---|---|
| 1 | **Fix what's broken in A** | Our fits used fake ages and no pitcher effect. Refit with real ages (done in code) + pitcher random effect and re-score. If we don't reach Depth Charts we learn the structure isn't the problem. | Component MAE vs Marcel/DC on 2020–2026 | A |
| 2 | **Per-game with starters, lineups, bullpen** | Public projection systems don't publish game odds; the market does. Team strength gets Brier .2462; both exchanges' closes get .2416–.2417 on the same 756 games. A regressed-FIP starter term takes **0.0014 of that 0.0046** (.2448); the remaining 0.0032 is lineups, bullpen state and a better pitcher model — measured, see market-benchmark-2026.md. | Walk-forward Brier vs. market closing lines | E |
| 3 | **Calibrated uncertainty** | Everyone publishes a point estimate. Contract valuation (Phase 6) needs a distribution; a 10th/90th band that actually covers 80% is a product no one sells. | Coverage tests (roadmap 5.7) | A, G |
| 4 | **Statcast-informed rates** | Marcel/Steamer/ZiPS use outcomes; batted-ball and swing data lead outcomes. Our PA-level models are positioned to use them and don't yet. | Same harness, add features | A |
| 5 | **Rookies and low-sample players** | Regression-to-mean systems are weakest exactly where hierarchical pooling (with minor-league / Statcast priors) is strongest. | Score the <200-PA-history cohort separately | A |
| 6 | **Multi-year horizons** | Steamer/ZiPS are one-year systems; long-term public projections are heuristics. Dynamic skill + component aging + health (Phase 5) is a structural difference. | Backtest 2010→2015 careers | A→Phase 5 |

What is **not** edge: September playoff odds, team-level per-game odds, and
any component where the harness says skill ≈ noise (BABIP: league average
ties Marcel).

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
