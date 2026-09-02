# North Star Architecture

**The one document every session works against.** The roadmap
([roadmap.md](roadmap.md)) is the dated plan; the factory design
([automation.md](automation.md)) is how work gets done in the background;
this is the *system* — what feeds what, where each piece stands, and the rule
that decides when a piece is allowed into production.

Last updated: Sept 1, 2026.

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
and it must stay that way until [A]–[C] beat their baselines.

## 2. Station status

| Station | What it is | Baseline to beat | Current best (Sept 1) | Status | Swap point |
|---|---|---|---|---|---|
| **A. Player rates** | K%, BB%, HR/PA, ISO, BABIP projections | Marcel (K% MAE .0261 on 2026) | Depth Charts .0234 · **ours .0271 (last)** | Built, **loses to Marcel** → not wired | `src/eval` providers; `data/projections/*.parquet` |
| **B. Playing time** | PA per player, rest of season | Recent-30-day PA share | — | Not built (roadmap 1.3) | `src/sim/strength.from_run_environment` inputs |
| **C. Team run env.** | Projected RS/G, RA/G per team from A×B | Season-to-date runs, regressed | — | Not built (roadmap 1.5) | `strength.from_run_environment(rs, ra)` |
| **D. Team strength** | Talent win% | Every team .500 (coin flip) | Regressed Pythagenpat, 60-game ballast | **Wired, live** | `src/sim/strength.py` |
| **E. Per-game P(win)** | Home win prob for one game | Home team always (Brier .2497) | log5 + HFA: **.2464** · Kalshi close **.2415** (876 games, measured — [market-benchmark-2026.md](market-benchmark-2026.md)) | Wired; no starter/lineup terms | `strength.home_win_prob` |
| **F. Season sim** | Monte Carlo, MLB tiebreakers, bracket | — (plumbing) | Within 1.6 pts of FanGraphs; coin-flip control within 1.9 | **Wired, validated** | `src/sim/season.py`, `standings.py`, `bracket.py` |
| **G. Odds** | P(playoffs/div/bye/pennant/WS), win bands | — | Live, 20k sims | **Wired, live** | `src/sim/odds.py` |
| **H. Site + archive** | Landing page, dated JSON, nightly job | — | Live; first snapshot 2026-09-01 | **Wired**; nightly first run pending | `scripts/run_playoff_odds.py`, `nightly-odds.yml` |
| **M. Market** | Daily archive of prices from **exchanges** (Kalshi, Polymarket: bid/ask, last, volume, open interest — public, no key) and **sportsbooks** (~30 books incl. Pinnacle via The Odds API, de-vigged); 2026 pre-game closes reconstructed for both exchanges; score E/A/G against all three | The market price itself | Kalshi ≈ Polymarket within ~1 pt on the same games | **Wired** — `market-snapshot.yml` 3×/day, closes in `market_closes_2026.parquet`; CLV / fill-aware ROI not yet | `src/market/` |

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
| 2 | **Per-game with starters, lineups, bullpen** | Public projection systems don't publish game odds; the market does. Team strength gets Brier .2464; the Kalshi close gets .2415 on the same 876 games. The **0.0049** between is pitcher-quality and lineup information — measured, see market-benchmark-2026.md. | Walk-forward Brier vs. market closing lines | E |
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
3. **Accuracy page** on the site from the scoreboard docs (roadmap 3, page 4).

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
