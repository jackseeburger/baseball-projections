# North Star Architecture

**The one document every session works against.** The roadmap
([roadmap.md](roadmap.md)) is the dated plan; the factory design
([automation.md](automation.md)) is how work gets done in the background;
this is the *system* — what feeds what, where each piece stands, and the rule
that decides when a piece is allowed into production.

Last updated: Sept 1, 2026.

## 0. The north star

**Beat the market.** Not FanGraphs, not Marcel — the betting market's
closing line, which is the best publicly available probability for a baseball
event and prices in everything the public systems know plus sharp money. If
our probabilities beat the de-vigged closing line out of sample, that is edge
that pays; if they don't, no amount of Bayesian elegance matters.

Everything else in this document is instrumentation toward that. The longer
arc is the same skeleton pointed at financial markets: **the market is the
baseline, walk-forward is the only valid test, calibration and sizing are
where the money is made or lost**, and the factory exists to run that loop
without fooling ourselves. Sports markets are the training ground because
they are less efficient than financial ones and settle every night.

The scoreboard that matters (station **M** below):

| Market | Our model | Metric | Money metric |
|---|---|---|---|
| Moneyline (game winner) | Station E | Log loss / Brier vs de-vigged closing line | Simulated ROI + closing-line value (CLV) on bets where \|ours − market\| > threshold |
| Totals (runs) | Station C + E | Same | Same |
| **Player props** (K, HR, hits, TB) | **Station A directly** | Log loss vs prop line | Same |
| Futures (division, pennant, WS) | Station G | Log loss at season checkpoints | Same, long-dated |

**Player props are the shortest path to money.** A K% model that beats the
strikeout-prop market monetizes station A on its own, without the whole
rollup working. It is also the cleanest possible test of whether the
Bayesian components have edge, because the prop line already embeds Steamer,
ZiPS, and the sharps.

Ground rules that carry over to finance unchanged:
1. **The market is the baseline** for every station that has one. Beating
   Marcel is a unit test; beating the closing line is the exam.
2. **Archive market lines daily starting now.** Closing lines cannot be
   reconstructed later; without the archive there is no backtest. Same
   principle as the odds snapshots (roadmap 3.1).
3. **Score in money terms, not just probability terms.** A model can have
   better log loss and lose money (edge concentrated where vig is highest)
   or worse log loss and make money (edge concentrated on mispriced tails).
   Report both; CLV is the leading indicator, ROI the lagging one.
4. **Walk-forward only; no peeking.** Every prediction uses only data
   available before the line closed. The harness's leakage guard is the
   most important line of code in the repo.
5. **Sizing is part of the model.** Fractional Kelly on the calibrated
   edge; a miscalibrated model with correct point estimates still loses.
   Calibration (edge pocket #3) is therefore not a nicety.
6. **Deep learning earns its place the same way** — where the data is
   high-dimensional and sequential (pitch-level tracking, swing paths,
   order flow later), scored against the same baselines in the same harness.

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
| **E. Per-game P(win)** | Home win prob for one game | Home team always (Brier .2497) | log5 + HFA: **.2478** · market ≈ .240–.245 | Wired; no starter/lineup terms | `strength.home_win_prob` |
| **F. Season sim** | Monte Carlo, MLB tiebreakers, bracket | — (plumbing) | Within 1.6 pts of FanGraphs; coin-flip control within 1.9 | **Wired, validated** | `src/sim/season.py`, `standings.py`, `bracket.py` |
| **G. Odds** | P(playoffs/div/bye/pennant/WS), win bands | — | Live, 20k sims | **Wired, live** | `src/sim/odds.py` |
| **H. Site + archive** | Landing page, dated JSON, nightly job | — | Live; first snapshot 2026-09-01 | **Wired**; nightly first run pending | `scripts/run_playoff_odds.py`, `nightly-odds.yml` |
| **M. Market** | Daily archive of moneyline / totals / props closing lines; de-vig; score E/A/G against them; simulated ROI + CLV | The closing line itself | — | **Not built** — highest-priority new station | `src/market/` (to create); The Odds API (key needed), SportsbookReview history |

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
| 2 | **Per-game with starters, lineups, bullpen** | Public projection systems don't publish game odds; the market does. Team strength gets Brier .2478; market ≈ .243. The 0.005 between is pitcher-quality and lineup information. | Walk-forward Brier vs. market closing lines | E |
| 3 | **Calibrated uncertainty** | Everyone publishes a point estimate. Contract valuation (Phase 6) needs a distribution; a 10th/90th band that actually covers 80% is a product no one sells. | Coverage tests (roadmap 5.7) | A, G |
| 4 | **Statcast-informed rates** | Marcel/Steamer/ZiPS use outcomes; batted-ball and swing data lead outcomes. Our PA-level models are positioned to use them and don't yet. | Same harness, add features | A |
| 5 | **Rookies and low-sample players** | Regression-to-mean systems are weakest exactly where hierarchical pooling (with minor-league / Statcast priors) is strongest. | Score the <200-PA-history cohort separately | A |
| 6 | **Multi-year horizons** | Steamer/ZiPS are one-year systems; long-term public projections are heuristics. Dynamic skill + component aging + health (Phase 5) is a structural difference. | Backtest 2010→2015 careers | A→Phase 5 |

What is **not** edge: September playoff odds, team-level per-game odds, and
any component where the harness says skill ≈ noise (BABIP: league average
ties Marcel).

## 5. Sequencing from here

Keyless work (runs in any cloud session now):
0. **Station M, archive first** — nightly job pulls today's moneyline,
   totals, and available props and writes a dated snapshot. One free Odds
   API key (`ODDS_API_KEY`, 500 req/month) covers a daily pull. Backfill
   2026 closing lines from SportsbookReview history so E can be scored
   against the market immediately, not next year.
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
