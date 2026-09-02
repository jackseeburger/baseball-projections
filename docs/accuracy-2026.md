# 2026 Accuracy Scoreboard — where we actually stand

**As of Sept 1, 2026** (season 85% complete). Reproduce with
`python scripts/score_2026_projections.py` and
`python scripts/backtest_game_odds.py --season 2026`.

This page exists to answer one question honestly: **do our models have edge?**
Right now the answer is no, and the numbers below say exactly where.

## 1. Preseason player projections vs. 2026 actuals

All systems scored on the **same 263 hitters** (≥150 PA), trials-weighted.
Public systems as captured Apr 9; ours generated Apr 10. Lower is better.

| Component | Depth Charts | ZiPS | Steamer | Marcel | **Bayes (ours)** | League avg |
|---|---|---|---|---|---|---|
| K% MAE | **.0234** | .0240 | .0256 | .0261 | .0271 | .0487 |
| BB% MAE | **.0156** | .0163 | .0162 | .0173 | .0170 | .0250 |
| HR/PA MAE | **.0081** | .0085 | .0087 | .0088 | .0091 | .0115 |
| ISO MAE | **.0289** | .0316 | .0317 | .0321 | .0333 | .0403 |
| BABIP MAE | **.0211** | .0214 | .0221 | .0235 | .0239 | .0252 |

**Reading it:**

- **Our five Bayesian components rank last among the real systems on every
  component and roughly tie Marcel.** They are not broken — they beat league
  average by the same margin everyone does — but a hierarchical model with
  HSGP aging curves is currently delivering Marcel-level accuracy. That is the
  finding, and it is exactly what the roadmap's "expect the numbers to be worse
  than you thought" warned.
- **Depth Charts wins everything.** It is a *consensus* (Steamer + ZiPS blend)
  with hand-curated playing time. Consensus beating any single model is the
  normal state of the world; it is also the bar.
- **The spread between systems is small.** Depth Charts to Marcel on K% is
  .0027 MAE on a stat with a .048 league-average error. Most of what's
  predictable is captured by "weight the last three years and regress."

The likely reasons ours trails, in the order to test with the harness:
1. **Fake ages** — every projection here was made with `birth_year = debut - 24`
   (fixed in 0.1, not yet refit). The aging term was learning noise.
2. **No pitcher effect** — a batter's K% is partly who he faced (0.5).
3. **Over-regression** — the hyperpriors may shrink stars toward the mean
   harder than the data warrants; check by scoring the top/bottom deciles.

## 2. Day-to-day: per-game win probability, walk-forward

1,757 games of 2026 predicted one day at a time using only prior results.

| Model | Brier | Log loss |
|---|---|---|
| Pythagenpat, 100-game ballast | **.2476** | **.6884** |
| Pythagenpat, 60 (production) | .2478 | .6887 |
| Pythagenpat, 30 | .2484 | .6901 |
| Home team always (53.5%) | .2497 | .6926 |
| Raw win% into log5 | .2529 | .7002 |

**Reading it:** the production strength model beats "always pick the home
team" by 0.002 Brier. That is real but tiny — single MLB games are close to
coin flips, and a team-level strength number is a blunt instrument. Betting
markets score around .240–.245 here because they price the things we ignore:
**starting pitcher, lineup, bullpen availability, park, weather.** That is
the per-game edge frontier, not team strength.

Calibration is good in the .40–.65 band; the extreme buckets are tiny.

## 2b. Why "close to FanGraphs on playoff odds" means nothing in September

Mean absolute gap to FanGraphs playoff odds, Sept 1, 8,000 sims each:

| Strength model | P(playoffs) | P(division) | P(WS) | Exp. wins |
|---|---|---|---|---|
| **No model — every team is a .500 coin flip** | 1.94 | 1.78 | 1.30 | 1.06 |
| Ours (regressed Pythagenpat) | 1.63 | 2.10 | 1.37 | 0.66 |

A simulator with **no team-strength model whatsoever** lands within 2 points of
FanGraphs. With 26 games left, playoff odds are ~90% standings arithmetic; the
strength model only moves the margin (mostly in expected wins). So the
1.5-point agreement is a check that the *plumbing* is right — schedule,
tiebreakers, bracket — not evidence of modeling skill. Playoff odds become a
model test in April, not September.

## 3. So where is edge?

Not in September playoff odds (85% of the season is banked; being within
1.5 pts of FanGraphs is arithmetic, not modeling), and not in team-level
per-game odds. The plausible pockets, each testable in the harness:

| Pocket | Why public systems may be weak there | How we'd know |
|---|---|---|
| **Per-game with starters/lineups/bullpen** | Public projection systems don't publish game odds; the market does, and it's beatable at the margins with better pitcher-quality models (Stuff+, roadmap cut list for v1). | Brier vs. market closing line, walk-forward. |
| **Uncertainty, not point estimates** | Public systems publish a number; contract and trade valuation need a *distribution*. A calibrated 10th/90th percentile is a product no one is selling. | Coverage tests (roadmap 5.7). |
| **Long horizons (3–5 yrs)** | Steamer/ZiPS are one-year systems; ZiPS' long-term is a heuristic. Dynamic skill + aging + health (Phase 5) is a real structural difference. | Backtest 2010→2015 careers. |
| **Statcast-informed components** | Marcel and friends use outcomes; batted-ball and swing data carry signal about *future* outcomes (xwOBA-style). Our PA-level models are positioned to use it and currently don't. | Same harness, add features, watch MAE. |
| **Rookies / low-sample players** | Where regression-to-mean systems are weakest; hierarchical pooling across minor-league and Statcast data should shine. | Score the <200-PA-history cohort separately. |

The honest position: the infrastructure is the asset. Most model ideas will
not beat Depth Charts, and the harness will say so in minutes rather than
letting us believe otherwise for a season. The ones that survive are the
edge.
