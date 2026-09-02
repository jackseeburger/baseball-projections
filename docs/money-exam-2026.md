# The Money Exam — 2026 per-game P(win) priced against the exchanges

Station E scored in dollars against station M. Brier says whether a
probability is good ([market-benchmark-2026.md](market-benchmark-2026.md));
this says what it would have been worth after the spread and the fee — the
three things accuracy cannot see (architecture.md §0): a **hurdle**, because
you pay the ask and the fee rather than the mid; **selectivity**, because you
are only paid where you *disagree* with the market; and **sizing**, because a
fractional-Kelly stake on a miscalibrated edge still loses.

Produced by:

```
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet \
    --out data/parquet/game_preds_2026.parquet
python scripts/money_exam.py --markdown
```

**Headline: no model here is worth betting.** Every one of them loses money at
every threshold on both venues once the spread and the fee are charged, and
the one that loses least is the *worst* model in the room. Nothing about that
is a surprise — no model is at the market on Brier (0.24454 vs 0.24156) — and
recording it is the point. The instrument exists so that the day a model does
reach the market, the answer to "is it worth trading?" is a command, not an
argument.

## Method

**The join.** One row per game from the walk-forward backtest (every quantity
rebuilt from games, appearances and posted lineups strictly *before* that
game's date) merged on `game_pk` to the exchange's reconstructed pre-pitch
close (the last quote before first pitch, median 15 minutes out). 756 games —
the set both venues priced, the same population `market-benchmark-2026.md`
scores.

**The trade rule.** Buy YES on the home team at the **ask** when the model is
above it, buy NO at **1 − bid** when the model is below the bid, and do
nothing inside the spread. Comparisons are strict and edges are measured
against the quote we would actually cross, not the mid, so the market's own
price never trades and neither does a model that merely rounds it.
Kalshi's book is used as quoted — its median close spread on these games is
exactly **1¢**. Polymarket's reconstruction only recovers a mid, so a
configurable half-spread (default 1¢) is assumed around it.

**Stakes.** Two rules, both reported. *Flat* is one unit of capital at risk
per bet. *Quarter-Kelly* stakes `0.25 · (p − cost)/(1 − cost)` of a fixed
1-unit bankroll, capped at 5% on any one game; stakes do not compound, so the
result does not depend on the order the games happened to fall in. A unit of
stake at cost `c` buys `1/c` contracts, so ROI = profit ÷ stake is comparable
across prices.

**Settlement.** From `home_won`. A winning YES bought at the ask returns
`1 − ask − fee` per contract; a winning NO bought at `1 − bid` returns
`bid − fee`; a loser forfeits the stake and still pays the fee.

**Metrics.** Per model, venue, threshold and staking rule: bets, hit rate,
total return, ROI, mean edge taken, the CLV proxy, max drawdown, and a **95%
bootstrap CI on ROI** from 2,000 draws that resample *games* with replacement
and re-form the ratio inside each draw.

**Controls.** `market` — the close itself, which sits inside the spread by
construction, never trades, and returns exactly 0. It is the anchor: a model
whose CI does not clear zero has not beaten *doing nothing*. `random_edge` —
the close plus N(0, 3 pts), which manufactures disagreement out of noise and
pays the spread and the fee for it; it is what an ROI looks like when the edge
is fake. `home_constant` — the no-information model, which disagrees with the
market constantly and therefore bets the most.

### Fees

**Kalshi**, taker: `round_up_to_cent(0.07 · C · P · (1 − P))`, C contracts at
price P in dollars. It is maximal at a coin flip — 1.75¢, which rounds to 2¢
— and vanishes in the tails. Every one of our 405 Kalshi bets at the 2-pt
threshold sat near enough to even money to pay the full **2¢ per contract**,
which on a mean cost of 47¢ is **4.4% of stake** before anything else happens.
Maker fees are set to zero here: resting a limit order is the trade we would
actually want, but a maker fill is not guaranteed, so the exam charges the
taker.

*Source.* Kalshi's fee-schedule page (`kalshi.com/docs/fees`) was unreachable
from this environment (HTTP 429 through the agent proxy), and the API docs no
longer carry a fees page — the surviving official page is the rounding rule,
[docs.kalshi.com/getting_started/fee_rounding](https://docs.kalshi.com/getting_started/fee_rounding)
("trade_fee = ceil_6dp(model_fee)", accumulated per order across fills), which
confirms the round-*up* direction but not the constant. The 0.07 is therefore
the **published formula as reported by secondary sources**, not a first-party
read; it is a parameter (`--kalshi-fee-rate`), not a constant, so correcting it
is a flag. We round up per *contract*, which is exact for a one-contract order
and slightly conservative (charges more) for larger ones.

**Polymarket**: no taker fee on most sports markets as of 2026; the cost is
the spread, modelled as the half-spread above (`--half-spread`). Secondary
sources report a sports taker fee introduced during 2026 (0.03 in March, 0.05
in July); `--polymarket-fee-rate` prices that, and the sensitivity table below
shows what it would do.

## Kalshi — 756 games, quoted bid/ask, taker fee rate 0.07

**flat 1u**

| Model | n ≥0pt | n ≥2pt | n ≥4pt | n ≥6pt | ROI ≥0pt | ROI ≥2pt | ROI ≥4pt | ROI ≥6pt |
|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 682 | 405 | 206 | 93 | -13.0% | -11.6% | -4.7% | +8.4% |
| pythag_60_sp_lu | 688 | 421 | 222 | 99 | -10.4% | -11.2% | -9.5% | -0.6% |
| pythag_60_sp | 695 | 427 | 226 | 107 | -8.5% | -10.6% | -10.7% | +1.3% |
| pythag_60 | 708 | 495 | 309 | 195 | -7.3% | -8.1% | -11.4% | -11.8% |
| home_constant (control) | 712 | 560 | 420 | 313 | -6.5% | -8.3% | -7.2% | -3.6% |
| random_edge (control) | 653 | 319 | 100 | 21 | -5.6% | -6.2% | -2.9% | +4.6% |
| market (control) | 0 | 0 | 0 | 0 | +0.0% | +0.0% | +0.0% | +0.0% |

At edge ≥ 2 pts:

| Model | n bets | hit | staked | return | ROI | ROI 95% CI | mean edge | CLV | max DD | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 405 | 0.437 | 405.0u | -46.98u | -11.6% | (-22.3%, -1.3%) | 4.60 pt | +5.10 pt | 53.02u | 17.76u |
| pythag_60_sp_lu | 421 | 0.435 | 421.0u | -47.16u | -11.2% | (-22.1%, -0.2%) | 4.73 pt | +5.21 pt | 48.56u | 18.73u |
| pythag_60_sp | 427 | 0.436 | 427.0u | -45.12u | -10.6% | (-20.9%, -0.8%) | 4.77 pt | +5.24 pt | 48.25u | 19.09u |
| pythag_60 | 495 | 0.432 | 495.0u | -40.03u | -8.1% | (-18.0%, +1.8%) | 5.92 pt | +6.41 pt | 59.60u | 22.99u |
| home_constant (control) | 560 | 0.409 | 560.0u | -46.47u | -8.3% | (-18.0%, +1.6%) | 7.84 pt | +8.34 pt | 54.26u | 27.02u |
| random_edge (control) | 319 | 0.498 | 319.0u | -19.75u | -6.2% | (-17.6%, +5.0%) | 3.63 pt | +4.04 pt | 24.58u | 12.87u |
| market (control) | 0 | — | 0.0u | +0.00u | +0.0% | — | — | — | 0.00u | 0.00u |

**quarter-Kelly (cap 5%)**

| Model | n ≥0pt | n ≥2pt | n ≥4pt | n ≥6pt | ROI ≥0pt | ROI ≥2pt | ROI ≥4pt | ROI ≥6pt |
|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 682 | 405 | 206 | 93 | -9.4% | -9.1% | -4.0% | +3.4% |
| pythag_60_sp_lu | 688 | 421 | 222 | 99 | -10.1% | -9.7% | -9.3% | -4.0% |
| pythag_60_sp | 695 | 427 | 226 | 107 | -10.0% | -10.3% | -9.9% | -1.2% |
| pythag_60 | 708 | 495 | 309 | 195 | -8.9% | -9.1% | -11.0% | -11.6% |
| home_constant (control) | 712 | 560 | 420 | 313 | -7.2% | -7.3% | -6.6% | -4.8% |
| random_edge (control) | 653 | 319 | 100 | 21 | -5.6% | -7.6% | -6.7% | -4.6% |
| market (control) | 0 | 0 | 0 | 0 | +0.0% | +0.0% | +0.0% | +0.0% |

At edge ≥ 2 pts:

| Model | n bets | hit | staked | return | ROI | ROI 95% CI | mean edge | CLV | max DD | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 405 | 0.437 | 8.8u | -0.80u | -9.1% | (-20.4%, +1.8%) | 4.60 pt | +5.10 pt | 0.99u | 0.38u |
| pythag_60_sp_lu | 421 | 0.435 | 9.3u | -0.90u | -9.7% | (-21.4%, +1.8%) | 4.73 pt | +5.21 pt | 0.97u | 0.41u |
| pythag_60_sp | 427 | 0.436 | 9.4u | -0.97u | -10.3% | (-21.2%, +0.3%) | 4.77 pt | +5.24 pt | 1.06u | 0.42u |
| pythag_60 | 495 | 0.432 | 12.7u | -1.15u | -9.1% | (-20.1%, +2.3%) | 5.92 pt | +6.41 pt | 1.50u | 0.60u |
| home_constant (control) | 560 | 0.409 | 17.4u | -1.27u | -7.3% | (-18.0%, +4.3%) | 7.84 pt | +8.34 pt | 1.68u | 0.88u |
| random_edge (control) | 319 | 0.498 | 6.1u | -0.46u | -7.6% | (-19.2%, +4.1%) | 3.63 pt | +4.04 pt | 0.49u | 0.24u |
| market (control) | 0 | — | 0.0u | +0.00u | +0.0% | — | — | — | 0.00u | 0.00u |

## Polymarket — 756 games, mid ± 1.0¢, no taker fee

**flat 1u**

| Model | n ≥0pt | n ≥2pt | n ≥4pt | n ≥6pt | ROI ≥0pt | ROI ≥2pt | ROI ≥4pt | ROI ≥6pt |
|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 611 | 342 | 170 | 64 | -7.1% | -5.1% | -0.1% | +2.8% |
| pythag_60_sp_lu | 623 | 367 | 189 | 73 | -8.0% | -5.5% | -2.0% | +0.1% |
| pythag_60_sp | 623 | 388 | 185 | 81 | -6.7% | -5.9% | -10.5% | +4.8% |
| pythag_60 | 650 | 451 | 278 | 162 | -5.5% | -7.4% | -8.2% | -1.7% |
| home_constant (control) | 665 | 527 | 389 | 289 | -3.3% | -5.6% | -5.8% | +3.7% |
| random_edge (control) | 569 | 255 | 65 | 15 | -2.8% | -4.0% | -10.8% | +4.2% |
| market (control) | 0 | 0 | 0 | 0 | +0.0% | +0.0% | +0.0% | +0.0% |

At edge ≥ 2 pts:

| Model | n bets | hit | staked | return | ROI | ROI 95% CI | mean edge | CLV | max DD | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 342 | 0.450 | 342.0u | -17.47u | -5.1% | (-16.6%, +5.7%) | 4.47 pt | +5.47 pt | 28.56u | 0.00u |
| pythag_60_sp_lu | 367 | 0.444 | 367.0u | -20.13u | -5.5% | (-16.7%, +5.4%) | 4.54 pt | +5.54 pt | 33.50u | 0.00u |
| pythag_60_sp | 388 | 0.441 | 388.0u | -22.76u | -5.9% | (-16.9%, +5.1%) | 4.48 pt | +5.48 pt | 39.37u | 0.00u |
| pythag_60 | 451 | 0.417 | 451.0u | -33.52u | -7.4% | (-18.2%, +2.7%) | 5.72 pt | +6.72 pt | 51.47u | 0.00u |
| home_constant (control) | 527 | 0.400 | 527.0u | -29.66u | -5.6% | (-15.4%, +4.3%) | 7.53 pt | +8.53 pt | 49.78u | 0.00u |
| random_edge (control) | 255 | 0.486 | 255.0u | -10.24u | -4.0% | (-16.7%, +8.7%) | 3.44 pt | +4.44 pt | 22.68u | 0.00u |
| market (control) | 0 | — | 0.0u | +0.00u | +0.0% | — | — | — | 0.00u | 0.00u |

**quarter-Kelly (cap 5%)**

| Model | n ≥0pt | n ≥2pt | n ≥4pt | n ≥6pt | ROI ≥0pt | ROI ≥2pt | ROI ≥4pt | ROI ≥6pt |
|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 611 | 342 | 170 | 64 | -5.4% | -4.5% | +0.2% | -1.1% |
| pythag_60_sp_lu | 623 | 367 | 189 | 73 | -6.1% | -5.4% | -3.3% | -2.7% |
| pythag_60_sp | 623 | 388 | 185 | 81 | -6.4% | -6.1% | -8.5% | +1.3% |
| pythag_60 | 650 | 451 | 278 | 162 | -5.2% | -5.7% | -6.3% | -2.0% |
| home_constant (control) | 665 | 527 | 389 | 289 | -3.8% | -4.4% | -4.7% | +0.8% |
| random_edge (control) | 569 | 255 | 65 | 15 | -2.7% | -4.8% | -12.4% | -1.7% |
| market (control) | 0 | 0 | 0 | 0 | +0.0% | +0.0% | +0.0% | +0.0% |

At edge ≥ 2 pts:

| Model | n bets | hit | staked | return | ROI | ROI 95% CI | mean edge | CLV | max DD | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| pythag_60_sp_lu_bp | 342 | 0.450 | 7.4u | -0.33u | -4.5% | (-16.6%, +7.0%) | 4.47 pt | +5.47 pt | 0.73u | 0.00u |
| pythag_60_sp_lu | 367 | 0.444 | 7.9u | -0.43u | -5.4% | (-17.0%, +6.6%) | 4.54 pt | +5.54 pt | 0.79u | 0.00u |
| pythag_60_sp | 388 | 0.441 | 8.2u | -0.50u | -6.1% | (-18.0%, +5.5%) | 4.48 pt | +5.48 pt | 0.90u | 0.00u |
| pythag_60 | 451 | 0.417 | 11.3u | -0.65u | -5.7% | (-18.2%, +5.9%) | 5.72 pt | +6.72 pt | 1.23u | 0.00u |
| home_constant (control) | 527 | 0.400 | 16.0u | -0.70u | -4.4% | (-15.6%, +7.0%) | 7.53 pt | +8.53 pt | 1.38u | 0.00u |
| random_edge (control) | 255 | 0.486 | 4.6u | -0.22u | -4.8% | (-17.9%, +8.5%) | 3.44 pt | +4.44 pt | 0.34u | 0.00u |
| market (control) | 0 | — | 0.0u | +0.00u | +0.0% | — | — | — | 0.00u | 0.00u |

## What the numbers say

**Every model loses, and the ranking is upside down.** At the 2-pt threshold
on Kalshi the full station-E stack returns **−11.6%** on 405 bets
(CI −22.3% to −1.3%) while the production `pythag_60` returns −8.1% and the
no-information `home_constant` returns −8.3%. The better model loses *more*.
That is not a paradox: Brier is an average over all 756 games, and money is an
average over only the 405 where the model disagreed with the price by more
than two points — exactly the games where being better on average buys
nothing. On the games it bet, the stack scores **0.2473** against the
exchange's **0.2425**: it is *worse than the market precisely where it thinks
the market is wrong*, and the more confidently a model deviates, the more of
its own error it converts into stake. The needed hit rate on that book of bets
was **50.1%** (mean cost 47¢ plus a 2¢ fee); it hit **43.7%**, a shortfall of
2.6 standard errors.

**The controls behave.** `market` never trades and returns exactly zero — it
is the anchor, and no model's 95% CI sits above it. `random_edge`, which is
the market's own price plus three points of noise, returns −6.2% at the same
threshold: pure fake edge costs roughly the spread and the fee, and every one
of the four real models did *worse* than pure fake edge. `home_constant` bets
most often and loses steadily. Nothing here separates the models from their
own controls: every CI at every threshold spans the others.

**The apparent wins are noise.** `pythag_60_sp_lu_bp` shows +8.4% at the 6-pt
threshold on Kalshi — on 93 bets, with a CI from −13.9% to +31.2%, from a
model whose 405-bet book at 2 pts loses 11.6%. `random_edge` shows +4.6% at
the same threshold on 21 bets. Thresholds are not free parameters to be
chosen: at 6 pts on Kalshi flat three cells look positive and one of them is
the noise control; on Polymarket at 6 pts five do, including both controls.

**Where the money goes.** Decomposing the stack's Kalshi ROI at 2 pts: taking
the close as a free fill and charging nothing — which also lets 460 rather
than 405 disagreements clear the threshold — it loses **−3.2%**; crossing
the quoted 1¢ book takes it to **−7.2%**; the 2¢-per-contract fee takes it to
**−11.6%**. So about 4.4 points of ROI is the fee, 4.0 points is the spread,
and 3.2 points is the model being wrong. Even at a **frictionless** fill —
trade every game at the close, no spread, no fee — the stack returns −3.2%
and `pythag_60` returns −0.9%. There is no cost structure that rescues this
book, because there is no edge underneath it.

**Polymarket looks better only because it costs less.** Same predictions, same
games, no fee, an assumed 1¢ half-spread instead of a real one: −5.1% instead
of −11.6%. Widen the assumption to 2¢ and it is −11.2%; add the 0.05 sports
taker fee that secondary sources report for 2026 and it is −7.7%. The two
venues' closes differ by 0.008 on average, so this is an assumption about
costs, not a finding about prices.

**Sensitivity** (edge ≥ 2 pts, flat, `pythag_60_sp_lu_bp`):

| Variant | n games | n bets | ROI | 95% CI |
|---|---|---|---|---|
| Kalshi, as quoted (headline) | 756 | 405 | -11.6% | (-22.3%, -1.3%) |
| Kalshi, all games it priced | 876 | 478 | -11.9% | (-21.7%, -2.6%) |
| Kalshi, fee waived (a maker fill) | 756 | 405 | -7.2% | (-17.9%, +3.1%) |
| Kalshi, frictionless (fill at the close) | 756 | 460 | -3.2% | (-13.3%, +7.0%) |
| Polymarket, 1¢ half-spread (headline) | 756 | 342 | -5.1% | (-16.6%, +5.7%) |
| Polymarket, 2¢ half-spread | 756 | 249 | -11.2% | (-24.4%, +1.8%) |
| Polymarket, 1¢ + 0.05 taker fee | 756 | 342 | -7.7% | (-19.2%, +3.0%) |

The 876-game Kalshi-only set (the venue priced 120 games Polymarket's archive
does not reach) gives the same answer as the 756-game common set, so the
headline is not an artifact of the intersection.

## What would have to be true for ROI to turn positive

Concretely, at Kalshi's prices: a bet at the mean cost of 47¢ pays a 2¢ fee,
so the model's probability has to be **about 2 points above the ask**, i.e.
roughly **2.5 points above the mid**, *and be right about it*, before a dollar
comes back. That is a statement about calibrated edge, not about disagreement:
we already disagree by 4.6 points on average and it is worth less than nothing.

In Brier terms the bar is arithmetic. If a model's disagreement with the price
were genuine — if its probability were the true one — the market's excess
Brier over it would be the mean squared disagreement on the games it bets.
That is 0.00314 on 405 of 756 games, or **0.0017 of Brier over the full set**
that the model *should* be winning by. It is instead **losing** by 0.0030. The
gap to close is therefore about **0.0047 of Brier** — the 0.0030 that station
E still concedes to the exchanges, plus the 0.0017 its own selectivity is
implicitly claiming. Park, weather, rest, travel and a better pitcher model
are the candidates
([market-benchmark-2026.md](market-benchmark-2026.md)); each of the last three
terms we added was worth 0.0003 or less, so this is several stations of work,
not one more feature.

Two cheaper routes exist and are worth naming. **Be a maker rather than a
taker** — resting a limit order inside the spread rather than crossing it is
worth 8.4 points of ROI here (the fee plus the spread), which is larger than
any edge we have found, though it trades adverse selection for it and cannot
be simulated from close data alone. And **find a less efficient contract**:
props and mid-liquidity markets, where the price does not already contain
Steamer, ZiPS and the sharps, which is the roadmap's stated shortest path to
money.

## Why this instrument and not Brier

Brier stays the gate (architecture.md §3): money never decides what is true,
because a model fit to the market's mistakes learns things that vanish when
the market adapts. But money is the final exam, and this table is why. A
proper scoring rule averages over every game, including the overwhelming
majority where we and the market agree and nothing is at stake. Trading
averages over the *selected* games where we disagree — a subset chosen by the
model's own errors, which is the worst-behaved subset there is. This table
shows a model that is better than `pythag_60` on Brier and worse than it in
dollars, and a model with a genuinely positive CLV proxy on every bet it
placed and a −11.6% return. Neither of those facts is visible in a Brier
score, and both of them are what actually happens to a bankroll. Money adds
the hurdle, the selectivity and the sizing; a station that clears Brier and
fails here has not earned a dollar of exposure.

## Caveats

- **The fill assumption is optimistic and deliberately so.** We assume we
  could have traded any size at the closing quote, at the top of book, with no
  impact and no queue. Real execution pays for depth and — for a price the
  market only reached on news we did not have — is exactly where the fill would
  not have been there. A positive ROI here would be *necessary*, not
  sufficient; the negative ROI here is decisive.
- **The CLV column is a proxy and is positive by construction.** With only a
  closing price there is no later price to compare against, so it reports how
  far the model sat from the close on the games it bet, signed toward the side
  taken — and the trade rule only fires when that quantity exceeds the spread.
  It is a check on how much disagreement each model is monetising and a
  placeholder for the real thing: entries at an earlier snapshot scored against
  the close, which `market-snapshot.yml` began archiving 3×/day on 2026-09-02.
- **Fees are second-hand.** See the fee section: the 0.07 constant could not be
  read first-party from this environment. It is a flag.
- **Polymarket's spread is assumed, not observed.** The backfill recovers a
  mid; the 1¢ default is a guess informed by Kalshi's observed 1¢ book. The
  sensitivity table shows what 2¢ does.
- **One venue-half-season, no vig-free reference.** 756 games from 2026-07-04
  to 09-02, no April–June (no exchange history) and no sportsbook close
  (Pinnacle archiving started 2026-09-02). A 405-bet book has a ±10-point ROI
  CI: this instrument can detect a catastrophe, not a 2% edge.
- **Settlement is from the exchange's own result field**, cross-checked
  against the schedule's winner; the join refuses to run if they disagree
  (they do not on any of these games).
- **No position limits, no bankroll constraint, no correlation.** Games on the
  same night settle independently here; in size they would not, and Kalshi's
  per-market position limits would bind before the stakes above did.
