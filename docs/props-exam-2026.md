# The Props Exam — 2026 player props priced against Kalshi

Station A scored in dollars against station M. The moneyline exam
([money-exam-2026.md](money-exam-2026.md)) ended on a flat finding — *every*
per-game model loses money at every threshold on both venues once the fee and
the spread are charged — and named the only two ways out: a better model, or
**a less efficient contract**. This is the second one.

A player prop is the cleanest less-efficient contract we can reach. It is
priced off exactly the component rates station A models; the architecture's
north star says so in as many words ("a K% model that beats the strikeout-prop
price monetizes station A on its own"); and it is where prediction-market
liquidity is thinnest, so if there is a soft price anywhere on Kalshi it is
here.

Produced by:

```
python scripts/backfill_prop_closes.py --season 2026     # closes + hourly candles
python scripts/props_exam.py --markdown                  # the taker tables
python scripts/props_exam.py --matchup on --maker --markdown   # Sept 3, below
```

The archive lives in `data/market/prop_closes_2026.parquet` and
`data/market/kalshi_prop_candles_2026.parquet`, both committed. It was
gitignored once, did not survive a container restart, and cost a full re-fetch
of tens of thousands of requests; the hourly candles cannot be re-fetched at
all once Kalshi ages the markets out.

**Headline: props are a much softer contract than moneylines, and it is still
not enough — because on props the fee is the whole loss.** The same model
family that returned **−11.6%** on Kalshi moneylines returns **−5.7%** here
(30,075 bets at edge ≥ 2 pts, 95% CI −8.3% to −3.1%), and 5.1 of those 5.7
points are the taker fee: waive it, as a resting limit order would, and the
result is **−0.6% (−3.2%, +2.0%)** — indistinguishable from zero, and equally
indistinguishable from a league-average-rate control that knows nothing about
which player it is pricing. The model plus the spread costs 0.7 points on
props against 7.2 on moneylines. That is a real and large difference in
contract efficiency. It buys a break-even, not an edge.

## Coverage

Kalshi's MLB prop series begin **2026-06-27**. This archive covers the last
four weeks of that history in full:

| | |
|---|---|
| Settled contracts archived | **66,896** (every one that traded) |
| Games / players | 397 / 639 |
| Dates | 2026-08-04 … 2026-09-02 |
| Player ids resolved | **99.0%** (687 rows unresolved) |
| Settled yes/no | 99.0% — the other 1.0% settle `scalar` (a scratch or a void) and are dropped |
| Median close | **15 minutes** before first pitch |
| Median book at the close | **1¢**, the same as the moneyline |

| Stat | contracts | games | players | lines | over rate | median pre-pitch volume |
|---|---|---|---|---|---|---|
| total bases | 19,617 | 397 | 456 | 7 | .265 | 8 |
| hits | 18,639 | 397 | 459 | 4 | .326 | 20 |
| home runs | 10,552 | 397 | 460 | 3 | .077 | 1,318 |
| RBI | 8,631 | 397 | 446 | 3 | .321 | 4 |
| strikeouts | 5,501 | 396 | 179 | 14 | .427 | 2,269 |
| stolen bases | 3,203 | 395 | 367 | 1 | .124 | 58 |
| outs recorded | 753 | 396 | 178 | 13 | .481 | 4,455 |

**53,119** of those are priced — the four stats the component table supports —
and 53,046 of those settled yes or no and are scored. The live archive is
wider than the backfill: one snapshot on 2026-09-02 carried **4,387 open prop
markets** across 263 players at a **99.5%** id-resolution rate, and the
snapshot job now writes them three times a day.

### The Sept 3 rebuild, and whether the headline reproduced

The archive above was gitignored and did not survive a container restart, so it
was rebuilt from Kalshi on Sept 3 and committed. The rebuild is **78,134
settled contracts over 457 games and 667 players, 2026-07-31 → 09-02** — the
same four weeks plus three days at the front and 60 more games, because the
listing pass ran a day later and Kalshi had settled more markets since. 99.0%
of names resolve. The listing found **149,372** settled contracts that traded
across the seven series; the fetch walks newest first at one candlestick
request per contract and was stopped at 80,000, so the remaining ~70,000 are
June and July and are still available to `--start` behind the same checkpoint.

**Restricted to the original window (2026-08-04 → 09-02) the headline
reproduces.** 54,310 scored contracts against 53,046 — the 2.4% difference is
the seven extra games:

| | published | rebuild, same window |
|---|---|---|
| scored contracts / games | 53,046 / 396 | 54,310 / 403 |
| Brier — ours | 0.15793 | 0.15759 |
| Brier — market | 0.15289 | 0.15247 |
| Brier — league-rate control | 0.15944 | 0.15901 |
| ROI, flat, edge ≥ 2 pts | −5.7% (−8.3%, −3.1%) | −5.9% (−8.6%, −3.2%) |
| n bets | 30,075 | 30,818 |
| random-edge control | −8.2% | −6.5% |

Every model column lands within 0.0004 of Brier and the ROI within 0.2 points.
The random-edge control moves the most (−8.2% → −6.5%), which is what a control
built from a fixed noise seed on a slightly different row set is supposed to
do — it is a null, not an estimate, and the point of it is the sign.

The two sections dated Sept 3 below are scored on the **whole** rebuilt window
(2026-07-31 → 09-02, 61,738 priced and settled contracts), not on the four
weeks the tables above use.

## Method

**The archive.** Kalshi lists seven MLB prop series — `KXMLBHR`, `KXMLBKS`,
`KXMLBHIT`, `KXMLBTB`, `KXMLBRBI`, `KXMLBSB`, `KXMLBOUTS` — one contract per
player per strike per game ("LaMonte Wade Jr.: 3+ hits?"). They reuse the game
event ticker, so the date and the matchup decode with the same parser the
moneyline uses; the player is named only in the title (`custom_strike` carries
a vendor UUID, not an MLBAM id), so `src/market/players.py` resolves the
printed name against the season roster list, folding accents and punctuation
and refusing to guess when a name maps to two ids. Polymarket carries the same
shape under `sportsMarketType` `baseball_player_home_runs` / `_strikeouts`,
far fewer of them, and is archived but not scored here.

**The close.** The last hourly candle ending at or before first pitch, as in
the moneyline backfill, with one difference that matters. A game market trades
every hour and its last print is current; a prop's last print can be hours
stale and sit outside a book that has since moved. So `p_over_close` is the
last trade **when it lies inside the closing quote** and the midpoint of that
quote otherwise. Without that rule the `market` control — which sits inside
the spread by construction and must never trade — placed bets, and a
random-edge control returned +13%: both were trading against prices nobody was
showing.

**Our probability.** Walk-forward, rebuilt from games strictly *before* each
game's own date:

* **Hitters.** Marcel-with-the-partial-season component rates (K, BB+HBP, HR
  per PA, ISO per AB, BABIP per BIP — `src/sim/lineups.py`, the arm
  [backtest-baselines.md](backtest-baselines.md) shows winning K% and HR
  intra-season), turned into per-plate-appearance event probabilities, times
  the plate appearances his **posted lineup slot** gets (4.6 leadoff down to
  3.7 ninth). Then P(count ≥ line) as a Binomial on that many trials —
  interpolated across the fractional expectation rather than rounded — and a
  Poisson on expected bases for total bases, which is not a count of successes
  in a fixed number of trials.
* **Pitchers.** Marcel K per batter faced (`src/sim/starters.py`) over the
  batters he is expected to face: 23, the league's average start, and
  optionally his own per-start number to date (`--pitcher-bf own`).

**What we do not price.** Three of the seven series are archived and left
unpriced, because the component table does not support them and a model we do
not believe would make the Brier look better than it is: **RBI** is a function
of who is on base in front of the hitter; **stolen bases** need a rate the
table does not carry; **outs recorded** is a manager's decision about pitch
count, leverage and score far more than a rate.

**The money exam.** The same `src/market/pnl.py` the moneylines went through:
buy YES at the ask when the model is above it, buy NO at 1 − bid when it is
below the bid, nothing inside the spread; Kalshi's taker fee
`round_up_to_cent(0.07 · C · P · (1 − P))`; flat and quarter-Kelly stakes;
2,000-draw bootstrap CIs. Two changes for props. First the bootstrap is
**clustered by game** — a hitter's 1+, 2+ and 3+ hits are one afternoon's at
bats, and a row-wise resample would call them three independent observations
and report a CI far tighter than the data supports. Second the controls are
the ones a prop needs: `market` (the close itself, anchored at exactly zero),
`random_edge` (the close plus 3 pts of noise — what an ROI looks like when the
edge is fake), and `league_rate` (the same contract priced with the *league's*
rates instead of the player's, which says how much of any skill is the player
rather than the shape of the distribution).

## Brier per stat — 53,046 settled contracts, 396 games

| Stat | n | games | players | over rate | **ours** | market | league-rate |
|---|---|---|---|---|---|---|---|
| hits | 18,192 | 395 | 458 | .326 | 0.16497 | **0.16391** | 0.16567 |
| home runs | 10,270 | 395 | 459 | .077 | 0.06823 | **0.06774** | 0.06884 |
| strikeouts | 5,382 | 395 | 177 | .426 | 0.16718 | **0.15132** | 0.17934 |
| total bases | 19,202 | 395 | 455 | .265 | 0.19664 | **0.18843** | 0.19642 |
| **all** | **53,046** | 396 | 636 | .266 | 0.15793 | **0.15289** | 0.15944 |

Read it in three parts.

1. **The market wins every stat.** It is closest on the two counting props
   built straight out of a per-PA rate — hits (.0011 of Brier) and home runs
   (.0005) — and furthest ahead on strikeouts (.0159) and total bases (.0082).
2. **Our player rates barely beat league-average rates.** The `league_rate`
   control prices the identical contract with the league's own component rates
   and the same expected plate appearances, so the gap between the two columns
   is the entire contribution of knowing *which hitter this is*: **0.0015** of
   Brier overall, and on total bases it is **negative** — the league's rates
   price the TB line better than the player's own do. Nearly all of what looks
   like skill in the "ours" column is the shape of the distribution, not the
   player.
3. **Strikeouts are where the model is worst and the price is best**, which is
   the opposite of the roadmap's expectation. Most of it is the workload
   assumption rather than the rate: giving every starter the league-average
   23 batters faced costs 0.0024 of Brier against using his own per-start
   number (0.16718 → 0.16475, `--pitcher-bf own`), and the remaining 0.0134 is
   the market knowing things about tonight's start that a season K/BF does
   not.

## Money — Kalshi, flat 1u, edge ≥ 2 pts

| Model | n bets | hit | staked | return | ROI | ROI 95% CI | mean edge | CLV | max DD | fees |
|---|---|---|---|---|---|---|---|---|---|---|
| marcel_partial | 30,075 | 0.509 | 30,075u | −1,705.58u | **−5.7%** | (−8.3%, −3.1%) | 6.61 pt | +7.14 pt | 1,843u | 1,514u |
| league_rate (control) | 32,933 | 0.508 | 32,933u | −1,959.45u | −5.9% | (−8.7%, −3.3%) | 7.40 pt | +7.98 pt | 2,118u | 1,714u |
| random_edge (control) | 19,761 | 0.432 | 19,761u | −1,611.88u | −8.2% | (−12.4%, −3.9%) | 3.65 pt | +4.12 pt | 1,674u | 1,894u |
| market (control) | 0 | — | 0u | +0.00u | +0.0% | — | — | — | 0u | 0u |

**ROI by threshold (flat 1u).** No threshold rescues it, and the curve slopes
the wrong way — being *more* selective is worse, which is what a model with no
real edge does when it charges a fee for each unit of selectivity.

| Model | n ≥0pt | n ≥2pt | n ≥4pt | n ≥6pt | ROI ≥0pt | ROI ≥2pt | ROI ≥4pt | ROI ≥6pt |
|---|---|---|---|---|---|---|---|---|
| marcel_partial | 47,284 | 30,075 | 20,031 | 12,998 | −4.1% | −5.7% | −6.6% | −6.5% |
| league_rate (control) | 48,686 | 32,933 | 23,166 | 15,431 | −3.9% | −5.9% | −7.1% | −6.6% |
| random_edge (control) | 43,416 | 19,761 | 6,391 | 1,464 | −7.5% | −8.2% | −7.1% | +0.2% |
| market (control) | 0 | 0 | 0 | 0 | +0.0% | +0.0% | +0.0% | +0.0% |

**Per stat** (flat 1u, edge ≥ 2 pts). Every stat loses; hits loses least and is
the only one whose CI touches zero.

| Stat | Model | n bets | hit | ROI | ROI 95% CI |
|---|---|---|---|---|---|
| hits | marcel_partial | 8,308 | 0.437 | −3.2% | (−7.3%, +1.3%) |
| | league_rate | 8,691 | 0.438 | −3.3% | (−7.5%, +1.3%) |
| | random_edge | 6,628 | 0.446 | −5.4% | (−11.4%, +1.4%) |
| home runs | marcel_partial | 2,633 | 0.227 | −12.6% | (−24.0%, −1.0%) |
| | league_rate | 4,360 | 0.331 | −14.8% | (−22.8%, −6.5%) |
| | random_edge | 3,695 | 0.315 | −39.3% | (−50.7%, −25.8%) |
| strikeouts | marcel_partial | 4,169 | 0.531 | −10.6% | (−17.6%, −3.7%) |
| | league_rate | 4,616 | 0.496 | −12.4% | (−19.7%, −5.2%) |
| | random_edge | 2,168 | 0.478 | −14.9% | (−21.5%, −8.2%) |
| total bases | marcel_partial | 14,965 | 0.592 | −4.5% | (−6.6%, −2.3%) |
| | league_rate | 15,266 | 0.601 | −3.0% | (−5.1%, −0.8%) |
| | random_edge | 7,185 | 0.477 | +7.4% | (+2.0%, +12.3%) |

## Sensitivity (edge ≥ 2 pts, flat)

| Variant | n bets | ROI | 95% CI | random-edge control |
|---|---|---|---|---|
| **As quoted (headline)** | 30,075 | **−5.7%** | (−8.3%, −3.1%) | −8.2% |
| Starter's own batters faced | 30,010 | −5.8% | (−8.4%, −3.1%) | −8.2% |
| **Fee waived (a maker fill)** | 30,075 | **−0.6%** | (−3.2%, +2.0%) | +1.4% |
| Frictionless (fill at the close) | 33,455 | −1.6% | (−4.2%, +1.1%) | +2.3% |
| Pre-pitch volume ≥ 25 contracts | 16,961 | −8.0% | (−11.3%, −4.7%) | −11.8% |

Two rows carry the argument.

**The fee is the loss.** Waiving it moves ROI from −5.7% to −0.6%, so
**5.1 of the 5.7 points are the taker fee** and everything else — the spread
we cross and the model's own errors — costs 0.7 points. On the moneyline the
same decomposition was 4.4 points of fee out of 11.6. The prop *price* is
soft; the prop *fee* is not, and because prop prices sit far from 50¢ the fee
is a larger fraction of a cheap contract than of an even-money one. But that
row is a ceiling, not a result: with the fee waived the random-edge control
also returns +1.4%, so a null with no information is indistinguishable from
our model there.

**Liquidity does not help.** Restricting to contracts that traded at least 25
times *before first pitch* makes the result worse (−8.0%): the contracts
people actually trade are the sharper ones. An earlier version of this filter
used the contract's whole-life volume and reported +20.6% ROI on hits — until
the random-edge control on the same subset returned +14.7%. Whole-life volume
includes in-play trading, and an in-play market trades heaviest once the
outcome is live, so filtering on it selects contracts that went the over's
way. It is recorded here because it is exactly the kind of leak a money table
is supposed to catch and a Brier score never would.

## Are props less efficient than moneylines?

**Yes, and by a lot — but the softness lands entirely in the part of the loss
that the fee then takes back, so the practical answer is still no.** The
comparison is apples to apples: same venue, same trade rule, same fee formula,
same fill assumption, overlapping dates. On Kalshi moneylines the full station-E
stack returned **−11.6%** on 405 bets at edge ≥ 2 pts, against a −6.2%
random-edge control; on props the station-A rate model returns **−5.7%** on
30,075 bets, against a −8.2% random-edge control. Two things changed and they
point opposite ways. First, we now beat the noise control instead of losing to
it (−5.7% vs −8.2%, where the moneyline model *lost* to its control, −11.6% vs
−6.2%) — that is a genuine signal: the prop price does not already contain
everything our rates know, and the model is at least trading in the right
direction. Second, the loss net of the fee shrank from 7.2 points to 0.7,
which is the efficiency gap the roadmap was betting on and it is real. What
did not change is that the market still wins on truth: it holds 0.0050 of
Brier over us on props against 0.0030 on moneylines — measured against a
*relatively* higher bar, since prop Brier is smaller throughout — and our
player-specific rates beat league-average rates by only 0.0015 of that.
The 30,075-bet sample is 74× the moneyline book and the CI is four times
tighter, so this is not an underpowered "we might be close": at these prices,
with a taker fee, props lose, and the interval excludes zero. The soft
contract exists and we found it; what it says is that the remaining route to
money on Kalshi props is **execution, not a better rate model** — resting a
limit order rather than crossing the spread is worth the entire 5.1 points,
which is larger than any edge measured here, and it is the one thing a close-
data backtest cannot simulate.

> **That last sentence was wrong, and Sept 3 measured it.** A close-data
> backtest cannot simulate a resting order; an hourly *candle* archive can, and
> the prop candles now exist. Quoting instead of crossing does not recover the
> fee — it returns **−7.2%** where crossing the same contracts returns −3.2% —
> because most of the orders are not maker orders at all and the ones that are
> get adversely selected. See
> [resting orders on props](#sept-3-2026--resting-orders-on-props). The
> paragraph above is left as written because it is what the evidence then
> supported, and because the shape of the error — reading a counterfactual row
> with the fee deleted as if it were a strategy — is the useful part.

## Sept 3, 2026 — the opposing pitcher in the hitter's price

The caveat this section answers was the last one in the list above: *a prop is
a matchup and this model prices only one side of it.* It now prices both, and
the term is on by default (`props.MATCHUP_DEFAULT`).

**What it is.** A log5 matchup factor per component, applied to the hitter's
own Marcel-with-partial rate:

```
rate = rate_hitter · rate_pitcher / rate_league
```

for K, BB+HBP and HR per plate appearance, ISO per at bat and BABIP per ball
in play — the five rates hits and total bases are built out of. It is an
identity, asserted rather than fitted, which is what [methods.md](methods.md)
§4 says to do with log5. Three details carry the work:

* **The pitcher is the pre-game *probable* starter**, from the same Stats API
  field station E reads, never the man who eventually took the mound.
* **A starter does not face the whole game.** He covers
  `starters.expected_starter_ip` of nine innings — the workload split station E
  already prices relief innings with — and the opposing club's own relief staff
  to date covers the rest, so the factor a hitter sees is
  `share · f(starter) + (1 − share) · f(pen)`. On this archive the mean
  expected start is 5.2 innings, so the pen carries 42% of it.
* **Pitcher strikeout props get the mirror**: the opposing club's *posted*
  card, plate-appearance weighted by lineup slot, its own recent cards where
  tonight's is not archived, the league — a factor of exactly 1.0 — where
  neither is.

A pitcher's rates *allowed* are computed by `lineups.marcel_rates`, the hitter
estimator pointed at the other id on the plate appearance, in the hitter's own
columns. That needed doubles and triples allowed, which were already in the
cached Stats API responses, so the whole term costs no new fetch.

**The one free constant is chosen out of sample.** `WEIGHT` scales the
adjustment — 1.0 is log5 exactly, 0.0 is the old price — and it is chosen on
the **first half of the window by date** (2026-07-31 → 08-16, 31,315 settled
contracts) and scored on the **second** (08-17 → 09-02, 30,423). The first
half prefers the boundary:

| weight | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| first-half Brier | 0.158812 | 0.158536 | 0.158313 | 0.158142 | **0.158026** |

That is a boundary solution and should be read as one: the first half is not
asking for a particular strength, it is asking for the identity undiluted, and
nothing on the grid would have let it ask for more.

### Brier per stat — 61,738 settled contracts, 455 games

| half | stat | n | games | over rate | current | **matchup** | market | league-rate |
|---|---|---|---|---|---|---|---|---|
| first | hits | 11,200 | 231 | .306 | 0.15777 | 0.15752 | **0.15639** | 0.15845 |
| first | home runs | 6,119 | 231 | .080 | 0.07053 | 0.07041 | **0.06983** | 0.07101 |
| first | strikeouts | 3,171 | 229 | .429 | 0.16978 | 0.16414 | **0.15633** | 0.18511 |
| first | total bases | 10,825 | 231 | .280 | 0.20658 | 0.20628 | **0.19694** | 0.20629 |
| **first** | **all** | **31,315** | 231 | .265 | 0.15881 | 0.15803 | **0.15349** | 0.16060 |
| second | hits | 9,888 | 224 | .347 | 0.17172 | 0.17111 | **0.17099** | 0.17249 |
| second | home runs | 5,773 | 224 | .076 | 0.06694 | 0.06681 | **0.06665** | 0.06759 |
| second | strikeouts | 2,985 | 223 | .420 | 0.16543 | 0.15975 | **0.14721** | 0.17239 |
| second | total bases | 11,777 | 224 | .243 | 0.18251 | 0.18205 | **0.17554** | 0.18249 |
| **second** | **all** | **30,423** | 224 | .262 | 0.15539 | 0.15444 | **0.15062** | 0.15644 |
| all | hits | 21,088 | 455 | .325 | 0.16431 | 0.16389 | **0.16324** | 0.16503 |
| all | home runs | 11,892 | 455 | .078 | 0.06878 | 0.06866 | **0.06829** | 0.06935 |
| all | strikeouts | 6,156 | 452 | .425 | 0.16767 | 0.16201 | **0.15191** | 0.17894 |
| all | total bases | 22,602 | 455 | .261 | 0.19403 | 0.19366 | **0.18579** | 0.19389 |
| **all** | **all** | **61,738** | 455 | .264 | 0.15713 | 0.15626 | **0.15207** | 0.15855 |

**Paired per contract, matchup − current.** Two arms priced on the same
contract on the same afternoon are paired by construction, so the difference of
the two Brier scores is a per-contract quantity; the standard error clusters on
the game, for the same reason the money bootstrap does. Negative means the
matchup price is better.

| half | stat | n | Brier(matchup) − Brier(current) | se | t |
|---|---|---|---|---|---|
| first | hits | 11,200 | −0.000250 | 0.000160 | −1.57 |
| first | home runs | 6,119 | −0.000116 | 0.000044 | −2.66 |
| first | strikeouts | 3,171 | −0.005641 | 0.001785 | −3.16 |
| first | total bases | 10,825 | −0.000299 | 0.000196 | −1.53 |
| **first** | **all** | **31,315** | **−0.000787** | 0.000208 | **−3.78** |
| second | hits | 9,888 | −0.000607 | 0.000180 | −3.37 |
| second | home runs | 5,773 | −0.000123 | 0.000045 | −2.71 |
| second | strikeouts | 2,985 | −0.005681 | 0.001814 | −3.13 |
| second | total bases | 11,777 | −0.000453 | 0.000196 | −2.31 |
| **second** | **all** | **30,423** | **−0.000954** | 0.000215 | **−4.43** |
| all | all | 61,738 | −0.000869 | 0.000150 | −5.80 |

### Did the pitcher close the gap?

**A fifth of it, and it is real.** On the scored half the market held 0.00477
of Brier over the current price; it holds 0.00382 over the matchup price. The
opposing pitcher is worth **0.00095**, it clears the gate — the sign is the
same on both halves, on every stat, and the second-half t is −4.4 with the
error clustered by game — and it leaves four fifths of the gap where it was.
So the term goes in (`props.MATCHUP_DEFAULT = True`) and the headline finding
of this document does not move: the market is still ahead, by about four times
what knowing the opposing pitcher was worth.

Three things are worth reading off the table.

1. **Nearly all of it is strikeouts.** The pitcher's own prop is the one where
   the matchup term has a lineup to work with and the rate it adjusts is the
   whole contract: 0.0057 of Brier on the scored half, six times the pooled
   gain, on 5% of the contracts. The K arm's gap to the market falls from
   0.0182 to 0.0125. On the hitter side the term moves hits by 0.0006, total
   bases by 0.0005 and home runs by 0.0001 — the right sign every time, and
   small, because a hitter's four or five plate appearances against one starter
   are a much smaller share of his night than a starter's whole start is of
   his.
2. **The league-rate control is the thing it beats most clearly.** Knowing
   *which hitter this is* was worth 0.0015 of Brier; the pooled matchup gain of
   0.0009 is more than half as much again, from a term that knows nothing about
   the hitter at all. Read against the headline that is the story: the price
   was missing about as much matchup as it was missing player.
3. **Total bases is still the weak arm and the pitcher does not fix it.** The
   league-rate control still ties us there (0.18249 vs 0.18205 on the scored
   half). The Poisson is the problem, not the opponent.

### Where the residual is

The market's remaining 0.0038 is, on this evidence, not the opposing pitcher
and not the player's own rates. Four things it can still be, in order of how
much of the archive they touch:

* **The distribution, on total bases.** TB is 37% of the priced contracts and
  sits 0.0065 behind the market on the scored half, against a pooled 0.0038. Bases arrive in lumps
  and a Poisson on expected bases is under-dispersed; this is the one place
  where a better *shape* rather than a better *rate* is the obvious fix.
* **Plate appearances.** Every hitter is given his slot's structural 4.6-to-3.7
  and nothing else — not the game's expected length, not the score, not a
  pinch-hitter. The starter's own workload is now in the strikeout price
  (`expected_starter_ip`); the hitter's is not.
* **Tonight.** Weather, park, catcher, umpire, bullpen state, a hitter playing
  hurt. None of it is in any arm here and all of it is in the price.
* **The rate model itself**, which the league-rate control says is worth only
  0.0015 to begin with.

### Caveats specific to the matchup term

- **The probables field cannot tell us it was a probable.** For a date already
  past, the Stats API serves the pitcher who actually started, so the archive
  cannot distinguish the announcement from the outcome. That is the same
  assumption station E's starter term runs on and the same defence applies —
  the exchange's close is a median 15 minutes before first pitch and knew about
  scratches too — but it means this is not a simulation of pricing the morning
  before. What *is* tested is that nothing later than the announcement reaches
  the price: the rates are cut strictly before the game date, and a unit test
  appends the line the starter actually threw, with the game's own date on it,
  and asserts the price does not move.
- **ISO allowed has no published stabilization point**, so it is regressed at
  the home-run point (1,300 batters faced, doubled), which is the heaviest on
  the board. That is deliberate — most of a pitcher's extra-base suppression is
  park and defence — but it is a choice, not a citation.
- **A hitter's plate appearances are split like innings.** The starter's share
  is `expected_starter_ip / 9` for every slot, when in fact the top of the
  order faces the starter for a larger share of its plate appearances than the
  bottom does. The correction is small and the direction is known.
- **The pen is a club aggregate, not tonight's arms.** Season-to-date relief
  innings for the club, regressed to the league. Who is actually available is
  knowable pre-game (`src/sim/reliever_usage.py` computes it for station E) and
  is not used here; station E measured that refinement as worth nothing on
  moneylines.

## Sept 3, 2026 — resting orders on props

The document's conclusion was that "the remaining route to money on Kalshi
props is **execution, not a better rate model** — resting a limit order rather
than crossing the spread is worth the entire 5.1 points". That was an argument
from the fee-waived row, which is a *ceiling*, not a strategy. This section
runs the strategy. **It does not reach zero. It is worse than crossing, and the
reason is adverse selection rather than fees.**

### Method

Identical to the moneyline maker exam
([money-exam-2026.md](money-exam-2026.md#maker-side--quoting-instead-of-crossing)),
one level down. For one contract and one model, with margin `m`: above the
close, post a YES bid at `P_model − m`; below it, a NO bid at
`(1 − P_model) − m`; equality quotes nothing. The bid is floored to the cent
grid. The order is live from T−24h to first pitch, one contract per market,
cancelled unfilled at first pitch, and it fills in the first archived hour
whose traded low reached the bid on non-zero volume — mirrored on the NO side,
at our own limit, never at a better price. Maker fee `0.0175 · P · (1 − P)`
(`--kalshi-maker-fee-rate`), not rounded to the cent. The control is the
model's own signed disagreements dealt to the wrong contracts. The bootstrap is
**clustered by game**, which the moneyline exam did not need and a prop book
does: a hitter's 1+, 2+ and 3+ hits are one afternoon's at bats. `m` is chosen
on the first half of the window by cents per posted contract and scored on the
second.

**The archive.** `data/market/kalshi_prop_candles_2026.parquet`: **516,666
hourly candles over 78,134 prop markets**, 2.7 MB, collected in the same pass
as the closes because Kalshi serves them only while a market is young. It is
much shallower than the moneyline archive and that matters: a game market is
open for the whole 24 hours, a prop market for a **median of 3 hours** (mean
6.6, max 24). **33% of archived prop hours traded**, against 97.9% of game
hours. A resting prop order therefore has, typically, three chances.

### Fill rate by margin — second half, one contract per posted order

| Model | m=0.00 | m=0.01 | m=0.02 | m=0.03 | m=0.05 |
|---|---|---|---|---|---|
| current | 0.720 | 0.609 | 0.501 | 0.410 | 0.279 |
| matchup | 0.720 | 0.602 | 0.487 | 0.393 | 0.259 |
| current (shuffled control) | 0.714 | 0.625 | 0.521 | 0.436 | 0.298 |
| matchup (shuffled control) | 0.714 | 0.618 | 0.513 | 0.419 | 0.278 |

### First half — ¢ per posted contract, where `m` is chosen

| Model | m=0.00 | m=0.01 | m=0.02 | m=0.03 | m=0.05 |
|---|---|---|---|---|---|
| current | −3.06¢ | −2.46¢ | −2.05¢ | −1.68¢ | **−1.06¢** |
| matchup | −2.69¢ | −2.14¢ | −1.70¢ | −1.42¢ | **−0.81¢** |
| current (shuffled control) | −3.13¢ | −2.43¢ | −1.88¢ | −1.57¢ | −0.83¢ |
| matchup (shuffled control) | −2.79¢ | −2.13¢ | −1.55¢ | −1.20¢ | −0.74¢ |

Every arm's training half prefers **m = 0.05**, the widest margin on the grid —
the same boundary solution the moneyline exam found, and it means the same
thing: what the first half wants is not a margin, it is *less trading*.

### Second half — the scored one (flat, one contract per order, 30,423 posted)

| Model | m | posted | fill | crossed | ¢/posted | ¢/filled | hit | ROI | ROI 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| current | 0.00 | 30,423 | 0.720 | 0.758 | −2.96¢ | −4.11¢ | .546 | −7.0% | (−8.1%, −5.9%) |
| current | 0.01 | 30,423 | 0.609 | 0.623 | −2.40¢ | −3.95¢ | .533 | −6.9% | (−8.3%, −5.6%) |
| current | 0.02 | 30,423 | 0.501 | 0.515 | −2.04¢ | −4.08¢ | .525 | −7.3% | (−8.8%, −5.7%) |
| current | 0.03 | 30,423 | 0.410 | 0.426 | −1.75¢ | −4.27¢ | .525 | −7.6% | (−9.3%, −5.8%) |
| **current** | **0.05** | 30,423 | 0.279 | 0.274 | **−1.15¢** | −4.12¢ | .531 | **−7.2%** | (−9.4%, −5.1%) |
| matchup | 0.00 | 30,423 | 0.720 | 0.741 | −2.56¢ | −3.55¢ | .561 | −6.0% | (−7.1%, −4.8%) |
| matchup | 0.01 | 30,423 | 0.602 | 0.599 | −2.09¢ | −3.46¢ | .552 | −5.9% | (−7.3%, −4.7%) |
| matchup | 0.02 | 30,423 | 0.487 | 0.490 | −1.67¢ | −3.43¢ | .548 | −5.9% | (−7.4%, −4.5%) |
| matchup | 0.03 | 30,423 | 0.393 | 0.397 | −1.44¢ | −3.66¢ | .551 | −6.3% | (−7.9%, −4.6%) |
| **matchup** | **0.05** | 30,423 | 0.259 | 0.247 | **−0.96¢** | −3.71¢ | .554 | **−6.3%** | (−8.3%, −4.3%) |
| current (shuffled) | 0.05 | 29,920 | 0.298 | 0.270 | −0.85¢ | −2.85¢ | .415 | −6.5% | (−8.4%, −4.5%) |
| matchup (shuffled) | 0.05 | 29,923 | 0.278 | 0.245 | −0.69¢ | −2.48¢ | .424 | −5.5% | (−7.4%, −3.7%) |

### The comparison the section exists for

All four rows below are the **same 30,423 second-half contracts**, so they can
be read against each other.

| How the trade is done | current | matchup |
|---|---|---|
| Taker, cross the close, edge ≥ 0 pts, fee charged | −3.2% | −3.1% |
| Taker, edge ≥ 2 pts, fee charged | −4.5% | −4.0% |
| **Taker, edge ≥ 0 pts, fee waived** (the ceiling the old conclusion rested on) | **+2.5%** | **+2.3%** |
| **Maker, rest at `P − m`, m = 0.05, maker fee charged** | **−7.2%** | **−6.3%** |

The fee-waived taker row is where the previous conclusion came from and it
still says what it said: with no fee, crossing the close is a hair above zero
(+2.5%, and the random-edge control on the same rows returns +3.2%, so the
"edge" is not one). The maker row is what actually happens when you try to earn
that fee back by quoting, and it is **ten points worse**. Quoting did not
recover the fee; it paid a different and larger cost.

**Two mechanisms, and the table separates them.**

*The order is usually not a maker order at all.* The `crossed` column counts
orders posted at or above the prevailing ask, which the exchange fills
immediately against a resting offer and charges the *taker* fee for. On
moneylines at a 5-pt margin that was 13% of orders. Here it is **27%**, and at
a zero margin **76%**, because a prop model's disagreement with the price is
large relative to the price: the mean edge on a traded prop is 6.6 points and the
median contract closes at 17. Bidding "my fair value minus five cents" for
a 12¢ home run is not providing liquidity, it is lifting the offer and paying
more than the ask to do it.

*The fills that are genuinely passive are adversely selected.* Push the margin
out until the order really is behind the market and the hit rate collapses:

| margin | 0.05 | 0.08 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|---|
| current — fill rate | 0.279 | 0.162 | 0.112 | 0.039 | 0.015 |
| current — hit rate | .531 | .499 | .473 | .406 | .337 |
| current — ROI | −7.2% | −8.4% | −8.7% | −11.6% | −12.3% |
| matchup — hit rate | .554 | .523 | .500 | .438 | .324 |
| matchup — ROI | −6.3% | −5.9% | −5.4% | −5.4% | −10.4% |

That is the signature of adverse selection and it is monotone: the further
inside our own fair value we insist on being paid, the rarer the fill and the
*worse* it is. Someone sold us that contract at that price because the contract
was worth less than that, and on a thin market three hours from first pitch —
a scratch, a card, a hitter dropped to eighth — they usually knew why. The
moneyline exam saw the opposite gradient (wider margin, smaller loss) because
there the saving was the taker fee and the spread on a liquid 50¢ contract;
here there is no comparable saving to be had, so what the gradient shows is
only the selection.

**So the answer is no.** Resting orders on props does not reach zero and does
not reach the taker's own number. The previous conclusion — that execution was
the remaining route — was drawn from a counterfactual row with the fee simply
deleted, and the row that actually simulates the execution says the deletion
was not free. What is left of that conclusion is narrower and still true: *if*
a fill can be had at the close without the taker fee, this book is about
break-even. Nothing here shows how to get one.

### Caveats — why this maker P&L is still an upper bound

- **Queue priority is assumed**, as on moneylines: any hour whose traded low
  reached our price fills us, at any volume. On a prop market that trades 33%
  of its hours, a price that only touched our level on two contracts would
  usually have left us behind the queue.
- **The granularity is one hour** and a prop market lives a median of three of
  them, so the archive frequently cannot say whether the low came before or
  after the order was posted.
- **No inventory, no requoting, no position limits, no correlation** between
  the dozens of contracts on the same afternoon — which, unlike on moneylines,
  is a large omission: a book of 30,000 orders across 224 games is not 30,000
  independent bets, and the clustered bootstrap widens the interval for that
  but does not model the exposure.
- **The archive is only as deep as the market's own life.** The rule rests the
  order for 24 hours; Kalshi's prop markets do not exist for 24 hours. A longer
  rest is not available to be tested here.

### Reproduce both Sept 3 sections

```
python scripts/backfill_prop_closes.py --season 2026        # closes + candles
python scripts/backfill_prop_closes.py --season 2026 --assemble-only

python scripts/props_exam.py --matchup off --start 2026-08-04 --end 2026-09-02 \
    --markdown                                              # the reproduction
python scripts/props_exam.py --matchup on --maker --markdown \
    --priced-out /tmp/priced.parquet                        # both sections
python scripts/props_exam.py --matchup on --fee-rate 0 --priced-in /tmp/priced.parquet \
    --thresholds 0.02 --stakings flat --markdown            # the fee-waived row
python scripts/props_exam.py --matchup on --maker --priced-in /tmp/priced.parquet \
    --maker-margins 0.05 0.08 0.10 0.15 0.20                # the adverse-selection grid
```

## Caveats

- **Four weeks, one venue, one season.** 2026-08-04 to 09-02. Kalshi's prop
  series only start 2026-06-27, so the whole universe is ten weeks old; this
  archive is the most recent four of them, and `--start` walks further back at
  ~19 candlestick requests a second.
- **Plate appearances per slot are structural, not fitted.** 4.6 leadoff down
  to 3.7 ninth, the same 1.24:1 spread `lineups.slot_pa_shares` derives from
  38 team plate appearances. A blowout, extra innings, an early exit or a
  pinch-hitter all move a hitter's real PA and none of them are modelled; the
  fractional interpolation across ⌊n⌋ and ⌈n⌉ softens the rounding error but
  not the variance.
- **A hitter pulled after the card was posted is priced as if he played.**
  The posted lineup is knowable before first pitch, which is the walk-forward
  guard; a late scratch is not. Kalshi voids those markets (`result: scalar`,
  1.0% of the archive) and they are dropped, so the exposure is a hitter who
  is *pinch-hit for*, not one who never appeared.
- **Three of the seven stats are not priced at all** — RBI, stolen bases,
  outs recorded. They are archived, so a sequencing model or a start-length
  model can be scored against them later without re-fetching anything.
- **The starter faces 23 batters.** The headline uses the league average; the
  sensitivity table shows his own per-start number, which helps the Brier
  (0.16718 → 0.16475) and not the money. Neither knows the bullpen state, the
  opposing lineup's own K rate, or the park. **The opposing lineup's K rate is
  in it since Sept 3** and is the largest single thing the matchup term buys;
  the bullpen state and the park are still not.
- **Total bases is priced as a Poisson on expected bases.** Bases arrive in
  lumps of 1, 2, 3 and 4, so the true distribution is more dispersed than
  Poisson and the high lines are systematically under-priced by this model.
  That is visible: TB is the stat where the league-rate control beats us.
- ~~**The opposing pitcher is not in the hitter's price, and the opposing
  lineup is not in the pitcher's.**~~ **Fixed and measured on Sept 3** — see
  [the matchup section](#sept-3-2026--the-opposing-pitcher-in-the-hitters-price).
  It was the largest single omission and it was worth less than it looked.
- **The fill assumption is the moneyline exam's, unchanged and optimistic**:
  one unit at the closing quote, top of book, no impact, no queue. Props are
  thinner than moneylines, so it is *more* optimistic here — the median
  contract traded 8 to 20 times before first pitch on the two biggest series.
- **Fees are second-hand.** The 0.07 constant is the published formula as
  reported by secondary sources; Kalshi's own fee page was unreachable from
  this environment. It is a flag (`--fee-rate`), and on this contract it is
  the entire finding, so it is the single number most worth verifying
  first-party.
- **Bets on the same game are correlated and the bootstrap knows it** (it
  resamples games, not rows); bets on the same *player across games* are not
  clustered, and neither is the fact that 14,965 total-bases bets and 8,308
  hits bets are largely the same hitters on the same afternoons.
