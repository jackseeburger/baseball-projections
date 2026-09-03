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
python scripts/backfill_prop_closes.py --season 2026 --start 2026-08-05 --end 2026-09-03
python scripts/props_exam.py --markdown
```

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
  opposing lineup's own K rate, or the park.
- **Total bases is priced as a Poisson on expected bases.** Bases arrive in
  lumps of 1, 2, 3 and 4, so the true distribution is more dispersed than
  Poisson and the high lines are systematically under-priced by this model.
  That is visible: TB is the stat where the league-rate control beats us.
- **The opposing pitcher is not in the hitter's price, and the opposing lineup
  is not in the pitcher's.** A prop is a matchup and this model prices only
  one side of it. That is the largest single omission, and it is most of what
  the market knows that we do not.
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
