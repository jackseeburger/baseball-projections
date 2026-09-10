# Hits props after fee, replicated on the July contracts

**BAS-93.** Layer 7. Follow-up to BAS-70's exploratory lead
(`docs/posterior-props.md`). Pre-registered 2026-09-10 14:30 UTC in the
Linear issue, before any fetch or run; this file mirrors that text.

## Why

BAS-70 ("Not pre-registered, and reported as such") found hits props
were the one after-fee positive on the board: fee-waived +9.5% / +12.1%
/ +11.4% (threshold @ 2 pts / posterior / matched) with second-half
intervals excluding zero, and +4.3% to +5.5% as quoted after the Kalshi
taker fee with intervals that include zero. It was a stat-level split of
a pooled test on 21,094 hits contracts spanning 2026-07-31 → 09-02,
flagged as a lead and not a result, "worth its own pre-registration on
the June and July contracts the archive has not yet fetched".

Those contracts exist and are still reachable. The archive starts on
07-31 only because the original candlestick fetch was stopped at 80,000
contracts (`docs/props-exam-2026.md`), not because of an API limit; the
remaining ~69,000 are July. Probed 2026-09-10 14:25 UTC from this
environment: Kalshi lists settled `KXMLBHIT` contracts from about
2026-07-05 (windows before 07-05 return nothing) and still serves hourly
candlesticks for a 2026-07-11 contract (five bars with bid, ask, price).
Candle retention is finite and unquantified, so this runs now.

This is a replication on data nobody has looked at, with every constant
frozen at its served value. It is the cleanest test the money layer has
had.

## What gets built

- **Fetch.** `scripts/backfill_prop_closes.py --season 2026 --start
  2026-07-05 --end 2026-07-31 --append`, hits series first, then the
  other six, resuming past the existing 80,000-ticker checkpoint;
  `--assemble-only --append` into the git-tracked
  `data/market/prop_closes_2026.parquet` and
  `kalshi_prop_candles_2026.parquet` (they are tracked on purpose; the
  appended archive is committed). Dedup on `(venue, market_id)`.
- **Score.** `scripts/props_exam.py --matchup on --posterior --start
  2026-07-05 --end 2026-07-30`, with **every free constant frozen at its
  served value, nothing chosen on the new window**: matchup weight 1.0,
  threshold 2 points, flat one unit, Kalshi taker rate 0.07,
  `DRAW_STREAMS=shared` (the committed archive's construction). No half
  split: the whole July window is out of sample. The rates behind every
  price are `marcel_partial + matchup` as served, built from data
  strictly before each game date.
- **Primary set:** hits contracts. Secondary: HR, TB, strikeouts
  (negative control), and the league-rate control `p_league` on the same
  contracts.

## Pre-registered predictions

1. **(The lead replicates.)** Hits, threshold @ 2 pts, July window:
   fee-waived ROI ≥ +4.0% (half of BAS-70's observed) with the 95%
   interval, clustered by game, excluding zero; and as-quoted ROI (after
   the 7% taker fee) > 0.
2. **(It is the player term, not the line shape.)** The league-rate
   control priced by the same rule on the same hits contracts is negative
   as quoted, and its fee-waived ROI is at least 5 points below the
   model's.
3. **(Negative control.)** Strikeout contracts lose as quoted, as in
   every prior section.

### Vacuity check, run before scoring

At least 10,000 settled hits contracts with volume above the archive's
`min_volume` in the July window; median candles per contract ≥ 3
(retention has not hollowed the closes); the July stat mix within a
factor of 1.5 of August's per stat. Below any of these the doc says the
window is too thin or too damaged to test, and the numbers are reported
as descriptive only.

### Failure conditions

If prediction 1 fails on either leg, the hits lead is recorded as noise
on the first data that could have confirmed it, and the paper ledger's
hits gate keeps running with that written next to it. If 1 holds and 2
fails, the edge is the contract shape, not the player, and the doc says
so. Nothing in serving changes either way: the ledger (BAS-77, Stage
0.1) is already running exactly this rule on live tickets, and it, not
this backtest, decides Stage 1.

## What ships

Nothing in serving. If 1 and 2 hold, the result is written into
`docs/props-exam-2026.md` as the first pre-registered after-fee
positive, the Stage 0 document records it as the ledger's prior, and the
next test named is the maker-side replay on the same July candles under
its own pre-registration.

## Risks recorded up front

Candle retention may be partial for early July (the vacuity check counts
it). July contains the All-Star break (07-13 → 07-16), so the window is
~22 game-days. July is before the archive's constants were chosen
(matchup weight and τ on 07-31 → 08-16), so it is out of sample for
them, but the tuned Marcel parameters (2020–2024) and the props pricing
code were built with knowledge of the August results. `p_over_sd` and
the posterior columns are computed for completeness and not tested. The
fee rate 0.07 is second-hand (`docs/props-exam-2026.md`). Kalshi's
listing before 07-05 returning nothing may be retention rather than
series inception; either way it is outside the window.

## Results (2026-09-10)

Evidence `data/eval/bas93/props_replication.json` and `.md`; runner
`scripts/bas93_replication.py`; `scripts/props_exam.py --whole-window`
(refuses to run unless τ and the matchup weight are fixed); the appended
archive `data/market/prop_closes_2026.parquet` (78,134 → 125,525 closes,
07-05 → 09-02, 750 games) and `kalshi_prop_candles_2026.parquet`
(516,666 → 794,734), every pre-existing row retained.

**Verdict: the hits lead does not replicate, and it was a half-window
artefact to begin with. Descriptive by the vacuity rule; nothing ships.**

### Fetch

Kalshi lists nothing settled before 2026-07-05 in any of the seven
series. 07-05 → 07-31: 47,939 traded contracts listed, 47,391 closes
fetched at ~8 workers with ~1,900 HTTP 429s absorbed by the backoff.
Retention: 98.1% of listed hits contracts returned a pre-first-pitch
close; every archived contract has at least one candle. The empty dates
(07-13 → 07-15) are the All-Star break, not fetch holes. One deviation:
`--end` bounds a close timestamp at midnight UTC, which would have kept 4
of 07-30's 10 games; a supplementary 07-30 → 08-01 pass closed that and
the mirror hole on 07-31 in the original archive (645 → 1,720 contracts
across the two dates).

### Vacuity check, run before scoring

| check | value | threshold | verdict |
|---|---|---|---|
| settled hits contracts | 13,578 | ≥ 10,000 | pass |
| median candles per contract | **2.0** | ≥ 3 | **fail** |
| July stat mix vs August, worst ratio | 1.07× | ≤ 1.5× | pass |

So by the pre-registration's own rule every number below is descriptive
and settles nothing. What the failing leg does and does not mean: the
median close is still 15 minutes before first pitch (as in August), 99.4%
settle yes or no, 99.0% of names resolve, every contract has a candle.
What it tracks is thinner trade: 56.3% of July hits contracts traded
before first pitch against 66.4% in August, median pre-pitch volume 4
against 20. The missing hours would damage a maker replay; the taker exam
scored here uses the close.

### Predictions, descriptive

Frozen constants (matchup weight 1.0, threshold 2 points, τ .65, taker
rate .07, flat one unit, shared draw streams), whole July window, 37,360
settled contracts, 292 games, 601 players.

1. **Hits @ 2 pts: fails all three legs.** 4,336 bets, fee-waived
   **+0.6%** (−4.4%, +6.1%), as quoted **−4.7%**. Floor was +4.0% with the
   interval excluding zero and as quoted > 0.
2. **League-rate control: fails on the gap.** 5,520 bets, as quoted
   −6.8% (negative, as predicted); fee-waived −1.6%, so the model sits
   +2.2 points above it against ≥ 5 required. The player term buys about
   two points; the contract shape buys the rest.
3. **Strikeouts: passes.** 2,631 bets, −6.1% as quoted.

Context, July, edge ≥ 2 pts (fee-waived / as quoted): HR −6.7% / −13.2%;
TB −1.5% / −5.1%; pooled −1.4% / −5.8%. Posterior rule at τ .65: hits
+2.5% / −3.9%. Brier (ours / matchup / market / league): hits .16708 /
.16676 / .16562 / .16776; pooled .16586 / .16523 / .16085 / .16745. The
market wins every stat. A fixed-seed random-edge null returns +6.9%
fee-waived on July hits, above the model's +0.6%: there is very little
signal in that cell.

### Where the lead came from

BAS-70's +9.5% / +12.1% is its **second half**; the machinery here
reproduces those figures exactly (+9.537%, +12.135%, the published
intervals and bet counts). On the **whole** August window the same rule is
+3.7% / +7.1% fee-waived and **negative as quoted** (−1.8% / −0.2%).
Scored the way this ticket scores July, August itself would have missed
prediction 1's floor. The lead was the second half of one month; July is
the first out-of-sample look at it and lands at +0.6%.

### What this means

- **For the ledger:** nothing changes in serving. The paper ledger
  (`docs/bankroll.md`, Stage 0.1) keeps running the hits gate with this
  written next to it; its pre-registered expectation (+2% to +6% after
  fees over the Stage 1 window) now carries a prior of "the archive says
  roughly zero fee-waived and negative as quoted".
- **For the props exam:** the market's price beats ours on every stat on
  the new month too, and the taker fee remains the whole of the loss and
  then some.
- **For the archive:** the July month is in the tracked parquets now, so
  every future props test has 750 games instead of 457.
- Caveats: 44% of July hits contracts never traded before first pitch, so
  their close is a quote midpoint and the fill assumption is more
  optimistic than August's; the fee rate .07 is second-hand and on hits it
  is the whole gap between +0.6% and −4.7%; 292 of 312 scheduled games are
  covered.
