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
