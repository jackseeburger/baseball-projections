# From the money exam to a bankroll

Tracked as BAS-77. The north star says *money is the exam*. This document
says how the exam becomes a bankroll, in stages, with the threshold for each
stage written down **before** any stage is reached — so a good week cannot
promote itself.

**Status: Stage 0 being built. No real money is involved at any stage
described here until Stage 2, and Stage 2 has a gate it has not met.**

## Where the exam stands

Everything after fees, everything out of sample, from
[money-exam-2026.md](money-exam-2026.md), [props-exam-2026.md](props-exam-2026.md)
and [posterior-props.md](posterior-props.md):

| Contract | Best strategy we have | Verdict |
| --- | --- | --- |
| Moneylines (Kalshi, as quoted) | −11.6% | every model loses; the market has our data and order flow |
| Props, pooled | −2.7% to −4.0% | the taker fee is the whole loss; break-even with it waived |
| **Props, hits only** | **+4.3% to +5.5%** | positive at the point estimate, interval crosses zero, **not pre-registered**; did not replicate on July (BAS-93: +0.6% fee-waived, −4.7% as quoted, and the August figure was a half-window artefact) |

Reinvesting scales an edge. It does not create one. Compounding a −3%
strategy compounds losses. So the bankroll question is not "how much" — it is
"has a positive expectation after fees been demonstrated, out of sample, on a
prediction written down first". Today the answer is: one candidate, no result.

## The stages

### Stage 0 — paper trade, live (now)

Every evening a job prices the open Kalshi props in the latest market
snapshot with the served props model, emits the tickets it *would* place,
and settles yesterday's from the exchange's own results. A paper bankroll of
1,000 units, quarter-Kelly capped at 5% of bankroll per ticket and 20% of it
across one slate (Amendment 1, below), fees charged as Kalshi charges them. Two ledgers per ticket: **taker** (cross the ask at
snapshot time, pay the fee) and **maker** (rest at the snapshot bid, fee
waived, filled only if a later snapshot shows the market traded through the
price — the fill assumption is written down and is pessimistic on purpose).

This is worth more than any backtest, because it removes the two things a
backtest cannot: selection of the archive after the fact, and hindsight on
which markets settled. The rules are frozen in code and the ledger is
committed, so the history cannot be edited.

**Pre-registered, before the first ticket:**

1. **Hits props are the primary test.** The one after-fee positive on the
   board, and the one that was not predicted — which is exactly why it needs
   a forward test rather than another look at the same archive. Prediction:
   taker ROI after fees on hits props is **positive**, between +2% and +6%,
   over the Stage 1 window. *(Evidence since: BAS-93 scored the July
   contracts with every constant frozen and found +0.6% fee-waived, −4.7%
   as quoted; the whole-August figure is negative as quoted too. The
   prediction stands as written; the prior behind it is now "roughly zero
   before the fee".)*
2. **Pooled props are the secondary test.** Prediction: taker ROI after fees
   is **between −4% and 0%** — that is, the pooled strategy does not clear
   the fee, as the archive says.
3. **Maker beats taker by 3 to 5 points** on ROI, and fills fewer than 60%
   of the tickets it rests.
4. **Strikeouts lose** under every rule, as they did in the archive.

**Vacuity check:** fewer than 150 hits tickets a week means the live market
is too thin for the Stage 1 window to close this season; report it and
extend the window into 2027 rather than lowering the bar.

#### **Amendment 1 (2026-09-09) — per-slate exposure**

**The finding.** The first real Stage 0 ledger ran on 2026-09-02 and the
paper bankroll went through zero on 2026-09-03. Nothing was wrong with the
code: the rule above caps each *ticket* at 5% of bankroll and says nothing
about how many tickets one evening may carry, and a Kalshi prop slate offers
hundreds of them simultaneously. The 2026-09-02 slate staked 9,069 units
against a 1,000-unit bankroll; across the whole archive the ledger staked
19,651 units, 19.7x the bankroll, before emission stopped — a non-positive
bankroll has no Kelly stake. As written, Stage 0 cannot be run: the bankroll
is destroyed long before the Stage 1 window's 1,000 settled tickets.

**The rule change.** The total stake across every ticket emitted from one
snapshot is capped at **20% of the running bankroll**. Each ticket keeps its
quarter-Kelly size capped at 5% of bankroll; if the sum of a slate's stakes
exceeds the cap, every stake in that slate is scaled down by the same factor,
pro rata. The cap is a constant in `src/market/paper.py` (`SLATE_CAP`), not a
flag — a rule that can be edited between runs is not a pre-registration.

**What this does not change.** The *sizing*, and only the sizing. Not the
selection rule (`pnl.decide` at the same 2-point threshold on the same arm),
not the served model, not the threshold, not the fee schedule, not the maker
fill assumption, not the four pre-registered predictions above, and not the
Stage 1 conditions. Scaling every stake in a slate by one common factor
cannot change which tickets are written or which of them win; it changes how
much each is worth.

**When it was made.** After seeing the ruin, and before any rerun. That
ordering is the thing that could be abused, so it is stated plainly: the
2026-09-02 and 2026-09-03 tickets were emitted, committed and published under
the unamended rule, the ruin was published as a finding, and this amendment
was written afterwards. No parameter was chosen by looking at a return.

**The window restarts.** The Stage 1 window — 1,000 settled tickets on the
primary test and 21 distinct game-days — **restarts at the amendment**. Every
ticket emitted before it stays in the ledger, flagged `rule_version="stage0"`,
is reported in the ROI tables, and counts toward nothing. Tickets from here
are `rule_version="stage0.1"` and they alone are scored by the gate. A window
that mixed two sizing rules would be a test of neither.

**The bankroll resets to 1,000.** A bankroll that has been through zero
cannot be sized from, so the amended rule starts again from the pre-registered
1,000 units. This is a reset of the *paper* bankroll and not a top-up of a
real one; the pre-amendment losses are not erased, they are recorded, and the
bankroll curve draws the two segments separately with the amendment marked
rather than drawing one line through a discontinuity that never happened.

**Vacuity implication.** At a 20% slate cap the maximum drawdown condition is
the binding one. The ticket count and the game-day count are now reachable —
a slate no longer risks the bankroll — and the ROI interval is a question of
sample size, but a rule that can lose a fifth of the bankroll in an evening
will show a realised trough that has to stay inside 1.5x what quarter Kelly
implies, and that is the condition most likely to fail. If Stage 1 is ever
met, expect the drawdown line to be the close one.

### Stage 1 — the gate to real money

All of the following, on the paper ledger, at the taker prices, after fees,
and none of them chosen after the fact:

- at least **1,000 settled tickets** on the primary test *and* at least
  **21 distinct game-days** — so that a hot week cannot clear the gate on
  its own;
- ROI after fees with a **95% bootstrap interval that excludes zero**;
- realised maximum drawdown no worse than **1.5×** what quarter-Kelly
  implies at the observed edge and variance, computed from the same ledger;
- the served model unchanged during the window, or the window restarted.

A gate met on the maker ledger only is not met: the maker fill assumption is
a model of execution, not execution.

### Stage 2 — a small real bankroll

Yours, in your account, sized so that the worst drawdown the ledger has
shown would be an annoyance and not a decision. The system produces tickets
and sizes; the placing is done by you, or by a script running under your
keys with three hard limits it cannot exceed: a maximum stake per ticket, a
daily loss stop, and a kill switch you can pull. The system never holds
funds and never places a bet itself — Kalshi is a regulated exchange with
identity requirements, the execution is legally yours, and the stop has to
be something you can reach.

Capital injections are keyed to the **same** metric that opened Stage 2 —
the rolling after-fee ROI and its interval on the *real* ledger — and never
to a good month. A withdrawal rule is written down at the same time as the
injection rule.

### What ends a stage

The gate rule runs in both directions. If the real ledger's rolling ROI
interval comes to include zero over a window as long as the one that opened
Stage 2, the bankroll drops back to paper until it re-clears. That is a
demotion, not a failure, and it is published like everything else.

## What this is for

A bankroll is not the goal of the system. It is the harshest exam the system
can sit, because a prediction market has our data and its own order flow,
and the only way to be paid is to be right where it is wrong. The factory's
product is a *verified* edge; the bankroll is the instrument that verifies
it. If it never opens, the exam has still done its job.
