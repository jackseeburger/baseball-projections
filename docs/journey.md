# Where we are on the way to the north star

**2026-09-09.** A dated snapshot, layer by layer, against
[architecture.md](architecture.md). Updated when a layer's status moves;
the model-by-model inventory is [target-system.md §2c](target-system.md).

## The north star

Build, from public data, what a team's R&D group builds: measure the
pitch → estimate true talent with hierarchical pooling → project playing
time → aggregate to games and seasons → price against the market. Truth
first, the market is the bar, money is the exam.

## Where each layer stands

**Layer 1 — measurement (pitch grain).** Substantially built.
- Hitting: contact quality (Statcast EV/LA) is **served** on all five
  components (BAS-72). Swing decisions were built and measured and turned
  out to be the same signal as contact quality on K%; not served (BAS-81).
- Pitching: stuff (velocity, movement, spin → whiff) is **served** on
  BB/BF and HR/BF; K/BF withheld by 0.001 of a t (BAS-79). Command /
  location was measured (BAS-76): it is a third of the pitch-level signal,
  but the pitcher-level residual over stuff does not persist year to year
  (r 0.40 against a 0.45 floor), so stage 2 was never run and nothing is
  served. The level aggregate with stuff as a control (BAS-87) then
  cleared the gate on BB/BF: −4.5% of error vs tuned Marcel and −1.5% on
  top of the served stuff engine at t −3.6, nothing on strikeouts or home
  runs. Serving it on the pitcher walk rate was tried and withheld (BAS-88):
  on the 2026 season alone the gain is 1.1% at t −1.1, below the
  significance bar, so the engine is built and dormant until a served
  season clears.
- Defence has no public tracking; that track stays thin.

**Layer 2 — true talent (hierarchical Bayes).** The contested layer.
Tuned Marcel serves today. The Bayes model draws with it on outcomes
alone on all three components it has been rolled to — K% (BAS-69), BB%
and HR/PA (BAS-73, the full grid: HR/PA's predicted win did not appear,
BB% loses by a hair) — and has not yet been fought *with* the measurements
inside it. The first structural fight, BAS-83, put the contact measurement in
as a regressor and lost by 19% of MAE on HR/PA; the diagnosis is errors in
variables, and the fix is the measurement model (BAS-85), not a re-tune. The pitcher-side hierarchical model (BAS-74) is next, with
stuff as a covariate from day one.

**Layer 3 — playing time.** Served: rest-of-season PA with the horizon
blend, IL and option return probabilities, the lineup-slot cap.

**Layers 4–6 — game and season arithmetic.** Served: starter, lineup,
bullpen-fatigue, park and defence terms in the nightly playoff odds; the
season sim with the postseason bracket. The per-game win model does not
beat the market (Brier .246 vs the market's .242) — the market is a real
bar here.

**Layer 7 — decision / market.** Built and forward-testing. Props priced
from layer 2 through the lineup/starter sim; the Kalshi/Polymarket archive
nightly; the money exam says taker fees eat the props edge and moneylines
lose; the Stage 0 paper ledger is live under Amendment 1
([bankroll.md](bankroll.md)) and needs 21 game-days and 1,000 settled
tickets before Stage 1 can be evaluated. Nothing suggests real money yet.

## Discipline in place

Pre-register → run → score against the doc → publish negatives → serve
only what beats the served engine, and (from BAS-82) only when the
covariate earns ≥ 1.0% of MAE on its own. Every model decision from
2026-09-09 is on the record.

## The plan from here, in order

1. **The layer-2 model track.** On outcomes alone the hierarchical model
   is a draw on every component (BAS-69, BAS-73), and that is what the
   math says it should be: a single-component random-effects model *is*
   Marcel with a learned ballast. The model wins only by carrying
   structure Marcel cannot express, so the track is three structural
   models in order, each pre-registered and each kept whatever the
   verdict: the covariates inside the likelihood (BAS-83: measured, and worse; the block imports the thin current window's noise at a coefficient fitted on full seasons, which is the case for a measurement model rather than a regressor), the
   joint multi-component model with correlated player effects (BAS-84,
   `docs/bayes-joint.md`: measured; the correlations are real in three
   seasons of four, the gain is a third of a percent on K% and BB% and
   nothing on HR/PA, and contact quality still beats it, so it is kept as
   the base for the next model and not served), and the measurement model where Statcast is a
   second observation of the same latent talent rather than a regressor
   (BAS-85, `docs/bayes-measurement.md`). Park factors now exist (BAS-86, `docs/park-factors.md`): persistent,
   real, and useless multiplied onto Marcel because Marcel's inputs already
   carry the park; their place is the hierarchical models' offset, to be
   scored there. After those: time and age as a process. The posterior
   width at layer 7 was tested (BAS-92, `docs/posterior-width.md`): the
   served props mean with the hierarchical model's per-player width in
   place of the Beta's, on BAS-70's own archive. **Vacuous by its own
   rule**: the single-component width is 1.2× the Beta's in almost the
   same order (Spearman .96), so it cannot reorder the bets; the
   `P(edge > 0)` rule ranks saturated favourites on top and they are the
   worst bucket under either width. Width that reorders contracts has to
   come from structure (joint or measurement posteriors). Two bugs fixed
   on the way: a shared pricing RNG that leaked one player's Beta into
   every later player's draws, and a `bayes_arm` kwarg that made every
   Bayes fit on main raise before sampling.
2. **BAS-74**, the pitcher hierarchical model with stuff inside it; a
   command *level* aggregate (BAS-76's follow-up) once pre-registered.
3. **ISO and BABIP** hierarchical models (their own denominators), after 1.
4. **Per-game win probability** — the largest unclosed gap to the market.
   Scoping found the per-game chain still prices every game from stock
   Marcel on both sides: none of the served layer-2 engines reach it.
   Putting them in was pre-registered and run (BAS-90,
   `docs/chain-engines.md`) and **lost**: tuned Marcel, stuff and contact
   each make the game price worse against the close (+.00076 Brier on the
   756 market games, same sign on all of 2025 and 2026, vacuity passed).
   The engines widen the rate tables by 18–37% and every downstream
   constant in the chain was tuned to the narrower ones; the tuned age
   curve alone is 70% of the cost. Nothing shipped, the switch stays in
   the code. The next lever is a spread-aware chain (re-fit the chain's
   constants to the engine tables on 2025 only, or shrink the engine
   tables by one pre-registered factor), under its own pre-registration:
   BAS-91 (`docs/chain-engines-matched.md`) matched each engine table's
   level and spread to the stock table's at the same as-of, parameter
   free, and isolated the age curve. **A real null:** the match removes
   essentially all of the engines' cost (pooled +.00026 → −.00002) and the
   candidate arm is the first engine arm not worse than stock on any full
   season, but its gain is −.00008 ± .00015 on 3,988 games. The per-game
   exam cannot see a 1% player-grain gain. Nothing shipped; the game price
   is moved by terms the chain lacks, not by a better rate table.
5. **Money** — let the ledger accumulate. Hits props were the one
   after-fee positive and stay the primary test. The Stage 1 gate decides.
   The July 2026 contracts the archive never fetched were pulled (47,391
   closes, the archive is now 750 games) and scored as a frozen-constant
   replication of that lead (BAS-93, `docs/props-replication.md`). **It
   did not replicate**: hits +0.6% fee-waived and −4.7% as quoted on July,
   and BAS-70's number turns out to be the second half of one month (the
   whole of August is negative as quoted). The market beats our price on
   every stat on the new month too. The ledger keeps running; its prior
   is now "roughly zero before the fee".

The honest summary: the measurement layer is real and serving; the talent
layer is the open question and this week's work puts the hierarchical
models in the fight they were designed for; the market layer is
instrumented and says we are not there yet.
