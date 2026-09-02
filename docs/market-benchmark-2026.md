# Market Benchmark — 2026 per-game P(win) vs. the exchanges

Station E scored against station M (docs/architecture.md §0: *the market is
the bar*). Produced by:

```
python scripts/backfill_market_closes.py --season 2026
python scripts/backtest_game_odds.py --season 2026 --min-games 20 \
    --market data/parquet/market_closes_2026.parquet
```

## What "market close" means here

For an exchange there is no bookmaker close; the closing line is the last
price before first pitch. Kalshi exposes hourly candlesticks per market, so
the close is the last candle ending at or before the scheduled first pitch —
median **15 minutes** before the game. That price already knows the starting
pitchers and lineups.

Coverage: Kalshi's `KXMLBGAME` series has settled markets from **2026-06-22**;
the first game with a pre-pitch candle is 06-26. Polymarket's closed-event
listing reaches back to early July. Neither venue can give us April–June.

## Scoreboard — 756 games priced by both venues, 2026-07-04 → 09-02

| Model | Brier | Log loss | Mean P(home) |
|---|---|---|---|
| **Kalshi close** | **0.2416** | **0.6759** | 0.537 |
| **Polymarket close** | **0.2417** | **0.6761** | 0.531 |
| **pythag_60_sp** (starters, new) | **0.2448** | **0.6827** | 0.533 |
| pythag_100 | 0.2462 | 0.6854 | 0.533 |
| pythag_60 (production) | 0.2462 | 0.6855 | 0.533 |
| pythag_160 | 0.2464 | 0.6858 | 0.533 |
| pythag_30 | 0.2465 | 0.6863 | 0.533 |
| win_pct_log5 | 0.2467 | 0.6867 | 0.533 |
| pythag_0 (no regression) | 0.2476 | 0.6886 | 0.533 |
| home_constant | 0.2489 | 0.6909 | 0.533 |

Realized home win rate on these games: 0.529. On the wider Kalshi-only
set (876 games from 06-26) the numbers are the same to the third decimal:
Kalshi 0.2415, pythag_60 0.2464.

`pythag_60_sp` is `pythag_60` with each side's runs-allowed rate moved by how
far its announced starting pitcher's regressed FIP sits from league average,
over the 5.5 innings an average start covers (`src/sim/starters.py`). It is the
same walk-forward harness: team rates from games before the date, pitcher rates
from appearances before the date, and the same log5 + HFA conversion. On the
full 1,773-game 2026 set (not just the 756 the exchanges priced) it scores
0.2465 against pythag_60's 0.2478 — the same 0.0013 gain, so the result is not
an artifact of the market subset. **Every starter slot was filled**: across
those 1,773 games only 2 fell back to `pythag_60` for a missing probable
(0 of the 756), and 22 of 3,546 starter slots had no prior history and were
scored at league average.

The two exchanges agree with each other closely — mean |Δ| = 0.008,
correlation 0.991 — so the benchmark is not an artifact of one venue.

## Reading it

- **The starting pitcher closes about 30% of the gap to the market.**
  pythag_60 → pythag_60_sp is 0.2462 → 0.2448, which is 0.0014 of the 0.0046
  that separated us from Kalshi. The market still wins by 0.0033 (t = 2.1 on
  the paired per-game difference, so *its* remaining edge is real). Our own
  gain is directionally consistent — 0.0014 on the 756 market games, 0.0013 on
  all 1,773 games of 2026, 0.0004 on 2025 — but on any one of those sets it is
  inside one standard error (t = −1.2 on the 756, −1.7 on the 1,773). Call it a
  real term of modest and not-yet-precisely-measured size, not a solved
  station. It clears the station E gate (§3) on the common game set; a second
  season of exchange history is what would make the size of the win certain.
- **The remaining 0.0033 is lineups, bullpens, and a better pitcher model.**
  The starter term correlates 0.68 with the market's own deviation from
  pythag_60, and regressing the market's deviation on ours gives a slope of
  1.21 — the market moves *further* on the same games we move on, in the same
  direction. So we are not over-reacting to pitchers; we are under-reacting,
  and we are missing whatever else it prices (who is actually in the lineup,
  rest and bullpen state, weather, park).
- **Team strength does carry real information.** pythag_60 beats the
  home-constant baseline by 0.0027, and the market beats pythag_60 by 0.0046.
  So roughly 37% of the distance from "know nothing" to "the market" is
  covered by regressed run differential alone, and another ~19% by the starter.
- **The starter term is what put probabilities in the tails.** pythag_60 put 6
  of the 756 games below 0.40 and 26 above 0.65; pythag_60_sp puts 24 and 42
  there, and they verify (0.373 predicted / 0.333 realized at the bottom).
  The top bucket is the soft spot: 42 games predicted 0.683, realized 0.619 —
  our biggest home favorites are still a little too confident.
- **Ballast barely matters** (30–160 games all within 0.0003). The signal we
  were missing was not in how we regress team strength — it was that we had no
  pitcher at all.

### How the pitcher term avoids fitting the test set

The pitcher rates are Marcel-standard: K, BB+HBP and HR per batter faced over
the current season plus the two before it at 5/4/3 recency weights, regressed
toward league average and pushed through the standard FIP coefficients
(13/3/−2) with the constant set so a league-average arm returns league RA/9.
Each component is regressed on its own published rate-stabilization point —
70 batters faced for strikeouts, 170 for walks, 1300 for home runs — so home
runs get regressed nearly twenty times harder than strikeouts, which is what
keeps FIP's 13× home-run coefficient from turning noise into a forecast. The
single free knob is how much harder than *reliability* a **projection** has to
regress, since the next start also has to absorb real talent change. That
multiplier was chosen walk-forward on **2025 only**, where the curve is flat
for anything from 2× to 6× (all within 0.00003 Brier); 2× was taken, giving
ballasts of 140 / 340 / 2600 batters faced. No constant was chosen by looking
at a 2026 score.

One deliberate departure from the obvious construction: the starter enters as a
*delta* from league average applied to the team's own runs-allowed rate, not as
`5.5/9 · FIP + 3.5/9 · team_RA`. FIP is park- and defense-neutral and team RA is
not, so the absolute-level blend quietly regresses 61% of every team's run
prevention toward the league mean — a Coors staff and a Petco staff both told
they allow league-average runs for 5.5 innings. Scored, that version came in at
0.2466, *worse* than pythag_60, and its correlation with the market's deviation
was only 0.48 against the delta form's 0.68. The team-regression it smuggled in
cost more than the pitcher information it added.

## What this does not yet show

- Sportsbook closes (Pinnacle) — the archive started 2026-09-02, so a
  book benchmark exists only from here forward.
- April–June, where no exchange history survives.
- Anything about *money*: this is truth scoring against the market's
  probability, not simulated P&L. CLV and fill-aware ROI come after a
  model that is at least at market on Brier.
- **A morning-of forecast.** `probablePitcher` for a past date returns the
  pitcher who actually started, which is what the exchanges' closes knew — the
  median Kalshi close is 15 minutes before first pitch — so the comparison is
  fair. It is not a simulation of predicting at 9am, where late scratches would
  cost a little.
- **Lineups, bullpen state, park and weather** — none of them are in the model
  yet, and the 0.0033 the market still holds is where they live.
