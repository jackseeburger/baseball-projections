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

## Scoreboard — 876 games, 2026-06-26 → 2026-09-02, common population

| Model | Brier | Log loss | Mean P(home) |
|---|---|---|---|
| **Kalshi close** | **0.2415** | **0.6757** | 0.537 |
| pythag_60 (production) | 0.2464 | 0.6860 | 0.533 |
| pythag_100 | 0.2464 | 0.6860 | 0.533 |
| pythag_160 | 0.2467 | 0.6865 | 0.533 |
| pythag_30 | 0.2467 | 0.6866 | 0.533 |
| win_pct_log5 | 0.2472 | 0.6877 | 0.532 |
| pythag_0 (no regression) | 0.2478 | 0.6891 | 0.533 |
| home_constant | 0.2494 | 0.6919 | 0.533 |

Realized home win rate on these games: 0.529.

## Reading it

- **The market beats us by 0.0049 Brier.** Every one of our models is a
  team-strength number that knows nothing about who is pitching. The market's
  last price before first pitch does. That difference is the measured size of
  the starting-pitcher + lineup term (architecture §4, row 2) — the next
  modeling ticket, and now it has a number to beat.
- **Team strength does carry real information.** pythag_60 beats the
  home-constant baseline by 0.0030, and the market beats pythag_60 by 0.0049.
  So roughly 40% of the distance from "know nothing" to "the market" is
  covered by regressed run differential alone.
- **Calibration of pythag_60 is fine in the middle and thin at the tails**
  (9 games below 0.40, 32 above 0.65). Any per-game model that produces
  sharper probabilities will need those tails to be right; the market's tails
  are where starters matter most.
- **Ballast barely matters** (30–160 games all within 0.0003). The signal we
  are missing is not in how we regress team strength.

## What this does not yet show

- Polymarket as a second column (pending in the same parquet).
- Sportsbook closes (Pinnacle) — the archive started 2026-09-02, so a
  book benchmark exists only from here forward.
- April–June, where no exchange history survives.
- Anything about *money*: this is truth scoring against the market's
  probability, not simulated P&L. CLV and fill-aware ROI come after a
  model that is at least at market on Brier.
