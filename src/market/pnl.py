"""Simulated P&L against the exchanges — the money exam (architecture.md §0).

Truth first; the market is the bar; **money is the exam**. Brier says whether a
probability is good; this module says what that probability would have been
worth after the spread and the fees, which is a different and harsher question.
It adds the three things accuracy alone cannot see:

    hurdle       you pay the ask and the fee, not the mid
    selectivity  you only get paid where you *disagree* with the market
    sizing       a fractional-Kelly stake on a miscalibrated edge still loses

Everything here is a pure function of two frames — the walk-forward predictions
(`scripts/backtest_game_odds.py --out`) and the reconstructed pre-game closes
(`src/market/backfill.py`) — joined on `game_pk`.

**The fill assumption is optimistic, deliberately and visibly so.** We assume we
could have traded one unit at the closing quote of the last pre-pitch candle
(median 15 minutes before first pitch) in whatever size we wanted, taking the
whole trade at the top of book. Real execution pays for depth, moves the price
it trades against, and — for a price the market only reached because of news we
did not have — is exactly where the fill would not have been there. So a
positive ROI here is a *necessary* condition for a strategy, not a sufficient
one; a negative ROI here is decisive.

Conventions
-----------
* `p` is always P(home wins). A YES contract on the home team pays $1 if the
  home team wins; a NO contract pays $1 if it loses.
* A **stake** is capital at risk, in units. One unit of stake at a cost of
  `c` dollars per contract buys `1/c` contracts, so `profit / stake` is an ROI
  that is comparable across prices. Buying a single contract is the special
  case `stake = c`.
* Thresholds and edges are in probability points (0.02 = 2 pts).

Fees
----
Kalshi's published taker fee is ``round_up_to_cent(0.07 · C · P · (1 − P))``
per order, where C is contracts and P the price in dollars — maximal at a
coin flip (1.75¢, rounded to 2¢ on one contract) and vanishing in the tails.
The official docs page that survives is the rounding rule
(https://docs.kalshi.com/getting_started/fee_rounding: "trade_fee =
ceil_6dp(model_fee)", accumulated per order across fills); the fee-schedule
page at kalshi.com was unreachable (HTTP 429) from this environment, so the
0.07 constant comes from the published formula as reported by secondary
sources. We round up per *contract*, which is the schedule's behaviour for a
one-contract order and slightly conservative (i.e. it charges more) for
larger ones.

Kalshi's **maker** fee is the same formula at a quarter of the rate,
``0.0175 · C · P · (1 − P)`` — 0.44¢ at a coin flip against the taker's 2¢.
That constant is second-hand for the same reason the taker's is (the
fee-schedule PDF 429s from this environment); two independent secondary
readings of the July 2026 schedule agree on "maker = 25% of taker" and on the
0.44¢ maximum. It is `--kalshi-maker-fee-rate`, not a constant.

Polymarket takes no fee on most sports markets, but the reconstructed close is
a mid, so the cost of crossing is modelled as a configurable half-spread
(default 1¢ each side). Every rate is a parameter, not a constant, so a change
in either venue's schedule is a flag rather than a rewrite.

Two exams live here. The **taker exam** crosses the closing quote and is the
one `scripts/money_exam.py` runs by default. The **maker exam** (bottom of
this module, `--maker`) rests a limit order through the 24 hours before first
pitch and asks whether the price came to it — the trade that stops paying the
spread and the taker fee, which together cost more than any edge station E has
shown. It needs a third frame, the hourly candle archive
(`scripts/backfill_kalshi_candles.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KALSHI_TAKER_RATE = 0.07        # round_up_to_cent(0.07 · C · P · (1-P))
KALSHI_MAKER_RATE = 0.0175      # a quarter of the taker rate; 0.44¢ max, at 50¢
POLYMARKET_TAKER_RATE = 0.0     # no taker fee on sports as of 2026; spread is the cost
DEFAULT_HALF_SPREAD = 0.01      # each side of the mid, in dollars
DEFAULT_KELLY_FRACTION = 0.25   # quarter Kelly
DEFAULT_KELLY_CAP = 0.05        # 5% of bankroll on any one game
BANKROLL = 1.0                  # fixed notional; stakes do not compound
BOOTSTRAP_DRAWS = 2_000
RANDOM_EDGE_SD = 0.03           # 3 pts of noise on the close — the null with a pulse
THRESHOLDS = (0.00, 0.02, 0.04, 0.06)
NO_BET = ""


@dataclass(frozen=True)
class Venue:
    """How one exchange prices and charges.

    `half_spread=None` means the venue quotes a real book and we cross it;
    a number means only a mid survives and we assume that spread around it.
    `maker_rate` is what the same venue charges for *providing* liquidity
    instead of taking it — the maker exam's hurdle.
    """

    name: str
    taker_rate: float = KALSHI_TAKER_RATE
    half_spread: float | None = None
    round_cents: bool = True
    maker_rate: float = 0.0


KALSHI = Venue("kalshi", KALSHI_TAKER_RATE, None, round_cents=True,
               maker_rate=KALSHI_MAKER_RATE)
POLYMARKET = Venue("polymarket", POLYMARKET_TAKER_RATE, DEFAULT_HALF_SPREAD,
                   round_cents=False, maker_rate=0.0)
VENUES = {v.name: v for v in (KALSHI, POLYMARKET)}


# ───────────────────────────── prices and fees ─────────────────────────────

def ceil_cents(x):
    """Round up to the next cent (the exchange's direction, never ours)."""
    return np.ceil(np.asarray(x, dtype=float) * 100 - 1e-9) / 100


def fee_per_contract(price, rate: float = KALSHI_TAKER_RATE,
                     round_cents: bool = True):
    """Kalshi's published taker fee for one contract at `price` dollars."""
    raw = rate * np.asarray(price, dtype=float) * (1 - np.asarray(price, dtype=float))
    return ceil_cents(raw) if round_cents else raw


def edge(p_model, p_market):
    """Signed disagreement with the market, in probability units."""
    return np.asarray(p_model, dtype=float) - np.asarray(p_market, dtype=float)


def quotes(df: pd.DataFrame, venue: Venue) -> tuple[np.ndarray, np.ndarray]:
    """(bid, ask) for P(home) at each venue's close.

    A real book is crossed as quoted; a mid-only venue gets the configured
    half-spread. Quoted books are still floored at the close ± 0 so a crossed
    or missing quote cannot manufacture an edge.
    """
    close = df["p_home_close"].to_numpy(dtype=float)
    if venue.half_spread is not None:
        hs = float(venue.half_spread)
        return np.clip(close - hs, 0.0, 1.0), np.clip(close + hs, 0.0, 1.0)
    bid = pd.to_numeric(df["bid"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(df["ask"], errors="coerce").to_numpy(dtype=float)
    bid = np.where(np.isnan(bid), close, bid)
    ask = np.where(np.isnan(ask), close, ask)
    lo, hi = np.minimum(bid, ask), np.maximum(bid, ask)
    return np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)


def decide(p_model, bid, ask, threshold: float = 0.0) -> pd.DataFrame:
    """Which side to take, at what cost, for how much edge.

    Buy YES on the home team at the **ask** when the model is above it, buy NO
    at ``1 − bid`` when the model is below the bid, and do nothing inside the
    spread — which is where the market's own price always sits, so the market
    as a model never trades. Comparisons are strict, so an edge of exactly the
    threshold is not a bet.
    """
    p = np.asarray(p_model, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    yes = p > ask + threshold
    no = p < bid - threshold
    side = np.where(yes, "yes", np.where(no, "no", NO_BET))
    cost = np.where(yes, ask, np.where(no, 1.0 - bid, np.nan))
    taken = np.where(yes, p - ask, np.where(no, bid - p, np.nan))
    return pd.DataFrame({"side": side, "cost": cost, "edge": taken})


# ───────────────────────────── stakes ─────────────────────────────

def kelly_stake(p_win, cost, fraction: float = DEFAULT_KELLY_FRACTION,
                cap: float = DEFAULT_KELLY_CAP, bankroll: float = BANKROLL):
    """Fractional Kelly on a binary contract, capped.

    A contract bought at `cost` pays $1, so the full-Kelly bankroll fraction is
    ``(p_win − cost) / (1 − cost)``. We take `fraction` of it and cap the
    result. Fees are not in the Kelly arithmetic — including them would shrink
    the stake, so leaving them out is the less flattering choice.
    """
    p = np.asarray(p_win, dtype=float)
    c = np.asarray(cost, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = (p - c) / (1.0 - c)
    full = np.where(np.isfinite(full), full, 0.0)
    return bankroll * np.clip(fraction * full, 0.0, cap)


def stakes(side, p_model, cost, staking: str = "flat",
           fraction: float = DEFAULT_KELLY_FRACTION, cap: float = DEFAULT_KELLY_CAP,
           bankroll: float = BANKROLL):
    """Stake per bet, in units. `flat` is one unit; `kelly` is capped fractional."""
    if staking == "flat":
        return np.ones(len(np.atleast_1d(cost)), dtype=float)
    if staking != "kelly":
        raise ValueError(f"unknown staking rule {staking!r}")
    p = np.asarray(p_model, dtype=float)
    p_win = np.where(np.asarray(side) == "yes", p, 1.0 - p)
    return kelly_stake(p_win, cost, fraction, cap, bankroll)


# ───────────────────────────── settlement ─────────────────────────────

def settle(side, cost, stake, home_won, venue: Venue = KALSHI):
    """Profit in stake units after the fee.

    One unit of stake buys ``1/cost`` contracts. A winning YES bought at the
    ask returns ``1 − ask − fee`` per contract; a winning NO bought at
    ``1 − bid`` returns ``bid − fee``.
    """
    side = np.asarray(side)
    cost = np.asarray(cost, dtype=float)
    stake = np.asarray(stake, dtype=float)
    won_home = np.asarray(home_won, dtype=bool)
    contracts = np.where(cost > 0, stake / np.where(cost > 0, cost, 1.0), 0.0)
    won = np.where(side == "yes", won_home, ~won_home)
    fee = contracts * fee_per_contract(cost, venue.taker_rate, venue.round_cents)
    gross = contracts * won.astype(float)
    return gross - stake - fee, fee


# ───────────────────────────── metrics ─────────────────────────────

def max_drawdown(profit) -> float:
    """Deepest peak-to-trough of the cumulative P&L path, in stake units."""
    p = np.asarray(profit, dtype=float)
    if p.size == 0:
        return 0.0
    equity = np.cumsum(p)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    return float(np.max(np.maximum(peak - equity, 0.0)))


def bootstrap_roi_ci(profit, stake, draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
                     alpha: float = 0.05, groups=None) -> tuple[float, float]:
    """Percentile CI on ROI, resampling *games* with replacement.

    Bets are one per game and settle independently, so the game is the
    resampling unit. The ratio ΣP/ΣS is re-formed inside each draw, which is
    what makes this a CI on ROI rather than on mean profit.

    `groups` makes that a **cluster** bootstrap: pass a per-bet label (the
    game) when several bets settle on the same event, as they do for player
    props — a hitter's 1+, 2+ and 3+ hits are one afternoon's at-bats, and a
    row-wise bootstrap would treat them as three independent observations and
    report a CI far tighter than the data supports.
    """
    profit = np.asarray(profit, dtype=float)
    stake = np.asarray(stake, dtype=float)
    n = profit.size
    if n == 0 or stake.sum() <= 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, n, size=(draws, n))
        tot_p = profit[idx].sum(axis=1)
        tot_s = stake[idx].sum(axis=1)
    else:
        labels, codes = np.unique(np.asarray(groups), return_inverse=True)
        order = np.argsort(codes, kind="stable")
        starts = np.searchsorted(codes[order], np.arange(labels.size))
        members = np.split(order, starts[1:])
        sum_p = np.array([profit[m].sum() for m in members])
        sum_s = np.array([stake[m].sum() for m in members])
        pick = rng.integers(0, labels.size, size=(draws, labels.size))
        tot_p = sum_p[pick].sum(axis=1)
        tot_s = sum_s[pick].sum(axis=1)
    roi = np.divide(tot_p, tot_s, out=np.zeros_like(tot_p), where=tot_s > 0)
    return (float(np.quantile(roi, alpha / 2)), float(np.quantile(roi, 1 - alpha / 2)))


def summarize(bets: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS,
              seed: int = 0, group_col: str | None = None) -> dict:
    """Money metrics for one model at one venue and one threshold.

    ROI is total profit over total stake. `clv` is the closing-line-value
    proxy, signed toward the side actually taken: how far the model sat from
    the close on the games it bet, positive when the model was on the side it
    liked. With a *closing* price it is not true CLV (there is no later price
    to compare to) — it is the leading indicator's stand-in, and it is the one
    number here that does not depend on how the games happened to land.
    """
    if len(bets) == 0:
        return {"n_bets": 0, "hit_rate": float("nan"), "total_staked": 0.0,
                "total_return": 0.0, "roi": 0.0, "roi_lo": 0.0, "roi_hi": 0.0,
                "mean_edge": float("nan"), "clv": float("nan"),
                "max_drawdown": 0.0, "total_fees": 0.0}
    profit = bets["profit"].to_numpy(dtype=float)
    stake = bets["stake"].to_numpy(dtype=float)
    groups = bets[group_col].to_numpy() if group_col and group_col in bets else None
    lo, hi = bootstrap_roi_ci(profit, stake, draws, seed, groups=groups)
    return {
        "n_bets": int(len(bets)),
        "hit_rate": float(bets["won"].mean()),
        "total_staked": float(stake.sum()),
        "total_return": float(profit.sum()),
        "roi": float(profit.sum() / stake.sum()) if stake.sum() else 0.0,
        "roi_lo": lo,
        "roi_hi": hi,
        "mean_edge": float(bets["edge"].mean()),
        "clv": float(bets["clv"].mean()),
        "max_drawdown": max_drawdown(profit),
        "total_fees": float(bets["fee"].sum()),
    }


# ───────────────────────────── the exam ─────────────────────────────

def join_closes(preds: pd.DataFrame, closes: pd.DataFrame, venue: str) -> pd.DataFrame:
    """Walk-forward predictions joined to one venue's closes on `game_pk`.

    Every prediction was built from games, appearances and lineups strictly
    before its own date; every close is the last quote before that game's
    first pitch. The join is the only place the two meet.

    A game suspended and resumed the next day is scored twice by the
    backtest; the exchange has one contract for it, so the first prediction
    — the one made before the original first pitch, which is the one the
    close is a quote on — is the one kept.
    """
    cols = ["game_pk", "p_home_close", "bid", "ask", "home_won", "minutes_before_pitch"]
    # Carried through for the maker exam, which needs to know which market's
    # candles belong to this game and when its order had to be cancelled.
    cols += [c for c in ("market_id", "game_start", "close_ts") if c in closes.columns]
    c = closes[closes["venue"] == venue][cols]
    df = preds.drop_duplicates("game_pk", keep="first").merge(
        c, on="game_pk", how="inner", validate="one_to_one")
    if "home_win" in df and df["home_won"].notna().any():
        mismatch = int((df["home_win"].astype(bool) != df["home_won"].astype(bool)).sum())
        if mismatch:
            raise ValueError(f"{venue}: {mismatch} games disagree on the winner")
    return df.sort_values("date").reset_index(drop=True)


def add_controls(df: pd.DataFrame, seed: int = 0,
                 sd: float = RANDOM_EDGE_SD) -> pd.DataFrame:
    """The two controls that need the venue's own price.

    `market` is the close itself — it sits inside the spread by construction,
    so it never trades and its ROI is exactly zero. It is the anchor: any
    model whose ROI is not clearly above zero has not beaten *doing nothing*.
    `random_edge` is the close plus N(0, sd) — a model with no information but
    the market's, which manufactures disagreement out of noise and pays the
    spread and the fee for it. It is what an ROI looks like when the edge is
    fake.
    """
    out = df.copy()
    out["market"] = out["p_home_close"].astype(float)
    rng = np.random.default_rng(seed)
    out["random_edge"] = np.clip(
        out["p_home_close"].astype(float) + rng.normal(0.0, sd, len(out)), 0.01, 0.99)
    return out


def bet_frame(df: pd.DataFrame, model: str, venue: Venue, threshold: float = 0.0,
              staking: str = "flat", fraction: float = DEFAULT_KELLY_FRACTION,
              cap: float = DEFAULT_KELLY_CAP, bankroll: float = BANKROLL) -> pd.DataFrame:
    """One row per bet actually placed, priced, staked and settled."""
    bid, ask = quotes(df, venue)
    d = decide(df[model].to_numpy(dtype=float), bid, ask, threshold)
    take = (d["side"] != NO_BET).to_numpy()
    if not take.any():
        return pd.DataFrame(columns=["date", "game_pk", "side", "cost", "edge",
                                     "stake", "profit", "fee", "won", "clv"])
    p = df[model].to_numpy(dtype=float)[take]
    close = df["p_home_close"].to_numpy(dtype=float)[take]
    side = d["side"].to_numpy()[take]
    cost = d["cost"].to_numpy(dtype=float)[take]
    stake = stakes(side, p, cost, staking, fraction, cap, bankroll)
    home_won = df["home_win"].to_numpy(dtype=bool)[take]
    profit, fee = settle(side, cost, stake, home_won, venue)
    won = np.where(side == "yes", home_won, ~home_won)
    signed_clv = np.where(side == "yes", p - close, close - p)
    return pd.DataFrame({
        "date": df["date"].to_numpy()[take],
        "game_pk": df["game_pk"].to_numpy()[take],
        "side": side, "cost": cost, "edge": d["edge"].to_numpy(dtype=float)[take],
        "stake": stake, "profit": profit, "fee": fee, "won": won,
        "clv": signed_clv,
    })


def evaluate(df: pd.DataFrame, model: str, venue: Venue, threshold: float = 0.0,
             staking: str = "flat", fraction: float = DEFAULT_KELLY_FRACTION,
             cap: float = DEFAULT_KELLY_CAP, draws: int = BOOTSTRAP_DRAWS,
             seed: int = 0, group_col: str | None = None) -> dict:
    """Metrics for one (model, venue, threshold, staking) cell."""
    bets = bet_frame(df, model, venue, threshold, staking, fraction, cap)
    row = {"venue": venue.name, "model": model, "threshold": threshold,
           "staking": staking, "n_games": int(len(df))}
    row.update(summarize(bets, draws=draws, seed=seed, group_col=group_col))
    return row


def run_exam(preds: pd.DataFrame, closes: pd.DataFrame, models: list[str],
             venues: list[Venue] | None = None,
             thresholds=THRESHOLDS, stakings=("flat", "kelly"),
             fraction: float = DEFAULT_KELLY_FRACTION, cap: float = DEFAULT_KELLY_CAP,
             draws: int = BOOTSTRAP_DRAWS, seed: int = 0) -> pd.DataFrame:
    """The whole grid: every model at every venue, threshold and staking rule."""
    venues = venues if venues is not None else list(VENUES.values())
    rows = []
    for venue in venues:
        df = add_controls(join_closes(preds, closes, venue.name), seed=seed)
        present = [m for m in models if m in df.columns]
        for model in present:
            for threshold in thresholds:
                for staking in stakings:
                    rows.append(evaluate(df, model, venue, threshold, staking,
                                         fraction, cap, draws, seed))
    return pd.DataFrame(rows)


# ═════════════════════════════ the maker exam ═════════════════════════════
#
# The taker exam above charges the whole cost of crossing the book: the spread
# plus the taker fee, which on these games is 4.0 + 4.4 points of ROI — more
# than any edge the station-E stack has ever shown. The obvious question is
# whether the trade survives if we stop crossing and start *quoting*: rest a
# limit order below our own fair value and let the market come to us.
#
# That question cannot be answered from a closing quote, because a resting
# order is filled by the price *path*, not by its last point. It can be
# answered from the hourly candle archive (`src/market/backfill.py`,
# `data/market/kalshi_candles_2026.parquet`), which keeps the traded high, low
# and volume of every hour of the 24 before first pitch.
#
# The rule simulated here, for one game and one model:
#
#   * the model is above the market's close → post a **YES** bid at
#     ``q = P_model(home) − m``; below it → post a **NO** bid at
#     ``q = (1 − P_model(home)) − m``. `m` is the margin: how far inside our own
#     fair value we insist on being paid for providing liquidity.
#   * the order is live from T−24h to first pitch, one contract per game, and
#     is **cancelled unfilled at first pitch** — no in-game exposure.
#   * it fills in the first hour whose traded low reached the bid on non-zero
#     volume (mirrored on the NO side: the first hour whose ``1 − high``
#     reached it), at the limit price. Price improvement is never assumed.
#
# What this buys and what it costs, stated up front: the fee falls from the
# taker's ~2¢ to the maker's ~0.44¢ and the spread is earned rather than paid,
# but the fills are *selected* — an order rests until someone wants to sell to
# it, and the someone who wants to sell to it is disproportionately someone who
# knows something (a scratched starter, a lineup, weather). That adverse
# selection is real and this simulation cannot see it, so a maker P&L is an
# upper bound in a way the taker P&L is not. See the caveats in
# docs/money-exam-2026.md.

MAKER_MARGINS = (0.00, 0.01, 0.02, 0.03, 0.05)
MAKER_HOURS = 24                # how long before first pitch the order rests
TICK = 0.01                     # Kalshi's price grid is whole cents
PRICE_FLOOR, PRICE_CEIL = 0.01, 0.99
EPS = 1e-9


def maker_fee_per_contract(price, rate: float = KALSHI_MAKER_RATE,
                           round_cents: bool = False):
    """Kalshi's maker fee for one contract at `price` dollars.

    Same shape as the taker fee with a quarter of the rate: 0.0175·P·(1−P),
    maximal at a coin flip (0.44¢). The rounding is *not* to the cent by
    default — the surviving first-party page
    (docs.kalshi.com/getting_started/fee_rounding) says the charged fee is
    ``ceil_6dp(model_fee)`` accumulated per order, and rounding a 0.44¢ fee up
    to a whole cent would more than double it. `round_cents=True` prices the
    less charitable reading.
    """
    return fee_per_contract(price, rate, round_cents)


def limit_price(p_model, p_ref, margin: float, tick: float = TICK):
    """(side, limit) for one game: which side to quote and at what price.

    Above the reference price we bid for YES at ``P − m``; below it we bid for
    NO at ``(1 − P) − m``. Equality quotes nothing. The bid is floored to the
    exchange's cent grid, which is the conservative direction — a lower bid is
    harder to fill, never easier — and clipped to [1¢, 99¢].

    **The limit price is a function of the model's probability alone.** No
    quantity from the candle archive enters it; the candles only answer
    whether the price came to it. That is the leakage guard, and it is
    unit-tested.
    """
    p = np.asarray(p_model, dtype=float)
    ref = np.asarray(p_ref, dtype=float)
    yes, no = p > ref, p < ref
    raw = np.where(yes, p - margin, np.where(no, (1.0 - p) - margin, np.nan))
    limit = np.floor(raw / tick + EPS) * tick
    limit = np.clip(limit, PRICE_FLOOR, PRICE_CEIL)
    side = np.where(yes, "yes", np.where(no, "no", NO_BET))
    limit = np.where(side == NO_BET, np.nan, limit)
    return side, limit


def first_fill(candles, side: str, limit: float, first_pitch_ts: int | None = None,
               start_ts: int | None = None):
    """The hour a resting order would have been filled in, or None.

    `candles` is the hourly OHLC for one market as a frame or a list of dicts
    (`src/market/backfill.CANDLE_COLUMNS`). A YES bid at `limit` is filled the
    first hour that **traded** (volume > 0) at or below it; a NO bid at `limit`
    is the same statement about ``1 − high``, because buying NO at `limit` is
    selling YES at ``1 − limit``. Hours outside [start, first pitch] are not
    the order's to be filled in, and an order that reaches first pitch unfilled
    is cancelled — this returns None and the game is a no-trade.

    Optimistic in exactly one way: it assumes our order was at the front of the
    queue at that price. It is not optimistic about the price — we are filled
    at our own limit, never at the better price the low may have printed.
    """
    if candles is None:
        return None                 # market never archived: it cannot fill
    if isinstance(candles, pd.DataFrame):
        if candles.empty:
            return None
        ts = candles["end_period_ts"].to_numpy(dtype="int64")
        low = candles["price_low"].to_numpy(dtype=float)
        high = candles["price_high"].to_numpy(dtype=float)
        vol = candles["volume"].to_numpy(dtype=float)
    else:
        rows = sorted(candles, key=lambda c: int(c["end_period_ts"]))
        if not rows:
            return None
        ts = np.array([int(c["end_period_ts"]) for c in rows], dtype="int64")
        low = np.array([np.nan if c.get("price_low") is None else c["price_low"]
                        for c in rows], dtype=float)
        high = np.array([np.nan if c.get("price_high") is None else c["price_high"]
                         for c in rows], dtype=float)
        vol = np.array([float(c.get("volume") or 0.0) for c in rows], dtype=float)
    order = np.argsort(ts, kind="stable")
    ts, low, high, vol = ts[order], low[order], high[order], vol[order]

    live = np.ones(ts.shape, dtype=bool)
    if start_ts is not None:
        live &= ts > start_ts
    if first_pitch_ts is not None:
        live &= ts <= first_pitch_ts
    touched = (low <= limit + EPS) if side == "yes" else ((1.0 - high) <= limit + EPS)
    hit = live & (vol > 0) & np.isfinite(low) & np.isfinite(high) & touched
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    return {"end_period_ts": int(ts[i]), "price_low": float(low[i]),
            "price_high": float(high[i]), "volume": float(vol[i])}


CANDLE_FIELDS = ["end_period_ts", "price_low", "price_high", "price_close",
                 "volume", "yes_bid_close", "yes_ask_close"]


def candle_index(candles: pd.DataFrame) -> dict:
    """market_id → the arrays the maker exam needs, built once per exam."""
    out = {}
    if candles is None or len(candles) == 0:
        return out
    cols = [c for c in CANDLE_FIELDS if c in candles.columns]
    for market_id, g in candles.sort_values("end_period_ts").groupby("market_id"):
        out[str(market_id)] = g[cols].reset_index(drop=True)
    return out


# ───────────────────────── posting, filling, settling ─────────────────────────

def maker_bet_frame(df: pd.DataFrame, model: str, candles: dict, margin: float,
                    venue: Venue = KALSHI, staking: str = "flat",
                    fraction: float = DEFAULT_KELLY_FRACTION,
                    cap: float = DEFAULT_KELLY_CAP, bankroll: float = BANKROLL,
                    hours: int = MAKER_HOURS, anchor: str = "close",
                    maker_round_cents: bool = False) -> pd.DataFrame:
    """One row per order *posted* — filled or not — priced and settled.

    `anchor` is what the model is compared against to pick a side: `close`
    (the pre-pitch close, the same reference the taker exam uses) or `post`
    (the traded price of the first archived hour, which is the price actually
    on the screen when the order goes in and therefore the only choice with no
    hindsight in it at all). The limit *price* never depends on either.

    Unfilled orders are kept, with zero stake and zero profit, because the
    denominator of a maker's return is the contracts he *posted*.
    """
    ref = df["p_home_close"].to_numpy(dtype=float)
    market_ids = df["market_id"].astype(str).to_numpy()
    first_pitch = df["first_pitch_ts"].to_numpy(dtype="int64")
    if anchor == "post":
        ref = np.array([_post_price(candles.get(m), fp - hours * 3600, fp, r)
                        for m, fp, r in zip(market_ids, first_pitch, ref)], dtype=float)
    elif anchor != "close":
        raise ValueError(f"unknown anchor {anchor!r}")

    side, limit = limit_price(df[model].to_numpy(dtype=float), ref, margin)
    posted = side != NO_BET
    if not posted.any():
        return pd.DataFrame(columns=["date", "game_pk", "side", "limit", "filled",
                                     "fill_ts", "marketable", "quoted", "contracts",
                                     "stake", "profit", "fee", "won", "edge"])
    p = df[model].to_numpy(dtype=float)[posted]
    side, limit = side[posted], limit[posted].astype(float)
    market_ids, first_pitch = market_ids[posted], first_pitch[posted]
    home_won = df["home_win"].to_numpy(dtype=bool)[posted]
    close = df["p_home_close"].to_numpy(dtype=float)[posted]

    fills = [first_fill(candles.get(m), s, q, int(fp), int(fp) - hours * 3600)
             for m, s, q, fp in zip(market_ids, side, limit, first_pitch)]
    filled = np.array([f is not None for f in fills], dtype=bool)
    fill_ts = np.array([f["end_period_ts"] if f else 0 for f in fills], dtype="int64")
    marketable = np.array([
        _marketable(candles.get(m), int(fp) - hours * 3600, int(fp), s, q)
        for m, fp, s, q in zip(market_ids, first_pitch, side, limit)], dtype=bool)

    p_win = np.where(side == "yes", p, 1.0 - p)
    if staking == "flat":
        quoted = np.ones(len(limit), dtype=float)             # one contract per game
    elif staking == "kelly":
        quoted = kelly_stake(p_win, limit, fraction, cap, bankroll) / limit
    else:
        raise ValueError(f"unknown staking rule {staking!r}")
    contracts = np.where(filled, quoted, 0.0)                 # only fills are held

    won = np.where(side == "yes", home_won, ~home_won)
    fee = contracts * maker_fee_per_contract(limit, venue.maker_rate, maker_round_cents)
    stake = contracts * limit
    profit = contracts * won.astype(float) - stake - fee
    out = pd.DataFrame({
        "date": df["date"].to_numpy()[posted],
        "game_pk": df["game_pk"].to_numpy()[posted],
        "side": side, "limit": limit, "filled": filled, "fill_ts": fill_ts,
        "marketable": marketable,
        "quoted": quoted, "contracts": contracts, "stake": stake,
        "profit": profit, "fee": fee,
        "won": np.where(filled, won, False),
        "edge": np.where(side == "yes", p - close, close - p),
    })
    # A zero-size order is not an order. Kelly declines to size a quote whose
    # limit is its own fair value, and those games are no-quotes, not no-fills.
    return out[out["quoted"] > 0].reset_index(drop=True)


def _first_live(cd, start_ts: int, first_pitch_ts: int, column: str):
    """`column` in the first archived hour the order is live in, or None."""
    if cd is None or len(cd) == 0 or column not in cd:
        return None
    ts = cd["end_period_ts"].to_numpy(dtype="int64")
    keep = (ts > start_ts) & (ts <= first_pitch_ts)
    if not keep.any():
        return None
    v = cd[column].to_numpy(dtype=float)[keep]
    good = np.isfinite(v)
    return float(v[good][0]) if good.any() else None


def _post_price(cd, start_ts: int, first_pitch_ts: int, fallback: float) -> float:
    """The traded price of the first hour the order is live in (the `post` anchor)."""
    v = _first_live(cd, start_ts, first_pitch_ts, "price_close")
    return float(fallback) if v is None else v


def _marketable(cd, start_ts: int, first_pitch_ts: int, side: str,
                limit: float) -> bool:
    """Would this "limit order" have crossed the book the moment it was posted?

    A bid at or above the prevailing ask is not a maker order at all: the
    exchange fills it immediately against the resting offer and charges the
    *taker* fee. The rule as specified can produce one — at a margin of zero
    the bid is the model's own price, which is above the market whenever the
    model disagrees at all — so the exam counts them and reports the rate
    rather than quietly booking them as maker fills.
    """
    if side == "yes":
        ask = _first_live(cd, start_ts, first_pitch_ts, "yes_ask_close")
        return ask is not None and limit >= ask - EPS
    bid = _first_live(cd, start_ts, first_pitch_ts, "yes_bid_close")
    return bid is not None and (1.0 - limit) <= bid + EPS


def maker_summarize(bets: pd.DataFrame, n_games: int, draws: int = BOOTSTRAP_DRAWS,
                    seed: int = 0, group_col: str | None = None) -> dict:
    """Money metrics for one maker cell.

    Three denominators, because a maker has three honest ones. **Per posted
    contract** is the one that matters — it charges the strategy for the games
    it wanted and did not get. **Per filled contract** says how good the fills
    themselves were. **Per game** puts every model on the same 800-odd-game
    footing whether or not it wanted to quote. ROI (profit over capital at
    risk) is reported too, because it is the number the taker table is in.

    `group_col` makes both bootstraps **clustered**, and props need it: one
    game carries dozens of contracts on the same afternoon's at bats, and
    resampling rows would report an interval far tighter than the data
    supports. On moneylines there is one order per game and it changes
    nothing.
    """
    empty = {"n_posted": 0, "n_filled": 0, "fill_rate": float("nan"),
             "marketable_rate": float("nan"),
             "contracts_posted": 0.0, "contracts_filled": 0.0,
             "hit_rate": float("nan"), "total_staked": 0.0, "total_return": 0.0,
             "roi": 0.0, "roi_lo": 0.0, "roi_hi": 0.0,
             "pnl_per_posted": float("nan"), "pnl_per_filled": float("nan"),
             "pnl_lo": float("nan"), "pnl_hi": float("nan"),
             "pnl_per_game": 0.0, "mean_edge": float("nan"),
             "mean_limit": float("nan"), "max_drawdown": 0.0, "total_fees": 0.0}
    if len(bets) == 0:
        return empty
    filled = bets[bets["filled"]]
    profit = bets["profit"].to_numpy(dtype=float)
    stake = bets["stake"].to_numpy(dtype=float)
    # The denominator that counts is what we *quoted*, filled or not: an order
    # that never traded still represents capital the strategy asked to risk.
    quoted = bets["quoted"].to_numpy(dtype=float)
    groups = bets[group_col].to_numpy() if group_col and group_col in bets else None
    roi_lo, roi_hi = bootstrap_roi_ci(profit, stake, draws, seed, groups=groups)
    pnl_lo, pnl_hi = bootstrap_roi_ci(profit, quoted, draws, seed, groups=groups)
    total = float(profit.sum())
    return {
        "n_posted": int(len(bets)),
        "n_filled": int(len(filled)),
        "fill_rate": float(bets["filled"].mean()),
        "marketable_rate": float(bets["marketable"].mean())
        if "marketable" in bets else float("nan"),
        "contracts_posted": float(quoted.sum()),
        "contracts_filled": float(filled["contracts"].sum()),
        "hit_rate": float(filled["won"].mean()) if len(filled) else float("nan"),
        "total_staked": float(stake.sum()),
        "total_return": total,
        "roi": total / stake.sum() if stake.sum() > 0 else 0.0,
        "roi_lo": roi_lo, "roi_hi": roi_hi,
        "pnl_per_posted": total / quoted.sum() if quoted.sum() else float("nan"),
        "pnl_per_filled": total / filled["contracts"].sum() if filled["contracts"].sum() else float("nan"),
        "pnl_lo": pnl_lo, "pnl_hi": pnl_hi,
        "pnl_per_game": total / n_games if n_games else 0.0,
        "mean_edge": float(bets["edge"].mean()),
        "mean_limit": float(bets["limit"].mean()),
        "max_drawdown": max_drawdown(profit[bets["filled"].to_numpy()]),
        "total_fees": float(bets["fee"].sum()),
    }


def maker_evaluate(df: pd.DataFrame, model: str, candles: dict, margin: float,
                   venue: Venue = KALSHI, staking: str = "flat",
                   fraction: float = DEFAULT_KELLY_FRACTION,
                   cap: float = DEFAULT_KELLY_CAP, hours: int = MAKER_HOURS,
                   anchor: str = "close", maker_round_cents: bool = False,
                   draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
                   group_col: str | None = None) -> dict:
    """Metrics for one (model, margin, staking) maker cell."""
    bets = maker_bet_frame(df, model, candles, margin, venue, staking, fraction,
                           cap, hours=hours, anchor=anchor,
                           maker_round_cents=maker_round_cents)
    row = {"venue": venue.name, "model": model, "margin": margin,
           "staking": staking, "n_games": int(len(df))}
    row.update(maker_summarize(bets, len(df), draws=draws, seed=seed,
                               group_col=group_col))
    return row


# ───────────────────────── controls and the split ─────────────────────────

def shuffled_edge(df: pd.DataFrame, model: str, seed: int = 0) -> np.ndarray:
    """The control: the close plus this model's *own* edges, dealt to the wrong games.

    A maker's return depends on the margin and therefore on how big the edges
    are, so the null has to have the same edge distribution as the model it
    controls for — N(0, 3 pts) would be a different strategy, not a control.
    Permuting the model's signed disagreements across games keeps that
    distribution exactly and destroys only the one thing under test: whether
    the disagreement is attached to the right game.
    """
    close = df["p_home_close"].to_numpy(dtype=float)
    e = df[model].to_numpy(dtype=float) - close
    rng = np.random.default_rng(seed)
    return np.clip(close + rng.permutation(e), PRICE_FLOOR, PRICE_CEIL)


def add_maker_controls(df: pd.DataFrame, models: list[str], seed: int = 0) -> tuple:
    """One shuffled-edge control per model, named `<model>__shuffled`."""
    out = df.copy()
    names = []
    for i, m in enumerate(models):
        if m not in out.columns:
            continue
        name = f"{m}__shuffled"
        out[name] = shuffled_edge(out, m, seed=seed + i)
        names.append(name)
    return out, names


def split_halves(df: pd.DataFrame, date_col: str = "date") -> tuple:
    """(first half, second half) of the game window by date.

    The margin is a free parameter and a free parameter chosen on the data it
    is scored on is not a result. It is chosen on the first half of the season
    and scored on the second, and both halves are reported so the reader can
    see whether the choice travelled.
    """
    d = df.sort_values(date_col).reset_index(drop=True)
    dates = pd.Index(sorted(d[date_col].astype(str).unique()))
    cut = dates[len(dates) // 2]
    first = d[d[date_col].astype(str) < cut].reset_index(drop=True)
    second = d[d[date_col].astype(str) >= cut].reset_index(drop=True)
    return first, second


def prepare_maker(preds: pd.DataFrame, closes: pd.DataFrame, venue: str = "kalshi",
                  hours: int = MAKER_HOURS) -> pd.DataFrame:
    """The joined frame the maker exam runs on: predictions + close + first pitch."""
    df = join_closes(preds, closes, venue)
    if "market_id" not in df.columns:
        raise ValueError("closes frame has no market_id; the candles cannot be joined")
    if "first_pitch_ts" not in df.columns:
        if "game_start" in df.columns:
            df["first_pitch_ts"] = [
                int(pd.Timestamp(str(s)).timestamp()) for s in df["game_start"]]
        else:
            # No first pitch on the frame: the close is the last pre-pitch
            # observation, so it is the tightest available upper bound.
            df["first_pitch_ts"] = df["close_ts"].astype("int64")
    return df


def maker_grid(df: pd.DataFrame, candles: pd.DataFrame, models: list[str],
               margins=MAKER_MARGINS, venue: Venue = KALSHI,
               stakings=("flat", "kelly"), fraction: float = DEFAULT_KELLY_FRACTION,
               cap: float = DEFAULT_KELLY_CAP, hours: int = MAKER_HOURS,
               anchor: str = "close", maker_round_cents: bool = False,
               draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
               split: bool = True, group_col: str | None = None,
               date_col: str = "date") -> pd.DataFrame:
    """Every model at every margin, on each half of the window, plus controls.

    Takes an already-joined frame — one row per order the strategy could post,
    carrying `date`, `game_pk`, `market_id`, `first_pitch_ts`, `p_home_close`
    and `home_win` — so a moneyline (one order per game) and a player prop
    (dozens per game) run the same grid rather than two copies of it.

    Rows carry a `half` label: `first` is where the margin may be chosen,
    `second` is where the number that counts is scored, and `all` is the whole
    window for reference.
    """
    df, control_names = add_maker_controls(df, models, seed=seed)
    index = candle_index(candles)
    halves = {"all": df}
    if split:
        first, second = split_halves(df, date_col)
        halves.update({"first": first, "second": second})
    rows = []
    for half, sub in halves.items():
        if len(sub) == 0:
            continue
        for model in [m for m in models if m in sub.columns] + control_names:
            for margin in margins:
                for staking in stakings:
                    row = maker_evaluate(sub, model, index, margin, venue, staking,
                                         fraction, cap, hours, anchor,
                                         maker_round_cents, draws, seed,
                                         group_col=group_col)
                    row["half"] = half
                    row["first_date"] = str(sub[date_col].min())
                    row["last_date"] = str(sub[date_col].max())
                    rows.append(row)
    return pd.DataFrame(rows)


def run_maker_exam(preds: pd.DataFrame, closes: pd.DataFrame, candles: pd.DataFrame,
                   models: list[str], margins=MAKER_MARGINS, venue: Venue = KALSHI,
                   stakings=("flat", "kelly"), fraction: float = DEFAULT_KELLY_FRACTION,
                   cap: float = DEFAULT_KELLY_CAP, hours: int = MAKER_HOURS,
                   anchor: str = "close", maker_round_cents: bool = False,
                   draws: int = BOOTSTRAP_DRAWS, seed: int = 0,
                   split: bool = True) -> pd.DataFrame:
    """The moneyline maker exam: join the closes, then run the grid."""
    df = prepare_maker(preds, closes, venue.name, hours)
    return maker_grid(df, candles, models, margins, venue, stakings, fraction,
                      cap, hours, anchor, maker_round_cents, draws, seed, split)


def choose_margin(res: pd.DataFrame, model: str, staking: str = "flat",
                  half: str = "first", by: str = "pnl_per_posted") -> float:
    """The margin that did best on the training half, by P&L per posted contract."""
    g = res[(res["model"] == model) & (res["staking"] == staking)
            & (res["half"] == half)]
    if g.empty or g[by].isna().all():
        return float("nan")
    return float(g.loc[g[by].idxmax(), "margin"])
