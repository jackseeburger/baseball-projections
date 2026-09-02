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
larger ones. Maker fees default to zero — resting a limit order is the trade
we would actually want, but a maker fill is not guaranteed, so the exam
charges the taker fee.

Polymarket takes no fee on most sports markets, but the reconstructed close is
a mid, so the cost of crossing is modelled as a configurable half-spread
(default 1¢ each side). Both rates are parameters, not constants, so a change
in either venue's schedule is a flag rather than a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KALSHI_TAKER_RATE = 0.07        # round_up_to_cent(0.07 · C · P · (1-P))
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
    """

    name: str
    taker_rate: float = KALSHI_TAKER_RATE
    half_spread: float | None = None
    round_cents: bool = True


KALSHI = Venue("kalshi", KALSHI_TAKER_RATE, None, round_cents=True)
POLYMARKET = Venue("polymarket", POLYMARKET_TAKER_RATE, DEFAULT_HALF_SPREAD,
                   round_cents=False)
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
                     alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI on ROI, resampling *games* with replacement.

    Bets are one per game and settle independently, so the game is the
    resampling unit. The ratio ΣP/ΣS is re-formed inside each draw, which is
    what makes this a CI on ROI rather than on mean profit.
    """
    profit = np.asarray(profit, dtype=float)
    stake = np.asarray(stake, dtype=float)
    n = profit.size
    if n == 0 or stake.sum() <= 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(draws, n))
    tot_p = profit[idx].sum(axis=1)
    tot_s = stake[idx].sum(axis=1)
    roi = np.divide(tot_p, tot_s, out=np.zeros_like(tot_p), where=tot_s > 0)
    return (float(np.quantile(roi, alpha / 2)), float(np.quantile(roi, 1 - alpha / 2)))


def summarize(bets: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS,
              seed: int = 0) -> dict:
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
    lo, hi = bootstrap_roi_ci(profit, stake, draws, seed)
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
    c = closes[closes["venue"] == venue][
        ["game_pk", "p_home_close", "bid", "ask", "home_won", "minutes_before_pitch"]
    ]
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
             seed: int = 0) -> dict:
    """Metrics for one (model, venue, threshold, staking) cell."""
    bets = bet_frame(df, model, venue, threshold, staking, fraction, cap)
    row = {"venue": venue.name, "model": model, "threshold": threshold,
           "staking": staking, "n_games": int(len(df))}
    row.update(summarize(bets, draws=draws, seed=seed))
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
