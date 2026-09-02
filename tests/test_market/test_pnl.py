"""Simulated P&L against the exchanges — synthetic games, exact arithmetic."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import pnl


def games(p_model, p_close, home_won, bid=None, ask=None, model="m"):
    """A predictions+closes frame already joined, the shape `evaluate` wants."""
    n = len(p_model)
    bid = [c - 0.01 for c in p_close] if bid is None else bid
    ask = [c + 0.01 for c in p_close] if ask is None else ask
    return pd.DataFrame({
        "date": [f"2026-07-{i + 1:02d}" for i in range(n)],
        "game_pk": range(1, n + 1),
        "home_win": home_won,
        model: p_model,
        "p_home_close": p_close,
        "bid": bid,
        "ask": ask,
    })


# ───────────────────────────── fees ─────────────────────────────

def test_taker_fee_at_known_prices():
    # Kalshi: round_up_to_cent(0.07 · C · P · (1-P)); worst case is a coin flip.
    assert pnl.fee_per_contract(0.50) == pytest.approx(0.02)      # 0.0175 → 2¢
    assert pnl.fee_per_contract(0.90) == pytest.approx(0.01)      # 0.0063 → 1¢
    assert pnl.fee_per_contract(0.99) == pytest.approx(0.01)      # 0.000693 → 1¢
    assert pnl.fee_per_contract(1.00) == pytest.approx(0.00)      # no risk, no fee


def test_taker_fee_unrounded_matches_the_formula():
    raw = pnl.fee_per_contract(0.5, rate=0.07, round_cents=False)
    assert raw == pytest.approx(0.07 * 0.5 * 0.5)
    assert pnl.fee_per_contract(0.5, rate=0.0) == 0.0             # Polymarket sports


def test_fee_is_symmetric_in_the_side_taken():
    # YES at ask a and NO at 1-a pay the same fee: P(1-P) is symmetric.
    assert pnl.fee_per_contract(0.62) == pnl.fee_per_contract(0.38)


# ───────────────────────────── the trade rule ─────────────────────────────

def test_no_trade_inside_the_spread():
    d = pnl.decide([0.50, 0.505, 0.49], bid=0.48, ask=0.52)
    assert list(d["side"]) == ["", "", ""]
    assert d["cost"].isna().all()


def test_buys_yes_at_the_ask_and_no_at_the_bid():
    d = pnl.decide([0.60, 0.40], bid=0.48, ask=0.52)
    assert list(d["side"]) == ["yes", "no"]
    assert d.loc[0, "cost"] == pytest.approx(0.52)          # pays the ask
    assert d.loc[1, "cost"] == pytest.approx(1 - 0.48)      # NO costs 1 - bid
    assert d.loc[0, "edge"] == pytest.approx(0.08)
    assert d.loc[1, "edge"] == pytest.approx(0.08)


def test_threshold_suppresses_marginal_disagreement():
    p = [0.55, 0.62]
    assert list(pnl.decide(p, 0.48, 0.52, threshold=0.00)["side"]) == ["yes", "yes"]
    assert list(pnl.decide(p, 0.48, 0.52, threshold=0.06)["side"]) == ["", "yes"]


def test_polymarket_quotes_are_the_mid_plus_a_half_spread():
    df = games([0.55], [0.50], [True], bid=[None], ask=[None])
    bid, ask = pnl.quotes(df, pnl.POLYMARKET)
    assert (bid[0], ask[0]) == pytest.approx((0.49, 0.51))


# ───────────────────────────── settlement ─────────────────────────────

def test_winning_yes_pays_one_minus_ask_minus_fee():
    ask = 0.52
    profit, fee = pnl.settle(["yes"], [ask], [ask], [True], pnl.KALSHI)   # 1 contract
    assert fee[0] == pytest.approx(pnl.fee_per_contract(ask))
    assert profit[0] == pytest.approx(1 - ask - fee[0])


def test_winning_no_pays_bid_minus_fee():
    bid, cost = 0.48, 1 - 0.48
    profit, fee = pnl.settle(["no"], [cost], [cost], [False], pnl.KALSHI)
    assert profit[0] == pytest.approx(bid - fee[0])


def test_losing_bet_loses_the_stake_and_still_pays_the_fee():
    ask = 0.52
    profit, fee = pnl.settle(["yes"], [ask], [ask], [False], pnl.KALSHI)
    assert profit[0] == pytest.approx(-ask - fee[0])
    assert fee[0] > 0


def test_one_unit_of_stake_buys_one_over_cost_contracts():
    # A unit staked on a 25¢ contract holds four contracts, so it pays 3u gross.
    profit, fee = pnl.settle(["yes"], [0.25], [1.0], [True], pnl.KALSHI)
    assert profit[0] == pytest.approx(3.0 - 4 * pnl.fee_per_contract(0.25))


# ───────────────────────────── stakes ─────────────────────────────

def test_kelly_is_the_edge_over_the_cost_times_the_fraction():
    f = pnl.kelly_stake(0.60, 0.50, fraction=0.25, cap=1.0)
    assert f == pytest.approx(0.25 * (0.60 - 0.50) / 0.50)


def test_kelly_cap_binds_on_a_large_edge():
    uncapped = pnl.kelly_stake(0.90, 0.50, fraction=0.25, cap=1.0)
    capped = pnl.kelly_stake(0.90, 0.50, fraction=0.25, cap=pnl.DEFAULT_KELLY_CAP)
    assert uncapped == pytest.approx(0.20) and uncapped > pnl.DEFAULT_KELLY_CAP
    assert capped == pytest.approx(pnl.DEFAULT_KELLY_CAP)


def test_kelly_never_stakes_on_a_negative_edge():
    assert pnl.kelly_stake(0.40, 0.50, fraction=0.25, cap=0.05) == 0.0


def test_flat_staking_is_one_unit_per_bet():
    s = pnl.stakes(["yes", "no"], [0.6, 0.4], [0.52, 0.52], staking="flat")
    assert list(s) == [1.0, 1.0]


# ───────────────────────────── metrics ─────────────────────────────

def test_clv_is_positive_toward_the_side_taken():
    # One YES above the close and one NO below it: both bet the side they liked.
    df = games([0.60, 0.40], [0.50, 0.50], [True, False])
    bets = pnl.bet_frame(df, "m", pnl.KALSHI)
    assert list(bets["side"]) == ["yes", "no"]
    assert bets.loc[0, "clv"] == pytest.approx(0.10)
    assert bets.loc[1, "clv"] == pytest.approx(0.10)
    assert bets["clv"].min() > 0            # the rule cannot take a negative one


def test_max_drawdown_is_the_deepest_peak_to_trough():
    assert pnl.max_drawdown([1.0, -0.5, -0.75, 2.0]) == pytest.approx(1.25)
    assert pnl.max_drawdown([1.0, 2.0]) == 0.0
    assert pnl.max_drawdown([]) == 0.0


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(7)
    profit = rng.normal(-0.05, 1.0, 300)
    stake = np.ones(300)
    lo, hi = pnl.bootstrap_roi_ci(profit, stake, draws=500, seed=1)
    point = profit.sum() / stake.sum()
    assert lo < point < hi


def test_bootstrap_ci_is_seeded_and_reproducible():
    profit, stake = np.arange(-10.0, 10.0), np.ones(20)
    assert pnl.bootstrap_roi_ci(profit, stake, draws=200, seed=3) == \
        pnl.bootstrap_roi_ci(profit, stake, draws=200, seed=3)


# ───────────────────────────── controls ─────────────────────────────

def test_market_as_a_model_never_trades_and_returns_exactly_zero():
    df = pnl.add_controls(games([0.6, 0.3, 0.5], [0.55, 0.35, 0.50],
                                [True, False, True]))
    row = pnl.evaluate(df, "market", pnl.KALSHI, threshold=0.0, draws=50)
    assert row["n_bets"] == 0
    assert row["roi"] == 0.0 and row["total_return"] == 0.0
    assert row["max_drawdown"] == 0.0


def test_market_never_trades_on_a_mid_only_venue_either():
    df = pnl.add_controls(games([0.6], [0.55], [True], bid=[None], ask=[None]))
    assert pnl.evaluate(df, "market", pnl.POLYMARKET, draws=50)["n_bets"] == 0


def test_random_edge_control_is_seeded_and_trades():
    df = games([0.5] * 200, [0.5] * 200, [True, False] * 100)
    a = pnl.add_controls(df, seed=11)["random_edge"].to_numpy()
    b = pnl.add_controls(df, seed=11)["random_edge"].to_numpy()
    assert np.allclose(a, b)
    assert pnl.evaluate(pnl.add_controls(df, seed=11), "random_edge",
                        pnl.KALSHI, draws=50)["n_bets"] > 0


# ───────────────────────────── end to end ─────────────────────────────

def test_run_exam_grid_covers_every_cell_and_joins_on_game_pk():
    preds = games([0.60, 0.40, 0.50], [0.55, 0.45, 0.50], [True, False, True])
    preds = preds.drop(columns=["p_home_close", "bid", "ask"])
    closes = pd.DataFrame({
        "venue": ["kalshi"] * 3,
        "game_pk": [1, 2, 3],
        "p_home_close": [0.55, 0.45, 0.50],
        "bid": [0.54, 0.44, 0.49], "ask": [0.56, 0.46, 0.51],
        "home_won": [True, False, True],
        "minutes_before_pitch": [15.0, 15.0, 15.0],
    })
    res = pnl.run_exam(preds, closes, ["m", "market"], [pnl.KALSHI],
                       thresholds=(0.0, 0.02), draws=50)
    assert len(res) == 2 * 2 * 2                      # models x thresholds x staking
    assert set(res["n_games"]) == {3}
    m0 = res[(res.model == "m") & (res.threshold == 0.0) & (res.staking == "flat")]
    assert int(m0["n_bets"].iloc[0]) == 2             # the 0.50 game is inside the spread


def test_join_refuses_closes_that_disagree_about_the_winner():
    preds = games([0.6], [0.55], [True]).drop(columns=["p_home_close", "bid", "ask"])
    closes = pd.DataFrame({"venue": ["kalshi"], "game_pk": [1], "p_home_close": [0.55],
                           "bid": [0.54], "ask": [0.56], "home_won": [False],
                           "minutes_before_pitch": [15.0]})
    with pytest.raises(ValueError, match="disagree"):
        pnl.join_closes(preds, closes, "kalshi")


def test_join_keeps_the_first_prediction_for_a_resumed_game():
    # A suspended game is scored on both dates; the exchange has one contract.
    preds = games([0.60, 0.30], [0.55, 0.55], [True, True]).drop(
        columns=["p_home_close", "bid", "ask"])
    preds["game_pk"] = [1, 1]
    closes = pd.DataFrame({"venue": ["kalshi"], "game_pk": [1], "p_home_close": [0.55],
                           "bid": [0.54], "ask": [0.56], "home_won": [True],
                           "minutes_before_pitch": [15.0]})
    df = pnl.join_closes(preds, closes, "kalshi")
    assert len(df) == 1 and df.loc[0, "m"] == pytest.approx(0.60)
