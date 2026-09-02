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


# ═══════════════════════════ the maker exam ═══════════════════════════

FIRST_PITCH = 1_800_000_000


def hour(h_before, low, high, volume=100.0, close=None):
    """One hourly candle `h_before` hours before first pitch."""
    return {"market_id": "M1", "game_pk": 1,
            "end_period_ts": FIRST_PITCH - h_before * 3600,
            "price_low": low, "price_high": high,
            "price_close": close if close is not None else high,
            "volume": volume}


def candles(rows):
    return pd.DataFrame(rows)


def maker_games(p_model, p_close, home_won, market_ids=None, model="m"):
    """A joined frame in the shape the maker exam wants."""
    n = len(p_model)
    return pd.DataFrame({
        "date": [f"2026-07-{i + 1:02d}" for i in range(n)],
        "game_pk": range(1, n + 1),
        "home_win": home_won,
        model: p_model,
        "p_home_close": p_close,
        "market_id": market_ids or [f"M{i + 1}" for i in range(n)],
        "first_pitch_ts": [FIRST_PITCH] * n,
    })


# ───────────────────────────── the fee ─────────────────────────────

def test_maker_fee_is_a_quarter_of_the_taker_fee():
    # Kalshi's July 2026 schedule: maker = 25% of taker, 0.44¢ max at 50¢.
    assert pnl.maker_fee_per_contract(0.50) == pytest.approx(0.004375)
    assert pnl.maker_fee_per_contract(0.50) == pytest.approx(
        0.25 * pnl.fee_per_contract(0.50, pnl.KALSHI_TAKER_RATE, round_cents=False))
    assert pnl.maker_fee_per_contract(0.50) < 0.0045          # under half a cent


def test_maker_fee_is_not_rounded_to_the_cent_by_default():
    # ceil_6dp per the surviving first-party rounding page; rounding a 0.44¢
    # fee up to a whole cent would more than double it.
    assert pnl.maker_fee_per_contract(0.50, round_cents=True) == pytest.approx(0.01)
    assert pnl.maker_fee_per_contract(0.50) < pnl.maker_fee_per_contract(
        0.50, round_cents=True)


# ───────────────────────── the limit price ─────────────────────────

def test_limit_price_quotes_yes_below_the_model_and_no_above_it():
    side, q = pnl.limit_price([0.60, 0.40], [0.50, 0.50], margin=0.02)
    assert list(side) == ["yes", "no"]
    assert q[0] == pytest.approx(0.58)          # 0.60 - 0.02 on the YES side
    assert q[1] == pytest.approx(0.58)          # (1 - 0.40) - 0.02 on the NO side


def test_limit_price_quotes_nothing_when_the_model_is_the_price():
    side, q = pnl.limit_price([0.50], [0.50], margin=0.01)
    assert list(side) == [""] and np.isnan(q[0])


def test_limit_price_floors_to_the_cent_grid_and_clips():
    side, q = pnl.limit_price([0.567, 0.999], [0.50, 0.50], margin=0.0)
    assert q[0] == pytest.approx(0.56)          # floored, never rounded up
    assert q[1] == pytest.approx(0.99)          # clipped to the top of the grid


def test_limit_price_is_a_function_of_the_model_alone():
    """The leakage guard: nothing from the candles may enter the order price.

    Same model probability, wildly different price paths — the limit must not
    move. Only the *side* looks at the market, and only at the close.
    """
    a = pnl.limit_price([0.62], [0.50], margin=0.03)[1]
    b = pnl.limit_price([0.62], [0.01], margin=0.03)[1]
    assert a[0] == pytest.approx(b[0]) == pytest.approx(0.59)


# ───────────────────────────── the fill rule ─────────────────────────────

def test_fills_in_the_first_hour_whose_low_reaches_the_bid():
    cs = candles([hour(5, low=0.60, high=0.62), hour(4, low=0.57, high=0.61),
                  hour(3, low=0.50, high=0.58)])
    fill = pnl.first_fill(cs, "yes", 0.58, FIRST_PITCH)
    assert fill is not None
    assert fill["end_period_ts"] == FIRST_PITCH - 4 * 3600     # the first one, not the deepest


def test_no_fill_when_the_low_never_reaches_the_bid():
    cs = candles([hour(5, low=0.60, high=0.62), hour(4, low=0.59, high=0.63)])
    assert pnl.first_fill(cs, "yes", 0.58, FIRST_PITCH) is None


def test_no_fill_on_zero_volume_however_low_the_print():
    # An hour that did not trade is a carried-forward quote, not a fill.
    cs = candles([hour(5, low=0.30, high=0.70, volume=0.0)])
    assert pnl.first_fill(cs, "yes", 0.58, FIRST_PITCH) is None
    cs2 = candles([hour(5, low=0.30, high=0.70, volume=0.0),
                   hour(4, low=0.55, high=0.60, volume=12.0)])
    assert pnl.first_fill(cs2, "yes", 0.58, FIRST_PITCH)["end_period_ts"] == \
        FIRST_PITCH - 4 * 3600


def test_no_side_mirrors_on_one_minus_the_high():
    # A NO bid at 0.58 is a YES offer at 0.42: it fills when the high reaches 0.42.
    cs = candles([hour(5, low=0.30, high=0.41), hour(4, low=0.35, high=0.45)])
    assert pnl.first_fill(cs, "no", 0.58, FIRST_PITCH)["end_period_ts"] == \
        FIRST_PITCH - 4 * 3600
    # A cheaper NO bid needs the YES price to run further up: 0.50 needs high ≥ 0.50.
    assert pnl.first_fill(cs, "no", 0.50, FIRST_PITCH) is None


def test_the_order_is_cancelled_at_first_pitch():
    # The price came to the bid, but only after the game started.
    cs = candles([hour(3, low=0.60, high=0.62), hour(-1, low=0.20, high=0.30)])
    assert pnl.first_fill(cs, "yes", 0.58, FIRST_PITCH) is None
    assert pnl.first_fill(cs, "yes", 0.58, first_pitch_ts=None) is not None


def test_the_order_is_not_live_before_it_is_posted():
    cs = candles([hour(30, low=0.10, high=0.20), hour(2, low=0.60, high=0.65)])
    start = FIRST_PITCH - 24 * 3600
    assert pnl.first_fill(cs, "yes", 0.58, FIRST_PITCH, start_ts=start) is None
    assert pnl.first_fill(cs, "yes", 0.58, FIRST_PITCH) is not None


def test_first_fill_accepts_dicts_and_an_empty_book():
    rows = [hour(2, low=0.55, high=0.60), hour(3, low=0.70, high=0.75)]
    assert pnl.first_fill(rows, "yes", 0.58, FIRST_PITCH)["end_period_ts"] == \
        FIRST_PITCH - 2 * 3600                      # unsorted input is sorted
    assert pnl.first_fill([], "yes", 0.58, FIRST_PITCH) is None
    assert pnl.first_fill(candles([]).assign(), "yes", 0.58) is None


# ───────────────────────── posting and settling ─────────────────────────

def test_a_filled_yes_pays_one_minus_the_limit_minus_the_maker_fee():
    df = maker_games([0.60], [0.50], [True])
    idx = {"M1": candles([hour(3, low=0.57, high=0.62)])}
    bets = pnl.maker_bet_frame(df, "m", idx, margin=0.02)
    assert bool(bets.loc[0, "filled"]) and bets.loc[0, "limit"] == pytest.approx(0.58)
    fee = pnl.maker_fee_per_contract(0.58)
    assert bets.loc[0, "profit"] == pytest.approx(1 - 0.58 - fee)
    assert bets.loc[0, "fee"] == pytest.approx(fee)


def test_a_filled_no_pays_one_minus_the_limit_when_the_home_team_loses():
    df = maker_games([0.40], [0.50], [False])
    idx = {"M1": candles([hour(3, low=0.35, high=0.45)])}       # 1 - high = 0.55 ≤ 0.58
    bets = pnl.maker_bet_frame(df, "m", idx, margin=0.02)
    assert list(bets["side"]) == ["no"] and bool(bets.loc[0, "filled"])
    assert bets.loc[0, "profit"] == pytest.approx(
        1 - 0.58 - pnl.maker_fee_per_contract(0.58))


def test_an_unfilled_order_costs_nothing_and_still_counts_as_posted():
    df = maker_games([0.60], [0.50], [True])
    idx = {"M1": candles([hour(3, low=0.62, high=0.65)])}
    bets = pnl.maker_bet_frame(df, "m", idx, margin=0.02)
    assert len(bets) == 1 and not bool(bets.loc[0, "filled"])
    assert bets.loc[0, "profit"] == 0.0 and bets.loc[0, "stake"] == 0.0
    row = pnl.maker_summarize(bets, n_games=1, draws=50)
    assert row["n_posted"] == 1 and row["n_filled"] == 0
    assert row["fill_rate"] == 0.0 and row["pnl_per_posted"] == 0.0


def test_a_market_with_no_archived_candles_never_fills():
    df = maker_games([0.60], [0.50], [True])
    bets = pnl.maker_bet_frame(df, "m", {}, margin=0.0)
    assert len(bets) == 1 and not bool(bets.loc[0, "filled"])


def test_a_bigger_margin_fills_less_often():
    df = maker_games([0.60, 0.60], [0.50, 0.50], [True, False],
                     market_ids=["M1", "M2"])
    idx = {"M1": candles([hour(3, low=0.585, high=0.61)]),
           "M2": candles([hour(3, low=0.585, high=0.61)])}
    tight = pnl.maker_bet_frame(df, "m", idx, margin=0.01)      # bid 0.59
    wide = pnl.maker_bet_frame(df, "m", idx, margin=0.03)       # bid 0.57
    assert tight["filled"].all() and not wide["filled"].any()


def test_flat_staking_is_one_contract_per_game():
    df = maker_games([0.60, 0.40], [0.50, 0.50], [True, False])
    idx = {"M1": candles([hour(3, low=0.50, high=0.62)]),
           "M2": candles([hour(3, low=0.35, high=0.60)])}
    bets = pnl.maker_bet_frame(df, "m", idx, margin=0.0, staking="flat")
    assert list(bets["quoted"]) == [1.0, 1.0]
    assert bets["filled"].all() and list(bets["contracts"]) == [1.0, 1.0]


def test_kelly_stakes_more_contracts_on_a_bigger_edge():
    df = maker_games([0.60, 0.90], [0.50, 0.50], [True, True])
    idx = {"M1": candles([hour(3, low=0.40, high=0.95)]),
           "M2": candles([hour(3, low=0.40, high=0.95)])}
    bets = pnl.maker_bet_frame(df, "m", idx, margin=0.05, staking="kelly")
    assert bets.loc[1, "contracts"] > bets.loc[0, "contracts"] > 0


def test_kelly_stakes_nothing_at_a_zero_margin():
    # At m = 0 the limit *is* the model's fair value, so the Kelly edge at the
    # fill price is zero and the rule declines to size it. Not a bug: it is
    # what "quote at fair value" means.
    df = maker_games([0.60], [0.50], [True])
    idx = {"M1": candles([hour(3, low=0.40, high=0.95)])}
    bets = pnl.maker_bet_frame(df, "m", idx, margin=0.0, staking="kelly")
    assert len(bets) == 0                 # a zero-size order is not an order
    assert pnl.maker_summarize(bets, n_games=1, draws=50)["n_posted"] == 0


# ───────────────────────────── the exam ─────────────────────────────

def test_shuffled_control_keeps_the_edge_distribution_and_loses_the_pairing():
    df = maker_games([0.60, 0.30, 0.55, 0.48], [0.50, 0.50, 0.50, 0.50],
                     [True, False, True, False])
    ctrl = pnl.shuffled_edge(df, "m", seed=3)
    edges = np.sort(np.abs(df["m"].to_numpy() - 0.50))
    assert np.allclose(np.sort(np.abs(ctrl - 0.50)), edges)
    assert not np.allclose(ctrl, df["m"].to_numpy())


def test_split_halves_partitions_the_window_by_date():
    df = maker_games([0.6] * 6, [0.5] * 6, [True] * 6)
    first, second = pnl.split_halves(df)
    assert len(first) + len(second) == 6
    assert first["date"].max() < second["date"].min()


def test_run_maker_exam_covers_the_grid_and_labels_both_halves():
    n = 8
    preds = pd.DataFrame({
        "date": [f"2026-07-{i + 1:02d}" for i in range(n)],
        "game_pk": range(1, n + 1),
        "home_win": [True, False] * (n // 2),
        "m": [0.60, 0.40] * (n // 2),
    })
    closes = pd.DataFrame({
        "venue": ["kalshi"] * n, "game_pk": range(1, n + 1),
        "p_home_close": [0.50] * n, "bid": [0.49] * n, "ask": [0.51] * n,
        "home_won": [True, False] * (n // 2), "minutes_before_pitch": [15.0] * n,
        "market_id": [f"M{i + 1}" for i in range(n)],
        "close_ts": [FIRST_PITCH - 900] * n,
    })
    cs = pd.concat([candles([hour(3, low=0.40, high=0.65)]).assign(market_id=f"M{i + 1}")
                    for i in range(n)])
    res = pnl.run_maker_exam(preds, closes, cs, ["m"], margins=(0.0, 0.02),
                             stakings=("flat",), draws=50)
    assert set(res["half"]) == {"all", "first", "second"}
    assert set(res["model"]) == {"m", "m__shuffled"}
    assert len(res) == 3 * 2 * 2                  # halves x models x margins
    allrows = res[(res["half"] == "all") & (res["model"] == "m")]
    assert set(allrows["n_games"]) == {n}
    assert (allrows["n_posted"] == n).all()       # every game gets a quote


def test_choose_margin_picks_the_training_halfs_best():
    res = pd.DataFrame({
        "model": ["m"] * 4, "staking": ["flat"] * 4,
        "half": ["first", "first", "second", "second"],
        "margin": [0.0, 0.02, 0.0, 0.02],
        "pnl_per_posted": [-0.01, 0.03, 0.05, -0.02],
    })
    assert pnl.choose_margin(res, "m") == pytest.approx(0.02)


def test_an_order_above_the_ask_is_flagged_as_marketable():
    """A bid at or above the resting offer is a taker order, not a maker one.

    At a margin of zero the rule quotes the model's own price, which is above
    the market whenever the model disagrees at all — the exam has to say so.
    """
    df = maker_games([0.60], [0.50], [True])
    quote = {"yes_bid_close": 0.50, "yes_ask_close": 0.51}
    idx = {"M1": candles([{**hour(6, low=0.49, high=0.52), **quote}])}
    aggressive = pnl.maker_bet_frame(df, "m", idx, margin=0.0)     # bid 0.60 > ask 0.51
    passive = pnl.maker_bet_frame(df, "m", idx, margin=0.15)       # bid 0.45 < ask
    assert bool(aggressive.loc[0, "marketable"])
    assert not bool(passive.loc[0, "marketable"])
    assert pnl.maker_summarize(aggressive, 1, draws=50)["marketable_rate"] == 1.0


def test_marketable_mirrors_on_the_no_side():
    df = maker_games([0.40], [0.50], [True])
    quote = {"yes_bid_close": 0.50, "yes_ask_close": 0.51}
    idx = {"M1": candles([{**hour(6, low=0.49, high=0.52), **quote}])}
    # NO at 0.60 is YES offered at 0.40, which the resting 0.50 bid lifts at once.
    aggressive = pnl.maker_bet_frame(df, "m", idx, margin=0.0)
    assert list(aggressive["side"]) == ["no"] and bool(aggressive.loc[0, "marketable"])
    passive = pnl.maker_bet_frame(df, "m", idx, margin=0.15)
    assert not bool(passive.loc[0, "marketable"])
