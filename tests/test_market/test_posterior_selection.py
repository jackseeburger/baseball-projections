"""`decide_posterior` — selecting on P(edge > 0) instead of |mean edge|.

docs/posterior-props.md's second prediction is that this beats the
threshold rule at a matched bet count, because the threshold rule cannot
tell a real 3-point edge from a 3-point edge sitting on a wide posterior.
These tests are about the mechanics the prediction depends on: the
degenerate case (a point posterior must reduce to the existing rule
exactly), monotonicity in `p_sd`, and that it never trades inside the
spread — plus that `decide` itself, and everything built on it, is
untouched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import pnl


# ───────────────────────────── decide_posterior ─────────────────────────────

def test_zero_sd_reduces_to_the_threshold_rule_at_zero():
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.1, 0.9, n)
    mid = rng.uniform(0.1, 0.9, n)
    bid, ask = mid - 0.02, mid + 0.02
    sd = np.zeros(n)

    baseline = pnl.decide(p, bid, ask, threshold=0.0)
    for tau in (0.1, 0.5, 0.55, 0.9, 0.99):
        posterior = pnl.decide_posterior(p, sd, bid, ask, tau)
        assert list(posterior["side"]) == list(baseline["side"])
        pd.testing.assert_series_equal(
            posterior["cost"], baseline["cost"], check_names=False)
        pd.testing.assert_series_equal(
            posterior["edge"], baseline["edge"], check_names=False)


def test_wider_posterior_bets_less_at_fixed_mean_edge_and_tau():
    # Fix the mean edge (p − ask = 0.03 throughout) and grow the posterior
    # sd: P(p_true > ask) shrinks toward 0.5 as the Normal widens around a
    # mean that is only barely on the right side of ask, so fewer contracts
    # clear any fixed tau > 0.5.
    n = 500
    ask = np.full(n, 0.50)
    bid = np.full(n, 0.46)
    p = np.full(n, 0.53)          # a constant 3-point mean edge
    tau = 0.75
    sds = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    n_bets = []
    for sd in sds:
        d = pnl.decide_posterior(p, np.full(n, sd), bid, ask, tau)
        n_bets.append(int((d["side"] != pnl.NO_BET).sum()))
    # All rows are identical, so each sd gives either "everyone bets" (0 or
    # n) or, at the crossover, could tip either way for all of them at once
    # — the count is non-increasing as sd grows.
    assert n_bets == sorted(n_bets, reverse=True)
    assert n_bets[0] == n            # sd=0.01: mean edge is ~3 sds out, easy bet
    assert n_bets[-1] == 0           # sd=0.32: mean edge is a rounding error


def test_wider_posterior_bets_less_across_a_mixed_book():
    # Same monotonicity claim with a realistic mix of edges rather than one
    # repeated row, comparing bet counts between two fixed sd levels.
    rng = np.random.default_rng(1)
    n = 2_000
    mid = rng.uniform(0.2, 0.8, n)
    bid, ask = mid - 0.02, mid + 0.02
    edge_sign = rng.choice([-1, 1], n)
    p = np.clip(mid + edge_sign * rng.uniform(0.0, 0.06, n), 0.01, 0.99)
    tau = 0.7

    tight = pnl.decide_posterior(p, np.full(n, 0.01), bid, ask, tau)
    loose = pnl.decide_posterior(p, np.full(n, 0.06), bid, ask, tau)
    n_tight = int((tight["side"] != pnl.NO_BET).sum())
    n_loose = int((loose["side"] != pnl.NO_BET).sum())
    assert n_loose < n_tight


def test_never_bets_inside_the_spread_for_tau_at_or_above_half():
    rng = np.random.default_rng(2)
    n = 5_000
    mid = rng.uniform(0.05, 0.95, n)
    half_spread = rng.uniform(0.005, 0.04, n)
    bid, ask = mid - half_spread, mid + half_spread
    p = rng.uniform(0.0, 1.0, n)                 # includes plenty inside [bid, ask]
    sd = rng.uniform(0.001, 0.3, n)

    for tau in (0.5, 0.6, 0.75, 0.9):
        d = pnl.decide_posterior(p, sd, bid, ask, tau)
        yes = d["side"].to_numpy() == "yes"
        no = d["side"].to_numpy() == "no"
        assert np.all(p[yes] > ask[yes])
        assert np.all(p[no] < bid[no])


def test_decide_posterior_never_bets_when_posterior_probability_below_tau():
    # A near-certain 1-point edge (tiny sd) should not clear a high tau even
    # though `decide` would take it at threshold 0.
    p = np.array([0.51])
    bid, ask = np.array([0.48]), np.array([0.50])
    sd = np.array([0.01])          # edge is ~1 sd out — clears a modest tau
    assert pnl.decide_posterior(p, sd, bid, ask, 0.8)["side"][0] == "yes"

    sd_wide = np.array([2.0])      # same edge, huge sd — should not clear tau
    assert pnl.decide_posterior(p, sd_wide, bid, ask, 0.6)["side"][0] == pnl.NO_BET


# ───────────────────────────── decide is untouched ─────────────────────────────

def test_existing_decide_behaviour_is_unchanged():
    p = np.array([0.10, 0.52, 0.60, 0.90])
    bid = np.array([0.05, 0.50, 0.55, 0.85])
    ask = np.array([0.15, 0.54, 0.65, 0.95])
    d = pnl.decide(p, bid, ask, threshold=0.0)
    assert list(d["side"]) == [pnl.NO_BET, pnl.NO_BET, pnl.NO_BET, pnl.NO_BET]
    d2 = pnl.decide(p, bid, ask, threshold=0.0)
    d3 = pnl.decide(np.array([0.20]), np.array([0.05]), np.array([0.15]), 0.0)
    assert d3["side"][0] == "yes"
    assert d3["cost"][0] == pytest.approx(0.15)
    assert d3["edge"][0] == pytest.approx(0.05)
    pd.testing.assert_frame_equal(d, d2)


# ───────────────────────────── bet_frame / evaluate rule threading ─────────────────────────────

def games(p_model, p_close, home_won, bid=None, ask=None, model="m", sd=None):
    n = len(p_model)
    bid = [c - 0.01 for c in p_close] if bid is None else bid
    ask = [c + 0.01 for c in p_close] if ask is None else ask
    d = {
        "date": [f"2026-07-{i + 1:02d}" for i in range(n)],
        "game_pk": range(1, n + 1),
        "home_win": home_won,
        model: p_model,
        "p_home_close": p_close,
        "bid": bid,
        "ask": ask,
    }
    if sd is not None:
        d["p_over_sd"] = sd
    return pd.DataFrame(d)


def test_bet_frame_default_rule_is_unchanged_by_the_new_argument():
    df = games([0.10, 0.60], [0.30, 0.50], [False, True])
    old = pnl.bet_frame(df, "m", pnl.KALSHI, threshold=0.02)
    new = pnl.bet_frame(df, "m", pnl.KALSHI, threshold=0.02, rule="threshold")
    pd.testing.assert_frame_equal(old, new)


def test_bet_frame_posterior_rule_uses_sd_col_and_tau():
    df = games([0.10, 0.60], [0.30, 0.50], [False, True],
              sd=[0.005, 0.20])
    bets = pnl.bet_frame(df, "m", pnl.KALSHI, rule="posterior", tau=0.9,
                         sd_col="p_over_sd")
    # Row 0: edge is far from the spread relative to its tiny sd -> bets.
    # Row 1: same-sized nominal edge but a wide sd -> does not clear tau=0.9.
    assert len(bets) == 1
    assert bets.iloc[0]["game_pk"] == 1


def test_bet_frame_posterior_rule_requires_sd_col_and_tau():
    df = games([0.10], [0.30], [False])
    with pytest.raises(ValueError):
        pnl.bet_frame(df, "m", pnl.KALSHI, rule="posterior")


def test_evaluate_threads_rule_through_and_reports_it():
    df = games([0.10, 0.60], [0.30, 0.50], [False, True], sd=[0.005, 0.20])
    row = pnl.evaluate(df, "m", pnl.KALSHI, rule="posterior", tau=0.9,
                       sd_col="p_over_sd")
    assert row["rule"] == "posterior"
    assert row["tau"] == 0.9
    assert row["n_bets"] == 1
