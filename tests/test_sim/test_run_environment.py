"""Unit tests for station C, the bottom-up team run environment
(src/sim/run_environment.py).

All synthetic — no network. What has to hold:
  * a club of league-average hitters scores exactly league RS/G, and one
    better hitter moves it by exactly the linear-weights amount his share of
    the plate appearances buys
  * who bats for a club, and how much, is read from appearances *strictly
    before* the date being predicted, split by the club he batted for
  * a rotation is the top-5 by starts in the window, weighted by starts, and a
    league-average rotation in front of a league-average pen allows exactly
    league RA/9
  * the blend is a nesting: weight 0 hands back the production top-down rates
    unchanged, so `pythag_C` is `pythag_60` to the last bit
  * nothing on or after the date being predicted ever enters (leakage)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sim import lineups as lu
from src.sim import run_environment as rn
from src.sim import strength

ROOT = Path(__file__).resolve().parents[2]

LG_RS9 = 4.30
LG_RA9 = 4.30
PA_PER_GAME = 38.0


# ─── fixtures ───

def hitting(batter, team_id, date, pa=4):
    return {"batter": batter, "team_id": team_id, "date": date, "pa": pa}


def pitching(pitcher, team, date, gs=1):
    return {"pitcher": pitcher, "team": team, "date": date, "gs": gs}


@pytest.fixture
def league():
    """A league rate dict from one pooled season of plausible counts."""
    counts = lu.normalize_counts(pd.DataFrame([{
        "batter": 1, "season": 2026, "pa": 100000.0, "ab": 89000.0,
        "h": 21800.0, "doubles": 4300.0, "triples": 400.0, "hr": 3000.0,
        "k": 22500.0, "bb": 8200.0, "hbp": 1100.0, "sf": 700.0,
    }]))
    return lu.league_rates(counts)


# ─── team_pa_shares: who bats, and how much ───

def test_shares_sum_to_one_within_each_club():
    logs = pd.DataFrame([hitting(b, 100, "2026-05-01") for b in range(1, 10)]
                        + [hitting(b, 200, "2026-05-01") for b in range(11, 20)])
    out = rn.team_pa_shares(logs, "2026-06-01", window_days=60)
    assert set(out["team_id"]) == {100, 200}
    for _, grp in out.groupby("team_id"):
        assert grp["share"].sum() == pytest.approx(1.0)


def test_no_hitter_takes_more_than_one_lineup_slot():
    # One man with ten times everybody else's plate appearances still cannot
    # bat more than an eighth of the time — that is lineup arithmetic.
    logs = pd.DataFrame([hitting(1, 100, "2026-05-01", pa=400)]
                        + [hitting(b, 100, "2026-05-01", pa=40) for b in range(2, 14)])
    out = rn.team_pa_shares(logs, "2026-06-01", window_days=60)
    assert out["share"].max() == pytest.approx(rn.MAX_PA_SHARE)
    assert out["share"].sum() == pytest.approx(1.0)


def test_a_traded_hitter_bats_for_the_club_he_batted_for():
    logs = pd.DataFrame([hitting(1, 100, "2026-05-01", pa=100),
                         hitting(1, 200, "2026-05-20", pa=100),
                         hitting(2, 100, "2026-05-01", pa=100),
                         hitting(3, 200, "2026-05-20", pa=100)])
    out = rn.team_pa_shares(logs, "2026-06-01", window_days=None)
    got = {(int(r.team_id), int(r.batter)): r.pa for r in out.itertuples(index=False)}
    assert got == {(100, 1): 100, (100, 2): 100, (200, 1): 100, (200, 3): 100}


def test_shares_never_see_the_date_being_predicted():
    logs = pd.DataFrame([hitting(1, 100, "2026-05-01", pa=100),
                         hitting(2, 100, "2026-05-01", pa=100),
                         # A monster game *on* the cutoff must not count.
                         hitting(2, 100, "2026-06-01", pa=900),
                         hitting(2, 100, "2026-06-02", pa=900)])
    out = rn.team_pa_shares(logs, "2026-06-01", window_days=None)
    assert out["share"].tolist() == pytest.approx([0.5, 0.5])


def test_a_hitter_outside_the_window_is_not_on_the_club():
    logs = pd.DataFrame([hitting(1, 100, "2026-03-25", pa=400),
                         hitting(2, 100, "2026-05-25", pa=100)])
    out = rn.team_pa_shares(logs, "2026-06-01", window_days=30)
    assert out["batter"].tolist() == [2]


def test_no_appearances_yet_is_an_empty_frame_not_a_crash():
    out = rn.team_pa_shares(pd.DataFrame(columns=["batter", "team_id", "date", "pa"]),
                            "2026-06-01")
    assert out.empty and list(out.columns) == rn.SHARE_COLS


# ─── team_rs9: hitters x playing time → runs ───

def test_a_club_of_league_average_hitters_scores_exactly_league_runs():
    logs = pd.DataFrame([hitting(b, 100, "2026-05-01") for b in range(1, 14)])
    shares = rn.team_pa_shares(logs, "2026-06-01", window_days=60)
    # Every hitter absent from the lookup, i.e. every hitter league average.
    rs9 = rn.team_rs9(shares, {}, LG_RS9, PA_PER_GAME)
    assert rs9.loc[100] == pytest.approx(LG_RS9)


def test_one_better_hitter_moves_runs_by_the_linear_weights_amount(league):
    """A hitter 20% better than the league at hitting home runs.

    His runs above average per plate appearance comes from `lineups`' linear
    weights, and the club's runs per game has to move by exactly that number
    times the plate appearances his share buys — no more (the estimator would
    be double-counting) and no less (it would be shrinking a real hitter).
    """
    rates = pd.DataFrame([{f"rate_{c}": league[f"rate_{c}"] for c in lu.COMPONENTS}],
                         index=pd.Index([1], name="batter"))
    rates.loc[1, "rate_hr"] = league["rate_hr"] * 1.20
    raa = lu.batter_runs_lookup(rates, league)[1]
    assert raa > 0

    logs = pd.DataFrame([hitting(b, 100, "2026-05-01") for b in range(1, 14)])
    shares = rn.team_pa_shares(logs, "2026-06-01", window_days=60)
    share = float(shares.loc[shares["batter"] == 1, "share"].iloc[0])

    base = rn.team_rs9(shares, {}, LG_RS9, PA_PER_GAME).loc[100]
    with_him = rn.team_rs9(shares, {1: raa}, LG_RS9, PA_PER_GAME).loc[100]
    assert with_him - base == pytest.approx(PA_PER_GAME * share * raa)


def test_an_average_hitter_and_a_hitter_with_no_history_are_the_same():
    logs = pd.DataFrame([hitting(b, 100, "2026-05-01") for b in range(1, 10)])
    shares = rn.team_pa_shares(logs, "2026-06-01", window_days=60)
    assert rn.team_rs9(shares, {1: 0.0}, LG_RS9, PA_PER_GAME).loc[100] == \
        pytest.approx(rn.team_rs9(shares, {}, LG_RS9, PA_PER_GAME).loc[100])


# ─── start_appearances and the rotation ───

def test_a_relief_outing_is_not_a_start():
    out = rn.start_appearances(pd.DataFrame([pitching(1, 100, "2026-05-01", gs=1),
                                             pitching(2, 100, "2026-05-01", gs=0)]))
    assert out["pitcher"].tolist() == [1]


def test_starts_and_relief_appearances_partition_every_outing():
    from src.sim import bullpen as bp
    logs = pd.DataFrame([
        {**pitching(1, 100, "2026-05-01", gs=1), "bf": 24, "outs": 18},
        {**pitching(2, 100, "2026-05-01", gs=0), "bf": 4, "outs": 3},
        {**pitching(1, 100, "2026-05-07", gs=0), "bf": 3, "outs": 3},
    ])
    assert len(rn.start_appearances(logs)) + len(bp.relief_appearances(logs)) == len(logs)


def test_a_rotation_is_the_top_five_by_starts():
    rows = []
    for pid, n in [(1, 6), (2, 5), (3, 5), (4, 4), (5, 4), (6, 1), (7, 1)]:
        rows += [pitching(pid, 100, f"2026-05-{d:02d}") for d in range(1, n + 1)]
    out = rn.rotation_window(rn.start_appearances(pd.DataFrame(rows)),
                             "2026-06-01", days=60, top_n=5)
    assert sorted(out["pitcher"]) == [1, 2, 3, 4, 5]
    assert out.loc[out["pitcher"] == 1, "starts"].iloc[0] == 6


def test_the_rotation_never_sees_the_start_being_predicted():
    starts = rn.start_appearances(pd.DataFrame([
        pitching(1, 100, "2026-05-30"), pitching(2, 100, "2026-06-01")]))
    out = rn.rotation_window(starts, "2026-06-01", days=30)
    assert out["pitcher"].tolist() == [1]


def test_a_starter_traded_away_leaves_the_rotation_when_his_starts_age_out():
    starts = rn.start_appearances(pd.DataFrame(
        [pitching(1, 100, f"2026-04-{d:02d}") for d in (1, 7, 13, 19, 25)]
        + [pitching(2, 100, f"2026-06-{d:02d}") for d in (1, 7)]))
    early = rn.rotation_window(starts, "2026-05-01", days=30)
    late = rn.rotation_window(starts, "2026-07-01", days=30)
    assert early["pitcher"].tolist() == [1]
    assert late["pitcher"].tolist() == [2]


def test_a_tie_on_starts_goes_to_the_man_who_started_more_recently():
    starts = rn.start_appearances(pd.DataFrame([
        pitching(1, 100, "2026-05-02"), pitching(2, 100, "2026-05-20")]))
    out = rn.rotation_window(starts, "2026-06-01", days=60, top_n=1)
    assert out["pitcher"].tolist() == [2]


def test_the_rotation_is_weighted_by_starts_not_by_head_count():
    rotation = pd.DataFrame([{"team": 100, "pitcher": 1, "starts": 9},
                             {"team": 100, "pitcher": 2, "starts": 1}])
    got = rn.rotation_ra9(rotation, {1: 3.0, 2: 8.0}, LG_RA9)[100]
    assert got == pytest.approx((9 * 3.0 + 1 * 8.0) / 10)


def test_a_starter_with_no_history_is_priced_at_league_average():
    rotation = pd.DataFrame([{"team": 100, "pitcher": 99, "starts": 5}])
    assert rn.rotation_ra9(rotation, {}, LG_RA9)[100] == pytest.approx(LG_RA9)


# ─── team_ra9: rotation + pen partition the nine innings ───

def test_an_average_staff_allows_exactly_league_runs():
    out = rn.team_ra9({100: LG_RA9}, {100: LG_RA9}, LG_RA9, team_ids=[100])
    assert out.loc[100] == pytest.approx(LG_RA9)


def test_the_rotation_covers_five_and_a_half_of_the_nine_innings():
    out = rn.team_ra9({100: LG_RA9 + 0.9}, {100: LG_RA9}, LG_RA9, team_ids=[100])
    assert out.loc[100] - LG_RA9 == pytest.approx(0.9 * rn.STARTER_IP / 9.0)


def test_a_missing_half_falls_back_to_the_league_for_that_half_only():
    out = rn.team_ra9({}, {100: LG_RA9 + 0.9}, LG_RA9, team_ids=[100])
    assert out.loc[100] - LG_RA9 == pytest.approx(0.9 * (9.0 - rn.STARTER_IP) / 9.0)


# ─── the blend is a nesting ───

def top_down_frame():
    return pd.DataFrame({"rs_pg": [4.8, 4.0], "ra_pg": [4.1, 4.6]},
                        index=pd.Index([100, 200], name="team_id"))


def bottom_up_frame():
    return pd.DataFrame({"rs_pg": [4.4, 4.4], "ra_pg": [4.4, 4.4]},
                        index=pd.Index([100, 200], name="team_id"))


def test_weight_zero_returns_the_production_rates_bit_for_bit():
    td = top_down_frame()
    out = rn.blend_run_env(bottom_up_frame(), td, weight=0.0)
    assert out.equals(td.loc[:, ["rs_pg", "ra_pg"]].astype(float))


def test_weight_one_returns_the_bottom_up_rates():
    out = rn.blend_run_env(bottom_up_frame(), top_down_frame(), weight=1.0)
    assert out["rs_pg"].tolist() == pytest.approx([4.4, 4.4])


def test_the_blend_is_linear_in_the_weight():
    out = rn.blend_run_env(bottom_up_frame(), top_down_frame(), weight=0.25)
    assert out.loc[100, "rs_pg"] == pytest.approx(0.25 * 4.4 + 0.75 * 4.8)


def test_a_club_the_bottom_up_half_could_not_price_keeps_its_own_rate():
    bu = bottom_up_frame()
    bu.loc[200, "rs_pg"] = np.nan
    bu = bu.drop(index=[])  # 100 priced, 200 half-priced
    out = rn.blend_run_env(bu, top_down_frame(), weight=1.0)
    assert out.loc[200, "rs_pg"] == pytest.approx(4.0)   # top-down, untouched
    assert out.loc[200, "ra_pg"] == pytest.approx(4.4)   # bottom-up, applied


def test_a_club_missing_from_the_bottom_up_frame_keeps_its_own_rates():
    out = rn.blend_run_env(bottom_up_frame().drop(index=[200]), top_down_frame(),
                           weight=1.0)
    assert out.loc[200].tolist() == pytest.approx([4.0, 4.6])


def test_bottom_up_rates_reindexes_onto_the_league():
    out = rn.bottom_up_rates(pd.Series({100: 4.9}), pd.Series({200: 4.1}),
                             team_ids=[100, 200])
    assert out.index.tolist() == [100, 200]
    assert np.isnan(out.loc[100, "ra_pg"]) and np.isnan(out.loc[200, "rs_pg"])


# ─── end to end: weight 0 is pythag_60, exactly ───

def _backtest_module():
    spec = importlib.util.spec_from_file_location(
        "backtest_game_odds", ROOT / "scripts/backtest_game_odds.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Game:
    def __init__(self, game_pk, home_id, away_id):
        self.game_pk, self.home_id, self.away_id = game_pk, home_id, away_id


def test_weight_zero_reproduces_pythag_60_and_pythag_60_sp_exactly():
    """The whole point of the nesting: `--c-weight 0` is the production model.

    Scored end to end on the real 2026 season this holds to the last bit —
    the paired Brier difference `pythag_C_sp - pythag_60_sp` is +0.00000 with
    a standard error of 0.00000 on all 2,105 games of 2025 and all 1,776 of
    2026 (docs/market-benchmark-2026.md). This is the same statement on a
    two-club synthetic, without the network.
    """
    bt = _backtest_module()
    tot = pd.DataFrame({"rs": [500.0, 400.0], "ra": [420.0, 470.0],
                        "w": [55.0, 45.0], "g": [100.0, 100.0]},
                       index=pd.Index([100, 200], name="team_id"))
    lg_ra9 = float(tot["ra"].sum() / tot["g"].sum())
    lg_rs9 = float(tot["rs"].sum() / tot["g"].sum())
    sp_day = {"lg_ra9": lg_ra9, "team": bt.team_rates(tot, bt.SP_BALLAST_GAMES),
              "probables": {7: (11, 22)}, "starter_ip": 5.5,
              "sp_ra9": {11: 3.2, 22: 5.1}}
    lu_day = {"runs_lookup": {}, "lg_rs9": lg_rs9}
    bp_day = {"pen": {100: (lg_ra9, 4.9), 200: (lg_ra9, 3.9)},
              "ra9": {1: 3.0, 2: 5.0}}
    c_ctx = {"hitter_logs": pd.DataFrame([hitting(1, 100, "2026-05-01"),
                                          hitting(2, 200, "2026-05-01")]),
             "starts": rn.start_appearances(pd.DataFrame(
                 [pitching(1, 100, "2026-05-01"), pitching(2, 200, "2026-05-01")])),
             "weight": 0.0, "share_window": 60, "rotation_days": 60,
             "rotation_top_n": 5, "pa_per_game": PA_PER_GAME}
    hfa = 0.54
    c_day = bt.run_env_day_context(tot, "2026-06-01", c_ctx, sp_day, lu_day, bp_day)
    got = bt.run_env_game_probs(_Game(7, 100, 200), c_day, sp_day, hfa)

    s = bt.strengths(tot, bt.SP_BALLAST_GAMES)
    want_60 = float(strength.home_win_prob(s[100], s[200], hfa))
    want_sp, _ = bt.starter_game_prob(_Game(7, 100, 200), sp_day, hfa)
    assert got[bt.C_MODEL] == want_60
    assert got[bt.C_SP_MODEL] == pytest.approx(want_sp, abs=1e-12)


def test_a_game_with_no_probable_falls_back_to_the_pitcher_free_model():
    bt = _backtest_module()
    tot = pd.DataFrame({"rs": [500.0, 400.0], "ra": [420.0, 470.0],
                        "w": [55.0, 45.0], "g": [100.0, 100.0]},
                       index=pd.Index([100, 200], name="team_id"))
    lg_ra9 = float(tot["ra"].sum() / tot["g"].sum())
    c_day = {"team": bt.team_rates(tot, bt.SP_BALLAST_GAMES), "lg_ra9": lg_ra9,
             "starter_ip": 5.5, "rs_missing": set(), "ra_missing": set()}
    sp_day = {"probables": {}, "sp_ra9": {}}
    got = bt.run_env_game_probs(_Game(7, 100, 200), c_day, sp_day, 0.54)
    assert got["c_sp_fallback"] is True
    assert got[bt.C_SP_MODEL] == got[bt.C_MODEL]
