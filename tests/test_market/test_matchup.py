"""The matchup term and the prop maker exam, on synthetic payloads — no network.

Three things are worth a test here and they are the three that can go wrong
silently: the log5 arithmetic (an identity, so it has a known answer), the
walk-forward guards (a prop price is a *pre-game* price, so neither the actual
starter nor tonight's line may reach it), and the fill rule on prop candles (a
resting order must not be filled by a price that printed after first pitch).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import backfill, kalshi, matchup, pnl, props
from src.sim import lineups as lu_model


# ───────────────────────────── the identity ─────────────────────────────

def test_log5_on_a_known_case():
    """A hitter twice the league's rate against a pitcher half of it is neutral."""
    assert matchup.log5(0.40, 0.10, 0.20) == pytest.approx(0.20)
    # A league-average pitcher returns the hitter unchanged...
    assert matchup.log5(0.31, 0.22, 0.22) == pytest.approx(0.31)
    # ...and a league-average hitter returns the pitcher.
    assert matchup.log5(0.22, 0.31, 0.22) == pytest.approx(0.31)
    # A pitcher who allows half the league's home runs halves the hitter's.
    assert matchup.log5(0.06, 0.015, 0.030) == pytest.approx(0.03)


def test_log5_falls_back_to_the_hitter_when_the_league_rate_is_missing():
    assert matchup.log5(0.31, 0.22, 0.0) == pytest.approx(0.31)


def test_apply_factor_interpolates_between_the_current_price_and_log5():
    rate, factor = 0.30, 1.5                    # a pitcher 50% over the league
    assert matchup.apply_factor(rate, factor, 0.0) == pytest.approx(rate)
    assert matchup.apply_factor(rate, factor, 1.0) == pytest.approx(0.45)
    assert matchup.apply_factor(rate, factor, 0.5) == pytest.approx(0.375)


def test_blend_factor_weights_the_starter_by_his_expected_innings():
    # Six innings of a strikeout machine, three of a league-average pen.
    assert matchup.blend_factor(1.30, 1.00, 6.0 / 9.0) == pytest.approx(1.20)
    assert matchup.blend_factor(1.30, 1.00, 0.0) == pytest.approx(1.00)
    assert matchup.blend_factor(1.30, 1.00, 1.0) == pytest.approx(1.30)


def test_a_missing_pitcher_is_the_league_and_leaves_the_price_alone():
    f = matchup.factors_from_rates(None, {"rate_k": 0.22})
    assert f["k"] == 1.0
    rates = {f"rate_{c}": 0.1 for c in matchup.COMPONENTS}
    assert matchup.matchup_rates(rates, f, 1.0)["rate_k"] == pytest.approx(0.1)


# ────────────────── a pitcher's rates, in the hitter's columns ──────────────

def _pitching_logs() -> pd.DataFrame:
    """Two starters and one reliever, one club, three dates."""
    return pd.DataFrame({
        "pitcher": [91, 91, 92, 93, 93],
        "season": [2026] * 5,
        "date": ["2026-08-01", "2026-08-07", "2026-08-01", "2026-08-02", "2026-08-08"],
        "team": [140] * 5,
        "gs": [1, 1, 1, 0, 0],
        "outs": [18, 18, 18, 3, 3],
        "bf": [24.0, 24.0, 24.0, 4.0, 4.0],
        "k": [10.0, 10.0, 2.0, 1.0, 1.0],
        "bb": [1.0, 1.0, 3.0, 0.0, 0.0],
        "hbp": [0.0] * 5,
        "hr": [0.0, 0.0, 2.0, 0.0, 0.0],
        "h": [4.0, 4.0, 8.0, 1.0, 1.0],
        "ab": [22.0, 22.0, 21.0, 4.0, 4.0],
        "doubles": [1.0, 1.0, 2.0, 0.0, 0.0],
        "triples": [0.0] * 5,
        "sf": [0.0] * 5,
    })


def test_allowed_counts_lands_in_the_hitter_columns():
    counts = matchup.allowed_counts(_pitching_logs())
    assert {"pitcher", "season", "pa", "ab", "bip", "k", "bbhbp", "hr", "xb",
            "hip", "date", "team", "gs"} <= set(counts.columns)
    row = counts.iloc[0]
    assert row["pa"] == 24.0                     # a batter faced is a plate appearance
    assert row["xb"] == 1.0                      # one double, no triples, no homers
    assert row["hip"] == 4.0                     # hits in play = H − HR
    assert row["bip"] == 22.0 - 10.0 - 0.0 + 0.0


def test_allowed_rates_use_only_appearances_before_the_date():
    counts = matchup.allowed_counts(_pitching_logs())
    league = {f"rate_{c}": v for c, v in
              (("k", 0.22), ("bbhbp", 0.09), ("hr", 0.031), ("iso", 0.16),
               ("babip", 0.295))}
    before = matchup.allowed_rates(
        matchup.counts_before(counts, "2026-08-07"), 2026, league)
    after = matchup.allowed_rates(
        matchup.counts_before(counts, "2026-08-08"), 2026, league)
    # Pitcher 91's second strikeout-heavy start is only in the later table.
    assert after.loc[91, "rate_k"] > before.loc[91, "rate_k"]


def test_pen_rates_read_relief_appearances_only_and_regress_to_the_league():
    counts = matchup.allowed_counts(_pitching_logs())
    league = {f"rate_{c}": v for c, v in
              (("k", 0.22), ("bbhbp", 0.09), ("hr", 0.031), ("iso", 0.16),
               ("babip", 0.295))}
    relief = matchup.relief_rows(counts)
    assert set(relief["pitcher"]) == {93}         # the two starters are out
    pen = matchup.pen_rates(relief, "2026-08-09", league)
    # Eight relief batters faced against a 140-BF ballast: nearly the league.
    assert pen[140]["rate_k"] == pytest.approx(0.22, abs=0.01)
    assert matchup.pen_rates(relief, "2026-08-01", league) == {}


def test_card_k_factor_weights_by_slot_and_falls_back_to_the_league():
    rates = pd.DataFrame({"rate_k": [0.30, 0.10]}, index=[1, 2])
    rates.index.name = "batter"
    assert matchup.card_k_factor([1, 2], rates, 0.20) == pytest.approx(1.0)
    # A card of one unknown hitter is the league exactly.
    assert matchup.card_k_factor([99], rates, 0.20) == pytest.approx(1.0)
    assert matchup.card_k_factor([], rates, 0.20) is None
    # The leadoff slot gets more plate appearances than the ninth, so the
    # weighted factor leans toward whoever bats first.
    lead = matchup.card_k_factor([1, 2], rates, 0.20, props.slot_pa)
    ninth = matchup.card_k_factor([2, 1], rates, 0.20, props.slot_pa)
    assert lead > 1.0 > ninth


def test_recent_cards_are_strictly_before_the_date():
    cards = {(140, "2026-08-01"): [1, 2], (140, "2026-08-05"): [3, 4],
             (141, "2026-08-04"): [5]}
    assert matchup.recent_card_ids(cards, 140, "2026-08-05") == [1, 2]
    assert matchup.recent_card_ids(cards, 140, "2026-08-06") == [3, 4, 1, 2]
    assert matchup.recent_card_ids(cards, 999, "2026-08-06") == []


# ───────────────────── the matchup price, end to end ─────────────────────

def _batter_ctx() -> dict:
    """One hitter with a long, ordinary prior season."""
    prior = pd.DataFrame({
        "batter": [1, 2], "season": [2025, 2025], "pa": [600, 600], "ab": [540, 540],
        "h": [140, 140], "doubles": [28, 28], "triples": [2, 2], "hr": [20, 20],
        "k": [120, 120], "bb": [55, 55], "hbp": [5, 5], "sf": [5, 5],
    })
    league_rows = pd.concat([prior] * 40, ignore_index=True)
    prior_counts = lu_model.normalize_counts(league_rows)
    logs = pd.DataFrame(columns=["batter", "season", "pa", "ab", "h", "doubles",
                                 "triples", "hr", "k", "bb", "hbp", "sf"])
    game_logs = lu_model.normalize_counts(logs)
    game_logs["date"] = []
    return {"season": 2026, "prior_counts": prior_counts, "game_logs": game_logs,
            "league": lu_model.league_rates(prior_counts)}


def _empty_pitcher_ctx() -> dict:
    cols = ["pitcher", "season", "bf", "k", "bbhbp", "hr", "outs"]
    return {"season": 2026, "league": {"rate_k": 0.22},
            "prior_counts": pd.DataFrame(columns=cols),
            "game_logs": pd.DataFrame(columns=[*cols, "date"])}


def _ace_logs(pitcher: int, dates, k: float = 16.0) -> pd.DataFrame:
    """A starter who strikes out two thirds of the batters he faces."""
    n = len(dates)
    return pd.DataFrame({
        "pitcher": [pitcher] * n, "season": [2026] * n, "date": list(dates),
        "team": [140] * n, "gs": [1] * n, "outs": [18] * n,
        "bf": [24.0] * n, "k": [k] * n, "bb": [1.0] * n, "hbp": [0.0] * n,
        "hr": [0.0] * n, "h": [3.0] * n, "ab": [23.0] * n,
        "doubles": [0.0] * n, "triples": [0.0] * n, "sf": [0.0] * n,
    })


def _matchup_ctx(logs: pd.DataFrame, probables: dict, league: dict) -> dict:
    counts = matchup.allowed_counts(logs)
    return {
        "ctx": {"season": 2026,
                "prior_counts": pd.DataFrame(columns=["pitcher", "season",
                                                      *matchup.COUNT_COLS]),
                "game_logs": counts,
                "starts": pd.DataFrame({"pitcher": logs["pitcher"],
                                        "date": logs["date"],
                                        "ip": logs["outs"] / 3.0}),
                "league": league},
        "probables": probables,
        "teams": {(700001, "home"): 140, (700001, "away"): 141},
        "cards": {}, "club_cards": {},
        "sides": {(700001, 1): "away"},
        "weight": 1.0,
    }


def _closes() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_pk": 700001, "game_date": "2026-08-20", "player_id": 1,
         "team_id": 141, "prop_stat": "hits", "prop_line": 0.5,
         "p_over_close": 0.60, "over_hit": True},
    ])


LEAGUE_ALLOWED = {"rate_k": 0.22, "rate_bbhbp": 0.09, "rate_hr": 0.031,
                  "rate_iso": 0.16, "rate_babip": 0.295}


def _price(logs, probables, weight=1.0):
    ctx = _matchup_ctx(logs, probables, LEAGUE_ALLOWED)
    ctx["weight"] = weight
    return props.price(_closes(), _batter_ctx(), _empty_pitcher_ctx(),
                       {(700001, 1): 3}, stats=("hits",), matchup_ctx=ctx)


def test_a_strikeout_machine_lowers_the_hitters_price():
    dates = [f"2026-08-{d:02d}" for d in range(1, 16)]
    priced = _price(_ace_logs(91, dates), {(700001, "home"): 91})
    assert priced.loc[0, "p_matchup"] < priced.loc[0, "p_model"]
    # And the weight interpolates: zero is exactly the current price.
    off = _price(_ace_logs(91, dates), {(700001, "home"): 91}, weight=0.0)
    assert off.loc[0, "p_matchup"] == pytest.approx(off.loc[0, "p_model"])


def test_an_unknown_probable_leaves_the_price_where_it_was():
    dates = [f"2026-08-{d:02d}" for d in range(1, 16)]
    priced = _price(_ace_logs(91, dates), {})       # no probable posted
    assert priced.loc[0, "p_matchup"] == pytest.approx(priced.loc[0, "p_model"])


def test_the_probable_is_what_is_read_not_the_man_who_actually_started():
    """The pre-game announcement moves the price; a post-game fact does not.

    Two guards in one case. Swapping the *probable* from an ace to a
    league-average arm moves the price, which is what makes this a matchup
    model at all. Appending the ace's appearance **on the game date itself** —
    the line he actually threw, which is only knowable afterwards — moves
    nothing, because every rate is cut strictly before the date being priced.
    """
    dates = [f"2026-08-{d:02d}" for d in range(1, 16)]
    ace = _ace_logs(91, dates)
    filler = _ace_logs(92, dates, k=5.0)            # about the league's rate
    logs = pd.concat([ace, filler], ignore_index=True)

    with_ace = _price(logs, {(700001, "home"): 91}).loc[0, "p_matchup"]
    with_filler = _price(logs, {(700001, "home"): 92}).loc[0, "p_matchup"]
    assert with_ace < with_filler                   # the probable is read

    # The ace in fact started this game and struck out 20; that line lands in
    # the archive with the game's own date on it.
    after = pd.concat([logs, _ace_logs(91, ["2026-08-20"], k=20.0),
                       _ace_logs(92, ["2026-08-20"], k=20.0)], ignore_index=True)
    assert _price(after, {(700001, "home"): 91}).loc[0, "p_matchup"] == \
        pytest.approx(with_ace)
    assert _price(after, {(700001, "home"): 92}).loc[0, "p_matchup"] == \
        pytest.approx(with_filler)


def _k_closes() -> pd.DataFrame:
    """One strikeout contract on a starter pitching for the home club."""
    return pd.DataFrame([
        {"game_pk": 700001, "game_date": "2026-08-20", "player_id": 91,
         "team_id": 140, "prop_stat": "k", "prop_line": 5.5,
         "p_over_close": 0.50, "over_hit": True},
    ])


def _k_pitcher_ctx() -> dict:
    """One starter at the league's strikeout rate, from a long prior season."""
    prior = pd.DataFrame({"pitcher": [91], "season": [2025], "bf": [700.0],
                          "k": [154.0], "bbhbp": [63.0], "hr": [21.0],
                          "outs": [500.0]})
    return {"season": 2026, "league": {"rate_k": 0.22, "rate_bbhbp": 0.09,
                                       "rate_hr": 0.03},
            "prior_counts": prior,
            "game_logs": pd.DataFrame(columns=["pitcher", "season", "bf", "k",
                                               "bbhbp", "hr", "outs", "date"])}


def _k_matchup_ctx(card: list) -> dict:
    ctx = _matchup_ctx(_ace_logs(91, ["2026-08-01"]), {}, LEAGUE_ALLOWED)
    ctx["cards"] = {(700001, "away"): card}
    return ctx


def test_a_strikeout_prone_card_raises_the_starters_price():
    """The mirror: the opposing posted card is in the pitcher's number."""
    batter_ctx = _batter_ctx()
    # Batter 1 whiffs in half his plate appearances; batter 2 never does.
    prior = pd.DataFrame({
        "batter": [1, 2], "season": [2025, 2025], "pa": [600, 600],
        "ab": [540, 540], "h": [140, 140], "doubles": [28, 28],
        "triples": [2, 2], "hr": [20, 20], "k": [300, 20], "bb": [55, 55],
        "hbp": [5, 5], "sf": [5, 5]})
    batter_ctx["prior_counts"] = pd.concat(
        [batter_ctx["prior_counts"], lu_model.normalize_counts(prior)],
        ignore_index=True)

    def price(card):
        return props.price(_k_closes(), batter_ctx, _k_pitcher_ctx(), {},
                           stats=("k",), matchup_ctx=_k_matchup_ctx(card))

    whiffers = price([1] * 9).loc[0, "p_matchup"]
    contact = price([2] * 9).loc[0, "p_matchup"]
    assert whiffers > contact
    # No card at all and no recent ones: the league, i.e. the current price.
    none = price([])
    assert none.loc[0, "p_matchup"] == pytest.approx(none.loc[0, "p_model"])


def test_paired_brier_is_negative_when_the_first_arm_is_better():
    priced = pd.DataFrame({
        "game_pk": [1, 1, 2, 2], "over_hit": [True, True, False, False],
        "p_matchup": [0.9, 0.8, 0.1, 0.3], "p_model": [0.5, 0.5, 0.5, 0.5]})
    out = props.paired_brier(priced, "p_matchup", "p_model")
    assert out["n"] == 4 and out["diff"] < 0
    # Two games, not four contracts: the clustered SE knows it.
    assert out["se"] > 0


# ───────────────────── resting orders on prop candles ─────────────────────

FIRST_PITCH = 1_788_000_000


def _candles(rows) -> pd.DataFrame:
    """(hours before first pitch, low, high, volume) → a candle frame."""
    return pd.DataFrame([{
        "market_id": "KXMLBHIT-X-1", "game_pk": 1,
        "end_period_ts": FIRST_PITCH - int(h * 3600),
        "yes_bid_close": 0.10, "yes_ask_close": 0.12,
        "price_open": lo, "price_high": hi, "price_low": lo, "price_close": hi,
        "volume": vol,
    } for h, lo, hi, vol in rows])[backfill.CANDLE_COLUMNS]


def test_the_fill_is_the_first_traded_hour_that_reached_the_limit():
    candles = _candles([(5, 0.30, 0.32, 50), (4, 0.24, 0.31, 40), (3, 0.20, 0.26, 60)])
    fill = pnl.first_fill(candles, "yes", 0.25, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600)
    assert fill["end_period_ts"] == FIRST_PITCH - 4 * 3600
    assert pnl.first_fill(candles, "yes", 0.19, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600) is None


def test_an_hour_that_did_not_trade_never_fills():
    candles = _candles([(5, 0.10, 0.11, 0)])
    assert pnl.first_fill(candles, "yes", 0.25, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600) is None


def test_a_candle_at_or_after_first_pitch_never_fills():
    """In-play prints are not a price a pre-game resting order could reach."""
    at = _candles([(0, 0.05, 0.06, 500)])
    after = pd.DataFrame([{**at.iloc[0].to_dict(),
                           "end_period_ts": FIRST_PITCH + 3600}])[backfill.CANDLE_COLUMNS]
    # The hour *ending exactly at* first pitch is the last one an order can be
    # filled in; anything later is in-game and is refused.
    assert pnl.first_fill(at, "yes", 0.25, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600) is not None
    assert pnl.first_fill(after, "yes", 0.25, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600) is None
    # And the archive itself refuses to keep them: a candle an hour after the
    # first pitch on the ticker is dropped before it is ever written.
    market = {"ticker": "KXMLBHIT-26AUG201840NYMTB-NYMJSOTO2-1",
              "event_ticker": "KXMLBHIT-26AUG201840NYMTB"}
    fp = int(pd.Timestamp(
        kalshi.parse_event(market["event_ticker"])["game_start"]).timestamp())

    def candle(ts):
        return {"end_period_ts": ts, "price": {"low_dollars": "0.10",
                                               "high_dollars": "0.20"},
                "yes_bid": {}, "yes_ask": {}, "volume_fp": 5}

    kept = backfill.prop_candle_rows(market, [candle(fp - 3600), candle(fp),
                                              candle(fp + 3600)])
    assert [r["end_period_ts"] for r in kept] == [fp - 3600, fp]


def test_the_no_side_is_the_yes_side_mirrored():
    candles = _candles([(5, 0.60, 0.70, 30), (4, 0.55, 0.74, 20)])
    # Buying NO at 27¢ is selling YES at 73¢: the hour whose high reached 74¢.
    fill = pnl.first_fill(candles, "no", 0.27, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600)
    assert fill["end_period_ts"] == FIRST_PITCH - 4 * 3600
    assert pnl.first_fill(candles, "no", 0.20, FIRST_PITCH,
                          FIRST_PITCH - 24 * 3600) is None


def _maker_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "date": "2026-08-20", "game_pk": 1, "market_id": "KXMLBHIT-X-1",
        "first_pitch_ts": FIRST_PITCH, "p_home_close": 0.30, "home_win": True,
        "model": 0.40,
    }])


def test_a_prop_maker_order_is_posted_below_our_own_price_and_kept_if_unfilled():
    candles = pnl.candle_index(_candles([(5, 0.38, 0.42, 10)]))
    bets = pnl.maker_bet_frame(_maker_frame(), "model", candles, 0.02)
    assert len(bets) == 1
    assert bets.loc[0, "side"] == "yes"
    assert bets.loc[0, "limit"] == pytest.approx(0.38)
    assert bool(bets.loc[0, "filled"])
    # Two cents wider and the market never comes to it, but the order still
    # counts against the strategy's denominator.
    none = pnl.maker_bet_frame(_maker_frame(), "model", candles, 0.05)
    assert len(none) == 1 and not bool(none.loc[0, "filled"])
    assert none.loc[0, "stake"] == 0.0 and none.loc[0, "profit"] == 0.0


def test_the_prop_maker_grid_clusters_its_bootstrap_by_game():
    """Three contracts on one afternoon are one observation, not three."""
    rows = []
    for i, (pk, won) in enumerate([(1, True), (1, True), (1, True),
                                   (2, False), (2, False), (2, False)]):
        rows.append({"date": "2026-08-2%d" % (pk,), "game_pk": pk,
                     "market_id": f"M{i}", "first_pitch_ts": FIRST_PITCH,
                     "p_home_close": 0.30, "home_win": won, "model": 0.40})
    frame = pd.DataFrame(rows)
    candles = pd.concat([_candles([(5, 0.30, 0.42, 10)]).assign(market_id=f"M{i}")
                         for i in range(6)], ignore_index=True)
    wide = pnl.maker_grid(frame, candles, ["model"], margins=(0.02,),
                          stakings=("flat",), draws=500, split=False,
                          group_col="game_pk")
    narrow = pnl.maker_grid(frame, candles, ["model"], margins=(0.02,),
                            stakings=("flat",), draws=500, split=False)
    w = wide.iloc[0], narrow.iloc[0]
    assert (w[0]["roi_hi"] - w[0]["roi_lo"]) > (w[1]["roi_hi"] - w[1]["roi_lo"])


def test_the_limit_price_is_a_function_of_the_model_and_the_margin_alone():
    """No quantity from the candle archive may enter the price we quote."""
    side, limit = pnl.limit_price(np.array([0.40]), np.array([0.30]), 0.02)
    assert side[0] == "yes" and limit[0] == pytest.approx(0.38)
    side, limit = pnl.limit_price(np.array([0.20]), np.array([0.30]), 0.02)
    assert side[0] == "no" and limit[0] == pytest.approx(0.78)
    side, _ = pnl.limit_price(np.array([0.30]), np.array([0.30]), 0.02)
    assert side[0] == pnl.NO_BET


def test_to_maker_frame_carries_the_market_and_the_cancel_time():
    priced = pd.DataFrame([{
        "game_pk": 1, "game_date": "2026-08-20",
        "game_start": "2026-08-20T18:40:00+00:00", "player_id": 11,
        "prop_stat": "hits", "prop_line": 0.5, "p_over_close": 0.60,
        "bid": 0.59, "ask": 0.61, "over_hit": True, "p_model": 0.75,
        "p_league": 0.60, "market_id": "KXMLBHIT-X-1"}])
    frame = props.to_maker_frame(priced)
    assert frame.loc[0, "market_id"] == "KXMLBHIT-X-1"
    assert frame.loc[0, "first_pitch_ts"] == \
        int(pd.Timestamp("2026-08-20T18:40:00+00:00").timestamp())
