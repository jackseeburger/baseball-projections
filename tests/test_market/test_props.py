"""Player props end to end on synthetic payloads — no network.

Covers the four joints where a prop can go wrong silently: the venue parse
(which player, which stat, which line), the name→MLBAM resolution, the
probability arithmetic, and the walk-forward guard on the rates.
"""
import gzip
import json
import sys
from math import comb, exp
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.market import backfill, kalshi, players, pnl, polymarket, props, snapshot
from src.market import teams as T
from src.market.schema import FIELDS, PROP_MARKET_TYPES, validate

TS = "2026-09-02T18:00:00+00:00"
EVENT = "KXMLBHIT-26SEP022010CWSHOU"

# One live-shaped market per prop series (trimmed from the 2026-09-02 probe).
PROP_MARKETS = [
    ("KXMLBHIT", "HOULWADE31", 3, "LaMonte Wade Jr.", "hits", "LaMonte Wade Jr.: 3+ hits?"),
    ("KXMLBHR", "HOUCVZQUEZ2", 2, "Christian Vázquez", "hr", "Christian Vázquez: 2+ home runs?"),
    ("KXMLBTB", "HOULWADE31", 4, "LaMonte Wade Jr.", "tb", "LaMonte Wade Jr.: 4+ total bases?"),
    ("KXMLBRBI", "HOULWADE31", 1, "LaMonte Wade Jr.", "rbi", "LaMonte Wade Jr.: 1+ RBIs?"),
    ("KXMLBSB", "MILLLARA18", 1, "Luis Lara", "sb", "Luis Lara: 1+ stolen bases?"),
    ("KXMLBKS", "PITBCHANDLER36", 8, "Bubba Chandler", "k", "Bubba Chandler: 8+ strikeouts?"),
    ("KXMLBOUTS", "CWSDMARTIN65", 14, "Davis Martin", "outs", "Davis Martin: 14+ Outs Recorded?"),
]


def make_market(series, player_code, n, name, title, event=None):
    event = event or f"{series}-26SEP022010CWSHOU"
    return {
        "ticker": f"{event}-{player_code}-{n}",
        "event_ticker": event,
        "title": title,
        "yes_sub_title": f"{name}: {n}+",
        "floor_strike": n - 0.5,
        "strike_type": "greater",
        "yes_bid_dollars": "0.1200", "yes_ask_dollars": "0.1300",
        "last_price_dollars": "0.1200", "volume_fp": "42.00",
        "status": "active",
        "custom_strike": {"baseball_player": "742ca939-106f-43e1-a8e6-0f9b6e23e82b"},
    }


# ───────────────────────────── venue parsing ─────────────────────────────

@pytest.mark.parametrize("series,code,n,name,stat,title", PROP_MARKETS)
def test_every_prop_series_parses(series, code, n, name, stat, title):
    r = kalshi.normalize(make_market(series, code, n, name, title), TS)
    assert r["market_type"] == kalshi.PROP_SERIES[series]
    assert r["prop_stat"] == stat
    assert r["prop_line"] == n - 0.5        # "3+" is the over on 2.5
    assert r["player_name"] == name
    assert r["outcome"] == "Over"
    assert r["game_date"] == "2026-09-02"
    validate(r)


def test_prop_ticker_yields_the_players_club():
    r = kalshi.normalize(*(make_market(*PROP_MARKETS[0][:4], PROP_MARKETS[0][5]), TS))
    assert r["team_abbrev"] == "HOU"        # HOU + an opaque player code
    two = kalshi.prop_team_id("KXMLBKS-26SEP021840SFPIT-SFLWEBB62-6")
    assert two == T.ABBREV_TO_ID["SF"]      # two-letter abbreviations too
    assert kalshi.prop_team_id("KXMLBGAME-26SEP042210WSHLAD-WSH") is None


def test_prop_label_parse_and_threshold():
    assert kalshi.parse_prop_label("Bubba Chandler: 8+") == ("Bubba Chandler", 7.5)
    assert kalshi.parse_prop_label("Washington") == (None, None)
    assert props.threshold(7.5) == 8 and props.threshold(0.5) == 1


def test_polymarket_prop_splits_by_sports_market_type():
    event = {"slug": "mlb-det-cle-2026-06-14-player-props",
             "startTime": "2026-06-14T18:10:00Z",
             "teams": [{"name": "Detroit Tigers", "abbreviation": "det", "ordering": "away"},
                       {"name": "Cleveland Guardians", "abbreviation": "cle", "ordering": "home"}],
             "markets": [
                 {"id": "1", "question": "Angel Martínez: Home Runs O/U 0.5",
                  "outcomes": '["Over","Under"]', "outcomePrices": '["0.2","0.8"]',
                  "sportsMarketType": "baseball_player_home_runs", "line": 0.5,
                  "active": True},
                 {"id": "2", "question": "Casey Mize: Strikeouts O/U 5.5",
                  "outcomes": '["Over","Under"]', "outcomePrices": '["0.5","0.5"]',
                  "sportsMarketType": "baseball_player_strikeouts", "line": 5.5,
                  "active": True},
                 {"id": "3", "question": "Casey Mize: Doubles O/U 0.5",
                  "outcomes": '["Over","Under"]', "outcomePrices": '["0.1","0.9"]',
                  "sportsMarketType": "baseball_player_doubles", "line": 0.5,
                  "active": True},
             ]}
    hr, k, unknown = polymarket.normalize_event(event, TS)
    assert (hr["market_type"], hr["prop_stat"], hr["prop_line"]) == ("prop_hr", "hr", 0.5)
    assert hr["player_name"] == "Angel Martínez" and hr["outcome"] == "Over"
    assert (k["market_type"], k["prop_stat"], k["prop_line"]) == ("prop_k", "k", 5.5)
    # An unmapped baseball_player_* stays the generic type and carries no
    # prop fields, so the schema guard cannot be tripped by a new venue stat.
    assert unknown["market_type"] == "player_prop" and unknown["prop_stat"] is None
    for r in (hr, k, unknown):
        validate(r)


# ───────────────────────────── schema ─────────────────────────────

def test_schema_rejects_prop_fields_on_a_non_prop():
    r = kalshi.normalize(make_market(*PROP_MARKETS[0][:4], PROP_MARKETS[0][5]), TS)
    r["market_type"] = "moneyline"
    with pytest.raises(ValueError, match="prop fields"):
        validate(r)


def test_schema_rejects_a_mismatched_prop_stat():
    r = kalshi.normalize(make_market(*PROP_MARKETS[0][:4], PROP_MARKETS[0][5]), TS)
    r["prop_stat"] = "hr"
    with pytest.raises(ValueError, match="does not match"):
        validate(r)


def test_snapshot_round_trip_keeps_the_prop_fields(tmp_path):
    recs = [kalshi.normalize(make_market(s, c, n, name, title), TS)
            for s, c, n, name, _, title in PROP_MARKETS]
    for i, r in enumerate(recs):
        r["player_id"] = 500000 + i
    path = snapshot.write(recs, TS, tmp_path)
    with gzip.open(path, "rt") as fh:
        raw = [json.loads(line) for line in fh]
    assert all(set(r) == set(FIELDS) for r in raw)
    back = snapshot.read(path)
    assert list(back["prop_stat"]) == [s for *_, s, _ in
                                       [(a, b, c, d, e, f) for a, b, c, d, e, f in PROP_MARKETS]]
    assert list(back["player_id"]) == [500000 + i for i in range(len(recs))]
    assert set(back["market_type"]) <= set(PROP_MARKET_TYPES)


# ───────────────────────── name resolution ─────────────────────────

STUB_PEOPLE = [
    {"id": 111, "fullName": "LaMonte Wade Jr."},
    {"id": 222, "fullName": "Christian Vázquez"},
    {"id": 333, "fullName": "Bubba Chandler"},
    {"id": 444, "fullName": "José Fermín"},
    {"id": 555, "fullName": "José Fermin"},     # a real 2026 collision
]


@pytest.fixture
def resolver():
    return players.NameResolver(2026, people=STUB_PEOPLE, search=False)


def test_resolver_folds_accents_and_punctuation(resolver):
    assert resolver.resolve("LaMonte Wade Jr.") == 111
    assert resolver.resolve("Lamonte Wade Jr") == 111
    assert resolver.resolve("Christian Vazquez") == 222      # Polymarket drops accents
    assert resolver.resolve("Christian Vázquez") == 222


def test_resolver_refuses_to_guess_an_ambiguous_name(resolver):
    assert resolver.resolve("José Fermín") is None
    assert "jose fermin" in resolver.ambiguous


def test_resolver_reports_misses(resolver):
    assert resolver.resolve("Nobody At All") is None
    assert "Nobody At All" in resolver.misses


def test_assign_player_ids_reports_the_resolution_rate(resolver):
    recs = [kalshi.normalize(make_market(s, c, n, name, title), TS)
            for s, c, n, name, _, title in PROP_MARKETS]
    stats = players.assign_player_ids(recs, resolver)
    assert stats["prop_markets"] == len(PROP_MARKETS) and stats["named"] == len(PROP_MARKETS)
    assert stats["resolved"] == 5   # Wade on 3 tickets, Vázquez, Chandler
    assert 0 < stats["resolution_rate"] < 1
    by_name = {r["player_name"]: r["player_id"] for r in recs}
    assert by_name["LaMonte Wade Jr."] == 111 and by_name["Luis Lara"] is None


def test_assign_player_ids_leaves_non_props_alone(resolver):
    ml = kalshi.normalize({"ticker": "KXMLBGAME-26SEP042210WSHLAD-WSH",
                           "event_ticker": "KXMLBGAME-26SEP042210WSHLAD",
                           "yes_sub_title": "Washington", "status": "active"}, TS)
    players.assign_player_ids([ml], resolver)
    assert ml["player_id"] is None and ml["prop_stat"] is None


# ───────────────────────── probability arithmetic ─────────────────────────

def test_binom_at_least_matches_the_closed_form():
    # P(at least 1 hit in 4 PA at .300) = 1 - .7^4
    assert props.binom_at_least(1, 4, 0.3) == pytest.approx(1 - 0.7 ** 4)
    # P(at least 2 in 5 at .250), summed by hand
    want = 1 - (0.75 ** 5 + comb(5, 1) * 0.25 * 0.75 ** 4)
    assert props.binom_at_least(2, 5, 0.25) == pytest.approx(want)


def test_binom_at_least_interpolates_a_fractional_n():
    lo = props.binom_at_least(1, 4, 0.3)
    hi = props.binom_at_least(1, 5, 0.3)
    mid = props.binom_at_least(1, 4.6, 0.3)
    assert lo < mid < hi
    assert mid == pytest.approx(0.4 * lo + 0.6 * hi)


def test_binom_at_least_edges():
    assert props.binom_at_least(0, 4, 0.3) == 1.0
    assert props.binom_at_least(5, 4, 0.3) == 0.0        # more than the trials
    assert props.binom_at_least(1, 4, 0.0) == 0.0
    assert props.binom_at_least(4, 4, 1.0) == pytest.approx(1.0)


def test_poisson_at_least_matches_the_closed_form():
    assert props.poisson_at_least(1, 1.4) == pytest.approx(1 - exp(-1.4))
    assert props.poisson_at_least(2, 1.4) == pytest.approx(1 - exp(-1.4) * (1 + 1.4))
    assert props.poisson_at_least(0, 1.4) == 1.0


def test_slot_pa_runs_from_leadoff_to_ninth():
    assert props.slot_pa(1) == 4.6 and props.slot_pa(9) == 3.7
    assert props.SLOT_PA[1] > props.SLOT_PA[5] > props.SLOT_PA[9]
    assert props.slot_pa(None) == props.DEFAULT_PA


def test_a_better_hitter_prices_higher_at_the_same_line():
    lg = {"rate_k": 0.22, "rate_bbhbp": 0.09, "rate_hr": 0.032, "rate_iso": 0.16,
          "rate_babip": 0.295, "nonab_share": 0.02, "sf_share": 0.005,
          "triple_share": 0.08}
    good = props.pa_outcome_probs({"rate_k": 0.14, "rate_bbhbp": 0.11, "rate_hr": 0.06,
                                   "rate_iso": 0.28, "rate_babip": 0.330}, lg)
    poor = props.pa_outcome_probs({"rate_k": 0.30, "rate_bbhbp": 0.06, "rate_hr": 0.012,
                                   "rate_iso": 0.08, "rate_babip": 0.270}, lg)
    assert good["hit"] > poor["hit"] and good["hr"] > poor["hr"] and good["tb"] > poor["tb"]
    for stat in ("hits", "hr", "tb"):
        assert props.batter_prop_prob(stat, 0.5, good, 4.6) > \
            props.batter_prop_prob(stat, 0.5, poor, 4.6)
    # A higher line is always less likely.
    assert props.batter_prop_prob("hits", 1.5, good, 4.6) < \
        props.batter_prop_prob("hits", 0.5, good, 4.6)


def test_unpriceable_stats_raise_rather_than_guess():
    with pytest.raises(ValueError):
        props.batter_prop_prob("rbi", 0.5, {"hit": 0.3, "hr": 0.04, "tb": 0.4}, 4.6)
    with pytest.raises(ValueError):
        props.pitcher_prop_prob("outs", 14.5, 0.25)
    assert set(props.UNPRICED) == {"rbi", "sb", "outs"}


# ───────────────────────────── leakage ─────────────────────────────

def _batter_ctx():
    """Two batters, identical priors; batter 2 hits 9 homers on the game date."""
    prior = pd.DataFrame({
        "batter": [1, 2], "season": [2025, 2025], "pa": [600, 600], "ab": [540, 540],
        "h": [140, 140], "doubles": [28, 28], "triples": [2, 2], "hr": [20, 20],
        "k": [120, 120], "bb": [55, 55], "hbp": [5, 5], "sf": [5, 5],
    })
    # A league frame wide enough for league_rates to be sane.
    league_rows = pd.concat([prior] * 40, ignore_index=True)
    prior_counts = props.lu_model.normalize_counts(league_rows)
    logs = pd.DataFrame({
        "batter": [1, 2, 2], "season": [2026, 2026, 2026],
        "date": ["2026-08-01", "2026-08-01", "2026-08-15"],
        "pa": [4, 4, 9], "ab": [4, 4, 9], "h": [1, 1, 9], "doubles": [0, 0, 0],
        "triples": [0, 0, 0], "hr": [0, 0, 9], "k": [1, 1, 0], "bb": [0, 0, 0],
        "hbp": [0, 0, 0], "sf": [0, 0, 0],
    })
    game_logs = props.lu_model.normalize_counts(logs)
    game_logs["date"] = logs["date"].to_numpy()
    return {"season": 2026, "prior_counts": prior_counts, "game_logs": game_logs,
            "league": props.lu_model.league_rates(prior_counts)}


def test_rates_use_only_games_strictly_before_the_date():
    ctx = _batter_ctx()
    before = props.batter_rates(ctx, "2026-08-15")
    after = props.batter_rates(ctx, "2026-08-16")
    assert before.loc[2, "rate_hr"] < after.loc[2, "rate_hr"]
    # On the day itself the two batters are still indistinguishable.
    assert before.loc[1, "rate_hr"] == pytest.approx(before.loc[2, "rate_hr"], abs=1e-9)


def test_price_never_sees_the_game_it_prices():
    ctx = _batter_ctx()
    pitchers = {"season": 2026, "league": {"rate_k": 0.22},
                "prior_counts": pd.DataFrame(columns=["pitcher", "season", "bf", "k",
                                                      "bbhbp", "hr", "outs"]),
                "game_logs": pd.DataFrame(columns=["pitcher", "season", "bf", "k",
                                                   "bbhbp", "hr", "outs", "date"])}
    closes = pd.DataFrame([
        {"game_pk": 700001, "game_date": "2026-08-15", "player_id": 2,
         "prop_stat": "hr", "prop_line": 0.5, "p_over_close": 0.12, "over_hit": True},
        {"game_pk": 700002, "game_date": "2026-08-16", "player_id": 2,
         "prop_stat": "hr", "prop_line": 0.5, "p_over_close": 0.12, "over_hit": False},
    ])
    slots = {(700001, 2): 3, (700002, 2): 3}
    priced = props.price(closes, ctx, pitchers, slots, stats=("hr",))
    assert len(priced) == 2
    same_day, next_day = priced.sort_values("game_date")["p_model"]
    assert same_day < next_day          # the 9-homer game only counts afterwards
    assert priced["exp_pa"].eq(props.SLOT_PA[3]).all()


def test_price_skips_a_hitter_who_was_not_in_the_posted_lineup():
    ctx = _batter_ctx()
    pitchers = {"season": 2026, "league": {"rate_k": 0.22},
                "prior_counts": pd.DataFrame(columns=["pitcher", "season", "bf", "k",
                                                      "bbhbp", "hr", "outs"]),
                "game_logs": pd.DataFrame(columns=["pitcher", "season", "bf", "k",
                                                   "bbhbp", "hr", "outs", "date"])}
    closes = pd.DataFrame([
        {"game_pk": 700001, "game_date": "2026-08-15", "player_id": 1,
         "prop_stat": "hr", "prop_line": 0.5, "p_over_close": 0.12, "over_hit": False},
    ])
    assert props.price(closes, ctx, pitchers, {}, stats=("hr",)).empty


# ───────────────────────────── the P&L join ─────────────────────────────

def _priced() -> pd.DataFrame:
    """Four contracts on two games; the model likes the over on both winners."""
    return pd.DataFrame([
        {"game_pk": 1, "game_date": "2026-08-01", "player_id": 11, "prop_stat": "hits",
         "prop_line": 0.5, "p_over_close": 0.60, "bid": 0.59, "ask": 0.61,
         "over_hit": True, "p_model": 0.75, "p_league": 0.60},
        {"game_pk": 1, "game_date": "2026-08-01", "player_id": 11, "prop_stat": "hits",
         "prop_line": 1.5, "p_over_close": 0.25, "bid": 0.24, "ask": 0.26,
         "over_hit": False, "p_model": 0.10, "p_league": 0.25},
        {"game_pk": 2, "game_date": "2026-08-02", "player_id": 12, "prop_stat": "hr",
         "prop_line": 0.5, "p_over_close": 0.12, "bid": 0.11, "ask": 0.13,
         "over_hit": True, "p_model": 0.30, "p_league": 0.12},
        {"game_pk": 2, "game_date": "2026-08-02", "player_id": 12, "prop_stat": "hr",
         "prop_line": 1.5, "p_over_close": 0.02, "bid": 0.01, "ask": 0.03,
         "over_hit": None, "p_model": 0.02, "p_league": 0.02},
    ])


def test_to_pnl_frame_drops_unsettled_and_renames():
    frame = props.to_pnl_frame(_priced())
    assert len(frame) == 3                      # the unsettled contract is gone
    assert {"date", "p_home_close", "home_win", "model", "league"} <= set(frame)
    assert frame["home_win"].tolist() == [True, False, True]


def test_pnl_runs_on_props_and_the_market_never_trades():
    frame = pnl.add_controls(props.to_pnl_frame(_priced()), seed=0)
    row = pnl.evaluate(frame, "model", pnl.KALSHI, 0.02, "flat", draws=200,
                       group_col="game_pk")
    assert row["n_bets"] == 3                   # two overs and one under, all ≥ 2 pts
    assert row["hit_rate"] == 1.0 and row["total_return"] > 0
    market = pnl.evaluate(frame, "market", pnl.KALSHI, 0.0, "flat", draws=200,
                          group_col="game_pk")
    assert market["n_bets"] == 0 and market["roi"] == 0.0


def test_cluster_bootstrap_is_wider_than_the_row_bootstrap():
    """Three contracts on one game are one observation, not three."""
    profit = pd.Series([0.4, 0.4, 0.4, -1.0, -1.0, -1.0]).to_numpy()
    stake = pd.Series([1.0] * 6).to_numpy()
    games = [1, 1, 1, 2, 2, 2]
    rows = pnl.bootstrap_roi_ci(profit, stake, draws=2000, seed=1)
    clustered = pnl.bootstrap_roi_ci(profit, stake, draws=2000, seed=1, groups=games)
    assert (clustered[1] - clustered[0]) > (rows[1] - rows[0])


def test_prop_frame_joins_game_pk_and_player_id(resolver):
    schedule = pd.DataFrame([{"game_pk": 777001, "date": "2026-09-02",
                              "home_id": T.ABBREV_TO_ID["HOU"],
                              "away_id": T.ABBREV_TO_ID["CWS"]}])
    rows = [{
        "venue": "kalshi", "game_date": "2026-09-02",
        "game_start": "2026-09-03T00:10:00+00:00",
        "home_id": T.ABBREV_TO_ID["HOU"], "away_id": T.ABBREV_TO_ID["CWS"],
        "team_id": T.ABBREV_TO_ID["HOU"], "player_name": "LaMonte Wade Jr.",
        "prop_stat": "hits", "prop_line": 2.5, "p_over_close": 0.08,
        "bid": 0.07, "ask": 0.09, "last_trade": 0.08, "close_ts": 1788000000,
        "minutes_before_pitch": 40.0, "volume_pre": 12.0, "volume_total": 42.0,
        "n_obs": 3, "market_id": f"{EVENT}-HOULWADE31-3", "result": "no",
        "over_hit": False,
    }]
    df = backfill.prop_frame(rows, schedule, resolver=resolver)
    assert list(df.columns) == backfill.PROP_CLOSE_COLUMNS
    assert df.loc[0, "game_pk"] == 777001 and df.loc[0, "player_id"] == 111


def test_candle_price_prefers_the_book_over_a_stale_print():
    stale = {"price": {"previous_dollars": "0.0900"},
             "yes_bid": {"close_dollars": "0.4000"},
             "yes_ask": {"close_dollars": "0.5400"}}
    price, bid, ask, last = backfill.candle_price(stale)
    assert (bid, ask, last) == (0.40, 0.54, 0.09)
    assert price == 0.47                      # the mid, not the print outside it
    live = {"price": {"close_dollars": "0.4500"},
            "yes_bid": {"close_dollars": "0.4400"},
            "yes_ask": {"close_dollars": "0.4600"}}
    assert backfill.candle_price(live)[0] == 0.45
    # A one-sided book with no print is not a price: an ask nobody is bidding
    # against says what a seller wants, not what the contract is worth.
    one_sided = {"price": {}, "yes_bid": {"close_dollars": "0.0000"},
                 "yes_ask": {"close_dollars": "0.0600"}}
    assert backfill.candle_price(one_sided) == (None, None, 0.06, None)
    assert backfill.candle_price({"price": {}, "yes_bid": {}, "yes_ask": {}})[0] is None
