"""Unit tests for the station E posted-lineup term (src/sim/lineups.py).

All synthetic — no network. What has to hold:
  * rates regress toward league average, harder for small samples, and each
    component regresses in its own denominator (PA / AB / BIP)
  * recent seasons outweigh old ones (Marcel 5/4/3)
  * a batter with no history scores exactly league average
  * the event decomposition reproduces the league exactly from league rates
  * the runs estimator returns league runs for a league-average lineup
  * the delta form leaves the team's runs-scored rate untouched for a
    league-average lineup
  * games on or after the date being predicted never enter the rates (leakage)
"""
import numpy as np
import pandas as pd
import pytest

from src.sim import lineups as lu

# A tidy league season: 100,000 PA with modern-ish rates.
#   K 22%, BB+HBP 9.5%, HR 3.0%, BABIP .295, ISO .160
LEAGUE_COUNTS = pd.DataFrame([{
    "batter": 0, "season": 2025,
    "pa": 100000, "ab": 89000, "h": 22050, "doubles": 4300, "triples": 350,
    "hr": 3000, "k": 22000, "bb": 8700, "hbp": 800, "sf": 700,
}])
LEAGUE = lu.league_rates(lu.normalize_counts(LEAGUE_COUNTS))
LG_R9 = 4.50


def counts(batter, season, pa, *, k, bb, hr, h, doubles=0, triples=0, hbp=0,
           sf=0, ab=None):
    return {"batter": batter, "season": season, "pa": pa,
            "ab": pa - bb - hbp - sf if ab is None else ab,
            "h": h, "doubles": doubles, "triples": triples, "hr": hr,
            "k": k, "bb": bb, "hbp": hbp, "sf": sf}


def frame(rows):
    return lu.normalize_counts(pd.DataFrame(rows))


def league_batter(pa=600):
    """A batter whose counts are the league's rates, scaled to `pa`."""
    s = pa / 100000.0
    row = LEAGUE_COUNTS.iloc[0].to_dict()
    row = {k: (v * s if k not in ("batter", "season") else v)
           for k, v in row.items()}
    row["batter"] = 1
    row["season"] = 2026
    return row


def league_rate_frame(index=(1,)):
    return pd.DataFrame(
        [{f"rate_{c}": LEAGUE[f"rate_{c}"] for c in lu.COMPONENTS}] * len(index),
        index=pd.Index(list(index), name="batter"))


# ─── normalize_counts ───

def test_normalize_counts_builds_the_five_denominators():
    out = frame([counts(1, 2026, 100, k=20, bb=8, hr=3, h=25, doubles=5,
                        triples=1, hbp=2, sf=1)])
    r = out.iloc[0]
    assert r["bbhbp"] == 10.0                       # walks and HBP folded
    assert r["ab"] == 100 - 8 - 2 - 1
    assert r["bip"] == r["ab"] - 20 - 3 + 1         # AB - K - HR + SF
    assert r["xb"] == 5 + 2 * 1 + 3 * 3             # ISO numerator
    assert r["hip"] == 25 - 3                       # BABIP numerator


def test_normalize_counts_tolerates_missing_columns():
    out = lu.normalize_counts(pd.DataFrame([{"batter": 1, "season": 2026,
                                             "pa": 10, "ab": 9, "h": 3, "k": 2}]))
    assert out.loc[0, "bbhbp"] == 0.0
    assert out.loc[0, "hip"] == 3.0


# ─── league_rates ───

def test_league_rates_use_each_component_s_own_denominator():
    assert LEAGUE["rate_k"] == pytest.approx(0.22)
    assert LEAGUE["rate_bbhbp"] == pytest.approx(0.095)
    assert LEAGUE["rate_hr"] == pytest.approx(0.03)
    bip = 89000 - 22000 - 3000 + 700
    assert LEAGUE["rate_babip"] == pytest.approx((22050 - 3000) / bip)
    assert LEAGUE["rate_iso"] == pytest.approx(
        (4300 + 2 * 350 + 3 * 3000) / 89000)
    assert LEAGUE["triple_share"] == pytest.approx(350 / 4650)


def test_league_rates_rejects_an_empty_pool():
    with pytest.raises(ValueError):
        lu.league_rates(frame([counts(1, 2026, 0, k=0, bb=0, hr=0, h=0)]))


# ─── marcel_rates: regression toward the league ───

def test_small_sample_is_pulled_further_toward_league_than_large_sample():
    small = lu.marcel_rates(frame([counts(1, 2026, 100, k=10, bb=10, hr=5, h=30,
                                          doubles=8)]), 2026, LEAGUE)
    big = lu.marcel_rates(frame([counts(1, 2026, 1000, k=100, bb=100, hr=50,
                                        h=300, doubles=80)]), 2026, LEAGUE)
    lg_k = LEAGUE["rate_k"]
    # both beat the league strikeout rate; the big sample is trusted further
    assert small.loc[1, "rate_k"] > big.loc[1, "rate_k"]
    assert big.loc[1, "rate_k"] < lg_k
    assert small.loc[1, "rate_k"] < lg_k


def test_each_component_regresses_on_its_own_ballast():
    """Same 300 PA: strikeout rate moves most, BABIP barely moves at all."""
    row = counts(1, 2026, 300, k=30, bb=28, hr=15, h=90, doubles=20)
    rates = lu.marcel_rates(frame([row]), 2026, LEAGUE)

    def pull(c, own):
        return (rates.loc[1, f"rate_{c}"] - LEAGUE[f"rate_{c}"]) / (own - LEAGUE[f"rate_{c}"])

    k_pull = pull("k", 30 / 300)
    babip_pull = pull("babip", (90 - 15) / (300 - 28 - 30 - 15))
    assert k_pull > 0.65          # 300 PA against a 120 PA ballast
    assert babip_pull < 0.20      # 250-odd BIP against a 1640 BIP ballast
    assert k_pull > 3 * babip_pull


def test_a_scalar_ballast_applies_to_every_component():
    row = counts(1, 2026, 200, k=60, bb=20, hr=10, h=50, doubles=12)
    rates = lu.marcel_rates(frame([row]), 2026, LEAGUE, ballast=100)
    norm = frame([row]).iloc[0]
    for c in lu.COMPONENTS:
        num, den = norm[lu.RATE_NUM[c]], norm[lu.RATE_DEN[c]]
        assert rates.loc[1, f"rate_{c}"] == pytest.approx(
            (num + 100 * LEAGUE[f"rate_{c}"]) / (den + 100))


def test_zero_plate_appearances_gives_exactly_league_average():
    rates = lu.marcel_rates(frame([counts(1, 2026, 0, k=0, bb=0, hr=0, h=0)]),
                            2026, LEAGUE)
    for c in lu.COMPONENTS:
        assert rates.loc[1, f"rate_{c}"] == pytest.approx(LEAGUE[f"rate_{c}"])


def test_ballast_size_controls_how_hard_we_regress():
    row = frame([counts(1, 2026, 400, k=40, bb=40, hr=25, h=120, doubles=30)])
    light = lu.marcel_rates(row, 2026, LEAGUE, ballast=50)
    heavy = lu.marcel_rates(row, 2026, LEAGUE, ballast=2000)
    assert light.loc[1, "rate_hr"] > heavy.loc[1, "rate_hr"] > LEAGUE["rate_hr"]


# ─── marcel_rates: recency weighting ───

def test_recent_season_outweighs_older_ones():
    """Identical samples, opposite signal: whichever is recent wins."""
    hot, cold = (dict(k=60, bb=60, hr=40, h=180, doubles=40),
                 dict(k=200, bb=20, hr=5, h=100, doubles=15))
    good_now = lu.marcel_rates(frame([counts(1, 2026, 600, **hot),
                                      counts(1, 2024, 600, **cold)]), 2026, LEAGUE)
    good_then = lu.marcel_rates(frame([counts(1, 2026, 600, **cold),
                                       counts(1, 2024, 600, **hot)]), 2026, LEAGUE)
    assert good_now.loc[1, "rate_hr"] > good_then.loc[1, "rate_hr"]
    assert good_now.loc[1, "rate_k"] < good_then.loc[1, "rate_k"]


def test_weights_are_five_four_three_normalized_to_the_current_season():
    rows = [counts(1, 2026, 100, k=30, bb=10, hr=5, h=25, doubles=5),
            counts(1, 2025, 100, k=20, bb=10, hr=5, h=25, doubles=5),
            counts(1, 2024, 100, k=10, bb=10, hr=5, h=25, doubles=5)]
    rates = lu.marcel_rates(frame(rows), 2026, LEAGUE, ballast=200)
    w = [1.0, 0.8, 0.6]
    num = 30 * w[0] + 20 * w[1] + 10 * w[2] + 200 * LEAGUE["rate_k"]
    den = 100 * sum(w) + 200
    assert rates.loc[1, "rate_k"] == pytest.approx(num / den)
    assert rates.loc[1, "pa_weighted"] == pytest.approx(100 * sum(w))


def test_seasons_outside_the_three_year_window_are_ignored():
    old = counts(1, 2019, 900, k=90, bb=90, hr=60, h=300, doubles=70)
    now = counts(1, 2026, 200, k=60, bb=20, hr=6, h=50, doubles=10)
    with_old = lu.marcel_rates(frame([now, old]), 2026, LEAGUE)
    without = lu.marcel_rates(frame([now]), 2026, LEAGUE)
    assert with_old.loc[1, "rate_k"] == pytest.approx(without.loc[1, "rate_k"])


def test_batter_with_no_rows_at_all_is_absent_from_the_table():
    rates = lu.marcel_rates(frame([counts(1, 2026, 200, k=50, bb=20, hr=8,
                                          h=50, doubles=10)]), 2026, LEAGUE)
    assert 2 not in rates.index
    assert lu.marcel_rates(frame([counts(1, 2019, 200, k=50, bb=20, hr=8,
                                         h=50, doubles=10)]), 2026, LEAGUE).empty


# ─── event_rates: the decomposition is exact at the league ───

def test_league_rates_reproduce_the_league_s_own_event_frequencies():
    ev = lu.event_rates(league_rate_frame(), LEAGUE)
    pa = 100000.0
    assert ev.loc[1, "bbhbp"] == pytest.approx(9500 / pa)
    assert ev.loc[1, "k"] == pytest.approx(22000 / pa)
    assert ev.loc[1, "hr"] == pytest.approx(3000 / pa)
    assert ev.loc[1, "d23"] == pytest.approx((4300 + 350) / pa)
    assert ev.loc[1, "b1"] == pytest.approx((22050 - 3000 - 4300 - 350) / pa)


def test_event_rates_sum_to_one():
    rows = [counts(1, 2026, 500, k=180, bb=20, hr=5, h=110, doubles=15),
            counts(2, 2026, 500, k=60, bb=80, hr=40, h=150, doubles=35,
                   triples=4)]
    ev = lu.event_rates(lu.marcel_rates(frame(rows), 2026, LEAGUE), LEAGUE)
    assert np.allclose(ev[list(lu.EVENTS)].sum(axis=1), 1.0)


# ─── runs_per_pa / lineup_r9 ───

def test_a_league_average_batter_is_worth_zero_runs_above_average():
    assert lu.runs_per_pa(league_rate_frame(), LEAGUE).loc[1] == pytest.approx(0.0)


def test_walks_and_homers_help_and_strikeouts_hurt():
    base = league_rate_frame().iloc[0].to_dict()

    def raa(**over):
        r = dict(base)
        r.update({f"rate_{k}": v for k, v in over.items()})
        return lu.runs_per_pa(r, LEAGUE)

    assert raa(hr=0.06) > 0
    assert raa(bbhbp=0.15) > 0
    assert raa(babip=0.340) > 0
    assert raa(k=0.35) < 0


def test_a_league_average_lineup_scores_exactly_league_runs():
    """The property the whole runs estimator turns on."""
    rates = league_rate_frame(range(1, 10))
    lookup = lu.batter_runs_lookup(rates, LEAGUE)
    assert lu.lineup_r9(range(1, 10), lookup, LG_R9) == pytest.approx(LG_R9)


def test_batters_with_no_history_are_treated_as_league_average():
    assert lu.lineup_r9(range(101, 110), {}, LG_R9) == pytest.approx(LG_R9)


def test_a_better_lineup_scores_more_runs():
    good = lu.marcel_rates(frame([counts(i, 2026, 3000, k=450, bb=400, hr=180,
                                         h=850, doubles=180, triples=15)
                                  for i in range(1, 10)]), 2026, LEAGUE)
    bad = lu.marcel_rates(frame([counts(i, 2026, 3000, k=900, bb=150, hr=30,
                                        h=650, doubles=110, triples=5)
                                 for i in range(1, 10)]), 2026, LEAGUE)
    r_good = lu.lineup_r9(range(1, 10), lu.batter_runs_lookup(good, LEAGUE), LG_R9)
    r_bad = lu.lineup_r9(range(1, 10), lu.batter_runs_lookup(bad, LEAGUE), LG_R9)
    assert r_good > LG_R9 > r_bad
    # and the spread is a plausible number of runs per game, not a blowup
    assert 0.5 < r_good - r_bad < 6.0


def test_lineup_slot_shares_favour_the_top_of_the_order():
    shares = lu.slot_pa_shares(38.0)
    assert shares.sum() == pytest.approx(1.0)
    assert (np.diff(shares) < 0).all()
    assert shares[0] / shares[-1] == pytest.approx(1.21, abs=0.05)


def test_a_star_helps_more_batting_leadoff_than_ninth():
    lookup = {1: 0.05}
    top = lu.lineup_r9([1] + list(range(2, 10)), lookup, LG_R9)
    bottom = lu.lineup_r9(list(range(2, 10)) + [1], lookup, LG_R9)
    assert top > bottom > LG_R9


# ─── blend_lineup_team: the delta form ───

def test_a_league_average_lineup_leaves_the_team_baseline_untouched():
    for team_rs in (3.6, 4.5, 5.4):
        assert lu.blend_lineup_team(LG_R9, team_rs, LG_R9) == pytest.approx(team_rs)


def test_a_better_lineup_raises_and_a_worse_one_lowers_expected_runs():
    assert lu.blend_lineup_team(5.2, 4.4, LG_R9) > 4.4
    assert lu.blend_lineup_team(3.8, 4.4, LG_R9) < 4.4


def test_the_weight_scales_the_delta():
    assert lu.blend_lineup_team(5.5, 4.4, LG_R9, weight=0.5) == pytest.approx(
        4.4 + 0.5 * (5.5 - LG_R9))
    assert lu.blend_lineup_team(5.5, 4.4, LG_R9, weight=0.0) == pytest.approx(4.4)


def test_team_context_survives_the_adjustment():
    """A good offence in front of an average lineup stays a good offence — the
    absolute-level blend this replaced would have dragged it to the mean."""
    good, bad = 5.4, 3.6
    assert (lu.blend_lineup_team(4.9, good, LG_R9)
            - lu.blend_lineup_team(4.9, bad, LG_R9)) == pytest.approx(good - bad)


def test_blend_is_vectorized():
    out = lu.blend_lineup_team(np.array([3.8, 5.2]), np.array([4.5, 4.5]), LG_R9)
    assert out.shape == (2,)
    assert out[0] < 4.5 < out[1]


# ─── team_lineup_baseline ───

def test_a_club_with_no_games_yet_is_league_average():
    assert lu.team_lineup_baseline([], LG_R9) == pytest.approx(LG_R9)
    # and with the regression turned off entirely there is still nothing to
    # divide by, so it must not blow up on the season's first date
    assert lu.team_lineup_baseline([], LG_R9, ballast_games=0.0) == pytest.approx(LG_R9)


def test_an_unregressed_baseline_is_just_the_club_mean():
    assert lu.team_lineup_baseline([5.0, 4.0], LG_R9, ballast_games=0.0) == pytest.approx(4.5)


def test_the_club_baseline_is_regressed_toward_the_league():
    prior = [5.5] * 20
    base = lu.team_lineup_baseline(prior, LG_R9, ballast_games=20.0)
    assert LG_R9 < base < 5.5
    assert base == pytest.approx((20 * 5.5 + 20 * LG_R9) / 40)


def test_more_games_pull_the_baseline_away_from_the_league():
    few = lu.team_lineup_baseline([5.5] * 10, LG_R9, ballast_games=20.0)
    many = lu.team_lineup_baseline([5.5] * 200, LG_R9, ballast_games=20.0)
    assert many > few > LG_R9


def test_measuring_against_the_club_baseline_isolates_the_news():
    """Today's lineup equal to the club's usual one is worth nothing extra,
    even for a club far from league average."""
    prior = [5.4] * 200
    base = lu.team_lineup_baseline(prior, LG_R9)
    assert lu.blend_lineup_team(base, 5.2, base) == pytest.approx(5.2)
    assert lu.blend_lineup_team(base - 0.4, 5.2, base) < 5.2


# ─── games_before: the leakage guard ───

def logs():
    return frame([
        dict(counts(1, 2026, 4, k=1, bb=1, hr=0, h=1), date="2026-04-01"),
        dict(counts(1, 2026, 5, k=2, bb=0, hr=1, h=2), date="2026-04-07"),
        dict(counts(1, 2026, 4, k=0, bb=2, hr=1, h=3), date="2026-04-13"),
    ]).assign(date=["2026-04-01", "2026-04-07", "2026-04-13"])


def test_only_games_strictly_before_the_date_are_counted():
    before = lu.games_before(logs(), "2026-04-13")
    assert before.loc[0, "pa"] == 9          # first two games only
    assert before.loc[0, "k"] == 3


def test_the_game_being_predicted_is_excluded():
    on_the_day = lu.games_before(logs(), "2026-04-13")
    day_after = lu.games_before(logs(), "2026-04-14")
    assert day_after.loc[0, "pa"] == 13
    assert on_the_day.loc[0, "pa"] < day_after.loc[0, "pa"]


def test_future_games_never_leak_into_the_rates():
    early = lu.marcel_rates(lu.games_before(logs(), "2026-04-07"), 2026, LEAGUE)
    late = lu.marcel_rates(lu.games_before(logs(), "2026-04-20"), 2026, LEAGUE)
    assert early.loc[1, "pa_weighted"] == pytest.approx(4)
    assert late.loc[1, "pa_weighted"] == pytest.approx(13)


def test_no_games_yet_yields_an_empty_frame():
    empty = lu.games_before(logs(), "2026-03-01")
    assert empty.empty
    assert lu.marcel_rates(empty, 2026, LEAGUE).empty
    assert lu.games_before(pd.DataFrame(), "2026-05-01").empty


def test_opening_day_lineup_with_prior_seasons_still_gets_rates():
    """No 2026 game yet, but 2025/2024 carry him — the walk-forward case on
    the first date of the season."""
    prior = frame([counts(1, 2025, 600, k=120, bb=60, hr=25, h=160, doubles=32),
                   counts(1, 2024, 600, k=120, bb=60, hr=25, h=160, doubles=32)])
    pool = pd.concat([prior, lu.games_before(logs(), "2026-03-01")],
                     ignore_index=True)
    rates = lu.marcel_rates(pool, 2026, LEAGUE)
    assert rates.loc[1, "pa_weighted"] == pytest.approx(600 * 0.8 + 600 * 0.6)


def test_an_absurd_line_cannot_produce_negative_event_probabilities():
    rates = pd.DataFrame([{"rate_k": 0.9, "rate_bbhbp": 0.4, "rate_hr": 0.3,
                           "rate_iso": 1.5, "rate_babip": 0.9}],
                         index=pd.Index([1], name="batter"))
    ev = lu.event_rates(rates, LEAGUE)
    for e in ("bbhbp", "k", "hr", "b1", "d23"):
        assert ev.loc[1, e] >= 0.0
