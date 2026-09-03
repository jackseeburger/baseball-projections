"""Unit tests for the station E starting-pitcher term (src/sim/starters.py).

All synthetic — no network. What has to hold:
  * rates regress toward league average, harder for small samples
  * recent seasons outweigh old ones (Marcel 5/4/3)
  * a pitcher with no history scores exactly league average
  * the game-level blend weights the starter by his share of the innings
  * appearances on or after the game date never enter the rates (leakage)
"""
import numpy as np
import pandas as pd
import pytest

from src.sim import starters as st

# A tidy league: 1000 BF, 22% K, 8% BB+HBP, 3% HR, 27 outs per 38.5 BF.
LEAGUE = {"rate_k": 0.22, "rate_bbhbp": 0.08, "rate_hr": 0.03, "bf_per_ip": 4.28}
LG_RA9 = 4.50


def counts(pitcher, season, bf, k, bbhbp, hr, outs=None):
    return {"pitcher": pitcher, "season": season, "bf": bf, "k": k,
            "bbhbp": bbhbp, "hr": hr, "outs": outs if outs is not None else bf * 0.7}


def frame(rows):
    return pd.DataFrame(rows)


# ─── normalize_counts ───

def test_normalize_counts_folds_walks_and_hbp():
    raw = pd.DataFrame([{"pitcher": 1, "season": 2025, "bf": 100, "k": 25,
                         "bb": 7, "hbp": 2, "hr": 3, "outs": 70}])
    out = st.normalize_counts(raw)
    assert list(out.columns) == ["pitcher", "season", "bf", "k", "bbhbp", "hr", "outs"]
    assert out.loc[0, "bbhbp"] == 9.0


def test_normalize_counts_tolerates_missing_hbp_and_outs():
    raw = pd.DataFrame([{"pitcher": 1, "season": 2025, "bf": 100, "k": 25,
                         "bb": 7, "hr": 3}])
    out = st.normalize_counts(raw)
    assert out.loc[0, "bbhbp"] == 7.0
    assert out.loc[0, "outs"] == 0.0


# ─── league_rates ───

def test_league_rates_are_pooled_per_batter_faced():
    lg = st.league_rates(frame([counts(1, 2025, 1000, 250, 80, 30, outs=700),
                                counts(2, 2025, 1000, 190, 80, 30, outs=700)]))
    assert lg["rate_k"] == pytest.approx(0.22)
    assert lg["rate_bbhbp"] == pytest.approx(0.08)
    assert lg["rate_hr"] == pytest.approx(0.03)
    assert lg["bf_per_ip"] == pytest.approx(3 * 2000 / 1400)


def test_league_rates_rejects_empty_pool():
    with pytest.raises(ValueError):
        st.league_rates(frame([counts(1, 2025, 0, 0, 0, 0, outs=0)]))


# ─── marcel_rates: regression toward the league ───

def test_small_sample_is_pulled_further_toward_league_than_large_sample():
    """Same rate, different sample: the 100-BF pitcher lands closer to league."""
    small = st.marcel_rates(frame([counts(1, 2026, 100, 40, 8, 3)]), 2026, LEAGUE, ballast=200)
    big = st.marcel_rates(frame([counts(1, 2026, 1000, 400, 80, 30)]), 2026, LEAGUE, ballast=200)
    lg_k = LEAGUE["rate_k"]
    assert lg_k < small.loc[1, "rate_k"] < big.loc[1, "rate_k"]
    # ballast 200 against 100 BF → two-thirds league, one-third the pitcher
    assert small.loc[1, "rate_k"] == pytest.approx((40 + 200 * lg_k) / 300)


def test_home_runs_are_regressed_far_harder_than_strikeouts():
    """FIP's 13x coefficient sits on the noisiest component; the default
    ballasts are what stop it from dominating."""
    df = frame([counts(1, 2026, 600, 300, 24, 30)])   # elite K, double the HR
    rates = st.marcel_rates(df, 2026, LEAGUE)
    k_pull = (rates.loc[1, "rate_k"] - LEAGUE["rate_k"]) / (0.5 - LEAGUE["rate_k"])
    hr_pull = (rates.loc[1, "rate_hr"] - LEAGUE["rate_hr"]) / (0.05 - LEAGUE["rate_hr"])
    assert k_pull > 0.75
    assert hr_pull < 0.25
    assert k_pull > 3 * hr_pull


def test_a_scalar_ballast_applies_to_every_component():
    rates = st.marcel_rates(frame([counts(1, 2026, 100, 40, 8, 3)]), 2026,
                            LEAGUE, ballast=100)
    for c in st.COMPONENTS:
        own = {"k": 0.40, "bbhbp": 0.08, "hr": 0.03}[c]
        assert rates.loc[1, f"rate_{c}"] == pytest.approx(
            (100 * own + 100 * LEAGUE[f"rate_{c}"]) / 200)


def test_zero_batters_faced_gives_exactly_league_average():
    rates = st.marcel_rates(frame([counts(1, 2026, 0, 0, 0, 0)]), 2026, LEAGUE)
    for c in st.COMPONENTS:
        assert rates.loc[1, f"rate_{c}"] == pytest.approx(LEAGUE[f"rate_{c}"])


def test_ballast_size_controls_how_hard_we_regress():
    df = frame([counts(1, 2026, 400, 160, 32, 12)])
    light = st.marcel_rates(df, 2026, LEAGUE, ballast=50)
    heavy = st.marcel_rates(df, 2026, LEAGUE, ballast=800)
    assert light.loc[1, "rate_k"] > heavy.loc[1, "rate_k"] > LEAGUE["rate_k"]


# ─── marcel_rates: recency weighting ───

def test_recent_season_outweighs_older_ones():
    """Identical samples, opposite signal: whichever is recent wins."""
    good_now = st.marcel_rates(frame([counts(1, 2026, 500, 150, 40, 15),
                                      counts(1, 2024, 500, 50, 40, 15)]), 2026, LEAGUE)
    good_then = st.marcel_rates(frame([counts(1, 2026, 500, 50, 40, 15),
                                       counts(1, 2024, 500, 150, 40, 15)]), 2026, LEAGUE)
    assert good_now.loc[1, "rate_k"] > good_then.loc[1, "rate_k"]


def test_weights_are_five_four_three_normalized_to_the_current_season():
    rates = st.marcel_rates(frame([counts(1, 2026, 100, 30, 8, 3),
                                   counts(1, 2025, 100, 20, 8, 3),
                                   counts(1, 2024, 100, 10, 8, 3)]), 2026,
                            LEAGUE, ballast=200)
    w = [1.0, 0.8, 0.6]
    num = 30 * w[0] + 20 * w[1] + 10 * w[2] + 200 * LEAGUE["rate_k"]
    den = 100 * sum(w) + 200
    assert rates.loc[1, "rate_k"] == pytest.approx(num / den)
    assert rates.loc[1, "bf_weighted"] == pytest.approx(100 * sum(w))


def test_seasons_outside_the_three_year_window_are_ignored():
    with_old = st.marcel_rates(frame([counts(1, 2026, 200, 60, 16, 6),
                                      counts(1, 2019, 900, 400, 16, 6)]), 2026, LEAGUE)
    without = st.marcel_rates(frame([counts(1, 2026, 200, 60, 16, 6)]), 2026, LEAGUE)
    assert with_old.loc[1, "rate_k"] == pytest.approx(without.loc[1, "rate_k"])


def test_pitcher_with_no_rows_at_all_is_absent_from_the_table():
    rates = st.marcel_rates(frame([counts(1, 2026, 200, 60, 16, 6)]), 2026, LEAGUE)
    assert 2 not in rates.index
    assert st.marcel_rates(frame([counts(1, 2019, 200, 60, 16, 6)]), 2026, LEAGUE).empty


# ─── fip_ra9: league average in, league RA/9 out ───

def test_league_average_pitcher_scores_exactly_league_ra9():
    rates = pd.DataFrame([{f"rate_{c}": LEAGUE[f"rate_{c}"] for c in st.COMPONENTS}],
                         index=pd.Index([1], name="pitcher"))
    assert st.fip_ra9(rates, LEAGUE, LG_RA9).loc[1] == pytest.approx(LG_RA9)


def test_missing_pitcher_falls_back_to_league_average():
    lookup = st.starter_ra9_lookup(
        st.marcel_rates(frame([counts(1, 2026, 500, 150, 30, 10)]), 2026, LEAGUE),
        LEAGUE, LG_RA9)
    assert 1 in lookup
    assert lookup.get(999, LG_RA9) == pytest.approx(LG_RA9)


def test_strikeouts_lower_and_homers_raise_the_run_estimate():
    def ra9(k, bbhbp, hr):
        rates = pd.DataFrame([{"rate_k": k, "rate_bbhbp": bbhbp, "rate_hr": hr}],
                             index=pd.Index([1], name="pitcher"))
        return st.fip_ra9(rates, LEAGUE, LG_RA9).loc[1]

    base = ra9(**{k.replace("rate_", ""): LEAGUE[k] for k in
                  ("rate_k", "rate_bbhbp", "rate_hr")})
    assert ra9(0.30, 0.08, 0.03) < base          # more strikeouts → fewer runs
    assert ra9(0.22, 0.08, 0.05) > base          # more homers → more runs
    assert ra9(0.22, 0.12, 0.03) > base          # more free passes → more runs


def test_fip_coefficients_are_the_standard_thirteen_three_minus_two():
    lg = dict(LEAGUE)
    rates = pd.DataFrame([{"rate_k": 0.22, "rate_bbhbp": 0.08, "rate_hr": 0.04}],
                         index=pd.Index([1], name="pitcher"))
    delta = st.fip_ra9(rates, lg, LG_RA9).loc[1] - LG_RA9
    assert delta == pytest.approx(13.0 * 0.01 * lg["bf_per_ip"])


# ─── blend_starter_team ───

def test_league_average_starter_leaves_the_team_baseline_untouched():
    """The property the whole design turns on: no pitcher signal, no change."""
    for team_ra in (3.6, 4.5, 5.4):
        assert st.blend_starter_team(LG_RA9, team_ra, LG_RA9) == pytest.approx(team_ra)


def test_starter_moves_the_team_by_his_share_of_the_innings():
    assert st.blend_starter_team(3.5, 4.0, LG_RA9, starter_ip=5.5) == pytest.approx(
        4.0 + 5.5 / 9 * (3.5 - LG_RA9))


def test_a_good_starter_lowers_and_a_bad_one_raises_expected_runs():
    assert st.blend_starter_team(3.0, 4.0, LG_RA9) < 4.0
    assert st.blend_starter_team(6.0, 4.0, LG_RA9) > 4.0


def test_team_context_survives_the_adjustment():
    """A good staff behind a league-average arm stays a good staff — the
    absolute-level blend this replaced would have dragged it to the mean."""
    good, bad = 3.6, 5.4
    assert st.blend_starter_team(LG_RA9, good, LG_RA9) < st.blend_starter_team(
        LG_RA9, bad, LG_RA9)
    # and the gap is preserved exactly, not compressed
    assert (st.blend_starter_team(4.0, bad, LG_RA9)
            - st.blend_starter_team(4.0, good, LG_RA9)) == pytest.approx(bad - good)


def test_blend_endpoints():
    assert st.blend_starter_team(3.0, 5.0, LG_RA9, starter_ip=0.0) == pytest.approx(5.0)
    assert st.blend_starter_team(3.0, 5.0, LG_RA9, starter_ip=9.0) == pytest.approx(
        5.0 + (3.0 - LG_RA9))


def test_blend_is_vectorized():
    out = st.blend_starter_team(np.array([3.0, 6.0]), np.array([4.5, 4.5]), LG_RA9)
    assert out.shape == (2,)
    assert out[0] < 4.5 < out[1]


# ─── appearances_before: the leakage guard ───

def logs():
    return pd.DataFrame([
        {"pitcher": 1, "season": 2026, "date": "2026-04-01", "bf": 20,
         "k": 6, "bbhbp": 2, "hr": 1, "outs": 15},
        {"pitcher": 1, "season": 2026, "date": "2026-04-07", "bf": 25,
         "k": 9, "bbhbp": 1, "hr": 0, "outs": 18},
        {"pitcher": 1, "season": 2026, "date": "2026-04-13", "bf": 30,
         "k": 12, "bbhbp": 3, "hr": 2, "outs": 21},
    ])


def test_only_appearances_strictly_before_the_date_are_counted():
    before = st.appearances_before(logs(), "2026-04-13")
    assert before.loc[0, "bf"] == 45          # first two starts only
    assert before.loc[0, "k"] == 15


def test_the_start_being_predicted_is_excluded():
    """The 04-13 line must not feed the 04-13 prediction."""
    on_the_day = st.appearances_before(logs(), "2026-04-13")
    day_after = st.appearances_before(logs(), "2026-04-14")
    assert day_after.loc[0, "bf"] == 75
    assert on_the_day.loc[0, "bf"] < day_after.loc[0, "bf"]


def test_future_appearances_never_leak_into_the_rates():
    early = st.marcel_rates(st.appearances_before(logs(), "2026-04-07"), 2026, LEAGUE)
    late = st.marcel_rates(st.appearances_before(logs(), "2026-04-20"), 2026, LEAGUE)
    assert early.loc[1, "bf_weighted"] == pytest.approx(20)
    assert late.loc[1, "bf_weighted"] == pytest.approx(75)


def test_no_appearances_yet_yields_an_empty_frame():
    empty = st.appearances_before(logs(), "2026-03-01")
    assert empty.empty
    assert st.marcel_rates(empty, 2026, LEAGUE).empty
    assert st.appearances_before(pd.DataFrame(), "2026-05-01").empty


def test_opening_day_starter_with_prior_seasons_still_gets_a_rate():
    """No 2026 log yet, but 2025/2024 carry him — this is the walk-forward
    case on the first date of the season."""
    prior = frame([counts(1, 2025, 600, 180, 48, 18),
                   counts(1, 2024, 600, 180, 48, 18)])
    pool = pd.concat([prior, st.appearances_before(logs(), "2026-03-01")],
                     ignore_index=True)
    rates = st.marcel_rates(pool, 2026, LEAGUE)
    assert rates.loc[1, "bf_weighted"] == pytest.approx(600 * 0.8 + 600 * 0.6)


def test_absurd_rates_are_clamped_to_a_positive_run_rate():
    """Pythagenpat needs RA/9 > 0; a nonsense line must not produce a negative."""
    rates = pd.DataFrame([{"rate_k": 0.95, "rate_bbhbp": 0.0, "rate_hr": 0.0}],
                         index=pd.Index([1], name="pitcher"))
    assert st.fip_ra9(rates, LEAGUE, LG_RA9).loc[1] == pytest.approx(st.MIN_RA9)


# ─── rate_table: the as-of-date assembly both callers share ───

def sp_inputs():
    """What `rate_inputs` returns, hand-built so no network is needed."""
    prior = frame([counts(1, 2025, 600, 132, 48, 18, outs=420),
                   counts(2, 2025, 600, 132, 48, 18, outs=420),
                   counts(1, 2024, 600, 132, 48, 18, outs=420),
                   counts(2, 2024, 600, 132, 48, 18, outs=420)])
    return {"season": 2026, "prior_counts": prior, "game_logs": logs(),
            "league": st.league_rates(prior)}


def test_rate_table_is_the_walk_forward_chain_in_one_call():
    inputs = sp_inputs()
    lg = inputs["league"]
    out = st.rate_table(inputs, "2026-04-13", LG_RA9)
    manual = st.starter_ra9_lookup(
        st.marcel_rates(
            pd.concat([inputs["prior_counts"],
                       st.appearances_before(inputs["game_logs"], "2026-04-13")],
                      ignore_index=True),
            2026, lg),
        lg, LG_RA9)
    assert out == manual


def test_rate_table_respects_the_as_of_cut():
    """Pitcher 1's April 13 start must not move his April 13 rate."""
    inputs = sp_inputs()
    on_the_day = st.rate_table(inputs, "2026-04-13", LG_RA9)
    day_after = st.rate_table(inputs, "2026-04-14", LG_RA9)
    assert on_the_day[1] != day_after[1]
    # Pitcher 2 has no 2026 logs at all, so no date moves him.
    assert on_the_day[2] == pytest.approx(day_after[2])


def test_a_pitcher_with_only_league_average_history_scores_league_ra9():
    inputs = sp_inputs()
    # Pitcher 2's prior seasons are exactly the league rates by construction.
    assert st.rate_table(inputs, "2026-03-01", LG_RA9)[2] == pytest.approx(LG_RA9)


# ─── game_home_prob: the whole per-game chain ───

TEAM_RATES = pd.DataFrame(
    {"rs_pg": [4.5, 4.5, 5.2], "ra_pg": [4.5, 4.5, 4.0]}, index=[108, 109, 110])


def test_two_league_average_starters_reproduce_plain_log5():
    """The property that makes this term safe to wire in: no pitcher
    information, no change to the production number."""
    from src.sim.strength import home_win_prob, pythagenpat
    p, no_history = st.game_home_prob(
        TEAM_RATES, 110, 109, (1, 2), {1: LG_RA9, 2: LG_RA9}, LG_RA9, hfa=0.54)
    plain = home_win_prob(pythagenpat(5.2, 4.0, 1.0), pythagenpat(4.5, 4.5, 1.0), 0.54)
    assert p == pytest.approx(float(plain))
    assert no_history == 0


def test_an_ace_on_the_mound_raises_his_side():
    even = st.game_home_prob(TEAM_RATES, 108, 109, (1, 2),
                             {1: LG_RA9, 2: LG_RA9}, LG_RA9, hfa=0.54)[0]
    home_ace = st.game_home_prob(TEAM_RATES, 108, 109, (1, 2),
                                 {1: 2.9, 2: LG_RA9}, LG_RA9, hfa=0.54)[0]
    away_ace = st.game_home_prob(TEAM_RATES, 108, 109, (1, 2),
                                 {1: LG_RA9, 2: 2.9}, LG_RA9, hfa=0.54)[0]
    assert away_ace < even < home_ace
    # Two identical teams and mirrored aces: near-symmetric about the even
    # number, but only near — Pythagenpat's exponent moves with the game's run
    # environment, so putting the ace on the home side is not the exact
    # reflection of putting him on the road side.
    assert (home_ace - even) == pytest.approx(even - away_ace, abs=0.01)


def test_a_starter_with_no_history_is_counted_and_scored_league_average():
    p, no_history = st.game_home_prob(TEAM_RATES, 108, 109, (7, 8), {}, LG_RA9,
                                      hfa=0.54)
    assert no_history == 2
    assert p == pytest.approx(0.54)      # two identical teams, both arms average


def test_hfa_still_applies_on_top_of_the_starter_adjustment():
    no_hfa = st.game_home_prob(TEAM_RATES, 108, 109, (1, 2),
                               {1: 3.2, 2: 5.2}, LG_RA9, hfa=0.5)[0]
    with_hfa = st.game_home_prob(TEAM_RATES, 108, 109, (1, 2),
                                 {1: 3.2, 2: 5.2}, LG_RA9, hfa=0.54)[0]
    assert with_hfa > no_hfa > 0.5


def test_starter_ip_zero_collapses_to_the_team_only_model():
    from src.sim.strength import home_win_prob, pythagenpat
    p = st.game_home_prob(TEAM_RATES, 110, 109, (1, 2), {1: 2.5, 2: 6.5},
                          LG_RA9, hfa=0.54, starter_ip=0.0)[0]
    plain = home_win_prob(pythagenpat(5.2, 4.0, 1.0), pythagenpat(4.5, 4.5, 1.0), 0.54)
    assert p == pytest.approx(float(plain))


# ─── expected_starter_ip: how deep this man goes, regressed toward 5.5 ───

def start_log(pitcher, date, outs, gs=1):
    return {"pitcher": pitcher, "date": date, "outs": outs, "gs": gs}


def test_only_starts_contribute_innings():
    out = st.start_innings(frame([start_log(1, "2026-05-01", 18),
                                  start_log(1, "2026-05-06", 6, gs=0)]))
    assert out["date"].tolist() == ["2026-05-01"]
    assert out["ip"].tolist() == [6.0]


def test_an_empty_or_reliever_only_log_gives_no_starts():
    assert st.start_innings(frame([])).empty
    assert st.start_innings(frame([start_log(1, "2026-05-01", 3, gs=0)])).empty


def test_no_starts_on_file_leaves_the_pitcher_absent():
    starts = st.start_innings(frame([start_log(1, "2026-05-01", 21)]))
    assert st.expected_starter_ip(starts, "2026-05-01") == {}
    assert st.expected_starter_ip(st.start_innings(frame([])), "2026-05-01") == {}


def test_a_deep_start_projects_deeper_than_the_prior():
    starts = st.start_innings(frame([start_log(1, "2026-05-01", 27)]))   # 9 IP
    assert st.expected_starter_ip(starts, "2026-05-02")[1] > st.STARTER_IP


def test_a_ballast_pulls_one_start_back_toward_the_prior():
    one = st.start_innings(frame([start_log(1, "2026-05-01", 27)]))
    hard = st.expected_starter_ip(one, "2026-05-02", ballast=20.0)[1]
    soft = st.expected_starter_ip(one, "2026-05-02", ballast=0.0)[1]
    assert st.STARTER_IP < hard < soft


def test_many_deep_starts_outweigh_a_ballast_that_one_does_not():
    one = st.start_innings(frame([start_log(1, "2026-05-01", 27)]))
    many = st.start_innings(frame([start_log(1, "2026-05-%02d" % d, 27)
                                   for d in range(1, 21)]))
    assert (st.expected_starter_ip(many, "2026-06-01", ballast=20.0)[1]
            > st.expected_starter_ip(one, "2026-06-01", ballast=20.0)[1])


def test_a_short_starter_projects_shorter_than_the_prior():
    starts = st.start_innings(frame([start_log(1, "2026-05-%02d" % d, 9)
                                     for d in range(1, 21)]))            # 3 IP
    assert st.expected_starter_ip(starts, "2026-06-01")[1] < st.STARTER_IP


def test_a_huge_ballast_reproduces_the_flat_split():
    starts = st.start_innings(frame([start_log(1, "2026-05-01", 27)]))
    ip = st.expected_starter_ip(starts, "2026-05-02", ballast=1e6)[1]
    assert ip == pytest.approx(st.STARTER_IP, abs=1e-4)


def test_the_projection_is_clipped_to_a_usable_range():
    starts = st.start_innings(frame([start_log(1, "2026-05-%02d" % d, 0)
                                     for d in range(1, 29)]))
    assert (st.expected_starter_ip(starts, "2026-06-01", ballast=0.0)[1]
            == st.MIN_STARTER_IP)


def test_tonight_s_start_never_sets_its_own_workload_split():
    """The leakage guard: how long this outing lasts is the outcome."""
    before = st.start_innings(frame([start_log(1, "2026-05-01", 27)]))
    after = st.start_innings(frame([start_log(1, "2026-05-01", 27),
                                    start_log(1, "2026-05-02", 3),
                                    start_log(1, "2026-05-09", 3)]))
    assert (st.expected_starter_ip(after, "2026-05-02")
            == st.expected_starter_ip(before, "2026-05-02"))
