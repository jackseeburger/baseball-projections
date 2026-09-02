"""The live rest-of-season projection — synthetic seasons, no network.

Four things have to hold or the number on the site is wrong:

  * **rates × PA is arithmetic, not a fit.** A .200 K% hitter with 100
    projected PA gets 20 projected strikeouts, and every counting stat in the
    line satisfies the identities the docstring claims.
  * **the league-average line returns the league wOBA.** The rate → line →
    wOBA path involves half a dozen constants; feed it the league's own rates
    and it has to come back where it started, otherwise every player's wOBA is
    off by the same silent bias.
  * **zero projected PA is not a projection.** An injured hitter has no
    rest-of-season line; he must not appear at all.
  * **the as-of cutoff is exclusive.** A game played on the morning of the
    projection has not happened yet. If it leaks in, every backtest number
    that justified this model is measuring something the live build doesn't do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projections import ros as ros_module
from src.projections.ros import (
    COMPONENT_ORDER,
    LEAGUE_HBP_RATE,
    LEAGUE_SF_RATE,
    OUTPUT_COLUMNS,
    WOBA_WEIGHTS,
    build_ros_projections,
    league_rates,
    marcel_rates,
    partial_season,
    ros_counting_line,
)

AS_OF = "2026-08-01"
SEASON = 2026

# The 2026 league line through Sept 1, from the PA parquet. The wOBA these
# rates imply is .3139 — see docs/ros-projections.md.
LEAGUE = {"k_rate": 0.2207, "bb_rate": 0.0892, "hr_rate": 0.0303,
          "babip": 0.2893, "iso": 0.1564}
LEAGUE_WOBA = 0.3139
LEAGUE_PA, LEAGUE_AB, LEAGUE_HBP, LEAGUE_SF = 157749, 140087, 1806, 1127


# ─── fixtures ────────────────────────────────────────────────────

def make_pa_frame(dates, batters=(1, 2), pa_per_day: int = 4) -> pd.DataFrame:
    """PA-level rows in the loader's schema: one strikeout in four, one walk."""
    rows = []
    for day, date in enumerate(pd.to_datetime(list(dates))):
        for batter in batters:
            for i in range(pa_per_day):
                rows.append({
                    "batter": batter, "game_pk": 1000 * day + batter,
                    "game_date": date, "game_year": SEASON,
                    "event": "strikeout" if i == 0 else "single",
                    "is_k": int(i == 0), "is_bb": int(i == 1), "is_hbp": 0,
                    "is_hit": int(i >= 2), "is_hr": int(i == 2),
                    "is_single": int(i == 3), "is_double": 0, "is_triple": 0,
                })
    columns = ["batter", "game_pk", "game_date", "game_year", "event", "is_k",
               "is_bb", "is_hbp", "is_hit", "is_hr", "is_single", "is_double",
               "is_triple"]
    return pd.DataFrame(rows, columns=columns)


def make_seasons(batters=(1, 2), seasons=(2023, 2024, 2025)) -> pd.DataFrame:
    """Prior full seasons in the harness schema — 600 PA of a league-ish line."""
    rows = []
    for batter in batters:
        for season in seasons:
            rows.append({
                "batter": batter, "season": season, "age": 27,
                "pa": 600, "ab": 540, "k": 130, "bb": 55, "hr": 20,
                "xb_points": 95, "bip": 390, "hits_in_play": 110,
            })
    return pd.DataFrame(rows)


def make_playing_time(pa_by_batter: dict[int, float], team_id: int = 100) -> pd.DataFrame:
    return pd.DataFrame([
        {"batter": b, "team_id": team_id, "cutoff_date": AS_OF,
         "games_remaining": 26, "pa_share": 0.1, "projected_pa_ros": pa}
        for b, pa in pa_by_batter.items()
    ])


# ─── rates x PA is arithmetic ────────────────────────────────────

def test_counting_stats_are_the_rate_times_the_plate_appearances():
    line = ros_counting_line(pa_ros=[100.0], k_rate=[0.20], bb_rate=[0.10],
                             hr_rate=[0.05], babip=[0.300], iso=[0.150],
                             hbp_rate=0.01, sf_rate=0.01)
    assert line["k"][0] == pytest.approx(20.0)
    assert line["bb"][0] == pytest.approx(10.0)
    assert line["hr"][0] == pytest.approx(5.0)
    assert line["hbp"][0] == pytest.approx(1.0)
    assert line["sf"][0] == pytest.approx(1.0)


def test_the_line_satisfies_the_identities_it_claims():
    """AB = PA − BB − HBP − SF, BIP = AB − K − HR + SF, H = babip·BIP + HR."""
    line = ros_counting_line([200.0, 350.0], [0.25, 0.18], [0.08, 0.12],
                             [0.04, 0.02], [0.310, 0.280], [0.200, 0.120],
                             hbp_rate=0.012, sf_rate=0.008).iloc[0]
    assert line["ab"] == pytest.approx(line["pa"] - line["bb"] - line["hbp"] - line["sf"])
    assert line["bip"] == pytest.approx(line["ab"] - line["k"] - line["hr"] + line["sf"])
    assert line["h"] == pytest.approx(0.310 * line["bip"] + line["hr"])
    assert line["xb_points"] == pytest.approx(0.200 * line["ab"])
    # 2B + 2·3B + 3·HR = ISO·AB, the definition of isolated power.
    assert (line["doubles"] + 2 * line["triples"] + 3 * line["hr"]) == pytest.approx(
        line["xb_points"])
    assert line["singles"] + line["doubles"] + line["triples"] + line["hr"] == pytest.approx(
        line["h"])


def test_woba_denominator_is_plate_appearances():
    """AB + BB + HBP + SF is PA by construction, so wOBA is a per-PA rate."""
    line = ros_counting_line([500.0], [0.22], [0.09], [0.03], [0.29], [0.16])
    assert (line["ab"] + line["bb"] + line["hbp"] + line["sf"])[0] == pytest.approx(500.0)


def test_doubling_the_plate_appearances_doubles_the_counts_not_the_rate():
    small = ros_counting_line([100.0], [0.22], [0.09], [0.03], [0.29], [0.16])
    big = ros_counting_line([200.0], [0.22], [0.09], [0.03], [0.29], [0.16])
    for stat in ["k", "bb", "hr", "ab", "bip", "h"]:
        assert big[stat][0] == pytest.approx(2 * small[stat][0])
    assert big["woba"][0] == pytest.approx(small["woba"][0])


def test_a_line_with_no_plate_appearances_has_no_woba():
    assert np.isnan(ros_counting_line([0.0], [0.2], [0.1], [0.03], [0.3], [0.15])["woba"][0])


# ─── the league-average sanity check ─────────────────────────────

def test_the_league_average_line_comes_back_at_the_league_woba():
    line = ros_counting_line(
        [600.0], [LEAGUE["k_rate"]], [LEAGUE["bb_rate"]], [LEAGUE["hr_rate"]],
        [LEAGUE["babip"]], [LEAGUE["iso"]],
        hbp_rate=LEAGUE_HBP / LEAGUE_PA, sf_rate=LEAGUE_SF / LEAGUE_PA)
    assert line["woba"][0] == pytest.approx(LEAGUE_WOBA, abs=0.002)


def test_the_weights_are_the_published_ones():
    """A typo in a weight moves every wOBA on the site; pin them."""
    assert WOBA_WEIGHTS == {"bb": 0.690, "hbp": 0.722, "single": 0.883,
                            "double": 1.244, "triple": 1.569, "hr": 2.015}


def test_league_rates_are_read_off_the_partial_season():
    pa = make_pa_frame(pd.date_range("2026-04-01", periods=10))
    partial = partial_season(pa, AS_OF)
    rates = league_rates(partial)
    assert rates["k_rate"] == pytest.approx(0.25)     # one K in four PA
    assert rates["bb_rate"] == pytest.approx(0.25)
    assert rates["hr_rate"] == pytest.approx(0.25)


def test_league_rates_fall_back_when_there_is_no_season_yet():
    rates = league_rates(partial_season(make_pa_frame([]), AS_OF))
    assert rates["hbp_rate"] == LEAGUE_HBP_RATE
    assert rates["sf_rate"] == LEAGUE_SF_RATE


# ─── the as-of cutoff ────────────────────────────────────────────

def test_the_partial_season_stops_the_day_before_the_as_of_date():
    """A game on the morning of the projection has not been played yet."""
    dates = list(pd.date_range("2026-07-25", periods=8))    # through 2026-08-01
    partial = partial_season(make_pa_frame(dates, batters=(1,)), AS_OF)
    assert partial["last_game_date"].max() == pd.Timestamp("2026-07-31")
    # Seven days of four PA, not eight.
    assert int(partial["pa"].sum()) == 7 * 4


def test_same_day_plate_appearances_do_not_move_the_projection():
    before = list(pd.date_range("2026-07-01", periods=30))
    on_the_day = make_pa_frame([pd.Timestamp(AS_OF)], batters=(1,), pa_per_day=40)
    seasons = make_seasons(batters=(1,))
    playing_time = make_playing_time({1: 100.0})

    without = build_ros_projections(AS_OF, seasons, make_pa_frame(before, batters=(1,)),
                                    playing_time)
    with_today = build_ros_projections(
        AS_OF, seasons,
        pd.concat([make_pa_frame(before, batters=(1,)), on_the_day], ignore_index=True),
        playing_time)
    assert without["k_rate_marcel"][0] == pytest.approx(with_today["k_rate_marcel"][0])
    assert without["woba_ros"][0] == pytest.approx(with_today["woba_ros"][0])


def test_the_partial_season_is_flagged_so_the_baselines_treat_it_as_partial():
    partial = partial_season(make_pa_frame(pd.date_range("2026-04-01", periods=5)), AS_OF)
    assert partial["partial"].all()
    assert set(partial["season"]) == {SEASON}


# ─── who gets a projection ───────────────────────────────────────

def test_zero_projected_plate_appearances_is_excluded():
    """An injured hitter projects to zero PA — he has no line to show."""
    pa = make_pa_frame(pd.date_range("2026-04-01", periods=40), batters=(1, 2))
    out = build_ros_projections(AS_OF, make_seasons(), pa,
                                make_playing_time({1: 120.0, 2: 0.0}))
    assert list(out["batter"]) == [1]


def test_a_hitter_with_no_history_gets_the_league_rate_not_a_hole():
    """Marcel with zero trials is the league rate; that is a projection."""
    pa = make_pa_frame(pd.date_range("2026-04-01", periods=40), batters=(1,))
    out = build_ros_projections(AS_OF, make_seasons(batters=(1,)), pa,
                                make_playing_time({1: 100.0, 99: 40.0}))
    rookie = out[out["batter"] == 99].iloc[0]
    league = league_rates(partial_season(pa, AS_OF))
    assert rookie["k_rate_marcel"] == pytest.approx(league["k_rate"])
    assert np.isfinite(rookie["woba_ros"])
    # No completed season to look back on, so the preseason control says nothing.
    assert pd.isna(rookie["k_rate_marcel_preseason"])


def test_a_traded_hitter_appears_once_on_the_club_with_the_playing_time():
    pa = make_pa_frame(pd.date_range("2026-04-01", periods=40), batters=(1,))
    playing_time = pd.concat([make_playing_time({1: 20.0}, team_id=100),
                              make_playing_time({1: 95.0}, team_id=200)],
                             ignore_index=True)
    out = build_ros_projections(AS_OF, make_seasons(batters=(1,)), pa, playing_time)
    assert len(out) == 1
    assert int(out["team_id"][0]) == 200
    assert out["pa_ros"][0] == pytest.approx(95.0)


# ─── the frame the site is handed ────────────────────────────────

def _built(**kwargs):
    pa = make_pa_frame(pd.date_range("2026-04-01", periods=40))
    return build_ros_projections(AS_OF, make_seasons(), pa,
                                 make_playing_time({1: 120.0, 2: 90.0}), **kwargs)


def test_output_columns_are_exactly_the_contract():
    assert list(_built().columns) == OUTPUT_COLUMNS


def test_every_component_carries_all_three_arms():
    out = _built()
    for component in COMPONENT_ORDER:
        prefix = {"k_rate": "k", "bb_rate": "bb", "hr_rate": "hr",
                  "babip": "babip", "iso": "iso"}[component]
        for arm in ("marcel", "marcel_preseason", "bayes"):
            assert f"{prefix}_rate_{arm}" in out.columns


def test_the_counting_columns_match_the_rates_and_the_playing_time():
    out = _built().set_index("batter")
    for batter, row in out.iterrows():
        assert row["k_ros"] == pytest.approx(row["k_rate_marcel"] * row["pa_ros"])
        assert row["bb_ros"] == pytest.approx(row["bb_rate_marcel"] * row["pa_ros"])
        assert row["hr_ros"] == pytest.approx(row["hr_rate_marcel"] * row["pa_ros"])


def test_the_live_arm_is_marcel_with_the_partial_season_the_control_is_without():
    """The two Marcel columns must actually differ once 2026 says something."""
    pa = make_pa_frame(pd.date_range("2026-04-01", periods=60), batters=(1,))
    out = build_ros_projections(AS_OF, make_seasons(batters=(1,)), pa,
                                make_playing_time({1: 100.0}))
    row = out.iloc[0]
    # The synthetic 2026 is a 25% K rate against a 21.7% career rate, so the
    # live arm has to sit above the preseason control.
    assert row["k_rate_marcel"] > row["k_rate_marcel_preseason"]


def test_marcel_rates_reuse_the_harness_providers():
    """Same training frame, same numbers as calling the baseline directly."""
    from src.eval.backtest import COMPONENTS
    from src.eval.baselines import marcel_tuned
    from src.eval.intraseason import build_training_frame

    pa = make_pa_frame(pd.date_range("2026-04-01", periods=40), batters=(1,))
    seasons = make_seasons(batters=(1,))
    partial = partial_season(pa, AS_OF)
    ours = marcel_rates(seasons, partial).set_index("batter")
    theirs = marcel_tuned(build_training_frame(seasons, partial, SEASON),
                          COMPONENTS["k_rate"], SEASON).set_index("batter")
    assert ours.loc[1, "k_rate_marcel"] == pytest.approx(theirs.loc[1, "predicted"])


def test_the_live_engine_is_the_tuned_marcel():
    """The wiring guard.

    The gate rule (architecture.md section 3) put `marcel_tuned` in production
    on the strength of the holdout in docs/backtest-baselines.md. This asserts
    the module actually calls it — and, because `marcel_tuned` at stock
    constants *is* `marcel` bit for bit, that the frozen params file is the one
    being read. A revert to stock Marcel, or a params file quietly replaced by
    defaults, fails here rather than shipping silently.
    """
    from src.eval import baselines
    from src.eval.baselines import STOCK_PARAMS, load_marcel_params

    assert ros_module.LIVE_ENGINE == "marcel_tuned"
    assert ros_module.LIVE_PROVIDERS == {
        "marcel": baselines.marcel_tuned,
        "marcel_preseason": baselines.marcel_tuned_preseason,
    }
    fitted = load_marcel_params(strict=True)
    assert fitted != STOCK_PARAMS, "marcel_params.json is stock; nothing is tuned"
    # Both arms move together: the preseason column is the control that
    # isolates in-season information, so it has to be the same model.
    assert set(ros_module.LIVE_PROVIDERS) == set(ros_module.MARCEL_ARMS)


def test_the_live_arm_differs_from_stock_marcel_on_a_tuned_component():
    """Not just wired — wired to something that actually changes the number.

    K% is one of the two components the tuning moved (ballast 200 -> 100,
    recency 5/4/3 -> 1/0.4/0.2), so the live column must not equal what stock
    Marcel would have produced on the same training frame.
    """
    from src.eval.backtest import COMPONENTS
    from src.eval.baselines import marcel
    from src.eval.intraseason import build_training_frame

    pa = make_pa_frame(pd.date_range("2026-04-01", periods=60), batters=(1, 2))
    seasons = make_seasons()
    partial = partial_season(pa, AS_OF)
    ours = marcel_rates(seasons, partial).set_index("batter")
    stock = marcel(build_training_frame(seasons, partial, SEASON),
                   COMPONENTS["k_rate"], SEASON).set_index("batter")
    assert ours.loc[1, "k_rate_marcel"] != pytest.approx(stock.loc[1, "predicted"])


def test_the_bayes_column_comes_from_the_projection_files():
    frames = {"k_rate": pd.DataFrame({"batter": [1, 2], "projection_year": [2026, 2026],
                                      "projected_k_rate": [0.111, 0.222]})}
    out = _built(bayes_frames=frames).set_index("batter")
    assert out.loc[1, "k_rate_bayes"] == pytest.approx(0.111)
    assert pd.isna(out.loc[1, "bb_rate_bayes"])


def test_the_bayes_column_ignores_other_projection_years():
    frames = {"k_rate": pd.DataFrame({"batter": [1, 1], "projection_year": [2027, 2026],
                                      "projected_k_rate": [0.900, 0.150]})}
    out = _built(bayes_frames=frames).set_index("batter")
    assert out.loc[1, "k_rate_bayes"] == pytest.approx(0.150)


def test_names_and_team_abbrevs_are_joined_when_supplied():
    teams = pd.DataFrame({"team_id": [100], "abbrev": ["NYY"]})
    out = _built(names={1: "Real Person", 2: "Other Person"}, teams=teams)
    assert set(out["name"]) == {"Real Person", "Other Person"}
    assert set(out["team_abbrev"]) == {"NYY"}


def test_the_frame_is_sorted_by_projected_woba():
    out = _built()
    assert out["woba_ros"].is_monotonic_decreasing


def test_an_empty_training_frame_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        build_ros_projections(AS_OF, make_seasons().iloc[:0], make_pa_frame([]),
                              make_playing_time({1: 100.0}))
