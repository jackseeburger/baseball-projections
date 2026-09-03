"""The live pitcher projection: the gated half, the structural half, and the cutoff.

Two things have to hold here that do not hold for the hitter module, and both
are about honesty rather than accuracy:

  * the rate columns are the arm that cleared the serving gate, fed exactly the
    training frame the harness builds at a cutoff — so the model on the page is
    the model that was scored;
  * the workload columns are structural and say so, and no test here pretends
    otherwise. What is tested about them is that they are arithmetic on the
    pitcher's own usage and that they cannot see the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import pitchers as pitcher_eval
from src.projections import pitcher_ros as pr

SEASON = 2026
AS_OF = "2026-08-01"


def pa_rows(pitcher: int, date: str, n: int, k: int = 0, bb: int = 0,
            hr: int = 0, hits: int = 0, game_pk: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(n):
        is_k = int(i < k)
        is_bb = int(k <= i < k + bb)
        is_hr = int(k + bb <= i < k + bb + hr)
        is_hit = int(k + bb <= i < k + bb + hits)
        event = ("strikeout" if is_k else "walk" if is_bb
                 else "home_run" if is_hr else "single" if is_hit else "field_out")
        rows.append({
            "batter": 900000 + i, "pitcher": pitcher, "game_pk": game_pk,
            "game_date": date, "game_year": SEASON, "event": event,
            "is_k": is_k, "is_bb": is_bb, "is_hbp": 0, "is_hit": is_hit,
            "is_hr": is_hr, "is_single": int(is_hit and not is_hr),
            "is_double": 0, "is_triple": 0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def pa():
    """One starter, one reliever, and a July line that is after some cutoffs."""
    frames = []
    for g, date in enumerate(["2026-04-10", "2026-05-10", "2026-06-10",
                              "2026-07-10"]):
        frames.append(pa_rows(100, date, 24, k=7, bb=2, hr=1, hits=5, game_pk=g))
        frames.append(pa_rows(200, date, 4, k=2, bb=0, hr=0, hits=1,
                              game_pk=100 + g))
    # A start after the as-of date: the leakage tests key on this one.
    frames.append(pa_rows(100, "2026-08-15", 24, k=0, bb=12, hr=6, hits=12,
                          game_pk=99))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def seasons():
    rows = []
    for season in (2024, 2025):
        rows.append({"pitcher": 100, "season": season, "bf": 700, "k": 190,
                     "bb": 55, "hbp": 5, "hr": 22, "ab": 640, "h": 155,
                     "sf": 5, "age": 28.0 + season - 2025})
        rows.append({"pitcher": 200, "season": season, "bf": 260, "k": 80,
                     "bb": 25, "hbp": 2, "hr": 8, "ab": 230, "h": 55,
                     "sf": 2, "age": 26.0 + season - 2025})
    return pitcher_eval.normalize_pitcher_seasons(pd.DataFrame(rows))


TEAM_OF = {100: 147, 200: 147}
GAMES_REMAINING = {147: 53}
GAMES_PLAYED = {147: 108}
GAMES_RECENT = {147: 26}


def build(pa, seasons, as_of=AS_OF, **kwargs):
    return pr.build_pitcher_projections(
        as_of, seasons, pa, team_of=TEAM_OF,
        team_games_played=GAMES_PLAYED, team_games_recent=GAMES_RECENT,
        games_remaining=GAMES_REMAINING, season=SEASON, **kwargs)


# ─── the served components ───────────────────────────────────────

def test_only_the_components_that_cleared_the_gate_are_served():
    assert pr.SERVED_COMPONENTS == ("p_k_rate", "p_bb_rate", "p_hr_rate", "p_babip")
    # The walks-plus-hit-batsmen rate is station E's, and is scored in the
    # harness, but a column labelled BB% has to mean walks.
    assert "p_bbhbp_rate" not in pr.SERVED_COMPONENTS


def test_the_engine_is_the_arm_the_harness_scored():
    assert pr.LIVE_ENGINE == "marcel_pitcher_tuned"
    assert pr.LIVE_PROVIDERS["marcel"] is pitcher_eval.marcel_pitcher_tuned


def test_the_output_has_a_column_per_component_and_arm(pa, seasons):
    out = build(pa, seasons)
    for component in pr.SERVED_COMPONENTS:
        prefix = pr.COMPONENT_PREFIX[component]
        for arm in pr.ARMS:
            assert f"{prefix}_rate_{arm}" in out.columns
    assert set(out["pitcher"]) == {100, 200}


# ─── the cutoff: nothing after `as_of` can move the number ───────

def test_a_start_after_the_as_of_date_cannot_move_the_projection(pa, seasons):
    """The whole leakage claim, on the serving path rather than the harness's.

    The 08-15 line is a disaster; dropping it from the input entirely must
    leave every projected rate and every projected batter faced identical.
    """
    truncated = pa[pd.to_datetime(pa["game_date"]) < pd.Timestamp(AS_OF)]
    full = build(pa, seasons).set_index("pitcher")
    cut = build(truncated, seasons).set_index("pitcher")
    numeric = [c for c in full.columns if full[c].dtype.kind == "f"]
    assert list(full.index) == list(cut.index)
    for column in numeric:
        assert np.allclose(full[column].to_numpy(), cut[column].to_numpy(),
                           rtol=0, atol=0, equal_nan=True), column


def test_a_game_on_the_as_of_date_itself_is_still_the_future(seasons):
    """The cutoff is exclusive: the morning's projection cannot contain a game
    that has not finished."""
    base = pa_rows(100, "2026-04-10", 24, k=8, game_pk=1)
    same_day = pd.concat([base, pa_rows(100, AS_OF, 24, k=0, bb=20, game_pk=2)],
                         ignore_index=True)
    a = pr.partial_season(base, AS_OF, SEASON)
    b = pr.partial_season(same_day, AS_OF, SEASON)
    assert int(a["bf"].sum()) == int(b["bf"].sum()) == 24


def test_moving_the_cutoff_later_does_change_the_number(pa, seasons):
    """Otherwise the guard above would also pass on a model that ignores 2026."""
    early = build(pa, seasons, as_of="2026-05-01").set_index("pitcher")
    late = build(pa, seasons, as_of="2026-09-01").set_index("pitcher")
    # 100's August was a catastrophe; seeing it must drop his projected K rate.
    assert late.loc[100, "k_rate_marcel"] < early.loc[100, "k_rate_marcel"]


# ─── the structural workload ─────────────────────────────────────

def test_role_is_read_off_the_workload_not_a_depth_chart():
    assert list(pr.role_of([24.0, 4.0, pr.STARTER_MIN_BF])) == ["SP", "RP", "SP"]


def test_the_starter_is_a_starter_and_the_reliever_is_a_reliever(pa, seasons):
    out = build(pa, seasons).set_index("pitcher")
    assert out.loc[100, "role"] == "SP"
    assert out.loc[200, "role"] == "RP"
    assert out.loc[100, "bf_ros"] > out.loc[200, "bf_ros"]


def test_projected_work_scales_with_the_games_the_club_has_left(pa, seasons):
    few = pr.build_pitcher_projections(
        AS_OF, seasons, pa, team_of=TEAM_OF, team_games_played=GAMES_PLAYED,
        team_games_recent=GAMES_RECENT, games_remaining={147: 10}, season=SEASON)
    many = build(pa, seasons)
    assert (many.set_index("pitcher").loc[100, "bf_ros"]
            > 4 * few.set_index("pitcher").loc[100, "bf_ros"])


def test_an_injured_pitchers_workload_is_scaled_by_his_expected_return(pa, seasons):
    full = build(pa, seasons).set_index("pitcher")
    half = build(pa, seasons, active_fraction={100: 0.5}).set_index("pitcher")
    assert half.loc[100, "bf_ros"] == pytest.approx(0.5 * full.loc[100, "bf_ros"])
    assert half.loc[200, "bf_ros"] == pytest.approx(full.loc[200, "bf_ros"])
    out = build(pa, seasons, active_fraction={100: 0.0})
    assert 100 not in set(out["pitcher"]), "a zero workload is not a projection"


def test_a_pitcher_not_on_a_staff_is_not_projected(pa, seasons):
    out = pr.build_pitcher_projections(
        AS_OF, seasons, pa, team_of={200: 147}, team_games_played=GAMES_PLAYED,
        team_games_recent=GAMES_RECENT, games_remaining=GAMES_REMAINING,
        season=SEASON)
    assert set(out["pitcher"]) == {200}


def test_the_workload_method_is_labelled_structural():
    assert pr.BF_METHOD == "structural"
    assert "not gated" in pr.BF_METHOD_NOTE


# ─── rates x workload ────────────────────────────────────────────

def test_a_league_average_line_comes_back_at_the_league_run_rate():
    """The FIP is re-centred, exactly as station E's is, so the number is
    readable next to an ERA and the model cannot shift the run environment."""
    league = {"p_k_rate": 0.22, "p_bb_rate": 0.085, "p_hr_rate": 0.030,
              "p_babip": 0.290, "bf_per_ip": 4.3}
    line = pr.ros_pitching_line([100.0], [0.22], [0.085], [0.030], league,
                                lg_ra9=4.40)
    assert line.loc[0, "fip"] == pytest.approx(4.40)
    assert line.loc[0, "k"] == pytest.approx(22.0)
    assert line.loc[0, "ip"] == pytest.approx(100.0 / 4.3)


def test_a_better_pitcher_gets_a_lower_fip():
    league = {"p_k_rate": 0.22, "p_bb_rate": 0.085, "p_hr_rate": 0.030,
              "p_babip": 0.290, "bf_per_ip": 4.3}
    line = pr.ros_pitching_line([100.0, 100.0], [0.32, 0.14], [0.05, 0.12],
                                [0.02, 0.04], league)
    assert line.loc[0, "fip"] < line.loc[1, "fip"]


def test_the_frame_is_sorted_by_projected_fip(pa, seasons):
    out = build(pa, seasons)
    assert out["fip_ros"].is_monotonic_increasing
