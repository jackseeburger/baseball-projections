"""Station B unit tests — synthetic rosters, no network.

Two teams, nine regulars and a bench bat each, plus the three cases the model
exists to handle: an injured regular, a call-up with no history, and a game
played exactly on the cutoff that must not be visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projections.playing_time import (
    BLEND_MIDPOINT_GAMES,
    BLEND_SCALE_GAMES,
    DEFAULT_BENCH_SHARE,
    MAX_PA_SHARE,
    METHODS,
    cap_shares,
    horizon_weight,
    is_active,
    is_injured,
    project_playing_time,
    realized_pa,
    score_projection,
    team_pa_per_game,
    walk_forward_scores,
    window_pa,
    window_pa_by_team,
)

CUTOFF = "2026-08-01"
TEAMS = (100, 200)
# batter ids: 1xx on team 100, 2xx on team 200.
REGULARS = {100: list(range(101, 110)), 200: list(range(201, 210))}
BENCH = {100: 110, 200: 210}
INJURED = 111          # a team-100 regular who lands on the 10-day IL
CALLUP = 112           # a team-100 hitter with zero prior plate appearances
OPTIONED = 113         # a team-100 hitter sent to the minors


def _game_dates(n_days: int, end: str = CUTOFF) -> list[pd.Timestamp]:
    """`n_days` consecutive dates ending the day *before* the cutoff."""
    last = pd.Timestamp(end) - pd.Timedelta(days=1)
    return list(pd.date_range(end=last, periods=n_days, freq="D"))


def make_game_logs(days: int = 90) -> pd.DataFrame:
    rows = []
    for date in _game_dates(days):
        for team, ids in REGULARS.items():
            for b in ids:
                rows.append({"batter": b, "team_id": team,
                             "date": date.date().isoformat(), "pa": 4})
            rows.append({"batter": BENCH[team], "team_id": team,
                         "date": date.date().isoformat(), "pa": 1})
        # The injured regular played every day, and keeps playing right up to
        # the cutoff, so only the roster status can zero him out.
        rows.append({"batter": INJURED, "team_id": 100,
                     "date": date.date().isoformat(), "pa": 4})
        rows.append({"batter": OPTIONED, "team_id": 100,
                     "date": date.date().isoformat(), "pa": 2})
    return pd.DataFrame(rows)


def make_team_logs(days: int = 90) -> pd.DataFrame:
    logs = make_game_logs(days)
    return (logs.groupby(["team_id", "date"], as_index=False)["pa"].sum())


def make_roster() -> pd.DataFrame:
    rows = []
    for team, ids in REGULARS.items():
        for b in ids:
            rows.append({"batter": b, "team_id": team, "status_code": "A"})
        rows.append({"batter": BENCH[team], "team_id": team, "status_code": "A"})
    rows += [
        {"batter": INJURED, "team_id": 100, "status_code": "D10"},
        {"batter": CALLUP, "team_id": 100, "status_code": "A"},
        {"batter": OPTIONED, "team_id": 100, "status_code": "RM"},
    ]
    return pd.DataFrame(rows)


def make_remaining(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({"team_id": list(TEAMS), "games_remaining": [n, n]})


@pytest.fixture
def frames():
    return make_roster(), make_game_logs(), make_team_logs(), make_remaining()


def project(frames, method="last_30", cutoff=CUTOFF):
    roster, logs, team_logs, remaining = frames
    return project_playing_time(roster, logs, remaining, cutoff,
                                team_logs=team_logs, method=method)


# --- status helpers ---

def test_status_helpers():
    assert is_active("A") and not is_active("D10")
    assert all(is_injured(c) for c in ("D7", "D10", "D15", "D60"))
    assert not is_injured("A") and not is_injured("RM")


# --- the shares ---

@pytest.mark.parametrize("method", METHODS)
def test_shares_sum_to_one_per_team(frames, method):
    proj = project(frames, method=method)
    totals = proj.groupby("team_id")["pa_share"].sum()
    assert set(totals.index) == set(TEAMS)
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-12)


def test_injured_and_optioned_project_to_zero(frames):
    proj = project(frames).set_index("batter")
    # The IL regular has more recent PA than anyone; only the status zeroes him.
    assert proj.loc[INJURED, "pa_share"] == 0.0
    assert proj.loc[INJURED, "projected_pa_ros"] == 0.0
    assert proj.loc[OPTIONED, "projected_pa_ros"] == 0.0
    assert proj.loc[101, "projected_pa_ros"] > 0


def test_season_share_baseline_does_not_zero_the_injured(frames):
    """The baseline is supposed to be worse in exactly this way."""
    proj = project(frames, method="season_share").set_index("batter")
    assert proj.loc[INJURED, "projected_pa_ros"] > 0


def test_no_history_hitter_gets_the_bench_default(frames):
    roster, logs, _, _ = frames
    proj = project(frames).set_index("batter")
    share = proj.loc[CALLUP, "pa_share"]
    assert 0 < share < proj.loc[101, "pa_share"]
    # The default is DEFAULT_BENCH_SHARE of the team's 30-day window total,
    # normalized against the eligible weights.
    team = roster[roster["team_id"] == 100]
    window = window_pa(logs, CUTOFF, 30)
    window = window[window["batter"].isin(team["batter"])]
    eligible = window[window["batter"].isin(
        team.loc[team["status_code"] == "A", "batter"])]
    raw = DEFAULT_BENCH_SHARE * float(window["pa"].sum())
    expected = raw / (float(eligible["pa"].sum()) + raw)
    assert share == pytest.approx(expected, rel=1e-9)


# --- the horizon blend ---

# A fixture where the two windows genuinely disagree: 101 was a regular for the
# first two months and has been benched for the last month, 102 the reverse.
# Every other regular is steady, so the season and 30-day shares differ for
# exactly two hitters and the blend has something to interpolate between.
FADING, RISING = 101, 102


def make_split_game_logs(days: int = 90) -> pd.DataFrame:
    logs = make_game_logs(days)
    recent = pd.to_datetime(logs["date"]) >= pd.Timestamp(CUTOFF) - pd.Timedelta(days=30)
    logs.loc[(logs["batter"] == FADING) & recent, "pa"] = 1
    logs.loc[(logs["batter"] == RISING) & ~recent, "pa"] = 1
    return logs


@pytest.fixture
def split_frames():
    logs = make_split_game_logs()
    team_logs = logs.groupby(["team_id", "date"], as_index=False)["pa"].sum()
    return make_roster(), logs, team_logs, make_remaining()


def _blend(frames, weight=None, games=30):
    roster, logs, team_logs, _ = frames
    remaining = make_remaining(games)
    return project_playing_time(roster, logs, remaining, CUTOFF, team_logs=team_logs,
                                method="blend", blend_weight=weight)


def test_horizon_weight_is_monotone_and_bounded():
    horizons = np.arange(0, 200, 1.0)
    w = horizon_weight(horizons)
    assert np.all(np.diff(w) < 0)                     # strictly decreasing in h
    # Saturates at both ends, a few scale lengths either side of the midpoint.
    assert horizon_weight(BLEND_MIDPOINT_GAMES - 5 * BLEND_SCALE_GAMES) > 0.99
    assert horizon_weight(BLEND_MIDPOINT_GAMES + 5 * BLEND_SCALE_GAMES) < 0.01
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert horizon_weight(BLEND_MIDPOINT_GAMES) == pytest.approx(0.5)
    assert horizon_weight(BLEND_MIDPOINT_GAMES + BLEND_SCALE_GAMES) < 0.5
    with pytest.raises(ValueError):
        horizon_weight(30, scale=0.0)


def test_blend_at_w_one_reproduces_last_30(split_frames):
    """The short-horizon limit is exactly the old model."""
    roster, logs, team_logs, remaining = split_frames
    last_30 = project_playing_time(roster, logs, remaining, CUTOFF,
                                   team_logs=team_logs, method="last_30")
    blended = _blend(split_frames, weight=1.0)
    np.testing.assert_allclose(blended["pa_share"].to_numpy(),
                               last_30["pa_share"].to_numpy(), atol=1e-12)


def test_blend_at_w_zero_is_the_season_share(split_frames):
    """The long-horizon limit is the season-to-date share of the active roster.

    Not the `season_share` *baseline*, which deliberately keeps no roster
    filter at all — the blend's own long half runs through the model's
    plumbing (IL zeroed, bench default), so what it reproduces is the season
    share renormalized over the hitters who can actually bat.
    """
    roster, logs, _, _ = split_frames
    shares = _blend(split_frames, weight=0.0).set_index("batter")["pa_share"]

    season = window_pa(logs, CUTOFF, None).set_index("batter")["pa"]
    team = roster[roster["team_id"] == 100]
    default = DEFAULT_BENCH_SHARE * float(season.reindex(team["batter"]).fillna(0.0).sum())
    active = team.loc[team["status_code"] == "A", "batter"]
    weights = season.reindex(active).fillna(default)
    expected = weights / weights.sum()
    assert expected.max() < MAX_PA_SHARE          # the cap must not be binding
    np.testing.assert_allclose(shares.reindex(active).to_numpy(),
                               expected.to_numpy(), atol=1e-12)


def test_blend_moves_monotonically_between_the_two_windows(split_frames):
    """Every hitter's share slides from his 30-day share to his season share."""
    ends = {w: _blend(split_frames, weight=w).set_index("batter")["pa_share"]
            for w in (0.0, 1.0)}
    grid = [_blend(split_frames, weight=w).set_index("batter")["pa_share"]
            for w in np.linspace(0.0, 1.0, 11)]
    moved = ends[1.0] - ends[0.0]
    assert abs(moved.loc[RISING]) > 1e-3 and abs(moved.loc[FADING]) > 1e-3
    for batter in ends[0.0].index:
        path = np.array([g.loc[batter] for g in grid])
        step = np.diff(path) * np.sign(moved.loc[batter] or 1.0)
        assert np.all(step >= -1e-12), f"{batter} is not monotone in w"


def test_blend_weight_follows_the_horizon(split_frames):
    """With no override, a longer horizon leans further on the season window.

    Monotone in the horizon, not merely different: every hitter's share slides
    the same direction as the horizon grows, toward his season share. (It does
    not *reach* it — the fitted weight only falls from about .84 to about .67
    across every horizon a projection is ever asked for.)
    """
    season = _blend(split_frames, weight=0.0).set_index("batter")["pa_share"]
    recent = _blend(split_frames, weight=1.0).set_index("batter")["pa_share"]
    horizons = [5, 15, 30, 60, 100, 162]
    shares = [_blend(split_frames, games=g).set_index("batter")["pa_share"]
              for g in horizons]
    toward_season = np.sign((season - recent).loc[[RISING, FADING]])
    assert set(toward_season) == {-1.0, 1.0}     # they move opposite ways
    for batter in (RISING, FADING):
        path = np.array([sh.loc[batter] for sh in shares])
        step = np.diff(path) * toward_season.loc[batter]
        assert np.all(step > 0), f"{batter} does not slide toward his season share"
        assert min(recent.loc[batter], season.loc[batter]) <= path.min()
        assert path.max() <= max(recent.loc[batter], season.loc[batter])


@pytest.mark.parametrize("games", [5, 20, 45, 80, 150])
def test_blend_shares_sum_to_one_and_zero_the_unavailable(split_frames, games):
    proj = _blend(split_frames, games=games)
    totals = proj.groupby("team_id")["pa_share"].sum()
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-12)
    indexed = proj.set_index("batter")
    assert indexed.loc[INJURED, "pa_share"] == 0.0
    assert indexed.loc[OPTIONED, "pa_share"] == 0.0
    assert indexed.loc[CALLUP, "pa_share"] > 0.0
    assert proj["pa_share"].max() <= MAX_PA_SHARE + 1e-12


@pytest.mark.parametrize("games", [10, 60])
def test_blend_respects_the_lineup_slot_cap(games):
    """A club with one hitter taking a third of its PA still gets capped."""
    ids = list(range(301, 311))
    rows = [{"batter": b, "team_id": 300, "status_code": "A"} for b in ids]
    roster = pd.DataFrame(rows)
    logs = pd.DataFrame([
        {"batter": b, "team_id": 300, "date": d.date().isoformat(),
         "pa": 8 if b == 301 else 2}
        for d in _game_dates(90) for b in ids
    ])
    team_logs = logs.groupby(["team_id", "date"], as_index=False)["pa"].sum()
    remaining = pd.DataFrame({"team_id": [300], "games_remaining": [games]})
    proj = project_playing_time(roster, logs, remaining, CUTOFF,
                                team_logs=team_logs, method="blend")
    # Uncapped he would be 8/26 = .308 of his club's plate appearances.
    assert proj["pa_share"].max() == pytest.approx(MAX_PA_SHARE)
    assert proj["pa_share"].sum() == pytest.approx(1.0)


def test_uniform_baseline_is_flat_over_the_active_roster(frames):
    proj = project(frames, method="uniform").set_index("batter")
    active = [*REGULARS[100], BENCH[100], CALLUP]
    shares = proj.loc[active, "pa_share"]
    np.testing.assert_allclose(shares.to_numpy(), 1.0 / len(active))
    assert proj.loc[INJURED, "pa_share"] == 0.0


# --- leakage ---

def test_by_team_splits_a_traded_hitter_between_his_clubs():
    """`window_pa_by_team` is `window_pa` keyed on the club he batted for.

    Station B pools a traded hitter's plate appearances because its roster
    frame already says which club he is on now. Station C
    (`src.sim.run_environment`) has no roster frame — it reads membership out
    of the appearances — so it needs the split.
    """
    logs = pd.DataFrame([
        {"batter": 1, "team_id": 100, "date": "2026-07-10", "pa": 40},
        {"batter": 1, "team_id": 200, "date": "2026-07-25", "pa": 20},
        {"batter": 2, "team_id": 100, "date": "2026-07-25", "pa": 30},
    ])
    pooled = window_pa(logs, CUTOFF, 30).set_index("batter")["pa"].to_dict()
    split = window_pa_by_team(logs, CUTOFF, 30)
    assert pooled == {1: 60, 2: 30}
    assert {(int(r.team_id), int(r.batter)): int(r.pa)
            for r in split.itertuples(index=False)} == {(100, 1): 40, (200, 1): 20,
                                                        (100, 2): 30}


def test_by_team_respects_the_window_and_the_cutoff():
    logs = pd.DataFrame([
        {"batter": 1, "team_id": 100, "date": "2026-05-01", "pa": 400},
        {"batter": 2, "team_id": 100, "date": "2026-07-31", "pa": 4},
        {"batter": 3, "team_id": 100, "date": CUTOFF, "pa": 400},
    ])
    assert window_pa_by_team(logs, CUTOFF, 30)["batter"].tolist() == [2]
    # window_days=None is the season to date, still strictly before the cutoff.
    assert sorted(window_pa_by_team(logs, CUTOFF, None)["batter"]) == [1, 2]


def test_by_team_drops_appearances_with_no_club_and_survives_an_empty_frame():
    logs = pd.DataFrame([{"batter": 1, "team_id": None, "date": "2026-07-31", "pa": 4},
                         {"batter": 2, "team_id": 100, "date": "2026-07-31", "pa": 4}])
    assert window_pa_by_team(logs, CUTOFF, 30)["batter"].tolist() == [2]
    empty = window_pa_by_team(pd.DataFrame(columns=["batter", "team_id", "date", "pa"]),
                              CUTOFF, 30)
    assert empty.empty and list(empty.columns) == ["team_id", "batter", "pa"]


def test_trailing_window_excludes_the_cutoff_day():
    """A game *on* the cutoff has not happened yet and must not be seen."""
    logs = pd.DataFrame([
        {"batter": 1, "team_id": 100, "date": "2026-07-31", "pa": 4},
        {"batter": 2, "team_id": 100, "date": "2026-08-01", "pa": 40},
        {"batter": 2, "team_id": 100, "date": "2026-08-02", "pa": 40},
    ])
    seen = window_pa(logs, CUTOFF, 30)
    assert seen.set_index("batter")["pa"].to_dict() == {1: 4}


def test_projection_ignores_games_on_and_after_the_cutoff(frames):
    roster, logs, team_logs, remaining = frames
    future = pd.DataFrame([
        {"batter": 101, "team_id": 100, "date": CUTOFF, "pa": 500},
        {"batter": 102, "team_id": 100, "date": "2026-08-15", "pa": 500},
    ])
    base = project(frames)
    leaked = project_playing_time(roster, pd.concat([logs, future], ignore_index=True),
                                  remaining, CUTOFF, team_logs=team_logs)
    pd.testing.assert_frame_equal(base, leaked)


def test_team_pa_per_game_ignores_the_cutoff_day():
    team_logs = pd.DataFrame([
        {"team_id": 100, "date": "2026-07-30", "pa": 40},
        {"team_id": 100, "date": "2026-07-31", "pa": 30},
        {"team_id": 100, "date": CUTOFF, "pa": 999},
    ])
    ppg = team_pa_per_game(team_logs, CUTOFF).set_index("team_id")
    assert ppg.loc[100, "team_games"] == 2
    assert ppg.loc[100, "pa_per_game"] == pytest.approx(35.0)


def test_realized_pa_covers_the_cutoff_day_inclusive():
    """The projection is responsible for the cutoff day onward — the split
    between what it may see and what it is scored on has no gap and no overlap."""
    logs = pd.DataFrame([
        {"batter": 1, "date": "2026-07-31", "pa": 4},
        {"batter": 1, "date": CUTOFF, "pa": 5},
        {"batter": 1, "date": "2026-09-02", "pa": 3},
        {"batter": 1, "date": "2026-09-03", "pa": 9},
    ])
    got = realized_pa(logs, CUTOFF, "2026-09-02").set_index("batter")["realized_pa"]
    assert got.loc[1] == 8


# --- the projected counts ---

def test_projected_pa_totals_match_team_pa_per_game_times_games(frames):
    roster, logs, team_logs, remaining = frames
    proj = project(frames)
    ppg = team_pa_per_game(team_logs, CUTOFF).set_index("team_id")["pa_per_game"]
    totals = proj.groupby("team_id")["projected_pa_ros"].sum()
    for team in TEAMS:
        assert totals[team] == pytest.approx(30 * ppg[team])


def test_missing_team_in_schedule_projects_zero_games(frames):
    roster, logs, team_logs, _ = frames
    remaining = pd.DataFrame({"team_id": [100], "games_remaining": [30]})
    proj = project_playing_time(roster, logs, remaining, CUTOFF, team_logs=team_logs)
    team200 = proj[proj["team_id"] == 200]
    assert (team200["games_remaining"] == 0).all()
    assert (team200["projected_pa_ros"] == 0).all()


# --- the lineup-slot ceiling ---

def test_cap_shares_water_fills_the_excess():
    shares = pd.Series([0.5, 0.2, 0.2, 0.1])
    teams = pd.Series([1, 1, 1, 1])
    out = cap_shares(shares, teams, cap=0.3)
    assert out.max() == pytest.approx(0.3)
    assert out.sum() == pytest.approx(1.0)
    # Redistribution is proportional, so the ordering below the cap survives.
    assert out[1] == pytest.approx(out[2])
    assert out[2] > out[3]


def test_cap_shares_is_a_no_op_below_the_cap():
    shares = pd.Series([0.4, 0.35, 0.25])
    teams = pd.Series([1, 1, 1])
    pd.testing.assert_series_equal(cap_shares(shares, teams, cap=0.5), shares)


def test_no_hitter_exceeds_one_lineup_slot(frames):
    """A club that just lost three regulars must not dump their plate
    appearances on whoever is left — the survivors stop at one lineup slot."""
    roster, logs, team_logs, remaining = frames
    hurt = roster["batter"].isin(REGULARS[100][:3])
    roster = roster.assign(status_code=roster["status_code"].where(~hurt, "D10"))
    proj = project_playing_time(roster, logs, remaining, CUTOFF, team_logs=team_logs)
    team100 = proj[proj["team_id"] == 100]
    assert team100["pa_share"].max() <= MAX_PA_SHARE + 1e-12
    assert team100["pa_share"].sum() == pytest.approx(1.0)


def test_cap_relaxes_when_a_team_has_too_few_hitters_to_satisfy_it(frames):
    """Fewer than eight eligible hitters cannot each stay under 1/8 and still
    account for the team's plate appearances; the shares must still sum to 1."""
    roster, logs, team_logs, remaining = frames
    keep = roster["batter"].isin([101, 102, BENCH[100]]) | (roster["team_id"] == 200)
    proj = project_playing_time(roster[keep].reset_index(drop=True), logs,
                                remaining, CUTOFF, team_logs=team_logs)
    team100 = proj[proj["team_id"] == 100]
    assert team100["pa_share"].sum() == pytest.approx(1.0)
    assert team100["pa_share"].max() == pytest.approx(1 / 3)


def test_baselines_are_not_capped(frames):
    """The cap is part of the model, not free help for the baseline."""
    roster, logs, team_logs, remaining = frames
    roster = roster[roster["batter"].isin([101, 102, 110])].reset_index(drop=True)
    proj = project_playing_time(roster, logs, remaining, CUTOFF,
                                team_logs=team_logs, method="uniform")
    assert proj["pa_share"].max() == pytest.approx(1 / 3)


def test_unknown_method_raises(frames):
    with pytest.raises(ValueError, match="unknown method"):
        project(frames, method="vibes")


# --- scoring ---

def test_perfect_projection_scores_zero_error():
    realized = pd.DataFrame({"batter": [1, 2, 3], "realized_pa": [120.0, 60.0, 0.0]})
    proj = pd.DataFrame({"batter": [1, 2, 3], "team_id": [100, 100, 100],
                         "projected_pa_ros": [120.0, 60.0, 0.0]})
    out = score_projection(proj, realized)
    assert out["mae"] == 0.0
    assert out["rmse"] == 0.0
    assert out["weighted_mae"] == 0.0
    assert out["weighted_rmse"] == 0.0
    assert out["n_hitters"] == 3


def test_scoring_universe_is_the_union_and_missing_sides_are_zero():
    realized = pd.DataFrame({"batter": [1, 9], "realized_pa": [100.0, 50.0]})
    proj = pd.DataFrame({"batter": [1, 2], "team_id": [100, 100],
                         "projected_pa_ros": [100.0, 30.0]})
    out = score_projection(proj, realized)
    # batter 9 (unprojected call-up) and batter 2 (projected, never played)
    # both count as full misses.
    assert out["n_hitters"] == 3
    assert out["mae"] == pytest.approx((0 + 30 + 50) / 3)


def test_shared_universe_scores_every_method_on_the_same_players():
    realized = pd.DataFrame({"batter": [1, 2], "realized_pa": [100.0, 50.0]})
    proj = pd.DataFrame({"batter": [1], "team_id": [100], "projected_pa_ros": [100.0]})
    out = score_projection(proj, realized, universe=[1, 2, 3])
    assert out["n_hitters"] == 3
    assert out["mae"] == pytest.approx(50 / 3)


def test_top9_capture_rewards_picking_the_right_nine():
    """Ten hitters, nine of whom take all the plate appearances."""
    ids = list(range(1, 11))
    realized = pd.DataFrame({"batter": ids,
                             "realized_pa": [100.0] * 9 + [0.0]})
    good = pd.DataFrame({"batter": ids, "team_id": [100] * 10,
                         "projected_pa_ros": [100.0] * 9 + [0.0]})
    bad = pd.DataFrame({"batter": ids, "team_id": [100] * 10,
                        "projected_pa_ros": [0.0] + [50.0] * 8 + [900.0]})
    assert score_projection(good, realized)["top9_capture"] == pytest.approx(1.0)
    assert score_projection(bad, realized)["top9_capture"] == pytest.approx(8 / 9)


def test_walk_forward_scores_every_method_at_every_cutoff(frames):
    roster, logs, team_logs, remaining = frames
    # Give the scoring window something to score against.
    post = pd.DataFrame([
        {"batter": b, "team_id": t, "date": d.date().isoformat(), "pa": 4}
        for t, ids in REGULARS.items() for b in ids
        for d in pd.date_range("2026-08-01", "2026-08-10", freq="D")
    ])
    all_logs = pd.concat([logs, post], ignore_index=True)
    table = walk_forward_scores({"2026-07-01": roster, CUTOFF: roster},
                                all_logs, team_logs,
                                {"2026-07-01": remaining, CUTOFF: remaining},
                                "2026-08-10")
    assert len(table) == 2 * len(METHODS)
    assert set(table["method"]) == set(METHODS)
    # A shared universe means every method at a cutoff sees the same n.
    assert table.groupby("cutoff")["n_hitters"].nunique().eq(1).all()
    assert table["mae"].notna().all()
