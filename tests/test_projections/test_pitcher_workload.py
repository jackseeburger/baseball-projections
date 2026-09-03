"""Station B-pitchers: the model's arithmetic, and the leakage guard.

The leakage test is the one that matters. Every method takes the *whole*
season's appearance log and filters it on the cutoff itself, so a method that
forgets to filter still runs and quietly scores brilliantly. These tests hand
each method a season whose post-cutoff rows are absurd — one pitcher facing
nine thousand batters the day after the cutoff — and assert that not one
projection moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.projections import pitcher_ros
from src.projections import pitcher_workload as W

TEAMS = (100, 200)
SEASON_START = pd.Timestamp("2024-04-01")
CUTOFF = pd.Timestamp("2024-05-01")
SCORE_END = pd.Timestamp("2024-05-30")

# id -> (team, role, every-nth-day, batters faced, first day, last day)
ROSTER_PLAN = {
    1: (100, "SP", 5, 25, 0, 59),      # a healthy starter, every fifth day
    2: (100, "RP", 2, 4, 0, 59),       # a busy reliever
    3: (100, "SP", 5, 24, 0, 9),       # a starter who stops on April 10
    4: (100, "RP", 3, 5, 35, 59),      # a call-up, all of it after the cutoff
    5: (200, "SP", 5, 22, 0, 59),
    6: (200, "RP", 2, 4, 0, 59),
    7: (200, "RP", 4, 3, 0, 59),
    8: (200, "SP", 5, 20, 0, 59),
}
# 9 is on the 40-man and never pitches at all.
STATUS = {1: "A", 2: "A", 3: "D60", 4: "A", 5: "A", 6: "A", 7: "A", 8: "A",
          9: "A"}
TEAM_OF = {**{p: plan[0] for p, plan in ROSTER_PLAN.items()}, 9: 200}


def _dates():
    return [SEASON_START + pd.Timedelta(days=d) for d in range(60)]


def team_games() -> pd.DataFrame:
    rows = []
    for i, day in enumerate(_dates()):
        for team in TEAMS:
            rows.append({"team_id": team, "date": day, "game_pk": 1000 + i})
    return pd.DataFrame(rows)


def schedule() -> pd.DataFrame:
    return pd.DataFrame([{"date": day, "game_pk": 1000 + i,
                          "home_id": TEAMS[0], "away_id": TEAMS[1]}
                         for i, day in enumerate(_dates())])


def appearances() -> pd.DataFrame:
    rows = []
    for pitcher, (team, role, every, bf, first, last) in ROSTER_PLAN.items():
        for offset in range(first, last + 1, every):
            day = SEASON_START + pd.Timedelta(days=offset)
            rows.append({"pitcher": pitcher, "date": day, "game_pk": 1000 + offset,
                         "team": team, "bf": float(bf),
                         "outs": float(bf) * 0.7, "gs": 1 if role == "SP" else 0,
                         "k": 0.0, "bb": 0.0, "hr": 0.0, "pitches": 0.0})
    return pd.DataFrame(rows)


def roster() -> pd.DataFrame:
    return pd.DataFrame([{"pitcher": p, "team_id": TEAM_OF[p],
                          "cutoff": CUTOFF.date().isoformat(),
                          "status_code": STATUS[p]}
                         for p in sorted(STATUS)])


def inputs(log: pd.DataFrame | None = None, **kwargs) -> W.CutoffInputs:
    prior = pd.DataFrame({"pitcher": [1, 2, 5], "bf": [700.0, 250.0, 600.0],
                          "outs": [500.0, 180.0, 430.0]})
    base = dict(cutoff=CUTOFF, score_end=SCORE_END,
                appearances=appearances() if log is None else log,
                team_games=team_games(), schedule=schedule(), roster=roster(),
                prior_totals=prior,
                # Pitcher 3 has been on the 60-day list since April 10.
                active_fraction=pd.Series({3: 0.4}),
                spell_start=pd.Series({3: pd.Timestamp("2024-04-11")}))
    base.update(kwargs)
    return W.CutoffInputs(**base)


def poisoned() -> pd.DataFrame:
    """The same season with every post-cutoff appearance made absurd.

    A method that reads past the cutoff cannot survive this: one row alone
    would swamp any pitcher's real workload.
    """
    log = appearances().copy()
    after = pd.to_datetime(log["date"]) >= CUTOFF
    log.loc[after, ["bf", "outs"]] = 9999.0
    log.loc[after, "gs"] = 1
    # And a pitcher who does not exist before the cutoff at all.
    extra = pd.DataFrame([{"pitcher": 99, "date": CUTOFF + pd.Timedelta(days=1),
                           "game_pk": 1031, "team": 100, "bf": 9999.0,
                           "outs": 9999.0, "gs": 1, "k": 0.0, "bb": 0.0,
                           "hr": 0.0, "pitches": 0.0}])
    return pd.concat([log, extra], ignore_index=True)


# --- the leakage guard --------------------------------------------------

@pytest.mark.parametrize("method", W.METHODS)
@pytest.mark.parametrize("unit", ["bf", "outs"])
def test_no_method_reads_past_the_cutoff(method, unit):
    """Post-cutoff rows, however extreme, move no projection at all."""
    clean = appearances()
    honest = W.project(inputs(clean[pd.to_datetime(clean["date"]) < CUTOFF]),
                       method, unit=unit)
    leaky = W.project(inputs(poisoned()), method, unit=unit)
    pd.testing.assert_frame_equal(
        honest.sort_values("pitcher").reset_index(drop=True),
        leaky.sort_values("pitcher").reset_index(drop=True),
        check_exact=False, rtol=1e-12)


def test_the_poison_would_have_been_visible():
    """The guard is worth something only if the poisoned rows are huge."""
    leaky = W.project(inputs(poisoned()), "season_rate", unit="bf")
    unfiltered = poisoned().groupby("pitcher")["bf"].sum()
    assert unfiltered.max() > 40_000
    assert leaky["projected"].max() < 1_000


def test_a_spell_starting_after_the_cutoff_is_ignored():
    """A transaction dated tomorrow cannot re-date today's projection.

    `blend_il` reads an unavailable pitcher's usage as of the day he went out,
    which is the one place in the model where a window ends somewhere other
    than the cutoff. A bad spell date in the future would walk that window
    forward into games that have not been played.
    """
    poison = poisoned()
    at_cutoff = W.project(inputs(poison, spell_start=pd.Series(dtype="datetime64[ns]")),
                          "blend_il", unit="bf")
    future = W.project(inputs(poison, spell_start=pd.Series({3: SCORE_END})),
                       "blend_il", unit="bf")
    pd.testing.assert_frame_equal(
        at_cutoff.sort_values("pitcher").reset_index(drop=True),
        future.sort_values("pitcher").reset_index(drop=True),
        check_exact=False, rtol=1e-12)


def test_window_totals_upper_bound_is_strict():
    log = appearances()
    on_the_day = log[pd.to_datetime(log["date"]) == CUTOFF]
    assert len(on_the_day) > 0
    totals = W.window_totals(log, CUTOFF, "bf", None)
    before = log[pd.to_datetime(log["date"]) < CUTOFF]
    assert totals["bf"].sum() == pytest.approx(before["bf"].sum())


def test_realized_includes_both_ends():
    log = appearances()
    real = W.realized(log, CUTOFF, SCORE_END, "bf")
    window = log[(pd.to_datetime(log["date"]) >= CUTOFF)
                 & (pd.to_datetime(log["date"]) <= SCORE_END)]
    assert real["realized"].sum() == pytest.approx(window["bf"].sum())


# --- the arithmetic -----------------------------------------------------

def test_role_comes_from_starts_not_from_batters_faced():
    assert list(W.role_of([5, 0, 3], [5, 10, 10])) == ["SP", "RP", "RP"]
    # An opener: one batter faced, but he started.
    assert list(W.role_of([6], [6])) == ["SP"]


def test_horizon_weight_decreases_and_is_bounded():
    w = W.horizon_weight([10, 30, 60, 90, 150])
    assert np.all(np.diff(w) < 0)
    assert np.all((w > 0) & (w < 1))
    assert W.horizon_weight(30.0) == pytest.approx(W.BLEND_WEIGHT_SHORT, abs=1e-6)
    assert W.horizon_weight(90.0) == pytest.approx(W.BLEND_WEIGHT_LONG, abs=1e-6)


def test_zero_projects_nothing_and_covers_the_staff():
    proj = W.project(inputs(), "zero")
    assert len(proj) == len(STATUS)
    assert (proj["projected"] == 0).all()


def test_season_rate_is_workload_per_club_game_times_games_left():
    proj = W.project(inputs(), "season_rate", unit="bf").set_index("pitcher")
    log = appearances()
    before = log[pd.to_datetime(log["date"]) < CUTOFF]
    bf = before.loc[before["pitcher"] == 1, "bf"].sum()
    played = 30           # April 1 .. April 30 inclusive
    left = 30             # May 1 .. May 30 inclusive
    assert proj.loc[1, "projected"] == pytest.approx(bf / played * left)


def test_last_season_uses_only_the_prior_season():
    proj = W.project(inputs(), "last_season", unit="bf").set_index("pitcher")
    assert proj.loc[1, "projected"] == pytest.approx(700.0 * 30 / 162.0)
    # Nobody's current season enters, so a pitcher with no prior year is zero.
    assert proj.loc[7, "projected"] == 0.0


def test_structural_is_the_served_function():
    """The harness scores the code the site runs, not a copy of it."""
    got = W.project(inputs(), "structural", unit="bf").set_index("pitcher")
    partial = W._structural_partial(inputs(), CUTOFF, "bf", None)
    recent = W._structural_partial(inputs(), CUTOFF, "bf", W.RECENT_DAYS)
    played = W.club_games(team_games(), pd.Timestamp("1900-01-01"), CUTOFF)
    recent_played = W.club_games(
        team_games(), CUTOFF - pd.Timedelta(days=W.RECENT_DAYS), CUTOFF)
    want = pitcher_ros.projected_batters_faced(
        partial, recent, played, recent_played,
        W.games_remaining(schedule(), CUTOFF, SCORE_END),
        pd.Series(TEAM_OF), pd.Series({3: 0.4})).set_index("pitcher")
    for pitcher in want.index:
        assert got.loc[pitcher, "projected"] == pytest.approx(
            want.loc[pitcher, "bf_ros"])


def test_structural_on_outs_keeps_the_served_role_call():
    """The served constants are in batters faced; outs are put on that scale.

    Without the rescale a starter averaging 17 outs would fall under
    `STARTER_MIN_BF` (12 *batters*) and be projected as a reliever, and the
    per-appearance regression would point at 22 outs a start instead of 15.
    The harness would then be scoring a different model and calling it the
    served one.
    """
    on_bf = W.project(inputs(), "structural", unit="bf").set_index("pitcher")
    on_outs = W.project(inputs(), "structural", unit="outs").set_index("pitcher")
    assert list(on_outs.loc[[1, 5, 8], "role"]) == ["SP", "SP", "SP"]
    assert list(on_outs.loc[[2, 6, 7], "role"]) == ["RP", "RP", "RP"]
    # Every appearance in the fixture is 0.7 outs per batter faced, so the two
    # projections differ by exactly that ratio.
    ratio = on_outs["projected"] / on_bf["projected"].where(on_bf["projected"] > 0)
    assert ratio.dropna().to_numpy() == pytest.approx(0.7, abs=1e-9)


def test_the_gate_only_moves_the_unavailable():
    gated = W.project(inputs(), "structural", unit="bf").set_index("pitcher")
    ungated = W.project(inputs(), "structural_nogate", unit="bf").set_index("pitcher")
    moved = [p for p in gated.index
             if abs(gated.loc[p, "projected"] - ungated.loc[p, "projected"]) > 1e-9]
    assert moved == [3]
    assert gated.loc[3, "projected"] < ungated.loc[3, "projected"]


def test_blend_zeroes_the_unavailable_and_blend_il_does_not():
    plain = W.project(inputs(), "blend", unit="bf").set_index("pitcher")
    with_il = W.project(inputs(), "blend_il", unit="bf").set_index("pitcher")
    assert plain.loc[3, "projected"] == 0.0
    assert with_il.loc[3, "projected"] > 0.0
    # The healthy pitchers are untouched by the injured-list treatment.
    healthy = [p for p in plain.index if p != 3]
    assert (plain.loc[healthy, "projected"].to_numpy()
            == pytest.approx(with_il.loc[healthy, "projected"].to_numpy()))


def test_blend_il_reads_the_injured_pitcher_before_he_went_out():
    """The station B fix: an empty trailing window is a symptom, not a fact."""
    with_il = W.project(inputs(), "blend_il", unit="bf").set_index("pitcher")
    # Pitcher 3 last pitched on April 6 and would have an empty 30-day window
    # at the cutoff read plainly; weighed at his spell start he is a starter
    # taking a normal turn, then scaled by the 0.4 active fraction.
    healthy_starter = W.project(inputs(), "blend", unit="bf").set_index(
        "pitcher").loc[1, "projected"]
    assert 0.15 * healthy_starter < with_il.loc[3, "projected"] < 0.8 * healthy_starter


def test_share_normalization_hits_the_club_total():
    proj = W.project(inputs(), "blend_il_share", unit="bf")
    per_game = W._club_unit_per_game(inputs(), "bf")
    left = W.games_remaining(schedule(), CUTOFF, SCORE_END)
    for team, group in proj.groupby("team_id"):
        assert group["projected"].sum() == pytest.approx(
            per_game.loc[team] * left.loc[team])


def test_calibration_scales_by_role():
    plain = W.project(inputs(), "blend", unit="bf").set_index("pitcher")
    scaled = W.project(inputs(), "blend", unit="bf",
                       calibration={"SP": 0.5, "RP": 2.0}).set_index("pitcher")
    for pitcher, row in plain.iterrows():
        factor = 0.5 if scaled.loc[pitcher, "role"] == "SP" else 2.0
        assert scaled.loc[pitcher, "projected"] == pytest.approx(
            row["projected"] * factor)


def test_the_served_method_is_the_one_the_site_stamps():
    """The harness's production key and the document's stamp cannot drift.

    They are deliberately different strings — one names a row of the
    scoreboard, the other names the model — so a test has to hold them
    together. The gate in docs/pitcher-workload.md moves both or neither.
    """
    assert W.PRODUCTION_METHOD in W.METHODS
    assert W.PRODUCTION_METHOD == "structural"
    assert pitcher_ros.BF_METHOD == "recent_usage"


def test_the_attrition_fraction_is_a_survival_curve():
    assert W.attrition_fraction(0.0, 0.002) == pytest.approx(1.0)
    assert W.attrition_fraction(50.0, 0.0) == pytest.approx(1.0)
    f = W.attrition_fraction([10.0, 60.0, 130.0], 0.002)
    assert np.all(np.diff(f) < 0)
    assert np.all((f > 0) & (f < 1))
    # A starter at a three-month horizon keeps about seven-eighths of it.
    assert W.attrition_fraction(130.0, 0.002) == pytest.approx(0.881, abs=0.005)


def test_the_hazard_only_shrinks_and_shrinks_more_at_a_long_horizon():
    plain = W.project(inputs(), "structural", unit="bf").set_index("pitcher")
    shrunk = W.project(inputs(), "structural_hazard", unit="bf").set_index("pitcher")
    assert (shrunk["projected"] <= plain["projected"] + 1e-12).all()
    ratio = (shrunk["projected"] / plain["projected"].where(plain["projected"] > 0)).dropna()
    assert (ratio < 1.0).all()
    # 30 games left in the fixture, so the haircut is small.
    assert ratio.min() > 0.9


# --- scoring ------------------------------------------------------------

def _projection(values: dict) -> pd.DataFrame:
    return pd.DataFrame({"pitcher": list(values), "team_id": 100,
                         "role": "SP", "projected": list(values.values())})


def _actual(values: dict) -> pd.DataFrame:
    return pd.DataFrame({"pitcher": list(values), "realized": list(values.values())})


def test_paired_difference_is_negative_when_the_first_is_better():
    a = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    b = pd.Series([2.0, 3.0, 4.0], index=[1, 2, 3])
    d = W.paired_difference(a, b)
    assert d["mean"] == pytest.approx(-1.0)
    assert d["n"] == 3
    assert d["se"] == pytest.approx(0.0)


def test_score_projection_counts_the_whole_universe():
    proj = _projection({1: 100.0, 2: 0.0})
    actual = _actual({1: 80.0, 3: 50.0})
    out = W.score_projection(proj, actual, universe=[1, 2, 3])
    assert out["n"] == 3
    assert out["mae"] == pytest.approx((20.0 + 0.0 + 50.0) / 3)
    assert out["bias"] == pytest.approx((20.0 + 0.0 - 50.0) / 3)


def test_score_projection_splits_by_role():
    proj = _projection({1: 100.0, 2: 10.0})
    actual = _actual({1: 80.0, 2: 20.0})
    roles = pd.Series({1: "SP", 2: "RP"})
    out = W.score_projection(proj, actual, universe=[1, 2], roles=roles)
    assert out["sp_mae"] == pytest.approx(20.0)
    assert out["rp_mae"] == pytest.approx(10.0)


def test_top_n_capture_picks_the_right_arms():
    proj = pd.DataFrame({"pitcher": [1, 2, 3], "team_id": 100, "role": "SP",
                         "projected": [10.0, 5.0, 1.0]})
    actual = _actual({1: 100.0, 2: 50.0, 3: 10.0})
    assert W.top_n_capture(proj, actual, 2, [1, 2, 3]) == pytest.approx(
        150.0 / 160.0)
