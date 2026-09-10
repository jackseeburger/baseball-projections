"""The live pitcher projection: the gated half, the structural half, and the cutoff.

Two things have to hold here that do not hold for the hitter module, and both
are about honesty rather than accuracy:

  * the rate columns are the arm that cleared the serving gate, fed exactly the
    training frame the harness builds at a cutoff — so the model on the page is
    the model that was scored;
  * the workload columns are the scored `recent_usage` model, and no test here
    pretends
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
    # Per component since BAS-79, and only where the *additive* arm cleared
    # the serving gate docs/pitching-stuff.md pre-registered: BB/BF (t -6.13)
    # and HR/BF (t -3.93). K/BF is withheld at t -2.38 against a bar of 2.5,
    # and BABIP was never scored — stuff has no mechanism for balls in play.
    # BAS-88 measured a third arm for BB/BF and withheld it too; see
    # `test_the_command_engine_exists_and_is_not_served` below.
    assert pr.LIVE_ENGINE == {
        "p_k_rate": "marcel_pitcher_tuned",
        "p_bb_rate": "stuff_additive",
        "p_hr_rate": "stuff_additive",
        "p_babip": "marcel_pitcher_tuned",
    }
    assert set(pr.LIVE_ENGINE) == set(pr.SERVED_COMPONENTS)
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


def test_the_workload_method_is_labelled_with_a_scored_name():
    """It carried `structural` while it was unscored; it is scored now.

    The stamp is what the site prints, so it has to change with the status of
    the model rather than trailing it — see docs/pitcher-workload.md.
    """
    assert pr.BF_METHOD == "recent_usage"
    assert "not gated" not in pr.BF_METHOD_NOTE
    assert "docs/pitcher-workload.md" in pr.BF_METHOD_NOTE


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


# --- the stuff engine (BAS-79) ------------------------------------------

def test_the_stuff_cutoff_is_the_month_boundary_not_the_as_of_date():
    """A build made mid-month reads the last month boundary, not a partial
    month — so a build on Sept 9 and one on Sept 30 read the same, Sept 1,
    cutoff (features through Aug 31)."""
    assert pr.stuff_cutoff("2026-09-09") == pd.Timestamp("2026-09-01")
    assert pr.stuff_cutoff("2026-09-30") == pd.Timestamp("2026-09-01")
    assert pr.stuff_cutoff("2026-09-01") == pd.Timestamp("2026-09-01")
    assert pr.stuff_features_through("2026-09-09") == pd.Timestamp("2026-08-31")


def stuff_bucket(pitcher, season, month, pitches, whiff_rate, velo):
    """One monthly stuff bucket, every count consistent with the rates."""
    from src.data.pitching_stuff import COUNT_COLUMNS

    row = {"pitcher": pitcher, "season": season, "month": month}
    row.update({c: 0.0 for c in COUNT_COLUMNS})
    swings = pitches * 0.45
    row["pitches"] = float(pitches)
    row["swings"] = swings
    row["p_whiff_sum"] = swings * whiff_rate
    row["p_csw_sum"] = pitches * (whiff_rate * 0.45 + 0.17)
    row["sum_velo"] = pitches * velo
    row["fb_pitches"] = pitches * 0.5
    row["fb_swings"] = swings * 0.5
    row["fb_p_whiff_sum"] = swings * 0.5 * whiff_rate
    row["nfb_pitches"] = pitches * 0.5
    row["nfb_swings"] = swings * 0.5
    row["nfb_p_whiff_sum"] = swings * 0.5 * whiff_rate
    return row


def test_the_engine_never_reads_a_bucket_from_the_as_of_month_or_later():
    """The leakage guard, at the level this ticket wires: an as-of date inside
    September must never see a September stuff bucket, even though the monthly
    artifact has one and the as-of date is well past the 1st.

    A September bucket of ten thousand unmissable pitches would move every
    covariate enormously if it leaked in; the features computed at the
    engine's own cutoff must be bit-for-bit what they would be if that row
    were never in the table at all.
    """
    from src.eval.stuff import features_at_cutoff

    as_of = "2026-09-09"
    cutoff = pr.stuff_cutoff(as_of)
    assert cutoff == pd.Timestamp("2026-09-01")

    normal = [stuff_bucket(1, 2026, m, 400, 0.24, 93.0) for m in (5, 6, 7, 8)]
    normal += [stuff_bucket(2, 2026, m, 400, 0.20, 91.0) for m in (5, 6, 7, 8)]
    clean_rows = pd.DataFrame(normal)
    # The same pitchers, plus an extreme September bucket for one of them.
    with_september = pd.DataFrame(
        normal + [stuff_bucket(1, 2026, 9, 10000, 0.95, 104.0)])

    clean = features_at_cutoff(clean_rows, cutoff, 2026)
    leaky = features_at_cutoff(with_september, cutoff, 2026)
    pd.testing.assert_frame_equal(
        clean.sort_values("player").reset_index(drop=True),
        leaky.sort_values("player").reset_index(drop=True))


def test_a_cutoff_that_is_not_a_month_boundary_is_refused_not_rounded():
    """Rounding a cutoff forward is leakage, so the window sum refuses one."""
    from src.eval.stuff import features_at_cutoff

    rows = pd.DataFrame([stuff_bucket(1, 2026, m, 400, 0.24, 93.0)
                         for m in (5, 6, 7, 8)])
    with pytest.raises(ValueError):
        features_at_cutoff(rows, pd.Timestamp("2026-09-09"), 2026)


def test_engine_providers_fall_back_to_marcel_without_stuff_inputs():
    """No monthly frame, no pa_dir, no as-of -> every component falls back to
    the tuned pitcher Marcel rather than raising, which is what the module
    docstring and the pre-registration's fourth serving prediction promise."""
    providers, used = pr.engine_providers()
    assert set(used.values()) == {pr.MARCEL_ENGINE}
    assert set(used) == set(pr.SERVED_COMPONENTS)
    assert all(f is pitcher_eval.marcel_pitcher_tuned
               for f in providers.values())


def test_the_projection_records_the_engine_that_actually_ran(pa, seasons):
    """Provenance rides on `.attrs`, per component, and says Marcel when the
    stuff artifact was not supplied — never the engine that was intended."""
    out = build(pa, seasons)
    assert out.attrs["pitcher_engine_used"] == {
        c: pr.MARCEL_ENGINE for c in pr.SERVED_COMPONENTS}
    assert out.attrs["stuff_features_through"] == "2026-07-31"  # AS_OF is Aug 1


def test_engine_providers_say_why_a_component_fell_back(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise FileNotFoundError("pa_outcomes_2017.parquet")

    monkeypatch.setattr(pr, "stuff_engine_provider", boom)
    with caplog.at_level("WARNING", logger="src.projections.pitcher_ros"):
        providers, used = pr.engine_providers(
            seasons_table=pd.DataFrame(), monthly=pd.DataFrame(), pa_dir=".",
            as_of="2026-09-09", components=("p_bb_rate",))
    assert used["p_bb_rate"] == "marcel_pitcher_tuned"
    assert "p_bb_rate" in caplog.text and "FileNotFoundError" in caplog.text


# --- the command engine, built and withheld (BAS-88) --------------------

def test_the_command_engine_exists_and_is_not_served():
    """BAS-88 built `command_additive` and did not ship it.

    Its serving pre-registration carried a vacuity clause on the coefficient
    of `cmd_csw` in the fit that would actually serve 2026, and that clause
    fired: the walk-forward fit on 2017-2025 puts `cmd_csw` at a
    pitcher-clustered t of -0.55 against a bar of |t| > 2. The same fit
    scored on 2026's own cutoffs beats `stuff_additive` by 1.11% of MAE at
    t -1.13, missing the pre-registered effect floor of 1.0% at |t| > 2.0 on
    its significance. Two clauses say withhold, so BB/BF is still
    `stuff_additive` and no served component names the command engine.

    This test is the guard on that decision: turning the engine on is a
    deliberate one-line change to `LIVE_ENGINE` in a ticket that earns it,
    and it should have to change this test too.
    """
    assert pr.COMMAND_ENGINE == "command_additive"
    assert pr.LIVE_ENGINE["p_bb_rate"] == pr.STUFF_ENGINE
    assert pr.COMMAND_ENGINE not in pr.LIVE_ENGINE.values()


def test_the_command_cutoff_is_the_stuff_cutoff():
    """Both blocks of the joint fit are read at the same month boundary, so a
    build cannot pair April command with August stuff."""
    for as_of in ("2026-09-09", "2026-09-30", "2026-09-01"):
        assert pr.command_cutoff(as_of) == pr.stuff_cutoff(as_of)
    assert pr.command_features_through("2026-09-09") == pd.Timestamp("2026-08-31")


def test_the_command_engine_never_reads_a_bucket_from_the_as_of_month():
    """The leakage guard on the command block, at the engine's own cutoff: an
    as-of date inside September must not see a September command bucket, even
    though the monthly artifact has one."""
    from src.data.pitching_command import COUNT_COLUMNS
    from src.eval.command import LEVEL_FEATURES, features_at_cutoff

    def bucket(pitcher, season, month, pitches, csw, zone_share):
        row = {"pitcher": pitcher, "season": season, "month": month}
        row.update({c: 0.0 for c in COUNT_COLUMNS})
        row["pitches"] = float(pitches)
        row["takens"] = pitches * 0.53
        row["cmd_csw_sum"] = pitches * csw
        row["in_zone"] = pitches * zone_share
        for r, share in (("heart", 0.25), ("shadow", 0.4), ("chase", 0.25),
                         ("waste", 0.1)):
            row[f"n_{r}"] = pitches * share
        return row

    as_of = "2026-09-09"
    cutoff = pr.command_cutoff(as_of)
    normal = [bucket(1, 2026, m, 400, 0.30, 0.48) for m in (5, 6, 7, 8)]
    normal += [bucket(2, 2026, m, 400, 0.26, 0.42) for m in (5, 6, 7, 8)]
    clean = features_at_cutoff(pd.DataFrame(normal), cutoff, 2026,
                               features=LEVEL_FEATURES)
    leaky = features_at_cutoff(
        pd.DataFrame(normal + [bucket(1, 2026, 9, 10000, 0.95, 0.99)]),
        cutoff, 2026, features=LEVEL_FEATURES)
    pd.testing.assert_frame_equal(
        clean.sort_values("player").reset_index(drop=True),
        leaky.sort_values("player").reset_index(drop=True))


def test_the_command_engine_falls_back_to_stuff_not_to_marcel(monkeypatch,
                                                              caplog):
    """The pre-registration's fallback is one rung at a time: a component
    whose command fit cannot be built is served on `stuff_additive`, which is
    a gated engine, rather than dropping all the way to Marcel."""
    monkeypatch.setitem(pr.LIVE_ENGINE, "p_bb_rate", pr.COMMAND_ENGINE)
    monkeypatch.setattr(pr, "command_engine_provider", _boom)
    sentinel = object()
    monkeypatch.setattr(pr, "stuff_engine_provider",
                        lambda *a, **k: sentinel)
    with caplog.at_level("WARNING", logger="src.projections.pitcher_ros"):
        providers, used = pr.engine_providers(
            seasons_table=pd.DataFrame(), monthly=pd.DataFrame(), pa_dir=".",
            as_of="2026-09-09", components=("p_bb_rate",),
            command_monthly=pd.DataFrame())
    assert used["p_bb_rate"] == pr.STUFF_ENGINE
    assert providers["p_bb_rate"] is sentinel
    assert "p_bb_rate" in caplog.text and "FileNotFoundError" in caplog.text


def test_a_missing_command_artifact_falls_back_to_stuff_and_says_so(
        monkeypatch, caplog):
    """No command artifact at all is the ordinary stripped-checkout case, and
    it is logged rather than silently serving a different engine."""
    monkeypatch.setitem(pr.LIVE_ENGINE, "p_bb_rate", pr.COMMAND_ENGINE)
    sentinel = object()
    monkeypatch.setattr(pr, "stuff_engine_provider",
                        lambda *a, **k: sentinel)
    with caplog.at_level("WARNING", logger="src.projections.pitcher_ros"):
        providers, used = pr.engine_providers(
            seasons_table=pd.DataFrame(), monthly=pd.DataFrame(), pa_dir=".",
            as_of="2026-09-09", components=("p_bb_rate",),
            command_monthly=None)
    assert used["p_bb_rate"] == pr.STUFF_ENGINE
    assert "no pitching-command artifact" in caplog.text


def test_both_engines_failing_still_lands_on_marcel(monkeypatch, caplog):
    """The bottom of the ladder: a worse projection, served honestly, and the
    document records Marcel rather than the engine that was intended."""
    monkeypatch.setitem(pr.LIVE_ENGINE, "p_bb_rate", pr.COMMAND_ENGINE)
    monkeypatch.setattr(pr, "command_engine_provider", _boom)
    monkeypatch.setattr(pr, "stuff_engine_provider", _boom)
    with caplog.at_level("WARNING", logger="src.projections.pitcher_ros"):
        providers, used = pr.engine_providers(
            seasons_table=pd.DataFrame(), monthly=pd.DataFrame(), pa_dir=".",
            as_of="2026-09-09", components=("p_bb_rate",),
            command_monthly=pd.DataFrame())
    assert used["p_bb_rate"] == pr.MARCEL_ENGINE
    assert providers["p_bb_rate"] is pitcher_eval.marcel_pitcher_tuned


def test_the_projection_stamps_both_feature_dates(pa, seasons):
    """Provenance for each block rides along, so the lag on either one is
    visible rather than implicit — even while the command engine is withheld."""
    out = build(pa, seasons)
    assert out.attrs["stuff_features_through"] == "2026-07-31"   # AS_OF is Aug 1
    assert out.attrs["command_features_through"] == "2026-07-31"


def _boom(*args, **kwargs):
    raise FileNotFoundError("pa_outcomes_2017.parquet")
