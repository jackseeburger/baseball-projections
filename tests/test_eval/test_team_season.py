"""The team walk-forward harness cannot see a game it has not been shown.

`src/eval/team_season.py` cuts a season at a date and projects the rest of it.
The whole value of the number that comes out depends on one property: nothing
on or after the cutoff reaches the inputs. These tests attack that property
directly rather than trusting the filters.

The central one is `TestExtremePostCutoff`: the same synthetic season is built
twice, identical up to the cutoff and *mirrored* after it — every remaining
game is a 20-0 blowout for the clubs that lost them in the first version. If
any post-cutoff information leaked, the two projections would differ. They are
asserted equal, column for column, including the Monte Carlo's own output,
because the seed and every input it draws from are the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import team_backtest as tb           # noqa: E402
from src.eval import team_season as ts             # noqa: E402
from src.sim import game_model as gm               # noqa: E402
from src.sim.bracket import (                      # noqa: E402
    MODERN, ONE_GAME_WILD_CARD, format_for_season,
)

SEASON = 2024
CUTOFF = "2024-05-01"
N_DATES = 60
CLUBS = 30


# ─── a synthetic 30-club season ───

def teams_frame() -> pd.DataFrame:
    ids = list(range(101, 101 + CLUBS))
    league = [103] * 15 + [104] * 15
    division = ([200] * 5 + [201] * 5 + [202] * 5
                + [203] * 5 + [204] * 5 + [205] * 5)
    return pd.DataFrame({
        "team_id": ids,
        "abbrev": [f"T{i:02d}" for i in range(CLUBS)],
        "name": [f"Team {i:02d}" for i in range(CLUBS)],
        "league_id": league, "division_id": division,
    })


def schedule_frame(*, mirror_after: str | None = None) -> pd.DataFrame:
    """A whole season, every club playing every date.

    `mirror_after` flips the winner and blows the score out to 20-0 for every
    game on or after that date, which is the "extreme post-cutoff" season.
    """
    teams = teams_frame()
    ids = teams["team_id"].to_numpy()
    rng = np.random.default_rng(7)
    rows, pk = [], 500_000
    for d in range(N_DATES):
        day = str((pd.Timestamp("2024-04-01") + pd.Timedelta(days=d)).date())
        order = rng.permutation(ids)
        for i in range(0, CLUBS, 2):
            pk += 1
            home, away = int(order[i]), int(order[i + 1])
            hs, as_ = (6, 3) if (home + d) % 2 == 0 else (2, 5)
            if mirror_after is not None and day >= mirror_after:
                hs, as_ = (0, 20) if hs > as_ else (20, 0)
            rows.append({"game_pk": pk, "date": day, "game_type": "R",
                         "status": "Final", "home_id": home, "away_id": away,
                         "home_score": hs, "away_score": as_})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def teams():
    return teams_frame()


@pytest.fixture(scope="module")
def schedule():
    return schedule_frame()


class TestSplit:
    def test_cut_is_strict_and_complete(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        assert split.played["date"].astype(str).max() < CUTOFF
        assert split.future["date"].astype(str).min() >= CUTOFF
        assert split.games_played + split.games_remaining == len(schedule)
        # A game *on* the cutoff is still to be played, as it is for the
        # nightly job, which runs before first pitch.
        assert (split.future["date"] == CUTOFF).any()

    def test_state_mirrors_the_split(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        assert len(split.state.completed) == split.games_played
        assert len(split.state.remaining) == split.games_remaining
        assert set(split.state.completed["game_pk"]) == set(split.played["game_pk"])

    def test_standings_sum_only_the_played_games(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        st = split.standings.set_index("team_id")
        assert int((st["wins"] + st["losses"]).sum()) == 2 * split.games_played
        # Hand-check one club against the raw frame.
        team_id = int(teams["team_id"].iloc[0])
        home = split.played[split.played["home_id"] == team_id]
        away = split.played[split.played["away_id"] == team_id]
        wins = int((home["home_score"] > home["away_score"]).sum()
                   + (away["away_score"] > away["home_score"]).sum())
        assert int(st.loc[team_id, "wins"]) == wins
        assert float(st.loc[team_id, "runs_scored"]) == float(
            home["home_score"].sum() + away["away_score"].sum())

    def test_club_games_remaining_is_per_club(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        rem = split.club_games_remaining()
        assert int(rem.sum()) == 2 * split.games_remaining
        assert set(rem.index) == set(teams["team_id"])

    def test_weekly_cutoffs_are_weekly_and_inside_the_season(self, schedule,
                                                             teams):
        cuts = ts.weekly_cutoffs(schedule, teams, step_days=7, skip_days=14,
                                 min_remaining=30)
        assert cuts == sorted(cuts)
        gaps = {(pd.Timestamp(b) - pd.Timestamp(a)).days
                for a, b in zip(cuts, cuts[1:])}
        assert gaps == {7}
        dates = schedule["date"].astype(str)
        assert cuts[0] >= str((pd.Timestamp(dates.min())
                               + pd.Timedelta(days=14)).date())
        assert cuts[-1] <= dates.max()


class TestLeakageGuard:
    def test_clean_split_passes(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        ts.assert_team_split_clean(split)

    def test_a_post_cutoff_game_in_played_raises(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        dirty = ts.TeamSplit(
            season=split.season, as_of=split.as_of, teams=split.teams,
            played=pd.concat([split.played, split.future.head(1)],
                             ignore_index=True),
            future=split.future, state=split.state,
            standings=split.standings, hfa=split.hfa)
        with pytest.raises(ValueError, match="on or after the cutoff"):
            ts.assert_team_split_clean(dirty)

    def test_a_pre_cutoff_game_in_future_raises(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        dirty = ts.TeamSplit(
            season=split.season, as_of=split.as_of, teams=split.teams,
            played=split.played,
            future=pd.concat([split.future, split.played.head(1)],
                             ignore_index=True),
            state=split.state, standings=split.standings, hfa=split.hfa)
        with pytest.raises(ValueError, match="before the cutoff"):
            ts.assert_team_split_clean(dirty)

    def test_final_standings_in_place_of_as_of_standings_raises(self, schedule,
                                                                teams):
        """The mistake the guard exists for.

        `fetch_standings(2016)` serves the *final* 2016 table. Handing that to
        a projection made on 2016-05-01 is the single most damaging thing that
        can go wrong in this harness and it leaves no trace in the output, so
        the guard reconciles the standings against the games instead.
        """
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        final = ts.standings_from_games(
            ts.regular_season_games(schedule, teams), teams)
        dirty = ts.TeamSplit(
            season=split.season, as_of=split.as_of, teams=split.teams,
            played=split.played, future=split.future, state=split.state,
            standings=final, hfa=split.hfa)
        with pytest.raises(ValueError, match="not summed from the pre-cutoff"):
            ts.assert_team_split_clean(dirty)

    def test_nothing_left_to_project_raises(self, schedule, teams):
        last = schedule["date"].astype(str).max()
        after = str((pd.Timestamp(last) + pd.Timedelta(days=1)).date())
        split = ts.split_season_at(schedule, teams, after, SEASON)
        with pytest.raises(ValueError, match="nothing left to project"):
            ts.assert_team_split_clean(split)

    def test_probables_past_the_window_raise(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        probables = pd.DataFrame({
            "game_pk": schedule["game_pk"], "date": schedule["date"],
            "game_type": "R",
            "home_sp_id": 1, "away_sp_id": 2})
        with pytest.raises(ValueError, match="starter window"):
            ts.assert_team_split_clean(split, probables=probables,
                                       window_days=7)
        trimmed = ts.probables_to_window(probables, CUTOFF, 7)
        ts.assert_team_split_clean(split, probables=trimmed, window_days=7)


# ─── the chain's own inputs ───

def _player_logs(cutoff_extreme: bool) -> dict:
    """A tiny two-club player-level season, with a post-cutoff blow-out option.

    Only the dates matter for what is being asserted: whether a row dated on
    or after the cutoff can reach a rate table. `cutoff_extreme` makes the
    post-cutoff rows absurd (every start a 30-strikeout shutout, every hitter
    a home run an inning), so a leak of any size moves a rate visibly.
    """
    dates = [str((pd.Timestamp("2024-04-01") + pd.Timedelta(days=d)).date())
             for d in range(N_DATES)]
    p_rows, h_rows = [], []
    for d, day in enumerate(dates):
        late = cutoff_extreme and day >= CUTOFF
        for ti, team in enumerate((101, 102)):
            sp = 1000 + 10 * ti + (d % 5)
            p_rows.append({"pitcher": sp, "season": SEASON, "date": day,
                           "game_pk": 900000 + d * 2 + ti, "game_type": "R",
                           "team": team, "outs": 18.0,
                           "bf": 24.0, "k": 30.0 if late else 6.0,
                           "bb": 0.0 if late else 2.0, "hbp": 0.0,
                           "hr": 0.0 if late else 1.0, "er": 0.0 if late else 3.0,
                           "gs": 1, "pitches": 95.0})
            for j in range(3):
                p_rows.append({"pitcher": 1500 + 10 * ti + j, "season": SEASON,
                               "date": day, "game_pk": 900000 + d * 2 + ti,
                               "game_type": "R", "team": team, "outs": 3.0,
                               "bf": 4.0, "k": 4.0 if late else 1.0, "bb": 0.0,
                               "hbp": 0.0, "hr": 0.0 if late else 1.0,
                               "er": 0.0 if late else 1.0, "gs": 0,
                               "pitches": 15.0})
            for b in range(9):
                h_rows.append({"batter": 2000 + 100 * ti + b, "season": SEASON,
                               "date": day, "game_pk": 900000 + d * 2 + ti,
                               "game_type": "R", "team_id": team,
                               "pa": 5, "ab": 5, "h": 5 if late else 1,
                               "doubles": 0, "triples": 0,
                               "hr": 5 if late else 0, "k": 0 if late else 1,
                               "bb": 0, "hbp": 0, "sf": 0})
    prior_p, prior_h = [], []
    for ti in range(2):
        for y in (SEASON - 2, SEASON - 1):
            for j in range(5):
                prior_p.append({"pitcher": 1000 + 10 * ti + j, "season": y,
                                "bf": 600.0, "k": 130.0, "bb": 50.0, "hbp": 6.0,
                                "hr": 20.0, "er": 70.0, "gs": 30,
                                "pitches": 2400.0, "outs": 450.0})
            for j in range(3):
                prior_p.append({"pitcher": 1500 + 10 * ti + j, "season": y,
                                "bf": 250.0, "k": 60.0, "bb": 25.0, "hbp": 3.0,
                                "hr": 8.0, "er": 30.0, "gs": 0,
                                "pitches": 1000.0, "outs": 180.0})
            for b in range(9):
                prior_h.append({"batter": 2000 + 100 * ti + b, "season": y,
                                "pa": 600.0, "ab": 540.0, "h": 140.0,
                                "doubles": 28.0, "triples": 2.0, "hr": 20.0,
                                "k": 130.0, "bb": 55.0, "hbp": 5.0, "sf": 4.0,
                                "age": 28})
    return {"pitching": pd.DataFrame(p_rows), "hitting": pd.DataFrame(h_rows),
            "prior_pitching": pd.DataFrame(prior_p),
            "prior_hitting": pd.DataFrame(prior_h)}


def _slate(logs: dict) -> gm.Slate:
    inputs = ts.chain_inputs_before(SEASON, logs["pitching"], logs["hitting"],
                                    logs["prior_pitching"],
                                    logs["prior_hitting"], CUTOFF)
    top_down = pd.DataFrame({"rs_pg": [4.5, 4.5], "ra_pg": [4.5, 4.5]},
                            index=pd.Index([101, 102], name="team_id"))
    return inputs, gm.build_slate(CUTOFF, inputs, top_down, 4.5, 4.5)


class TestChainInputs:
    def test_inputs_are_cut_and_the_guard_agrees(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        logs = _player_logs(cutoff_extreme=True)
        inputs, _ = _slate(logs)
        for frame in (inputs.pitcher_counts, inputs.relief, inputs.usage,
                      inputs.starts, inputs.start_ip, inputs.hitter_counts,
                      inputs.hitter_pa):
            assert frame["date"].astype(str).max() < CUTOFF
        ts.assert_team_split_clean(split, inputs=inputs)

    def test_an_uncut_input_frame_raises(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        logs = _player_logs(cutoff_extreme=True)
        uncut = gm.ChainInputs.from_logs(SEASON, logs["pitching"],
                                         logs["hitting"],
                                         logs["prior_pitching"],
                                         logs["prior_hitting"])
        with pytest.raises(ValueError, match="on or after the cutoff"):
            ts.assert_team_split_clean(split, inputs=uncut)

    def test_a_prior_frame_from_the_predict_year_raises(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        logs = _player_logs(cutoff_extreme=False)
        logs["prior_pitching"] = logs["prior_pitching"].assign(season=SEASON)
        inputs, _ = _slate(logs)
        with pytest.raises(ValueError, match="not a completed season"):
            ts.assert_team_split_clean(split, inputs=inputs)

    def test_extreme_post_cutoff_play_cannot_move_a_rate(self):
        """The chain's own tables are identical either side of the cutoff.

        One season has ordinary baseball after May 1; the other has every
        starter throwing a 30-strikeout shutout and every hitter homering five
        times a night. `build_slate` sees the same numbers in both.
        """
        _, calm = _slate(_player_logs(cutoff_extreme=False))
        _, wild = _slate(_player_logs(cutoff_extreme=True))
        pd.testing.assert_frame_equal(calm.team, wild.team)
        assert calm.sp_ra9 == wild.sp_ra9
        assert calm.runs_lookup == wild.runs_lookup
        assert calm.expected_ip == wild.expected_ip
        pd.testing.assert_series_equal(calm.talent(), wild.talent())


# ─── the projection itself ───

class TestExtremePostCutoff:
    """The whole projection is invariant to what happens after the cutoff."""

    @staticmethod
    def _project(mirror: bool) -> pd.DataFrame:
        teams = teams_frame()
        sched = schedule_frame(mirror_after=CUTOFF if mirror else None)
        split = ts.split_season_at(sched, teams, CUTOFF, SEASON)
        ts.assert_team_split_clean(split)
        frames = [ts.project(split, ts.strength_even(split), "record_500",
                             n_sims=60, seed=3),
                  ts.project(split, ts.strength_own_rate(split), "record_wpct",
                             n_sims=60, seed=3)]
        return pd.concat(frames, ignore_index=True)

    def test_projection_is_identical_when_the_rest_is_extreme(self):
        calm, wild = self._project(False), self._project(True)
        pd.testing.assert_frame_equal(calm, wild)

    def test_the_extreme_season_really_is_different(self):
        """The control on the control: the two seasons *do* end differently."""
        teams = teams_frame()
        calm = ts.final_records(schedule_frame(), teams)
        wild = ts.final_records(schedule_frame(mirror_after=CUTOFF), teams)
        merged = calm.merge(wild, on="team_id", suffixes=("_c", "_w"))
        assert (merged["final_wins_c"] != merged["final_wins_w"]).any()


class TestBaselineArms:
    def test_even_strength_is_a_coin_flip(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        s = ts.strength_even(split)
        assert set(np.unique(s.to_numpy())) == {0.5}

    def test_own_rate_tracks_the_record_and_is_capped(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        s = ts.strength_own_rate(split)
        st = split.standings.set_index("team_id")
        raw = st["wins"] / (st["wins"] + st["losses"])
        for t in split.state.team_ids:
            assert s[int(t)] == pytest.approx(
                min(max(float(raw[int(t)]), ts.WPCT_FLOOR), ts.WPCT_CEIL))
        assert s.between(ts.WPCT_FLOOR, ts.WPCT_CEIL).all()

    def test_preseason_reads_nothing_from_this_season(self, schedule, teams):
        """Two different cutoffs, one preseason vector — that is the arm."""
        prior = ts.standings_from_games(
            ts.regular_season_games(schedule, teams), teams)
        early = ts.split_season_at(schedule, teams, "2024-04-20", SEASON)
        late = ts.split_season_at(schedule, teams, "2024-05-15", SEASON)
        a = ts.strength_preseason(prior, early.state.team_ids)
        b = ts.strength_preseason(prior, late.state.team_ids)
        pd.testing.assert_series_equal(a, b)


class TestPlayoffFormat:
    def test_the_field_changes_in_2022(self):
        assert format_for_season(2016) is ONE_GAME_WILD_CARD
        assert format_for_season(2021) is ONE_GAME_WILD_CARD
        assert format_for_season(2022) is MODERN
        assert format_for_season(2025) is MODERN
        assert ONE_GAME_WILD_CARD.n_seeds == 5
        assert MODERN.n_seeds == 6

    def test_2020_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError, match="eight-club"):
            format_for_season(2020)

    def test_a_five_club_field_seats_five_clubs(self, schedule, teams):
        """P(playoffs) sums to the size of the field, per league."""
        split = ts.split_season_at(schedule, teams, CUTOFF, 2016)
        assert split.fmt is ONE_GAME_WILD_CARD
        frame = ts.project(split, ts.strength_even(split), "record_500",
                           n_sims=80, seed=5)
        for lg, g in frame.groupby("league_id"):
            assert g["p_playoffs"].sum() == pytest.approx(5.0, abs=1e-9)
            assert g["p_division"].sum() == pytest.approx(3.0, abs=1e-9)
            assert g["p_pennant"].sum() == pytest.approx(1.0, abs=1e-9)
        assert frame["p_ws"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_a_six_club_field_seats_six(self, schedule, teams):
        split = ts.split_season_at(schedule, teams, CUTOFF, 2024)
        frame = ts.project(split, ts.strength_even(split), "record_500",
                           n_sims=80, seed=5)
        for lg, g in frame.groupby("league_id"):
            assert g["p_playoffs"].sum() == pytest.approx(6.0, abs=1e-9)


class TestOutcomes:
    def test_final_records_add_up(self, schedule, teams):
        rec = ts.final_records(schedule, teams)
        assert int(rec["final_wins"].sum()) == len(schedule)
        assert int((rec["final_wins"] + rec["final_losses"]).sum()) == \
            2 * len(schedule)

    def test_postseason_outcomes_read_the_bracket(self, schedule, teams):
        ids = teams["team_id"].to_numpy()
        post = pd.DataFrame([
            {"game_pk": 1, "date": "2024-10-01", "game_type": "F",
             "status": "Final", "home_id": ids[0], "away_id": ids[1],
             "home_score": 3, "away_score": 1},
            {"game_pk": 2, "date": "2024-10-25", "game_type": "W",
             "status": "Final", "home_id": ids[0], "away_id": ids[16],
             "home_score": 2, "away_score": 5},
        ])
        out = ts.postseason_outcomes(pd.concat([schedule, post]), teams)
        got = out.set_index("team_id")
        assert got.loc[ids[0], "made_playoffs"] == 1
        assert got.loc[ids[1], "made_playoffs"] == 1
        assert got.loc[ids[2], "made_playoffs"] == 0
        assert got.loc[ids[0], "won_pennant"] == 1
        assert got.loc[ids[16], "won_pennant"] == 1
        assert got.loc[ids[1], "won_pennant"] == 0
        assert got.loc[ids[16], "won_ws"] == 1
        assert int(out["won_ws"].sum()) == 1


class TestScoring:
    def _scored(self, schedule, teams) -> pd.DataFrame:
        split = ts.split_season_at(schedule, teams, CUTOFF, SEASON)
        frame = ts.project(split, ts.strength_own_rate(split), "record_wpct",
                           n_sims=80, seed=11)
        outcomes = ts.final_records(schedule, teams).assign(
            season=SEASON, made_playoffs=0, won_division=0, won_pennant=0,
            won_ws=0)
        outcomes.loc[outcomes.index[:12], "made_playoffs"] = 1
        return tb.attach_outcomes(frame, outcomes)

    def test_final_and_rest_of_season_wins_error_are_the_same_number(
            self, schedule, teams):
        """Stated in the module docstring, asserted here so it stays true."""
        scored = self._scored(schedule, teams)
        assert np.allclose(scored["err_final_wins"], scored["err_rest_wins"])

    def test_brier_and_log_loss_are_the_textbook_ones(self, schedule, teams):
        scored = self._scored(schedule, teams)
        p = scored["p_playoffs"].to_numpy()
        y = scored["made_playoffs"].to_numpy()
        assert np.allclose(scored["brier_p_playoffs"], (p - y) ** 2)
        pc = np.clip(p, tb.EPS, 1 - tb.EPS)
        assert np.allclose(scored["logloss_p_playoffs"],
                           -(y * np.log(pc) + (1 - y) * np.log(1 - pc)))

    def test_clustered_mean_matches_the_sandwich_by_hand(self):
        d = pd.Series([1.0, 2.0, 3.0, 10.0])
        g = pd.Series(["a", "a", "b", "b"])
        mean, se, G = tb.clustered_mean(d, g)
        assert G == 2
        assert mean == pytest.approx(4.0)
        # cluster sums of (d - mean): a → -5, b → +5
        expected = np.sqrt((25 + 25) * (2 / 1) / 16)
        assert se == pytest.approx(expected)

    def test_clustered_se_exceeds_the_iid_one_when_clusters_agree(self):
        rng = np.random.default_rng(0)
        offsets = {s: rng.normal(0, 1.0) for s in range(8)}
        seasons = np.repeat(list(offsets), 30)
        d = pd.Series([offsets[s] + rng.normal(0, 0.05) for s in seasons])
        _, se, _ = tb.clustered_mean(d, pd.Series(seasons))
        iid = float(d.std(ddof=1) / np.sqrt(len(d)))
        assert se > 2 * iid

    def test_calibration_deciles_partition_the_rows(self, schedule, teams):
        scored = self._scored(schedule, teams)
        cal = tb.calibration(scored, "record_wpct", n_bins=5)
        assert len(cal) == 5
        assert int(cal["n"].sum()) == len(scored)
        assert list(cal["decile"]) == [1, 2, 3, 4, 5]

    def test_reliability_decomposes_the_brier_score(self, schedule, teams):
        scored = self._scored(schedule, teams)
        rel = tb.reliability(scored, "record_wpct", n_bins=5)
        # Murphy, with the within-bin term carried explicitly rather than
        # rounded away: the four together are the Brier score exactly.
        assert (rel["reliability"] - rel["resolution"] + rel["uncertainty"]
                + rel["residual"]) == pytest.approx(rel["brier"], abs=1e-12)
        assert abs(rel["residual"]) < 0.05
        assert rel["skill_score"] == pytest.approx(
            1.0 - rel["brier"] / rel["uncertainty"])
