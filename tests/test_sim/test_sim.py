"""Tests for the season simulator (roadmap Phase 2) on a toy league."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sim.bracket import play_postseason, play_series
from src.sim.odds import run_playoff_odds
from src.sim.season import from_schedule, simulate_remaining, tally
from src.sim.standings import TiebreakContext, break_tie, seed_league
from src.sim.strength import (
    estimate_hfa, home_win_prob, league_ra_per_game, log5, pythagenpat,
    regressed_run_rates, regressed_strength,
)


class TestStrength:
    def test_pythagenpat_symmetry_and_direction(self):
        assert pythagenpat(700, 700, 162) == pytest.approx(0.5)
        assert pythagenpat(800, 600, 162) > 0.6
        assert pythagenpat(600, 800, 162) < 0.4

    def test_log5_identities(self):
        assert log5(0.6, 0.6) == pytest.approx(0.5)
        assert log5(0.6, 0.5) == pytest.approx(0.6)
        assert log5(0.5, 0.4) == pytest.approx(0.6)

    def test_home_field_raises_home_prob(self):
        assert home_win_prob(0.5, 0.5, hfa=0.54) == pytest.approx(0.54)
        assert home_win_prob(0.5, 0.5, hfa=0.5) == pytest.approx(0.5)
        # Symmetric: P(home wins | A home) + P(home wins | B home) = 1 when no HFA
        p1 = home_win_prob(0.6, 0.45, hfa=0.5)
        p2 = home_win_prob(0.45, 0.6, hfa=0.5)
        assert p1 + p2 == pytest.approx(1.0)

    def test_regressed_strength_shrinks_toward_mean(self):
        st = pd.DataFrame({
            "team_id": [1, 2], "wins": [10, 5], "losses": [5, 10],
            "runs_scored": [90, 60], "runs_allowed": [60, 90],
        })
        raw = pythagenpat(90, 60, 15)
        s = regressed_strength(st, regress_games=60)
        assert 0.5 < s[1] < raw
        assert s[1] + s[2] == pytest.approx(1.0, abs=1e-6)

    def test_regressed_run_rates_are_labelled_by_team_id(self):
        """Rows carry team ids as *labels*, not as a reindex of the row order.

        Building the frame with `index=team_ids` would align the columns by
        label against `standings`' own 0..n index and silently produce NaNs
        for every team id that is not also a row position — which is all of
        them, since MLB team ids start at 108.
        """
        st = pd.DataFrame({
            "team_id": [108, 109, 110], "wins": [60, 50, 40],
            "losses": [40, 50, 60], "runs_scored": [500, 450, 400],
            "runs_allowed": [400, 450, 500],
        })
        rates = regressed_run_rates(st, regress_games=60)
        assert list(rates.index) == [108, 109, 110]
        assert rates.notna().all().all()
        lg = (500 + 450 + 400) / 300
        assert rates.loc[108, "rs_pg"] == pytest.approx((500 + 60 * lg) / 160)
        # A shuffled standings frame gives each team the same numbers.
        shuffled = regressed_run_rates(st.sample(frac=1, random_state=0), 60)
        pd.testing.assert_frame_equal(rates, shuffled.loc[rates.index])

    def test_regressed_strength_is_pythagenpat_of_the_rates(self):
        st = pd.DataFrame({
            "team_id": [108, 109], "wins": [10, 5], "losses": [5, 10],
            "runs_scored": [90, 60], "runs_allowed": [60, 90],
        })
        rates = regressed_run_rates(st, regress_games=60)
        s = regressed_strength(st, regress_games=60)
        for t in (108, 109):
            assert s[t] == pytest.approx(
                pythagenpat(rates.loc[t, "rs_pg"], rates.loc[t, "ra_pg"], 1.0))
        assert league_ra_per_game(st) == pytest.approx(150 / 30)

    def test_estimate_hfa_blends_prior(self):
        c = pd.DataFrame({"home_win": [True] * 100})
        assert 0.54 < estimate_hfa(c, prior=0.54, prior_games=100) < 1.0
        assert estimate_hfa(c.iloc[:0]) == 0.54


# ─── Toy league: 4 teams, two leagues × one division each ─────────────────
@pytest.fixture
def toy_teams():
    return pd.DataFrame({
        "team_id": [1, 2, 3, 4],
        "abbrev": ["AAA", "BBB", "CCC", "DDD"],
        "name": ["A", "B", "C", "D"],
        "league_id": [103, 103, 104, 104],
        "division_id": [201, 201, 204, 204],
    })


def _sched(rows):
    df = pd.DataFrame(rows, columns=["date", "home_id", "away_id", "home_score", "away_score"])
    df["status"] = np.where(df["home_score"].notna(), "Final", "Preview")
    df["game_type"] = "R"
    df["game_pk"] = range(len(df))
    return df


@pytest.fixture
def toy_schedule():
    # Completed: 1 beat 2 twice, 2 beat 1 once; 3 and 4 split.
    # A postponed 'Final' with no score must be dropped.
    return _sched([
        ("2026-04-01", 1, 2, 5, 3), ("2026-04-02", 2, 1, 2, 4), ("2026-04-03", 2, 1, 6, 1),
        ("2026-04-01", 3, 4, 3, 2), ("2026-04-02", 4, 3, 4, 1),
        ("2026-04-04", 1, 2, None, None), ("2026-04-05", 3, 4, None, None),
    ]).assign(status=lambda d: np.where(d.home_score.notna(), "Final", "Preview"))


class TestSeasonState:
    def test_split_and_postponed_drop(self, toy_teams, toy_schedule):
        toy_schedule.loc[len(toy_schedule)] = ["2026-04-06", 1, 2, None, None, "Final", "R", 99]
        state = from_schedule(toy_schedule, toy_teams)
        assert len(state.completed) == 5
        assert len(state.remaining) == 2

    def test_tally_conserves_games(self, toy_teams, toy_schedule):
        state = from_schedule(toy_schedule, toy_teams)
        strength = pd.Series({1: 0.6, 2: 0.4, 3: 0.5, 4: 0.5})
        hw = simulate_remaining(state, strength, 0.54, 50, np.random.default_rng(0))
        rec = tally(state, hw)
        total_games = len(state.completed) + len(state.remaining)
        assert np.all(rec.wins.sum(1) == total_games)
        assert np.all((rec.wins + rec.losses).sum(1) == 2 * total_games)
        # Team 1 has 2 wins banked, team 2 has 1.
        assert rec.wins[:, 0].min() >= 2 and rec.wins[:, 1].min() >= 1


def _twelve_team_league(seed: int = 5):
    """6 teams per league, 3 divisions of 2 — the smallest league the bracket accepts."""
    rng = np.random.default_rng(seed)
    teams = pd.DataFrame({
        "team_id": range(1, 13),
        "abbrev": [f"T{i}" for i in range(1, 13)],
        "name": [f"Team {i}" for i in range(1, 13)],
        "league_id": [103] * 6 + [104] * 6,
        "division_id": [200, 200, 201, 201, 202, 202, 203, 203, 204, 204, 205, 205],
    })
    strength = pd.Series({i: float(v) for i, v in
                          zip(range(1, 13), np.linspace(0.62, 0.38, 12))})
    rows, ids = [], list(range(1, 13))
    for d in range(40):
        perm = rng.permutation(ids)
        for h, a in zip(perm[::2], perm[1::2]):
            done = d < 30
            hs, as_ = (rng.integers(0, 8), rng.integers(0, 8)) if done else (None, None)
            if done and hs == as_:
                hs += 1
            rows.append((f"2026-04-{d+1:02d}" if d < 30 else f"2026-05-{d-29:02d}",
                         int(h), int(a), hs, as_))
    return teams, strength, from_schedule(_sched(rows), teams)


class TestOverrides:
    """Per-game P(home) overrides (station E's starting-pitcher term).

    The nightly job knows who is pitching in the next few days and nothing
    about the rest of the schedule, so an override has to be surgical: it
    replaces the log5 probability for the games it names and leaves every
    other game's draw bit-identical.
    """

    def _state(self, toy_teams):
        # Four remaining games so there is something to leave alone.
        sched = _sched([
            ("2026-04-01", 1, 2, 5, 3), ("2026-04-01", 3, 4, 3, 2),
            ("2026-04-04", 1, 2, None, None), ("2026-04-05", 3, 4, None, None),
            ("2026-04-06", 2, 1, None, None), ("2026-04-07", 4, 3, None, None),
        ])
        return from_schedule(sched, toy_teams)

    STRENGTH = pd.Series({1: 0.6, 2: 0.4, 3: 0.5, 4: 0.5})

    def _draw(self, state, overrides=None, n=400, seed=7):
        return simulate_remaining(state, self.STRENGTH, 0.54, n,
                                  np.random.default_rng(seed),
                                  p_home_overrides=overrides)

    def test_game_pk_rides_along_to_the_remaining_frame(self, toy_teams):
        state = self._state(toy_teams)
        assert "game_pk" in state.remaining.columns
        assert list(state.remaining["game_pk"]) == [2, 3, 4, 5]

    def test_no_overrides_is_the_unchanged_path(self, toy_teams):
        state = self._state(toy_teams)
        assert np.array_equal(self._draw(state), self._draw(state, overrides={}))
        assert np.array_equal(self._draw(state), self._draw(state, overrides=None))

    def test_override_changes_only_the_named_games(self, toy_teams):
        state = self._state(toy_teams)
        base = self._draw(state)
        # Override the second remaining game (game_pk 3) only.
        got = self._draw(state, overrides={3: 0.99})
        cols = list(state.remaining["game_pk"])
        target = cols.index(3)
        others = [i for i in range(len(cols)) if i != target]
        assert np.array_equal(base[:, others], got[:, others])
        assert got[:, target].mean() > base[:, target].mean()

    def test_probability_one_always_wins_and_zero_always_loses(self, toy_teams):
        state = self._state(toy_teams)
        pks = list(state.remaining["game_pk"])
        got = self._draw(state, overrides={pks[0]: 1.0, pks[2]: 0.0})
        assert got[:, 0].all()
        assert not got[:, 2].any()

    def test_unknown_game_pk_is_ignored(self, toy_teams):
        """A game that went final between the probables fetch and the sim."""
        state = self._state(toy_teams)
        assert np.array_equal(self._draw(state), self._draw(state, overrides={9999: 1.0}))

    def test_overrides_need_a_game_pk_column(self, toy_teams):
        state = self._state(toy_teams)
        state.remaining.drop(columns=["game_pk"], inplace=True)
        with pytest.raises(KeyError):
            self._draw(state, overrides={3: 1.0})

    def test_run_playoff_odds_threads_the_overrides_through(self):
        """The bracket needs six seeds a league, so this one uses the big toy."""
        teams, strength, state = _twelve_team_league()
        plain = run_playoff_odds(state, strength, 0.54, n_sims=200, seed=3)
        # Hand every remaining game to the home team.
        forced = {int(pk): 1.0 for pk in state.remaining["game_pk"]}
        forced_odds = run_playoff_odds(state, strength, 0.54, n_sims=200,
                                       seed=3, p_home_overrides=forced)
        hosted = state.remaining["home_id"].value_counts()
        banked = plain.set_index("team_id")["wins"]
        got = forced_odds.set_index("team_id")["mean_wins"]
        for t in teams["team_id"]:
            assert got[t] == pytest.approx(banked[t] + hosted.get(t, 0))
        # And the odds themselves moved. (Every team makes the playoffs in a
        # 6-team league, so p_playoffs is a constant 1.0 here — use p_division.)
        assert not np.allclose(
            plain.sort_values("team_id")["p_division"].to_numpy(),
            forced_odds.sort_values("team_id")["p_division"].to_numpy())

    def test_overrides_leave_untouched_games_identical_end_to_end(self):
        """Override one game; every other game's simulated outcome is unchanged."""
        _, strength, state = _twelve_team_league()
        rng_args = (strength, 0.54, 200, np.random.default_rng(11))
        base = simulate_remaining(state, *rng_args)
        pk = int(state.remaining["game_pk"].iloc[3])
        got = simulate_remaining(state, strength, 0.54, 200,
                                 np.random.default_rng(11),
                                 p_home_overrides={pk: 1.0})
        cols = [i for i in range(base.shape[1]) if i != 3]
        assert np.array_equal(base[:, cols], got[:, cols])
        assert got[:, 3].all()


class TestTiebreak:
    def _ctx(self, toy_teams, schedule, hw):
        state = from_schedule(schedule, toy_teams)
        rec = tally(state, hw)
        return state, TiebreakContext.build(state, rec, hw, np.random.default_rng(1))

    def test_head_to_head_decides(self, toy_teams):
        # 1 and 2 both 2-2 overall, but 1 is 2-1 vs 2 (one win vs an outsider each way).
        sched = _sched([
            ("2026-04-01", 1, 2, 5, 3), ("2026-04-02", 2, 1, 2, 4), ("2026-04-03", 2, 1, 6, 1),
            ("2026-04-04", 1, 3, 1, 2), ("2026-04-05", 2, 4, 2, 1),
            ("2026-04-06", 3, 4, 2, 1), ("2026-04-06", 4, 3, 2, 1),
        ])
        state, ctx = self._ctx(toy_teams, sched, np.zeros((1, 0), dtype=bool))
        assert ctx.records.wins[0, 0] == ctx.records.wins[0, 1] == 2
        assert break_tie([0, 1], 0, ctx) == [0, 1]
        assert break_tie([1, 0], 0, ctx) == [0, 1]

    def test_falls_through_to_intradivision(self, toy_teams):
        # Even H2H (1-1); team 1 wins an extra intradivision game vs... only 2 is in-division,
        # so give both an interleague loss but team 2's loss is intradivision? Simplify:
        # 1 vs 2 split; then team 1 beats 3 (interleague), team 2 loses to 4 (interleague).
        # Both 1-1... make wins equal: 1: W vs 2, W vs 3, L vs 4 = 2-1 ; 2: W vs 1, W vs 4, L vs 3 = 2-1
        # H2H 1-1. intradiv both 1-1. Then falls to intraleague-last-half... which is the same
        # 1-1 for both → coin flip. Assert only that a valid ordering is returned.
        sched = _sched([
            ("2026-04-01", 1, 2, 5, 3), ("2026-04-02", 2, 1, 4, 2),
            ("2026-04-03", 1, 3, 3, 1), ("2026-04-04", 4, 1, 3, 1),
            ("2026-04-03", 2, 4, 3, 1), ("2026-04-04", 3, 2, 3, 1),
            ("2026-04-05", 3, 4, 2, 1),
        ])
        state, ctx = self._ctx(toy_teams, sched, np.zeros((1, 0), dtype=bool))
        assert sorted(break_tie([0, 1], 0, ctx)) == [0, 1]

    def test_seed_league_structure(self, toy_teams, toy_schedule):
        state = from_schedule(toy_schedule, toy_teams)
        hw = np.ones((1, len(state.remaining)), dtype=bool)  # home teams win remaining
        rec = tally(state, hw)
        ctx = TiebreakContext.build(state, rec, hw, np.random.default_rng(0))
        seeds = seed_league(0, 103, ctx)
        assert seeds.division_winners == [0]        # team 1 (row 0) leads AL
        assert seeds.wild_cards == [1]


class TestBracket:
    def test_dominant_team_wins_series(self):
        strength = np.array([0.99, 0.01])
        rng = np.random.default_rng(0)
        wins = sum(play_series(0, 1, 3, strength, 0.54, rng) == 0 for _ in range(200))
        assert wins == 200

    def test_postseason_returns_champion_from_seeds(self):
        strength = np.full(12, 0.5)
        seeds = {103: [0, 1, 2, 3, 4, 5], 104: [6, 7, 8, 9, 10, 11]}
        res = play_postseason(seeds, np.full(12, 90), strength, 0.54, np.random.default_rng(0))
        assert res.champion in range(12)
        assert set(res.pennant) == {103, 104}


class TestEndToEnd:
    def test_probabilities_are_coherent(self):
        """12-team league (6 per league, 3 divisions of 2) through the full pipeline."""
        rng = np.random.default_rng(5)
        teams = pd.DataFrame({
            "team_id": range(1, 13),
            "abbrev": [f"T{i}" for i in range(1, 13)],
            "name": [f"Team {i}" for i in range(1, 13)],
            "league_id": [103] * 6 + [104] * 6,
            "division_id": [200, 200, 201, 201, 202, 202, 203, 203, 204, 204, 205, 205],
        })
        strength = pd.Series({i: float(v) for i, v in
                              zip(range(1, 13), np.linspace(0.62, 0.38, 12))})
        rows = []
        ids = list(range(1, 13))
        for d in range(40):
            perm = rng.permutation(ids)
            for h, a in zip(perm[::2], perm[1::2]):
                done = d < 30
                hs, as_ = (rng.integers(0, 8), rng.integers(0, 8)) if done else (None, None)
                if done and hs == as_:
                    hs += 1
                rows.append((f"2026-04-{d+1:02d}" if d < 30 else f"2026-05-{d-29:02d}",
                             int(h), int(a), hs, as_))
        state = from_schedule(_sched(rows), teams)
        odds = run_playoff_odds(state, strength, 0.54, n_sims=300, seed=1)

        assert len(odds) == 12
        # Per league: 6 playoff spots = every team; 3 division winners, 2 byes, 1 pennant.
        for lg in (103, 104):
            g = odds[odds.league_id == lg]
            assert g["p_division"].sum() == pytest.approx(3.0)
            assert g["p_bye"].sum() == pytest.approx(2.0)
            assert g["p_pennant"].sum() == pytest.approx(1.0)
        assert odds["p_ws"].sum() == pytest.approx(1.0)
        assert ((odds["p_playoffs"] >= odds["p_division"]) &
                (odds["p_division"] >= odds["p_bye"]) &
                (odds["p_pennant"] >= odds["p_ws"])).all()
        assert (odds["wins_p5"] <= odds["wins_p50"]).all() and (odds["wins_p50"] <= odds["wins_p95"]).all()
