"""Postseason rotations in the bracket.

The four properties that make the term trustworthy: it cycles the rotation
the way a real series does, it is a no-op when no rotation is supplied, an
ace actually helps and helps most in the game he starts, and two identical
clubs stay a coin flip.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.sim.bracket import (
    DEFAULT_ROTATION_SIZE, Rotations, play_postseason, play_series,
    series_game_probs, strength_with_starter,
)
from src.sim.strength import home_win_prob

LG_RA9 = 4.5          # a plausible league runs-allowed per nine


def rotation(*deltas: float, first_id: int = 100) -> list[tuple[int, float]]:
    """`[(pitcher_id, ra9_delta), ...]` with ids that are easy to eyeball."""
    return [(first_id + i, d) for i, d in enumerate(deltas)]


def rotations(by_team: dict[int, list[tuple[int, float]]]) -> Rotations:
    return Rotations(by_team=by_team, lg_ra9=LG_RA9)


def series_win_rate(high: int, low: int, wins_needed: int, strength, hfa,
                    rots: Rotations | None, n: int = 20_000,
                    seed: int = 7) -> float:
    """Share of `n` simulated series the higher seed wins."""
    rng = np.random.default_rng(seed)
    wins = sum(play_series(high, low, wins_needed, strength, hfa, rng, rots) == high
               for _ in range(n))
    return wins / n


class TestRotationCycling:
    def test_four_man_rotation_repeats_in_game_five(self):
        rots = rotations({0: rotation(-1.0, -0.5, 0.0, 0.5)})
        assert len(rots.by_team[0]) == DEFAULT_ROTATION_SIZE
        for game_idx in range(4):
            assert rots.starter(0, game_idx) == rots.starter(0, game_idx + 4)
        assert rots.starter(0, 0)[0] == 100      # game 1 is the first listed
        assert rots.starter(0, 4)[0] == 100      # game 5 is the same arm
        assert rots.starter(0, 1)[0] == 101

    def test_the_probability_table_repeats_with_the_rotation(self):
        """Game 5 of a best-of-7 is priced exactly like game 1."""
        strength = np.array([0.55, 0.50])
        rots = rotations({0: rotation(-1.0, 0.0, 0.0, 0.0),
                          1: rotation(0.0, 0.0, 0.0, 0.0)})
        at_home, _ = series_game_probs(0, 1, 4, strength, 0.54, rots)
        assert len(at_home) == 7
        assert at_home[4] == pytest.approx(at_home[0])
        assert at_home[5] == pytest.approx(at_home[1])
        assert at_home[0] > at_home[1]           # the ace game is the better one

    def test_a_short_rotation_wraps_on_its_own_length(self):
        """A three-man rotation repeats every three games, not every four."""
        rots = rotations({0: rotation(-1.0, 0.0, 0.5)})
        assert rots.starter(0, 3) == rots.starter(0, 0)
        assert rots.starter(0, 4) == rots.starter(0, 1)


class TestFallback:
    def test_no_rotations_reproduces_the_seeded_result_exactly(self):
        strength = np.array([0.58, 0.47, 0.52, 0.50])
        for wins_needed in (2, 3, 4):
            a = series_win_rate(0, 1, wins_needed, strength, 0.54, None, n=2000)
            b = series_win_rate(0, 1, wins_needed, strength, 0.54,
                                rotations({}), n=2000)
            assert a == b

    def test_a_team_without_a_rotation_keeps_team_strength(self):
        """Only the side that has a rotation moves; the other is untouched."""
        strength = np.array([0.55, 0.50])
        flat = series_game_probs(0, 1, 4, strength, 0.54, None)[0][0]
        # Team 1 has a league-average rotation, team 0 none at all.
        rots = rotations({1: rotation(0.0, 0.0, 0.0, 0.0)})
        at_home, _ = series_game_probs(0, 1, 4, strength, 0.54, rots)
        assert at_home == pytest.approx(np.full(7, flat))

    def test_a_zero_delta_rotation_is_a_no_op(self):
        """The Pythagenpat inversion round-trips, so league-average is identity."""
        for p in (0.42, 0.5, 0.61):
            assert strength_with_starter(p, 0.0, 2 * LG_RA9) == p

    def test_postseason_falls_back_per_series(self):
        strength = np.full(12, 0.5)
        seeds = {103: [0, 1, 2, 3, 4, 5], 104: [6, 7, 8, 9, 10, 11]}
        plain = play_postseason(seeds, np.full(12, 90), strength, 0.54,
                                np.random.default_rng(0))
        empty = play_postseason(seeds, np.full(12, 90), strength, 0.54,
                                np.random.default_rng(0), rotations({}))
        assert plain.champion == empty.champion
        assert plain.pennant == empty.pennant


class TestAceEffect:
    """A pitcher a full run per nine better than league is worth real games."""

    ACE = -1.0

    def test_the_ace_lifts_game_one_above_game_two(self):
        strength = np.array([0.50, 0.50])
        rots = rotations({0: rotation(self.ACE, 0.0, 0.0, 0.0),
                          1: rotation(0.0, 0.0, 0.0, 0.0)})
        at_home, on_road = series_game_probs(0, 1, 4, strength, 0.54, rots)
        assert at_home[0] > at_home[1]
        assert on_road[0] > on_road[1]
        # And game 2, with both sides league average, is the untouched price.
        flat_home = home_win_prob(0.50, 0.50, 0.54)
        assert at_home[1] == pytest.approx(flat_home)

    def test_the_ace_beats_a_flat_rotation_over_a_series(self):
        strength = np.array([0.50, 0.50])
        flat = rotations({0: rotation(0.0, 0.0, 0.0, 0.0),
                          1: rotation(0.0, 0.0, 0.0, 0.0)})
        with_ace = rotations({0: rotation(self.ACE, 0.0, 0.0, 0.0),
                              1: rotation(0.0, 0.0, 0.0, 0.0)})
        p_flat = series_win_rate(0, 1, 4, strength, 0.54, flat)
        p_ace = series_win_rate(0, 1, 4, strength, 0.54, with_ace)
        # The ace starts two of at most seven games; a one-run-per-nine edge
        # over 5.5 innings is worth a few points of series probability, which
        # is far outside the ±0.7 pt standard error of 20,000 series.
        assert p_ace > p_flat + 0.01

    def test_the_effect_is_monotone_in_the_delta(self):
        strength = np.array([0.50, 0.50])
        probs = [series_game_probs(
            0, 1, 4, strength, 0.54,
            rotations({0: rotation(d, 0.0, 0.0, 0.0)}))[0][0]
            for d in (0.75, 0.0, -0.75, -1.5)]
        assert probs == sorted(probs)


class TestSymmetry:
    def test_identical_teams_and_rotations_are_a_coin_flip(self):
        """HFA off (0.5), so the only asymmetry left would be a bug."""
        strength = np.array([0.5, 0.5])
        rot = rotation(-1.0, -0.3, 0.2, 0.6)
        rots = rotations({0: list(rot), 1: [(200 + i, d) for i, (_, d) in enumerate(rot)]})
        for wins_needed in (2, 3, 4):
            p = series_win_rate(0, 1, wins_needed, strength, 0.5, rots)
            # 20,000 series → standard error 0.0035; 4 SE is a generous band.
            assert p == pytest.approx(0.5, abs=0.015)

    def test_a_matchup_table_is_symmetric_under_swapping_the_sides(self):
        strength = np.array([0.54, 0.48])
        a = rotation(-0.8, 0.1, 0.3, 0.0)
        b = [(300 + i, d) for i, d in enumerate((-0.2, -0.6, 0.4, 0.1))]
        at_home, _ = series_game_probs(0, 1, 4, strength, 0.54,
                                       rotations({0: a, 1: b}))
        _, on_road = series_game_probs(1, 0, 4, strength, 0.54,
                                       rotations({0: a, 1: b}))
        # `on_road` from the mirrored call is P(1 wins visiting game k), which
        # is the complement of P(0 wins hosting it).
        assert at_home == pytest.approx(1.0 - on_road)


class TestRunEnvironment:
    def test_a_bigger_run_environment_damps_the_same_delta(self):
        """A run per nine is a smaller share of a high-scoring game."""
        low_env = strength_with_starter(0.5, -1.0, 2 * 3.5)
        high_env = strength_with_starter(0.5, -1.0, 2 * 5.5)
        assert low_env > high_env > 0.5

    def test_per_team_run_env_is_used_when_supplied(self):
        strength = np.array([0.5, 0.5])
        env = np.array([2 * 3.5, 2 * 3.5])
        default = Rotations({0: rotation(-1.0)}, lg_ra9=5.5)
        with_env = Rotations({0: rotation(-1.0)}, lg_ra9=5.5, run_env=env)
        a = series_game_probs(0, 1, 4, strength, 0.54, default)[0][0]
        b = series_game_probs(0, 1, 4, strength, 0.54, with_env)[0][0]
        assert b > a


class TestOddsIntegration:
    def test_run_playoff_odds_threads_rotations_into_the_bracket(self):
        """A league-wide ace for one club must lift its P(WS), same seed."""
        import pandas as pd

        from src.sim.odds import run_playoff_odds
        from src.sim.season import from_schedule

        teams = pd.DataFrame({
            "team_id": range(1, 13),
            "abbrev": [f"T{i}" for i in range(1, 13)],
            "name": [f"Team {i}" for i in range(1, 13)],
            "league_id": [103] * 6 + [104] * 6,
            "division_id": [200, 200, 201, 201, 202, 202,
                            203, 203, 204, 204, 205, 205],
        })
        rows = []
        for i, home in enumerate(range(1, 13)):
            away = 1 + (home % 12)
            rows.append({"game_pk": 9000 + i, "date": "2026-09-20",
                         "game_datetime": "", "status": "Preview",
                         "game_type": "R", "home_id": home, "away_id": away,
                         "home_score": None, "away_score": None})
        state = from_schedule(pd.DataFrame(rows), teams)
        strength = pd.Series({t: 0.5 for t in range(1, 13)})

        plain = run_playoff_odds(state, strength, 0.54, n_sims=400, seed=11)
        rots = Rotations({0: rotation(-2.0, -2.0, -2.0, -2.0)}, lg_ra9=LG_RA9)
        laden = run_playoff_odds(state, strength, 0.54, n_sims=400, seed=11,
                                 rotations=rots)
        p = lambda df: float(df.set_index("abbrev").loc["T1", "p_ws"])  # noqa: E731
        assert p(laden) > p(plain)
