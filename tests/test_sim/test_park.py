"""Park factors: the arithmetic, the centring, and the two leakage guards.

The term is a multiplier on a run environment, so the things that can go wrong
with it are the things that can go wrong with any multiplier: it can be built
from the wrong games (a game on or after the date being predicted), it can be
built the wrong way round (a hitter's park scoring below 1), it can quietly
move the league instead of redistributing it, and it can fail to switch off
when it is meant to be off. One test each, on synthetic games where the right
answer is known by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sim import park  # noqa: E402
from src.sim.strength import pythagenpat  # noqa: E402

HITTERS_PARK, PITCHERS_PARK, NEUTRAL = 4001, 4002, 4003


def synthetic_games(runs_at_hitters_park: float = 12.0,
                    runs_elsewhere: float = 8.0,
                    season: str = "2024") -> pd.DataFrame:
    """A four-club league where exactly one park inflates runs and nothing else.

    Clubs 1 and 2 host at `HITTERS_PARK` and `PITCHERS_PARK`; every club is
    equally good, so any factor that comes out is the park and only the park.
    """
    rows, pk = [], 0
    hosts = {1: HITTERS_PARK, 2: PITCHERS_PARK, 3: NEUTRAL, 4: NEUTRAL}
    for home in (1, 2, 3, 4):
        for away in (1, 2, 3, 4):
            if home == away:
                continue
            for i in range(10):
                pk += 1
                venue = hosts[home]
                runs = (runs_at_hitters_park if venue == HITTERS_PARK
                        else 6.0 if venue == PITCHERS_PARK else runs_elsewhere)
                rows.append({"game_pk": pk, "date": f"{season}-05-{i % 28 + 1:02d}",
                             "status": "Final", "game_type": "R",
                             "venue_id": venue, "home_id": home, "away_id": away,
                             "home_score": runs / 2, "away_score": runs / 2})
    return pd.DataFrame(rows)


class TestTheFactorItself:
    def test_a_hitters_park_scores_above_one_and_a_pitchers_park_below(self):
        games = park.completed_venue_games(synthetic_games())
        f = park.run_factors(games, ballast=0.0)
        assert f[HITTERS_PARK] > 1.05
        assert f[PITCHERS_PARK] < 0.95
        # The third park sits between them, not at exactly 1: its home clubs'
        # road mix contains the other two parks, which is the known bias of the
        # home/road estimator and is why the factors are centred on the league
        # afterwards rather than assumed to be centred already.
        assert f[PITCHERS_PARK] < f[NEUTRAL] < f[HITTERS_PARK]

    def test_the_league_mean_factor_is_exactly_one(self):
        """Park redistributes the league's runs; it never moves the league."""
        games = park.completed_venue_games(synthetic_games())
        f = park.run_factors(games, ballast=0.0)
        n = games.groupby("venue_id").size()
        weighted = sum(n[v] * f[v] for v in f) / n.sum()
        assert weighted == pytest.approx(1.0, abs=1e-12)

    def test_ballast_pulls_the_factor_toward_one_monotonically(self):
        games = park.completed_venue_games(synthetic_games())
        seen = [park.run_factors(games, ballast=b)[HITTERS_PARK]
                for b in (0.0, 100.0, 400.0, 1600.0)]
        assert seen == sorted(seen, reverse=True)
        assert seen[-1] == pytest.approx(1.0, abs=0.02)

    def test_an_infinite_ballast_is_no_park_term_at_all(self):
        """The setting that makes the sweep a clean nesting."""
        games = park.completed_venue_games(synthetic_games())
        f = park.run_factors(games, ballast=float("inf"))
        assert set(f.values()) == {1.0}

    def test_a_neutral_site_nobody_hosts_twice_is_left_alone(self):
        """One game at a venue is not evidence about a park."""
        games = synthetic_games()
        odd = games.iloc[[0]].copy()
        odd["game_pk"] = 99999
        odd["venue_id"] = 5150
        odd["home_score"] = 20.0
        odd["away_score"] = 20.0
        pooled = park.completed_venue_games(pd.concat([games, odd]))
        f = park.run_factors(pooled, ballast=park.BALLAST_GAMES)
        assert f[5150] == pytest.approx(1.0, abs=0.05)

    def test_a_schedule_with_no_venue_column_falls_back_to_the_home_club(self):
        games = synthetic_games().drop(columns=["venue_id"])
        out = park.completed_venue_games(games)
        assert (out["venue_id"] == out["home_id"]).all()


class TestExposureAndNeutralisation:
    def test_exposure_is_the_mean_park_of_the_games_a_club_played(self):
        games = park.completed_venue_games(synthetic_games())
        f = park.run_factors(games, ballast=0.0)
        e = park.team_exposure(games, f, [1, 2, 3, 4])
        # Club 1 hosts the hitters' park, club 2 the pitchers' park.
        assert e[1] > e[3] > e[2]
        assert float((e * 1.0).mean()) == pytest.approx(1.0, abs=1e-9)

    def test_no_exposure_reproduces_the_plain_regression_exactly(self):
        """`park_ballast = inf` has to leave the production rates untouched."""
        rs = pd.Series({1: 700.0, 2: 600.0}, name="rs")
        ra = pd.Series({1: 650.0, 2: 620.0}, name="ra")
        g = pd.Series({1: 140.0, 2: 140.0}, name="g")
        out = park.neutral_run_rates(rs, ra, g, 60.0, exposure=None)
        lg_rs = rs.sum() / g.sum()
        expected = (rs[1] + 60.0 * lg_rs) / (g[1] + 60.0)
        assert out.loc[1, "rs_pg"] == pytest.approx(expected, abs=1e-12)
        ones = pd.Series(1.0, index=rs.index)
        same = park.neutral_run_rates(rs, ra, g, 60.0, exposure=ones)
        assert np.allclose(out.to_numpy(), same.to_numpy())

    def test_neutralising_divides_the_totals_not_the_finished_rate(self):
        """Dividing afterwards would divide the league ballast too.

        A club in a 10% hitters' park should come back at its own runs over
        1.10, regressed toward the *league* — not at the whole regressed
        number over 1.10, which would drag it below the league it was
        regressed toward.
        """
        rs = pd.Series({1: 700.0, 2: 700.0})
        ra = pd.Series({1: 700.0, 2: 700.0})
        g = pd.Series({1: 140.0, 2: 140.0})
        e = pd.Series({1: 1.10, 2: 0.90})
        out = park.neutral_run_rates(rs, ra, g, 60.0, exposure=e)
        lg = float(rs.sum() / g.sum())
        naive = ((rs[1] + 60.0 * lg) / (g[1] + 60.0)) / 1.10
        assert out.loc[1, "rs_pg"] > naive
        assert out.loc[1, "rs_pg"] == pytest.approx(
            (rs[1] / 1.10 + 60.0 * lg) / (g[1] + 60.0), abs=1e-12)

    def test_a_club_with_no_games_yet_is_left_at_one(self):
        empty = park.completed_venue_games(pd.DataFrame())
        e = park.team_exposure(empty, {HITTERS_PARK: 1.2}, [1, 2])
        assert list(e) == [1.0, 1.0]


class TestTheGameItself:
    def test_the_park_scales_both_rates_and_leaves_the_ratio_alone(self):
        rs, ra = park.apply_factor(5.0, 4.0, 1.10)
        assert rs / ra == pytest.approx(5.0 / 4.0, abs=1e-12)
        assert rs == pytest.approx(5.5, abs=1e-12)

    def test_the_same_ratio_in_a_bigger_park_is_a_more_extreme_win_probability(self):
        """Why park belongs in a win probability at all.

        Pythagenpat's exponent rises with the run environment, so a club that
        outscores its opponent by 25% wins more often in a high-scoring park —
        the run difference is bigger against the same noise.
        """
        low = pythagenpat(5.0, 4.0, 1.0)
        rs, ra = park.apply_factor(5.0, 4.0, 1.15)
        assert pythagenpat(rs, ra, 1.0) > low

    def test_a_venue_nobody_priced_is_a_multiplier_of_one(self):
        assert park.factor(None, {1: 1.2}) == 1.0
        assert park.factor(99, {1: 1.2}) == 1.0
        assert park.factor(1, {}) == 1.0


class TestLeakage:
    def test_the_factors_are_built_from_prior_seasons_only(self, monkeypatch):
        """No game of the season being scored may reach a park factor."""
        asked = []

        def fake_schedule(start, end):
            asked.append(start[:4])
            return synthetic_games(season=start[:4])

        import src.data.mlb_stats_api as api
        monkeypatch.setattr(api, "fetch_schedule", fake_schedule)
        park.fetch_prior_factors(2026, prior_seasons=2)
        assert asked == ["2024", "2025"]
        assert "2026" not in asked

    def test_exposure_ignores_games_on_or_after_the_date(self):
        """The cut the caller makes: a game today cannot price today's park.

        A club sent to a 20-run park on the target date and every date after it
        has exactly the exposure it had before — the frame is filtered strictly
        before, so tonight's park cannot inform tonight's neutralisation.
        """
        games = park.completed_venue_games(synthetic_games())
        as_of = "2024-05-20"
        before = games[games["date"] < as_of]
        f = park.run_factors(games, ballast=0.0)
        base = park.team_exposure(before, f, [1, 2, 3, 4])

        future = games.iloc[:10].copy()
        future["date"] = as_of
        future["venue_id"] = HITTERS_PARK
        future["home_id"] = 3
        with_future = pd.concat([games, future], ignore_index=True)
        after = park.team_exposure(
            with_future[with_future["date"] < as_of], f, [1, 2, 3, 4])
        assert np.allclose(base.to_numpy(), after.to_numpy())
