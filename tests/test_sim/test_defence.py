"""Team defence: the balls-in-play arithmetic, the road filter, the leakage cut.

The term reads the same kind of thing it predicts — hits on balls in play —
so the guard that matters most is the date cut: a club's defence tonight may
not be informed by the balls it fields tonight, and the same cut has to catch
the second game of a doubleheader reading the first. The rest is arithmetic
with a known answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sim import defence  # noqa: E402

HOME_BY_GAME = {1: 100, 2: 200, 3: 100, 4: 200}


def log(pitcher, team, date, game_pk, ab=30, k=8, hr=1, sf=0, h=9, outs=27.0):
    return {"pitcher": pitcher, "season": 2026, "date": date,
            "game_pk": game_pk, "game_type": "R", "team": team, "outs": outs,
            "bf": ab + 4, "k": k, "bb": 3, "hbp": 1, "hr": hr, "er": 4,
            "gs": 1, "pitches": 95, "h": h, "ab": ab, "sf": sf}


class TestTheCounts:
    def test_balls_in_play_and_hits_on_them_are_the_fip_complement(self):
        logs = pd.DataFrame([log(1, 100, "2026-05-01", 2, ab=30, k=8, hr=1,
                                 sf=1, h=9)])
        out = defence.bip_counts(logs, HOME_BY_GAME)
        assert len(out) == 1
        assert out.loc[0, "bip"] == pytest.approx(30 - 8 - 1 + 1)
        assert out.loc[0, "hits_bip"] == pytest.approx(9 - 1)

    def test_home_games_are_dropped_because_that_is_the_park_not_the_defence(self):
        logs = pd.DataFrame([
            log(1, 100, "2026-05-01", 1),   # club 100 at home
            log(1, 100, "2026-05-02", 2),   # club 100 on the road
        ])
        out = defence.bip_counts(logs, HOME_BY_GAME)
        assert list(out["date"]) == ["2026-05-02"]
        both = defence.bip_counts(logs, HOME_BY_GAME, road_only=False)
        assert len(both) == 2

    def test_two_pitchers_in_one_road_game_are_one_club_line(self):
        logs = pd.DataFrame([log(1, 100, "2026-05-02", 2, ab=20, k=5, h=6),
                             log(2, 100, "2026-05-02", 2, ab=10, k=3, h=3)])
        out = defence.bip_counts(logs, HOME_BY_GAME)
        assert len(out) == 1
        assert out.loc[0, "bip"] == pytest.approx((20 - 5 - 1) + (10 - 3 - 1))

    def test_a_game_whose_park_is_unknown_is_dropped_not_guessed(self):
        logs = pd.DataFrame([log(1, 100, "2026-05-02", 77)])
        assert len(defence.bip_counts(logs, HOME_BY_GAME)) == 0


class TestTheEstimate:
    def _counts(self):
        # Two clubs, same balls in play, one fielding 20 points better.
        rows = []
        for i in range(50):
            rows.append({"team": 100, "date": f"2026-05-{i % 28 + 1:02d}",
                         "bip": 24.0, "hits_bip": 6.6, "outs": 27.0})
            rows.append({"team": 200, "date": f"2026-05-{i % 28 + 1:02d}",
                         "bip": 24.0, "hits_bip": 7.4, "outs": 27.0})
        return pd.DataFrame(rows)

    def test_the_better_defence_gets_a_negative_runs_allowed_delta(self):
        deltas, diag = defence.team_defence(self._counts(), "2026-09-01",
                                            ballast=0.0)
        assert deltas[100] < 0 < deltas[200]
        assert diag["lg_babip"] == pytest.approx(7.0 / 24.0, abs=1e-9)

    def test_the_deltas_sum_to_zero_across_a_league_of_equal_samples(self):
        deltas, _ = defence.team_defence(self._counts(), "2026-09-01",
                                         ballast=0.0)
        assert sum(deltas.values()) == pytest.approx(0.0, abs=1e-9)

    def test_the_ballast_shrinks_the_delta_monotonically(self):
        seen = [abs(defence.team_defence(self._counts(), "2026-09-01",
                                         ballast=b)[0][100])
                for b in (0.0, 1000.0, 4000.0, 16000.0)]
        assert seen == sorted(seen, reverse=True)

    def test_an_infinite_ballast_is_no_defence_term_at_all(self):
        deltas, _ = defence.team_defence(self._counts(), "2026-09-01",
                                         ballast=float("inf"))
        assert set(deltas.values()) == {0.0}

    def test_the_size_is_hits_times_the_run_value_of_a_hit(self):
        """0.020 of BABIP over ~24 balls in play a game is about 0.36 runs."""
        deltas, diag = defence.team_defence(self._counts(), "2026-09-01",
                                            ballast=0.0)
        expected = (6.6 / 24.0 - 7.0 / 24.0) * diag["bip_per9"] * \
            defence.RUNS_PER_BIP_HIT
        assert deltas[100] == pytest.approx(expected, abs=1e-9)

    def test_a_club_with_no_balls_in_play_keeps_its_component_rate(self):
        ra9 = pd.Series({100: 4.5, 200: 4.5, 300: 4.5})
        out = defence.apply_deltas(ra9, {100: -0.2, 200: 0.2})
        assert out[300] == pytest.approx(4.5)
        assert out[100] == pytest.approx(4.3)


class TestLeakage:
    def _logs(self):
        rows = [log(1, 100, f"2026-05-{d:02d}", 2, ab=30, k=8, hr=1, h=9)
                for d in range(1, 20)]
        return pd.DataFrame(rows)

    def test_a_game_on_the_date_cannot_inform_the_date(self):
        base = defence.bip_counts(self._logs(), HOME_BY_GAME)
        today = pd.DataFrame([log(1, 100, "2026-05-20", 2, ab=40, k=0, hr=0,
                                  h=40)])
        with_today = defence.bip_counts(
            pd.concat([self._logs(), today], ignore_index=True), HOME_BY_GAME)
        before = defence.team_defence(base, "2026-05-20", ballast=0.0)[0]
        after = defence.team_defence(with_today, "2026-05-20", ballast=0.0)[0]
        assert before == after

    def test_tomorrows_game_cannot_either(self):
        base = defence.bip_counts(self._logs(), HOME_BY_GAME)
        future = pd.DataFrame([log(1, 100, "2026-06-01", 2, ab=40, k=0, hr=0,
                                   h=40)])
        with_future = defence.bip_counts(
            pd.concat([self._logs(), future], ignore_index=True), HOME_BY_GAME)
        assert (defence.team_defence(base, "2026-05-20", ballast=0.0)[0]
                == defence.team_defence(with_future, "2026-05-20",
                                        ballast=0.0)[0])

    def test_the_first_game_of_a_doubleheader_does_not_reach_the_second(self):
        """Both games carry the same date, so the strict cut excludes both."""
        logs = pd.concat([self._logs(),
                          pd.DataFrame([log(1, 100, "2026-05-19", 4, ab=40,
                                            k=0, hr=0, h=40)])],
                         ignore_index=True)
        counts = defence.bip_counts(logs, HOME_BY_GAME)
        assert defence.counts_before(counts, "2026-05-19")["bip"].sum() == \
            defence.counts_before(
                defence.bip_counts(self._logs(), HOME_BY_GAME),
                "2026-05-19")["bip"].sum()

    def test_no_history_at_all_is_no_term(self):
        empty = pd.DataFrame(columns=defence.COUNT_COLS)
        assert defence.team_defence(empty, "2026-05-01")[0] == {}
