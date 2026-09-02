"""Tests for the intra-season (dated cutoff) harness on synthetic PA frames."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval import backtest, score
from src.eval.backtest import COMPONENTS
from src.eval.baselines import (
    SEASON_TO_DATE_BALLAST, league_average, marcel, marcel_preseason,
    previous_season, season_to_date,
)
from src.eval.intraseason import (
    aggregate_pa, assert_split_clean, backtest_intraseason,
    build_training_frame, partial_and_realized, split_at_cutoff,
)

# Flags the PA-outcomes parquet carries, keyed by event, so the synthetic
# frames below match the real schema exactly.
EVENT_FLAGS = {
    "strikeout":     dict(is_k=1),
    "walk":          dict(is_bb=1),
    "hit_by_pitch":  dict(is_hbp=1),
    "single":        dict(is_hit=1, is_single=1),
    "double":        dict(is_hit=1, is_double=1),
    "triple":        dict(is_hit=1, is_triple=1),
    "home_run":      dict(is_hit=1, is_hr=1),
    "field_out":     dict(),
    "sac_fly":       dict(),
    "sac_bunt":      dict(),
    "catcher_interf": dict(),
}
FLAG_COLS = ["is_k", "is_bb", "is_hbp", "is_hit", "is_hr",
             "is_single", "is_double", "is_triple"]


def pa_rows(batter: int, date: str, events: dict[str, int], game_pk: int = 1):
    """Expand {event: count} into PA-level rows for one batter on one date."""
    rows = []
    for event, n in events.items():
        for _ in range(n):
            row = {"batter": batter, "game_pk": game_pk, "game_date": date,
                   "game_year": int(date[:4]), "event": event}
            row.update({c: 0 for c in FLAG_COLS})
            row.update(EVENT_FLAGS[event])
            rows.append(row)
    return rows


class TestAggregation:
    def test_matches_hand_counts(self):
        """One batter, a hand-countable line."""
        pa = pd.DataFrame(pa_rows(1, "2026-04-01", {
            "strikeout": 3, "walk": 2, "hit_by_pitch": 1, "single": 4,
            "double": 2, "triple": 1, "home_run": 2, "field_out": 8,
            "sac_fly": 1, "sac_bunt": 1, "catcher_interf": 1,
        }))
        agg = aggregate_pa(pa).iloc[0]

        assert agg["pa"] == 26           # every row is a plate appearance
        assert agg["k"] == 3
        assert agg["bb"] == 2
        assert agg["hbp"] == 1
        assert agg["h"] == 9             # 4 + 2 + 1 + 2
        assert agg["hr"] == 2
        assert agg["sf"] == 1 and agg["sh"] == 1 and agg["ci"] == 1
        # AB = PA − BB − HBP − SF − SH − interference
        assert agg["ab"] == 26 - 2 - 1 - 1 - 1 - 1 == 20
        assert agg["xb_points"] == 2 + 2 * 1 + 3 * 2 == 10
        assert agg["bip"] == 20 - 3 - 2 + 1 == 16
        assert agg["hits_in_play"] == 9 - 2 == 7
        assert agg["season"] == 2026
        assert agg["games"] == 1

    def test_counts_do_not_overflow_int8(self):
        """The outcome flags are int8 on disk; 3*hr silently wraps if kept."""
        pa = pd.DataFrame(pa_rows(1, "2026-04-01", {"home_run": 60}))
        assert aggregate_pa(pa).iloc[0]["xb_points"] == 180

    def test_multiple_batters_and_games(self):
        rows = (pa_rows(1, "2026-04-01", {"strikeout": 2}, game_pk=1)
                + pa_rows(1, "2026-04-02", {"single": 1}, game_pk=2)
                + pa_rows(2, "2026-04-01", {"walk": 3}, game_pk=1))
        agg = aggregate_pa(pd.DataFrame(rows)).set_index("batter")
        assert agg.loc[1, "pa"] == 3 and agg.loc[1, "games"] == 2
        assert agg.loc[2, "pa"] == 3 and agg.loc[2, "bb"] == 3
        assert str(agg.loc[1, "first_game_date"].date()) == "2026-04-01"
        assert str(agg.loc[1, "last_game_date"].date()) == "2026-04-02"

    def test_empty_frame_returns_schema(self):
        out = aggregate_pa(pd.DataFrame(columns=["batter"]))
        assert out.empty and "hits_in_play" in out.columns


class TestSplit:
    def test_cutoff_day_belongs_to_the_realized_side(self):
        pa = pd.DataFrame(
            pa_rows(1, "2026-06-30", {"strikeout": 1})
            + pa_rows(1, "2026-07-01", {"strikeout": 1})
            + pa_rows(1, "2026-07-02", {"strikeout": 1})
        )
        before, after = split_at_cutoff(pa, "2026-07-01")
        assert len(before) == 1 and len(after) == 2

    def test_partial_and_realized_flags(self):
        pa = pd.DataFrame(pa_rows(1, "2026-06-01", {"strikeout": 5})
                          + pa_rows(1, "2026-08-01", {"walk": 5}))
        partial, realized = partial_and_realized(pa, "2026-07-01", 2026)
        assert bool(partial.iloc[0]["partial"]) is True
        assert bool(realized.iloc[0]["partial"]) is False
        assert partial.iloc[0]["k"] == 5 and realized.iloc[0]["bb"] == 5


class TestLeakageGuard:
    @pytest.fixture
    def clean(self):
        pa = pd.DataFrame(pa_rows(1, "2026-06-01", {"strikeout": 5})
                          + pa_rows(1, "2026-08-01", {"strikeout": 5}))
        return partial_and_realized(pa, "2026-07-01", 2026)

    def test_clean_split_passes(self, clean):
        partial, realized = clean
        assert_split_clean(partial, realized, "2026-07-01", 2026)

    def test_realized_row_before_cutoff_raises(self, clean):
        partial, realized = clean
        realized.loc[0, "first_game_date"] = pd.Timestamp("2026-06-15")
        with pytest.raises(ValueError, match="realized row"):
            assert_split_clean(partial, realized, "2026-07-01", 2026)

    def test_training_row_after_cutoff_raises(self, clean):
        partial, realized = clean
        partial.loc[0, "last_game_date"] = pd.Timestamp("2026-07-15")
        with pytest.raises(ValueError, match="training row"):
            assert_split_clean(partial, realized, "2026-07-01", 2026)

    def test_training_season_after_predict_year_raises(self, clean):
        partial, realized = clean
        partial.loc[0, "season"] = 2027
        partial.loc[0, "last_game_date"] = pd.Timestamp("2027-04-01")
        with pytest.raises(ValueError, match="> predict year"):
            assert_split_clean(partial, realized, "2026-07-01", 2026)

    def test_realized_wrong_season_raises(self, clean):
        partial, realized = clean
        realized.loc[0, "season"] = 2025
        with pytest.raises(ValueError, match="must be entirely season"):
            assert_split_clean(partial, realized, "2026-07-01", 2026)

    def test_build_training_frame_drops_full_predict_year(self):
        seasons = pd.DataFrame([
            {"batter": 1, "season": 2025, "pa": 600, "k": 120, "age": 27},
            {"batter": 1, "season": 2026, "pa": 600, "k": 200, "age": 28},
        ])
        partial = pd.DataFrame([{"batter": 1, "season": 2026, "pa": 200,
                                 "k": 40, "partial": True}])
        train = build_training_frame(seasons, partial, 2026)
        # The season table's own 2026 row (the full year) must not survive.
        assert len(train) == 2
        assert train[train.season == 2026]["k"].tolist() == [40]
        assert train[train.season == 2026]["age"].tolist() == [28]


def _train_frame(partial_pa: int, partial_k_rate: float = 0.50) -> pd.DataFrame:
    """Two full seasons at .20 K% plus a partial 2026 at `partial_k_rate`.

    A second batter holds the league rate at .20 so regression targets are
    stable across the partial sizes being compared.
    """
    rows = []
    for season in (2024, 2025):
        for batter in (1, 2):
            rows.append({"batter": batter, "season": season, "pa": 600,
                         "k": 120, "age": 27, "partial": False})
    rows.append({"batter": 1, "season": 2026, "pa": partial_pa,
                 "k": round(partial_pa * partial_k_rate), "age": 28,
                 "partial": True})
    rows.append({"batter": 2, "season": 2026, "pa": 600, "k": 120,
                 "age": 28, "partial": True})
    return pd.DataFrame(rows)


class TestBaselinesWithPartialSeasons:
    spec = COMPONENTS["k_rate"]

    def test_marcel_weights_the_partial_season_by_pa(self):
        small = marcel(_train_frame(50), self.spec, 2026).set_index("batter")
        big = marcel(_train_frame(500), self.spec, 2026).set_index("batter")
        control = marcel_preseason(_train_frame(50), self.spec, 2026).set_index("batter")
        # A hot partial season pulls the projection up, and 500 PA of it pulls
        # ten times as hard as 50 — that is Marcel weighting by trials.
        assert control.loc[1, "predicted"] < small.loc[1, "predicted"]
        assert small.loc[1, "predicted"] < big.loc[1, "predicted"]

    def test_marcel_preseason_ignores_the_partial_season(self):
        hot = marcel_preseason(_train_frame(500), self.spec, 2026)
        cold = marcel_preseason(_train_frame(500, partial_k_rate=0.05),
                                self.spec, 2026)
        pd.testing.assert_frame_equal(hot, cold)

    def test_previous_season_uses_the_last_full_season(self):
        pred = previous_season(_train_frame(500), self.spec, 2026).set_index("batter")
        assert pred.loc[1, "predicted"] == pytest.approx(0.20)  # 2025, not 2026

    def test_league_average_is_the_rate_through_the_cutoff(self):
        # Batter 1 at .50 over 500 PA, batter 2 at .20 over 600 PA.
        pred = league_average(_train_frame(500), self.spec, 2026)
        expected = (250 + 120) / (500 + 600)
        assert pred["predicted"].nunique() == 1
        assert pred["predicted"].iloc[0] == pytest.approx(expected)

    def test_season_to_date_regresses_with_the_component_ballast(self):
        train = _train_frame(500)
        pred = season_to_date(train, self.spec, 2026).set_index("batter")
        league = (250 + 120) / 1100
        b = SEASON_TO_DATE_BALLAST["k_rate"]
        assert pred.loc[1, "predicted"] == pytest.approx(
            (250 + b * league) / (500 + b))

    def test_season_to_date_with_zero_pa_is_league_average(self):
        train = _train_frame(500)
        train = pd.concat([train, pd.DataFrame([
            {"batter": 3, "season": 2026, "pa": 0, "k": 0,
             "age": 28, "partial": True}])], ignore_index=True)
        pred = season_to_date(train, self.spec, 2026).set_index("batter")
        league = (250 + 120) / 1100
        assert pred.loc[3, "predicted"] == pytest.approx(league)


class TestBacktestWithCutoff:
    """A synthetic 2026 season plus two prior seasons, cut at a date."""

    @pytest.fixture
    def pa(self):
        """Three batters, 40 PA in each of April..August 2026.

        K rates by batter: .15 / .25 / .35, identical before and after any
        month boundary, so a cutoff never changes the truth being scored.
        """
        rows = []
        for month in range(4, 9):
            for batter, k in [(1, 6), (2, 10), (3, 14)]:
                rows += pa_rows(
                    batter, f"2026-0{month}-10",
                    {"strikeout": k, "field_out": 40 - k}, game_pk=month,
                )
        return pd.DataFrame(rows)

    @pytest.fixture
    def seasons(self):
        rows = []
        for season in (2024, 2025):
            for batter, rate in [(1, 0.15), (2, 0.25), (3, 0.35)]:
                rows.append({"batter": batter, "season": season, "age": 27,
                             "pa": 600, "k": int(round(rate * 600))})
        return pd.DataFrame(rows)

    def test_end_to_end(self, pa, seasons):
        results = backtest("k_rate", cutoff_date="2026-07-01", seasons=seasons,
                           pa_frame=pa, min_trials=50)
        assert set(results["model"]) == {
            "marcel", "marcel_preseason", "previous_season",
            "league_average", "season_to_date"}
        # Two months of 40 PA are scored, not the whole season.
        assert results["trials"].max() == 80
        s = score(results).set_index("model")
        assert s.loc["previous_season", "mae"] < s.loc["league_average", "mae"]

    def test_cutoff_before_first_game_matches_the_season_level_split(
            self, pa, seasons
    ):
        """With nothing banked in-season the harness must reduce to the
        season-level backtest on the same numbers."""
        season_2026 = aggregate_pa(pa, 2026)
        full = pd.concat([seasons, season_2026], ignore_index=True)
        plain = backtest("k_rate", 2025, 2026, seasons=full, min_trials=50)
        cut = backtest("k_rate", cutoff_date="2026-01-01", seasons=seasons,
                       pa_frame=pa, min_trials=50,
                       providers={"marcel": marcel,
                                  "previous_season": previous_season,
                                  "league_average": league_average})
        cols = ["model", "batter", "predicted", "realized_successes",
                "realized_rate", "trials"]
        pd.testing.assert_frame_equal(
            plain[cols].sort_values(["model", "batter"]).reset_index(drop=True),
            cut[cols].sort_values(["model", "batter"]).reset_index(drop=True),
        )

    def test_cutoff_after_last_game_leaves_nothing_to_score(self, pa, seasons):
        with pytest.raises(ValueError, match="realized frame is empty"):
            backtest("k_rate", cutoff_date="2026-12-01", seasons=seasons,
                     pa_frame=pa, min_trials=50)

    def test_later_cutoff_scores_fewer_trials(self, pa, seasons):
        trials = {}
        for cutoff in ["2026-05-01", "2026-07-01", "2026-08-01"]:
            r = backtest_intraseason("k_rate", cutoff, seasons=seasons,
                                     pa_frame=pa, min_trials=1)
            trials[cutoff] = int(r[r.model == "marcel"]["trials"].sum())
        assert trials["2026-05-01"] > trials["2026-07-01"] > trials["2026-08-01"]

    def test_pa_frame_required_with_a_cutoff(self, seasons):
        with pytest.raises(ValueError, match="pa_frame"):
            backtest("k_rate", cutoff_date="2026-07-01", seasons=seasons)

    def test_pa_frame_rejected_without_a_cutoff(self, pa, seasons):
        with pytest.raises(ValueError, match="only meaningful with cutoff_date"):
            backtest("k_rate", 2025, seasons=seasons, pa_frame=pa)

    def test_providers_never_see_post_cutoff_data(self, pa, seasons):
        seen = {}

        def spy(train, spec, predict_year):
            seen["max_date"] = train["last_game_date"].max()
            seen["max_season"] = int(train["season"].max())
            return previous_season(train, spec, predict_year)

        backtest("k_rate", cutoff_date="2026-07-01", seasons=seasons,
                 pa_frame=pa, min_trials=50, providers={"spy": spy})
        assert seen["max_season"] == 2026
        assert seen["max_date"] < pd.Timestamp("2026-07-01")
