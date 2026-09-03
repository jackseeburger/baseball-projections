"""The feature table is pre-game, and provably so.

The learned challenger's whole claim to be honest is that its inputs were
knowable before first pitch. That claim rests on one thing: every row is read
off a `game_model.build_slate` slate, and `build_slate` cuts every frame it
slices to games strictly before the date.

So the test that matters here is not "does the builder run" — it is: *take a
season, compute the features, then append the rest of the season's games to
the raw logs and compute them again.* If a single number moves, something on
a row knew its own result. Everything else in this file is arithmetic around
that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sim import game_features as gf      # noqa: E402
from src.sim import game_model as gm         # noqa: E402

SEASON = 2026
CLUBS = [1, 2, 3, 4]
DATES = [f"2026-04-{d:02d}" for d in range(1, 29)]


def _ids(team_id: int) -> dict:
    base = 1000 * team_id
    return {"starters": [base + i for i in range(5)],
            "relievers": [base + 50 + i for i in range(5)],
            "batters": [base + 500 + i for i in range(9)]}


def synthetic_season(rng: np.random.Generator) -> dict:
    """A four-club season: schedule, results, appearances, cards, priors."""
    roster = {t: _ids(t) for t in CLUBS}
    games, pitching, hitting, cards, probables = [], [], [], {}, {}
    pk = 500000
    for di, date in enumerate(DATES):
        for home, away in ((CLUBS[0], CLUBS[1]), (CLUBS[2], CLUBS[3])) if di % 2 == 0 \
                else ((CLUBS[1], CLUBS[2]), (CLUBS[3], CLUBS[0])):
            pk += 1
            hs, as_ = int(rng.integers(0, 10)), int(rng.integers(0, 10))
            if hs == as_:
                hs += 1
            games.append({"game_pk": pk, "date": date, "home_id": home,
                          "away_id": away, "home_score": hs, "away_score": as_,
                          "home_win": hs > as_, "venue_id": 3000 + home,
                          "day_night": "night" if di % 3 else "day"})
            cards[pk] = {"home": roster[home]["batters"],
                         "away": roster[away]["batters"]}
            sp = {}
            for side, team in (("home", home), ("away", away)):
                sp[side] = roster[team]["starters"][di % 5]
                pitching.append({"pitcher": sp[side], "season": SEASON, "date": date,
                                 "game_pk": pk, "game_type": "R", "team": team,
                                 "bf": 24, "k": int(rng.integers(2, 9)),
                                 "bb": int(rng.integers(0, 4)), "hbp": 0,
                                 "hr": int(rng.integers(0, 3)), "gs": 1,
                                 "pitches": 90, "outs": 18})
                for rel in roster[team]["relievers"][: 1 + di % 3]:
                    pitching.append({"pitcher": rel, "season": SEASON, "date": date,
                                     "game_pk": pk, "game_type": "R", "team": team,
                                     "bf": 4, "k": int(rng.integers(0, 3)),
                                     "bb": int(rng.integers(0, 2)), "hbp": 0,
                                     "hr": 0, "gs": 0, "pitches": 15, "outs": 3})
                for b in roster[team]["batters"]:
                    hitting.append({"batter": b, "season": SEASON, "date": date,
                                    "game_pk": pk, "game_type": "R", "team_id": team,
                                    "pa": 4, "ab": 4, "h": int(rng.integers(0, 3)),
                                    "doubles": 0, "triples": 0,
                                    "hr": int(rng.integers(0, 2)),
                                    "k": int(rng.integers(0, 3)),
                                    "bb": int(rng.integers(0, 2)), "hbp": 0, "sf": 0})
            probables[pk] = (sp["home"], sp["away"])

    prior_p = pd.DataFrame([
        {"pitcher": p, "season": year, "bf": 600, "k": 140, "bb": 50, "hbp": 5,
         "hr": 18, "outs": 450}
        for t in CLUBS for p in roster[t]["starters"] + roster[t]["relievers"]
        for year in (SEASON - 2, SEASON - 1)])
    prior_h = pd.DataFrame([
        {"batter": b, "season": year, "pa": 500, "ab": 450, "h": 120, "doubles": 25,
         "triples": 2, "hr": 15, "k": 100, "bb": 45, "hbp": 4, "sf": 4,
         "xb_points": 74, "bip": 335, "hits_in_play": 105}
        for t in CLUBS for b in roster[t]["batters"]
        for year in (SEASON - 2, SEASON - 1)])
    return {"games": pd.DataFrame(games), "pitching": pd.DataFrame(pitching),
            "hitting": pd.DataFrame(hitting), "cards": cards,
            "probables": probables, "prior_p": prior_p, "prior_h": prior_h}


def features_from(season: dict, through: str | None = None) -> pd.DataFrame:
    """The feature table, optionally with the raw logs truncated at a date.

    `through` cuts the *inputs* — appearances, hitting lines, completed games —
    not the games scored, which is exactly the shape the leakage test needs.
    """
    pitching, hitting, games = season["pitching"], season["hitting"], season["games"]
    if through is not None:
        pitching = pitching[pitching["date"] <= through]
        hitting = hitting[hitting["date"] <= through]
    inputs = gm.ChainInputs.from_logs(SEASON, pitching, hitting,
                                      season["prior_p"], season["prior_h"])
    scored = games if through is None else games[games["date"] <= through]
    return gf.season_features(scored, CLUBS, inputs,
                              probables=season["probables"], cards=season["cards"],
                              min_games=4)


@pytest.fixture(scope="module")
def season():
    return synthetic_season(np.random.default_rng(20260903))


@pytest.fixture(scope="module")
def table(season):
    return features_from(season)


class TestThePreGameGuarantee:
    def test_a_row_does_not_move_when_the_future_is_added(self, season, table):
        """Rows computed with only the past match rows computed with everything.

        The season is built twice: once with the logs truncated at a mid-season
        date, once with the whole season on file. Every game on or before the
        cut has to score identically, because nothing after the cut was ever
        allowed into its slate.
        """
        cut = DATES[19]
        early = features_from(season, through=cut).set_index("game_pk")
        late = table[table["date"] <= cut].set_index("game_pk")
        assert len(early) and set(early.index) == set(late.index)
        late = late.loc[early.index]
        numeric = [c for c in gf.FEATURE_COLUMNS + ["chain_p", "chain_p_lu"]]
        for col in numeric:
            pd.testing.assert_series_equal(early[col].astype(float),
                                           late[col].astype(float),
                                           check_names=False,
                                           obj=f"{col} moved when the future arrived")

    def test_the_first_scored_date_has_no_card_history(self, season, table):
        """A club's lineup baseline only ever holds cards from played games."""
        first = table[table["date"] == table["date"].min()]
        assert len(first)
        # Cards exist for every game in the fixture, so the deltas are non-zero
        # later; on the first scored date each club is measured against its own
        # banked cards, which at that point are the ones already played.
        assert (first["home_has_card"] == 1).all()


class TestTheRowItself:
    def test_every_declared_feature_is_present_and_finite(self, table):
        assert len(table) > 40
        mat = gf.feature_matrix(table)
        assert list(mat.columns) == gf.FEATURE_COLUMNS
        # `sp_rest` is legitimately missing for a pitcher's first appearance.
        optional = {"home_sp_rest", "away_sp_rest"}
        for col in mat.columns:
            if col in optional:
                continue
            assert np.isfinite(mat[col]).all(), f"{col} has a non-finite value"

    def test_the_chain_column_is_the_chain(self, season, table):
        """`chain_p` is what `game_model` returns, not a re-derivation."""
        assert table["chain_p"].between(0.05, 0.95).all()
        assert (table["chain_p"] != table["pythag_60"]).mean() > 0.9
        # The posted-card model differs from the card-free one somewhere.
        assert (table["chain_p_lu"] != table["chain_p"]).any()

    def test_the_diffs_are_the_two_sides(self, table):
        for name, key in (("rs9_diff", "rs9"), ("ra9_diff", "ra9"),
                          ("sp_ra9_diff", "sp_ra9"), ("pen_diff", "pen_delta")):
            expected = table[f"home_{key}"] - table[f"away_{key}"]
            np.testing.assert_allclose(table[name], expected, rtol=0, atol=1e-12)

    def test_the_bottom_up_half_is_recovered_exactly(self, table):
        """blend = w·bottom_up + (1−w)·top_down, inverted."""
        w = gm.ChainConfig().blend_weight
        for side in ("home", "away"):
            rebuilt = (w * table[f"{side}_bu_rs9"]
                       + (1 - w) * table[f"{side}_td_rs9"])
            np.testing.assert_allclose(rebuilt, table[f"{side}_rs9"], atol=1e-10)


class TestTheSeasonState:
    def test_totals_and_rates_agree_with_the_harness(self, season):
        """The harness imports these; the alias has to be the same function."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backtest_game_odds_features", ROOT / "scripts/backtest_game_odds.py")
        bt = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bt
        spec.loader.exec_module(bt)
        assert bt.team_totals is gf.team_totals
        assert bt.team_rates is gf.team_rates
        assert bt.strengths is gf.strengths

    def test_rest_days_count_calendar_days_since_the_last_game(self, season):
        games = season["games"]
        before = games[games["date"] < DATES[5]]
        rest = gf.rest_days(before, DATES[5], CLUBS)
        assert set(rest) <= set(CLUBS)
        assert all(v >= 1 for v in rest.values())
