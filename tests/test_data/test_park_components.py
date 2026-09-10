"""The per-component park factors (BAS-86, docs/park-factors.md).

Nothing here needs pymc, a network or the real PA parquets: the factors are
arithmetic over a counts table, so the tests build a synthetic league with a
park effect put in on purpose and check that the same number comes back out.

The one test that reads the committed artifact skips when it is absent, so a
checkout without `data/features/park_factors.parquet` still passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import park_components as pk  # noqa: E402

TEAMS = [f"T{i:02d}" for i in range(8)]


def synthetic_pa(seasons=(2021, 2022, 2023), hr_boost=None, pa_per_pair=500,
                 seed=0) -> pd.DataFrame:
    """A round-robin league: every club hosts every other club, equally often.

    `hr_boost` is {park: multiplier} applied to the HR probability of *every*
    PA in that park, which is exactly what a park factor claims to measure.

    The recovered factor is close to the injected multiplier but not equal to
    it, and the gap is a property of the method rather than of this fixture:
    every rival's "elsewhere" pool contains the extreme park (a 1/(n-1) share
    of their road schedule), so an extreme park drags its own comparison
    baseline toward itself and the split comes back amplified — about +7% on a
    1.5x park in this eight-club league, and proportionally less in a real
    thirty-club one. The assertions below bracket the effect rather than
    pinning it, for that reason.
    """
    rng = np.random.default_rng(seed)
    hr_boost = hr_boost or {}
    rows = []
    for season in seasons:
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                for bat_team, topbot in ((away, "Top"), (home, "Bot")):
                    n = pa_per_pair
                    p_hr = 0.04 * float(hr_boost.get(home, 1.0))
                    is_hr = (rng.random(n) < p_hr).astype(int)
                    is_k = (rng.random(n) < 0.22).astype(int)
                    is_bb = np.where(is_k + is_hr > 0, 0,
                                     (rng.random(n) < 0.09).astype(int))
                    hit = np.where(is_hr == 1, 1,
                                   np.where(is_k + is_bb > 0, 0,
                                            (rng.random(n) < 0.30).astype(int)))
                    rows.append(pd.DataFrame({
                        "game_year": season, "home_team": home,
                        "away_team": away, "inning_topbot": topbot,
                        "event": np.where(is_hr == 1, "home_run", "field_out"),
                        "is_k": is_k, "is_bb": is_bb, "is_hbp": 0,
                        "is_hit": hit, "is_hr": is_hr,
                        "is_double": 0, "is_triple": 0,
                        "_bat_team": bat_team,
                    }))
    return pd.concat(rows, ignore_index=True)


class TestCounts:
    def test_denominators_match_the_harness_season_identities(self):
        """`pa_counts`'s per-PA AB and BIP flags have to sum to the same
        totals `src.eval.intraseason.aggregate_pa` computes from season
        counts, or the ISO and BABIP factors are measured on a denominator
        the harness does not score."""
        from src.eval.intraseason import aggregate_pa

        pa = synthetic_pa(seasons=(2021,))
        pa = pa.assign(batter=1, game_pk=1, game_date="2021-05-01")
        season = aggregate_pa(pa, 2021)
        counts = pk.pa_counts(pa)
        assert counts["k_rate_d"].sum() == int(season["pa"].sum())
        assert counts["iso_d"].sum() == int(season["ab"].sum())
        assert counts["babip_d"].sum() == int(season["bip"].sum())
        assert counts["babip_n"].sum() == int(season["hits_in_play"].sum())
        assert counts["iso_n"].sum() == int(season["xb_points"].sum())
        assert counts["k_rate_n"].sum() == int(season["k"].sum())

    def test_batting_team_is_read_off_the_half_inning(self):
        pa = synthetic_pa(seasons=(2021,), pa_per_pair=5)
        counts = pk.pa_counts(pa)
        merged = pa.groupby(["game_year", "home_team", "_bat_team"]).size()
        counts = counts.set_index(["game_year", "park", "bat_team"])["k_rate_d"]
        for key, n in merged.items():
            assert counts.loc[key] == n


class TestFactors:
    @pytest.mark.parametrize("seed", [0, 3, 6])
    def test_an_injected_park_effect_comes_back(self, seed):
        """Three seeds, because one seed passing is a fact about that seed."""
        pa = synthetic_pa(hr_boost={"T00": 1.5, "T01": 0.6}, seed=seed)
        out = pk.regress(pk.window_raw(pk.pa_counts(pa), (2021, 2022, 2023)), 0.0)
        hr = out[out["component"] == "hr_rate"].set_index("park")["factor"]
        neutral = hr[[t for t in TEAMS if t not in ("T00", "T01")]]
        assert 1.4 < hr["T00"] / neutral.mean() < 1.8
        assert 0.48 < hr["T01"] / neutral.mean() < 0.70
        assert neutral.between(0.85, 1.15).all()

    def test_both_halves_see_the_same_effect(self):
        """A park effect is a fact about the park, so the home club's split
        and the visitors' split have to agree. They are averaged into `raw`,
        and a build where only one half moved would be measuring the tenant."""
        pa = synthetic_pa(hr_boost={"T00": 1.5})
        raw = pk.window_raw(pk.pa_counts(pa), (2021, 2022, 2023))
        row = raw[(raw["component"] == "hr_rate") & (raw["park"] == "T00")].iloc[0]
        assert 1.35 < row["home_ratio"] < 1.9
        assert 1.35 < row["visitor_ratio"] < 1.9
        assert row["home_ratio"] == pytest.approx(row["visitor_ratio"], rel=0.20)

    def test_infinite_ballast_is_exactly_no_park_term(self):
        raw = pk.window_raw(pk.pa_counts(synthetic_pa(hr_boost={"T00": 1.5})),
                            (2021, 2022, 2023))
        out = pk.regress(raw, float("inf"))
        assert (out["factor"] == 1.0).all()

    def test_ballast_shrinks_toward_one_monotonically(self):
        raw = pk.window_raw(pk.pa_counts(synthetic_pa(hr_boost={"T00": 1.5})),
                            (2021, 2022, 2023))
        spread = []
        for b in (0.0, 4000.0, 16000.0, 64000.0):
            out = pk.regress(raw, b)
            hr = out[out["component"] == "hr_rate"]["factor"]
            spread.append(float(np.abs(np.log(hr)).max()))
        assert spread == sorted(spread, reverse=True)

    def test_factors_are_centred_on_the_league(self):
        raw = pk.window_raw(pk.pa_counts(synthetic_pa(hr_boost={"T00": 1.5})),
                            (2021, 2022, 2023))
        out = pk.regress(raw, pk.FROZEN_BALLAST)
        for _, g in out.groupby("component"):
            assert float(np.average(g["factor"], weights=g["n"])) == pytest.approx(1.0)

    def test_zero_ballast_keeps_the_raw_ordering(self):
        raw = pk.window_raw(pk.pa_counts(synthetic_pa(hr_boost={"T00": 1.5})),
                            (2021, 2022, 2023))
        out = pk.regress(raw, 0.0)
        hr = out[out["component"] == "hr_rate"]
        assert (hr.sort_values("raw")["factor"].to_numpy()
                == pytest.approx(np.sort(hr["factor"].to_numpy())))


class TestWalkForward:
    def test_the_window_is_strictly_before_the_stamped_season(self):
        have = [2019, 2020, 2021, 2022, 2023, 2024]
        assert pk.window_for(2024, have) == (2021, 2022, 2023)
        assert pk.window_for(2022, have) == (2019, 2020, 2021)

    def test_a_missing_season_is_skipped_not_shortened(self):
        have = [2018, 2019, 2021, 2022]
        assert pk.window_for(2022, have) == (2018, 2019, 2021)

    def test_the_stamped_season_cannot_reach_its_own_factor(self):
        """The leakage guard, stated as a fact about the numbers: a huge
        effect present ONLY in the stamped season must not move that
        season's factor at all."""
        base = synthetic_pa(seasons=(2021, 2022, 2023), pa_per_pair=300)
        leak = synthetic_pa(seasons=(2024,), hr_boost={"T00": 3.0},
                            pa_per_pair=300, seed=1)
        counts = pk.pa_counts(pd.concat([base, leak], ignore_index=True))
        f = pk.factors_for_season(counts, 2024)
        hr = f[f["component"] == "hr_rate"].set_index("park")["factor"]
        assert hr["T00"] == pytest.approx(1.0, abs=0.1)
        assert list(f["window"].iloc[0]) == [2021, 2022, 2023]


class TestBallastChoice:
    def test_loso_windows_leave_the_season_out_and_stay_window_length(self):
        w = pk.loso_windows((2015, 2016, 2017, 2018, 2019))
        for held_out, window in w.items():
            assert held_out not in window
            assert len(window) == pk.WINDOW_SEASONS

    def test_choose_ballast_takes_the_pooled_minimum(self):
        table = pd.DataFrame([
            {"component": "hr_rate", "ballast": 0.0, "season": None, "rmse": 0.2},
            {"component": "hr_rate", "ballast": 100.0, "season": None, "rmse": 0.1},
            {"component": "hr_rate", "ballast": 200.0, "season": None, "rmse": 0.3},
            {"component": "hr_rate", "ballast": 100.0, "season": 2015, "rmse": 0.01},
        ])
        assert pk.choose_ballast(table) == {"hr_rate": 100.0}

    def test_ties_break_toward_more_regression(self):
        table = pd.DataFrame([
            {"component": "k_rate", "ballast": 100.0, "season": None, "rmse": 0.1},
            {"component": "k_rate", "ballast": 900.0, "season": None, "rmse": 0.1},
        ])
        assert pk.choose_ballast(table) == {"k_rate": 900.0}

    def test_the_frozen_ballast_covers_every_component(self):
        assert set(pk.FROZEN_BALLAST) == set(pk.COMPONENT_NAMES)
        assert all(np.isfinite(v) and v >= 0 for v in pk.FROZEN_BALLAST.values())


class TestWideTable:
    def test_columns_are_the_names_the_rate_models_look_for(self):
        """`prepare_model_data` reads `RateComponent.park_factor_col` off the
        frame; a column named anything else leaves that component at a
        neutral offset and says so only in a log line."""
        from src.models.pa_components import RATE_COMPONENTS

        counts = pk.pa_counts(synthetic_pa(hr_boost={"T00": 1.5}))
        wide = pk.wide_table(pk.build_table(counts, [2024]))
        for comp in RATE_COMPONENTS.values():
            assert comp.park_factor_col in wide.columns
        assert {"team", "game_year"} <= set(wide.columns)
        assert not wide.duplicated(["team", "game_year"]).any()

    def test_wide_and_long_carry_the_same_numbers(self):
        counts = pk.pa_counts(synthetic_pa(hr_boost={"T00": 1.5}))
        long = pk.build_table(counts, [2024])
        wide = pk.wide_table(long).set_index(["team", "game_year"])
        for _, row in long.iterrows():
            col = pk.FACTOR_COLUMNS[row["component"]]
            assert wide.loc[(row["team"], row["game_year"]), col] == pytest.approx(
                row["factor"])


@pytest.fixture(scope="module")
def table():
    path = Path(__file__).parent.parent.parent / pk.DEFAULT_PATH
    if not path.exists():
        pytest.skip(f"{path} not built")
    return pd.read_parquet(path)


class TestBuiltArtifact:
    """The committed file itself, when it is there."""

    def test_every_component_column_is_present_and_positive(self, table):
        for col in pk.FACTOR_COLUMNS.values():
            assert col in table.columns
            assert (table[col] > 0).all()
            assert table[col].between(0.5, 2.0).all()

    def test_thirty_parks_a_season_centred_on_one(self, table):
        """K% is exact: `n_pa` is its own denominator, and its centring weight.
        The others are centred on their own trials (balls in play, at-bats),
        which the wide file does not carry, so a PA-weighted mean of those is
        only near 1 — near enough that an offset cannot move a league level."""
        for year, g in table.groupby("game_year"):
            assert len(g) == 30, year
            assert float(np.average(g["k_park_factor"], weights=g["n_pa"])) == (
                pytest.approx(1.0, abs=1e-9))
            for col in pk.FACTOR_COLUMNS.values():
                assert float(np.average(g[col], weights=g["n_pa"])) == (
                    pytest.approx(1.0, abs=2e-3))

    def test_the_sidecar_records_the_frozen_ballast_and_the_sweep(self, table):
        import json

        path = (Path(__file__).parent.parent.parent / pk.DEFAULT_PATH
                ).with_suffix("").with_suffix(".meta.json")
        meta = json.loads(path.read_text())
        assert meta["ballast"] == {k: float(v) for k, v
                                   in pk.FROZEN_BALLAST.items()}
        assert meta["loso_persistence"]
        assert meta["ballast_chosen_on"] == list(pk.BALLAST_SEASONS)
