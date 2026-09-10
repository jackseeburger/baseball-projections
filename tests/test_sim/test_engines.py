"""The engine switch under station E's chain (src/sim/engines.py).

The one thing that has to hold before any of this can be scored: **rung 0 is
the chain as served, to the last bit.** Both rate tables now run through
`ChainEngines`, so if the stock path through it differed from the arithmetic
those two modules used before — by a rounding, by a ballast unit, by a player
who falls out of a join — every number in the market benchmark would move and
nothing would fail. So the reproduction is asserted with `==` on the arrays,
not `approx`, at three levels: the pitcher table, the hitter table, and the
`build_slate` slate both callers price a game from.

The rest is the ladder's own arithmetic: it nests, the corrections are the
served providers' arithmetic, and the covariate cutoff is never rounded
forward. All synthetic — no network, no PA parquet, no pymc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sim import engines as eng
from src.sim import game_model as gm
from src.sim import lineups as lu
from src.sim import starters as st

P_LEAGUE = {"rate_k": 0.22, "rate_bbhbp": 0.08, "rate_hr": 0.03,
            "bf_per_ip": 4.28}
H_LEAGUE = {"rate_k": 0.225, "rate_bbhbp": 0.090, "rate_hr": 0.032,
            "rate_iso": 0.155, "rate_babip": 0.295,
            "nonab_share": 0.012, "sf_share": 0.006, "triple_share": 0.10}
LG_RA9 = 4.50


def p_counts(pitcher, season, bf, k, bbhbp, hr):
    return {"pitcher": pitcher, "season": season, "bf": bf, "k": k,
            "bbhbp": bbhbp, "hr": hr, "outs": bf * 0.7}


def h_counts(batter, season, pa, k, bbhbp, hr, xb, hip):
    ab = pa - bbhbp
    return {"batter": batter, "season": season, "pa": pa, "ab": ab,
            "bip": ab - k - hr, "k": k, "bbhbp": bbhbp, "hr": hr,
            "xb": xb, "hip": hip, "doubles": 0.0, "triples": 0.0, "sf": 0.0}


PITCHERS = pd.DataFrame([
    p_counts(543037, 2026, 480, 129, 41, 17),
    p_counts(543037, 2025, 620, 174, 52, 22),
    p_counts(543037, 2024, 590, 152, 60, 24),
    p_counts(605400, 2026, 310, 95, 22, 9),
    p_counts(605400, 2024, 700, 210, 55, 30),
    # A pitcher who is only in the oldest season in the window.
    p_counts(111111, 2024, 150, 30, 20, 8),
    # ...and one entirely outside it, who must not appear at all.
    p_counts(222222, 2019, 900, 200, 70, 30),
])
BATTERS = pd.DataFrame([
    h_counts(660271, 2026, 480, 100, 60, 25, 130, 90),
    h_counts(660271, 2025, 620, 130, 80, 32, 170, 120),
    h_counts(660271, 2024, 590, 140, 70, 28, 150, 110),
    h_counts(592450, 2026, 300, 90, 30, 12, 70, 55),
    h_counts(592450, 2023, 640, 150, 75, 35, 180, 125),
    h_counts(514888, 2024, 200, 40, 18, 5, 40, 40),
])


# ─── rung 0 reproduces the served chain, to the last bit ───

def test_the_stock_pitcher_table_is_the_one_station_e_already_computed():
    """`marcel_rates` now routes through `ChainEngines`; rung 0 must be a no-op.

    Pinned against both the station A provider called directly with station E's
    constants and the pre-provider arithmetic `_marcel_rates_legacy` keeps, so
    a change in either the switch or the provider fails here rather than in the
    fifth decimal of a Brier score.
    """
    from src.eval.pitchers import pitcher_rates

    through_switch = st.marcel_rates(PITCHERS, 2026, P_LEAGUE)
    direct = pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                           params=st.marcel_params(),
                           components=st.PITCHER_COMPONENTS)
    legacy = st.marcel_rates(PITCHERS, 2026, P_LEAGUE, legacy=True)

    assert list(through_switch.index) == list(direct.index)
    assert 222222 not in set(through_switch.index)
    for column in ["bf_weighted", *st.RATE_COLS]:
        assert np.array_equal(through_switch[column].to_numpy(),
                              direct[column].to_numpy())
        assert through_switch[column].to_numpy() == pytest.approx(
            legacy[column].reindex(through_switch.index).to_numpy())


def test_the_stock_hitter_table_is_the_one_lineups_already_computed():
    """The new hitter path on stock params *is* `lineups.marcel_rates`.

    This is the load-bearing one: unlike the pitcher side there was no station
    A provider under `lineups.marcel_rates` before, so the tuned path is a new
    estimator and the only thing that says it is the same estimator is that at
    stock constants it returns the same five numbers. Ballast units (real PA
    here, average-year weight in `MarcelParams`) are exactly what a mistake
    would hide in.

    The agreement is to floating-point round-off rather than literally
    bit-for-bit, and necessarily so: the harness accumulates `sum(w_i * n_i)`
    against a ballast of `p.ballast * mean(w)` where this module accumulates
    `sum((w_i/w_0) * n_i)` against `ballast`, which is the same number by
    algebra and not by IEEE 754. `starters.marcel_params` has carried the same
    translation, and the same caveat, since station E started reading station
    A's provider. It does not reach the served chain, because rung 0 does not
    take this path at all — `lineups.marcel_rates` runs its own arithmetic
    unless an engine says otherwise, which is what the slate test below pins
    with exact equality.
    """
    served = lu.marcel_rates(BATTERS, 2026, H_LEAGUE)
    through_switch = eng.ChainEngines().hitter_rates(
        BATTERS, 2026, H_LEAGUE, stock=lu.marcel_params())

    assert list(through_switch.index) == list(served.index)
    assert list(through_switch.columns) == ["pa_weighted", *lu.RATE_COLS]
    for column in through_switch.columns:
        assert through_switch[column].to_numpy() == pytest.approx(
            served[column].to_numpy(), abs=0, rel=1e-12)


def test_the_stock_hitter_table_gives_the_same_runs_lookup():
    """...and therefore the same runs above average per PA, which is what the
    chain actually reads off the table.

    Worth its own assertion because `runs_per_pa` differences the five rates
    against the league's, so a relative error on a rate becomes a much larger
    relative error on a number near zero. It stays at 1e-11 of a run.
    """
    served = lu.batter_runs_lookup(lu.marcel_rates(BATTERS, 2026, H_LEAGUE),
                                   H_LEAGUE)
    switched = lu.batter_runs_lookup(
        eng.ChainEngines().hitter_rates(BATTERS, 2026, H_LEAGUE,
                                        stock=lu.marcel_params()),
        H_LEAGUE)
    assert served.keys() == switched.keys()
    for batter, value in served.items():
        assert switched[batter] == pytest.approx(value, abs=1e-11, rel=1e-9)


# Two clubs, two starters and a reliever each, three hitters each.
CLUB_OF = {543037: 100, 605400: 101, 111111: 100, 777777: 101}
STARTS = {543037, 605400}


def _slate_inputs():
    """A two-club synthetic season, in the frames `ChainInputs` carries."""
    dates = pd.date_range("2026-04-01", periods=20).strftime("%Y-%m-%d")
    p_rows, h_rows = [], []
    for i, d in enumerate(dates):
        for pid, club in CLUB_OF.items():
            started = pid in STARTS
            p_rows.append({"pitcher": pid, "season": 2026, "date": d,
                           "team": club, "gs": int(started),
                           "game_pk": 900000 + i, "game_type": "R",
                           "bf": 24 if started else 6,
                           "k": 6 if started else 2, "bb": 2, "hbp": 0,
                           "hr": 1, "outs": 18 if started else 3,
                           "pitches": 90 if started else 20,
                           "ab": 22 if started else 5, "h": 5, "sf": 0})
        for bid in (660271, 592450, 514888):
            h_rows.append({"batter": bid, "season": 2026, "date": d,
                           "team_id": 100 + bid % 2, "pa": 4, "ab": 4, "h": 1,
                           "doubles": 0, "triples": 0, "hr": 0, "k": 1,
                           "bb": 0, "hbp": 0, "sf": 0})
    pitching = pd.DataFrame(p_rows)
    hitting = pd.DataFrame(h_rows)
    prior_p = PITCHERS[PITCHERS["season"] < 2026].rename(
        columns={"bbhbp": "bb"}).assign(hbp=0.0)
    prior_h = BATTERS[BATTERS["season"] < 2026].rename(
        columns={"bbhbp": "bb", "hip": "h_in_play"})
    prior_h = prior_h.assign(h=prior_h["h_in_play"] + prior_h["hr"], hbp=0.0)
    return gm.ChainInputs.from_logs(2026, pitching, hitting, prior_p, prior_h)


def test_the_default_slate_prices_a_game_exactly_as_it_did_before():
    """`build_slate` on the default `ChainConfig` must reach the two rate
    tables the chain reached before the switch existed, bit for bit.

    The slate is where every term downstream reads its inputs, so equality of
    `sp_ra9` and `runs_lookup` is equality of every probability the chain
    returns for every game of that date.
    """
    inputs = _slate_inputs()
    top_down = pd.DataFrame({"rs_pg": [4.4, 4.6], "ra_pg": [4.5, 4.5]},
                            index=pd.Index([100, 101], name="team_id"))
    slate = gm.build_slate("2026-04-15", inputs, top_down, 4.5, 4.5)
    assert slate.config.engines is eng.STOCK

    counts = pd.concat([inputs.pitcher_prior_counts,
                        st.appearances_before(inputs.pitcher_counts,
                                              "2026-04-15")],
                       ignore_index=True)
    expected_sp = st.starter_ra9_lookup(
        st.marcel_rates(counts, 2026, inputs.pitcher_league,
                        ballast=dict(st.BALLAST_BF), legacy=True),
        inputs.pitcher_league, 4.5)
    assert slate.sp_ra9.keys() == expected_sp.keys()
    for pid, ra9 in expected_sp.items():
        assert slate.sp_ra9[pid] == pytest.approx(ra9)

    h_counts_frame = pd.concat(
        [inputs.hitter_prior_counts,
         lu.games_before(inputs.hitter_counts, "2026-04-15")],
        ignore_index=True)
    expected_lu = lu.batter_runs_lookup(
        lu.marcel_rates(h_counts_frame, 2026, inputs.hitter_league,
                        ballast=dict(lu.BALLAST)),
        inputs.hitter_league)
    assert slate.runs_lookup.keys() == expected_lu.keys()
    for bid, raa in expected_lu.items():
        assert slate.runs_lookup[bid] == raa      # bit for bit, not approx


# ─── the ladder nests ───

def test_the_rungs_are_a_nesting():
    with pytest.raises(ValueError):
        eng.ChainEngines(tuned=True, contact=True)
    with pytest.raises(ValueError):
        eng.ChainEngines(stuff=True)
    assert eng.ChainEngines(tuned=True, stuff=True, contact=True).contact


def test_the_tuned_rung_actually_moves_the_pitcher_table():
    """Rung 1 is not a relabelling: the fitted constants and the age curve have
    to reach the rates, or the null prediction is vacuously true."""
    ages = {543037: 33.0, 605400: 24.0, 111111: 28.0}
    stock = st.marcel_rates(PITCHERS, 2026, P_LEAGUE)
    tuned = eng.ChainEngines(tuned=True, ages=ages).pitcher_rates(
        PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())
    assert not np.allclose(stock["rate_k"].to_numpy(),
                           tuned["rate_k"].to_numpy())
    assert list(stock.index) == list(tuned.index)


def test_the_recalibration_control_keeps_the_ballasts_and_drops_the_age_curve():
    ages = {543037: 33.0, 605400: 24.0, 111111: 28.0}
    tuned = eng.ChainEngines(tuned=True, ages=ages)
    control = eng.ChainEngines(tuned=True, ages=ages, age_slopes=False)
    a = tuned.pitcher_rates(PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())
    b = control.pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                              stock=st.marcel_params())
    # p_k_rate is the one pitcher component whose fit has age slopes at all.
    assert not np.allclose(a["rate_k"].to_numpy(), b["rate_k"].to_numpy())
    # ...and the ones that fell back to stock params carry no age term, so the
    # control cannot move them.
    assert np.array_equal(a["rate_hr"].to_numpy(), b["rate_hr"].to_numpy())


def test_zeroing_the_slopes_leaves_every_other_constant_alone():
    from src.eval.baselines import MarcelParams

    params = {"x": MarcelParams(ballast=425.0, weights=(1.0, 0.4, 0.2),
                                peak_age=26.0, age_slope_young=0.012,
                                age_slope_old=-0.006, league_mode="weighted3")}
    out = eng.zero_slopes(params)["x"]
    assert (out.ballast, out.weights, out.peak_age, out.league_mode) == (
        425.0, (1.0, 0.4, 0.2), 26.0, "weighted3")
    assert out.age_slope_young == 0.0 and out.age_slope_old == 0.0


# ─── the correction, and its cutoff ───

class _Fit:
    """A `StuffFit`/`ContactFit` stand-in: the `predict` contract, nothing else."""

    def __init__(self, intercept, coefs):
        self.features = tuple(coefs)
        self.coef = {"intercept": intercept, "base": 1.0, **coefs}

    def predict(self, base, z):
        out = self.coef["intercept"] + self.coef["base"] * np.asarray(base)
        for f in self.features:
            out = out + self.coef[f] * z[f].to_numpy(dtype="float64")
        return out


def _monthly_stub():
    """Two pitchers with one tracked covariate, in `features_at_cutoff`'s shape."""
    return pd.DataFrame({"player": [543037, 605400],
                         "xwhiff": [1.5, -0.5]}).set_index("player")


def test_a_correction_is_added_to_the_rate_and_a_missing_player_gets_zero():
    engine = eng.ChainEngines(
        tuned=True, stuff=True, predict_year=2026,
        stuff_fits={"bbhbp": _Fit(0.004, {"xwhiff": -0.01})},
        stuff_monthly=pd.DataFrame())
    # Prime the per-boundary memo instead of building covariates from a monthly
    # artifact: this test is about what the correction does to a rate, and it
    # pins the memo key at the same time.
    engine._z[("pitcher", "2026-09-01")] = _monthly_stub()
    base = engine.pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                                stock=st.marcel_params(), as_of="2026-09-09")
    engine_off = eng.ChainEngines(tuned=True, ages={})
    plain = engine_off.pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                                     stock=st.marcel_params())

    assert base.loc[543037, "rate_bbhbp"] == pytest.approx(
        plain.loc[543037, "rate_bbhbp"] + 0.004 - 0.01 * 1.5)
    # 111111 is not in the covariate frame: z = 0, so he gets the intercept
    # only — the recalibrated baseline, not a dropped row.
    assert base.loc[111111, "rate_bbhbp"] == pytest.approx(
        plain.loc[111111, "rate_bbhbp"] + 0.004)
    # ...and a component the correction does not cover is untouched.
    assert base.loc[543037, "rate_k"] == plain.loc[543037, "rate_k"]


def test_the_covariate_cutoff_is_the_last_boundary_and_never_rounded_forward():
    from src.eval.contact import assert_month_boundary

    for as_of, expected in (("2026-09-09", "2026-09-01"),
                            ("2026-09-01", "2026-09-01"),
                            ("2026-09-30", "2026-09-01"),
                            ("2026-04-02", "2026-04-01")):
        cutoff = eng.month_boundary(as_of)
        assert str(cutoff.date()) == expected
        assert cutoff <= pd.Timestamp(as_of)
        assert_month_boundary(cutoff)          # the served guard accepts it


def test_the_age_map_is_the_chadwick_age_of_record():
    bd = pd.DataFrame({"batter": [1, 2, 3],
                       "birth_year": [1993, 1999, None],
                       "birth_month": [6, 12, None],
                       "birth_day": [30, 1, None]})
    ages = eng.age_map(2026, bd)
    assert ages[1] == pytest.approx(33.0, abs=0.01)
    assert ages[2] == pytest.approx(26.58, abs=0.01)
    assert 3 not in ages                        # unknown birthdate, no age


# ─── the level and spread match ───

def test_matching_with_every_other_flag_off_is_the_stock_table_bit_for_bit():
    """`match` is the identity at rung 0, and identically so.

    The point of the flag is to move a *rung's* table onto the stock table's
    moments; at rung 0 the table already is the stock table, so putting it
    through `(x - m) * (s/s) + m` would round it in the last bit for nothing.
    `_match` returns early instead, and this asserts the early return with
    `==` on the arrays rather than `approx` — every published Brier in
    docs/market-benchmark-2026.md is computed on this table.
    """
    served = st.marcel_rates(PITCHERS, 2026, P_LEAGUE)
    switched = eng.ChainEngines(match=True).pitcher_rates(
        PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())
    assert list(switched.index) == list(served.index)
    for column in ["bf_weighted", *st.RATE_COLS]:
        assert np.array_equal(switched[column].to_numpy(),
                              served[column].to_numpy())

    # ...and on the hitter side, where `lineups.marcel_rates` keeps its own
    # arithmetic unless the engine is tuned, so a match-only engine is the
    # served function itself.
    lu_served = lu.marcel_rates(BATTERS, 2026, H_LEAGUE)
    lu_switched = lu.marcel_rates(BATTERS, 2026, H_LEAGUE,
                                  engine=eng.ChainEngines(match=True))
    assert lu_served.equals(lu_switched)


def _weighted(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = np.asarray(x)[m], np.asarray(w)[m]
    mean = float((w * x).sum() / w.sum())
    return mean, float(np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))


def test_the_matched_pitcher_table_carries_the_stock_moments():
    """Weighted mean and weighted sd, to 1e-9, on the effective sample.

    This is the whole claim of the transform, stated as an equality rather
    than as "closer to". The weights are `bf_weighted`, which the tuned params
    do not change on the pitcher side, so both tables are weighted the same and
    the equality is arithmetic rather than a coincidence of this frame.
    """
    ages = {543037: 33.0, 605400: 24.0, 111111: 28.0}
    stock = eng.ChainEngines().pitcher_rates(
        PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())
    unmatched = eng.ChainEngines(tuned=True, ages=ages).pitcher_rates(
        PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())
    matched = eng.ChainEngines(tuned=True, ages=ages, match=True).pitcher_rates(
        PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())

    assert list(matched.index) == list(stock.index)
    moved = False
    for column in st.RATE_COLS:
        want = _weighted(stock[column].to_numpy(),
                         stock["bf_weighted"].to_numpy())
        got = _weighted(matched[column].to_numpy(),
                        matched["bf_weighted"].to_numpy())
        assert got[0] == pytest.approx(want[0], abs=1e-9)
        assert got[1] == pytest.approx(want[1], abs=1e-9)
        # The rung's own ordering survives — matching is affine with a positive
        # scale, so it cannot reorder players — and the matched table is not
        # just the stock table back again.
        assert np.array_equal(np.argsort(matched[column].to_numpy()),
                              np.argsort(unmatched[column].to_numpy()))
        if not np.allclose(matched[column].to_numpy(),
                           stock[column].to_numpy()):
            moved = True
    assert moved


def test_the_matched_hitter_table_carries_the_stock_moments():
    """The hitter side, where the tuned recency weights move `pa_weighted` too.

    Each table is weighted by the effective sample it carries, so the equality
    asserted is between the matched table's moments under its own weights and
    the stock table's under the stock weights — which is what the chain reads
    when it weights a lineup.
    """
    ages = {660271: 31.0, 592450: 34.0, 514888: 27.0}
    stock = eng.ChainEngines().hitter_rates(
        BATTERS, 2026, H_LEAGUE, stock=lu.marcel_params())
    matched = eng.ChainEngines(tuned=True, ages=ages, match=True).hitter_rates(
        BATTERS, 2026, H_LEAGUE, stock=lu.marcel_params())
    assert list(matched.index) == list(stock.index)
    for column in lu.RATE_COLS:
        want = _weighted(stock[column].to_numpy(),
                         stock["pa_weighted"].to_numpy())
        got = _weighted(matched[column].to_numpy(),
                        matched["pa_weighted"].to_numpy())
        assert got[0] == pytest.approx(want[0], abs=1e-9)
        assert got[1] == pytest.approx(want[1], abs=1e-9)


def test_a_player_missing_from_the_stock_table_keeps_his_rung_value():
    """No row is dropped and no row is invented; an absent one is left alone.

    Both tables come off the same count frame, so this is not expected to fire
    in the chain — it is here because the alternative failure (a player
    silently taking someone else's level because he had no stock row) would be
    invisible in a Brier score.
    """
    ages = {543037: 33.0, 605400: 24.0, 111111: 28.0}
    engine = eng.ChainEngines(tuned=True, ages=ages, match=True)
    stock = eng.ChainEngines().pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                                             stock=st.marcel_params())
    # Prime the memo with a stock table one pitcher short.
    engine._stock[("pitcher", "None", len(PITCHERS))] = stock.drop(index=111111)
    matched = engine.pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                                   stock=st.marcel_params())
    plain = eng.ChainEngines(tuned=True, ages=ages).pitcher_rates(
        PITCHERS, 2026, P_LEAGUE, stock=st.marcel_params())
    assert list(matched.index) == list(plain.index)
    for column in st.RATE_COLS:
        assert matched.loc[111111, column] == plain.loc[111111, column]
        assert matched.loc[543037, column] != plain.loc[543037, column]


def test_the_stock_table_is_memoised_per_side_and_date():
    engine = eng.ChainEngines(tuned=True, ages={543037: 33.0}, match=True)
    for _ in range(3):
        engine.pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                             stock=st.marcel_params(), as_of="2026-08-15")
    engine.pitcher_rates(PITCHERS, 2026, P_LEAGUE,
                         stock=st.marcel_params(), as_of="2026-08-16")
    assert set(engine._stock) == {("pitcher", "2026-08-15", len(PITCHERS)),
                                  ("pitcher", "2026-08-16", len(PITCHERS))}


def test_weighted_moments_ignores_rows_with_no_weight():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, np.nan, 0.0])
    mean, sd = eng.weighted_moments(x, w)
    assert mean == pytest.approx(1.5)
    assert sd == pytest.approx(0.5)
    assert all(np.isnan(v) for v in eng.weighted_moments(x, np.zeros(4)))


def test_build_engines_carries_the_two_new_switches():
    """`--engine-match` / `--engine-no-age` reach the object, at every rung."""
    built = eng.build_engines(0, 2026, match=True)
    assert built.match and not built.tuned
    birthdates = pd.DataFrame({"batter": [543037], "birth_year": [1991],
                               "birth_month": [7], "birth_day": [3]})
    r1 = eng.build_engines(1, 2026, age_slopes=False, match=True,
                           birthdates=birthdates)
    assert (r1.tuned, r1.age_slopes, r1.match) == (True, False, True)
    r1b = eng.build_engines(1, 2026, recalibration=True, birthdates=birthdates)
    assert (r1b.age_slopes, r1b.match) == (False, False)
