"""The measurement model (BAS-85, docs/bayes-measurement.md).

`src/models/pa_measurement.py` takes the joint graph over K% and HR/PA and
adds three *observed* channels to it — barrels per batted ball and mean exit
velocity on the power latent, whiffs per swing on the contact latent. Four
claims have to hold, and this file checks all four:

1. **The channel windows are walk-forward.** A month bucket at or after the
   cutoff's month never reaches a channel, and a season before the cutoff's
   year is read whole. This is the same rule
   `src.models.pa_covariates.season_cutoff` states as a date; here it is
   checked as counts, on buckets built so that including the cutoff's own
   month would change every number.
2. **An unobserved cell contributes nothing.** A batter with no tracked
   contact in his window is not an observation of a zero barrel rate; he is
   absent from the channel entirely. That is the structural difference from
   BAS-83, whose covariate put him at the league mean with full weight.
3. **The graph is the joint graph plus the channels.** Every variable
   `build_joint_model` declares is still declared, with the same dims, and
   the additions are exactly the loadings, the intercepts, `sigma_ev` and the
   three observed nodes — so a `measurement` fit's projections come out
   through `pa_joint.component_posterior` unchanged.
4. **The log-probability is finite** at the initial point, both variants, on
   a tiny synthetic dataset.

The first two need neither pymc nor arviz — the channel preparation is plain
numpy and pandas, which is why it lives above the graph builder's local pymc
import, the same split `src/models/pa_covariates.py` uses. Everything that
builds a graph is marked `needs_pymc`. **No sampling anywhere:** MCMC belongs
on the grid.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.pa_measurement import (  # noqa: E402
    CHANNELS, CHANNEL_COMPONENT, MEASUREMENT_COMPONENTS, MIN_BBE, MIN_SWINGS,
    channel_arrays, intercept_name, loading_name, measurement_param_summary,
    prepare_channels,
)

needs_pymc = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("pymc", "arviz")),
    reason="MCMC deps (pymc/arviz) are not installed in CI",
)

BATTERS = np.array([1000, 1001, 1002])
SEASONS = np.array([2023, 2024])


def _contact_monthly() -> pd.DataFrame:
    """Contact buckets where every month carries a different, recognisable
    number of batted balls, so a window that took one month too many is a
    different total rather than a slightly different one.

    Batter 1002 has no 2024 bucket before June at all — he is the "no tracked
    contact in his window" case claim 2 is about.
    """
    rows = []
    for season in (2023, 2024):
        for month in (4, 5, 6, 7):
            for i, b in enumerate(BATTERS):
                if b == 1002 and season == 2024 and month < 6:
                    continue
                bbe = 10.0 * (month + i)
                rows.append({
                    "side": "hitter", "player": int(b), "season": season,
                    "month": month, "bbe": bbe,
                    "sum_ev": 88.0 * bbe, "sum_ev2": (88.0 ** 2 + 210.0) * bbe,
                    "n_barrel": np.floor(0.08 * bbe),
                })
    # A pitcher-side row with the same ids, to prove the side filter bites.
    rows.append({"side": "pitcher", "player": 1000, "season": 2024,
                 "month": 4, "bbe": 9999.0, "sum_ev": 0.0, "sum_ev2": 0.0,
                 "n_barrel": 9999.0})
    return pd.DataFrame(rows)


def _swing_monthly() -> pd.DataFrame:
    rows = []
    for season in (2023, 2024):
        for month in (4, 5, 6, 7):
            for i, b in enumerate(BATTERS):
                z = 100.0 * (month + i)
                rows.append({
                    "batter": int(b), "season": season, "month": month,
                    "pitches_z": 3 * z, "pitches_o": 3 * z,
                    "swings_z": z, "swings_o": 0.5 * z,
                    "contact_z": 0.8 * z, "contact_o": 0.3 * z,
                    "whiff_z": 0.2 * z, "whiff_o": 0.2 * z,
                    "called_z": z,
                })
    return pd.DataFrame(rows)


# --- 1. the walk-forward window --------------------------------------------

def test_the_cutoff_month_and_everything_after_it_is_excluded():
    """A June 1 cutoff sees April and May of 2024, and never June.

    The buckets are monthly, so the only honest read of a June 1 cutoff is
    "through May 31"; a June bucket contains games played after the cutoff
    and admitting it would be leakage, which is exactly what
    `docs/contact-quality.md` §2 refuses.
    """
    arr = channel_arrays(BATTERS, SEASONS, _contact_monthly(), _swing_monthly(),
                         "2024-06-01")
    # batter 1000, 2024: April (40) + May (50), not June (60) or July (70)
    assert arr["bbe"][0, 1] == 40.0 + 50.0
    # batter 1001, 2024: 50 + 60
    assert arr["bbe"][1, 1] == 50.0 + 60.0


def test_a_season_before_the_cutoff_year_is_read_whole():
    """2023 is finished at a 2024 cutoff, so all four of its buckets count —
    the month filter is about the cutoff's *own* season and nothing else."""
    arr = channel_arrays(BATTERS, SEASONS, _contact_monthly(), _swing_monthly(),
                         "2024-06-01")
    assert arr["bbe"][0, 0] == 40.0 + 50.0 + 60.0 + 70.0


def test_a_later_cutoff_strictly_adds_batted_balls():
    """Monotone in the cutoff, which is what "additive monthly buckets" means
    and what makes an August cell a superset of the same batter's May one."""
    may = channel_arrays(BATTERS, SEASONS, _contact_monthly(), _swing_monthly(),
                         "2024-05-01")
    aug = channel_arrays(BATTERS, SEASONS, _contact_monthly(), _swing_monthly(),
                         "2024-08-01")
    assert (aug["bbe"] >= may["bbe"]).all()
    assert aug["bbe"][0, 1] > may["bbe"][0, 1]


def test_the_pitcher_side_of_the_contact_artifact_is_not_read():
    """The artifact stacks both sides under one `player` id. Reading the
    pitcher rows would credit a hitter with the contact he *allowed*, which
    for a pitcher who never bats is a 9999-batted-ball season."""
    arr = channel_arrays(BATTERS, SEASONS, _contact_monthly(), _swing_monthly(),
                         "2024-06-01")
    assert arr["bbe"].max() < 9999.0


def test_a_batter_the_model_does_not_carry_is_dropped():
    """The channels are indexed by the model's own batter array; a monthly
    row for anyone else has no cell to land in."""
    contact = _contact_monthly()
    extra = contact[contact["player"] == 1000].assign(player=4242)
    arr = channel_arrays(BATTERS, SEASONS, pd.concat([contact, extra]),
                         _swing_monthly(), "2024-06-01")
    plain = channel_arrays(BATTERS, SEASONS, contact, _swing_monthly(),
                           "2024-06-01")
    np.testing.assert_array_equal(arr["bbe"], plain["bbe"])


def test_a_renamed_artifact_column_raises_rather_than_zeroing():
    with pytest.raises(ValueError, match="n_barrel"):
        channel_arrays(BATTERS, SEASONS,
                       _contact_monthly().rename(columns={"n_barrel": "barrels"}),
                       _swing_monthly(), "2024-06-01")


# --- 2. an unobserved cell is absent, not a zero ---------------------------

def test_a_cell_with_no_tracked_contact_is_not_an_observation():
    """Batter 1002 has no 2024 bucket before June, so at a June 1 cutoff his
    2024 cell carries no barrel and no EV observation at all.

    This is the whole structural difference from BAS-83. There, the same
    batter got `x = 0` — the league mean — multiplied by a pooled
    coefficient at full weight. Here he simply does not appear in the
    channel's index arrays, so his likelihood contribution is exactly zero
    and the outcome channel is all the model has about him.
    """
    ch = prepare_channels(BATTERS, SEASONS, _contact_monthly(),
                          _swing_monthly(), "2024-06-01")
    cells = set(zip(ch.barrel_batter.tolist(), ch.barrel_season.tolist()))
    assert (2, 1) not in cells        # batter 1002, season 2024
    assert (2, 0) in cells            # his finished 2023 is there
    assert (0, 1) in cells


def test_a_thin_cell_is_dropped_below_the_batted_ball_floor():
    contact = _contact_monthly()
    thin = contact["player"].eq(1000) & contact["season"].eq(2024)
    contact.loc[thin, ["bbe", "sum_ev", "sum_ev2", "n_barrel"]] = [
        1.0, 88.0, 88.0 ** 2, 0.0]
    ch = prepare_channels(BATTERS, SEASONS, contact, _swing_monthly(),
                          "2024-06-01")
    cells = set(zip(ch.barrel_batter.tolist(), ch.barrel_season.tolist()))
    assert (0, 1) not in cells
    assert MIN_BBE > 1


def test_every_channel_carries_its_own_trial_count():
    """A binomial channel's `n` is the batter's real exposure, so a thin
    window contributes little *by construction* rather than by a weight
    someone chose."""
    ch = prepare_channels(BATTERS, SEASONS, _contact_monthly(),
                          _swing_monthly(), "2024-06-01")
    assert (ch.barrel_n > 0).all()
    assert (ch.barrel_k <= ch.barrel_n).all()
    assert (ch.whiff_n >= MIN_SWINGS).all()
    assert (ch.whiff_k <= ch.whiff_n).all()
    assert ch.barrel_n.sum() == ch.ev_n.sum()   # one window, two channels


def test_exit_velocity_is_standardized_by_the_per_batted_ball_sd():
    """`sigma_ev / sqrt(BBE)` is only the standard error of a cell mean if
    the scale is the per-*ball* sd, so that is what `ev_sd` has to be — here
    sqrt(210) by construction of the fixture's `sum_ev2`."""
    ch = prepare_channels(BATTERS, SEASONS, _contact_monthly(),
                          _swing_monthly(), "2024-06-01")
    assert ch.ev_mu == pytest.approx(88.0)
    assert ch.ev_sd == pytest.approx(np.sqrt(210.0), rel=1e-6)
    # Every cell has the same mean EV in this fixture, so every z is zero.
    np.testing.assert_allclose(ch.ev_z, 0.0, atol=1e-9)


def test_more_successes_than_trials_is_an_error():
    contact = _contact_monthly()
    contact["n_barrel"] = contact["bbe"] * 2
    with pytest.raises(ValueError, match="more successes than trials"):
        prepare_channels(BATTERS, SEASONS, contact, _swing_monthly(),
                         "2024-06-01")


# --- 3 & 4. the graph -------------------------------------------------------

def _pa_rows(n_batters=10, seasons=(2023, 2024), seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batters):
        power = rng.normal()
        k_rate = 1 / (1 + np.exp(-(-1.2 + 0.3 * power)))
        hr_rate = 1 / (1 + np.exp(-(-3.4 + 0.4 * power)))
        for s in seasons:
            for _ in range(int(rng.integers(60, 90))):
                rows.append({
                    "batter": 1000 + b, "game_year": s,
                    "game_date": pd.Timestamp(f"{s}-06-15"),
                    "stand": "R" if b % 2 == 0 else "L",
                    "is_k": int(rng.random() < k_rate),
                    "is_bb": int(rng.random() < 0.09),
                    "is_hr": int(rng.random() < hr_rate),
                    "home_team": "HOU" if b % 3 else "SEA",
                    "away_team": "SEA" if b % 3 else "HOU",
                    "inning_topbot": "Top",
                })
    return pd.DataFrame(rows)


def _fixture_channels(batters, seasons, seed=5):
    """Channel artifacts covering a fitted batter/season grid."""
    rng = np.random.default_rng(seed)
    cq, sw = [], []
    for b in batters:
        for s in seasons:
            for month in (4, 5):
                bbe = float(rng.integers(30, 60))
                ev = 86.0 + rng.normal(0, 2)
                cq.append({"side": "hitter", "player": int(b), "season": int(s),
                           "month": month, "bbe": bbe, "sum_ev": ev * bbe,
                           "sum_ev2": (ev ** 2 + 200.0) * bbe,
                           "n_barrel": float(rng.binomial(int(bbe), 0.08))})
                z = float(rng.integers(200, 400))
                sw.append({"batter": int(b), "season": int(s), "month": month,
                           "pitches_z": 3 * z, "pitches_o": 3 * z,
                           "swings_z": z, "swings_o": 0.5 * z,
                           "contact_z": 0.8 * z, "contact_o": 0.3 * z,
                           "whiff_z": float(rng.binomial(int(z), 0.2)),
                           "whiff_o": float(rng.binomial(int(0.5 * z), 0.3)),
                           "called_z": z})
    return pd.DataFrame(cq), pd.DataFrame(sw)


@pytest.fixture(scope="module")
def built():
    """`(joint_data, channels)` for the graph tests, prepared once."""
    from src.models.pa_joint import prepare_joint_data

    data = prepare_joint_data(_pa_rows(), None, min_pa=1,
                              components=MEASUREMENT_COMPONENTS)
    cq, sw = _fixture_channels(data.shared["batters"], data.shared["seasons"])
    channels = prepare_channels(data.shared["batters"], data.shared["seasons"],
                                cq, sw, "2025-06-01")
    return data, channels


@needs_pymc
@pytest.mark.parametrize("ability_walk", [False, True])
def test_the_graph_is_the_joint_graph_plus_exactly_the_channels(built, ability_walk):
    """Nothing the joint model declares is dropped, and the additions are
    the loadings, the intercepts, `sigma_ev` and the three observed nodes.

    That is what lets `pa_joint.component_posterior` hand a one-component
    view of this trace to `pa_rate.generate_projections` unchanged: a
    measurement fit's projection arithmetic is the single-component arm's,
    so any difference between the arms is the posterior.
    """
    from src.models.pa_joint import build_joint_model
    from src.models.pa_measurement import build_measurement_model
    from src.models.pa_rate import ModelOptions

    data, channels = built
    options = ModelOptions(ability_walk=ability_walk)
    plain = {v.name for v in build_joint_model(data, options).named_vars.values()}
    full = {v.name for v in
            build_measurement_model(data, channels, options).named_vars.values()}

    assert plain <= full, "the measurement model dropped a joint variable"
    added = full - plain
    expected = {loading_name(c) for c in CHANNELS}
    expected |= {intercept_name(c) for c in CHANNELS}
    expected |= {"sigma_ev", "barrel_obs", "ev_obs", "whiff_obs"}
    expected |= {"barrel_batter_idx", "barrel_season_idx", "barrel_n",
                 "ev_batter_idx", "ev_season_idx", "ev_n",
                 "whiff_batter_idx", "whiff_season_idx", "whiff_n"}
    assert added == expected


@needs_pymc
@pytest.mark.parametrize("ability_walk", [False, True])
def test_the_logp_is_finite_at_the_initial_point(built, ability_walk):
    from src.models.pa_measurement import build_measurement_model
    from src.models.pa_rate import ModelOptions

    data, channels = built
    model = build_measurement_model(
        data, channels, ModelOptions(ability_walk=ability_walk))
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp), f"logp is {logp}"


@needs_pymc
def test_the_channels_actually_enter_the_likelihood(built):
    """A channel that is declared but disconnected would pass every test
    above and answer prediction 1 with the prior. So: change the observed
    barrel counts and the log-probability has to move."""
    from dataclasses import replace

    from src.models.pa_measurement import build_measurement_model
    from src.models.pa_rate import ModelOptions

    data, channels = built
    options = ModelOptions(ability_walk=False)
    base = build_measurement_model(data, channels, options)
    at = base.initial_point()
    lp0 = base.compile_logp()(at)

    bumped = replace(channels, barrel_k=np.zeros_like(channels.barrel_k))
    other = build_measurement_model(data, bumped, options)
    assert other.compile_logp()(other.initial_point()) != lp0


@needs_pymc
def test_a_component_the_channels_need_is_refused(built):
    """The whiff channel loads on the K% latent. A fit without K% has no
    such latent, and building the model against it would silently load the
    channel on whatever component sat at that index."""
    from src.models.pa_joint import prepare_joint_data
    from src.models.pa_measurement import build_measurement_model

    _, channels = built
    data = prepare_joint_data(_pa_rows(), None, min_pa=1,
                              components=("bb_rate", "hr_rate"))
    with pytest.raises(ValueError, match="whiff"):
        build_measurement_model(data, channels, None)


def test_every_channel_names_a_component_the_model_fits():
    assert set(CHANNEL_COMPONENT.values()) <= set(MEASUREMENT_COMPONENTS)
    assert set(CHANNEL_COMPONENT) == set(CHANNELS)


# --- reading the predictions off -------------------------------------------

class _FakePosterior(dict):
    def __contains__(self, k):   # xarray Datasets answer `in` on variables
        return dict.__contains__(self, k)


def _fake_trace(**arrays):
    from types import SimpleNamespace

    class _Var:
        def __init__(self, v):
            self.values = np.asarray(v)
            self.dims = ()

    return SimpleNamespace(
        posterior=_FakePosterior({k: _Var(v) for k, v in arrays.items()}))


def test_the_summary_reports_an_interval_and_a_tail_mass_per_loading():
    """Prediction 1 is "the interval excludes zero", so an interval — not a
    mean and an sd — is what the fit record has to carry."""
    rng = np.random.default_rng(3)
    trace = _fake_trace(
        lambda_barrel=0.6 + 0.05 * rng.standard_normal(400),
        lambda_ev=0.4 + 0.05 * rng.standard_normal(400),
        lambda_whiff=0.7 + 0.05 * rng.standard_normal(400),
        alpha_barrel=rng.standard_normal(400),
        alpha_ev=rng.standard_normal(400),
        alpha_whiff=rng.standard_normal(400),
        sigma_ev=1.0 + 0.01 * rng.standard_normal(400),
    )
    out = measurement_param_summary(trace)
    for channel in CHANNELS:
        s = out[loading_name(channel)]
        assert s["q2.5"] > 0 and s["p_gt_0"] == 1.0
    assert "max_abs_loading_corr" in out
    assert out["max_abs_loading_corr"] < 0.95


def test_the_summary_converts_the_exit_velocity_loading_to_miles_an_hour():
    """A standardized loading is not a number anyone can check against
    public work; miles an hour per sd of power is."""
    ch = prepare_channels(BATTERS, SEASONS, _contact_monthly(),
                          _swing_monthly(), "2024-06-01")
    trace = _fake_trace(lambda_ev=np.full(100, 0.5))
    out = measurement_param_summary(trace, ch)
    assert out["lambda_ev"]["mph_per_sd"] == pytest.approx(0.5 * ch.ev_sd)


def test_the_summary_is_empty_without_a_posterior():
    assert measurement_param_summary(None) == {}
