"""The dated-cutoff path and its leakage guard (BAS-59).

The claim under test is the one the whole fair-fight comparison rests on: a
plate appearance dated on or after the cutoff cannot move a single posterior
draw. The synthetic fixture makes that visible — the post-cutoff rows are
*all strikeouts*, so if any of them leaked, the fitted K rate would move
enormously and no test would have to be subtle about it.

Two levels:

  * the pymc-free level (runs in CI): the cells fed to the likelihood are
    byte-identical whether or not the extreme post-cutoff rows exist, and the
    guard raises when handed a leaky frame;
  * the pymc level (skipped where pymc is not installed, i.e. CI): the
    compiled log-probability of the two models agrees exactly at the same
    parameter point, which is the actual statement "the posterior cannot
    move".
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.aggregation import aggregate_binomial_cells
from src.models.cutoff import apply_cutoff, assert_no_post_cutoff, cutoff_exposure

CUTOFF = "2026-07-01"

# The model module needs pymc and arviz, which requirements-ci.txt leaves out
# on purpose (MCMC runs on Modal, not in CI). Skip only the tests that build a
# model; the cells-and-guard half above runs everywhere.
needs_pymc = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("pymc", "arviz")),
    reason="MCMC deps (pymc/arviz) are not installed in CI",
)


def _pa_rows(dates, k_values, batters, pitchers=None):
    n = len(dates)
    return pd.DataFrame({
        "batter": batters,
        "pitcher": pitchers if pitchers is not None else [900 + i % 3 for i in range(n)],
        "game_date": pd.to_datetime(dates),
        "game_year": [pd.Timestamp(d).year for d in dates],
        "stand": ["R"] * n,
        "is_k": k_values,
        "home_team": ["HOU"] * n,
        "away_team": ["SEA"] * n,
        "inning_topbot": ["Top"] * n,
    })


@pytest.fixture
def before_and_after():
    """(pre-cutoff frame, pre + extreme post-cutoff frame).

    Pre-cutoff: two prior seasons and a partial 2026, ~25% strikeouts.
    Post-cutoff: 400 PA that are *every one* a strikeout, dated on and after
    the cutoff — including one exactly on it, which the strict `<` must drop.
    """
    rng = np.random.default_rng(59)
    dates, ks, batters = [], [], []
    for season in (2024, 2025, 2026):
        for i in range(300):
            day = 1 + i % 25
            month = 4 + (i // 25) % 3          # April-June, all before Jul 1
            dates.append(f"{season}-{month:02d}-{day:02d}")
            ks.append(int(rng.random() < 0.25))
            batters.append(100 + i % 4)
    before = _pa_rows(dates, ks, batters)

    late_dates, late_batters = [], []
    for i in range(400):
        # The first row sits exactly on the cutoff: strictly-before must cut it.
        day = 1 + i % 28
        month = 7 + (i // 28) % 2
        late_dates.append(f"2026-{month:02d}-{day:02d}")
        late_batters.append(100 + i % 4)
    late = _pa_rows(late_dates, [1] * 400, late_batters)
    assert (pd.to_datetime(late["game_date"]) >= pd.Timestamp(CUTOFF)).all()

    leaky = pd.concat([before, late], ignore_index=True)
    return before, leaky


# ─── the pymc-free half: runs everywhere, including CI ────────────────────

def test_apply_cutoff_is_strictly_before(before_and_after):
    _, leaky = before_and_after
    cut = apply_cutoff(leaky, CUTOFF)
    assert len(cut) < len(leaky)
    assert pd.to_datetime(cut["game_date"]).max() < pd.Timestamp(CUTOFF)
    # A game played exactly on the cutoff is withheld, matching
    # intraseason.split_at_cutoff.
    on_the_day = leaky[pd.to_datetime(leaky["game_date"]) == pd.Timestamp(CUTOFF)]
    assert len(on_the_day) > 0
    assert not set(on_the_day.index) & set(cut.index)


def test_cut_frame_equals_the_clean_frame(before_and_after):
    before, leaky = before_and_after
    cut = apply_cutoff(leaky, CUTOFF).reset_index(drop=True)
    pd.testing.assert_frame_equal(cut, before.reset_index(drop=True))


def test_extreme_post_cutoff_rows_do_not_change_the_likelihood(before_and_after):
    """The cells handed to the Binomial are identical either way.

    Post-cutoff rows are 400 straight strikeouts; if a single one reached the
    likelihood, `k` in some cell would rise. Cell equality is the pymc-free
    statement of "the posterior cannot move", since the likelihood is a
    function of the cells alone.
    """
    before, leaky = before_and_after
    keys = ["batter", "game_year", "stand_idx"]

    def cells(df):
        d = apply_cutoff(df, CUTOFF)
        return aggregate_binomial_cells(
            d.assign(stand_idx=(d["stand"] == "R").astype(int)),
            cell_cols=tuple(keys),
            carry_cols=(),
        ).sort_values(keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(cells(before), cells(leaky))
    assert cells(leaky)["k"].sum() < cells(leaky)["n"].sum()   # not all strikeouts


def test_guard_raises_on_a_leaky_frame(before_and_after):
    _, leaky = before_and_after
    with pytest.raises(ValueError, match="leakage"):
        assert_no_post_cutoff(leaky, CUTOFF)
    # And passes on the cut one.
    assert_no_post_cutoff(apply_cutoff(leaky, CUTOFF), CUTOFF)


def test_guard_needs_a_date_column():
    df = pd.DataFrame({"batter": [1], "is_k": [0]})
    with pytest.raises(KeyError, match="game_date"):
        assert_no_post_cutoff(df, CUTOFF)
    with pytest.raises(KeyError, match="game_date"):
        apply_cutoff(df, CUTOFF)


def test_no_cutoff_is_a_passthrough(before_and_after):
    _, leaky = before_and_after
    assert apply_cutoff(leaky, None) is leaky
    assert_no_post_cutoff(leaky, None)          # no raise


def test_exposure_reports_the_partial_season(before_and_after):
    _, leaky = before_and_after
    exp = cutoff_exposure(leaky, CUTOFF)
    assert exp["partial_season"] == 2026
    assert exp["partial_pa"] == 300
    assert exp["prior_pa"] == 600
    assert exp["last_game"] < "2026-07-01"


# ─── the pymc half: the posterior itself. Skipped without pymc (CI). ──────


def _prepared(df):
    from src.models.pa_k_rate import prepare_model_data

    return prepare_model_data(df, None, min_pa=1, cutoff_date=CUTOFF,
                              include_pitcher=True)


@needs_pymc
def test_prepared_data_is_identical_with_and_without_leakage(before_and_after):
    before, leaky = before_and_after
    a, b = _prepared(before), _prepared(leaky)
    assert a["n_obs"] == b["n_obs"]
    assert a["n_pa"] == b["n_pa"] == 900
    np.testing.assert_array_equal(a["k"], b["k"])
    np.testing.assert_array_equal(a["n_trials"], b["n_trials"])
    np.testing.assert_array_equal(a["batter_idx"], b["batter_idx"])
    np.testing.assert_array_equal(a["pitcher_idx"], b["pitcher_idx"])


@needs_pymc
def test_logp_is_identical_so_no_draw_can_move(before_and_after):
    """The real statement: same log-probability at the same parameter point.

    Two models, one built from the clean frame and one from the frame with 400
    extreme post-cutoff strikeouts appended, evaluated at an identical point.
    NUTS explores a posterior only through this function, so an exact match
    means no post-cutoff PA can move any draw of any chain.
    """
    from src.models.pa_k_rate import build_model

    before, leaky = before_and_after
    m_clean = build_model(_prepared(before))
    m_leaky = build_model(_prepared(leaky))

    rng = np.random.default_rng(7)
    point = {}
    for name, value in m_clean.initial_point().items():
        point[name] = np.asarray(value) + 0.1 * rng.standard_normal(np.shape(value))

    logp_clean = m_clean.compile_logp()(point)
    logp_leaky = m_leaky.compile_logp()(point)
    assert float(logp_clean) == pytest.approx(float(logp_leaky), rel=0, abs=1e-9)


@needs_pymc
def test_the_test_would_notice_leakage(before_and_after):
    """Control: without the cutoff the extreme rows do move the logp.

    A leakage guard that cannot fail is not a guard, so this asserts the
    fixture is strong enough to have caught one.
    """
    from src.models.pa_k_rate import build_model, prepare_model_data

    before, leaky = before_and_after
    m_clean = build_model(prepare_model_data(before, None, min_pa=1,
                                             include_pitcher=True))
    m_leaky = build_model(prepare_model_data(leaky, None, min_pa=1,
                                             include_pitcher=True))
    point = m_clean.initial_point()
    assert float(m_clean.compile_logp()(point)) != pytest.approx(
        float(m_leaky.compile_logp()(point)), rel=1e-6)
