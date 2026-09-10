"""The BAS-87 stage-2 runner's arms and controls, on a synthetic panel.

Two claims are worth a test and neither needs real data:

1. **The stuff-only additive control is the served engine.** The whole
   pre-registration turns on it — the covariate-only share and the increment
   over `stuff_additive` are the same number only because the recalibration
   control with the baseline pinned *is* the fit
   `src.eval.stuff.fit_live_stuff` serves. If that ever stops being true the
   serving verdict is being read off the wrong control.
2. **The permutation control permutes only the command block.** A shuffle
   that also moved the stuff columns would be testing something else, and one
   that changed a column's marginal would not be a permutation at all.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.command import LEVEL_FEATURES
from src.eval.stuff import FEATURES as STUFF_FEATURES
from src.eval.stuff import fit_stuff

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "run_command_level_backtest",
        ROOT / "scripts/run_command_level_backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _module()


@pytest.fixture
def panel():
    """Three seasons of cells with covariates that really do carry signal.

    The realized rate is a baseline plus a command term plus noise, so an arm
    that reads the command block should beat one that does not, and a permuted
    arm should not.
    """
    rng = np.random.default_rng(20260910)
    rows = []
    for season in (2024, 2025, 2026):
        for cutoff in ("05-01", "07-01"):
            n = 200
            base = 0.08 + rng.normal(0, 0.006, n)
            z = {f: rng.normal(0, 1, n) for f in
                 tuple(STUFF_FEATURES) + tuple(LEVEL_FEATURES)}
            realized = (base + 0.004 * z["cmd_csw"] - 0.002 * z["xcsw"]
                        + rng.normal(0, 0.004, n))
            trials = rng.integers(120, 600, n).astype("float64")
            rows.append(pd.DataFrame({
                "component": "p_bb_rate", "side": "pitcher", "season": season,
                "cutoff": f"{season}-{cutoff}", "player": np.arange(n),
                "base": base, "realized_rate": realized,
                "realized_successes": realized * trials, "trials": trials,
                "pre_trials": rng.integers(50, 400, n).astype("float64"),
                "cmd_pitches_raw": rng.integers(200, 3000, n
                                                ).astype("float64"),
                **z}))
    return pd.concat(rows, ignore_index=True)


def test_the_additive_recalibration_control_is_the_served_stuff_engine(
        mod, panel):
    """`stuff_additive` in this script is the same call the serving path makes,
    so its coefficients must equal a stuff-only additive fit on the same rows."""
    res = mod.walk_forward(panel, ["p_bb_rate"], [2026], LEVEL_FEATURES)
    got = res[res["model"] == mod.SERVED_ARM]
    want = fit_stuff(panel[panel["season"] < 2026], "p_bb_rate",
                     features=STUFF_FEATURES, fixed_base=True)
    assert want.coef["base"] == 1.0
    direct = np.clip(want.predict(
        panel[panel["season"] == 2026]["base"].to_numpy(dtype="float64"),
        panel[panel["season"] == 2026]), 1e-4, 0.999)
    np.testing.assert_allclose(got["predicted"].to_numpy(), direct)


def test_the_command_arm_beats_the_stuff_only_control_when_command_matters(
        mod, panel):
    """The panel's realized rate is built with a real `cmd_csw` term, so the
    joint arm has to be closer to it than the control that cannot see it."""
    res = mod.walk_forward(panel, ["p_bb_rate"], [2025, 2026], LEVEL_FEATURES)
    inc = mod.paired(res, mod.ADDITIVE_ARM, mod.SERVED_ARM)
    assert inc["diff"] < 0
    assert inc["t"] < -2.0
    # Both cluster columns are reported and neither is degenerate.
    assert inc["n_clusters"] == res["player"].nunique()
    assert inc["n_cells"] == 4


def test_the_shuffle_permutes_the_command_block_and_only_it(mod, panel):
    """Same marginal, wrong pitcher — and the stuff columns untouched."""
    out = mod.shuffle_z(panel, LEVEL_FEATURES, seed=7)
    for (_, _), g in out.groupby(["season", "cutoff"]):
        want = panel[(panel["season"] == g["season"].iloc[0])
                     & (panel["cutoff"] == g["cutoff"].iloc[0])]
        for f in LEVEL_FEATURES:
            np.testing.assert_allclose(np.sort(g[f].to_numpy()),
                                       np.sort(want[f].to_numpy()))
        for f in STUFF_FEATURES:
            np.testing.assert_allclose(g[f].to_numpy(), want[f].to_numpy())
    assert not np.allclose(out["cmd_csw"].to_numpy(),
                           panel["cmd_csw"].to_numpy())


def test_the_permuted_arm_lands_on_the_stuff_only_control(mod, panel):
    """The pipeline must not be able to fit the split. A command vector
    attached to the wrong pitcher may cost a little; it must not pay."""
    shuffled = mod.shuffle_z(panel, LEVEL_FEATURES, seed=3)
    res = mod.walk_forward(panel, ["p_bb_rate"], [2025, 2026], LEVEL_FEATURES,
                           shuffled=shuffled)
    got = mod.paired(res, "command_level_additive_shuffled", mod.SERVED_ARM)
    assert got["t"] > -2.0


def test_the_serving_rule_is_the_one_architecture_md_carries(mod):
    """A floor of 1.0% of the served baseline's MAE at |t| > 2.0 (BAS-82)."""
    assert (mod.SERVE_MIN_PCT, mod.SERVE_MIN_T) == (1.0, 2.0)
    rows = [
        {"component": "clears", "arm": mod.ADDITIVE_ARM, "base": mod.BASE_ARM,
         "diff": -1e-4, "pct": -3.0, "t": -5.0, "t_cell": -5.0,
         "win_rate": 0.55},
        {"component": "clears", "arm": mod.ADDITIVE_ARM, "base": mod.SERVED_ARM,
         "diff": -1e-5, "pct": -1.2, "t": -2.4, "t_cell": -2.4,
         "win_rate": 0.52},
        {"component": "clears", "arm": mod.FREE_ARM, "base": mod.BASE_ARM,
         "diff": -1e-4, "pct": -3.1, "t": -5.1, "t_cell": -5.1,
         "win_rate": 0.55},
        # Same total gain, but under the floor on the covariate's own share.
        {"component": "floored", "arm": mod.ADDITIVE_ARM, "base": mod.BASE_ARM,
         "diff": -1e-4, "pct": -3.0, "t": -5.0, "t_cell": -5.0,
         "win_rate": 0.55},
        {"component": "floored", "arm": mod.ADDITIVE_ARM,
         "base": mod.SERVED_ARM, "diff": -1e-6, "pct": -0.9, "t": -3.0,
         "t_cell": -3.0, "win_rate": 0.51},
        {"component": "floored", "arm": mod.FREE_ARM, "base": mod.BASE_ARM,
         "diff": -1e-4, "pct": -3.1, "t": -5.1, "t_cell": -5.1,
         "win_rate": 0.55},
    ]
    got = mod.stage2_table(rows, ["clears", "floored"]).set_index("component")
    assert bool(got.loc["clears", "serves"])
    assert not bool(got.loc["floored", "serves"])
