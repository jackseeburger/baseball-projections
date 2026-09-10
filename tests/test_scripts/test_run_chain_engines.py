"""BAS-91's arms, predictions and verdict (scripts/run_chain_engines.py).

Two things are worth pinning about a scorer. The **arms** have to be the seven
the pre-registration names — an arm that quietly walks with the age curve on
when its name says off produces a clean table of numbers that answer a
different question — so the flags each arm shells out with are asserted here
rather than read off a docstring. And the **verdict** is the last thing anybody
reads, so each prediction is exercised in both directions on synthetic frames:
a pass has to be able to fail.

Synthetic throughout: no parquet, no network, no market archive.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_chain_engines", ROOT / "scripts/run_chain_engines.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


# ─── the arms ───

def test_the_seven_arms_are_the_pre_registered_ones():
    assert mod.ARM_SETS["bas91"] == ("served", "recal", "r1_match", "contact",
                                     "r3_noage", "r3_match", "r3_noage_match")
    # rung, age curve, matched — the table in docs/chain-engines-matched.md.
    def shape(name):
        spec = mod.RUNGS[name]
        return (spec["rung"], not spec.get("no_age", False)
                and not spec.get("recalibration", False),
                spec.get("match", False))
    assert shape("served") == (0, True, False)
    assert shape("recal") == (1, False, False)
    assert shape("r1_match") == (1, True, True)
    assert shape("contact") == (3, True, False)
    assert shape("r3_noage") == (3, False, False)
    assert shape("r3_match") == (3, True, True)
    assert shape("r3_noage_match") == (3, False, True)


def test_each_arm_shells_out_with_the_flags_its_name_claims(tmp_path, monkeypatch):
    """`walk` is the only place an arm becomes a command line."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen[tuple(cmd)] = kwargs
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"")
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    for name in mod.ARM_SETS["bas91"]:
        mod.walk(2026, name, tmp_path, 20)
    by_arm = {}
    for cmd in seen:
        out = Path(cmd[cmd.index("--out") + 1]).name
        arm = out[len("chain_engines_"):-len("_2026.parquet")]
        by_arm[arm] = set(cmd)
    assert "--engine-match" in by_arm["r3_noage_match"]
    assert "--engine-no-age" in by_arm["r3_noage_match"]
    assert "--engine-match" in by_arm["r3_match"]
    assert "--engine-no-age" not in by_arm["r3_match"]
    assert "--engine-no-age" in by_arm["r3_noage"]
    assert "--engine-match" not in by_arm["r3_noage"]
    assert "--engine-match" in by_arm["r1_match"]
    assert "--engine-recalibration" in by_arm["recal"]
    assert by_arm["served"].isdisjoint(
        {"--engine-match", "--engine-no-age", "--engine-recalibration"})
    assert "3" in by_arm["r3_noage_match"] and "1" in by_arm["r1_match"]


# ─── the predictions ───

def _arm(base, y, target_diff, target_t, noise_sd, rng):
    """One arm's probabilities: `served` moved by a shift with two parts.

    The truth-correlated part, `d_i * (2y_i - 1)`, is what makes the arm better
    or worse than served; solving its mean for a wanted paired Brier difference
    (Newton, on a function whose derivative is −1 to four decimals) and setting
    its spread from a wanted paired t puts both numbers the predictions are
    decided on into the fixture rather than leaving them to luck. The
    independent part moves the price without moving the score, which is what
    the vacuity check reads and what a real arm mostly does.
    """
    n = len(base)
    sd = abs(target_diff) * np.sqrt(n) / abs(target_t)
    z = rng.normal(0.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)
    e = rng.normal(0.0, noise_sd, n)
    e = e - e.mean()
    mu = -target_diff

    def realized(mu):
        p = np.clip(base + (mu + sd * z) * (2 * y - 1) + e, 1e-6, 1 - 1e-6)
        return float(np.mean((p - y) ** 2 - (base - y) ** 2)), p

    for _ in range(50):
        got, p = realized(mu)
        if abs(got - target_diff) < 1e-12:
            break
        mu += got - target_diff
    return realized(mu)[1]


def _frame(n=800, arms=None, slopes=None, seed=0):
    """A synthetic scored set: a calibrated `served` column plus one per arm.

    `served` is drawn as a real forecast — the outcomes are Bernoulli draws on
    it — so its logistic recalibration slope is near 1 and prediction 1's ".03
    of served" is a statement about the arms rather than about the fixture.
    `arms` maps an arm to `(paired Brier difference vs served, paired t)` and
    optionally a third element, the sd of the score-neutral part of its shift;
    `slopes` stretches an arm's log-odds instead, which is how an arm gets a
    slope of its own without moving its Brier much.
    """
    rng = np.random.default_rng(seed)
    base = np.clip(0.5 + rng.normal(0.0, 0.09, n), 0.05, 0.95)
    y = (rng.random(n) < base).astype(float)
    df = pd.DataFrame({"home_win": y, "served": base})
    for arm in mod.ARM_SETS["bas91"][1:]:
        spec = (arms or {}).get(arm, (0.0005, 3.0))
        p = _arm(base, y, spec[0], spec[1],
                 spec[2] if len(spec) > 2 else 0.006, rng)
        stretch = (slopes or {}).get(arm)
        if stretch is not None:
            odds = np.log(p / (1 - p))
            p = np.clip(1.0 / (1.0 + np.exp(-odds * stretch)), 1e-6, 1 - 1e-6)
        df[arm] = p
    # The production team-strength column the market agreement is measured
    # against: near `served` but not identical to it, so the correlation of two
    # deviations from it is defined.
    df["pythag_60"] = np.clip(base + rng.normal(0.0, 0.02, n), 0.02, 0.98)
    return df


def _predictions(market, all_2026=None, all_2025=None):
    all_2026 = market if all_2026 is None else all_2026
    all_2025 = market if all_2025 is None else all_2025
    pooled = pd.concat([all_2025, all_2026], ignore_index=True)
    return mod.score_predictions_matched(market, all_2026, all_2025, pooled)


# The pre-registration's own picture, as numbers: the candidate helps, the age
# curve costs at matched tables, and matching alone does not rescue rung 1.
PASSING = {"r3_noage_match": (-0.00040, -4.0),
           "r3_match": (-0.00005, -4.0),
           "r1_match": (0.00042, 12.0)}


def test_every_prediction_passes_when_the_numbers_say_so():
    out = _predictions(_frame(arms=PASSING))
    assert out["1_matching_restores_calibration"]["passes"]
    assert out["2_age_curve_costs_at_matched_tables"]["passes"]
    assert out["3_ship_test_r3_noage_match_vs_served"]["passes"]
    assert out["4_null_matching_does_not_rescue_rung1"]["passes"]
    assert out["vacuity_mean_abs_delta_p_home"]["passes"]
    verdict = mod.verdict_matched(out)
    assert verdict["ships"]
    assert not any(verdict[k] for k in verdict if k != "ships")


def test_prediction_1_fails_when_a_matched_arm_is_overconfident():
    """A matched arm whose logistic slope is far from served's is the spread
    story wrong: the transform was supposed to put it back."""
    out = _predictions(_frame(arms=PASSING, slopes={"r3_match": 1.30}))
    p1 = out["1_matching_restores_calibration"]
    assert not p1["passes"]
    assert p1["worst_gap"] > 0.03
    assert mod.verdict_matched(out)["matching_is_not_the_fix"]


def test_prediction_2_fails_when_the_age_curve_costs_nothing():
    arms = {**PASSING, "r3_match": (-0.00040, -4.0)}   # the same arm twice over
    out = _predictions(_frame(arms=arms))
    p2 = out["2_age_curve_costs_at_matched_tables"]
    assert abs(p2["market"]["diff"]) < 0.00030
    assert not p2["passes"]


def test_prediction_2_fails_on_sign_when_the_seasons_disagree():
    market = _frame(arms=PASSING, seed=1)
    flipped = _frame(arms={**PASSING, "r3_match": (-0.00090, -4.0)}, seed=2)
    out = _predictions(market, all_2026=market, all_2025=flipped)
    p2 = out["2_age_curve_costs_at_matched_tables"]
    assert p2["market"]["diff"] >= 0.00030          # the market clause holds
    assert not p2["same_sign_on_all_sets"] and not p2["passes"]


def test_prediction_3_fails_when_the_candidate_is_worse():
    out = _predictions(_frame(arms={**PASSING,
                                    "r3_noage_match": (0.00040, 4.0)}))
    p3 = out["3_ship_test_r3_noage_match_vs_served"]
    assert p3["pooled"]["diff"] > 0 and not p3["passes"] and not p3["ships"]
    verdict = mod.verdict_matched(out)
    assert verdict["ship_test_failed"] and verdict["ship_test_wrong_sign"]
    assert not verdict["ships"]


def test_the_ship_rule_needs_the_t_as_well_as_the_threshold():
    """Prediction 3 holding is not enough: docs/chain-engines-matched.md ships
    only at t <= -2.0 on the pooled set, and its own power statement says
    -.00020 is t ~ -0.8, so the two clauses have to be separable."""
    out = _predictions(_frame(arms={**PASSING,
                                    "r3_noage_match": (-0.00021, -1.2)}))
    p3 = out["3_ship_test_r3_noage_match_vs_served"]
    assert p3["passes"]
    assert p3["pooled"]["t"] > -2.0
    assert not p3["ships"]
    assert not mod.verdict_matched(out)["ships"]


def test_prediction_4_records_m1_dropped_when_rung1_lands_on_served():
    out = _predictions(_frame(arms={**PASSING, "r1_match": (0.00002, 0.2)}))
    p4 = out["4_null_matching_does_not_rescue_rung1"]
    assert not p4["passes"] and p4["within_00010_of_served"]
    assert mod.verdict_matched(out)[
        "m1_dropped_spread_was_the_mechanism_at_rung1"]


def test_the_vacuity_check_fires_when_the_arms_price_the_same_game():
    out = _predictions(_frame(arms={**PASSING,
                                    "r3_noage_match": (-0.00001, -8.0, 0.0)}))
    vac = out["vacuity_mean_abs_delta_p_home"]
    assert not vac["passes"] and vac["value"] < 0.003
    assert mod.verdict_matched(out)["vacuous"]


def test_market_agreement_reports_one_row_per_arm():
    market = _frame(arms=PASSING)
    market["kalshi_close"] = market["served"]
    market["polymarket_close"] = market["served"]
    out = mod.market_agreement(market, mod.ARM_SETS["bas91"])
    assert set(out) == set(mod.ARM_SETS["bas91"])
    assert out["served"]["residual_vs_kalshi"] == pytest.approx(0.0, abs=1e-12)
    assert out["r3_noage_match"]["residual_vs_kalshi"] < 0
