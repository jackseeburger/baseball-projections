"""BAS-94's scoring pass (docs/bayes-prior-mean.md).

`scripts/analyze_bas94.py` turns a cell parquet, a borrowed comparator parquet
and a fit-record JSON into a verdict on five pre-registered predictions. The
predictions are what they are — this file does not re-litigate them — but
*reading them off* has to be right, and it is easy to get wrong in ways a real
run would hide:

* vacuity (prediction 1) is **two** claims joined by "and": a coefficient
  excluding zero at >= 90% of cutoffs AND the prior mean's between-player sd
  being >= 30% of `sigma_ability`. A model whose gammas are all significant
  and whose target barely moves has to FAIL, because that is exactly the
  relabelled league mean the prediction exists to rule out;
* the HR/PA half of it is "barrel **or** EV", so one of three coefficients
  excluding zero at a cutoff is enough there, while K% needs whiff;
* "the August gap is at most half the May gap" (prediction 4) is a statement
  about a *gain* shrinking — an arm that lost in May and lost less in August
  must not read as a pass;
* the comparator arms are spliced in from another ticket's grid, so only the
  cells this run actually scored may come across, and `marcel_tuned` — the one
  arm both grids computed — has to be audited rather than assumed;
* the width half of prediction 5 is read on the RATE interval, not the
  predictive one, since the predictive band adds the same binomial term to
  both arms.

Every case here is built from a hand-made frame with the answer known in
advance. Nothing samples and nothing reads a real artifact.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "analyze_bas94", ROOT / "scripts/analyze_bas94.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bas94 = _load()

MAY = ("2024-05-01", "2024-05-13")
AUG = ("2024-08-01", "2024-08-05")


def _cells(scale_by_month, arms=(bas94.ARM_A,), base=bas94.WALK_ARM,
           n_batters=40, seed=4, component="hr_rate",
           cutoffs=MAY + AUG, season=2024):
    """Cells where each arm's absolute error is a known multiple of the base's.

    The paired difference is then exactly `(scale - 1)` times the base's MAE,
    so the percentage the script reports is known before it runs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cutoff in cutoffs:
        month = int(pd.Timestamp(cutoff).month)
        for b in range(n_batters):
            realized = float(rng.uniform(0.02, 0.05))
            err = float(rng.uniform(0.002, 0.01))
            trials = int(rng.integers(150, 400))
            common = {"component": component, "batter": 1000 + b,
                      "realized_successes": int(realized * trials),
                      "realized_rate": realized, "trials": trials,
                      "season": season, "cutoff": cutoff}
            rows.append({**common, "model": base,
                         "predicted": realized + err,
                         "pred_q10": realized + err - 0.01,
                         "pred_q90": realized + err + 0.01,
                         "pred_sd": 0.004})
            for arm in arms:
                scale = scale_by_month[arm][month]
                rows.append({**common, "model": arm,
                             "predicted": realized + err * scale,
                             "pred_q10": realized + err * scale - 0.005,
                             "pred_q90": realized + err * scale + 0.005,
                             "pred_sd": 0.002})
    return pd.DataFrame(rows)


def _fit(arm, component, cutoff, gammas, sd_ratio=0.6, rhat=1.01,
         divergences=0):
    return {
        "arm": arm, "component": component, "cutoff": cutoff,
        "variant": "ability_walk+prior_contact", "sampler": "numpyro",
        "elapsed_s": 90.0,
        "diagnostics": {"max_rhat": rhat, "max_rhat_var": "z_step",
                        "min_ess_bulk": 300.0, "divergences": divergences},
        "prior_mean_params": {
            "set": "contact",
            "features": sorted(gammas),
            "gamma": {name: {"mean": 0.2 if excl else 0.0, "sd": 0.05,
                             "q05": 0.1 if excl else -0.1,
                             "q95": 0.3 if excl else 0.1,
                             "excludes_zero": excl}
                      for name, excl in gammas.items()},
            "prior_mean_sd_between_players": sd_ratio * 0.4,
            "sigma_ability": 0.4,
            "sd_ratio": sd_ratio,
            "sd_ratio_q05": sd_ratio * 0.9, "sd_ratio_q95": sd_ratio * 1.1,
        },
    }


# ─── prediction 1: both halves ─────────────────────────────────────────────

class TestPrediction1:
    def _fits(self, gammas, sd_ratio):
        return [_fit(bas94.ARM_A, "hr_rate", c, gammas, sd_ratio)
                for c in MAY + AUG]

    def test_holds_when_a_coefficient_moves_and_the_target_spreads(self):
        out = bas94.score_prediction_1(
            self._fits({"gamma_barrel": True, "gamma_ev_mean": False,
                        "gamma_ev90": False}, 0.6))
        assert out["gamma"]["hr_rate"]["share"] == 1.0
        assert out["sd_ratio"]["hr_rate"]["mean"] == pytest.approx(0.6)
        assert out["holds"] is True

    def test_fails_when_the_target_barely_moves(self):
        """A significant gamma whose prior mean spreads players by 5% of
        `sigma_ability` is a relabelled league mean. Both halves are required
        and this is the one that catches it."""
        out = bas94.score_prediction_1(
            self._fits({"gamma_barrel": True}, 0.05))
        assert out["gamma"]["hr_rate"]["holds"] is True
        assert out["sd_ratio"]["hr_rate"]["holds"] is False
        assert out["holds"] is False

    def test_fails_when_no_coefficient_excludes_zero(self):
        out = bas94.score_prediction_1(
            self._fits({"gamma_barrel": False, "gamma_ev_mean": False,
                        "gamma_ev90": False}, 0.9))
        assert out["gamma"]["hr_rate"]["n_excluding_zero"] == 0
        assert out["holds"] is False

    def test_hr_pa_passes_a_cutoff_on_barrel_or_ev(self):
        """"barrel or EV": one of the three excluding zero is enough, and the
        per-coefficient counts still show which one did it."""
        fits = [_fit(bas94.ARM_A, "hr_rate", c,
                     {"gamma_barrel": False, "gamma_ev_mean": False,
                      "gamma_ev90": True}, 0.6) for c in MAY + AUG]
        out = bas94.score_prediction_1(fits)
        g = out["gamma"]["hr_rate"]
        assert g["share"] == 1.0 and g["holds"] is True
        assert g["per_coefficient"]["gamma_barrel"]["n_excluding_zero"] == 0
        assert g["per_coefficient"]["gamma_ev90"]["n_excluding_zero"] == 4

    def test_k_rate_needs_whiff(self):
        fits = [_fit(bas94.ARM_A, "k_rate", c,
                     {"gamma_whiff": False, "gamma_barrel": True}, 0.6)
                for c in MAY + AUG]
        out = bas94.score_prediction_1(fits)
        assert out["gamma"]["k_rate"]["n_excluding_zero"] == 0
        assert out["holds"] is False

    def test_the_ninety_percent_threshold_is_a_share_not_all(self):
        """"at >= 90% of cutoffs", so one miss in ten passes and one in five
        does not."""
        cuts = [f"2024-0{4 + i // 3}-{1 + 7 * (i % 3):02d}" for i in range(10)]
        fits = [_fit(bas94.ARM_A, "hr_rate", c,
                     {"gamma_barrel": i > 0}, 0.6)
                for i, c in enumerate(cuts)]
        out = bas94.score_prediction_1(fits)
        assert out["gamma"]["hr_rate"]["share"] == pytest.approx(0.9)
        assert out["holds"] is True

        fits = [_fit(bas94.ARM_A, "hr_rate", c,
                     {"gamma_barrel": i > 1}, 0.6)
                for i, c in enumerate(cuts)]
        assert bas94.score_prediction_1(fits)["holds"] is False


# ─── predictions 2 and 4: the regime split ─────────────────────────────────

class TestRegimeSplit:
    def _splits(self, may_scale, aug_scale, component="hr_rate"):
        cells = _cells({bas94.ARM_A: {5: may_scale, 8: aug_scale}},
                       component=component)
        return {component: bas94.regime_split(cells, bas94.ARM_A,
                                              bas94.WALK_ARM, component)}

    def test_may_gain_is_measured_as_a_percentage_of_the_base_mae(self):
        splits = self._splits(0.90, 0.99)
        may = splits["hr_rate"]["may"]
        assert may["pct_of_base_mae"] == pytest.approx(-10.0, abs=0.01)

    def test_prediction_2_holds_on_a_ten_percent_may_gain(self):
        out = bas94.score_prediction_2(self._splits(0.90, 0.99))
        assert out["hr_rate"]["holds"] is True

    def test_prediction_2_fails_on_a_one_percent_may_gain(self):
        out = bas94.score_prediction_2(self._splits(0.99, 0.995))
        assert out["hr_rate"]["holds"] is False

    def test_prediction_2_thresholds_differ_by_component(self):
        """3% on HR/PA, 2% on K% — a 2.5% gain passes one and not the other."""
        hr = bas94.score_prediction_2(self._splits(0.975, 0.99, "hr_rate"))
        k = bas94.score_prediction_2(self._splits(0.975, 0.99, "k_rate"))
        assert hr["hr_rate"]["holds"] is False
        assert k["k_rate"]["holds"] is True

    def test_prediction_4_holds_when_the_gain_shrinks(self):
        out = bas94.score_prediction_4(self._splits(0.90, 0.98))
        assert out["hr_rate"]["holds"] is True

    def test_prediction_4_fails_when_the_gain_does_not_shrink(self):
        out = bas94.score_prediction_4(self._splits(0.90, 0.89))
        assert out["hr_rate"]["holds"] is False

    def test_prediction_4_fails_when_may_was_not_a_gain(self):
        """An arm 10% worse in May and 2% worse in August "shrank", and is not
        the mechanism the prediction describes."""
        out = bas94.score_prediction_4(self._splits(1.10, 1.02))
        assert out["hr_rate"]["may_is_a_gain"] is False
        assert out["hr_rate"]["holds"] is False


# ─── prediction 3 ──────────────────────────────────────────────────────────

class TestPrediction3:
    def test_a_gain_against_the_walk_and_a_draw_against_the_served_engine(self):
        pooled_walk = {"hr_rate": {"pct_of_base_mae": -3.0, "diff": -0.0003,
                                   "clustered_by_player_t": -4.0},
                       "k_rate": {"pct_of_base_mae": -2.0, "diff": -0.0005,
                                  "clustered_by_player_t": -3.0}}
        pooled_contact = {"hr_rate": {"pct_of_base_mae": 0.4, "diff": 0.00004,
                                      "clustered_by_player_t": 0.6}}
        out = bas94.score_prediction_3(pooled_walk, pooled_contact)
        assert out["holds"] is True

    def test_a_gain_without_the_t_fails_on_hr_pa(self):
        pooled_walk = {"hr_rate": {"pct_of_base_mae": -3.0, "diff": -0.0003,
                                   "clustered_by_player_t": -1.2}}
        out = bas94.score_prediction_3(pooled_walk, {})
        assert out["vs_bayes_walk_hr_rate"]["holds"] is False

    def test_the_draw_band_is_two_sided(self):
        """"within +-1.5%" — a 3% WIN over the served engine is outside the
        band the pre-registration wrote down, and is reported as such rather
        than quietly upgraded to a pass."""
        out = bas94.score_prediction_3(
            {}, {"hr_rate": {"pct_of_base_mae": -3.0, "diff": -0.0003,
                             "clustered_by_player_t": -4.0}})
        assert out["vs_contact_additive_hr_rate"]["holds"] is False


# ─── prediction 5 ──────────────────────────────────────────────────────────

class TestPrediction5:
    def test_width_is_compared_on_the_rate_interval(self):
        """Arm A's rate band is +-0.005 and the walk's +-0.01, so arm A is the
        narrower one; the predictive band adds the same binomial term to both
        and must not be what decides it."""
        cells = _cells({bas94.ARM_A: {5: 0.95, 8: 0.99}})
        out = bas94.score_prediction_5(cells, "hr_rate")
        assert out["width_scored_on"] == "rate_only"
        assert out["may_width_arm_a"] == pytest.approx(0.01, abs=1e-9)
        assert out["may_width_bayes_walk"] == pytest.approx(0.02, abs=1e-9)
        assert out["narrower_at_may"] is True

    def test_an_arm_with_no_interval_scores_none_not_zero(self):
        """`contact_additive` makes no interval claim; reporting 0% coverage
        would read as a catastrophic failure of an arm that never made it."""
        cells = _cells({bas94.ARM_A: {5: 0.95, 8: 0.99}})
        contact = cells[cells["model"] == bas94.WALK_ARM].copy()
        contact["model"] = bas94.CONTACT_ARM
        for c in ("pred_q10", "pred_q90", "pred_sd"):
            contact[c] = np.nan
        out = bas94.score_prediction_5(
            pd.concat([cells, contact], ignore_index=True), "hr_rate")
        assert out["predictive"][bas94.CONTACT_ARM]["covered"] is None


# ─── the comparator splice ─────────────────────────────────────────────────

class TestTheSplice:
    def _write(self, tmp_path):
        own = _cells({bas94.ARM_A: {5: 0.95, 8: 0.99}})
        marcel = own[own["model"] == bas94.WALK_ARM].copy()
        marcel["model"] = bas94.MARCEL_ARM
        own = pd.concat([own[own["model"] == bas94.ARM_A], marcel],
                        ignore_index=True)

        other = _cells({bas94.ARM_A: {5: 0.95, 8: 0.99, 9: 0.99}},
                       cutoffs=MAY + AUG + ("2024-09-01",))
        walk = other[other["model"] == bas94.WALK_ARM].copy()
        contact = walk.copy()
        contact["model"] = bas94.CONTACT_ARM
        marcel2 = walk.copy()
        marcel2["model"] = bas94.MARCEL_ARM
        other = pd.concat([walk, contact, marcel2], ignore_index=True)

        (tmp_path / "own").mkdir()
        (tmp_path / "other").mkdir()
        own.to_parquet(tmp_path / "own/cells_bayes.parquet", index=False)
        other.to_parquet(tmp_path / "other/cells_bayes.parquet", index=False)
        return tmp_path / "own", tmp_path / "other"

    def test_only_the_comparator_arms_and_only_our_cells_come_across(self, tmp_path):
        a, b = self._write(tmp_path)
        cells, audit = bas94.load_cells(a, b)
        assert set(cells["model"]) == {bas94.ARM_A, bas94.MARCEL_ARM,
                                       bas94.WALK_ARM, bas94.CONTACT_ARM}
        # The September cutoff exists only in the borrowed grid and must not
        # inflate a pooled table with a cell no prior-mean arm ever saw.
        assert "2024-09-01" not in set(cells["cutoff"])
        assert audit["comparator_cells_matched"] == audit["comparator_cells_wanted"]

    def test_marcel_is_audited_rather_than_assumed(self, tmp_path):
        a, b = self._write(tmp_path)
        _, audit = bas94.load_cells(a, b)
        assert audit["marcel_audit"]["n_common"] > 0
        assert audit["marcel_audit"]["max_abs_diff"] == pytest.approx(0.0)

    def test_a_disagreeing_marcel_shows_up_in_the_audit(self, tmp_path):
        a, b = self._write(tmp_path)
        other = pd.read_parquet(b / "cells_bayes.parquet")
        bad = other["model"] == bas94.MARCEL_ARM
        other.loc[bad, "predicted"] = other.loc[bad, "predicted"] + 0.01
        other.to_parquet(b / "cells_bayes.parquet", index=False)
        _, audit = bas94.load_cells(a, b)
        assert audit["marcel_audit"]["max_abs_diff"] == pytest.approx(0.01)

    def test_a_missing_comparator_grid_is_reported_not_fatal(self, tmp_path):
        a, _ = self._write(tmp_path)
        cells, audit = bas94.load_cells(a, tmp_path / "nowhere")
        assert "comparators" in audit
        assert set(cells["model"]) == {bas94.ARM_A, bas94.MARCEL_ARM}


# ─── convergence ───────────────────────────────────────────────────────────

class TestConvergence:
    def test_a_bad_rhat_is_named_and_its_whole_cell_leaves_the_sensitivity(self):
        fits = [_fit(bas94.ARM_A, "hr_rate", c, {"gamma_barrel": True})
                for c in MAY]
        fits.append(_fit(bas94.ARM_A, "hr_rate", AUG[0],
                         {"gamma_barrel": True}, rhat=1.9))
        bad = bas94.nonconverged_fits(fits)
        assert list(bad) == [f"{bas94.ARM_A}|hr_rate|{AUG[0]}"]

        cells = _cells({bas94.ARM_A: {5: 0.95, 8: 0.99}})
        clean = bas94.drop_nonconverged(cells, bad)
        # Every arm's rows for that cell go, not just the broken arm's.
        assert AUG[0] not in set(clean["cutoff"])
        assert set(clean["model"]) == set(cells["model"])

    def test_a_clean_grid_names_nothing(self):
        fits = [_fit(bas94.ARM_A, "hr_rate", c, {"gamma_barrel": True})
                for c in MAY + AUG]
        assert bas94.nonconverged_fits(fits) == {}

    def test_the_diagnostics_summary_counts_what_it_says(self):
        fits = [_fit(bas94.ARM_A, "hr_rate", c, {"gamma_barrel": True},
                     rhat=1.01, divergences=i) for i, c in enumerate(MAY + AUG)]
        s = bas94.fit_diagnostics_summary(fits)[bas94.ARM_A]
        assert s["n_fits"] == 4
        assert s["divergences"]["total"] == 0 + 1 + 2 + 3
        assert s["divergences"]["n_fits_with_any"] == 3
        assert s["max_rhat"]["n_above_ceiling"] == 0
        assert s["samplers"] == ["numpyro"]


# ─── the verdict ───────────────────────────────────────────────────────────

def _payload(p1, p2, contact_t=0.5, contact_diff=0.00001):
    return {
        "prediction_1_vacuity": {"holds": p1},
        "prediction_2_may": {"holds": p2},
        "prediction_3_pooled": {"holds": False},
        "prediction_4_mechanism": {"holds": False},
        "prediction_5_calibration": {"hr_rate": {"holds": False}},
        "pooled_vs_contact_additive": {
            "hr_rate": {"diff": contact_diff,
                        "clustered_by_player_t": contact_t}},
    }


class TestVerdict:
    def test_prediction_1_failing_closes_the_structural_track(self):
        v = bas94.verdict(_payload(False, True))
        assert "structural track is closed" in v["headline"]
        assert v["ships"] is False

    def test_prediction_2_failing_with_1_holding_ships_nothing(self):
        v = bas94.verdict(_payload(True, False))
        assert "Nothing ships" in v["headline"]

    def test_beating_the_served_engine_opens_a_serving_ticket(self):
        v = bas94.verdict(_payload(True, True, contact_t=-3.0,
                                   contact_diff=-0.0004))
        assert v["ships"] is True
        assert "serving ticket" in v["headline"]

    def test_a_draw_with_the_served_engine_stays_behind_the_flag(self):
        v = bas94.verdict(_payload(True, True))
        assert v["ships"] is False
        assert "behind the flag" in v["headline"]


# ─── end to end ────────────────────────────────────────────────────────────

def test_build_payload_and_result_md_render(tmp_path):
    cells = _cells({bas94.ARM_A: {5: 0.95, 8: 0.99},
                    bas94.ARM_B: {5: 0.97, 8: 1.01}},
                   arms=(bas94.ARM_A, bas94.ARM_B))
    contact = cells[cells["model"] == bas94.WALK_ARM].copy()
    contact["model"] = bas94.CONTACT_ARM
    marcel = contact.copy()
    marcel["model"] = bas94.MARCEL_ARM
    cells = pd.concat([cells, contact, marcel], ignore_index=True)
    fits = [_fit(bas94.ARM_A, "hr_rate", c, {"gamma_barrel": True})
            for c in MAY + AUG]

    payload = bas94.build_payload(cells, fits, {"own_rows": len(cells)})
    payload["nonconverged_fits"] = {}
    payload["verdict"] = bas94.verdict(payload)
    md = bas94.render_result_md(payload)

    assert bas94.ARM_A in md and bas94.ARM_B in md
    assert "Verdict:" in md
    # JSON-serialisable with the default the script uses.
    json.dumps(payload, default=float)
