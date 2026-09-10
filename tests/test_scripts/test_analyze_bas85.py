"""BAS-85's scoring pass (docs/bayes-measurement.md).

`scripts/analyze_bas85.py` turns a cell parquet and a fit-record JSON into a
verdict on five pre-registered predictions. The predictions are what they are
— this file does not re-litigate them — but *reading them off* has to be
right, and it is easy to get wrong in ways a real run would hide:

* the May/August split (prediction 2) is on the cutoff's month, and "beats by
  3%" is a signed threshold that a sign error would turn into its opposite;
* "the gap is <= 1.5% in August" is about the size of the difference in
  either direction, so an arm that is 4% *worse* in August must fail it;
* coverage (prediction 4) is `None` for an arm with no interval, not 0% —
  `contact_additive` does not make the claim, and scoring it as a total
  failure would be a lie in the headline; and the interval that gets scored
  is the predictive one, because the thing it is scored against is a rate
  over a finite number of trials and the rate's own posterior does not carry
  that noise;
* the loadings are recorded on one fit record per component out of one fit,
  so counting cutoffs has to deduplicate, or "excludes zero at every cutoff"
  gets easier to satisfy the more components are fit.

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
        "analyze_bas85", ROOT / "scripts/analyze_bas85.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bas85 = _load()


def _cells(gap_by_month, n_batters=40, seed=4):
    """Cells where `bayes_measurement_walk` beats `contact_additive` by a
    known absolute-error margin per cutoff month.

    Both arms predict the same realised rate plus an error; the measurement
    arm's error is the contact arm's scaled by `gap_by_month[month]`, so the
    paired difference is exactly (scale - 1) times the base's MAE and the
    percentage the script reports is known before it runs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for month, scale in gap_by_month.items():
        cutoff = f"2024-{month:02d}-01"
        for b in range(n_batters):
            realized = 0.03 + 0.01 * rng.random()
            err = 0.004 + 0.001 * rng.random()
            for model, s in ((bas85.CONTACT_ARM, 1.0),
                             (bas85.MEASUREMENT_ARM, scale),
                             (bas85.WALK_ARM, 1.2)):
                rows.append({
                    "component": "hr_rate", "model": model, "batter": 1000 + b,
                    "predicted": realized + s * err,
                    "realized_successes": 10, "realized_rate": realized,
                    "trials": 300, "season": 2024, "cutoff": cutoff,
                })
    return pd.DataFrame(rows)


# --- prediction 2: the regime split ----------------------------------------

def test_the_split_is_on_the_cutoffs_month():
    cells = _cells({5: 0.9, 8: 0.99})
    split = bas85.regime_split(cells, bas85.MEASUREMENT_ARM, bas85.CONTACT_ARM,
                               "hr_rate")
    assert set(split) == {"may", "august"}
    assert split["may"]["cutoffs"] == ["2024-05-01"]
    assert split["august"]["cutoffs"] == ["2024-08-01"]


def test_a_real_may_gain_and_a_closed_august_gap_holds():
    """The pre-registered shape: 10% better at May, 1% at August."""
    cells = _cells({5: 0.90, 8: 0.99})
    split = bas85.regime_split(cells, bas85.MEASUREMENT_ARM, bas85.CONTACT_ARM,
                               "hr_rate")
    scored = bas85.score_prediction_2(split)
    assert scored["may"]["pct_of_base_mae"] < bas85.MAY_GAIN_PCT
    assert scored["august"]["holds"]
    assert scored["holds"]


def test_a_may_gain_too_small_to_meet_the_threshold_fails():
    """1% is a gain and is not the pre-registered one. A test that called
    this a pass would let any positive number through."""
    cells = _cells({5: 0.99, 8: 0.995})
    scored = bas85.score_prediction_2(
        bas85.regime_split(cells, bas85.MEASUREMENT_ARM, bas85.CONTACT_ARM,
                           "hr_rate"))
    assert not scored["may"]["holds"]
    assert not scored["holds"]


def test_an_august_gap_in_the_wrong_direction_fails():
    """"The gap is <= 1.5%" is about the size of the difference either way.
    An arm 5% WORSE in August has not converged with the served engine; it
    has diverged from it, which is a different result and not this one."""
    cells = _cells({5: 0.90, 8: 1.05})
    scored = bas85.score_prediction_2(
        bas85.regime_split(cells, bas85.MEASUREMENT_ARM, bas85.CONTACT_ARM,
                           "hr_rate"))
    assert scored["august"]["pct_of_base_mae"] > 0
    assert not scored["august"]["holds"]
    assert not scored["holds"]


def test_a_missing_regime_is_not_a_pass():
    """A grid that never ran an August cutoff cannot satisfy prediction 2 by
    having nothing to fail on."""
    cells = _cells({5: 0.90})
    scored = bas85.score_prediction_2(
        bas85.regime_split(cells, bas85.MEASUREMENT_ARM, bas85.CONTACT_ARM,
                           "hr_rate"))
    assert "august" not in scored
    assert not scored["holds"]


# --- prediction 3: pooled ---------------------------------------------------

def test_pooled_needs_a_gain_on_hr_and_a_draw_on_k():
    good = bas85.score_prediction_3({
        "hr_rate": {"diff": -0.0009, "clustered_by_player_t": -3.1},
        "k_rate": {"diff": 0.0002},
    })
    assert good["holds"]
    # A gain on HR/PA that comes with a real move on K% is not the
    # pre-registered result: the model was supposed to leave K% alone.
    assert not bas85.score_prediction_3({
        "hr_rate": {"diff": -0.0009, "clustered_by_player_t": -3.1},
        "k_rate": {"diff": 0.0031},
    })["holds"]
    # A gain the clustered t does not support.
    assert not bas85.score_prediction_3({
        "hr_rate": {"diff": -0.0009, "clustered_by_player_t": -1.1},
        "k_rate": {"diff": 0.0002},
    })["holds"]


# --- prediction 4: coverage -------------------------------------------------

def _coverage_cells(fraction_inside, model, n=200, month=5, seed=9,
                    sd=0.0001):
    """Cells where a known fraction of realised rates land inside the band.

    The rate-only band is written directly (`pred_q10`/`pred_q90`). `pred_sd`
    is tiny by default so that the *predictive* band, which is the posterior
    sd and the binomial sd in quadrature, is dominated by the binomial term
    and is a known width — which lets the two readings be tested apart.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        realized = 0.03
        inside = i < int(round(fraction_inside * n))
        rows.append({
            "component": "hr_rate", "model": model, "batter": 1000 + i,
            "predicted": 0.03, "realized_successes": 10,
            "realized_rate": realized if inside else 0.30,
            "trials": 300, "season": 2024,
            "cutoff": f"2024-{month:02d}-01",
            "pred_q10": 0.02 if inside else 0.05,
            "pred_q90": 0.04 if inside else 0.06,
            "pred_sd": sd,
        })
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def test_coverage_counts_cells_not_trials():
    """The claim is "how often is an interval right", and every batter's
    interval is one claim — weighting by playing time would answer a
    different question."""
    cells = _coverage_cells(0.8, bas85.MEASUREMENT_ARM)
    cov = bas85.coverage(cells, bas85.MEASUREMENT_ARM, "hr_rate")
    assert cov["n"] == 200
    assert cov["covered"] == pytest.approx(0.8)
    assert cov["mean_width"] > 0


def test_the_predictive_band_is_wider_than_the_rate_band():
    """The realised rate carries binomial noise the rate's posterior does
    not, so the interval that can cover it is the wider one. If these two
    ever came out equal the binomial term would have been dropped."""
    cells = _coverage_cells(0.8, bas85.MEASUREMENT_ARM)
    rate = bas85.coverage(cells, bas85.MEASUREMENT_ARM, "hr_rate")
    pred = bas85.coverage(cells, bas85.MEASUREMENT_ARM, "hr_rate",
                          predictive=True)
    assert pred["mean_width"] > rate["mean_width"]
    # 300 trials at p=0.03: binomial sd ~0.0099, so an 80% band is ~+-0.0127.
    assert pred["mean_width"] == pytest.approx(2 * 1.2816 * 0.00985, rel=0.02)


def test_the_verdict_is_taken_on_the_predictive_reading(monkeypatch):
    """Both readings are reported; only one is scored, and the payload says
    which — a reader who quotes the wrong one should be able to see that it
    is not the number the verdict came from."""
    cells = pd.concat([_coverage_cells(0.80, bas85.MEASUREMENT_ARM),
                       _coverage_cells(0.60, bas85.WALK_ARM, seed=10)],
                      ignore_index=True)
    scored = bas85.score_prediction_4(cells, "hr_rate")
    assert scored["scored_on"] == "predictive"
    assert set(scored["predictive"]) == set(scored["rate_only"])
    assert (scored["predictive"][bas85.MEASUREMENT_ARM]["mean_width"]
            > scored["rate_only"][bas85.MEASUREMENT_ARM]["mean_width"])


def test_an_arm_with_no_interval_scores_none_not_zero():
    """`contact_additive` has no posterior at all. Reporting 0% coverage for
    it would read as a catastrophic miscalibration of an arm that never made
    the claim."""
    cells = _coverage_cells(0.8, bas85.MEASUREMENT_ARM)
    plain = cells.drop(columns=["pred_q10", "pred_q90", "pred_sd"]).assign(
        model=bas85.CONTACT_ARM)
    for predictive in (False, True):
        cov = bas85.coverage(plain, bas85.CONTACT_ARM, "hr_rate",
                             predictive=predictive)
        assert cov["covered"] is None
        assert cov["n"] == 0


def test_the_prediction_needs_both_halves():
    """80% coverage for `measurement` AND under 75% for `bayes_walk` at May.
    A measurement arm that is well calibrated while the walk arm is too, is
    a calibration result, not the pre-registered one."""
    meas = _coverage_cells(0.80, bas85.MEASUREMENT_ARM)
    walk_bad = _coverage_cells(0.60, bas85.WALK_ARM, seed=10)
    walk_ok = _coverage_cells(0.80, bas85.WALK_ARM, seed=11)
    assert bas85.score_prediction_4(
        pd.concat([meas, walk_bad], ignore_index=True), "hr_rate")["holds"]
    assert not bas85.score_prediction_4(
        pd.concat([meas, walk_ok], ignore_index=True), "hr_rate")["holds"]
    # And a measurement arm whose interval is far too wide fails on its own.
    assert not bas85.score_prediction_4(
        pd.concat([_coverage_cells(0.97, bas85.MEASUREMENT_ARM, seed=12),
                   walk_bad], ignore_index=True), "hr_rate")["holds"]


# --- predictions 1 and 5: the loadings --------------------------------------

def _fit(cutoff, component, loadings, sigma=0.30, max_corr=0.4):
    return {
        "cutoff": cutoff, "component": component,
        "measurement_params": {
            **{name: {"mean": m, "q2.5": lo, "q97.5": hi}
               for name, (m, lo, hi) in loadings.items()},
            "sigma_ability": {"k_rate": sigma, "hr_rate": sigma},
            "max_abs_loading_corr": max_corr,
        },
    }


LOADS = {"lambda_barrel": (0.6, 0.4, 0.8),
         "lambda_ev": (0.5, 0.3, 0.7),
         "lambda_whiff": (0.7, 0.5, 0.9)}
STRADDLES = {**LOADS, "lambda_ev": (0.05, -0.3, 0.4)}


def test_one_fit_recorded_per_component_counts_as_one_cutoff():
    """Two components come out of one measurement fit, so the same posterior
    lands on two records. Counting them as two cutoffs would make "excludes
    zero at every cutoff" easier the more components are fit, which is
    exactly backwards."""
    fits = [_fit("2024-05-01", "k_rate", LOADS),
            _fit("2024-05-01", "hr_rate", LOADS)]
    by_cutoff = bas85.loadings_by_cutoff(fits)
    assert by_cutoff["lambda_ev"] == {"2024-05-01": {
        "mean": 0.5, "q2.5": 0.3, "q97.5": 0.7, "mph_per_sd": None}}
    assert bas85.score_prediction_1(by_cutoff)["lambda_ev"]["n_cutoffs"] == 1


def test_prediction_1_needs_every_cutoff_not_most_of_them():
    fits = [_fit("2024-05-01", "hr_rate", LOADS),
            _fit("2024-08-01", "hr_rate", STRADDLES)]
    scored = bas85.score_prediction_1(bas85.loadings_by_cutoff(fits))
    assert scored["lambda_barrel"]["holds"]
    assert scored["lambda_ev"]["n_excluding_zero"] == 1
    assert scored["lambda_ev"]["n_cutoffs"] == 2
    assert not scored["lambda_ev"]["holds"]
    assert not scored["holds"]


def test_prediction_1_with_no_fits_at_all_fails():
    """Nothing recorded is not evidence for the prediction."""
    assert not bas85.score_prediction_1({})["holds"]


def test_prediction_5_catches_a_collapsed_latent_and_a_degenerate_pair():
    ok = [_fit("2024-05-01", "hr_rate", LOADS)]
    assert bas85.score_prediction_5(ok)["holds"]

    flat = [_fit("2024-05-01", "hr_rate", LOADS, sigma=0.005)]
    scored = bas85.score_prediction_5(flat)
    assert scored["sigma_ability"]["hr_rate"]["n_below_floor"] == 1
    assert not scored["holds"]

    twin = [_fit("2024-05-01", "hr_rate", LOADS, max_corr=0.98)]
    scored = bas85.score_prediction_5(twin)
    assert len(scored["degenerate"]) == 1
    assert not scored["holds"]


def test_prediction_5_with_no_fits_at_all_fails():
    assert not bas85.score_prediction_5([])["holds"]


# --- per-season and provenance ----------------------------------------------

def test_by_season_separates_a_one_season_result_from_a_repeated_one():
    """A gain that exists in one season and not the other is a different
    claim from one that repeats, and the pooled table cannot tell them
    apart. The grid runs one season per process and is scored after each, so
    this is the read that partial results actually get."""
    a = _cells({5: 0.90, 8: 0.99})
    b = _cells({5: 1.02, 8: 1.01}, seed=7)
    b["season"] = 2025
    b["cutoff"] = b["cutoff"].str.replace("2024", "2025", regex=False)
    per = bas85.by_season(pd.concat([a, b], ignore_index=True), "hr_rate")

    assert sorted(per) == [2024, 2025]
    got = {s: [r for r in rows if r["base"] == bas85.CONTACT_ARM][0]
           for s, rows in per.items()}
    assert got[2024]["pct_of_base_mae"] < -3.0     # the real gain
    assert got[2025]["pct_of_base_mae"] > 0        # and the season without it


def test_a_grid_mixing_samplers_is_flagged_not_pooled_silently(tmp_path, capsys):
    """Two measurement fits from different backends are not one table.

    NumPyro and PyMC disagree on this graph (the HR latent has a scale/sign
    ridge -- see data/eval/bas85/NUMPYRO_NON_IDENTIFIABILITY.md), so a
    checkpoint that resumed under a different `--bayes-sampler` would pool
    two incompatible posteriors into one row and nothing downstream would
    say so.
    """
    cells = _cells({5: 0.90, 8: 0.99})
    cells.to_parquet(tmp_path / "cells_bayes.parquet", index=False)
    fits = [_fit("2024-05-01", "hr_rate", LOADS),
            _fit("2024-08-01", "hr_rate", LOADS)]
    fits[0]["sampler"], fits[0]["parameterisation"] = "pymc", "centred"
    fits[1]["sampler"], fits[1]["parameterisation"] = "numpyro", "centred"
    (tmp_path / "bayes_fits.json").write_text(json.dumps(fits))

    sys.argv = ["analyze_bas85.py", "--in-dir", str(tmp_path)]
    bas85.main()
    out = capsys.readouterr().out
    assert "more than one" in out and "not comparable" in out

    payload = json.loads((tmp_path / "analysis_bas85.json").read_text())
    assert len(payload["provenance"]) == 2


# --- the converged-only sensitivity -----------------------------------------

def test_a_chain_disagreement_is_caught_from_the_fit_record_alone():
    """`max_abs_loading_corr` at ~1 and a high R-hat both say the two chains
    landed on different scales of the same latent. Read off the record, so it
    costs nothing and works on a checkpoint a running grid is still writing."""
    ok = _fit("2024-05-01", "hr_rate", LOADS, max_corr=0.56)
    ok["diagnostics"] = {"max_rhat": 1.04}
    ridge = _fit("2024-06-24", "hr_rate", LOADS, max_corr=0.999)
    ridge["diagnostics"] = {"max_rhat": 1.856}
    noisy = _fit("2024-07-08", "hr_rate", LOADS, max_corr=0.60)
    noisy["diagnostics"] = {"max_rhat": 1.9}

    bad = bas85.degenerate_fits([ok, ridge, noisy])
    assert set(bad) == {"2024-06-24", "2024-07-08"}
    assert "loading corr" in bad["2024-06-24"] and "R-hat" in bad["2024-06-24"]
    assert "R-hat" in bad["2024-07-08"]


def test_a_broken_fit_is_reported_both_ways_and_never_dropped_silently(
        tmp_path, capsys):
    """The pre-registered analysis scores every cutoff. A non-converged fit
    still moves the pooled MAE -- its projections are averaged over a chain
    whose latent collapsed, so they carry about half the spread -- so the
    converged-only table is a *sensitivity* printed beside the headline, with
    the excluded cutoffs named, not a quiet filter."""
    good = _cells({5: 0.90})
    broken = _cells({7: 1.30}, seed=8)
    pd.concat([good, broken], ignore_index=True).to_parquet(
        tmp_path / "cells_bayes.parquet", index=False)
    fits = [_fit("2024-05-01", "hr_rate", LOADS, max_corr=0.5),
            _fit("2024-07-01", "hr_rate", LOADS, max_corr=0.999)]
    for f in fits:
        f["diagnostics"] = {"max_rhat": 1.05}
    (tmp_path / "bayes_fits.json").write_text(json.dumps(fits))

    sys.argv = ["analyze_bas85.py", "--in-dir", str(tmp_path)]
    bas85.main()
    out = capsys.readouterr().out
    assert "did not converge" in out and "2024-07-01" in out

    payload = json.loads((tmp_path / "analysis_bas85.json").read_text())
    assert list(payload["degenerate_fits"]) == ["2024-07-01"]
    assert payload["converged_only"]["excluded_cutoffs"] == ["2024-07-01"]
    assert payload["converged_only"]["n_cutoffs_kept"] == 1
    # The headline still carries every cutoff.
    pooled = [r for r in payload["comparisons"]["hr_rate"]
              if r["base"] == bas85.CONTACT_ARM][0]
    assert pooled["n"] > payload["converged_only"]["comparisons"]["hr_rate"][0]["n"]


def test_no_degenerate_fits_means_no_sensitivity_section():
    """A clean grid should not grow a second set of tables saying the same
    thing as the first."""
    assert bas85.degenerate_fits([]) == {}
    ok = _fit("2024-05-01", "hr_rate", LOADS, max_corr=0.5)
    ok["diagnostics"] = {"max_rhat": 1.02}
    assert bas85.degenerate_fits([ok]) == {}


# --- the whole pass ---------------------------------------------------------

def test_the_script_runs_end_to_end_and_writes_its_verdict(tmp_path, capsys):
    """The renderers and `main` on a frame with every arm present — the path
    a grid actually takes, so a formatting error in a table nobody unit-tests
    still fails here."""
    cells = _cells({5: 0.90, 8: 0.99})
    cov = pd.concat([_coverage_cells(0.80, bas85.MEASUREMENT_ARM),
                     _coverage_cells(0.60, bas85.WALK_ARM, seed=10)],
                    ignore_index=True)
    pd.concat([cells, cov], ignore_index=True).to_parquet(
        tmp_path / "cells_bayes.parquet", index=False)
    (tmp_path / "bayes_fits.json").write_text(
        pd.Series([_fit("2024-05-01", "hr_rate", LOADS),
                   _fit("2024-08-01", "hr_rate", LOADS)]).to_json(orient="values"))

    sys.argv = ["analyze_bas85.py", "--in-dir", str(tmp_path)]
    bas85.main()
    out = capsys.readouterr().out
    assert "prediction_1_loadings" in out and "HOLDS" in out
    assert "channel loadings by cutoff" in out
    payload = pd.read_json(tmp_path / "analysis_bas85.json", typ="series")
    assert payload["prediction_1_loadings"]["holds"] is True
    assert payload["prediction_2_early_season"]["holds"] is True
