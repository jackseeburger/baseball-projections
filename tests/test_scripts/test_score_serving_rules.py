"""The covariate-share serving rules (BAS-82), driven by a synthetic cell table."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "score_serving_rules", ROOT / "scripts" / "score_serving_rules.py")
ssr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ssr)


def synthetic_cells(n_clusters=200, per_cluster=3, effect=-1.0, noise=1.0,
                    cluster_sd=1.0, seed=7):
    """A player-clustered paired-difference table with a known effect.

    Each player gets a shared offset — the thing that makes rows within a
    player correlated and the clustered SE larger than the naive one — plus
    independent per-cell noise.
    """
    rng = np.random.default_rng(seed)
    offs = rng.normal(0.0, cluster_sd, n_clusters)
    rows = []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            rows.append({"player": c, "trials": 100.0,
                         "d": effect + offs[c] + rng.normal(0.0, noise)})
    return pd.DataFrame(rows)


def test_clustered_stats_matches_the_harness_estimator():
    cells = ssr.cells_from_frame(synthetic_cells())
    got = ssr.clustered_stats(cells)
    from src.eval.tuning import paired_abs_error_diff

    # Rebuild the same numbers through the harness's own function by handing it
    # a predicted/realized pair whose absolute-error difference is `d`.
    raw = synthetic_cells()
    a = pd.DataFrame({"cell": range(len(raw)), "predicted": raw["d"].abs(),
                      "realized_rate": 0.0, "trials": raw["trials"],
                      "player": raw["player"]})
    b = pd.DataFrame({"cell": range(len(raw)),
                      "predicted": np.where(raw["d"] >= 0, 0.0, -raw["d"]),
                      "realized_rate": 0.0, "trials": raw["trials"]})
    a["predicted"] = np.where(raw["d"] >= 0, raw["d"], 0.0)
    ref = paired_abs_error_diff(a, b, id_col="cell", cluster_col="player")
    assert got["diff"] == pytest.approx(ref["diff"], rel=1e-9)
    assert got["se"] == pytest.approx(ref["se"], rel=1e-9)
    assert got["n_clusters"] == ref["n_clusters"] == 200


def test_clustering_widens_the_se_relative_to_ignoring_it():
    cells = ssr.cells_from_frame(synthetic_cells(cluster_sd=2.0))
    clustered = ssr.clustered_stats(cells)
    naive = ssr.clustered_stats(cells.assign(cluster=range(len(cells))))
    assert clustered["se"] > naive["se"]


def test_bootstrap_is_seeded_and_reproducible():
    # A knife-edge effect, where the seed can actually move the answer.
    cells = ssr.cells_from_frame(synthetic_cells(effect=-0.24, seed=9))
    assert 2.0 < abs(ssr.clustered_stats(cells)["t"]) < 3.0
    a = ssr.cluster_bootstrap_p(cells, n_boot=500, seed=11)
    b = ssr.cluster_bootstrap_p(cells, n_boot=500, seed=11)
    c = ssr.cluster_bootstrap_p(cells, n_boot=500, seed=12)
    assert a["p_gt_bar"] == b["p_gt_bar"]
    assert a["p_gt_bar"] != c["p_gt_bar"]  # a seed is a real moving part


def test_bootstrap_separates_a_strong_effect_from_no_effect():
    strong = ssr.cells_from_frame(synthetic_cells(effect=-1.0, seed=3))
    none = ssr.cells_from_frame(synthetic_cells(effect=0.0, seed=3))
    assert ssr.cluster_bootstrap_p(strong, n_boot=500)["p_gt_bar"] > 0.9
    assert ssr.cluster_bootstrap_p(none, n_boot=500)["p_gt_bar"] < 0.2


def test_reconstruction_reproduces_the_committed_summary_exactly():
    cells = ssr.reconstruct_cells(diff=-0.00015, se=6e-05, n=4954,
                                  n_clusters=955, seed=5)
    got = ssr.clustered_stats(cells)
    assert got["diff"] == pytest.approx(-0.00015, rel=1e-9)
    assert got["se"] == pytest.approx(6e-05, rel=1e-9)
    assert got["n"] == 4954 and got["n_clusters"] == 955
    # And the bootstrap on it lands on the closed form it is bound to.
    p = ssr.cluster_bootstrap_p(cells, n_boot=2000, seed=5)["p_gt_bar"]
    assert p == pytest.approx(ssr.p_normal(got["t"]), abs=0.05)


def test_p_normal_at_the_bar_is_a_half():
    assert ssr.p_normal(2.5) == pytest.approx(0.5, abs=1e-6)


def _share(t, pct, diff=-1e-4, n=4954, k=955):
    return {"diff": diff, "se": abs(diff / t), "t": t, "n": n,
            "n_clusters": k, "pct": pct}


def test_the_rules_on_a_synthetic_knife_edge():
    # BB/BF's shape: straddles the bar across two baselines, effect above the
    # floor. Two-baseline holds the status quo, the effect floor clears it.
    share = _share(-2.41, -0.85)
    cal = _share(-2.58, -0.93)
    assert ssr.verdict_current(share) == ssr.WITHHOLD
    assert ssr.verdict_two_baseline(share, cal, ssr.WITHHOLD) == ssr.WITHHOLD
    assert ssr.verdict_effect_floor(share) == ssr.CLEAR
    # Under on both baselines: two-baseline withholds outright.
    assert ssr.verdict_two_baseline(_share(-1.86, -0.64), _share(-1.54, -0.46),
                                    ssr.CLEAR) == ssr.WITHHOLD
    # Over on both: it clears outright.
    assert ssr.verdict_two_baseline(_share(-3.7, -3.1), _share(-3.7, -3.1),
                                    ssr.WITHHOLD) == ssr.CLEAR
    # No calibrated baseline exists -> the status quo, whatever it was.
    assert ssr.verdict_two_baseline(share, None, ssr.CLEAR) == ssr.CLEAR


def test_effect_floor_rejects_a_significant_but_tiny_share():
    # pitcher BB% under contact quality: t is small AND the effect is nil.
    assert ssr.verdict_effect_floor(_share(-0.53, -0.068)) == ssr.WITHHOLD
    # A large-n, highly significant share worth 0.1% of MAE is still refused.
    assert ssr.verdict_effect_floor(_share(-6.0, -0.10)) == ssr.WITHHOLD
    # And a large share that is not significant is refused too.
    assert ssr.verdict_effect_floor(_share(-1.5, -4.0)) == ssr.WITHHOLD


def test_bootstrap_verdict_band_falls_back_to_the_status_quo():
    assert ssr.verdict_bootstrap(0.85, ssr.WITHHOLD) == ssr.CLEAR
    assert ssr.verdict_bootstrap(0.10, ssr.CLEAR) == ssr.WITHHOLD
    assert ssr.verdict_bootstrap(0.50, ssr.CLEAR) == ssr.CLEAR
    assert ssr.verdict_bootstrap(0.50, ssr.WITHHOLD) == ssr.WITHHOLD


def test_score_decision_on_a_synthetic_per_cell_table():
    decision = {
        "decision": "synthetic/hitter/x", "ticket": "TEST", "side": "hitter",
        "component": "x", "served_today": False, "baseline": "base",
        "control": "base_recal", "total_pct": -2.0, "total_t": -3.0,
        "cells": synthetic_cells(effect=-1.0, seed=4).rename(
            columns={"player": "player", "trials": "trials"}),
        "covariate_share": _share(-4.0, -2.0),
        "covariate_share_calibrated": None,
    }
    row = ssr.score_decision(decision, seed=1, n_boot=400)
    assert row["bootstrap_source"] == "per_cell"
    assert row["verdict_current"] == ssr.CLEAR
    assert row["bootstrap_p_gt_2_5"] > 0.9
    assert row["changes"] == []


def test_load_decisions_covers_every_committed_serving_decision():
    decisions = ssr.load_decisions()
    names = {d["decision"] for d in decisions}
    assert {f"contact/hitter/{c}" for c in
            ("k_rate", "bb_rate", "hr_rate", "babip", "iso")} <= names
    assert {f"stuff/pitcher/{c}" for c in
            ("p_k_rate", "p_bb_rate", "p_bbhbp_rate", "p_hr_rate")} <= names
    stuff = {d["decision"]: d for d in decisions}
    # BAS-80 gave the stuff decisions a second baseline; contact quality has none.
    assert stuff["stuff/pitcher/p_bb_rate"]["covariate_share_calibrated"] is not None
    assert stuff["contact/hitter/iso"]["covariate_share_calibrated"] is None
    # The knife edge the ticket is about, read straight out of the evidence.
    assert stuff["stuff/pitcher/p_bb_rate"]["covariate_share"]["t"] == \
        pytest.approx(-2.4134, abs=1e-3)
    assert stuff["stuff/pitcher/p_bb_rate"]["covariate_share_calibrated"]["t"] == \
        pytest.approx(-2.5820, abs=1e-3)


def test_committed_matrix_matches_a_rerun():
    import json
    payload = json.loads(
        (ROOT / "data" / "eval" / "serving_rules.json").read_text())
    got = {d["decision"]: d for d in payload["decisions"]}
    for d in ssr.load_decisions():
        row = got[d["decision"]]
        assert row["covariate_t"] == pytest.approx(
            d["covariate_share"]["t"], rel=1e-9)
        assert row["verdict_current"] == ssr.verdict_current(d["covariate_share"])
