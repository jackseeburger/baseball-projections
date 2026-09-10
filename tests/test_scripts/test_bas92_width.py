"""BAS-92 wiring: the width flag on the exam, and the analysis's own helpers.

Nothing here fits a model. `scripts/run_bayes_width.py` imports pymc only
inside the function that samples (via `src.eval.bayes_arm`), so it is
importable in CI, which installs `requirements-ci.txt` and has neither pymc
nor arviz; that importability is itself one of the assertions below.
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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


props_exam = _load("props_exam")
analyze = _load("analyze_bas92")
from src.market import pnl, props  # noqa: E402


def test_run_bayes_width_imports_without_a_sampler():
    """The fit script has to be importable where pymc is not installed —
    same rule `src.eval.bayes_arm` states for itself."""
    module = _load("run_bayes_width")
    assert module.CUTOFFS == ("2026-07-15", "2026-08-01", "2026-08-15",
                              "2026-09-01")
    assert set(module.COMPONENTS) == set(props.BAYES_WIDTH_COMPONENTS)
    assert module.VARIANT == "ability_walk"


def test_price_with_forwards_the_width_arm_and_defaults_to_off(monkeypatch):
    seen = {}

    def fake_price(closes, batter_ctx, pitcher_ctx, slots, stats=None,
                   pitcher_bf="fixed", matchup_ctx=None, bayes_width=None):
        seen["bayes_width"] = bayes_width
        return pd.DataFrame()

    monkeypatch.setattr(props_exam.props, "price", fake_price)
    ctx = {"batter_ctx": {}, "pitcher_ctx": {}, "slots": {}, "matchup_ctx": None}
    props_exam.price_with(pd.DataFrame(), ctx, ("hr",), "fixed")
    assert seen["bayes_width"] is None
    sentinel = object()
    props_exam.price_with(pd.DataFrame(), ctx, ("hr",), "fixed",
                          bayes_width=sentinel)
    assert seen["bayes_width"] is sentinel


def _quoted_frame(n=200, seed=3):
    rng = np.random.default_rng(seed)
    mid = rng.uniform(0.2, 0.8, n)
    return pd.DataFrame({
        "date": ["2026-08-20"] * n,
        "game_pk": rng.integers(1, 40, n),
        "prop_stat": "hr",
        "p_home_close": mid,
        "bid": mid - 0.02, "ask": mid + 0.02,
        "matchup": np.clip(mid + rng.normal(0, 0.06, n), 0.01, 0.99),
        "p_over_sd": rng.uniform(0.002, 0.06, n),
        "home_win": rng.random(n) < mid,
    })


def test_p_edge_positive_is_the_decision_boundary_decide_posterior_uses():
    """`P(edge > 0)` read out as a number has to agree, row for row, with the
    rule that thresholds it — otherwise the tercile table in prediction 3 is
    ranking on something other than what prediction 2 selects on."""
    frame = _quoted_frame()
    venue = props_exam.venue_for(pnl.KALSHI_TAKER_RATE, False)
    p_edge = analyze.p_edge_positive(frame, "matchup", "p_over_sd", venue)
    assert ((p_edge >= 0.0) & (p_edge <= 1.0)).all()
    bid, ask = pnl.quotes(frame, venue)
    for tau in (0.55, 0.7, 0.9):
        d = pnl.decide_posterior(frame["matchup"].to_numpy(dtype=float),
                                 frame["p_over_sd"].to_numpy(dtype=float),
                                 bid, ask, tau)
        assert np.array_equal((d["side"] != pnl.NO_BET).to_numpy(), p_edge > tau)


def test_prediction_1_reads_vacuity_off_the_primary_stat():
    """A width that is a fixed multiple of the Beta's is perfectly rank
    correlated with it — different numbers, identical *ordering* — which is
    exactly the case the pre-registration calls vacuous, and the Spearman
    half of the check is what catches it."""
    n = 400
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"prop_stat": "hr",
                         "p_over_sd": rng.uniform(0.005, 0.03, n)})
    scaled = base.copy()
    scaled["p_over_sd"] = base["p_over_sd"] * 2.0        # every sd changes by 100%
    res = analyze.prediction_1(base, scaled)
    assert res["by_stat"]["hr"]["changed_share"] == pytest.approx(1.0)
    assert res["by_stat"]["hr"]["spearman"] == pytest.approx(1.0)
    assert res["passes"] is False

    shuffled = base.copy()
    shuffled["p_over_sd"] = rng.permutation(base["p_over_sd"].to_numpy())
    res2 = analyze.prediction_1(base, shuffled)
    assert res2["by_stat"]["hr"]["spearman"] < 0.9
    assert res2["passes"] is True


def test_tercile_table_holds_the_bets_fixed_and_only_reorders_them():
    frame = _quoted_frame(n=600, seed=7)
    fee_waived = props_exam.venue_for(0.0, False)
    as_quoted = props_exam.venue_for(pnl.KALSHI_TAKER_RATE, False)
    table = analyze.tercile_table(frame, fee_waived, as_quoted, draws=50, seed=0)
    bets = pnl.bet_frame(frame, "matchup", as_quoted,
                         threshold=analyze.HEADLINE_THRESHOLD)
    assert table["n"] == len(bets)
    assert sum(r["n_bets"] for r in table["terciles"]) == len(bets)
    assert [r["rule"] for r in table["terciles"]] == ["bottom", "mid", "top"]
    lo = [r["p_edge_lo"] for r in table["terciles"]]
    assert lo == sorted(lo)
