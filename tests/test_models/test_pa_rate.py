"""The component-parameterised PA rate model (BAS-73).

`src/models/pa_rate.py` is `pa_k_rate.py` with the numerator, the league
prior and the age curve's direction lifted into a `RateComponent`. Two
claims have to hold for that to be a refactor rather than a rewrite, and
this file checks both:

1. **K% is unchanged, bit-for-bit.** `test_k_rate_is_bit_for_bit_unchanged`
   fits `tests/test_models/reference_k_rate.py`'s fixture through
   `src.models.pa_k_rate` in all four `ModelOptions` combinations and
   compares every posterior mean and every projected rate against
   `k_rate_reference.json`, which was generated from the pre-split
   `pa_k_rate.py`. Exact equality, no tolerance: the fixture is deliberately
   sampled with the pure-Python `pymc` NUTS backend at a fixed seed so a
   fresh process reproduces the same draws.

2. **BB% and HR/PA are the same model with their own direction.** The age
   curve is a valley for K% (a bigger rate is worse) and a hill for the other
   two, and `TestAgeDirectionPerComponent` pins that per component at the
   graph level rather than trusting the sign in a comment.

Everything here needs pymc/arviz, which `requirements-ci.txt` leaves out on
purpose (MCMC runs on Modal, not in CI) — same guard as
`tests/test_models/test_pa_k_rate_options.py`, and the whole file is skipped
where they are not installed. The registry and the age-direction pin at the bottom
need neither: they read `src/models/pa_components.py`, which is why that
module exists.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.test_models.reference_k_rate import (  # noqa: E402
    FAST_SAMPLER, FIXTURE_SEASONS, fixture_pa_rows, summarise_k_rate,
)

HAS_PYMC = all(importlib.util.find_spec(m) is not None
               for m in ("pymc", "arviz"))
needs_pymc = pytest.mark.skipif(
    not HAS_PYMC, reason="MCMC deps (pymc/arviz) are not installed in CI")

REFERENCE_JSON = Path(__file__).parent / "k_rate_reference.json"


# ─── 1. the equivalence pin ─────────────────────────────────────────────────

@needs_pymc
def test_k_rate_is_bit_for_bit_unchanged():
    """Every K% posterior mean and every projected rate, exactly as the
    pre-BAS-73 module produced them.

    Exact (`==`, not `approx`) on purpose. "Close" is what a refactor that
    quietly moved a prior would also produce, and the whole point of the
    fixture — one chain, 25 draws, the seeded pure-Python sampler — is that
    the draws are reproducible to the last bit, so anything short of equality
    is a real change and should fail here.
    """
    from src.models import pa_k_rate

    reference = json.loads(REFERENCE_JSON.read_text())
    got = summarise_k_rate(pa_k_rate)

    assert set(got) == set(reference), "option combos have drifted"
    for combo, expected in reference.items():
        actual = got[combo]
        assert set(actual["posterior_means"]) == set(expected["posterior_means"]), (
            f"{combo}: the trace's variable set changed")
        for name, value in expected["posterior_means"].items():
            assert actual["posterior_means"][name] == value, (
                f"{combo}: posterior mean of {name!r} moved")
        assert actual["batters"] == expected["batters"]
        assert actual["projected_k_rate"] == expected["projected_k_rate"], (
            f"{combo}: projected K% moved")


@needs_pymc
def test_the_wrapper_and_the_general_module_build_the_same_k_rate_graph():
    """`pa_k_rate` binds `component="k_rate"` and nothing else.

    The equivalence test above proves the numbers; this one proves the
    mechanism, cheaply and without sampling, so a failure says *where*.
    """
    from src.models import pa_k_rate, pa_rate

    pa = fixture_pa_rows()
    via_wrapper = pa_k_rate.prepare_model_data(pa, None, min_pa=1)
    via_general = pa_rate.prepare_model_data(pa, None, min_pa=1,
                                             component="k_rate")
    assert via_wrapper["component"] == "k_rate"
    assert via_wrapper["league_init_mu"] == via_general["league_init_mu"] == -1.27
    np.testing.assert_array_equal(via_wrapper["k"], via_general["k"])
    np.testing.assert_array_equal(via_wrapper["n_trials"], via_general["n_trials"])

    names = {v.name for v in pa_k_rate.build_model(via_wrapper).free_RVs}
    assert names == {v.name for v in pa_rate.build_model(via_general).free_RVs}


# ─── 2. the component registry ──────────────────────────────────────────────

class TestComponentRegistry:
    def test_the_three_pa_binomials_are_registered(self):
        from src.models.pa_components import RATE_COMPONENTS

        assert set(RATE_COMPONENTS) == {"k_rate", "bb_rate", "hr_rate"}

    def test_bb_rate_counts_walks_only_not_walks_plus_hbp(self):
        """`src.eval.backtest.COMPONENTS["bb_rate"]` scores `bb / pa`, and
        `src.eval.intraseason.aggregate_pa` builds `bb` from `is_bb` with
        `hbp` in a column of its own. Fitting `is_bb + is_hbp` would score the
        model against a numerator it was never fit on, and HBP is ~1% of PA
        against BB's ~8.5% — big enough to matter, small enough to miss."""
        from src.eval.backtest import COMPONENTS
        from src.models.pa_components import RATE_COMPONENTS

        assert RATE_COMPONENTS["bb_rate"].numerator == "is_bb"
        assert COMPONENTS["bb_rate"].successes == "bb"
        assert COMPONENTS["bb_rate"].trials == "pa"

    def test_hr_rate_counts_home_runs_per_pa(self):
        from src.eval.backtest import COMPONENTS
        from src.models.pa_components import RATE_COMPONENTS

        assert RATE_COMPONENTS["hr_rate"].numerator == "is_hr"
        assert COMPONENTS["hr_rate"].trials == "pa"

    def test_babip_and_iso_are_refused_with_a_reason(self):
        """Not per-PA binomials — different denominators, out of scope until
        they get their own cells."""
        from src.models.pa_components import get_component

        for name in ("babip", "iso", "nonsense"):
            with pytest.raises(ValueError, match="unknown component"):
                get_component(name)

    def test_none_is_k_rate_so_old_call_sites_keep_their_meaning(self):
        from src.models.pa_components import get_component

        assert get_component(None).name == "k_rate"


@needs_pymc
class TestLeagueInitPrior:
    """K% pins the prior mean; BB% and HR/PA read it off the earliest
    training season, which is the season the league random walk actually
    starts at."""

    def _data(self, component):
        from src.models.pa_rate import prepare_model_data

        return prepare_model_data(fixture_pa_rows(), None, min_pa=1,
                                  component=component)

    def test_k_rate_stays_pinned(self):
        assert self._data("k_rate")["league_init_mu"] == -1.27

    @pytest.mark.parametrize("component", ["bb_rate", "hr_rate"])
    def test_derived_from_the_earliest_season(self, component):
        from src.models.pa_components import RATE_COMPONENTS

        pa = fixture_pa_rows()
        data = self._data(component)
        numerator = RATE_COMPONENTS[component].numerator
        first = pa[pa["game_year"] == pa["game_year"].min()]
        p = (first[numerator].sum() + 0.5) / (len(first) + 1.0)
        assert data["league_init_mu"] == pytest.approx(
            float(np.log(p / (1 - p))), abs=1e-9)
        # And it is the *first* season, not the pooled window: the fixture's
        # rates differ enough across its three seasons for that to bite.
        pooled = pa[numerator].mean()
        assert data["league_init_mu"] != pytest.approx(
            float(np.log(pooled / (1 - pooled))), abs=1e-12)

    def test_the_prior_reaches_the_graph(self):
        """A derived mean that never gets wired into `pm.Normal` would look
        right in the data dict and do nothing at all."""
        from src.models.pa_rate import build_model

        data = self._data("hr_rate")
        model = build_model(data)
        mu = model["league_init"].owner.inputs[2].eval()
        assert float(mu) == pytest.approx(data["league_init_mu"], abs=1e-12)
        assert float(mu) < -3.0  # HR/PA is a ~3% event: logit ≈ -3.4


# ─── 3. the age curve's direction, per component ────────────────────────────

@needs_pymc
class TestAgeDirectionPerComponent:
    """`src.eval.tuning.AGE_DIRECTION` says K% is the one component where a
    bigger number is a worse hitter. On the constrained curve that is the
    difference between a valley and a hill, and it is a sign — the cheapest
    thing in the model to get backwards and the hardest to notice in a
    backtest, since a wrong-signed age term still fits *something*.

    Both branches are exercised at fixed parameters: `peak_age` pinned to 27,
    both slopes to 0.01, so the only thing under test is the sign.
    """

    def _terms(self, component):
        from src.models.pa_rate import ModelOptions, build_model, prepare_model_data

        data = prepare_model_data(fixture_pa_rows(), None, min_pa=1,
                                  component=component)
        model = build_model(data, ModelOptions(constrained_age=True))

        point = model.initial_point()
        lo, hi = 25.0, 31.0
        frac = (27.0 - lo) / (hi - lo)
        point["peak_frac_logodds__"] = np.array(np.log(frac / (1 - frac)))
        point["slope_young_log__"] = np.array(np.log(0.01))
        point["slope_old_log__"] = np.array(np.log(0.01))

        outs = model.replace_rvs_by_values(
            [model["peak_age"], model["slope_young"], model["slope_old"]])
        f = model.compile_fn(outs, inputs=model.value_vars, point_fn=True,
                             on_unused_input="ignore")
        peak_age, slope_young, slope_old = f(point)
        assert peak_age == pytest.approx(27.0, abs=1e-6)

        from src.models.pa_components import RATE_COMPONENTS
        sign = -RATE_COMPONENTS[component].age_direction

        def term(age):
            d = age - peak_age
            return float(sign * (slope_old * d if d > 0 else -slope_young * d))

        return term(22.0), term(27.0), term(34.0)

    def test_k_rate_is_a_valley(self):
        """K% is lowest at the peak: a young hitter's strikeout rate falls as
        he approaches it and rises again after — the shape
        docs/bayes-variants.md's fitted `slope_old = 0.0080` describes."""
        young, peak, old = self._terms("k_rate")
        assert peak == pytest.approx(0.0, abs=1e-12)
        assert young > peak and old > peak

    @pytest.mark.parametrize("component", ["bb_rate", "hr_rate"])
    def test_bb_and_hr_are_hills(self, component):
        """A skill you want more of peaks and declines: both a 22- and a
        34-year-old sit *below* the 27-year-old."""
        young, peak, old = self._terms(component)
        assert peak == pytest.approx(0.0, abs=1e-12)
        assert young < peak and old < peak

    @pytest.mark.parametrize("component,expected_sign",
                             [("k_rate", 1.0), ("bb_rate", -1.0),
                              ("hr_rate", -1.0)])
    def test_the_quadratic_prior_mean_follows_the_same_direction(
            self, component, expected_sign):
        """The free quadratic asserts no sign, but its `beta_age2` prior mean
        should still lean the right way: a U for K%, an inverted U for the
        others."""
        from src.models.pa_rate import build_model, prepare_model_data

        data = prepare_model_data(fixture_pa_rows(), None, min_pa=1,
                                  component=component)
        model = build_model(data)
        mu = float(model["beta_age2"].owner.inputs[2].eval())
        assert mu == pytest.approx(expected_sign * 0.005, abs=1e-12)

    def test_the_projection_side_reapplies_the_same_sign(self):
        """`_age_term_function` rebuilds the curve from the trace, where the
        slopes are stored as the positive HalfNormals they are. If it did not
        reapply `curve_sign`, a BB% projection would age like a K% one — and
        nothing else in the pipeline would notice."""
        from src.models.pa_rate import _age_term_function

        class _Var:
            def __init__(self, v):
                self.values = np.array([[v]])

        post = {"peak_age": _Var(27.0), "slope_young": _Var(0.01),
                "slope_old": _Var(0.01)}
        k = _age_term_function(post, 1, age_direction=-1.0)
        bb = _age_term_function(post, 1, age_direction=1.0)
        k34, bb34 = float(k(34.0).item()), float(bb(34.0).item())
        assert k34 > 0 and bb34 < 0
        assert k34 == pytest.approx(-bb34, abs=1e-12)


# ─── 4. projection column names per component ───────────────────────────────

@needs_pymc
class TestGenerateProjectionsColumnsPerComponent:
    """One tiny real fit per component. The column names are what the arm
    and the harness join on, so a component whose frame still says
    `projected_k_rate` would be scored as K% under a BB% label."""

    def _fit(self, component):
        from src.models.pa_rate import (
            build_model, prepare_model_data, sample_model,
        )

        data = prepare_model_data(fixture_pa_rows(), None, min_pa=1,
                                  component=component)
        trace = sample_model(build_model(data), **FAST_SAMPLER)
        return data, trace

    @pytest.mark.parametrize("component", ["k_rate", "bb_rate", "hr_rate"])
    def test_columns_are_named_for_the_component(self, component):
        from src.models.pa_rate import generate_projections

        data, trace = self._fit(component)
        proj = generate_projections(trace, data, projection_year=2026)
        assert set(proj.columns) == {
            "batter", "stand", "age", f"projected_{component}",
            f"{component}_std", f"{component}_lower", f"{component}_upper",
            "posterior_mean_ability", "total_pa", f"career_{component}",
            "last_season", "unseen",
        }
        assert proj[f"projected_{component}"].between(0, 1).all()

    @pytest.mark.parametrize("component", ["bb_rate", "hr_rate"])
    def test_unseen_batters_use_the_same_column_names(self, component):
        from src.models.pa_rate import generate_projections

        data, trace = self._fit(component)
        unseen = pd.DataFrame({"batter": [9998, 9999], "age": [np.nan, 24.0]})
        proj = generate_projections(trace, data, projection_year=2026,
                                    unseen=unseen)
        rows = proj[proj["unseen"]]
        assert set(rows["batter"]) == {9998, 9999}
        assert rows[f"projected_{component}"].between(0, 1).all()

    def test_hr_rate_projects_a_home_run_rate_not_a_strikeout_rate(self):
        """The obvious end-to-end sanity check: HR/PA is a ~3% event and K%
        a ~22% one, so a component that silently fitted the wrong numerator
        would land an order of magnitude off."""
        from src.models.pa_rate import generate_projections

        data, trace = self._fit("hr_rate")
        proj = generate_projections(trace, data, projection_year=2026)
        assert proj["projected_hr_rate"].median() < 0.10


# ─── 5. constants that must not drift from src.eval ────────────────────────

def test_age_direction_matches_the_tuning_module():
    """`pa_rate` copies `AGE_DIRECTION` rather than importing it, to keep the
    model package's zero dependency on `src.eval` — the same trade
    `AGE_PEAK_WINDOW` already makes. That is only defensible while something
    notices when one copy moves, since a flipped sign here would age BB% like
    K% and still fit. Needs no pymc: both are plain dicts."""
    from src.eval import tuning
    from src.models import pa_components

    for name, direction in pa_components.AGE_DIRECTION.items():
        assert direction == tuning.AGE_DIRECTION[name], name


@needs_pymc
def test_the_wrapper_still_exports_the_age_peak_window():
    """`tests/test_models/test_pa_k_rate_options.py` pins
    `pa_k_rate.AGE_PEAK_WINDOW` against `src.eval.tuning`; the wrapper has to
    keep re-exporting it for that check to keep meaning anything."""
    from src.eval import tuning
    from src.models import pa_k_rate

    assert pa_k_rate.AGE_PEAK_WINDOW == tuning.AGE_PEAK_WINDOW


# ─── 6. park factors, per component (BAS-86, docs/park-factors.md) ──────────

@needs_pymc
class TestParkFactorLoading:
    """`load_park_factors` picks the file up; `prepare_model_data` picks the
    *component's own column* out of it.

    The K% path when no file exists is the one that has to stay bit-for-bit
    (test 1 above fits under exactly that condition, passing `None`), so the
    first test here states that condition directly rather than leaving it
    implied by the reference fit.
    """

    # `fixture_pa_rows` is HOU hosting SEA in the top of the inning, so the
    # *batting* club — the one `prepare_model_data` keys the offset on, since
    # a hitter's park is his own club's — is SEA in every row. Both clubs are
    # in the table, in every fixture season, so a test that asserts SEA's
    # numbers is asserting the lookup and not the absence of HOU's.
    FACTORS = {"SEA": {"k": 0.90, "bb": 1.25, "hr": 0.70},
               "HOU": {"k": 1.10, "bb": 0.80, "hr": 1.40}}

    def _pf(self):
        rows = [{"team": team, "game_year": year,
                 "k_park_factor": f["k"], "bb_park_factor": f["bb"],
                 "hr_park_factor": f["hr"]}
                for team, f in self.FACTORS.items()
                for year in FIXTURE_SEASONS]
        return pd.DataFrame(rows)

    def test_no_file_anywhere_is_none_and_a_zero_offset(self, tmp_path):
        from src.models.pa_rate import load_park_factors, prepare_model_data

        assert load_park_factors(tmp_path / "nope.parquet") is None
        data = prepare_model_data(fixture_pa_rows(), None, min_pa=1)
        assert (data["log_pf_k"] == 0.0).all()

    def test_the_features_artifact_wins_over_the_legacy_parquet(self, tmp_path,
                                                                monkeypatch):
        from src.models import pa_rate

        new = tmp_path / "features.parquet"
        old = tmp_path / "legacy.parquet"
        self._pf().to_parquet(new, index=False)
        self._pf().assign(k_park_factor=2.0).to_parquet(old, index=False)
        monkeypatch.setattr(pa_rate, "PARK_FACTOR_PATHS", (new, old))
        assert (pa_rate.load_park_factors()["k_park_factor"] != 2.0).all()

        monkeypatch.setattr(pa_rate, "PARK_FACTOR_PATHS", (tmp_path / "x", old))
        assert (pa_rate.load_park_factors()["k_park_factor"] == 2.0).all()

    @pytest.mark.parametrize("component,factor", [
        ("k_rate", 0.90), ("bb_rate", 1.25), ("hr_rate", 0.70)])
    def test_each_component_reads_its_own_column(self, component, factor):
        """A shared table with three columns in it: BB% must not be handed
        K%'s factor."""
        from src.models.pa_rate import prepare_model_data

        data = prepare_model_data(fixture_pa_rows(), self._pf(), min_pa=1,
                                  component=component)
        assert np.allclose(data["log_pf_k"], np.log(factor))

    def test_a_missing_component_column_is_neutral_not_borrowed(self):
        from src.models.pa_rate import prepare_model_data

        pf = self._pf().drop(columns=["hr_park_factor"])
        data = prepare_model_data(fixture_pa_rows(), pf, min_pa=1,
                                  component="hr_rate")
        assert (data["log_pf_k"] == 0.0).all()

    def test_the_joint_model_stacks_one_column_per_component(self):
        from src.models.pa_joint import prepare_joint_data

        joint = prepare_joint_data(fixture_pa_rows(), self._pf(), min_pa=1,
                                   components=("k_rate", "bb_rate", "hr_rate"))
        expected = np.log([0.90, 1.25, 0.70])
        assert np.allclose(joint.log_pf, expected[None, :])

    def test_the_committed_artifact_covers_every_component(self):
        from src.data.park_components import DEFAULT_PATH
        from src.models.pa_components import RATE_COMPONENTS
        from src.models.pa_rate import PROJECT_ROOT, load_park_factors

        path = PROJECT_ROOT / DEFAULT_PATH
        if not path.exists():
            pytest.skip(f"{path} not built")
        pf = load_park_factors(path)
        for comp in RATE_COMPONENTS.values():
            assert comp.park_factor_col in pf.columns
