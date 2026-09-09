"""The multi-variant bayes sweep's plumbing (BAS-69): cache isolation across
variants, checkpoint round-tripping (including refusing a pre-variant one),
and the clustered-vs-unclustered comparison table the analyze stage renders.

Nothing here samples — every test that would otherwise need a real MCMC fit
monkeypatches `src.eval.bayes_arm.fit_bayes_k_rate`, the same convention
`tests/test_eval/test_bayes_arm.py` uses, so this file collects and passes
without pymc/arviz installed (CI's requirements-ci.txt has neither).

The one test that matters most is `test_two_variants_never_share_a_fit`: it
is what would have caught the exact bug this whole sweep design exists to
avoid — a fit from one variant silently scored as another's.
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
        "run_intraseason_backtest_dense",
        ROOT / "scripts/run_intraseason_backtest_dense.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dense = _load()

from src.eval.backtest import COMPONENTS  # noqa: E402
from src.eval.bayes_arm import BayesArmConfig, BayesFit  # noqa: E402

CONTACT_ARM_NAME = dense.CONTACT_ARM


# --- variant <-> config round trip -------------------------------------------

class TestVariantConfig:
    def test_every_known_variant_round_trips(self):
        for variant in dense.VARIANT_ARM_NAMES:
            config = dense._variant_config(variant)
            assert config.variant() == variant

    def test_arm_names_and_variant_names_are_inverse_maps(self):
        assert dense.ARM_NAME_VARIANT == {
            arm: variant for variant, arm in dense.VARIANT_ARM_NAMES.items()
        }
        for variant, arm in dense.VARIANT_ARM_NAMES.items():
            assert dense.ARM_NAME_VARIANT[arm] == variant


# --- the cache isolation guard -----------------------------------------------

def _fake_projections(batters):
    return pd.DataFrame({
        "batter": list(batters),
        "projected_k_rate": [0.22] * len(batters),
    })


class TestCacheIsolation:
    """`bayes_k_rate_provider`'s memoization keys on (cutoff_date,
    predict_year) only — no variant. `make_variant_providers` is the code
    that has to make that safe by never handing two variant names the same
    provider object (except the deliberate "bayes" alias for "flat"). These
    tests drive that function directly, standing in for the real MCMC fit
    with a monkeypatched `fit_bayes_k_rate` so no pymc is needed.
    """

    def test_two_variants_never_share_a_fit(self, monkeypatch):
        calls = []

        def fake_fit(cutoff_date, predict_year, config, unseen=None):
            calls.append(config.variant())
            return BayesFit(
                cutoff_date=str(cutoff_date), predict_year=int(predict_year),
                projections=_fake_projections([1, 2]),
                diagnostics={"max_rhat": 1.0, "min_ess_bulk": 500,
                            "divergences": 0, "bfmi": [0.9], "bfmi_ok": True,
                            "healthy": True},
                config=config, trace=None, model_data=None, data_summary={},
            )

        monkeypatch.setattr("src.eval.bayes_arm.fit_bayes_k_rate", fake_fit)

        train = pd.DataFrame({"batter": [1, 2], "season": [2026, 2026]})
        spec = COMPONENTS["k_rate"]
        fits: list[dict] = []
        providers = dense.make_variant_providers(
            "2026-07-01", 2026, ["flat", "ability_walk"], {}, fits)

        assert set(providers) == {"bayes_flat", "bayes_walk", "bayes"}
        # "bayes" is the flat variant's alias: same object, same cache.
        assert providers["bayes"] is providers["bayes_flat"]
        assert providers["bayes_walk"] is not providers["bayes_flat"]

        providers["bayes_flat"](train, spec, 2026)
        providers["bayes_walk"](train, spec, 2026)
        assert calls == ["flat", "ability_walk"], (
            "each variant must fit exactly once, and never the other's config")

        # Calling flat again (directly, and via the "bayes" alias) must hit
        # the cache, not fit again — this is the whole point of the alias
        # costing zero extra MCMC.
        providers["bayes_flat"](train, spec, 2026)
        providers["bayes"](train, spec, 2026)
        assert calls == ["flat", "ability_walk"], "cache hits must not refit"

        assert {f["variant"] for f in fits} == {"flat", "ability_walk"}
        assert all("elapsed_s" in f for f in fits), (
            "wall time must be attached once the real fit (not a cache hit) "
            "returns"
        )

    def test_a_mislabeled_fit_raises_instead_of_silently_scoring(self, monkeypatch):
        """The defense-in-depth check: if a fit ever comes back reporting a
        different variant than the one its provider was built for — which
        should be structurally impossible given one `bayes_k_rate_provider`
        call per variant, but is exactly the failure mode a future refactor
        could reintroduce — `make_variant_providers` must raise, not score
        the wrong model under the right name.
        """
        def fake_fit_always_flat(cutoff_date, predict_year, config, unseen=None):
            # Always reports "flat", regardless of what was actually asked
            # for — simulates the cache (or provider) getting cross-wired.
            wrong_config = BayesArmConfig()
            return BayesFit(
                cutoff_date=str(cutoff_date), predict_year=int(predict_year),
                projections=_fake_projections([1, 2]),
                diagnostics={"max_rhat": 1.0, "min_ess_bulk": 500, "divergences": 0},
                config=wrong_config, trace=None, model_data=None, data_summary={},
            )

        monkeypatch.setattr("src.eval.bayes_arm.fit_bayes_k_rate", fake_fit_always_flat)

        train = pd.DataFrame({"batter": [1, 2], "season": [2026, 2026]})
        spec = COMPONENTS["k_rate"]
        providers = dense.make_variant_providers(
            "2026-07-01", 2026, ["ability_walk"], {}, [])

        with pytest.raises(RuntimeError, match="cache isolation broken"):
            providers["bayes_walk"](train, spec, 2026)

    def test_only_flat_gets_the_bayes_alias(self):
        providers = dense.make_variant_providers(
            "2026-07-01", 2026, ["constrained_age"], {}, [])
        assert set(providers) == {"bayes_age"}
        assert "bayes" not in providers


# --- variant_param_summary (the vacuity check) -------------------------------

class _FakeVar:
    def __init__(self, values):
        self.values = np.asarray(values)


class TestVariantParamSummary:
    def test_pulls_only_the_variant_own_params(self):
        posterior = {
            "sigma_step": _FakeVar([[0.01, 0.02], [0.015, 0.025]]),  # 2x2 chains x draws
            "beta_hand": _FakeVar([[0.1, 0.1], [0.1, 0.1]]),  # not this variant's own
        }
        trace = type("FakeTrace", (), {"posterior": posterior})()
        config = dense._variant_config("ability_walk")
        out = dense.variant_param_summary(trace, config)
        assert set(out) == {"sigma_step"}
        assert out["sigma_step"]["mean"] == pytest.approx(0.0175)

    def test_missing_param_name_is_skipped_not_raised(self):
        """The model side may land under a different name than the
        pre-registration's math used — this should say what it found, not
        assume a naming convention."""
        trace = type("FakeTrace", (), {"posterior": {}})()
        config = dense._variant_config("ability_walk")
        assert dense.variant_param_summary(trace, config) == {}

    def test_flat_variant_has_no_own_params(self):
        posterior = {"sigma_step": _FakeVar([0.02])}
        trace = type("FakeTrace", (), {"posterior": posterior})()
        config = dense._variant_config("flat")
        assert dense.variant_param_summary(trace, config) == {}

    def test_no_trace_returns_empty(self):
        config = dense._variant_config("ability_walk")
        assert dense.variant_param_summary(None, config) == {}


# --- checkpoint round trip ---------------------------------------------------

CHECKPOINT_COLS = ["component", "model", "batter", "predicted",
                   "realized_successes", "realized_rate", "trials",
                   "season", "cutoff"]


def _checkpoint_rows(models: list[str], season: int, cutoff: str,
                     batters=(1, 2), component: str = "k_rate") -> pd.DataFrame:
    rows = []
    for model in models:
        for b in batters:
            rows.append({
                "component": component, "model": model, "batter": b,
                "predicted": 0.22, "realized_successes": 20,
                "realized_rate": 0.20, "trials": 100,
                "season": season, "cutoff": cutoff,
            })
    return pd.DataFrame(rows, columns=CHECKPOINT_COLS)


class TestCheckpointRoundTrip:
    def test_a_variant_era_checkpoint_loads_and_reports_done_variants(self, tmp_path):
        path = tmp_path / "cells_bayes.parquet"
        df = _checkpoint_rows(
            ["marcel_tuned", "marcel", "bayes_flat", "bayes_walk", "bayes"],
            2026, "2026-07-01")
        df.to_parquet(path, index=False)

        loaded = dense._load_bayes_checkpoint(path)
        assert len(loaded) == len(df)

        cell = loaded[(loaded["season"] == 2026) & (loaded["cutoff"] == "2026-07-01")]
        done = dense._done_variants_for_cell(cell)
        assert done == {"flat", "ability_walk"}
        # a variant nobody has fit yet is correctly reported as not done
        assert not {"constrained_age"} <= done

    def test_missing_checkpoint_is_empty_not_an_error(self, tmp_path):
        loaded = dense._load_bayes_checkpoint(tmp_path / "does_not_exist.parquet")
        assert loaded.empty

    def test_pre_variant_checkpoint_is_rejected_with_a_clear_message(self, tmp_path):
        path = tmp_path / "cells_bayes.parquet"
        # Pre-variant era: only "bayes", none of the new arm names.
        df = _checkpoint_rows(["marcel_tuned", "marcel", "bayes"], 2026, "2026-07-01")
        df.to_parquet(path, index=False)

        with pytest.raises(RuntimeError, match="pre-variant-sweep checkpoint"):
            dense._load_bayes_checkpoint(path)

    def test_fits_json_round_trips_and_rejects_legacy(self, tmp_path):
        path = tmp_path / "bayes_fits.json"
        path.write_text('[{"cutoff": "2026-07-01", "variant": "flat", "arm": "bayes_flat"}]')
        assert dense._load_bayes_fits(path) == [
            {"cutoff": "2026-07-01", "variant": "flat", "arm": "bayes_flat"}]

        legacy = tmp_path / "legacy_fits.json"
        legacy.write_text('[{"cutoff": "2026-07-01", "scale": "2x500 draws"}]')
        with pytest.raises(RuntimeError, match="before variants existed"):
            dense._load_bayes_fits(legacy)

    def test_done_variants_for_a_never_checkpointed_cell_is_empty(self):
        assert dense._done_variants_for_cell(None) == set()


# --- the analyze-stage comparison table --------------------------------------

def _synthetic_bayes_frame() -> pd.DataFrame:
    """Two (season, cutoff) cells, four models, enough players per cell that
    the clustered/unclustered SEs differ but stay finite. `bayes_flat` and
    `bayes_walk` are each worse than `marcel_tuned` by a small, consistent
    margin (so the sign of `diff` is unambiguous) and `bayes_walk` is closer
    to `marcel_tuned` than `bayes_flat` is (so the table can show the
    pre-registration's hoped-for ordering on synthetic data, without
    asserting real numbers exist yet).
    """
    rng = np.random.default_rng(7)
    rows = []
    for season, cutoff in [(2022, "2022-07-01"), (2024, "2024-07-01")]:
        batters = np.arange(1, 61)
        realized = 0.20 + rng.normal(0, 0.03, len(batters))
        models = {
            "marcel_tuned": realized + rng.normal(0, 0.010, len(batters)),
            "marcel": realized + rng.normal(0, 0.014, len(batters)),
            "bayes_flat": realized + rng.normal(0, 0.010, len(batters)) + 0.004,
            "bayes_walk": realized + rng.normal(0, 0.010, len(batters)) + 0.001,
        }
        # "bayes" ships as an exact alias of "bayes_flat" in the real sweep
        # (make_variant_providers) — carried here too, added *after* every
        # rng draw above so it doesn't perturb their sequence, so this frame
        # exercises build_analysis's pre-existing "bayes"-named payload keys
        # (bayes_paired_by_cell etc.) the same way real checkpoint data
        # would, instead of leaving them silently empty.
        models["bayes"] = models["bayes_flat"]
        for model, predicted in models.items():
            for b, r, p in zip(batters, realized, predicted):
                rows.append({
                    "component": "k_rate", "model": model, "batter": int(b),
                    "predicted": float(p), "realized_successes": float(r) * 500,
                    "realized_rate": float(r), "trials": 500.0,
                    "season": season, "cutoff": cutoff,
                })
    return pd.DataFrame(rows)


class TestVariantComparisonTable:
    def test_reports_clustered_and_unclustered_with_the_wrong_one_labelled(self):
        bayes = _synthetic_bayes_frame()
        table = dense.build_variant_comparison_table(bayes)

        assert not table.empty
        assert set(table["arm"]) == {"bayes_flat", "bayes_walk"}
        # bayes_flat/bayes_walk vs {marcel_tuned, marcel}, plus bayes_walk vs
        # bayes_flat and bayes_flat vs bayes_walk — an arm against itself is
        # correctly excluded, and `contact_additive` (a base since BAS-83) is
        # absent from this synthetic frame, so it contributes no rows.
        assert len(table) == 2 * 2 + 2
        assert CONTACT_ARM_NAME not in set(table["base"])

        row = table[(table["arm"] == "bayes_flat") & (table["base"] == "marcel_tuned")].iloc[0]
        assert row["diff"] > 0  # bayes_flat really is worse, by construction
        assert row["n_cells_scored"] == 2
        assert row["arm_wins_cells"] + row["arm_loses_cells"] <= row["n_cells_scored"]
        assert row["unclustered_over_player_clustered_t_ratio"] > 1, (
            "clustering (54 rows/player in the real doc) should widen the SE "
            "here too, since every player recurs across both synthetic cells"
        )
        # every column the spec asks for is present
        for col in ("clustered_by_player_se", "clustered_by_player_t",
                    "clustered_by_cell_se", "clustered_by_cell_t",
                    "unclustered_se_WRONG", "unclustered_t_WRONG"):
            assert col in row

    def test_a_variant_against_itself_is_excluded(self):
        bayes = _synthetic_bayes_frame()
        table = dense.build_variant_comparison_table(bayes)
        assert not ((table["arm"] == table["base"])).any()

    def test_empty_frame_renders_a_message_not_a_crash(self):
        assert "no variant comparisons" in dense.render_variant_table(pd.DataFrame())

    def test_render_produces_one_line_per_row(self):
        bayes = _synthetic_bayes_frame()
        table = dense.build_variant_comparison_table(bayes)
        text = dense.render_variant_table(table)
        # header + separator + one row per comparison
        assert len(text.splitlines()) == len(table) + 2


# --- build_analysis exercised end to end on synthetic checkpoints -----------

class TestBuildAnalysisSynthetic:
    """Exercises the whole analyze code path (build_analysis -> the variant
    table -> render) against synthetic checkpoints, standing in for the real
    `data/eval/dense_intraseason/` output (gitignored, absent in a fresh
    worktree) per the task's "do not run the real sweep" instruction."""

    def test_build_analysis_runs_and_carries_the_variant_table(self, tmp_path):
        bayes = _synthetic_bayes_frame()
        bayes_path = tmp_path / "cells_bayes.parquet"
        bayes.to_parquet(bayes_path, index=False)
        cheap_path = tmp_path / "cells_cheap.parquet"  # deliberately absent
        out_json = tmp_path / "analysis.json"

        payload = dense.build_analysis(cheap_path, bayes_path, out_json)

        assert out_json.exists()
        assert "bayes_variant_comparison" in payload
        assert len(payload["bayes_variant_comparison"]) > 0
        rendered = dense.render_variant_table(
            pd.DataFrame(payload["bayes_variant_comparison"]))
        assert "bayes_flat" in rendered and "bayes_walk" in rendered


class TestVariantParamNamesMatchTheModel:
    """`variant_param_summary` skips a name the trace does not carry, which is
    the right behaviour at runtime and a silent failure at authoring time: the
    first sweep recorded no peak age at all because this table said `peak` and
    the model says `peak_age`. Pin the names against the model's own source so
    a rename is caught here rather than discovered in an empty results column.

    Reads `src/models/pa_rate.py` as text — the module imports pymc, which
    CI does not install. That is the file the model graph lives in since
    BAS-73 made the component a parameter; `src/models/pa_k_rate.py` is now a
    wrapper that pins `component="k_rate"` and declares no variables of its
    own.
    """

    def _model_source(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        return (root / "src" / "models" / "pa_rate.py").read_text()

    def test_every_declared_param_is_named_in_the_model(self):
        import re

        source = self._model_source()
        declared = {p for names in dense.VARIANT_OWN_PARAMS.values() for p in names}
        for name in sorted(declared):
            # The covariate coefficients are declared in a loop over the
            # aggregate names, so their literal is an f-string prefix rather
            # than a whole name; matching the prefix plus checking the
            # aggregate is one this model knows is the same guarantee.
            if name.startswith("beta_cov_"):
                from src.models.pa_covariates import CONTACT_COVARIATES

                assert re.search(r'pm\.Normal\(f"beta_cov_\{name\}"', source), (
                    "src/models/pa_rate.py no longer declares beta_cov_<aggregate>"
                )
                assert name[len("beta_cov_"):] in CONTACT_COVARIATES, (
                    f"{name!r} is in VARIANT_OWN_PARAMS but names no aggregate "
                    f"in src.models.pa_covariates.CONTACT_COVARIATES"
                )
                continue
            pattern = rf'pm\.(?:Deterministic|HalfNormal|Normal|Beta)\(\s*"{name}"'
            assert re.search(pattern, source), (
                f"{name!r} is in VARIANT_OWN_PARAMS but no PyMC variable of that "
                f"name exists in src/models/pa_rate.py — variant_param_summary "
                f"would silently record nothing for it"
            )

    def test_the_walk_and_the_age_curve_each_declare_their_own_parameters(self):
        assert "sigma_step" in dense.VARIANT_OWN_PARAMS["ability_walk"]
        assert "peak_age" in dense.VARIANT_OWN_PARAMS["constrained_age"]
        assert "beta_cov_barrel" in dense.VARIANT_OWN_PARAMS["contact"]


class TestDegenerateClusteredT:
    """A clustered t backed by one cluster is missing, not enormous.

    The first rendering of this table, run against a sweep that had scored a
    single (season, cutoff) pair, printed `t(cell) = 2.3e15` beside an honest
    `t(player) = 1.68`. With one cluster the between-cluster variance has no
    degrees of freedom and the ratio is whatever the last rounding error was.
    A number that reads as a statistic and is not one has to be suppressed
    where it is computed, not hidden in the formatting.
    """

    def test_one_cluster_nans_the_t_and_se(self):
        out = dense._suppress_degenerate_t(
            {"diff": 0.001, "n": 300, "se": 1e-18, "t": 2.3e15, "n_clusters": 1})
        assert out["t"] != out["t"]        # NaN
        assert out["se"] != out["se"]
        assert out["diff"] == 0.001        # everything else survives
        assert out["n"] == 300

    def test_two_or_more_clusters_pass_through_untouched(self):
        original = {"diff": 0.001, "n": 300, "se": 0.0004, "t": 2.5, "n_clusters": 2}
        assert dense._suppress_degenerate_t(original) == original

    def test_a_missing_cluster_count_is_treated_as_degenerate(self):
        out = dense._suppress_degenerate_t({"t": 5.0, "se": 0.1})
        assert out["t"] != out["t"]

    def test_the_renderer_prints_a_dash_for_a_suppressed_t(self):
        import pandas as pd

        row = {
            "arm": "bayes_walk", "base": "marcel_tuned", "n": 338,
            "n_cells_scored": 1, "diff": 0.00033,
            "clustered_by_player_t": 0.39,
            "clustered_by_cell_t": None,          # JSON round-trip turns NaN into null
            "unclustered_t_WRONG": 0.39,
            "unclustered_over_player_clustered_t_ratio": 1.0,
            "arm_wins_cells": 0, "arm_loses_cells": 1,
        }
        text = dense.render_variant_table(pd.DataFrame([row]))
        assert "nan" not in text.lower()
        assert "-" in text
        assert "bayes_walk  " in text or "bayes_walk " in text   # column not run together


# --- BAS-73: components --------------------------------------------------------

class TestComponentIsolation:
    """One fit per (component, cutoff, variant). The failure mode is the same
    one `TestCacheIsolation` guards for variants — a fit served under the
    wrong label — one axis over, and it is worse here because the numbers
    would look plausible: a K% fit returned for a BB% request comes back as a
    well-formed frame of rates near .22 that the harness will happily score
    against BB% realizations.
    """

    def _fake_fit(self, calls):
        def fake_fit(cutoff_date, predict_year, config, unseen=None):
            calls.append((config.component, config.variant()))
            return BayesFit(
                cutoff_date=str(cutoff_date), predict_year=int(predict_year),
                projections=pd.DataFrame({
                    "batter": [1, 2],
                    config.projected_col(): [0.22, 0.22],
                }),
                diagnostics={"max_rhat": 1.0, "min_ess_bulk": 500,
                             "divergences": 0},
                config=config, trace=None, model_data=None, data_summary={},
            )
        return fake_fit

    def test_each_component_gets_its_own_fit(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr("src.eval.bayes_arm.fit_bayes_k_rate",
                            self._fake_fit(calls))

        train = pd.DataFrame({"batter": [1, 2], "season": [2026, 2026]})
        fits: list[dict] = []
        for component in ("k_rate", "bb_rate", "hr_rate"):
            providers = dense.make_variant_providers(
                "2026-07-01", 2026, ["flat"], {}, fits, component=component)
            providers["bayes_flat"](train, COMPONENTS[component], 2026)

        assert calls == [("k_rate", "flat"), ("bb_rate", "flat"),
                         ("hr_rate", "flat")]
        assert [f["component"] for f in fits] == ["k_rate", "bb_rate", "hr_rate"]

    def test_the_provider_cache_keys_on_the_component(self, monkeypatch):
        """Directly against `bayes_k_rate_provider`, not through the sweep:
        the cache is where a component-blind key would actually bite."""
        from src.eval.bayes_arm import BayesArmConfig, bayes_k_rate_provider

        calls: list[tuple] = []
        monkeypatch.setattr("src.eval.bayes_arm.fit_bayes_k_rate",
                            self._fake_fit(calls))
        train = pd.DataFrame({"batter": [1, 2], "season": [2026, 2026]})

        bb = bayes_k_rate_provider("2026-07-01", 2026,
                                   BayesArmConfig(component="bb_rate"))
        out = bb(train, COMPONENTS["bb_rate"], 2026)
        assert list(out.columns) == ["batter", "predicted"]
        bb(train, COMPONENTS["bb_rate"], 2026)          # cache hit
        assert calls == [("bb_rate", "flat")], "a repeat must not refit"

        # And the same provider refuses the component it was not built for
        # rather than serving the cached BB% fit under a K% label.
        with pytest.raises(ValueError, match="built to fit 'bb_rate'"):
            bb(train, COMPONENTS["k_rate"], 2026)

    def test_a_fit_reporting_the_wrong_component_raises(self, monkeypatch):
        """Defense in depth, mirroring the variant check: if a fit ever comes
        back for a different component than its provider was built for, the
        sweep must raise rather than record it under the asked-for name."""
        def fake_fit_always_k(cutoff_date, predict_year, config, unseen=None):
            wrong = BayesArmConfig(component="k_rate")
            return BayesFit(
                cutoff_date=str(cutoff_date), predict_year=int(predict_year),
                projections=pd.DataFrame({"batter": [1],
                                          "projected_bb_rate": [0.08]}),
                diagnostics={}, config=wrong, trace=None, model_data=None,
                data_summary={},
            )

        monkeypatch.setattr("src.eval.bayes_arm.fit_bayes_k_rate",
                            fake_fit_always_k)
        providers = dense.make_variant_providers(
            "2026-07-01", 2026, ["flat"], {}, [], component="bb_rate")
        train = pd.DataFrame({"batter": [1], "season": [2026]})
        with pytest.raises(RuntimeError, match="component isolation broken"):
            providers["bayes_flat"](train, COMPONENTS["bb_rate"], 2026)


class TestComponentKeyedCheckpoints:
    """A 2026-07-01 BB% cell and a 2026-07-01 K% cell are different
    measurements. Before the component joined the key, the two-key groupby
    that rebuilds `cells` would collapse them into one entry and the second
    write would silently replace the first — losing a whole component's
    sweep with no error anywhere."""

    def test_two_components_at_one_cutoff_are_two_cells(self, tmp_path):
        path = tmp_path / "cells_bayes.parquet"
        df = pd.concat([
            _checkpoint_rows(["marcel_tuned", "bayes_flat"], 2026,
                             "2026-07-01", component="k_rate"),
            _checkpoint_rows(["marcel_tuned", "bayes_walk"], 2026,
                             "2026-07-01", component="bb_rate"),
        ], ignore_index=True)
        df.to_parquet(path, index=False)

        loaded = dense._load_bayes_checkpoint(path)
        cells = {k: g for k, g in
                 loaded.groupby(["component", "season", "cutoff"])}
        assert set(cells) == {("k_rate", 2026, "2026-07-01"),
                              ("bb_rate", 2026, "2026-07-01")}
        assert dense._done_variants_for_cell(
            cells[("k_rate", 2026, "2026-07-01")]) == {"flat"}
        assert dense._done_variants_for_cell(
            cells[("bb_rate", 2026, "2026-07-01")]) == {"ability_walk"}

    def test_a_done_k_rate_cell_does_not_mark_bb_rate_done(self, tmp_path):
        """The resume check reads `cells.get((component, season, cutoff))`;
        this is the assertion that says a finished K% sweep does not make the
        BB% sweep look finished too."""
        path = tmp_path / "cells_bayes.parquet"
        _checkpoint_rows(["bayes_flat"], 2026, "2026-07-01",
                         component="k_rate").to_parquet(path, index=False)
        loaded = dense._load_bayes_checkpoint(path)
        cells = {k: g for k, g in
                 loaded.groupby(["component", "season", "cutoff"])}
        assert dense._done_variants_for_cell(
            cells.get(("bb_rate", 2026, "2026-07-01"))) == set()


class TestBayesComponentsCli:
    def test_the_default_is_k_rate_alone(self):
        """Nothing existing changes: a sweep that names no component runs the
        one it always ran."""
        assert dense.DEFAULT_BAYES_COMPONENTS == ["k_rate"]

    def test_only_the_three_per_pa_binomials_are_offered(self):
        assert set(dense.BAYES_COMPONENTS) == {"k_rate", "bb_rate", "hr_rate"}

    def test_the_offered_set_is_exactly_what_the_model_registers(self):
        """The sweep's list and the model's registry must not drift: a
        component offered here but unknown there fails at the first fit, and
        one registered there but missing here is simply never swept."""
        from src.models.pa_components import RATE_COMPONENTS

        assert set(dense.BAYES_COMPONENTS) == set(RATE_COMPONENTS)

    def test_run_bayes_rejects_a_component_the_arm_cannot_fit(self):
        with pytest.raises(ValueError, match="unknown bayes component"):
            dense.run_bayes(pd.DataFrame(), {}, (2026,), [],
                            components=["babip"])


# --- the joint variants (BAS-84) ---------------------------------------------

class TestJointVariants:
    """`--variants joint_walk` has to reach `BayesArmConfig(joint=True,
    ability_walk=True)` and nothing else, and must not change what a command
    that names no variants fits."""

    def test_the_joint_variants_round_trip_through_the_config(self):
        for variant in ("joint", "joint+ability_walk"):
            config = dense._variant_config(variant)
            assert config.variant() == variant
            assert config.joint is True
        assert dense._variant_config("joint+ability_walk").ability_walk is True
        assert dense._variant_config("joint").ability_walk is False

    def test_the_doc_spellings_resolve_to_the_config_vocabulary(self):
        assert dense.resolve_variant("joint_walk") == "joint+ability_walk"
        assert dense.resolve_variant("joint_flat") == "joint"
        # Everything else passes through untouched, so an unknown name still
        # produces the caller's own error rather than a KeyError here.
        assert dense.resolve_variant("ability_walk") == "ability_walk"
        assert dense.resolve_variant("nonsense") == "nonsense"

    def test_an_alias_and_its_target_are_the_same_config(self):
        assert (dense._variant_config("joint_walk")
                == dense._variant_config("joint+ability_walk"))

    def test_every_alias_points_at_a_real_variant(self):
        for alias, target in dense.VARIANT_ALIASES.items():
            assert target in dense.VARIANT_ARM_NAMES
            assert alias not in dense.VARIANT_ARM_NAMES

    def test_the_joint_arms_have_their_own_names_on_the_board(self):
        assert dense.VARIANT_ARM_NAMES["joint+ability_walk"] == "bayes_joint_walk"
        assert dense.VARIANT_ARM_NAMES["joint"] == "bayes_joint"
        assert dense.ARM_NAME_VARIANT["bayes_joint_walk"] == "joint+ability_walk"

    def test_the_default_sweep_is_still_the_four_single_component_variants(self):
        """The joint arms are opt-in. `DEFAULT_VARIANTS` used to be read off
        `VARIANT_ARM_NAMES`, which would have quadrupled the cost of every
        command that names no variants the moment a new arm was registered."""
        assert dense.DEFAULT_VARIANTS == [
            "flat", "ability_walk", "constrained_age",
            "ability_walk+constrained_age"]

    def test_the_variant_summary_records_nothing_misleading_for_a_joint_fit(self):
        """A joint trace carries no scalar `sigma_step` -- it is a vector,
        one per component. The summary must skip it rather than average three
        components' step sizes into one number; the per-component values go
        into the fit record's `joint_params` instead."""
        config = dense._variant_config("joint+ability_walk")
        posterior = {"sigma_step_k_rate": _FakeVar([[0.10, 0.12]])}
        trace = type("FakeTrace", (), {"posterior": posterior})()
        assert dense.variant_param_summary(trace, config) == {}


# --- the park arm (BAS-86) ---------------------------------------------------

def _synthetic_park_frame() -> pd.DataFrame:
    """Two cells, three models, HR/PA. `marcel_tuned_park` is better than
    `marcel_tuned` by a small consistent margin and the half-strength arm by
    half of it, so every sign in the table below is unambiguous."""
    rng = np.random.default_rng(11)
    rows = []
    for season, cutoff in [(2024, "2024-07-01"), (2025, "2025-07-01")]:
        batters = np.arange(1, 41)
        realized = 0.030 + rng.normal(0, 0.006, len(batters))
        err = rng.normal(0, 0.004, len(batters))
        models = {
            "marcel_tuned": realized + err + 0.0010,
            "marcel_tuned_park": realized + err + 0.0004,
            "marcel_tuned_park_half": realized + err + 0.0007,
        }
        for model, predicted in models.items():
            for b, r, p in zip(batters, realized, predicted):
                rows.append({
                    "component": "hr_rate", "model": model, "batter": int(b),
                    "predicted": float(p), "realized_successes": float(r) * 300,
                    "realized_rate": float(r), "trials": 300.0,
                    "season": season, "cutoff": cutoff,
                })
    return pd.DataFrame(rows)


def _park_fixture():
    """A two-batter season table + PA frame + factor table, enough for
    `run_cheap` to produce one real cell."""
    dates = pd.date_range("2024-04-01", periods=120, freq="D")
    rows = []
    # Both play at COL; batter 1 is the home club's, batter 2 the visitor's,
    # so "which park is his" is the batting club and not the venue.
    for batter, home, away, topbot in ((1, "COL", "SF", "Bot"),
                                       (2, "COL", "SF", "Top")):
        for i, d in enumerate(dates):
            rows.append({
                "batter": batter, "pitcher": 900 + batter, "game_pk": i,
                "game_date": d, "game_year": 2024, "event": "strikeout",
                "is_k": int(i % 4 == 0), "is_bb": 0, "is_hbp": 0,
                "is_hit": int(i % 5 == 0), "is_hr": int(i % 10 == 0),
                "is_single": int(i % 5 == 0) - int(i % 10 == 0),
                "is_double": 0, "is_triple": 0,
                "home_team": home, "away_team": away, "inning_topbot": topbot,
            })
    pa = pd.concat([pd.DataFrame(rows)] * 3, ignore_index=True)
    seasons = pd.DataFrame({
        "batter": [1, 2, 1, 2], "season": [2022, 2022, 2023, 2023],
        "pa": [500] * 4, "ab": [450] * 4, "hr": [20, 12, 22, 10],
        "k": [110] * 4, "bb": [45] * 4, "hbp": [3] * 4, "sf": [2] * 4,
        "h": [120] * 4, "doubles": [25] * 4, "triples": [2] * 4,
        "xb_points": [89, 65, 95, 59], "bip": [300] * 4,
        "hits_in_play": [100] * 4, "age": [27] * 4,
    })
    pf = pd.DataFrame({
        "team": ["COL", "SF"], "game_year": [2024, 2024],
        "hr_park_factor": [1.30, 0.70], "k_park_factor": [0.90, 1.05],
    })
    return seasons, pa, pf


class TestParkArmAnalysis:
    def test_the_gain_is_reported_as_a_share_of_the_baseline_mae(self):
        """architecture.md §3's floor is written in percent of the served
        baseline's MAE, so the table has to carry that percent and not just
        the raw difference."""
        cells = _synthetic_park_frame()
        row = dense.arm_comparison(cells, "marcel_tuned_park", "marcel_tuned",
                                   "hr_rate")
        assert row["diff"] < 0                      # the arm wins, by construction
        assert row["base_mae"] > row["arm_mae"] > 0
        assert row["pct_of_base"] == pytest.approx(
            100 * row["diff"] / row["base_mae"])
        assert row["pct_of_base"] < 0
        assert row["season"] is None

    def test_per_season_rows_are_scored_on_that_season_only(self):
        cells = _synthetic_park_frame()
        row = dense.arm_comparison(cells, "marcel_tuned_park", "marcel_tuned",
                                   "hr_rate", season=2024)
        assert row["season"] == 2024
        assert row["n"] == 40
        assert row["n_cells_scored"] == 1

    def test_the_analysis_covers_both_arms_pooled_and_per_season(self):
        out = dense.park_arm_analysis(_synthetic_park_frame())
        arms = {r["arm"] for r in out["pooled"]}
        assert arms == {"marcel_tuned_park", "marcel_tuned_park_half"}
        assert {r["season"] for r in out["by_season"]} == {2024, 2025}
        # half-strength is half the correction, so it should sit between the
        # base and the full arm — the ordering the diagnostic exists to show.
        full = [r for r in out["pooled"] if r["arm"] == "marcel_tuned_park"][0]
        half = [r for r in out["pooled"] if r["arm"] == "marcel_tuned_park_half"][0]
        assert full["diff"] < half["diff"] < 0

    def test_no_park_arm_in_the_frame_is_an_empty_section(self):
        cells = _synthetic_park_frame()
        assert dense.park_arm_analysis(
            cells[cells["model"] != "marcel_tuned_park"]) == {}

    def test_the_rendered_table_survives_a_json_round_trip(self):
        """`build_analysis` writes the payload to JSON and the analyze stage
        renders it back, so NaN has become None by the time it is printed."""
        import json

        out = dense.park_arm_analysis(_synthetic_park_frame())
        rows = json.loads(json.dumps(out["pooled"]))
        text = dense.render_park_table(rows)
        assert "marcel_tuned_park" in text and "% of base" in text
        assert dense.render_park_table([]) == "(no park-arm rows)"

    def test_build_analysis_carries_the_park_section(self, tmp_path):
        cheap = tmp_path / "cells_cheap.parquet"
        _synthetic_park_frame().to_parquet(cheap, index=False)
        payload = dense.build_analysis(cheap, tmp_path / "missing.parquet",
                                       tmp_path / "analysis.json")
        assert payload["park_arm"]["base"] == "marcel_tuned"
        assert payload["park_arm"]["pooled"]

    def test_the_cheap_sweep_adds_the_arm_without_moving_the_common_set(self,
                                                                       tmp_path):
        """The whole reason the park arm can join the cheap cells at all: it
        projects exactly the batters `marcel_tuned` projects, so the other
        arms' populations are untouched."""
        seasons, pa, pf = _park_fixture()
        without = dense.run_cheap(seasons, {2024: pa}, ["hr_rate"], (2024,),
                                  ["05-01"], tmp_path, min_trials=1)
        with_arm = dense.run_cheap(seasons, {2024: pa}, ["hr_rate"], (2024,),
                                   ["05-01"], tmp_path, min_trials=1,
                                   park_factors=pf)
        assert "marcel_tuned_park" in set(with_arm["model"])
        base_before = without[without["model"] == "marcel_tuned"]
        base_after = with_arm[with_arm["model"] == "marcel_tuned"]
        assert set(base_before["batter"]) == set(base_after["batter"])
        assert (base_before.sort_values("batter")["predicted"].to_numpy()
                == pytest.approx(
                    base_after.sort_values("batter")["predicted"].to_numpy()))

    def test_the_arm_moves_a_batter_in_the_direction_of_his_park(self, tmp_path):
        seasons, pa, pf = _park_fixture()
        cells = dense.run_cheap(seasons, {2024: pa}, ["hr_rate"], (2024,),
                                ["05-01"], tmp_path, min_trials=1,
                                park_factors=pf)
        base = cells[cells["model"] == "marcel_tuned"].set_index("batter")
        arm = cells[cells["model"] == "marcel_tuned_park"].set_index("batter")
        # batter 1 plays in the 1.30 park, batter 2 in the 0.70 one
        assert arm.loc[1, "predicted"] == pytest.approx(
            base.loc[1, "predicted"] * 1.30)
        assert arm.loc[2, "predicted"] == pytest.approx(
            base.loc[2, "predicted"] * 0.70)


# --- the measurement variants (BAS-85) ---------------------------------------

class TestMeasurementVariants:
    """`--variants measurement_walk` has to reach
    `BayesArmConfig(measurement=True, joint=True, ability_walk=True)`, get
    its own arm name on the board, and keep the posterior interval
    docs/bayes-measurement.md's prediction 4 is scored on.
    """

    def test_the_measurement_variants_round_trip_through_the_config(self):
        for variant in ("measurement", "measurement+ability_walk"):
            config = dense._variant_config(variant)
            assert config.variant() == variant
            assert config.measurement is True
            # The channels read a latent the joint graph writes, so a
            # measurement fit is a joint fit — with the two components the
            # pre-registration names, which `joint_components` decides.
            assert config.joint is True
        assert dense._variant_config("measurement+ability_walk").ability_walk
        assert not dense._variant_config("measurement").ability_walk

    def test_the_doc_spelling_resolves_to_the_config_spelling(self):
        assert dense.resolve_variant("measurement_walk") == (
            "measurement+ability_walk")
        assert dense.resolve_variant("measurement_flat") == "measurement"
        assert (dense._variant_config("measurement_walk")
                == dense._variant_config("measurement+ability_walk"))

    def test_the_measurement_arms_have_their_own_names_on_the_board(self):
        assert dense.VARIANT_ARM_NAMES["measurement+ability_walk"] == (
            "bayes_measurement_walk")
        assert dense.VARIANT_ARM_NAMES["measurement"] == "bayes_measurement"
        assert dense.ARM_NAME_VARIANT["bayes_measurement_walk"] == (
            "measurement+ability_walk")

    def test_they_are_not_in_the_default_sweep(self):
        """A command written before BAS-85 fits what it always fit."""
        assert not any("measurement" in v for v in dense.DEFAULT_VARIANTS)

    def test_the_coverage_interval_is_kept_for_exactly_two_arms(self):
        """Prediction 4 compares `measurement`'s 80% interval against
        `bayes_walk`'s, so those two arms carry the 10th and 90th percentiles
        and nothing else does — an interval on every arm would change every
        arm's projection frame for the sake of two."""
        assert dense._variant_config("measurement_walk").extra_quantiles == (
            dense.COVERAGE_QUANTILES)
        assert dense._variant_config("ability_walk").extra_quantiles == (
            dense.COVERAGE_QUANTILES)
        assert dense._variant_config("flat").extra_quantiles == ()
        assert dense._variant_config("joint_walk").extra_quantiles == ()
