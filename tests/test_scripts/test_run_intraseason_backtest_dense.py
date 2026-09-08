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
                     batters=(1, 2)) -> pd.DataFrame:
    rows = []
    for model in models:
        for b in batters:
            rows.append({
                "component": "k_rate", "model": model, "batter": b,
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
        # bayes_flat — bayes_flat vs bayes_flat is correctly excluded.
        assert len(table) == 2 * 2 + 1

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
