"""Guard against modal_functions/app.py defining a model again (issue #86).

The point of issue #86 was that Modal's weekly production refit ran an
inlined copy of every hitter-rate model that had quietly diverged from the
`src/` code the test suite and the backtest harness gate — the same name,
two different models, and nothing noticed. The fix was making
`modal_functions/app.py` call `src.models.*` instead of inlining PyMC
model code a second time. This test is what keeps it fixed: it fails the
moment someone pastes a `pm.Model(...)` block back into `app.py`, with a
message that says why that is not just a style problem.

Two tiny models are allowed to stay inlined — `smoke_test` and
`wandb_integration_test` sample a few points from `N(0.26, 0.05)` to prove
the image builds, the volumes mount, and wandb logging works end-to-end.
That is infrastructure validation, not a hitter-rate model, and each is
capped at a small line count so the allowlist itself cannot quietly become
the next place to hide a real model under an infrastructure-sounding name.

Static/source-level only. `pymc` is not installed in CI on purpose
(`requirements-ci.txt`: "MCMC runs on Modal, not in CI"), so this test
must never `import modal_functions.app` or `pymc` — it reads `app.py` as
text and parses it with `ast`, nothing more.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_PY = ROOT / "modal_functions" / "app.py"

# Functions allowed to build a tiny illustrative PyMC model, and the most
# lines any one of them may span before the allowlist itself becomes a
# place to hide a real model.
ALLOWED_MODEL_FUNCTIONS = {"smoke_test", "wandb_integration_test"}
# wandb_integration_test's actual model is two lines (mu, sigma on a
# five-point observed array) but the function also logs diagnostics,
# tables, plots and an artifact to prove the whole wandb pipeline works —
# 130 lines covers that honestly while staying an order of magnitude
# under the ~350-750 line real models this issue removed from this file.
ALLOWED_FUNCTION_MAX_LINES = 130

# Calls that mean "this is building a model", not "this is calling one" —
# distribution constructors and model-structure primitives. This is the
# exact list of pm.* names the four inlined copies issue #86 found actually
# used (train_pa_k_rate, train_iso_model, train_babip_model each built a
# pm.Model with a subset of these), plus the sampler itself and a few
# distributions common enough elsewhere in PyMC to be worth catching too.
BANNED_CALL_PATTERN = re.compile(
    r"\bpm\.("
    r"Model|Normal|HalfNormal|Binomial|Bernoulli|Deterministic|Data|"
    r"ZeroSumNormal|InverseGamma|HalfCauchy|Beta|Gamma|StudentT|Poisson|"
    r"Uniform|Exponential|LogNormal|Dirichlet|MvNormal|sample|"
    r"gp\.HSGP|gp\.cov"
    r")\s*\("
)


def _function_line_ranges(source: str) -> dict[str, tuple[int, int]]:
    """name -> (first_line, last_line), 1-indexed inclusive, decorators
    included, for every function defined anywhere in the module."""
    tree = ast.parse(source)
    ranges: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            ranges[node.name] = (start, node.end_lineno)
    return ranges


def test_app_py_parses():
    """Baseline: the file is valid Python before this test goes source-diffing it."""
    ast.parse(APP_PY.read_text())


def test_allowlisted_model_functions_still_exist_and_are_small():
    """If these get renamed or grow, the allowlist below is checking the
    wrong thing — or hiding a real model behind an infrastructure name."""
    ranges = _function_line_ranges(APP_PY.read_text())
    for name in ALLOWED_MODEL_FUNCTIONS:
        assert name in ranges, (
            f"expected an allowlisted smoke-test function named {name!r} in "
            f"{APP_PY.relative_to(ROOT)} — if it was renamed, update "
            f"ALLOWED_MODEL_FUNCTIONS in this test to match, not just to "
            f"make it pass"
        )
        start, end = ranges[name]
        n_lines = end - start + 1
        assert n_lines <= ALLOWED_FUNCTION_MAX_LINES, (
            f"{name}() in {APP_PY.relative_to(ROOT)} is {n_lines} lines — "
            f"over the {ALLOWED_FUNCTION_MAX_LINES}-line budget for a smoke "
            f"test. If it grew because a real model got pasted in under an "
            f"infrastructure-sounding name, that is exactly the failure "
            f"this guard exists to catch; if it grew for a legitimate "
            f"reason, split the new code into its own function so it is "
            f"NOT in ALLOWED_MODEL_FUNCTIONS."
        )


def test_no_inlined_model_outside_the_smoke_test_allowlist():
    """The test that actually matters (issue #86).

    `modal_functions/app.py` must CALL a model from `src/models/`, never
    define one. This is not a style preference: `src/models/pa_k_rate.py`
    is what the test suite imports, what `src/eval/bayes_arm.py`
    backtests, and what the gate rule (docs/architecture.md #3) scores
    against Marcel. A second, inlined copy in `app.py` can silently drift
    from that one — and did: the production weekly refit spent months
    fitting an HSGP age curve with no opposing-pitcher term while every
    published number about "the Bayesian K% model" was scored against a
    quadratic age curve with one. Two different models, one name, and
    nothing noticed until someone went and diffed them by hand. See
    docs/modal-src-divergence.md for the full account.

    If this test fails, the fix is almost never "update this test" — it is
    "import the model from src/models/ instead of pasting it in here."
    """
    source = APP_PY.read_text()
    ranges = _function_line_ranges(source)
    lines = source.splitlines()

    allowed_line_numbers: set[int] = set()
    for name in ALLOWED_MODEL_FUNCTIONS:
        if name not in ranges:
            continue
        start, end = ranges[name]
        allowed_line_numbers.update(range(start, end + 1))

    offending = []
    for lineno, line in enumerate(lines, start=1):
        if lineno in allowed_line_numbers:
            continue
        if BANNED_CALL_PATTERN.search(line):
            offending.append((lineno, line.strip()))

    assert not offending, (
        "modal_functions/app.py defines a PyMC model outside the "
        f"{sorted(ALLOWED_MODEL_FUNCTIONS)} smoke-test allowlist — this is "
        "exactly the failure mode issue #86 exists to prevent (see "
        "docs/modal-src-divergence.md). The gated model lives in "
        "src/models/*.py; modal_functions/app.py should import and call "
        "it, not paste in a second copy that can quietly diverge from it "
        "again. Offending line(s):\n"
        + "\n".join(f"  app.py:{ln}: {text}" for ln, text in offending)
    )


def test_train_functions_call_into_src_models():
    """The positive half of the guard above: the three training entrypoints
    each actually reach into src/models/ for their model code, rather than
    merely lacking a pm.Model call by accident (e.g. because the function
    body was gutted rather than rewired).
    """
    source = APP_PY.read_text()
    ranges = _function_line_ranges(source)
    lines = source.splitlines()
    for fn_name, module_name in [
        ("train_pa_k_rate", "pa_k_rate"),
        ("train_iso_model", "iso_rate"),
        ("train_babip_model", "babip_rate"),
    ]:
        assert fn_name in ranges, (
            f"{fn_name} is missing from {APP_PY.relative_to(ROOT)} — the "
            f"workflow_dispatch component it backs (modal-refit.yml) would "
            f"have nothing to call"
        )
        start, end = ranges[fn_name]
        body = "\n".join(lines[start - 1:end])
        assert f"src.models.{module_name}" in body, (
            f"{fn_name}() in {APP_PY.relative_to(ROOT)} no longer imports "
            f"src.models.{module_name} — it should call the model from "
            f"src/models/, not rebuild it locally."
        )
