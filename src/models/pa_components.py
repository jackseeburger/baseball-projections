"""The per-PA rate component registry (BAS-73), with no modelling dependency.

`src/models/pa_rate.py` imports pymc and arviz at module level, and
`requirements-ci.txt` leaves both out on purpose — MCMC runs on Modal, not
in CI. Everything a caller needs in order to *name* a component, validate
it, or read its column names lives here instead, so the eval harness and
the dense sweep can resolve `bb_rate` without importing a sampler. The
model module re-exports all of it; nothing that imported these names from
`pa_rate` has to change.
"""
from __future__ import annotations

from dataclasses import dataclass

# Age window a constrained peak is allowed to fall in, and the reference age
# quadratic model centers on. Matches `src.eval.tuning.AGE_PEAK_WINDOW`,
# which is where the same constraint (peak inside 25-31, opposite-signed
# slopes so the curve turns over instead of running as a level) was first
# imposed on tuned Marcel's age curve; duplicated here as a constant rather
# than imported so this module's only dependency on `src.eval` stays what it
# already was (none) — the numbers are copied on purpose, not accidentally.
AGE_PEAK_WINDOW = (25.0, 31.0)
# Sign of "a bigger number is a better hitter", per component — copied from
# `src.eval.tuning.AGE_DIRECTION` for the same reason AGE_PEAK_WINDOW above
# is copied (this module keeps its zero dependency on `src.eval`), and pinned
# against the original by
# `tests/test_models/test_pa_rate.py::test_age_direction_matches_the_tuning_module`.
# It is what turns the constrained age curve from a valley into a hill: K%
# alone is the component where a bigger rate is a *worse* hitter, so its
# curve falls toward the peak and rises after it, while BB% and HR/PA peak
# and decline like the counting skills they are.
AGE_DIRECTION = {"k_rate": -1.0, "bb_rate": 1.0, "hr_rate": 1.0}


@dataclass(frozen=True)
class RateComponent:
    """Everything that differs between K%, BB% and HR/PA, in one object.

    The model graph, the cell structure and the likelihood are identical
    across the three; what a component carries is which PA column counts as a
    success, what the fitted columns are called on the way out, and two
    priors that genuinely depend on the rate's scale and shape.

    league_init_mu:
        Prior mean for `league_init`, on the logit scale. `None` means "read
        it off the training data's earliest season" — the honest default,
        since the whole point of the league random walk is that the level is
        data, and a hard-coded number for a rate that has moved (HR/PA ran
        .0363 in 2019 and .0286 in 2022, a 0.25 swing on the logit) is a
        thumb on the scale that nobody would notice.

        `k_rate` is the one component that pins it, at -1.27. That is the
        number the published K% sweep (docs/bayes-variants.md) was fit under,
        and changing it would break the bit-for-bit reproduction that says
        this refactor did not move the K% arm. The pin costs nothing in
        substance: the empirical earliest-season K% logit runs -1.21 to -1.26
        across 2019-2026, so the pin sits within 0.06 of what the data-driven
        rule would pick, against a prior sd of 0.3 — a fifth of one prior
        standard deviation, on a parameter the league walk re-estimates from
        hundreds of thousands of plate appearances anyway.

    age_direction:
        +1 when a bigger rate is a better hitter (BB%, HR/PA), -1 when it is
        worse (K%). Sets which way the constrained age curve turns and the
        sign of the quadratic's `beta_age2` prior mean. See
        `AGE_DIRECTION` and `build_model`'s age block.
    """
    name: str                    # "k_rate"
    numerator: str               # PA-parquet column that counts a success
    league_init_mu: float | None
    age_direction: float
    park_factor_col: str         # column in park_factors.parquet, if present

    @property
    def projected_col(self) -> str:
        return f"projected_{self.name}"

    @property
    def obs_name(self) -> str:
        """Name of the Binomial likelihood variable in the trace.

        K% keeps the historical `obs_k` — it is written into every saved K%
        trace and into `log_to_wandb`'s artifact metadata, and renaming it
        would orphan those for no gain.
        """
        return "obs_k" if self.name == "k_rate" else f"obs_{self.name}"

    @property
    def career_col(self) -> str:
        return f"career_{self.name}"

    def out_columns(self) -> dict:
        """The four posterior-summary column names `generate_projections`
        writes, keyed by the suffix that names them."""
        return {
            "mean": self.projected_col,
            "std": f"{self.name}_std",
            "lower": f"{self.name}_lower",
            "upper": f"{self.name}_upper",
        }


RATE_COMPONENTS: dict[str, RateComponent] = {
    # -1.27 is the pin the K% sweep was run under; see `league_init_mu`.
    "k_rate": RateComponent("k_rate", "is_k", -1.27,
                            AGE_DIRECTION["k_rate"], "k_park_factor"),
    # `is_bb` alone, matching `src.eval.backtest.COMPONENTS["bb_rate"]`,
    # whose `bb` column is built from `is_bb` with `hbp` kept separate. See
    # the module docstring.
    "bb_rate": RateComponent("bb_rate", "is_bb", None,
                             AGE_DIRECTION["bb_rate"], "bb_park_factor"),
    "hr_rate": RateComponent("hr_rate", "is_hr", None,
                             AGE_DIRECTION["hr_rate"], "hr_park_factor"),
}
DEFAULT_COMPONENT = "k_rate"


def get_component(component: str | RateComponent | None) -> RateComponent:
    """Resolve a component name to its spec. `None` is K%, the historical
    default, so every pre-BAS-73 call site keeps its meaning."""
    if component is None:
        return RATE_COMPONENTS[DEFAULT_COMPONENT]
    if isinstance(component, RateComponent):
        return component
    if component not in RATE_COMPONENTS:
        raise ValueError(
            f"unknown component {component!r}; this model is a per-PA "
            f"binomial and serves {sorted(RATE_COMPONENTS)} — BABIP (per "
            f"ball in play) and ISO (not a count of trials at all) need "
            f"their own denominators"
        )
    return RATE_COMPONENTS[component]
