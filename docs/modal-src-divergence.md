# The Modal/src divergence — full audit (issue #86)

**Run:** Sept 8, 2026. **What this answers:** exactly how
`modal_functions/app.py`'s four training functions differed from their
`src/` counterparts before this issue, which published numbers came from
which copy, whether the "no cross-module imports" comment was ever true,
and what changed here versus what was deliberately left alone.

**What this does not answer:** whether the HSGP age curve is better than
the quadratic one, or whether the opposing-pitcher term should ship. Those
are modelling questions with their own gate (architecture.md #3) — see
[Not in scope](#not-in-scope).

## The premise: was "no cross-module imports" ever true?

No. `modal.Image.add_local_python_source(module_name)` ships a local
Python package into a function's container image — checked against
Modal's own docs (not assumed), and it predates the `modal>=0.64` this
repo pins by a long way. `add_local_dir` does the equivalent for a plain
directory. The comment in `modal_functions/app.py` — "All model code is
inlined (Modal requirement — no cross-module imports)" — described a
constraint that does not exist. It cost roughly 3,200 lines of duplicated
model code and, per the rest of this document, a production model that
quietly stopped matching the one every published number was scored
against.

## train_pa_k_rate vs src/models/pa_k_rate.py — the gated model

This is the pair the issue was filed about, and the audit found more
daylight between them than the two differences already known when it was
filed.

| | `src/models/pa_k_rate.py` (gated) | `modal_functions/app.py::train_pa_k_rate` (production, before this PR) |
|---|---|---|
| Age curve | Quadratic in centered age: `beta_age * age_c + beta_age2 * age_c**2` | HSGP (Hilbert Space GP), Matern-5/2 kernel, 20 basis functions |
| Opposing pitcher | Optional zero-mean random effect (`include_pitcher=True`, BAS-59) | Absent — no `sigma_pitcher`, `z_pitcher`, or `pitcher_idx` anywhere in the file |
| Age computation | `src.data.birthdates.seasonal_age` — age as of **June 30**, using the Chadwick register's birth year *and* month/day | `game_year - birth_year` — a **calendar-year** age, and the birth year source (PA row, or a `hitter_seasons.parquet` median, or `first_year-24`) never carried month/day even when it came from the register, so June-30 precision was never reachable |
| Cell aggregation | Exact rewrite of the per-PA Binomial: cells keyed on `(batter, season, team, stand[, pitcher])`, so a traded player or a switch-hitter's at-bats against each hand/team get their own cell and their own park factor | Aggregated all the way to **(batter, season)**: team is the *modal* team that season and stand is whichever value came first in the data for that player-year, so a traded player's other team's park factor and a switch-hitter's off-side at-bats are silently dropped. `log_pf_k` is the *mean* log park factor across the season rather than an exact per-cell value. The docstring's claim ("Mathematically equivalent to per-PA Bernoulli") is true of `src/models/aggregation.py`'s cell key, not of this one. |
| Batter filter | One filter: career PA ≥ `min_pa` | Two filters: drop anyone whose **best single season** is under 80 PA (an undocumented pitcher heuristic), *then* career PA ≥ `min_pa` — a different batter universe than the one the harness scores |
| Dated cutoff (BAS-59) | `cutoff_date` — PA strictly before the cutoff train the model, the rest are withheld, with a leakage assertion (`src/models/cutoff.py`) | No cutoff mechanism at all; always trains on everything in `pa_dir` |
| `target_accept` default | 0.9 (`SAMPLER_KWARGS`) | 0.95 — the workflow never overrides it, so this was a live, if minor, default divergence |
| Sampler backend | NumPyro NUTS | NumPyro NUTS (same) |
| `min_pa` default | 50 | 50 (same) |
| Projections | Single projection year (+ a population-level draw for batters the fit never saw) | Five projection years, extrapolating the league random walk further each year, via linear interpolation of the HSGP age effect across training ages |

Two of these (age curve, pitcher effect) were already known from the
issue. The rest — the aggregation granularity, the pitcher-filter
heuristic, the age computation, the missing cutoff support, and the
`target_accept` default — were not written down anywhere before this
audit.

## train_iso_model / train_babip_model vs src/ — no counterpart exists

The issue asked to find the `src/` counterpart for these two "if nothing
does, say so — that is itself a finding." Nothing does. `src/models/`
contained exactly one model file, `pa_k_rate.py`, before this PR. There is
no ISO or BABIP model anywhere in `src/`, no harness provider for either
in `src/eval/`, and no backtest comparing either to Marcel the way
`src/eval/bayes_arm.py` does for K% (BAS-59). These two components have
never been gated, in either copy.

That makes "make Modal import `src/`" a different operation for these two
than for K%: there was nothing to route to. What this PR did instead —
and why — is in [What changed](#what-changed).

## assemble_projections — arithmetic, and a dependency on files nothing produces

`assemble_projections` is not a PyMC model: it is pure pandas/numpy
arithmetic (component rates → AVG/OBP/SLG → wOBA → wRC+ → oWAR) over
already-computed projection parquets, and it was never in scope for the
Phase 3 guard test for that reason. Two findings about it:

1. **No `src/` counterpart.** The issue's pointer to
   `src/projections/` and `src/models/aggregation.py` doesn't land:
   `aggregation.py` is the Binomial *cell* aggregation the K% model uses
   (batter-season Binomial counts), unrelated to assembling five rate
   components into a batting line. The closest kin are
   `src/models/marcel.py`'s own wOBA/wRC+ derivation (different constants
   — `LG_WOBA` .315 vs. this function's .310, `LG_R_PA` .120 vs. .116) and
   `scripts/build_career_war.py`, which is the module that actually owns
   career WAR uncertainty bands today and is off-limits for this PR (owned
   by another agent). There is no single "the" assembly module to import.
2. **It depends on files nothing in this repository builds.** It reads
   `{k_rate,bb_rate,hr_rate,iso,babip}_projections_{year}.parquet` from
   `/models/projections`. Only three of those five have a producer
   anywhere in `modal_functions/app.py`, on this branch, on
   `origin/feature/assembly`, or in git history at all —
   `git log --all -S"train_bb_rate_model"` and the equivalent for
   `train_hr_rate_model` return nothing. `bb_rate` and `hr_rate`
   projections exist today only as the static files committed 2026-04-10
   (see below); nothing regenerates them. Running the `assembly` or `all`
   `workflow_dispatch` component against a fresh Modal volume will fail at
   that read.

This PR corrected the false "Modal requirement" framing wherever it
appeared and documented the missing-producer risk directly in the
function's docstring, but did not build `train_bb_rate_model` /
`train_hr_rate_model` or invent a new assembly module — both are new
modelling and infrastructure work with their own gate, not a divergence
to close (see [Not in scope](#not-in-scope)).

## Which published artifacts came from which copy

This is the part of the audit that matters most, because it is the part a
reader has no way to check without doing exactly this archaeology.

### `data/projections/*_projections_2026.parquet` — the `bayes_preseason` file

Five files (`k_rate`, `bb_rate`, `hr_rate`, `iso`, `babip`) plus five
matching `*_aging_curve_2026.parquet` files were committed 2026-04-10 in
`af8b179` ("feat: career WAR projections with Bayesian uncertainty bands"
— "Merge assembly branch (all 5 component models)"). This is the file
`docs/accuracy-2026.md` §1 calls "Bayes (ours)" ("public systems as
captured Apr 9; ours generated Apr 10"), the file
`docs/ros-projections.md` and `docs/backtest-baselines.md`'s BAS-59
section call `bayes_preseason`, and the file every "our Bayesian
components lose to Marcel" comparison on the site's accuracy page has
been reading since.

**It came from the Modal copies, not from `src/`.** Two independent lines
of evidence:

- **Schema.** `k_rate_projections_2026.parquet` has columns
  `projection_year`, `projected_age`, `k_rate_10`, `k_rate_90`, and a
  companion `k_rate_aging_curve_2026.parquet` with 100 rows of
  `age_effect_mean/lower/upper` on an age grid from 20 to 42.
  `src/models/pa_k_rate.py::generate_projections` produces none of that —
  no `projection_year` column (it projects one year), no `_10`/`_90`
  percentiles, and nothing to tabulate an aging curve from, because its
  age term is a two-coefficient quadratic. The committed schema matches
  Modal's `train_pa_k_rate` output field-for-field (percentiles, the
  `age_effect_*` HSGP table) and matches nothing `src/` has ever produced.
- **Git history.** `train_pa_k_rate`, `train_iso_model`, and
  `train_babip_model`'s bodies in `modal_functions/app.py` were last
  touched 2026-04-05/04-06 (`19d3527`, `ee31c53`, `54af327`) and were
  byte-for-byte unchanged through Sept 2026 — the only commits to touch
  this file between then and this PR (`73b9968`, `5073ea5`) modified
  `generate_birth_years_on_volume` and added the refit workflow, not the
  model bodies. The April 10 commit's date and content line up exactly
  with those unmodified copies.

**The BB% and HR% halves have no recoverable code.** `assemble_projections`
expects `bb_rate` and `hr_rate` projection files (see above), and the
April 10 commit includes them with the same Modal-shaped schema, but no
`train_bb_rate_model` / `train_hr_rate_model` function exists anywhere in
this repository's git history, including the unmerged
`origin/feature/assembly` branch the April 10 commit message refers to.
Whatever produced those two files was run once, its output was committed,
and its code was never checked in. There is no way to regenerate or audit
it from this repository alone.

**The existing docs already hedge this correctly, mostly.**
`docs/backtest-baselines.md`'s BAS-59 section says outright that
`bayes_preseason` is "the legacy April 10 projection file — a different
code path, no opposing-pitcher term, fit under the old `cutoff_year`
semantics," and refuses to use it as a clean information-content control
for exactly that reason. `docs/accuracy-2026.md` §1 independently names
"HSGP aging curves" and "No pitcher effect" as the likely reasons the
Bayesian row trails — both true of the Modal copy specifically. Neither
doc claims `bayes_preseason` and the BAS-59 `bayes` arm (below) are the
same model; I did not find a place that needed correcting beyond adding
this document as the citable source for *which* code path produced it.
Nothing here regenerates or alters that file — it is a historical
artifact of a run that already shipped, and Phase 1's instruction not to
quietly "fix" something that already produced a published result applies
to it directly.

### The BAS-59 "fair fight" and every other walk-forward K% number

`src/eval/bayes_arm.py` imports `src.models.pa_k_rate` directly — `load_pa_data`,
`prepare_model_data`, `build_model`, `sample_model`, `generate_projections`.
There is no path from any backtest or accuracy script into
`modal_functions/app.py`. Every number in the BAS-59 "fair fight" table
(`bayes` refit at the cutoff, `bayes_withheld`, the opposing-pitcher LOO/
gate result), the whole of `docs/backtest-baselines.md`'s
"fair-fight" section, and `docs/accuracy-2026.md`'s later BAS-59
addendum, came from `src/models/pa_k_rate.py`. **None of it is affected by
anything in this PR** — the model code those numbers were scored against
is exactly the model code this PR now also runs on Modal going forward.

## What changed

**train_pa_k_rate** now imports and calls `src.models.pa_k_rate` —
`load_pa_data → prepare_model_data → build_model → sample_model →
generate_projections → log_to_wandb`. Concretely, this makes the next
Monday refit (and any future `workflow_dispatch` run) use: the quadratic
age curve, the exact `(batter, season, team, stand[, pitcher])` cell
aggregation, the career-PA-only batter filter, `target_accept=0.9`, and —
when `/data/parquet/birthdates.parquet` exists on the volume — real
June-30 seasonal ages from the Chadwick register instead of
`first_year-23`. `include_pitcher` and `cutoff_date` are exposed as new
optional parameters, both defaulting to what the weekly refit has always
done (no pitcher term, no cutoff), so this rewiring does not silently
change Monday's default output; turning `include_pitcher=True` on is now
possible for the "score it at all three cutoffs" follow-up
`docs/backtest-baselines.md` already asked for.

Because `src.models.pa_k_rate.prepare_model_data` reads a `birthdates`
DataFrame with the full Chadwick schema (year *and* month/day) rather
than the calendar-year `birth_year` the PA rows already carry, a new
function, `generate_birthdates_on_volume`, writes that file
(`/data/parquet/birthdates.parquet`) via `src.data.birthdates.fetch_register`.
This needs to run once against the live volume before a refit gets real
seasonal ages — see [What's still unvalidated](#whats-still-unvalidated).
The existing `generate_birth_years_on_volume` (year-only, calendar-age)
was left in place rather than removed, since nothing in this audit found
what else might read `batter_birth_years.parquet`.

**train_iso_model / train_babip_model** were extracted verbatim into new
modules, `src/models/iso_rate.py` and `src/models/babip_rate.py` (sharing
HSGP-age-effect and calendar-age helpers via a new `src/models/rate_hsgp.py`),
with no change to priors, likelihood, the HSGP age curve, data preparation,
or aggregation. `modal_functions/app.py` now imports and calls them. This
is not "fixing a difference that produced a published result" — there was
only ever one copy of these two models, so there was nothing to diverge
from; the extraction gives Modal something to `import` instead of a
second inlined copy, which is what the Phase 3 guard test requires, and
gives these two components a `src/` home for the first time.

**assemble_projections** — see above: the false "Modal requirement"
framing was corrected in the module docstring and this function's
docstring now says plainly what it depends on that nothing produces. No
model code changed because there was none to change.

**Not changed:** the age-curve functional form question, whether the
opposing-pitcher term ships, the April 10 `bayes_preseason` file itself,
`scripts/build_career_war.py`, and anything in `src/eval/bayes_arm.py`.

## Not in scope

Per the issue: whether the HSGP age term or the quadratic curve is the
better model, and whether the opposing-pitcher term should ship, are
modelling questions with their own gate (architecture.md #3). This audit
and the wiring change only establish that the production refit now runs
the same code the harness gates — not that the harness has decided in
either curve's favor. Building `train_bb_rate_model` / `train_hr_rate_model`,
or a canonical assembly module, is new work with the same property: out
of scope here, flagged above for whoever picks it up next.

## What's still unvalidated

This sandbox cannot reach Modal — its client speaks gRPC and the session
proxy does not pass it — so nothing here was run *on Modal*. What was
validated without it:

- `python -c "import ast; ast.parse(open('modal_functions/app.py').read())"` — passes.
- Importing `modal_functions.app` as a module (constructs `modal.App`,
  `modal.Volume`, `modal.Secret`, and the `pymc_image` build steps,
  including `.add_local_python_source("src")`) — succeeds locally with no
  credentials, confirming the image and app definitions are well-formed.
- The Phase 3 guard test
  (`tests/test_models/test_modal_no_inline_model.py`) — fails against the
  pre-PR file (proving it is a real check) and passes against the current one.
- **The actual model logic, end to end, with real PyMC sampling**: each of
  `train_pa_k_rate.get_raw_f()`, `train_iso_model.get_raw_f()`, and
  `train_babip_model.get_raw_f()` was called directly (Modal's own
  mechanism for invoking the undecorated function locally — no `.remote()`,
  no gRPC, no Modal backend involved) against synthetic PA data shaped
  like the real `pa_outcomes` schema, with `data_volume`/`models_volume`
  stubbed to no-ops. All three loaded data, filtered, aggregated, built
  the model, sampled with NumPyro, generated projections, and (for K%)
  logged the summary — including the pitcher-effect path
  (`include_pitcher=True`) and the missing-birthdates fallback warning,
  which fired exactly as written. `full pytest -q` is green.

What this cannot validate: the real R2/Modal-volume data — schema drift,
actual PA volume, or `pa_outcomes` columns this sandbox has no access to;
wandb logging against the real `wandb-baseball` secret; sampling time and
memory at the real scale (8 GB / 4 CPU, ~1.9M PA, full `n_draws=2000`);
and, above all, whether `generate_birthdates_on_volume`'s network fetch of
the Chadwick register actually succeeds from inside a Modal container.

**The exact `workflow_dispatch` invocation that would validate the rest:**

```
# 1. One-time, before the first refit under this PR (writes birthdates.parquet
#    with real month/day so ages are seasonal, not calendar-year):
modal run modal_functions/app.py::generate_birthdates_on_volume

# 2. component=smoke_test, fast=true — cheapest possible check that the
#    image builds (including add_local_python_source("src")) and the
#    volumes mount on Modal's actual infrastructure:
gh workflow run modal-refit.yml -f component=smoke_test -f fast=true

# 3. component=k_rate, fast=true — the rewired function, tiny sample,
#    should complete in a couple of minutes and log a wandb run tagged
#    "k-rate", "bayesian", "pa-level" (src.models.pa_k_rate's tags, not
#    the old "hsgp"/"binomial" ones):
gh workflow run modal-refit.yml -f component=k_rate -f fast=true

# 4. component=iso and component=babip, fast=true — the extracted modules:
gh workflow run modal-refit.yml -f component=iso -f fast=true
gh workflow run modal-refit.yml -f component=babip -f fast=true

# 5. Only once 2-4 are green: a full-scale run (fast=false, the workflow's
#    defaults — draws=2000, tune=1500, chains=4) to confirm the 8GB/4CPU
#    budget still holds at the real ~1.9M-PA scale and the sampling time
#    is in the same ballpark the old inlined copy ran at.
gh workflow run modal-refit.yml -f component=k_rate
```

I did not run any of these — they need `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`,
which live only in this repo's GitHub Actions secrets.
