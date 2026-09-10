# Park factors for the hierarchical models

**BAS-86.** Station A. Pre-registered 2026-09-10 08:15 UTC in the Linear
issue, before any run; this file mirrors that text.

## Why

`src/models/pa_rate.py` (and now `pa_joint.py`) loads
`data/parquet/park_factors.parquet` as a logit offset and falls back to a
neutral park with a log line when it is absent. The file has never been
committed, uploaded to R2, or built by any script: every Bayes arm ever
scored (BAS-69, BAS-73, BAS-83, BAS-84) ran park-neutral. Marcel does not
use park factors either, so the comparisons were fair, but it is missing
structure teams have, and HR/PA, the component every model has struggled
to beat a ballast on, is the one where park matters most.

## What gets built

`scripts/build_park_factors.py` → `data/features/park_factors.parquet`
(team, game_year, component, factor): per-park, per-season multiplicative
factors for K, BB, HR, BABIP and ISO from `pa_outcomes`, home/away paired
(each club's rate at home against the same club's rate away, and visitors'
rate in the park against their rate elsewhere, averaged), regressed toward
1 with a three-season window and a pseudo-count ballast chosen once on
2015–2019 by leave-one-season-out persistence and frozen. Written as of
each season strictly before it is used (walk-forward: the 2024 factor uses
2021–2023). `pa_rate.load_park_factors` reads it unchanged (the
`k_park_factor` column is kept for compatibility; other components are
added). A `marcel_tuned_park` arm on the dense harness: tuned Marcel with
the batter's home-park offset applied to the projected rate, the cheapest
test of whether the factor carries information.

## Pre-registered predictions

1. **Persistence:** year-over-year correlation of the regressed HR factor
   across parks ≥ 0.6; of the K factor ≥ 0.4. Below that the factor is
   noise at this reduction (vacuity: stop).
2. **HR/PA:** `marcel_tuned_park` beats `marcel_tuned` on the dense
   harness (2022/2024/2025/2026 × 12 cutoffs, common set, SE clustered by
   player) by ≥ 0.5% of MAE, |t| > 2.5.
3. **K% and BB%:** |Δ| < 0.3% of MAE (parks barely move them).
4. **Under §3's effect floor** (1.0% at |t| > 2.0) prediction 2's gain
   does not clear serving on its own; the factor's value is as the offset
   the Bayes arms already expect, to be re-scored in BAS-85's follow-up.

### Vacuity check

The sd of the regressed log HR factor across the 30 parks in a season must
be ≥ 0.05; below that there is no park signal to test.

## What ships

Nothing in this pass. Serving is its own ticket under architecture.md §3.
