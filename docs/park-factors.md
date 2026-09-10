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

## Results (2026-09-10, run after the pre-registration above)

**The factors are real and persistent; applied to tuned Marcel they make
every component worse, because the park is already in Marcel.** Nothing
is served. Evidence: `data/eval/park_factors_stage1.json`; the artifact is
`data/features/park_factors.parquet` with its sidecar
`park_factors.meta.json` (the frozen ballasts, the persistence tables, the
known park changes).

### The artifact

330 rows (30 clubs × 2016–2026), wide: `k_park_factor`, `bb_park_factor`,
`hr_park_factor`, `babip_park_factor`, `iso_park_factor`, `n_pa`.
Home/away paired on both halves (host club here vs elsewhere; visitors here
vs those same visitors elsewhere, weighted), averaged, regressed toward 1
with a pseudo-count ballast in the component's own trials, renormalised to
a trials-weighted mean of 1 per season, windows Y−3..Y−1 (2020 included,
unweighted, so the 2021–2023 stamps are ~20% more regressed). Ballasts
chosen once on 2015–2019 by leave-one-season-out RMSE of the log factor
and frozen: K 2,000, BB 8,000, HR 3,000, BABIP 3,000, ISO 4,000 trials.
No park term at all is 30–50% worse in RMSE than the chosen ballast on
every component.

### Vacuity: PASS

sd of the regressed log HR factor across the 30 parks: 0.121 (2022),
0.114 (2024), 0.110 (2025), 0.098 (2026), all above 0.05.

### Scoring the predictions

1. **Persistence — HOLDS.** Year-over-year correlation of the stamped
   regressed factor, pooled 2017–2026: HR 0.92 (bar 0.60), K 0.95 (bar
   0.40); ISO 0.91, BABIP 0.93, BB 0.86. The caveat belongs next to it:
   consecutive stamped factors share two of three window seasons, so this
   is persistence of the *artifact*. On single-season raw splits with no
   overlap the same correlation is HR 0.44, K 0.70, BABIP 0.62, ISO 0.38,
   BB 0.29, and HR would miss its bar. The prediction was written about
   the regressed factor and holds as written; both numbers are recorded.
2. **HR/PA — FAILED, and the arm is worse, not short.** `marcel_tuned_park`
   vs `marcel_tuned`: +3.36% of MAE, t(player) 4.39, cells 0–48, against
   a bar of ≥ 0.5% better at |t| > 2.5.
3. **K% and BB% — FAILED.** K% +6.83% (t 5.21, 0–48), BB% +1.43%
   (t 3.04, 7–41), against |Δ| < 0.3%.
4. **Effect floor — holds vacuously.** There is no gain for the floor to
   judge.

Pooled (arm − base; positive is worse; n = 14,294; SE clustered by player):

| component | Δ MAE | % of base | t (player) | t (cell) | W–L |
| --- | --- | --- | --- | --- | --- |
| K% | +0.00195 | +6.83% | 5.21 | 21.25 | 0–48 |
| HR/PA | +0.00034 | +3.36% | 4.39 | 16.22 | 0–48 |
| ISO | +0.00110 | +2.99% | 4.12 | 11.32 | 0–48 |
| BABIP | +0.00061 | +2.20% | 2.45 | 6.93 | 8–40 |
| BB% | +0.00026 | +1.43% | 3.04 | 10.73 | 7–41 |

Every season, every component, the same sign. A half-strength diagnostic
arm (√factor, not pre-registered) loses by about half as much.

### Why it loses, measured

The factor each batter received is recoverable from the cells, so the
realised rate, Marcel's projection and the arm's projection were each
regressed on the log factor (trials-weighted, cell-demeaned):

| component | realised | marcel_tuned | arm | arm − realised |
| --- | --- | --- | --- | --- |
| HR/PA | +0.21 | +0.24 | +1.22 | +1.01 |
| BABIP | +0.28 | +0.11 | +1.13 | +0.85 |
| ISO | +0.01 | +0.11 | +1.12 | +1.11 |
| K% | −0.03 | −0.15 | +0.84 | +0.87 |
| BB% | −0.52 | −0.42 | +0.60 | +1.12 |

`marcel_tuned` already tilts with the park about as much as the outcome
does: its inputs are rates the hitter accrued in that same park mix, and
the rest-of-season sample is that mix again. The arm adds a whole park
effect where roughly none was missing; the excess is ≈ 1.0 on every
component, the shape of a double count, and the half-strength arm losing
by half is the same fact seen twice. K% is worst because the realised tilt
there is ≈ 0: a K% park factor has essentially no relationship with a
hitter's rest-of-season K rate, so any correction is noise.

The factors are not thereby useless; the pre-registration applied them in
the wrong place. Where an offset like this can pay is a model whose
ability term is park-free, which is what the hierarchical models' logit
offset is for, or a batter who changes parks between his training window
and the projected season. Neither is tested here.

### What changes now

Committing the artifact where `load_park_factors` looks means the
hierarchical arms (`pa_rate`, `pa_joint`, `bayes_arm`) pick up
non-neutral, per-component, walk-forward park offsets for the first time.
The loader logs the path and columns it finds. Bayes-arm numbers produced
from a checkout containing this commit are not comparable cell-for-cell
to the published ones (BAS-69/73/83/84, all park-neutral), and the BAS-85
grid now running predates it and stays park-neutral. Whether the offset
helps the Bayes arms is BAS-85's follow-up question, pre-registered there;
the graph also carries a free `park_effect[team]` alongside the fixed
offset, and given the tilt above those two are close to competing for the
same variance. Neither was changed here.

### Things that complicate the reading

- **Clubs that changed parks.** The key is the hosting club (the PA
  parquet has no venue id), so a club carries its old park for up to three
  seasons: ATH (Coliseum through 2024, Sacramento from 2025) is stamped
  0.83 for HR in 2025; TB (Tropicana, then Steinbrenner Field in 2025,
  Tropicana again in 2026) is stamped wrong in both directions. Recorded in
  the sidecar. Two parks of thirty cannot produce a t of 4–5. Neutral-site
  games are charged to the nominal home club.
- **Humidors and fences are what a three-season window smears.** AZ's HR
  factor takes three stamped seasons to arrive at its post-humidor level;
  BAL runs 1.28 → 0.91 → 1.03 across its wall changes.
- **2026 is partial** and its common set is smaller (n ≈ 257 vs ≈ 317)
  because a pre-existing preseason Bayes arm joins those cells.
- **The estimator is mildly self-referential**: every rival's "elsewhere"
  pool contains the extreme park, so an extreme park drags its own
  baseline toward itself; proportionally small with 30 clubs, bracketed
  in the test fixture.
