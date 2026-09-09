# The serving rule, and why it has an effect floor

**BAS-82.** 2026-09-09. Evidence: `data/eval/serving_rules.json`, built by
`scripts/score_serving_rules.py` from the committed evidence of every
serving decision to date.

## The problem

BAS-80 found that pitcher BB/BF's covariate-only |t| is 2.41 against the
current pitcher Marcel and 2.58 against a calibrated one half a percent of
MAE away — straddling the 2.5 bar architecture.md §3 had just adopted —
and K/BF sits at 2.499 against the same bar. A serving rule that flips on
±0.1 of a t is measuring how many players were scored, not whether the
measurement is worth serving.

## Three candidates, scored on fourteen decisions

Every decision made under the covariate-share idea — contact quality's ten
components (BAS-72), stuff's four (BAS-79, on both baselines from BAS-80)
— re-scored under the rule in force and three candidates. `cov %` is the
covariate-only share as a percent of the served baseline's MAE.

| decision | served | cov % | cov t | cov t (calibrated) | current (\|t\|>2.5) | two-baseline | floor 0.75% + \|t\|>2.0 | bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contact / hitter K% | yes | −1.45 | −4.37 | — | clear | clear | clear | clear |
| contact / hitter BB% | yes | −1.46 | −2.96 | — | clear | clear | clear | clear |
| contact / hitter HR/PA | yes | −4.60 | −6.27 | — | clear | clear | clear | clear |
| contact / hitter BABIP | yes | −1.59 | −3.32 | — | clear | clear | clear | clear |
| contact / hitter ISO | yes | −5.23 | −7.09 | — | clear | clear | clear | clear |
| contact / pitcher K/BF | no | −0.23 | −0.98 | — | withhold | withhold | withhold | withhold |
| contact / pitcher BB/BF | no | −0.07 | −0.53 | — | withhold | withhold | withhold | withhold |
| contact / pitcher (BB+HBP)/BF | no | −0.05 | −0.31 | — | withhold | withhold | withhold | withhold |
| contact / pitcher HR/BF | no | −4.39 | −6.14 | — | clear | clear | clear | clear |
| contact / pitcher BABIP | no | −1.61 | −2.88 | — | clear | clear | clear | clear |
| stuff / pitcher K/BF | no | −3.09 | −3.73 | −3.73 | clear | clear | clear | clear |
| **stuff / pitcher BB/BF** | **yes** | **−0.85** | **−2.41** | **−2.58** | withhold | status quo | clear | coin flip (0.47) |
| stuff / pitcher (BB+HBP)/BF | no | −0.63 | −1.86 | −1.54 | withhold | withhold | withhold | withhold |
| stuff / pitcher HR/BF | yes | −2.20 | −3.50 | −3.50 | clear | clear | clear | clear |

(K/BF's stuff arm is withheld on the *total* gate, t 2.499; its covariate
share clears every rule. Contact quality's pitcher rows were never served
because the doc's §8 judgement was hitters first; the walk rates are
refused on the floor, not the t, which is the right refusal for the right
reason.)

- **Two-baseline margin** changes nothing: on the only rows with a second
  baseline it does the right thing without moving the answer, and on the
  other twelve it is inert because no calibrated baseline exists. It is
  also only available where a calibration ticket has been run, and BAS-80
  declined to ship its calibration, so the second baseline is a
  measurement that no longer exists in production.
- **Bootstrap** changes nothing and *cannot* on this evidence: a cluster
  bootstrap of a clustered t on the same data is asymptotically the t. Its
  0.474 for BB/BF is Φ(2.41−2.5)+Φ(−2.41−2.5) to three digits. A seeded
  random number generator inside a serving gate, for a restatement of a
  number already in the JSON, is the worst trade of the three.
- **Effect floor + looser t** is the only candidate that addresses the
  complaint: a bare significance threshold is a function of `n_clusters`,
  so with 955 pitchers it will eventually certify a share worth a twentieth
  of a percent while a useful share measured on 300 players is refused.
  Its honest cost is that the floor is a number someone chose, and the
  candidate as drawn (0.75%) passed BB/BF by 0.10 of a point — the same
  knife edge one layer down.

## The rule adopted

**Covariate-only share ≥ 1.0% of the served baseline's MAE, at |t| > 2.0.**

The floor is not fitted to any component in the table. It is the gain
that tuning Marcel's own constants was worth on the hitter side (1.1% of
MAE, contact-quality.md §4), rounded down: a measurement that adds a
nightly build, a sidecar, a fallback path and a provenance field has to be
worth at least what re-tuning the baseline's constants was worth. Under
it, on this table, stuff's BB/BF (0.85%) would be withheld and everything
else is unchanged — which is stated here so that nobody can later say the
floor was chosen to keep BB/BF served. It was not chosen to un-serve it
either: BAS-79 stands under the rule in force when it was made, and this
rule applies from the next serving decision (BAS-76's, first).

## Two limits of the evidence

1. The committed JSONs carry pooled summaries, not per-cell differences,
   so the bootstrap column was computed on a seeded cluster panel rescaled
   to reproduce each row's `diff` and `se` — which makes it a restatement
   of the t by construction. `scripts/score_serving_rules.py` takes a real
   per-cell table when one is available.
2. Contact quality committed no `contact_additive_recal` arm, so the
   covariate share of the *served* additive shape was never measured for
   BAS-72; the free fit's `contact` vs `contact_recal` is what the table
   scores. BAS-81's swing decisions were scored after this table was built
   (K% share t −2.96 at 66% of a 1.57% gain ≈ 1.0%; BB% t −2.10 at 0.77%)
   and would be withheld on BB% and at the edge on K% — consistent with
   its own verdict that nothing ships.
