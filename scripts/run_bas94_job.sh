#!/usr/bin/env bash
# One BAS-94 shard: both prior-mean arms, one season, one component, the
# twelve biweekly cutoffs. A shard is one process because NumPyro's JIT never
# reclaims its compilations and a process dies at around 70 of them; 24 fits
# is comfortably under that and four seasons in one process would not be.
#
# Two shards at a time is what this box fits: each fit runs two chains and
# NumPyro pins two cores per process on four cores. K% shards take about twice
# as long as HR/PA ones, so the two queues are fed shard by shard as slots
# free rather than season by season, which is the difference between four
# hours and six.
#
#   scripts/run_bas94_job.sh 2024 hr_rate
set -euo pipefail

YEAR="${1:?usage: run_bas94_job.sh <season> <component>}"
COMP="${2:?usage: run_bas94_job.sh <season> <component>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/eval/bas94/run/${YEAR}_${COMP}"
VARIANTS="ability_walk+prior_contact,ability_walk+prior_contact_cur"

mkdir -p "$OUT"
exec python "$ROOT/scripts/run_intraseason_backtest_dense.py" \
    --stage bayes \
    --bayes-seasons "$YEAR" \
    --bayes-components "$COMP" \
    --variants "$VARIANTS" \
    --bayes-sampler numpyro \
    --bayes-draws 500 --bayes-tune 500 --bayes-chains 2 \
    --out-dir "$OUT" \
    >>"$OUT/run.log" 2>&1
