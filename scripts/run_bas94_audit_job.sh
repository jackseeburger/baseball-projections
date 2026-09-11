#!/usr/bin/env bash
# One sampler-audit shard: `bayes_walk` alone, one season, one component,
# refit under numpyro on the same cells the prior-mean arms ran on.
#
# BAS-94 borrows `bayes_walk` from the BAS-85 grid, which drew it under pymc.
# These shards say what that borrowing costs, if anything — see
# scripts/bas94_sampler_audit.py. Written to a directory of their own so the
# refit never reaches the headline checkpoint: the published comparator is the
# pymc one, and the audit is a check on it, not a replacement for it.
#
#   scripts/run_bas94_audit_job.sh 2024 hr_rate
set -euo pipefail

YEAR="${1:?usage: run_bas94_audit_job.sh <season> <component>}"
COMP="${2:?usage: run_bas94_audit_job.sh <season> <component>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/eval/bas94/sampler_audit/${YEAR}_${COMP}"

mkdir -p "$OUT"
exec python "$ROOT/scripts/run_intraseason_backtest_dense.py" \
    --stage bayes \
    --bayes-seasons "$YEAR" \
    --bayes-components "$COMP" \
    --variants ability_walk \
    --bayes-sampler numpyro \
    --bayes-draws 500 --bayes-tune 500 --bayes-chains 2 \
    --out-dir "$OUT" \
    >>"$OUT/run.log" 2>&1
