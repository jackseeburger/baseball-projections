#!/usr/bin/env bash
# One BAS-94 season: the two prior-mean arms on K% and HR/PA at the twelve
# biweekly cutoffs, one process per component.
#
# Two processes and not one because NumPyro's JIT never reclaims its
# compilations and a process dies at around 70 of them (the dense sweep's own
# module docstring); two and not four because each fit runs two chains and
# this box has four cores. Each process gets its own --out-dir, since two
# processes sharing a checkpoint path each write back only the cells they
# loaded at startup.
#
#   scripts/run_bas94_season.sh 2024
set -euo pipefail

YEAR="${1:?usage: run_bas94_season.sh <season>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/data/eval/bas94/run"
VARIANTS="ability_walk+prior_contact,ability_walk+prior_contact_cur"

mkdir -p "$RUN_DIR"
pids=()
for comp in hr_rate k_rate; do
    out="$RUN_DIR/${YEAR}_${comp}"
    mkdir -p "$out"
    python "$ROOT/scripts/run_intraseason_backtest_dense.py" \
        --stage bayes \
        --bayes-seasons "$YEAR" \
        --bayes-components "$comp" \
        --variants "$VARIANTS" \
        --bayes-sampler numpyro \
        --bayes-draws 500 --bayes-tune 500 --bayes-chains 2 \
        --out-dir "$out" \
        >"$out/run.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
