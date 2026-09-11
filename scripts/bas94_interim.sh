#!/usr/bin/env bash
# Merge the BAS-94 shards and write the interim analysis for a season.
#
# Run after each season so a partial grid is readable: the scorer runs on
# whatever cells exist, and the interim file names the seasons it covers.
#
# Every season given is scored and nothing else. Shards are queued across
# seasons, so by the time a season finishes the next one has usually written
# half its cutoffs, and a "complete seasons plus half of one" table is not a
# thing anyone can read: the half-season's population is a different one at
# every cutoff it is missing.
#
#   scripts/bas94_interim.sh 2024            # scores 2024
#   scripts/bas94_interim.sh 2025 2024 2025  # scores both, filed under 2025
set -euo pipefail

YEAR="${1:?usage: bas94_interim.sh <label-season> [seasons...]}"
shift || true
SEASONS=("${@:-$YEAR}")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/eval/bas94"

python "$ROOT/scripts/merge_bas94_checkpoints.py" \
    --run-dir "$OUT/run" --out-dir "$OUT" | tee "$OUT/merge_after_${YEAR}.txt"
python "$ROOT/scripts/analyze_bas94.py" --in-dir "$OUT" \
    --seasons "${SEASONS[@]}" \
    --out "$OUT/analysis_bas94.json" --result-md "$OUT/RESULT.md" \
    > "$OUT/interim_after_${YEAR}.txt" 2>&1
tail -40 "$OUT/interim_after_${YEAR}.txt"
