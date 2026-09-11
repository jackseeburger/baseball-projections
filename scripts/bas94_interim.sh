#!/usr/bin/env bash
# Merge the BAS-94 shards and write the interim analysis for a season.
#
# Run after each season so a partial grid is readable: the scorer runs on
# whatever cells exist, and the interim file names the seasons it covers.
#
#   scripts/bas94_interim.sh 2024
set -euo pipefail

YEAR="${1:?usage: bas94_interim.sh <season>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/eval/bas94"

python "$ROOT/scripts/merge_bas94_checkpoints.py" \
    --run-dir "$OUT/run" --out-dir "$OUT" | tee "$OUT/merge_after_${YEAR}.txt"
python "$ROOT/scripts/analyze_bas94.py" --in-dir "$OUT" \
    --out "$OUT/analysis_bas94.json" --result-md "$OUT/RESULT.md" \
    > "$OUT/interim_after_${YEAR}.txt" 2>&1
tail -40 "$OUT/interim_after_${YEAR}.txt"
