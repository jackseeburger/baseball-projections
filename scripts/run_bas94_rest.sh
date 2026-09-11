#!/usr/bin/env bash
# Start the remaining BAS-94 shards once the 2024 K% shard frees its slot.
#
# The box fits two shards; 2024's K% shard is still running when the queue for
# the rest is armed, so the queue waits for it rather than oversubscribing
# four cores with six chains — which slows every fit without finishing the
# grid any sooner.
#
# The gate counts finished cells in that shard's own log rather than asking
# pgrep whether the process is alive. A pgrep gate on a command-line fragment
# matches any shell whose command line happens to contain the fragment —
# including the very loop doing the asking, which then waits for itself
# forever. This cost half an hour of an idle core the first time.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/data/eval/bas94/run/2024_k_rate/run.log"
CELLS=12
while [ "$(grep -c 'INFO bayes: ' "$LOG" 2>/dev/null || echo 0)" -lt "$CELLS" ]; do
    sleep 30
done
exec "$ROOT/scripts/run_bas94_queue.sh" \
    2025:k_rate 2022:hr_rate 2022:k_rate 2026:hr_rate 2026:k_rate
