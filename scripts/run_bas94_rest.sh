#!/usr/bin/env bash
# Start the remaining BAS-94 shards once the 2024 K% shard frees its slot.
#
# The box fits two shards; 2024's K% shard is still running when the queue for
# the rest is armed, so the queue waits for it rather than oversubscribing
# four cores with six chains — which slows every fit without finishing the
# grid any sooner.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while pgrep -f "bayes-seasons 2024" >/dev/null 2>&1; do sleep 30; done
exec "$ROOT/scripts/run_bas94_queue.sh" \
    2025:k_rate 2022:hr_rate 2022:k_rate 2026:hr_rate 2026:k_rate
