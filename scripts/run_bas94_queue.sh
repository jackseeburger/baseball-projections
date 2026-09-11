#!/usr/bin/env bash
# Feed BAS-94 shards to two slots, taking the next one as a slot frees.
#
# K% shards take about twice as long as HR/PA ones (145 s a fit against 81 s),
# so running the grid season by season leaves a core pair idle for half of
# every season. `xargs -P 2` pulls the next shard the moment a slot opens
# instead, which on this grid is the difference between about six hours and
# about four. Shards are listed in the pre-registered season order
# (2024, 2025, 2022, 2026) with K% first inside each season, so the longest
# job of a season starts first and a season finishes as early as the schedule
# allows — interims stay in order.
#
#   scripts/run_bas94_queue.sh 2025:k_rate 2025:hr_rate 2022:k_rate ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf '%s\n' "$@" | xargs -P 2 -I{} bash -c \
    'set -- $(echo "{}" | tr ":" " "); "$0" "$1" "$2"' \
    "$ROOT/scripts/run_bas94_job.sh"
