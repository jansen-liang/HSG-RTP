#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_ROOT/logs/grid_after_ablations.log"
COMPLETION_MARKER="$REPO_ROOT/evaluation_results/component_ablations_complete_20260809"

mkdir -p "$REPO_ROOT/logs"
while [[ ! -f "$COMPLETION_MARKER" ]]; do
  if ! tmux has-session -t component_ablation_queue 2>/dev/null; then
    printf '[%s] component queue exited before completion marker\n' "$(date '+%F %T')" >> "$LOG_FILE"
    exit 1
  fi
  printf '[%s] waiting for component_ablation_queue\n' "$(date '+%F %T')" >> "$LOG_FILE"
  sleep 60
done

printf '[%s] starting GRID author-code adaptation\n' "$(date '+%F %T')" >> "$LOG_FILE"
bash "$REPO_ROOT/scripts/run_grid_author_adaptation.sh" >> "$LOG_FILE" 2>&1
printf '[%s] GRID author-code adaptation complete\n' "$(date '+%F %T')" >> "$LOG_FILE"
