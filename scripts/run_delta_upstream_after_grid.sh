#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GRID_SUMMARY="${HSG_RTP_GRID_SUMMARY:-evaluation_results/grid_author_retrained_direct70/summary.json}"
LOG_FILE="logs/delta_upstream_after_grid.log"
PYTHON="${HSG_RTP_PYTHON:-/home/swzz/anaconda3/gra/bin/python}"
MODEL_PATH="${HSG_RTP_MODEL_PATH:-/home/swzz/data/HLR/models/Qwen3-8B}"
DATASET="${HSG_RTP_EVAL_DATA:-pipeline/output/task_split/test_corrected_streaming.jsonl}"

mkdir -p logs
while [[ ! -f "$GRID_SUMMARY" ]]; do
    if ! tmux has-session -t grid_after_ablations 2>/dev/null; then
        printf '[%s] GRID queue exited before producing %s\n' \
            "$(date '+%F %T')" "$GRID_SUMMARY" >> "$LOG_FILE"
        exit 1
    fi
    printf '[%s] waiting for GRID result\n' "$(date '+%F %T')" >> "$LOG_FILE"
    sleep 60
done

printf '[%s] starting upstream DELTA smoke evaluation\n' "$(date '+%F %T')" >> "$LOG_FILE"
"$PYTHON" scripts/evaluate_external_baseline.py \
    --method delta_upstream \
    --model-path "$MODEL_PATH" \
    --model-name Qwen3-8B \
    --dataset "$DATASET" \
    --output-dir evaluation_results/delta_upstream_partial_smoke3_20260809 \
    --max-input-tokens 8192 \
    --max-new-tokens 2048 \
    --limit 3 >> "$LOG_FILE" 2>&1

printf '[%s] starting upstream DELTA full evaluation\n' "$(date '+%F %T')" >> "$LOG_FILE"
"$PYTHON" scripts/evaluate_external_baseline.py \
    --method delta_upstream \
    --model-path "$MODEL_PATH" \
    --model-name Qwen3-8B \
    --dataset "$DATASET" \
    --output-dir evaluation_results/delta_upstream_partial_final70_20260809 \
    --max-input-tokens 8192 \
    --max-new-tokens 2048 \
    >> "$LOG_FILE" 2>&1
printf '[%s] upstream DELTA partial adaptation complete\n' "$(date '+%F %T')" >> "$LOG_FILE"
