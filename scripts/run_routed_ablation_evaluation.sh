#!/usr/bin/env bash

set -euo pipefail

CHECKPOINT_ENV="${1:-}"
ABLATION="${2:-}"
OUTPUT_DIR="${3:-}"
LIMIT="${4:-}"
if [[ -z "$CHECKPOINT_ENV" || -z "$ABLATION" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 CHECKPOINT_ENV ABLATION OUTPUT_DIR [LIMIT]" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$CHECKPOINT_ENV" ]]; then
    echo "Checkpoint environment file not found: $CHECKPOINT_ENV" >&2
    exit 1
fi

set -a
source "$CHECKPOINT_ENV"
set +a

GLOBAL_CHECKPOINT="${GLOBAL_CHECKPOINT:-${NO_HSG_GLOBAL_CHECKPOINT:-}}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${NO_HSG_LOCAL_CHECKPOINT:-}}"
if [[ -z "$GLOBAL_CHECKPOINT" || -z "$LOCAL_CHECKPOINT" ]]; then
    echo "Checkpoint environment must define global and local checkpoints" >&2
    exit 1
fi

PYTHON="${HSG_RTP_PYTHON:-/home/swzz/anaconda3/gra/bin/python}"
MODEL_PATH="${HSG_RTP_MODEL_PATH:-/home/swzz/data/HLR/models/Qwen3-8B}"
DATASET="${HSG_RTP_EVAL_DATA:-pipeline/output/task_split/test_corrected_streaming.jsonl}"
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

exec "$PYTHON" scripts/evaluate_routed_checkpoints.py \
    --global-checkpoint "$GLOBAL_CHECKPOINT" \
    --local-checkpoint "$LOCAL_CHECKPOINT" \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --ablation "$ABLATION" \
    --lightweight-controller \
    "${LIMIT_ARGS[@]}"
