#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-full}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${HSG_RTP_ENV_DIR:-/home/swzz/anaconda3/gra}"
DEEPSPEED="$ENV_DIR/bin/deepspeed"
MODEL_PATH="${HSG_RTP_MODEL_PATH:-/home/swzz/data/HLR/models/Qwen3-8B}"
RUN_ROOT="${HSG_RTP_NO_HSG_RUN_ROOT:-checkpoints/no_hsg_protocol_matched}"
LOG_ROOT="${HSG_RTP_NO_HSG_LOG_ROOT:-logs/no_hsg_protocol_matched}"
FORCE_RETRAIN="${HSG_RTP_FORCE_RETRAIN:-0}"
MAX_BATCH_ARGS=()
VALIDATION_ARGS=(--val_data_path pipeline/output/task_split/test_corrected_streaming.jsonl)
if [[ "$MODE" == "smoke" ]]; then
    RUN_ROOT="${RUN_ROOT}_smoke"
    LOG_ROOT="${LOG_ROOT}_smoke"
    MAX_BATCH_ARGS=(--max_train_batches 2 --max_val_samples 2)
    VALIDATION_ARGS=()
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME='^lo,docker,veth'

latest_checkpoint() {
    local stage_dir="$1"
    local epoch="$2"
    [[ -d "$stage_dir" ]] || return 0
    find "$stage_dir" -mindepth 2 -maxdepth 2 -type d -name "epoch_${epoch}" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_stage() {
    local stage_name="$1"
    local port="$2"
    shift 2
    echo "[$(date '+%F %T')] Starting $stage_name"
    "$DEEPSPEED" --num_gpus=2 --master_port="$port" train_streaming.py \
        --model_path "$MODEL_PATH" \
        --gpu_nums 2 \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --ablation no_hsge \
        "${MAX_BATCH_ARGS[@]}" \
        "$@" 2>&1 | tee "$LOG_ROOT/${stage_name}.log"
}

STAGE1_DIR="$RUN_ROOT/stage1_base"
STAGE1_CHECKPOINT="$(latest_checkpoint "$STAGE1_DIR" 1)"
if [[ -z "$STAGE1_CHECKPOINT" || "$FORCE_RETRAIN" == "1" ]]; then
    run_stage stage1_base 29610 \
        --data_path pipeline/output/task_split/train.jsonl \
        "${VALIDATION_ARGS[@]}" \
        --save_dir "$STAGE1_DIR" \
        --epochs 1 \
        --chunk_size 100 \
        --deepspeed_config configs/train_stage1.json
    STAGE1_CHECKPOINT="$(latest_checkpoint "$STAGE1_DIR" 1)"
fi
test -n "$STAGE1_CHECKPOINT"

STAGE2_DIR="$RUN_ROOT/stage2_recovery"
STAGE2_CHECKPOINT="$(latest_checkpoint "$STAGE2_DIR" 2)"
if [[ -z "$STAGE2_CHECKPOINT" || "$FORCE_RETRAIN" == "1" ]]; then
    run_stage stage2_recovery 29611 \
        --data_path pipeline/output/task_split/train_mixed_recovery.jsonl \
        "${VALIDATION_ARGS[@]}" \
        --save_dir "$STAGE2_DIR" \
        --epochs 2 \
        --resume_from_checkpoint "$STAGE1_CHECKPOINT" \
        --chunk_size 100 \
        --deepspeed_config configs/train_mixed_recovery.json
    STAGE2_CHECKPOINT="$(latest_checkpoint "$STAGE2_DIR" 2)"
fi
test -n "$STAGE2_CHECKPOINT"

STAGE3_DIR="$RUN_ROOT/stage3_sequence_budget"
STAGE3_CHECKPOINT="$(latest_checkpoint "$STAGE3_DIR" 3)"
if [[ -z "$STAGE3_CHECKPOINT" || "$FORCE_RETRAIN" == "1" ]]; then
    run_stage stage3_sequence_budget 29612 \
        --data_path pipeline/output/task_split/train_mixed_recovery.jsonl \
        --save_dir "$STAGE3_DIR" \
        --epochs 3 \
        --resume_from_checkpoint "$STAGE2_CHECKPOINT" \
        --chunk_size 200 \
        --deepspeed_config configs/train_mixed_recovery.json
    STAGE3_CHECKPOINT="$(latest_checkpoint "$STAGE3_DIR" 3)"
fi
test -n "$STAGE3_CHECKPOINT"

STAGE4_DIR="$RUN_ROOT/stage4_global_fix"
STAGE4_CHECKPOINT="$(latest_checkpoint "$STAGE4_DIR" 4)"
if [[ -z "$STAGE4_CHECKPOINT" || "$FORCE_RETRAIN" == "1" ]]; then
    run_stage stage4_global_fix 29613 \
        --data_path pipeline/output/task_split/train_short_global_fix_lora_only.jsonl \
        --save_dir "$STAGE4_DIR" \
        --epochs 4 \
        --resume_from_checkpoint "$STAGE3_CHECKPOINT" \
        --chunk_size 200 \
        --deepspeed_config configs/train_short_global_fix_lora_only.json \
        --freeze_non_lora
    STAGE4_CHECKPOINT="$(latest_checkpoint "$STAGE4_DIR" 4)"
fi
test -n "$STAGE4_CHECKPOINT"

cat > "$RUN_ROOT/checkpoints.env" <<EOF
NO_HSG_LOCAL_CHECKPOINT=$STAGE3_CHECKPOINT
NO_HSG_GLOBAL_CHECKPOINT=$STAGE4_CHECKPOINT
EOF

echo "No-HSG training complete"
echo "Local checkpoint:  $STAGE3_CHECKPOINT"
echo "Global checkpoint: $STAGE4_CHECKPOINT"
