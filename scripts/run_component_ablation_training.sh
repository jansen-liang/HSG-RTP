#!/usr/bin/env bash

set -euo pipefail

ABLATION="${1:-}"
MODE="${2:-full}"
case "$ABLATION" in
    no_global_topology|no_object_tokens|no_graph_updates_history) ;;
    *)
        echo "Usage: $0 {no_global_topology|no_object_tokens|no_graph_updates_history} [smoke|full]" >&2
        exit 2
        ;;
esac
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "Mode must be smoke or full" >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_DIR="${HSG_RTP_ENV_DIR:-/home/swzz/anaconda3/gra}"
DEEPSPEED="$ENV_DIR/bin/deepspeed"
MODEL_PATH="${HSG_RTP_MODEL_PATH:-/home/swzz/data/HLR/models/Qwen3-8B}"
RUN_ROOT="${HSG_RTP_ABLATION_RUN_ROOT:-checkpoints/${ABLATION}_protocol_matched}"
LOG_ROOT="${HSG_RTP_ABLATION_LOG_ROOT:-logs/${ABLATION}_protocol_matched}"
FORCE_RETRAIN="${HSG_RTP_FORCE_RETRAIN:-0}"
MAX_BATCH_ARGS=()
VALIDATION_ARGS=()
if [[ "${HSG_RTP_INTERMEDIATE_VALIDATION:-0}" == "1" ]]; then
    VALIDATION_ARGS=(--val_data_path pipeline/output/task_split/test_corrected_streaming.jsonl)
fi
if [[ "$MODE" == "smoke" ]]; then
    RUN_ROOT="${RUN_ROOT}_smoke"
    LOG_ROOT="${LOG_ROOT}_smoke"
    MAX_BATCH_ARGS=(--max_train_batches 2 --max_val_samples 2)
    VALIDATION_ARGS=()
fi

if [[ "$ABLATION" == "no_graph_updates_history" ]]; then
    STAGE1_DATA="pipeline/output/task_split/train_static_history.jsonl"
    MIXED_DATA="pipeline/output/task_split/train_mixed_recovery_static_history.jsonl"
    GLOBAL_DATA="pipeline/output/task_split/train_short_global_fix_lora_only_static_history.jsonl"
else
    STAGE1_DATA="pipeline/output/task_split/train.jsonl"
    MIXED_DATA="pipeline/output/task_split/train_mixed_recovery.jsonl"
    GLOBAL_DATA="pipeline/output/task_split/train_short_global_fix_lora_only.jsonl"
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
    echo "[$(date '+%F %T')] Starting $ABLATION/$stage_name"
    "$DEEPSPEED" --num_gpus=2 --master_port="$port" train_streaming.py \
        --model_path "$MODEL_PATH" \
        --gpu_nums 2 \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --ablation "$ABLATION" \
        "${MAX_BATCH_ARGS[@]}" \
        "$@" 2>&1 | tee "$LOG_ROOT/${stage_name}.log"
}

STAGE1_DIR="$RUN_ROOT/stage1_base"
STAGE1_CHECKPOINT="$(latest_checkpoint "$STAGE1_DIR" 1)"
if [[ -z "$STAGE1_CHECKPOINT" || "$FORCE_RETRAIN" == "1" ]]; then
    run_stage stage1_base 29620 \
        --data_path "$STAGE1_DATA" \
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
    run_stage stage2_recovery 29621 \
        --data_path "$MIXED_DATA" \
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
    run_stage stage3_sequence_budget 29622 \
        --data_path "$MIXED_DATA" \
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
    run_stage stage4_global_fix 29623 \
        --data_path "$GLOBAL_DATA" \
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
ABLATION=$ABLATION
LOCAL_CHECKPOINT=$STAGE3_CHECKPOINT
GLOBAL_CHECKPOINT=$STAGE4_CHECKPOINT
EOF

echo "$ABLATION training complete"
echo "Local checkpoint:  $STAGE3_CHECKPOINT"
echo "Global checkpoint: $STAGE4_CHECKPOINT"
