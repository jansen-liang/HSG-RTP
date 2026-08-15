#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTHOR_ROOT="${GRID_AUTHOR_ROOT:-/home/swzz/disk2T/grid/ridsg}"
GRID_PYTHON="${GRID_PYTHON:-/home/swzz/anaconda3/grid/bin/python}"
CONFIG="${GRID_CONFIG:-$AUTHOR_ROOT/hparams_hsg_rtp.cfg}"
RAW_TRAIN="$REPO_ROOT/third_party/GRID/dataset/hsg_rtp_train"
RAW_TEST="$REPO_ROOT/third_party/GRID/dataset/hsg_rtp_test"
PREPROCESSED_ROOT="$REPO_ROOT/third_party/GRID/preprocess_data"
PREPROCESSED_TRAIN="$PREPROCESSED_ROOT/hsg_rtp_train_clip"
PREPROCESSED_TEST="$PREPROCESSED_ROOT/hsg_rtp_test_clip"
LOG_DIR="$REPO_ROOT/logs/grid_author_adaptation"

mkdir -p "$PREPROCESSED_ROOT" "$LOG_DIR"
export LD_LIBRARY_PATH="/home/swzz/anaconda3/grid/lib:${LD_LIBRARY_PATH:-}"

"$GRID_PYTHON" "$REPO_ROOT/scripts/preprocess_grid_author.py" \
  --author-root "$AUTHOR_ROOT" \
  --config "$CONFIG" \
  --raw-data "$RAW_TRAIN" \
  --output "$PREPROCESSED_TRAIN" \
  --device 0 \
  --resume 2>&1 | tee "$LOG_DIR/preprocess_train.log"

"$GRID_PYTHON" "$REPO_ROOT/scripts/preprocess_grid_author.py" \
  --author-root "$AUTHOR_ROOT" \
  --config "$CONFIG" \
  --raw-data "$RAW_TEST" \
  --output "$PREPROCESSED_TEST" \
  --device 0 \
  --resume 2>&1 | tee "$LOG_DIR/preprocess_test.log"

cd "$AUTHOR_ROOT"
"$GRID_PYTHON" train.py \
  --parser_mode trainer \
  --config_path "$CONFIG" \
  --preprocessed_data_path "$PREPROCESSED_TRAIN" \
  --fit_flag true \
  --from_ckpt_flag false \
  --experiment_name hsg_rtp_author_adaptation \
  --gpu_devices 0,1 2>&1 | tee "$LOG_DIR/train.log"

CHECKPOINT="$(find "$AUTHOR_ROOT/logs/hsg_rtp_author_adaptation" -name '*.ckpt' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
test -n "$CHECKPOINT"
printf '%s\n' "$CHECKPOINT" > "$LOG_DIR/checkpoint.txt"

"$GRID_PYTHON" train.py \
  --parser_mode predictor \
  --config_path "$CONFIG" \
  --preprocessed_data_path "$PREPROCESSED_TEST" \
  --fit_flag false \
  --from_ckpt_flag true \
  --use_ckpt_config false \
  --ckpt_path "$CHECKPOINT" \
  --experiment_name hsg_rtp_author_adaptation_test \
  --gpu_devices 0 2>&1 | tee "$LOG_DIR/test.log"

cd "$REPO_ROOT"
"$GRID_PYTHON" scripts/evaluate_grid_author_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --dataset pipeline/output/task_split/test_corrected_streaming.jsonl \
  --output-dir evaluation_results/grid_author_retrained_direct70 \
  --author-root "$AUTHOR_ROOT" \
  --config "$CONFIG" \
  --device cuda:0 2>&1 | tee "$LOG_DIR/rollout.log"
