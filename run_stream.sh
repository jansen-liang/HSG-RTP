#!/bin/bash

# =================================================================
# 1. 环境激活与系统变量
# =================================================================
# Activate an existing Conda environment when Conda is available.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${HSG_RTP_CONDA_ENV:-${HLR_CONDA_ENV:-hsg-rtp}}"
fi

# 双 RTX 2080 Ti 环境变量设置
mkdir -p ./data/tmp
export TMPDIR=$(pwd)/data/tmp
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker,veth
export UCX_TLS=^ib,^ucx,^rdmacm
export UCX_NET_DEVICES=
export CUDA_LAUNCH_BLOCKING=0

# =================================================================
# 2. 路径配置 (全部采用相对当前文件夹的路径)
# =================================================================
MODE="train"  # 可选模式: train , inference
if [ "$1" != "" ]; then
    MODE=$1
fi

# 【核心修改点】训练集与验证集相对路径
DATA_PATH="${HSG_RTP_TRAIN_DATA:-${HLR_TRAIN_DATA:-./pipeline/output/task_split/train.jsonl}}"
VAL_DATA_PATH="${HSG_RTP_EVAL_DATA:-${HLR_EVAL_DATA:-./pipeline/output/task_split/test.jsonl}}"

# 其他资源路径
MODEL_PATH="${HSG_RTP_MODEL_PATH:-${HLR_MODEL_PATH:-Qwen/Qwen3-8B}}"
TRAIN_DS_CONFIG="./configs/train.json"  
INFER_DS_CONFIG="./configs/inference.json"
SAVE_DIR="./checkpoints"
CHECKPOINT_PATH="${HSG_RTP_LORA_CHECKPOINT:-${HLR_LORA_CHECKPOINT:-./checkpoints/streaming_qwen_latest/epoch_1}}"

# =================================================================
# 3. 训练参数（双卡，每卡约 22GB）
# =================================================================
NUM_GPUS=2
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
EPOCHS=3

# 流式加载块大小；显存占用仍以首次完整训练为准
CHUNK_SIZE=100  

echo "Running streaming version in $MODE mode..."

# =================================================================
# 4. 执行命令逻辑
# =================================================================
case $MODE in
    "train")
        echo "Starting streaming training with DeepSpeed..."
        deepspeed --num_gpus=$NUM_GPUS --master_port=29503 train_streaming.py \
            --data_path "$DATA_PATH" \
            --val_data_path "$VAL_DATA_PATH" \
            --model_path "$MODEL_PATH" \
            --save_dir "$SAVE_DIR" \
            --epochs $EPOCHS \
            --use_lora \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --chunk_size $CHUNK_SIZE \
            --deepspeed_config "$TRAIN_DS_CONFIG" \
            --gpu_nums $NUM_GPUS
        ;;
    "inference")
        echo "Starting streaming inference and evaluation..."
        mkdir -p "./pipeline/output/result"
        deepspeed --num_gpus=1 --master_port=29504 inference_streaming.py \
            --model_path "$MODEL_PATH" \
            --data_path "$VAL_DATA_PATH" \
            --lora_path "$CHECKPOINT_PATH" \
            --max_samples 60 \
            --batch_size 1 \
            --chunk_size 50 \
            --output_path "./pipeline/output/result/streaming_results_$(date +%Y%m%d_%H%M%S).json" \
            --verbose
        ;;
esac
