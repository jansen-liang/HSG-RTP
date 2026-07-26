#!/bin/bash

# =================================================================
# 1. 环境激活与系统变量
# =================================================================
# 激活你的 hlr 环境 (使用相对 ~ 路径)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hlr

# 环境变量优化：确保 4090 在单卡模式下不因为网络检查而卡顿
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
DATA_PATH="./pipeline/output/hlr_dataset_20251107_162657.jsonl"
VAL_DATA_PATH="./pipeline/output/hlr_dataset_20251018_222538.jsonl"

# 其他资源路径
MODEL_PATH="./models/Qwen3-8B"
TRAIN_DS_CONFIG="./configs/train.json"  
INFER_DS_CONFIG="./configs/inference.json"
SAVE_DIR="./checkpoints"
CHECKPOINT_PATH="./checkpoints/streaming_qwen_latest/epoch_1"

# =================================================================
# 3. 训练参数 (48GB 显存专属优化)
# =================================================================
NUM_GPUS=1
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
EPOCHS=60    

# 流式加载块大小。48GB 显存非常大，设为 100 可以让 GPU 吞吐更顺畅
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
            --batch_size 4 \
            --chunk_size 50 \
            --output_path "./pipeline/output/result/streaming_results_$(date +%Y%m%d_%H%M%S).json" \
            --verbose
        ;;
esac