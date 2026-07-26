#!/bin/bash

# 解析命令行参数
MODE="train"  # train , inference
if [ "$1" != "" ]; then
    MODE=$1
fi

# 公共参数
DATA_PATH="./pipeline/output/hlr_dataset_20251107_162657.jsonl"
VAL_DATA_PATH="./pipeline/output/hlr_dataset_20251018_222538.jsonl"
MODEL_PATH="./models/Qwen3-8B"
TRAIN_DS_CONFIG="./configs/train.json"
INFER_DS_CONFIG="./configs/inference.json"
SAVE_DIR="./checkpoints"
CHECKPOINT_PATH="./checkpoints/20251002_150038/epoch_1"

# LoRA参数
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 训练参数
EPOCHS=60
LR=2e-5
CHUNK_SIZE=400

echo "Running in $MODE mode..."

case $MODE in
    "train")
        echo "Starting training..."
        deepspeed --num_gpus=1 --master_port=29501 train.py \
            --data_path $DATA_PATH \
            --val_data_path $VAL_DATA_PATH \
            --model_path $MODEL_PATH \
            --save_dir $SAVE_DIR \
            --epochs $EPOCHS \
            --lr $LR \
            --use_lora \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --chunk_size $CHUNK_SIZE \
            --deepspeed_config $TRAIN_DS_CONFIG
        ;;
    "inference")
        echo "Starting inference with DeepSpeed..."
        deepspeed --num_gpus=1 --master_port=29502 inference.py \
            --model_path $MODEL_PATH \
            --data_path $DATA_PATH \
            --use_lora \
            --lora_checkpoint $CHECKPOINT_PATH \
            --lora_r $LORA_R \
            --lora_alpha $LORA_ALPHA \
            --lora_dropout $LORA_DROPOUT \
            --deepspeed_config $INFER_DS_CONFIG \
            --max_samples 10 \
            --output_json "./pipeline/output/inference_results_$(date +%Y%m%d_%H%M%S).json"
        ;;
esac