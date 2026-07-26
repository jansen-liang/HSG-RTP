# HLR: Hierarchical Large-scale Robot

HLR是一个基于层次化场景图的机器人指令理解与规划系统，结合了大语言模型（LLM）和图神经网络，实现对复杂机器人任务的智能规划。

## Public-release configuration

The repository does not contain model weights, private datasets, checkpoints, or API credentials. Copy `.env.example` to `.env` and configure only the services and local paths you use. API-backed agents read credentials from environment variables; credentials must never be written into source files.

```bash
cp .env.example .env
export LIGHTAI_API_KEY=<your-lightai-key>   # only when using a LightAI-backed agent
export ZHIPUAI_API_KEY=<your-zhipuai-key>   # only when using a ZhipuAI-backed agent
export BLTCY_API_KEY=<your-bltcy-key>       # only when using a BLTCY-backed agent
export HLR_MODEL_PATH="Qwen/Qwen3-8B"
```

Generated datasets are written to `pipeline/output/` and are intentionally excluded from version control. Model weights and checkpoints are also excluded and must be downloaded according to their original licenses.

## 📋 项目概述

本项目使用层次化场景图（Hierarchical Scene Graph）来表示环境信息，通过融合指令编码和场景图编码，训练一个能够理解自然语言指令并生成机器人动作序列的模型。

## Canonical Graph IR

仓库现在提供了一套独立的 canonical scene graph 基础设施，放在 `graph_ir/` 下，用来统一旧版 `pipeline/sg/scene_graph.py`、编辑器导出 schema、以及 `HLR_dataset` 的 OOP `nodes + edges` schema。

核心模块：
- `graph_ir/graph.py`：typed property graph，统一为 `Node(type, subtype, attrs, states)` 和 `Edge(relation, category, attrs)`
- `graph_ir/compilers.py`：旧 schema / editor schema / OOP schema -> canonical IR
- `graph_ir/ontology.py`：稳定关系本体与关系归一化
- `graph_ir/ids.py`：稳定、可复现的 deterministic ID
- `graph_ir/rules.py`：动作的 graph rewrite 规则骨架
- `graph_ir/validation.py`：graph-level 验证，包括连通性、关系合法性和可执行动作验证
- `graph_ir/generation.py`：可控生成约束与稳定命名分配器

快速使用：

```bash
python -m graph_ir.cli pipeline/sg/scene_graph.py --scene HOTEL
python -m graph_ir.cli HLR_dataset/data/scene_graphs/hospital_scene_0.json
python -m graph_ir.cli pipeline/sg/generated/hotel.py --dump /tmp/hotel_canonical.json
```

### 主要特性
- 🏗️ **层次化场景表示**：支持宏观区→房间→物品的三层结构
- 🤖 **指令理解**：基于Qwen3-8B的自然语言指令处理
- 🧠 **图神经网络**：使用图编码器处理复杂的空间关系
- ⚡ **高效训练**：支持DeepSpeed分布式训练和LoRA微调
- 📊 **可视化工具**：提供场景图3D可视化功能

## 🛠️ 环境配置

### 系统要求
- **操作系统**：Linux (Ubuntu 18.04+)
- **Python版本**：Python 3.8+
- **CUDA版本**：CUDA 12.1+
- **GPU要求**：至少需要1张GPU（推荐3张或以上用于分布式训练）
- **内存要求**：至少16GB RAM，推荐32GB+

### 1. 创建Conda环境

```bash
# 创建Python环境
conda create -n hlr python=3.10
conda activate hlr

# 安装CUDA支持的PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. 安装依赖包

```bash
# 克隆项目
git clone <your-repo-url>
cd HLR

# 安装Python依赖
pip install -r requirements.txt

# 安装DeepSpeed（如果上述步骤未成功安装）
pip install deepspeed
```

### 3. 验证环境

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 📦 第三方模型下载

项目依赖以下预训练模型，需要下载并放置到指定目录：

### 1. Qwen3-8B 大语言模型

**下载方式一：Hugging Face**
```bash
# 安装git-lfs
git lfs install

# 下载模型到指定目录
cd models/
git clone https://huggingface.co/Qwen/Qwen2.5-8B-Instruct Qwen3-8B
```

**下载方式二：ModelScope**
```bash
# 使用ModelScope下载
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-8B-Instruct', cache_dir='./models/Qwen3-8B')
"
```

### 2. all-MiniLM-L6-v2 文本编码模型

```bash
# 下载文本编码模型
cd models/
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

### 目录结构检查
确保模型目录结构如下：
```
models/
├── Qwen3-8B/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-*.safetensors
│   └── ...
└── all-MiniLM-L6-v2/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── ...
```

## 🚀 运行方法

### 1. 准备数据

确保数据文件存在：
```bash
# 训练数据
data/instruction/robot_instruction_dataset_1000_20250924_220912.jsonl
# 验证数据  
data/instruction/robot_instruction_dataset_50_20250920_223347.jsonl
```

### 2. 修改配置参数

编辑 `run.sh` 文件中的关键参数：

```bash
# 核心路径配置（必须修改）
DATA_PATH="/path/to/your/training/data.jsonl"           # 训练数据路径
VAL_DATA_PATH="/path/to/your/validation/data.jsonl"    # 验证数据路径
MODEL_PATH="/path/to/your/Qwen3-8B"                   # Qwen模型路径
TEXT_MODEL_PATH="/path/to/your/all-MiniLM-L6-v2"      # 文本编码器路径

# 训练参数（可选修改）
BATCH_SIZE=3        # 单GPU批次大小，根据显存调整
EPOCHS=100          # 训练轮数
LR=2e-5             # 学习率
--num_gpus=3        # GPU数量，根据实际硬件调整

# LoRA微调参数
--lora_r 8          # LoRA秩，控制参数量
--lora_alpha 16     # LoRA缩放因子
--lora_dropout 0.1  # Dropout率
```

### 3. DeepSpeed配置调整

编辑 `configs/deepspeed_config.json`：

```json
{
    "train_batch_size": 18,                    # 总批次大小 = num_gpus * micro_batch_size * gradient_accumulation
    "train_micro_batch_size_per_gpu": 3,       # 单GPU批次大小，根据显存调整
    "gradient_accumulation_steps": 2,          # 梯度累积步数
    "zero_optimization": {
        "stage": 2,                            # ZeRO stage，2表示分片优化器状态
        "offload_optimizer": {
            "device": "cpu"                    # 优化器状态转移到CPU
        }
    }
}
```

**重要参数说明**：
- 如果显存不足，减少 `train_micro_batch_size_per_gpu`
- 如果想保持总批次大小，相应增加 `gradient_accumulation_steps`
- ZeRO stage 3可以进一步节省显存，但会增加通信开销

### 4. 开始训练

```bash
# 给脚本执行权限
chmod +x run.sh

# 启动训练
./run.sh
```

### 5. 监控训练

训练过程中会输出：
- 训练损失和验证指标
- Jaccard相似度和LCS比率
- 模型预测样例
- WandB可视化（如果配置）

检查点保存在：
```
checkpoints/TIMESTAMP/
├── epoch_1/
├── epoch_3/
└── ...
```

### 6. 推理测试

```bash
# 使用训练好的模型进行推理
python inference.py \
    --model_path ./checkpoints/TIMESTAMP/epoch_X \
    --input_text "请帮我把红酒从酒吧拿到大厅" \
    --scene_graph_file ./data/sg/scene_graph.py
```

## 📊 性能监控

### 训练指标
- **Loss**: 训练损失，应逐渐下降
- **Jaccard相似度**: 预测动作与真实动作的交集比例
- **LCS比率**: 最长公共子序列比率，衡量动作顺序的正确性

### 系统监控
```bash
# GPU使用情况
nvidia-smi

# 内存使用
htop

# GPU实时监控
nvtop
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：减小批次大小
   # 在run.sh中修改：BATCH_SIZE=1
   # 在deepspeed_config.json中修改："train_micro_batch_size_per_gpu": 1
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径是否正确
   ls -la models/Qwen3-8B/
   ls -la models/all-MiniLM-L6-v2/
   ```

3. **DeepSpeed初始化失败**
   ```bash
   # 检查环境变量
   echo $CUDA_VISIBLE_DEVICES
   # 重新安装DeepSpeed
   pip uninstall deepspeed -y && pip install deepspeed
   ```

4. **数据加载错误**
   ```bash
   # 检查数据格式
   head -n 1 data/instruction/your_data.jsonl
   # 确保JSONL格式正确
   ```

### 调试建议
- 使用较小的数据集进行快速验证
- 单GPU模式测试：`--num_gpus=1`
- 启用详细日志：在代码中添加 `print` 语句
- 检查磁盘空间是否充足

## 📁 项目结构

```
HLR/
├── train.py              # 训练主程序
├── eval.py               # 验证评估
├── inference.py          # 推理脚本
├── run.sh                # 启动脚本
├── requirements.txt      # Python依赖
├── configs/
│   └── deepspeed_config.json  # DeepSpeed配置
├── utils/
│   ├── hlr.py           # 主模型定义
│   ├── dataloader.py    # 数据加载器
│   ├── graph_encoder.py # 图编码器
│   └── instruction_encoder.py # 指令编码器
├── data/
│   ├── instruction/     # 指令数据集
│   └── sg/              # 场景图相关
├── models/              # 预训练模型目录
└── checkpoints/         # 模型检查点
```

## 🎯 使用技巧

1. **显存优化**：使用gradient checkpointing和混合精度训练
2. **数据并行**：多GPU训练可显著加速
3. **学习率调整**：根据验证指标动态调整学习率
4. **早停策略**：监控验证损失，避免过拟合
