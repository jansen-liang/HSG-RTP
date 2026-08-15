# HSG-RTP

**基于层次化场景图的状态感知闭环机器人任务规划**

[English](README.md) · [开源许可证](LICENSE)

![HSG-RTP 方法概览](assets/hsg-rtp-overview.png)

HSG-RTP 是面向多房间、多楼层和长时程任务的机器人规划研究代码库。系统使用层次化场景图（Hierarchical Scene Graph, HSG）维护环境拓扑、对象状态和机器人状态，通过独立的全局/局部图视图进行子任务规划与动作落地，并在执行失败后进行结构化恢复。

> **发布状态：** 仓库包含模型、数据生成、训练、严格评估、闭环恢复、消融实验和外部基线适配代码。模型权重、生成数据集、实验输出和私有凭据不会随仓库发布，已审计的定量结果列于下文。

## 方法概览

```text
自然语言指令 + 当前 HSG
          │
          ├── 全局视图：房间拓扑与宏观区域
          │        └── 生成全局子任务计划
          │
          └── 局部视图：当前房间与显式对象
                   └── 生成下一条可执行动作
                              │
                    解析 → 检查 → 执行 → 验证
                              │
                    提交状态或触发恢复/重规划
```

核心能力包括：

- **双视图 HSG 编码：** 全局路径使用 GATv2 编码房间拓扑，局部路径保留独立对象 token。
- **共享规划模型：** 冻结 Qwen3-8B 主干，训练 LoRA adapter 和 HSG encoder。
- **事务式执行：** 对 `goto`、`scan`、`pick`、`place`、`press`、`wait` 进行前置条件检查和状态更新，只提交通过验证的状态。
- **闭环恢复：** 支持初始计划修复、临时技能失败自动重试、局部纠错重试和未完成全局计划替换。

## 安装

推荐环境为 Linux、Python 3.10、CUDA 12.1 和 NVIDIA GPU。

```bash
git clone https://github.com/jansen-liang/HSG-RTP.git
cd HSG-RTP

conda create -n hsg-rtp python=3.10 -y
conda activate hsg-rtp
pip install -r requirements.txt
```

默认基础模型为 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)。可以直接使用 Hugging Face 模型标识，也可以设置本地路径：

```bash
cp .env.example .env
export HSG_RTP_MODEL_PATH=Qwen/Qwen3-8B
export HSG_RTP_TRAIN_DATA=pipeline/output/task_split/train.jsonl
export HSG_RTP_EVAL_DATA=pipeline/output/task_split/test.jsonl
```

旧的 `HLR_*` 环境变量和 `HLR_dataset/` 目录暂时保留，用于兼容已有脚本和 checkpoint；新代码建议使用 `HSG_RTP_*`。

## 数据准备

生成任务记录和流式训练样本：

```bash
./build_data.sh
```

按任务身份进行确定性划分，避免同一任务的不同步骤泄漏到训练集和测试集：

```bash
python scripts/split_task_dataset.py \
  pipeline/output/hsg_rtp_dataset_<timestamp>.json \
  --output-dir pipeline/output/task_split \
  --test-ratio 0.2 \
  --seed 42
```

可选生成 recovery-augmented 训练数据：

```bash
python scripts/build_mixed_recovery_dataset.py \
  --input pipeline/output/task_split/train.jsonl \
  --output pipeline/output/task_split/train_mixed_recovery.jsonl
```

## 训练

```bash
export HSG_RTP_MODEL_PATH=Qwen/Qwen3-8B
export HSG_RTP_TRAIN_DATA=pipeline/output/task_split/train.jsonl
export HSG_RTP_EVAL_DATA=pipeline/output/task_split/test.jsonl
./run_stream.sh train
```

参考训练设置：

| 配置 | 数值 |
| --- | --- |
| 基础模型 | 冻结的 Qwen3-8B |
| LoRA | rank 16，alpha 32，dropout 0.1 |
| 优化器 | AdamW，学习率 `2e-5` |
| 有效 batch size | 16 |
| 训练轮数 | 3 |
| 精度 | FP16 |
| 分布式训练 | 2 GPU，DeepSpeed ZeRO-2 |
| 节点/前缀/输出长度 | 128 / 512 / 448 tokens |

## 推理

```bash
export HSG_RTP_LORA_CHECKPOINT=checkpoints/<run>/epoch_<n>
./run_stream.sh inference
```

推理程序会直接读取 checkpoint 中的 `adapter_config.json`，自动使用正确的 LoRA rank、alpha 和 dropout。

## 严格任务级评估

首先使用参考动作校准数据集：

```bash
python evaluate_tasks.py \
  pipeline/output/task_split/test.jsonl \
  --output-dir evaluation_results/oracle
```

评估模型输出：

```bash
python evaluate_tasks.py \
  pipeline/output/task_split/test.jsonl \
  --predictions evaluation_results/model_predictions.jsonl \
  --output-dir evaluation_results/model
```

- **Plan SR：** 检查全局计划格式、房间引用、拓扑可达性、顺序与任务覆盖。
- **Strict Exec SR：** 严格执行所有局部动作，要求每一步满足前置条件并最终达到任务目标。
- 无法由参考动作完成的无效 benchmark task 会单独报告。

## 实验结果

结果使用修正后的留出测试集：共提交 70 个任务，其中 66 个由当前模拟器支持。下表所有方法均使用相同的 bounded controller；独立训练的 No-HSG 模型不接收图 token。

| 方法 | Plan SR (%) | Strict Exec SR (%) | Global Jaccard | Global LCS | Local Jaccard | Local LCS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B, zero-shot | 92.42 | 56.06 | 0.624 | 0.733 | 0.128 | 0.117 |
| **HSG-RTP** | **93.94** | **80.30** | **0.634** | **0.739** | **0.354** | **0.312** |
| No-HSG | 86.36 | 77.27 | 0.580 | 0.689 | 0.315 | 0.284 |
| 不使用全局拓扑 | 90.91 | 81.82 | 0.581 | 0.683 | 0.339 | 0.302 |
| 不使用独立对象 token | 92.42 | 84.85 | 0.613 | 0.711 | 0.342 | 0.305 |
| 不使用图更新/历史 | 89.39 | 40.91 | 0.614 | 0.714 | 0.101 | 0.091 |

在 1,616 个留出的单步规划样本上，HSG-RTP 的 Jaccard 为 0.7785，LCS ratio 为 0.7922，全局/局部模式准确率为 98.87%，样本加权 loss 为 0.0722。

外部方法的原生接口、控制器和输出协议不同，因此单独报告，不与上表或彼此作直接对比。

| 方法 | 适配方式 | Plan SR (%) | Exec SR (%) |
| --- | --- | ---: | ---: |
| SayPlan | 基于论文重实现，使用 Qwen3-8B | 19.70 | 10.61 |
| SayCan | 官方 notebook 适配，使用符号 affordance | N/A | 42.42 |
| GRID | 本地作者代码，重新训练 RN50 | N/A | 1.52 |
| DELTA | 上游部分适配，使用固定 problem builder | 18.18 | 15.15 |

具体协议和适配边界见 [docs/baseline_protocol.md](docs/baseline_protocol.md)。本地存在未入库的评估产物时，可使用 `python scripts/audit_paper_results.py` 核验结果摘要。

## 闭环恢复

`evaluation/` 提供对象移动、通路阻塞和一次性技能失败等可控扰动，并记录自动重试、局部恢复和全局计划替换过程。

```bash
python scripts/evaluate_recovery_checkpoint.py \
  --checkpoint checkpoints/<run>/epoch_<n> \
  --model-path Qwen/Qwen3-8B \
  --dataset pipeline/output/task_split/test.jsonl \
  --baseline-results evaluation_results/recovery_baseline.json \
  --output evaluation_results/recovery_model.json
```

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

图编码器与模型加载测试需要安装 `requirements.txt` 中的 PyTorch 和 PyTorch Geometric。

## 引用

正式发表后将补充引用信息。当前使用本项目时，请引用仓库地址。

## 许可证

仓库原创代码采用 [Apache License 2.0](LICENSE)。预训练模型、数据集、模拟器和其他第三方组件继续遵循各自的许可证。
