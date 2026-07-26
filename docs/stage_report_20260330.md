# HLR 阶段性总结

日期：2026-03-30

这份文档面向当前 HLR 的 scene graph 改造阶段，目标是回答三件事：

1. 目前已经做了什么，做到什么程度。
2. 接下来打算怎么做，哪里还不够硬，哪里还没想清楚。
3. 相关参考文献如何分组，哪些最值得直接借鉴。

## 1. 当前阶段目标

当前工作的主线不是继续堆更大的 LLM，而是把 HLR 从“基于已有符号场景图做动作生成”推进到“基于任务导向动态图做可执行规划与评测”。

围绕这个目标，当前阶段聚焦五件事：

1. 定义唯一 `canonical graph IR`
2. 把生成器从随机拼装改成可控生成
3. 统一语义与稳定 ID
4. 打通静态图和动态图转移
5. 建立 graph-level 验证与 benchmark 协议

## 2. 已完成工作

### 2.1 统一图表示层

**方法**

- 新增 `graph_ir/` 模块，统一引入 typed property graph：
  - `graph_ir/graph.py`：`CanonicalNode(id, type, subtype, attrs, states)` 与 `CanonicalEdge(source, target, relation, category, attrs)`
  - `graph_ir/ontology.py`：统一关系本体与 relation alias
  - `graph_ir/ids.py`：deterministic stable ID
  - `graph_ir/compilers.py`：兼容旧 schema、editor schema、OOP `nodes + edges` schema
  - `graph_ir/rules.py`：动作的 graph rewrite 规则骨架
  - `graph_ir/validation.py`：graph-level 验证
  - `graph_ir/generation.py`：可控生成约束与稳定命名基础设施
  - `graph_ir/cli.py`：命令行编译与验证入口

**结果**

- 目前已经能把三套图格式编译到同一个 canonical IR：
  - 旧版 `pipeline/sg/scene_graph.py`
  - editor 导出的 `pipeline/sg/generated/*.py`
  - `HLR_dataset` 的 OOP `nodes + edges` JSON
- canonical IR 已支持以下统一能力：
  - 节点/边统一表示
  - 关系归一化
  - 稳定 ID
  - 动作合法性检查
  - 图级验证
- 已验证样例：
  - `python -m graph_ir.cli pipeline/sg/scene_graph.py --scene HOTEL`
    - `schema=legacy_scene`
    - `nodes=133`
    - `edges=213`
    - `valid=True`
  - `python -m graph_ir.cli HLR_dataset/data/scene_graphs/hospital_scene_0.json`
    - `schema=oop_scene`
    - `nodes=50`
    - `edges=72`
    - `valid=True`
- smoke test 已通过：
  - `pytest -q tests/test_graph_ir_smoke.py`
  - 结果：`4 passed in 0.23s`

### 2.2 对现有 HLR 主链的结构诊断

**方法**

- 逐文件检查了现有 scene graph 消费链：
  - `pipeline/utils/graph_utils.py`
  - `pipeline/utils/sample_builder.py`
  - `utils/graph_encoder.py`
  - `pipeline/utils/state_manager.py`
  - `pipeline/utils/action_planner.py`
- 对已有生成数据做了规模统计：
  - `pipeline/output/hlr_dataset_20251129_155602.json`

**结果**

- 当前 HLR 已经有一个比较清楚的层次式数据链：
  - task generation
  - action planning
  - streaming sample build
  - state synchronization
- 当前系统的优点：
  - `sample_builder.py` 已做到“每个样本使用动作执行前的真实状态”
  - 这是后续做动态图 benchmark 的重要基础
- 当前系统的主要结构性短板：
  - `get_global_view()` 会把全局图裁成“房间楼层 + 邻接”，对象级关系基本丢失
  - `get_local_view()` 只保留当前房间，且对物体数量做截断
  - `HierarchicalSceneGraphEncoder.encode_local_scene()` 主要把物体转成 `type/state/affordance` 文本后聚合，关系推理很弱
  - `SceneGraphStateManager` 仍然是 dict 级状态更新，不是 graph-native 执行器
  - `action_planner.py` 里的电梯按钮状态仍有随机模拟成分，难以作为严格专家轨迹
- 当前数据规模概况：
  - 350 条任务记录
  - 5 个场景：`hotel / supermarket / allensville / office / pudu`
  - 任务类型分布：
    - `tidying`: 132
    - `delivery`: 116
    - `guidance`: 102
  - 平均动作长度：23.50
  - 最长动作长度：103
  - streaming samples 总数：8534
    - global：308
    - local：8226

### 2.3 问题定义已经收敛

**方法**

- 对照 HLR 现状与 MomaGraph、SayPlan、Taskography 等工作，重新界定本项目的研究落点。
- 将 scene graph 改造目标收敛到“五件事”。

**结果**

- 目前的研究定位已经比较明确：
  - 不是做“纯视觉 scene graph generation”
  - 也不是只做“文本到动作序列”
  - 而是做 `task-oriented executable graph world model`
- 这意味着后续工作要把三件事连成闭环：
  - 图表示
  - 图执行
  - 图评测

### 2.4 文献线索已初步整理

**方法**

- 已形成一份与当前改造直接相关的文献笔记：
  - `docs/scene_graph_research_notes.md`

**结果**

- 已经把 scene graph 和 planning 之间最相关的代表性工作收敛成几条主线，足够支持下一阶段的系统设计与 related work 撰写。

## 3. 未完成工作与下一步方案

这一部分按“打算怎么做 + 当前问题”来写。

### 3.1 任务条件子图检索

**打算怎么做**

- 在 `compile_to_canonical()` 之后、进入 encoder 之前，新增 `instruction-conditioned subgraph retriever`
- 输入：
  - 指令
  - 当前 canonical graph
  - agent 当前状态
- 输出：
  - 一个 task-relevant subgraph，而不是现在的固定 global/local 裁图
- 第一版先做规则或启发式检索：
  - room relevance
  - object mention matching
  - affordance matching
  - shortest-path room expansion
- 第二版再考虑学习式检索器

**当前问题**

- 目前没有现成的“任务相关子图”监督标签
- 如果直接学习检索器，需要先决定 supervision 从哪里来：
  - 规则伪标签
  - planner 轨迹反推
  - 外部数据集迁移
- 还没完全想清楚“全局子图”和“局部子图”之间的边界如何定义

### 3.2 假设边与动态剪枝

**打算怎么做**

- 在 canonical graph 中显式允许不确定功能边：
  - 例如 `controls`
  - `connected_to`
  - `powered_by`
- 给边加入状态：
  - `hypothesis`
  - `confirmed`
  - `rejected`
  - `confidence`
- 执行动作后，根据观测更新这些边
- 这部分直接借鉴 MomaGraph 的思路，但会落到 HLR 的 graph executor 里

**当前问题**

- 还缺“观测模型”的定义
- 目前 `scan/open/press/use` 之后，系统还没有一个统一的 observation schema
- 也还没决定用硬规则更新，还是做轻量 belief update

### 3.3 将 `state_manager` 迁移到 graph rewrite executor

**打算怎么做**

- 让 `pipeline/utils/state_manager.py` 不再直接维护 legacy dict
- 改为调用 `graph_ir/rules.py` 中的规则：
  - precondition check
  - effect apply
  - graph mutation
- legacy scene 只作为输入格式，通过 compiler 先转成 canonical IR

**当前问题**

- 现有动作定义还不统一：
  - `goto`
  - `pick`
  - `place`
  - `scan`
  - `press`
  - `wait`
- 电梯、按钮、容器、部件对象等边界情况还需要收口
- 当前 `state_manager` 和旧 planner 的字段假设很多，替换时容易牵动训练主链

### 3.4 可控生成器

**打算怎么做**

- 用 `graph_ir/generation.py` 里的 `GenerationConstraints` 作为接口，逐步替换掉当前随机拼装逻辑
- 显式控制：
  - 楼层数
  - 房间数
  - 跨层边密度
  - 房间邻接密度
  - 对象密度
  - 容器深度
  - 最短计划长度
  - 任务可达性

**当前问题**

- 现在只是有约束定义和稳定命名器，还没有真正接到 `scene_factory.py`
- 最难的部分不是“采样图”，而是“保证图和任务同时成立”
- 后续必须显式验证：
  - 图连通
  - 目标可满足
  - 计划长度达到设定范围
  - 不存在 trivially easy 的路径

### 3.5 新 benchmark 与评测协议

**打算怎么做**

- 在现有 Jaccard/LCS 之外，增加更适合 graph-based agent 的评测：
  - graph validity
  - executability
  - success rate
  - cost ratio to expert
  - replanning recovery rate
  - hidden functional relation discovery
- 评测输入应区分：
  - oracle graph
  - retrieved subgraph
  - predicted graph

**当前问题**

- 当前 GT 轨迹来自启发式 planner，不是严格最优专家
- `action_planner.py` 中电梯按钮状态还带随机模拟
- 如果 benchmark 要站住，需要先明确：
  - 专家计划如何定义
  - 失败算子如何统计
  - 重规划时机如何规定

### 3.6 视觉侧建图暂不作为第一优先级

**打算怎么做**

- 当前阶段优先把“图表示-图执行-图评测”打通
- 视觉输入生成 scene graph 可以作为下一阶段扩展：
  - 接外部 VLM
  - 对接 MomaGraph-style task-oriented graph prediction

**当前问题**

- HLR 现在的优势不在视觉建图，而在符号规划链
- 如果过早转去做视觉侧，容易把当前最应该补的执行器和 benchmark 地基搁置

## 4. 目前最关键的未决问题

这几项是当前最值得提前盯住的风险点。

### 4.1 canonical IR 会不会和旧主链长期并存

- 现在 `graph_ir/` 已经可用，但训练与推理主链还没有真正迁过去
- 如果长期“双轨并存”，维护成本会越来越高
- 因此下一阶段至少要明确一个迁移边界：
  - 先迁 executor
  - 还是先迁 encoder 输入

### 4.2 动态图更新到底是“真实状态”还是“agent belief”

- 这一点非常关键
- 如果是“真实状态图”，规则会简单一些，但不适合部分可观测环境
- 如果是“belief graph”，就必须定义观测不确定性和图更新机制
- 目前更可行的路线是：
  - 先做 deterministic state-aware graph
  - 再逐步扩展为 hypothesis graph

### 4.3 benchmark 想要证明什么

- 如果 benchmark 只是证明“scene graph 比纯文本好”，那难度不够
- 更有价值的目标是证明：
  - 任务条件子图检索是否比全图编码更有效
  - graph rewrite executor 是否能提高可执行性
  - 动态图更新是否能提升重规划成功率

## 5. 参考文献

下面分两部分整理：scene graph 方向与任务规划方向。

### 5.1 Scene Graph 方向

#### 方向 A：在线增量建图

这类工作把 scene graph 看成机器人探索过程中逐步长出来的世界模型，重点在于在线更新、地图优化和跨层表示。

- Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization  
  https://arxiv.org/abs/2201.13360
- Embodied Semantic Scene Graph Generation  
  https://proceedings.mlr.press/v164/li22e.html

#### 方向 B：开放词汇 / 功能型 scene graph

这类工作强调 scene graph 不只是 `on/in/neighbor`，还要包含开放词汇语义、功能关系和可交互部件。

- Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation  
  https://arxiv.org/abs/2403.17846
- Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces  
  https://arxiv.org/abs/2503.19199

#### 方向 C：面向规划的 scene graph

这类工作把 scene graph 当成 planning 的中间层，核心问题是 task-relevant subgraph、可执行性、以及规划复杂度控制。

- SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning  
  https://arxiv.org/abs/2307.06135
- TASKOGRAPHY: Evaluating robot task planning over large 3D scene graphs  
  https://arxiv.org/abs/2207.05006
- Optimal Scene Graph Planning with Large Language Model Guidance  
  https://arxiv.org/abs/2309.09182

#### 方向 D：状态感知统一图

这类工作进一步把“静态结构 + 动态状态 + 任务相关关系”统一到一张图里，并配套 benchmark。

- MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning  
  https://openreview.net/forum?id=3eTr9dGwJv

### 5.2 任务规划方法

这一部分只总结和当前 HLR 最相关的具身任务规划路线，不展开传统自动规划全史。

#### 方向 A：直接让语言模型做任务分解

这类方法不显式建图，也不一定有外部 planner，优势是简单直接，缺点是可执行性和状态跟踪通常较弱。

- Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents  
  https://arxiv.org/abs/2201.07207

#### 方向 B：技能/可供性约束下的 LLM 规划

这类方法不让 LLM 直接“随便想”，而是让它在技能库、affordance 或执行反馈约束下规划。

- Do As I Can, Not As I Say: Grounding Language in Robotic Affordances  
  https://arxiv.org/abs/2204.01691
- Inner Monologue: Embodied Reasoning through Planning with Language Models  
  https://arxiv.org/abs/2207.05608
- Code as Policies: Language Model Programs for Embodied Control  
  https://arxiv.org/abs/2209.07753

#### 方向 C：从语言生成规划域或符号模型

这类方法的核心不是让 LLM 直接输出动作，而是让它帮助生成可供传统 planner 使用的规划域。

- Large Language Models as Planning Domain Generators  
  https://arxiv.org/abs/2405.06650

#### 方向 D：scene-graph-conditioned planning

这类方法最接近我们当前路线，即先用 scene graph 建模环境，再在图上做规划或重规划。

- SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning  
  https://arxiv.org/abs/2307.06135
- Optimal Scene Graph Planning with Large Language Model Guidance  
  https://arxiv.org/abs/2309.09182
- Task and Motion Planning in Hierarchical 3D Scene Graphs  
  https://arxiv.org/abs/2403.08094

## 6. 对当前 HLR 的直接启发

综合以上文献与当前代码状态，当前最值得坚持的路线不是“再换更大的 LLM”，而是：

1. 用 canonical IR 统一表示层
2. 用 task-conditioned retriever 代替固定裁图
3. 用 graph rewrite rules 统一执行与状态更新
4. 用 graph-level validation 建立严格评测
5. 在这个基础上，再考虑视觉侧 task-oriented graph generation

一句话概括当前阶段的判断：

**HLR 已经具备 scene graph 规划系统的雏形，但还缺一个统一、可执行、可验证的图世界模型。当前新增的 `graph_ir/` 是这个世界模型的第一层地基。**
