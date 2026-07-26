# Scene Graph Research Notes

这份笔记只记录和当前 HLR 改造最相关的脉络，按“怎么生成图”和“图怎么用于规划”两条线整理。

## 1. 代表性工作

### Hydra
- Paper: [Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization](https://arxiv.org/abs/2201.13360)
- 核心做法：从传感器流在线增量构图，边探索边生成 layered 3D scene graph。
- 对我们最有用的点：
  - scene graph 不是一次性静态产物，而是随探索持续增长
  - 图要和几何地图、拓扑 places、rooms 共同优化
  - 适合支撑“动态图 + graph rewrite”路线

### HOV-SG
- Paper: [Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation](https://arxiv.org/abs/2403.17846)
- 核心做法：把开放词汇语义和层次 3D scene graph 结合，服务大空间导航。
- 对我们最有用的点：
  - 大图需要层次压缩和按需展开
  - zone / room / object 的分层表示非常关键
  - 适合指导 canonical IR 的 node type 设计

### SayPlan
- Project: [SayPlan](https://sayplan.github.io/)
- 核心做法：用 3D scene graph 做 subgraph semantic search，再由 LLM + classical planner + simulator 做 iterative replanning。
- 对我们最有用的点：
  - 不喂全图，先做 task-relevant subgraph retrieval
  - scene graph simulator 要能验证计划是否可执行
  - 和我们后续的“子图检索 + 分层规划 + 执行校验”几乎同方向

### Taskography
- Paper: [Taskography: Evaluating robot task planning over large 3D scene graphs](https://openreview.net/forum?id=nWLt35BU1z_)
- 核心做法：直接把 large 3D scene graph 当规划 benchmark。
- 对我们最有用的点：
  - benchmark 不只看图长什么样，还要控制任务难度和图规模
  - 需要明确 graph-induced planning cost 和 executability

### MomaGraph
- Paper: [MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Models for Embodied Task Planning](https://openreview.net/forum?id=3eTr9dGwJv)
- 核心做法：用 state-aware unified scene graph 表示任务相关环境状态，并配套数据集和 benchmark。
- 对我们最有用的点：
  - 统一 schema 非常重要
  - 静态结构和动态状态必须放在一张图里
  - scene graph 本身可以成为 planning supervision 的核心载体

### Embodied Semantic Scene Graph Generation
- Paper: [Embodied Semantic Scene Graph Generation](https://openreview.net/forum?id=FCoh4OLZ1Gg)
- 核心做法：agent 自主探索环境，增量构建 scene graph。
- 对我们最有用的点：
  - “生成场景图”本身可以被定义成 embodied task
  - 图构建和探索策略是耦合的

### Open-Vocabulary Functional 3D Scene Graphs
- Paper: [Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](https://arxiv.org/abs/2503.19199)
- 核心做法：不仅建空间关系图，还显式建功能关系图。
- 对我们最有用的点：
  - 规划图不能只保留 on / in / neighbor
  - 还需要 controls / functional / interaction 关系

## 2. 目前主流做法可以粗分为四类

### A. 手工 / 模板 / 符号生成
- 典型特点：
  - schema 清楚
  - 容易配 precondition / effect
  - 但容易缺几何 realism 和多样性
- 对应我们当前 HLR 的基础形态

### B. 感知驱动的增量建图
- 典型特点：
  - 从 RGB-D / SLAM / exploration 里动态长出图
  - 强调在线更新和不确定性
- 代表：Hydra, Embodied Semantic Scene Graph Generation

### C. 开放词汇 / VLM 驱动的 3D scene graph
- 典型特点：
  - object / room / function 的语义来自 foundation models
  - 更强调开放类别和语言 grounding
- 代表：HOV-SG, Open-Vocabulary Functional 3D Scene Graphs

### D. 面向任务规划的状态感知图
- 典型特点：
  - 图是为 planning / manipulation 服务，不只是表示环境
  - 节点和边要带 affordance、state、precondition、effect
- 代表：SayPlan, Taskography, MomaGraph

## 3. 对 HLR 五个目标的直接启发

### 1. Canonical graph IR
- 参考：MomaGraph, HOV-SG
- 结论：
  - 一定要统一成 typed property graph
  - 静态结构和动态状态必须共存

### 2. 可控生成
- 参考：Taskography
- 结论：
  - 生成器不能只是随机拼装
  - 要显式控制图规模、拓扑复杂度、目标计划长度

### 3. 稳定语义和稳定 ID
- 参考：Hydra, HOV-SG
- 结论：
  - 图节点的类型本体和关系本体要先收口
  - ID 必须 deterministic，不然难复现实验

### 4. 静态图和动态图转移打通
- 参考：Hydra, SayPlan, MomaGraph
- 结论：
  - action 需要图级 precondition / effect
  - scene graph simulator 是 benchmark 的核心组成部分

### 5. Graph-level 验证
- 参考：SayPlan, Taskography
- 结论：
  - 必须验证连通性、可达性、任务可满足性、执行合法性
  - 不能只做字段检查
