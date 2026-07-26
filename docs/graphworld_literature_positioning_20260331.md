# GraphWorld-24/7 文献定位与研究差异化

日期：2026-03-31

本文档围绕我们当前想推进的方向整理文献，不再以“scene graph 能不能表示环境”为主问题，而是把目标明确为：

**构建一个适合长期 RL 的、自演化的、社会-时间感知的图世界模型（persistent self-evolving socio-temporal graph world model），并在此基础上建立 always-on service robot benchmark / environment。**

核心关注两个问题：

1. 现有 graph-agent / scene graph / world model 工作做到哪一步了？
2. 现有任务规划工作，为什么还没有真正走到“无限循环、自演化环境中的长期服务决策”？

## 1. 结论先行

现有工作已经分别证明了下面几件事：

- 图表示对于机器人任务规划是有价值的。
- 动态 scene graph 比静态 scene graph 更贴近真实环境。
- task-relevant subgraph retrieval 比直接喂全图更可扩展。
- 长时程任务需要 closed-loop feedback、memory 和 replanning。
- 世界模型对长期决策与策略学习非常重要。

但到目前为止，还没有哪条主线真正把下面这些要素统一到同一个 benchmark / environment 里：

- **分层拓扑图**
- **功能与部件关系**
- **动作导致的 graph rewrite**
- **外生随机事件**
- **时间流逝导致的状态变化**
- **长期循环任务流**
- **RL-compatible reward and evaluation**
- **持续运行而非一次性 episode 的服务目标**

这正是我们要切入的位置。

## 2. 图相关工作：四条主线

### 2.1 方向一：层次化 / 拓扑化 / 可定位的 scene graph

这条线的重点是：如何把大空间组织成机器人可用的层次拓扑结构。

代表工作：

- [S-Graphs+: Real-time Localization and Mapping leveraging Hierarchical Representations](https://arxiv.org/abs/2212.11770)
- [S-Graphs 2.0 -- A Hierarchical-Semantic Optimization and Loop Closure for SLAM](https://arxiv.org/abs/2502.18044)
- [Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization](https://arxiv.org/abs/2201.13360)
- [Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation (HOV-SG)](https://arxiv.org/abs/2403.17846)

这些工作的共同特点：

- 强调 floor / room / object 的层次组织
- 强调与 SLAM / localization / navigation 的结合
- 强调大规模空间的拓扑压缩

局限：

- 更偏“地图”和“层次表示”
- 通常不直接建模长期任务流
- 很少把社会规则、时间规则和外生事件写进图状态

对我们的启发：

- 我们应该继承其 **层次拓扑图能力**
- 但要把图从“定位图 / 语义地图”升级成 **graph world model**

### 2.2 方向二：动态 scene graph、在线更新与记忆

这条线的重点是：图不是静态快照，而是随着观测和环境变化不断更新。

代表工作：

- [Modeling Dynamic Environments with Scene Graph Memory](https://arxiv.org/abs/2305.17537)
- [Embodied Semantic Scene Graph Generation](https://proceedings.mlr.press/v164/li22e.html)
- [Hi-Dyna Graph: Hierarchical Dynamic Scene Graph for Robotic Autonomy in Human-Centric Environments](https://arxiv.org/abs/2506.00083)
- [OST-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding](https://arxiv.org/abs/2507.07984)

这些工作的共同特点：

- 强调 partial observability
- 强调 online update
- 强调图记忆和时序理解
- 强调 agent 边探索边构图

局限：

- 更新通常主要由“观测增长”驱动
- 较少引入 **任务持续到达** 和 **服务循环**
- 较少形式化“世界自己在变化”的长期任务层

对我们的启发：

- 我们必须把 **动态图更新** 作为环境内核，而不是附属模块
- 但更新来源要更完整：
  - agent action
  - exogenous event
  - temporal decay / temporal activation

### 2.3 方向三：功能型 / 任务导向 scene graph

这条线的重点是：图不只表示空间位置，还要表示“谁控制谁、什么可操作、和任务有关的关系是什么”。

代表工作：

- [Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](https://arxiv.org/abs/2503.19199)
- [SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://sayplan.github.io/)
- [EmbodiedRAG: Dynamic 3D Scene Graph Retrieval for Efficient and Scalable Robot Task Planning](https://arxiv.org/abs/2410.23968)
- [MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Model for Embodied Task Planning](https://arxiv.org/abs/2512.16909)

这些工作的共同特点：

- 图中引入 affordance / predicate / object state / functional relation
- 强调 task-relevant subgraph
- 强调 Graph-then-Plan 或 graph-grounded planning

局限：

- 多数工作仍然把图视为“某个任务时刻的中间表示”
- 即便动态图更新存在，也往往局限于当前任务上下文
- 很少将图扩展为 **长期运行服务世界的统一状态接口**

对我们的启发：

- 任务相关子图检索必须成为 GraphWorld-24/7 的核心能力
- 但我们要进一步引入：
  - recurring jobs
  - delayed effects
  - world health
  - social constraints

### 2.4 方向四：world model / 自演化任务世界

这条线与我们的方向最接近，但目前仍没有完全命中我们要做的点。

代表工作：

- [Learning 3D Persistent Embodied World Models](https://arxiv.org/abs/2505.05495)
- [Embodied AI Agents: Modeling the World](https://arxiv.org/abs/2506.22355)
- [Affordance-Graphed Task Worlds: Self-Evolving Task Generation for Scalable Embodied Learning](https://arxiv.org/abs/2602.12065)
- [AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation](https://arxiv.org/abs/2508.07770)

这些工作的共同特点：

- 明确把“世界模型”或“任务世界”提到更核心的位置
- 强调 memory、simulation、policy learning、self-evolution 或 task generation

局限：

- `Learning 3D Persistent Embodied World Models` 更偏“预测式 3D persistent WM”，不是服务机器人 benchmark
- `Embodied AI Agents: Modeling the World` 是宏观研究纲领，不是具体 graph-world benchmark
- `AGT-World` 已经开始触及 self-evolution，但更偏任务生成与 policy self-improvement
- `AgentWorld` 很强于 interactive simulation，但其核心仍偏 mobile manipulation platform，而不是 socio-temporal persistent service world

对我们的启发：

- 我们的标题和概念应该明确往 **graph world model** 上走
- 但与上述工作不同，我们的世界模型应当是：
  - **持续运行**
  - **受社会规则与物理规则共同约束**
  - **具有长期服务目标**
  - **适合 RL 训练和长期规划评测**

## 3. 现有任务规划工作：做到了什么，没做到什么

### 3.1 LLM + affordance / skill grounding

代表工作：

- [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)](https://arxiv.org/abs/2204.01691)
- [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608)
- [Code as Policies: Language Model Programs for Embodied Control](https://arxiv.org/abs/2209.07753)
- [LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models](https://arxiv.org/abs/2212.04088)

这些工作的贡献：

- 证明 LLM 可以做高层任务拆解
- 证明 affordance / skill grounding 很关键
- 证明执行反馈能提升闭环表现

但这些工作大多仍然是：

- 给定一个任务
- 在一个有限 episode 内完成
- 世界主要围绕当前任务服务

它们通常不要求 agent 在一个 **持续运行的世界** 中长期维持系统稳定。

### 3.2 graph-conditioned planning

代表工作：

- [TASKOGRAPHY: Evaluating robot task planning over large 3D scene graphs](https://arxiv.org/abs/2207.05006)
- [SayPlan](https://arxiv.org/abs/2307.06135)
- [Optimal Scene Graph Planning with Large Language Model Guidance](https://arxiv.org/abs/2309.09182)
- [EmbodiedRAG](https://arxiv.org/abs/2410.23968)
- [MomaGraph](https://arxiv.org/abs/2512.16909)

这些工作的贡献：

- 证明 scene graph 可以支撑 task planning
- 证明 subgraph retrieval 和 executable feedback 很重要
- 证明 task-oriented graph 比纯视觉/纯文本更稳定

但它们大多还没有进入下面这个 setting：

- 任务不断到达
- 世界自己变化
- 有周期任务、随机任务、高优先级任务
- agent 需要在长期服务效用上做权衡

也就是说，它们还不是 **always-on service planning**。

### 3.3 长时程 embodied planning

代表工作：

- [ReLEP: A Novel Framework for Real-world Long-horizon Embodied Planning](https://arxiv.org/abs/2409.15658)
- [LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks](https://arxiv.org/abs/2506.00411)
- [VestaBench: An Embodied Benchmark for Safe Long-Horizon Planning Under Multi-Constraint and Adversarial Settings](https://aclanthology.org/2025.emnlp-industry.149/)
- [λ: A Benchmark for Data-Efficiency in Long-Horizon Indoor Mobile Manipulation Robotics](https://lambdabenchmark.github.io/)

这些工作的贡献：

- 开始认真讨论 long-horizon tasks
- 引入多约束、安全性、数据效率和真实机器人验证

但它们通常仍然有一个共同限制：

- **episode 是有限的**
- **任务集合是有限的**
- **世界并不持续自演化**

它们测试的是“长任务”，不是“长期运行”。

### 3.4 open-ended / continual embodied agents

代表工作：

- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- [AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents](https://arxiv.org/abs/2405.14573)
- [Towards Adaptive, Continual Embodied Agents](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-220.pdf)
- [GOAT-Bench: A Benchmark for Multi-Modal Lifelong Navigation](https://arxiv.org/abs/2404.06609)
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)
- [C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](https://arxiv.org/abs/2510.20685)

这些工作的贡献：

- open-ended、continual、lifelong 这些概念已经被讨论
- 表明“固定 episode benchmark”正在成为瓶颈

但它们的不足也很清楚：

- 很多工作不在真实服务场景中
- 很多工作不使用 graph world model 作为统一状态接口
- 很多工作仍偏技能积累、导航终身学习、开放探索，而不是长期服务系统维护

换句话说，它们提供了“长期学习”的动机，但还没有给出一个：

**面向服务机器人、以 graph world model 为核心、带物理与社会规则、支持长期 RL 的 persistent benchmark。**

## 4. 我们和现有工作的关键差异

可以把差异压成一句话：

**我们不是在已有静态 scene graph 上做一个更强 planner，而是在构建一个持续自演化的图世界模型，让机器人在其中长期学习“下一步该做什么、不该做什么”。**

更具体地说，我们要强调五个差异：

### 4.1 从“scene graph”到“graph world model”

现有很多工作仍然把图当成：

- 当前环境快照
- 当前任务的中间表示
- 输入 LLM 的结构化上下文

而我们要把图提升为：

- 持续状态接口
- 动作转移接口
- 反馈接口
- 评测接口

### 4.2 从“长任务”到“长期运行”

现有 long-horizon planning 多数是在一个长任务里走很多步。

而我们要研究的是：

- 世界 24/7 持续运行
- agent 不断接到新任务
- 任务之间彼此竞争资源与时间
- world state 不因单个任务结束而 reset

### 4.3 从“agent 触发更新”到“世界自演化”

现有动态图更新很多仍偏：

- robot observe -> update graph
- robot act -> update graph

而我们还要加入：

- 外生事件
- 周期事件
- 随时间发生的状态变化
- 社会规则触发的可达性变化

即：

**图的变化不只是 agent 引起的，而是世界自己在变。**

### 4.4 从“任务成功率”到“长期服务效用”

现有 benchmark 常测：

- success / fail
- action match
- subgoal accuracy

而我们应测：

- cumulative service utility
- high-priority response rate
- backlog control
- world health
- norm compliance
- replanning recovery

### 4.5 从“planning-only”到“RL-compatible environment”

现有大量工作仍把评测建立在：

- supervised plan generation
- heuristic execution
- finite-horizon evaluation

而我们希望环境同时支持：

- heuristic agent
- planner-based agent
- LLM agent
- RL scheduler / policy

并证明：

**在这种 persistent graph world 中，RL 的确能学到长期服务策略。**

## 5. 我们的引擎应该继承哪些“各家之长”

为了让 GraphWorld-24/7 真正站住，不是凭空造个新世界，而是要系统继承现有工作里已经被验证有效的优点。

### 5.1 从层次化图工作中继承

- floor / zone / room / object / component 分层
- 拓扑图与语义图结合
- 可在大空间中高效检索相关子图

来源：

- Hydra
- S-Graphs+
- HOV-SG

### 5.2 从 task-oriented graph 工作中继承

- affordance
- function / control relations
- task-relevant subgraph retrieval
- graph-grounded planning

来源：

- SayPlan
- EmbodiedRAG
- OpenFunGraph
- MomaGraph

### 5.3 从 closed-loop planning 工作中继承

- 计划不是一次性输出
- 执行失败后要重规划
- simulator / executor 必须提供反馈

来源：

- SayCan
- Inner Monologue
- SayPlan
- ReLEP

### 5.4 从 persistent world model 工作中继承

- persistent memory
- world state evolution
- planning through future state reasoning

来源：

- Learning 3D Persistent Embodied World Models
- Embodied AI Agents: Modeling the World

### 5.5 从 lifelong / continual learning 工作中继承

- 不 reset 的长期训练视角
- open-ended curriculum
- 长期知识保留

来源：

- Voyager
- LIBERO
- GOAT-Bench
- C-NAV

## 6. 我们的环境应具备的最小能力

基于以上文献，我们的引擎至少应支持以下能力，才能形成真正的研究差异：

### 6.1 状态层

- persistent graph state
- state-aware nodes
- relation-aware edges
- temporal validity / temporal cooldown / occupancy

### 6.2 更新层

- action-driven graph rewrite
- event-driven graph update
- time-driven graph evolution

### 6.3 任务层

- periodic jobs
- stochastic jobs
- urgent jobs
- background maintenance jobs

### 6.4 反馈层

- 动作成功 / 失败反馈
- graph consistency feedback
- world-health feedback
- task completion feedback

### 6.5 学习层

- RL reward
- delayed reward
- long-horizon evaluation
- policy / planner / LLM 多类型 agent 接口

## 7. 建议的论文定位

可以把论文定位收敛成下面这段话：

**现有图规划工作已经证明了 task-oriented scene graph 的价值，现有长期具身规划工作也开始关注 long-horizon decision-making，但二者大多仍停留在有限 episode 和当前任务上下文中。我们提出一个持续自演化的 graph world model benchmark，使机器人必须在受物理规则、社会规则和时间事件共同驱动的世界中长期运行，并通过 RL 或规划不断推理“下一步该做什么，不该做什么”。**

## 8. 当前最推荐的标题与关键词

### 标题

**GraphWorld-24/7: A Persistent Self-Evolving Graph World Model for Always-On Service Robot Learning**

### 核心关键词

- persistent graph world model
- self-evolving environment
- socio-temporal graph
- always-on service robot
- long-horizon reinforcement learning
- graph-based continual planning

## 9. 下一步建议

如果继续往下推进，最优先的不是再搜更多 paper，而是开始把上面的“差异化”压成可实现的环境定义。建议紧接着做三件事：

1. 定义 GraphWorld-24/7 的正式状态空间  
   包括节点类型、边类型、temporal fields、job schema、reward schema。

2. 定义环境更新方程  
   形式化 `G_{t+1} = f(G_t, a_t, e_t, \Delta t)`，明确 action update、event update、time update。

3. 定义 benchmark protocol  
   明确：
   - episode 长度
   - 任务到达机制
   - 长期服务指标
   - baseline 类型

只有这三件事开始落地，这篇文章才会真正从“很好的想法”变成“非常强的研究问题”。  
