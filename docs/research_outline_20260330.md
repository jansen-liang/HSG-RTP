# HLR 研究定位与下一阶段方案

日期：2026-03-30

这份文档是在阶段性总结基础上，进一步把当前工作整理成更接近论文或开题材料的形式。重点回答三个问题：

1. 我们到底想解决什么问题。
2. 相比已有工作，我们的创新点可能落在哪里。
3. 下一阶段最值得优先推进的实验和系统改造是什么。

## 1. 一句话研究问题

**能否把任务导向的动态图表示、可执行的图规则系统、以及严格的图级评测结合起来，构建一个真正服务于长时程具身任务规划的 graph world model？**

更具体地说，我们关心的不是“scene graph 能不能生成出来”，而是：

- scene graph 能不能作为任务规划的中间状态层
- scene graph 能不能随着动作和观测动态更新
- scene graph 能不能支撑可执行性验证、重规划与 benchmark

## 2. 当前判断

当前文献已经证明了三件事：

1. scene graph 适合作为具身任务的结构化环境表示
2. 任务相关子图比“整图平铺”更适合作为规划输入
3. 动态状态和功能关系对 manipulation planning 很关键

但目前还没有哪条路线把下面三层完整打通：

1. **统一图表示**
2. **图级动作转移与执行验证**
3. **面向 planning 的 benchmark protocol**

这正是 HLR 现在最有机会补的位置。

## 3. 和已有工作的关系

### 3.1 MomaGraph 解决了什么

MomaGraph 最强的地方在于：

- 用统一的 state-aware scene graph 表示任务相关环境状态
- 强调空间关系、功能关系和部件级交互元素的统一
- 引入任务导向 scene graph 数据集与 benchmark
- 用 `Graph-then-Plan` 证明先建任务图再规划是有价值的

它非常适合支撑我们的以下动机：

- scene graph 不应只是静态快照
- scene graph 不应只保留 `on / in / neighbor`
- 任务相关子图和动态图更新是必要的
- benchmark 不能只看最终动作文本

### 3.2 MomaGraph 没有完全替我们解决什么

MomaGraph 并没有完全替我们解决下面这些问题：

- 多套 schema 如何统一到一个长期可维护的 canonical IR
- graph executor 如何定义动作 precondition / effect
- scene graph 如何成为严格的执行状态，而不只是中间表示
- benchmark 如何围绕 executability、replanning、cost ratio 建立协议

这意味着如果我们继续沿 HLR 路线推进，最自然的 claim 不是“我们也能做任务图”，而是：

**我们把任务导向图进一步做成了可执行、可验证、可 benchmark 的 graph world model。**

### 3.3 HLR 现在的独特位置

和 MomaGraph 相比，HLR 当前更强的不是建图，而是以下两点：

- 已有一条比较完整的 symbolic planning / sample building / state synchronization 链路
- 已经有面向长时程任务的层次化训练与推理形态

换句话说：

- MomaGraph 更像“怎么得到对任务有用的图”
- HLR 更适合补“这张图怎么真正驱动规划、执行和评测”

## 4. 我们建议的核心主张

### 4.1 主张一：scene graph 应该被重新定义为 graph world model

在 HLR 的后续工作里，scene graph 不再只是：

- 环境描述
- 输入序列化
- 训练辅助特征

而应当同时承担三种角色：

1. **状态表示层**
2. **动作转移层**
3. **评测与验证层**

只有这三者统一，graph-based agent 才不容易退化成“把图翻译成 prompt 再让 LLM 猜动作”。

### 4.2 主张二：任务条件子图检索是关键，而不是全图编码

HLR 当前的全局图/局部图裁剪还是固定逻辑，不是任务条件的。

下一阶段更重要的问题是：

- 指令到底需要哪些节点和边
- 哪些房间和对象与当前子任务相关
- 何时需要补充更大上下文

因此最该做的不是一上来换更强图编码器，而是先引入：

**instruction-conditioned subgraph retriever**

### 4.3 主张三：动态图更新应该落到图规则和边状态上

如果只维护 object state，而不维护 relation state，很多真正困难的任务关系其实捕捉不到。

例如：

- 哪个按钮控制哪盏灯
- 哪个旋钮控制哪个灶眼
- 哪个 handle 对应哪扇 window / cabinet door

因此下一阶段动态图更新不应只改 node states，还要改：

- edge existence
- edge confidence
- hypothesis / confirmed / rejected 边状态

## 5. 建议的系统框架

建议把 HLR 的下一阶段系统写成下面这条主链：

`Canonical Graph IR -> Task-conditioned Subgraph Retrieval -> Hierarchical Planner -> Graph Rewrite Executor -> Graph-level Evaluation`

### 5.1 Canonical Graph IR

作用：

- 统一 legacy / editor / OOP schema
- 统一 node / edge / state / relation ontology
- 为 planner、executor、validator 提供共同接口

当前状态：

- 已完成第一版地基

### 5.2 Task-conditioned Subgraph Retrieval

作用：

- 替换现在固定 global/local 裁图
- 根据 instruction、agent location、pending plan 检索 task-relevant subgraph

优先级：

- 很高

原因：

- 它直接决定 scene graph 是否真正“面向任务”

### 5.3 Hierarchical Planner

作用：

- 在 global 子图上做 room / zone 级 planning
- 在 local 子图上做 object / component 级 planning

当前状态：

- HLR 已有雏形

下一步重点：

- 不再直接吃 legacy scene dict，而是吃 canonical graph 或 retrieved subgraph

### 5.4 Graph Rewrite Executor

作用：

- 统一动作 precondition / effect
- 执行过程中更新 graph state 与 relation state
- 支持 legality check 与 replanning

当前状态：

- 已有初版规则骨架

下一步重点：

- 让 `state_manager` 接入这套规则

### 5.5 Graph-level Evaluation

作用：

- 验证图是否有效
- 验证动作是否可执行
- 统计成功率、代价、恢复率

当前状态：

- 已有初版 validator

下一步重点：

- 建立 benchmark protocol 和对比基线

## 6. 最可能的创新点

如果把论文创新点压缩成几条，我认为最自然的是下面四条。

### 6.1 统一的 canonical scene graph IR

不是单纯定义一种数据结构，而是让：

- 多源场景数据
- 动作规则
- 执行器
- 评测器

都围绕同一套 IR 工作。

### 6.2 任务条件子图检索

和固定 global/local 划分相比，任务条件子图检索更贴近 planning 真实需求，也更符合 SayPlan、MomaGraph 这类工作的经验。

### 6.3 可执行的 graph rewrite executor

这一点非常关键，因为它会把“场景图研究”从表示学习推进到：

- action legality
- state transition
- replanning trigger
- benchmark grounding

### 6.4 面向规划的 graph benchmark protocol

当前很多 scene graph 工作把 benchmark 重点放在：

- 图预测准确率
- 局部关系识别
- action sequence matching

而我们可以补充：

- executability
- planning cost
- replanning recovery
- hidden relation discovery

## 7. 下一阶段实验设计

### 7.1 实验主问题

建议下一阶段围绕下面三个问题组织实验：

1. 任务条件子图检索是否比固定裁图更有效
2. graph rewrite executor 是否能提高可执行性
3. 动态关系更新是否能提高重规划成功率

### 7.2 建议 baseline

#### Baseline A：当前 HLR

- 固定 global/local 裁图
- legacy dict state manager
- 原始 planner 与训练链

#### Baseline B：text-only scene serialization

- 不使用结构化图编码
- 只把 scene 文本化后喂给模型

#### Baseline C：oracle full graph

- 使用完整 canonical graph
- 不做 task-conditioned retrieval

#### Baseline D：retrieved subgraph

- 使用 instruction-conditioned subgraph
- 其余模块尽量与 C 对齐

#### Baseline E：retrieved subgraph + graph rewrite executor

- 用于验证执行器本身的增益

### 7.3 建议指标

#### 图层指标

- graph validity
- relation validity
- inverse consistency
- task subgraph recall

#### 规划层指标

- success rate
- executability
- invalid action rate
- plan cost ratio

#### 动态层指标

- replanning recovery rate
- hidden functional relation identification
- post-action graph consistency

## 8. 最值得先做的三个里程碑

### Milestone 1：把 executor 迁到 canonical graph

目标：

- 用 graph rewrite rules 替换旧 dict state update

完成标准：

- 至少 `goto / pick / place / scan / press / open / close` 接入新执行器
- 旧样本构建流程可以跑通

### Milestone 2：做 task-conditioned subgraph retrieval

目标：

- 替换当前固定 global/local 裁图

完成标准：

- 给出规则版 retriever
- 支持对比 full graph / fixed view / retrieved subgraph

### Milestone 3：建立 benchmark protocol

目标：

- 从“像不像 GT”推进到“能不能执行、执行得好不好”

完成标准：

- 增加 executability、cost ratio、replanning recovery 等指标
- 建立一组最小可复现实验

## 9. 当前最现实的风险

### 9.1 改动会牵动主链

如果直接把训练和推理主链全部改到 canonical graph 上，短期风险较大。

更稳妥的策略是：

1. 先迁 executor
2. 再迁 retriever
3. 最后迁 encoder 输入

### 9.2 缺少任务相关子图监督

这会影响 retrieval 的训练方案。

短期建议：

- 先用规则构伪标签
- 用 planner 轨迹做弱监督

### 9.3 专家轨迹不够“硬”

当前启发式 planner 和随机电梯按钮状态会影响 benchmark 的说服力。

短期建议：

- 先把随机成分去掉
- 再决定是否引入 classical planner 或 search-based expert

## 10. 当前推荐的研究叙事

如果要把当前工作压成一句较清晰的研究叙事，我建议这样写：

**现有 scene graph 工作已经证明了任务导向图表示的重要性，但仍缺少一个统一、可执行、可验证的 graph world model。我们在 HLR 上补齐这一层，使 scene graph 不仅是环境描述，更是规划、执行与评测共享的状态接口。**

再压缩一点，就是：

**从 task-oriented scene graph，走向 executable graph world model。**
