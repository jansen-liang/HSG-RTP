# 🚀 项目代号：H-Sim (Hierarchical Simulation & Reasoning Framework) 核心逻辑大纲

### Part 1. 数据引擎：程序化多层场景图生成器 (The World)

**核心逻辑：** 解决现有数据集“平层化”、“简单化”的痛点，构建支持跨楼层、长序列任务的复杂环境。

* **1.1 基础资源层 (Confirmed)**
* **物体库 (Object Library):** `/opt/data/.../objects` (按房间分类的元数据或模型)。
Base Assets: 自动导入 BEHAVIOR-1K 物体（含物理属性、交互逻辑）。

Custom Assets: 自动导入 Objaverse 模型（针对医用/工业设备）。

Logic Mapping: 应用我之前说的映射表（如 Turnstile -> Inherits Door logic）。
* **规则生成器 (Rule Generator):** 利用 LLM 基于 Prompt (常识、物理、空间) 生成房间内的布局规则。


* **1.2 场景图构建层 (Confirmed)**
* **层次化结构:** Building -> Floor -> Room -> Object。
* **关键连接点:** 电梯 (Elevator)、楼梯 (Stairs) 作为跨层连接边；门 (Door) 作为房间连接边。
* **场景类型:** 
| 序号 | 场景名称 (Type) | 核心特征 (Why this?) | 典型跨层逻辑 |
| --- | --- | --- | --- |
| **1** | **大型综合超市** (Hypermarket) | 货架密集，物品多。仓库在楼上，卖场在楼下。 | 补货任务：从二楼仓库取货 -> 货梯 -> 一楼货架。 |
| **2** | **豪华酒店** (Luxury Hotel) | 严格的服务流程，多房间，长走廊。 | 送餐任务：一楼厨房取餐 -> 客梯 -> 15楼客房 -> 敲门交接。 |
| **3** | **综合三甲医院** (General Hospital) | 区域权限复杂（污染/清洁区），设备多。 | 样本转运：3楼病房取血样 -> 专用电梯 -> 1楼化验室。 |
| **4** | **高层办公楼** (Office Building) | 结构重复但微小差异，涉及门禁多。 | 文件递送：前台取快递 -> 刷卡进闸机 -> 电梯 -> 20楼经理办公室。 |
| **5** | **高档住宅小区** (Residential) | 包含室外（花园）到室内，涉及公共与私人空间。 | 倒垃圾/取外卖：从12楼家里出来 -> 电梯 -> 1楼大堂 -> 室外垃圾桶。 |
| **6** | **大学图书馆/教学楼** (University) | 空间极大，静音要求，书籍/设备归位。 | 归还设备：从5楼教室拿投影仪 -> 楼梯/电梯 -> 1楼设备科。 |



* **🤔 待讨论/需细化 (TBD):**
* **物体属性的具体化:** 既然要做状态变更，物体库必须包含 `States` 定义（例如：微波炉必须有 `isOpen`, `isRunning`, `contains` 属性）。数据集中是否有这些元数据？如果没有，需要用脚本批量挂载。
* **图的表示形式:** 是用 NetworkX 生成纯拓扑图，还是包含 3D 坐标 (x,y,z) 的几何图？（建议：包含粗略坐标，否则无法计算导航距离）。



---

### Part 2. 任务引擎：基于符号规划的数据生成 (The Data)

**核心逻辑：** 利用 PDDL + 传统规划器生成“完美示范”数据，解决长序列逻辑的 Ground Truth 获取问题。

* **2.1 动作空间定义 (Action Space) (Confirmed)**
* **分层动作:**
* **High-Level (Subtask):** 导航、整理、制作、清洁、传送。
* **Low-Level (Atomic):**
**移动类:** `Walk` (走), `Run` (跑 - 紧急任务用), `Turn` (转身/转向).
* **手部操作类:** `Grasp` (抓), `Release` (放), `Twist` (拧 - 用于瓶盖/旋钮).
* **力交互类:** `Push` (推), `Pull` (拉).
* **交互触发类:** `Press` (按 - 按钮/开关), `Swipe` (刷 - 卡/身份证).
* **被动/特殊类:** `Wait` (等待), `Handover` (递给人类).




* **2.2 状态机与执行器 (Executor) (Confirmed)**
* **功能:** 接收 `(Action, Object)`，检查 `Precondition`，更新 `Scene Graph` (拓扑关系 + 节点状态)。
* **交互逻辑:** 门 (推拉 -> 开关)，灯 (按 -> 亮灭)，微波炉 (按 -> 运行)，物体 (抓放 -> 边连接改变)。


* **2.3 数据生成流水线 (Confirmed)**
* 定义 PDDL Domain (动作定义) 和 Problem (初始/目标状态)。
* 使用 FD (Fast Downward) 规划器求解得到 Action Sequence。
* 格式化为: `Instruction -> [Subtask 1 -> Actions...] -> [Subtask 2 -> Actions...]`。


* **🤔 待讨论/需细化 (TBD):**
* **子任务模版 (Templates) 的多样性:** 需要穷举多少种模版才能覆盖长序列任务？（建议：设计 10-15 个核心模版，通过排列组合生成无限任务）。
* **失败处理:** PDDL 生成的是完美路径，但训练数据是否需要包含“失败-重试”的负样本？(SFT阶段可能不需要，RL阶段需要)。



---

### Part 3. 模型架构：双分支分层推理机 (The Brain)

**核心逻辑:** 结合结构化图编码与大语言模型，实现分层推理。

* **3.1 编码器设计 (Confirmed)**
* **Text Encoder:** 处理 Instruction。
* **Graph Encoder:** 双分支分层场景图编码器 (你的 HLR 遗产)。


* **3.2 推理主干 (Backbone) (Confirmed)**
* Base Model: Qwen (或其他开源 LLM)。
* 目标: 降低参数量，高效推理。


* **3.3 分层输出机制 (Gate/MoE) (Drafting)**
* **方案:** 引入 Gate Attention 或 MoE 模块。
* **机制:** 动态判断当前 Token 是生成 `Subtask` (规划模式) 还是 `Action` (执行模式)。


* **🤔 待讨论/需细化 (TBD):**
* **上下文长度 (Context Length):** 如果场景图很大（多层大楼），全部 Token 化通过 Graph Encoder 是否会撑爆显存？是否需要设计“动态子图提取” (只看当前房间+邻居)？
* **模态融合:** 图特征 (Graph Embedding) 是通过 Cross-Attention 注入 LLM，还是直接作为 Soft Prompt 拼接到 Input Embedding 前面？（后者工程量小，前者效果可能好）。



---

### Part 4. 训练策略：SFT + Online RL (The Learning)

**核心逻辑:** 先教规矩 (SFT)，再教策略 (RL)。

* **4.1 第一阶段: SFT (Supervised Fine-Tuning) (Confirmed)**
* 数据来源: PDDL 生成的完美序列。
* 目标: 学习输出格式 (JSON/Special Tokens)、基础逻辑、图文对应关系。


* **4.2 第二阶段: Online RL / RLHF (Confirmed)**
* **环境:** 你的 `executor.py` 作为模拟器。
* **流程:** 模型输出 Action -> Executor 更新图 -> 返回新 State -> 模型继续。
* **评价:** 奖励函数 (Reward Function) + Value Model。


* **🤔 待讨论/需细化 (TBD):**
* **奖励函数设计 (Reward Engineering):**
* Step Reward: 动作合法性 (例如：没开门就想走 -> -1)。
* Subtask Reward: 完成子目标 (例如：成功拿到物体 -> +5)。
* Sparse Reward: 最终任务完成 (例如：把药送到了 -> +100)。


* **Value Model 的训练:** 是单独训练一个 Critic 网络，还是复用 Actor 网络加一个 Head？



---

### Part 5. 总结与实验对比 (Experiment)

* **Baseline:** React, CoT-Planning, 纯文本 LLM (无图编码器), 传统的 TAMP 方法。
* **Metrics:** 任务成功率 (SR), 路径最优率 (SPL), 子任务完成率, 跨楼层推理准确率。

---

### 💡 接下来最优先需要确认的技术点 (Priority Checklist)

建议按照以下顺序进行技术验证，这也是我们接下来对话的重点：

1. **[High Priority] 状态定义的完备性:**
* 我们要先列一个表，确定到底支持哪些 `Atomic Actions` 和哪些 `Object States`。如果这个没定死，后面的 PDDL 和 Executor 都没法写。


2. **[Medium Priority] PDDL 与 Executor 的接口:**
* PDDL 求解出来的动作序列，如何无缝转化为 Executor 能读懂的指令？两者需要共用一套命名空间 (Namespace)。


3. **[Hard] Gate/MoE 的具体实现:**
* 是想自己魔改 Transformer 结构（加 Gate 层），还是仅在 Prompt 层面做控制？（魔改结构工作量大，但创新性强）。

