# TongTest 的任务生成机制及其对 HLR 方法的启发

## 1. TongTest 中任务是如何生成的

TongTest 的任务生成不是从固定任务标签或人工模板开始的，而是从动态具身环境中的状态空间开始。文章首先提出 DEPSI 环境，即 dynamic embodied physical and social interactions，用来覆盖智能体在真实生活中会遇到的物理交互、社会交互、环境变化和人机反馈。在这个框架下，任务被定义为环境状态之间的转移，而不是一个静态类别。

具体来说，TongTest 将任务形式化为：

```text
T = (phi_initial, phi_target)
```

其中 `phi_initial` 表示满足初始条件的一组环境状态，`phi_target` 表示满足目标条件的一组环境状态。由于 DEPSI 环境具有动态性和复杂性，文章没有要求每次任务都从完全相同的状态开始或结束，而是用“等价状态集合”来定义任务的起点和终点。例如，只要杯子处于“空”的状态，就可以属于某个初始状态集合；只要杯子处于“装满水”的状态，就可以属于目标状态集合。因此，一个“倒水”任务可以被理解为从 `cup.empty = true` 到 `cup.filled = true` 的状态转移。

为了表示这些状态，TongTest 使用 compositional graphical model，也就是 parse graph，作为基本知识表示形式。文章强调 parse graph 用来解析任意场景中的三类关系：

- 空间关系：物体、场所、智能体之间的位置和拓扑关系；
- 时间关系：事件和动作在时间上的先后、持续和变化；
- 因果关系：动作、物理规律、社会约束与状态变化之间的因果链。

在 Fig. 2 中，文章进一步把 parse graph 展开为 S-PG、T-PG 和 C-PG，即 spatial parse graph、temporal parse graph 和 causal parse graph。任务可以从这些空间、时间、因果图中采样出来，并沿着图上的路径分解为子任务。

在 parse graph 的基础上，TongTest 定义了 fluent space。这里的 fluent 指随时间变化的变量或属性，例如物体位置、物体状态、智能体状态、社会关系、信念状态、任务相关对象的可用性等。parse graph 表达环境结构，fluent space 表达图中可以变化的属性。二者结合后，一个复杂环境中的所有可能场景配置都可以被表示为连续的 DEPSI 状态空间。

因此，TongTest 的任务生成过程可以概括为：

```text
DEPSI environment
-> parse graph representation
-> fluent space over dynamic attributes
-> sample initial and target state configurations
-> define task as state transition
-> decompose task into subtasks through graph sampling
```

这种机制使得任务生成具有组合性。系统可以通过组合不同的对象、物理状态、社会状态和动作生成大量任务。例如：

```text
cup.empty -> cup.filled
table.messy -> table.clean
object.position = unsafe -> object.position = safe
food.temperature = cold -> food.temperature = hot
human.need = unserved -> human.need = served
```

这些状态转移都可以成为任务。文章认为，传统 benchmark 的任务通常是有限且预定义的，而 TongTest 的目标是支持 infinite task generation，即通过在 DEPSI 的连续状态空间中采样不同配置，生成开放组合的任务。

另一个关键点是 self-driven task generation。TongTest 不仅讨论平台如何生成测试任务，也强调 AGI 应能在开放环境中主动生成任务。文章认为，真正的通用智能不应只等待人类给出细粒度指令，而应在观察环境后知道下一步需要做什么。例如，一个杯子危险地放在桌子边缘时，如果智能体主动把杯子移到安全位置，这种行为就体现了由安全价值驱动的自主任务生成。

因此，TongTest 中的任务生成包含两个层面：

- 平台层面的任务生成：测试平台基于 parse graph、fluent space 和 DEPSI 配置生成大规模任务，用于评估智能体的能力；
- 智能体层面的任务生成：智能体在没有明确人类目标时，根据观察、价值系统和因果理解主动发现任务并执行。

这两个层面都由文章提出的 value-causality-behavior chain 支撑。价值系统提供任务生成的内在驱动力，因果理解约束任务完成路径，行为序列则实现从初始状态到目标状态的转移。换言之，TongTest 的任务不是简单地从任务模板中随机抽取，而是由“图结构中的动态状态变化”和“价值驱动的目标选择”共同产生。

## 2. 如何用 TongTest 优化 HLR 的任务生成方法

HLR 当前的数据生成流程已经具备与 TongTest 对齐的基础。现有 pipeline 从层次场景图中读取房间、楼层、邻接关系、物体、物体状态和 affordance，再生成 `delivery`、`tidying`、`guidance` 三类任务，随后通过规则规划器生成可执行动作序列，并通过状态管理器更新场景图，最终形成 global/local streaming samples。这个流程已经包含“图表示”“状态变化”“任务分解”和“执行前状态监督”等关键成分。

但是，从 TongTest 的角度看，HLR 当前的任务生成仍然主要是 template-conditioned graph sampling：系统先选择一个人工定义的任务类型，再从图中采样物体、房间和路径。为了更紧密地结合 TongTest，可以将其升级为 graph-fluent task generation，即先从图状态本身发现任务需求，再生成目标状态和动作序列。

具体映射关系如下：

```text
TongTest parse graph
-> HLR hierarchical scene graph

TongTest fluent space
-> HLR object, relation, agent, and room states

TongTest task transition
-> HLR initial graph state to target graph state

TongTest subtask decomposition
-> HLR global plan plus local action sequence

TongTest value-driven task generation
-> HLR safety, cleanliness, service, accessibility, and efficiency triggers
```

在 HLR 中，层次场景图可以被视为面向机器人长程任务的 parse graph。楼层、房间、电梯、走廊和邻接边对应空间图；动作序列、执行历史和 streaming step 对应时间图；物体状态变化、设备方法、动作前置条件和执行结果对应因果图。物体和智能体的动态属性则构成 HLR 的 fluent space，例如：

```text
agent.position
agent.inventory
object.room
object.relation.on / relation.in
object.temperature
object.wetness
object.cleanliness
object.availability
container.open_state
device.supported_methods
room.accessibility
```

基于这个表示，HLR 的任务生成可以从“任务类型优先”改为“状态转移优先”。系统首先扫描 scene graph 中的 fluents，检测是否存在需要改变的状态，再为这些状态生成目标配置。例如：

```text
dirty(object) -> clean(object)
wet(towel) -> dry(towel)
cold(food) -> hot(food)
object.on = unsafe_surface -> object.on = safe_surface
package.at = lobby -> package.at = target_room
trash.on = table -> trash.in = bin
```

这些状态转移可以进一步被组织为价值触发器。参考 TongTest 的 value-driven task generation，HLR 可以设计一组与机器人服务场景相关的价值维度：

- 安全价值：危险物体、边缘物体、阻塞通道、尖锐物品需要被转移到安全位置；
- 清洁价值：脏污、潮湿、杂乱或垃圾状态触发清洁和整理任务；
- 服务价值：食物、包裹、文件、药品等对象根据场景需求触发递送任务；
- 舒适价值：湿毛巾、冷食物、凌乱桌面等状态触发面向用户体验的任务；
- 可达价值：目标房间、设备或对象不可直接到达时，生成跨楼层导航或中间子任务。

这样，HLR 就可以从图中自动生成任务，而不是只从预设模板中抽样。一个 TongTest-style HLR 任务生成流程可以写成：

```text
1. Load hierarchical scene graph G.
2. Extract fluent variables F from nodes, edges, and agent state.
3. Detect value or state triggers over F.
4. Sample target graph state G_target according to trigger rules.
5. Validate preconditions and available affordances.
6. Decompose graph transition G_initial -> G_target into a global plan.
7. Compile global plan into local executable actions.
8. Execute actions with the state manager and generate streaming samples.
```

例如，对于“杯子在桌边”的场景，系统不需要先选择 `tidying` 任务模板，而是可以从图中检测到安全触发器：

```text
Initial:
cup.relation = on(table_edge)
cup.stability = unsafe

Trigger:
safety violation

Target:
cup.relation = on(stable_surface)
cup.stability = safe

Generated task:
move cup to a safe surface

Global plan:
goto(current_room): secure(cup)

Local actions:
scan(current_room)
pick(cup)
place(cup, stable_surface)
```

再如，对于“冷食物需要送到房间”的场景，系统可以生成一个组合任务：

```text
Initial:
food.temperature = cold
food.location = restaurant
target_room = room_301

Target:
food.temperature = hot
food.location = room_301

Global plan:
goto(restaurant): heat(food)
goto(room_301): deliver(food)

Local actions:
goto(restaurant)
pick(food)
open(microwave)
place(food, microwave)
use(microwave, food)
pick(food)
goto(room_301)
place(food, table)
```

这比当前单纯的 `delivery` 更接近 TongTest，因为任务目标来自 fluent transition，而不是来自人工指定的任务类型。

在论文或方法描述中，可以把改进后的 HLR 表述为：

```text
Inspired by TongTest, we formulate long-horizon task generation as graph-fluent transition sampling. A hierarchical scene graph serves as a task-generating parse graph, while dynamic object, relation, room, and agent attributes form the fluent space. Tasks are generated by detecting value- or state-driven triggers in the graph, sampling target graph states, and decomposing the resulting transition into global plans and local executable actions.
```

对应中文表述为：

```text
受 TongTest 启发，我们将长程任务生成建模为层次场景图上的 fluent 状态转移采样。场景图提供空间、时间和因果关系的结构化表示，物体、关系、房间和智能体的动态属性构成 fluent space。系统通过检测图中的状态触发器和价值触发器自动构造目标图状态，并将初始图到目标图的转移分解为全局计划和局部可执行动作。
```

这样改写后，HLR 与 TongTest 的结合不只是引用层面的相似，而是在方法层面形成清晰对应：TongTest 提供任务生成的理论框架，HLR 则在多楼层机器人长程规划场景中实现了一个具体的图状态转移式数据生成管线。
