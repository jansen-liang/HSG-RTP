# HLR 场景生成方案

本文档只描述 HLR 中的场景生成模块。该模块的目标是生成多楼层现代化建筑场景，并输出可用于后续任务生成和规划的数据结构。

场景生成的主线如下：

```text
Building Type
-> Floor Function
-> Room Topology
-> Object Layout
-> Hierarchical Scene Graph
```

## 1. 六类楼宇场景

HLR 定义六类现代化多楼层建筑：

```python
S_types = {
    "hotel",        # 酒店
    "hospital",     # 医院
    "residential",  # 住宅楼
    "school",       # 教学楼
    "office",       # 办公室 / 写字楼
    "factory"       # 大型工厂
}
```

| 楼宇类型 | 主要功能 | 典型空间 |
| --- | --- | --- |
| 酒店 | 住宿、餐饮、会议、后勤服务 | 大堂、客房、餐厅、会议室、服务间 |
| 医院 | 门诊、检查、住院、药品流转 | 挂号区、诊室、病房、药房、护士站 |
| 住宅楼 | 日常居住、公共服务、设备管理 | 门厅、住户单元、走廊、设备间 |
| 教学楼 | 教学、实验、办公、自习 | 教室、实验室、教师办公室、自习区 |
| 办公室 | 办公、会议、接待、文档流转 | 前台、开放办公区、会议室、打印区 |
| 大型工厂 | 生产、仓储、质检、维护 | 车间、仓库、质检室、控制室、维修区 |

## 2. 楼层功能生成

给定楼宇类型后，首先采样楼层数 `N_floors`，再为每层选择功能标签：

```python
floor_func = select_floor_function(floor_idx, N_floors, building_type)
```

不同楼宇的楼层功能先验如下：

```python
BuildingFloorFunctions = {
    "hotel": [
        "lobby_floor",
        "guest_room_floor",
        "restaurant_floor",
        "conference_floor",
        "service_floor"
    ],
    "hospital": [
        "outpatient_floor",
        "emergency_floor",
        "examination_floor",
        "inpatient_floor",
        "surgery_floor",
        "pharmacy_logistics_floor"
    ],
    "residential": [
        "entrance_floor",
        "residential_floor",
        "public_facility_floor",
        "equipment_floor"
    ],
    "school": [
        "classroom_floor",
        "laboratory_floor",
        "faculty_office_floor",
        "library_study_floor",
        "public_service_floor"
    ],
    "office": [
        "reception_floor",
        "open_office_floor",
        "meeting_floor",
        "executive_floor",
        "service_floor"
    ],
    "factory": [
        "production_floor",
        "warehouse_floor",
        "quality_control_floor",
        "control_room_floor",
        "maintenance_floor",
        "office_support_floor"
    ]
}
```

楼层功能应符合基本建筑逻辑。例如酒店底层通常是大堂，高层多为客房；医院低层更可能是门诊、急诊和检查区；住宅楼大部分楼层是重复住宅层；工厂通常包含生产、仓储、质检和控制功能。

## 3. 房间拓扑生成

每个楼层功能会被展开成一个房间拓扑图：

```text
G_floor = (R, E)
```

其中 `R` 是房间节点集合，`E` 是房间之间的连接边。连接边可以表示门、走廊、电梯厅、楼梯间等通行关系。

典型房间类型如下：

| 楼宇类型 | 楼层功能 | 房间类型 |
| --- | --- | --- |
| 酒店 | `lobby_floor` | lobby, reception, lounge, luggage_room, public_restroom, elevator_hall |
| 酒店 | `guest_room_floor` | corridor, guest_room, linen_room, housekeeping_room, elevator_hall, staircase |
| 医院 | `outpatient_floor` | registration, waiting_area, consultation_room, treatment_room, pharmacy, public_restroom |
| 医院 | `inpatient_floor` | nurse_station, ward, treatment_room, storage_room, doctor_office, elevator_hall |
| 住宅楼 | `residential_floor` | corridor, apartment_unit, elevator_hall, staircase, utility_room |
| 教学楼 | `classroom_floor` | corridor, classroom, teacher_lounge, storage_room, public_restroom |
| 教学楼 | `laboratory_floor` | lab, preparation_room, equipment_room, safety_room, corridor |
| 办公室 | `open_office_floor` | open_office, meeting_room, phone_booth, pantry, print_room, corridor |
| 工厂 | `production_floor` | workshop, assembly_area, machine_area, material_buffer, control_station |
| 工厂 | `warehouse_floor` | storage_zone, loading_area, shelf_area, packing_area, forklift_route |

拓扑可以由规则生成器产生，也可以由大模型提出候选后再做代码校验：

```text
floor function
-> propose room graph
-> validate topology
-> repair or resample
-> accept room topology
```

拓扑校验主要检查三件事：

- 连通性：每个可访问房间都能从入口、电梯厅或楼梯到达；
- 合理性：房间类型与楼层功能匹配，连接关系符合建筑常识；
- 多样性：同类建筑可以有重复结构，但不同样本之间不能完全复制。

## 4. 物体库与房间布局

房间拓扑确定后，对每个房间从对象库中选择物体并生成 layout。layout 需要给出物体实例、位置、朝向、支撑关系和可达区域。

HLR 采用 BEHAVIOR-1K / OmniGibson 作为主要对象语义库。原因是 BEHAVIOR 提供对象类别、affordance 和状态语义，适合映射到场景图中的对象状态和动作前置条件。

当前项目中可用的 BEHAVIOR 导出文件是：

```text
HLR_dataset/data/objects/behavior_full.yaml
```

本地统计如下：

| 对象池 | 数量 |
| --- | ---: |
| 唯一对象名 | 3373 |
| synset 条目 | 3484 |
| affordance 类型 | 50 |
| 带默认状态的对象名 | 1007 |
| 可摆放实体对象 `nonSubstance` | 2274 |
| 任务相关可交互对象 | 1407 |

当前已有场景子库：

| 场景子库 | 唯一对象名 |
| --- | ---: |
| hospital | 313 |
| hotel | 277 |
| residential | 277 |
| office | 335 |
| teaching_building | 66 |
| library | 252 |
| supermarket | 917 |

六类楼宇中目前已有酒店、医院、住宅楼、办公室、教学楼五类子库，去重后覆盖 560 个唯一对象名。大型工厂还没有单独的 `factory_generated.yaml`，可以先从 BEHAVIOR 主库中筛选工具、容器、设备、材料、清洁用品和运输相关对象。

房间到对象的先验示例：

```python
RoomObjectPrior = {
    "guest_room": [
        "bed", "nightstand", "desk", "chair", "wardrobe", "lamp", "towel"
    ],
    "consultation_room": [
        "doctor_desk", "chair", "examination_bed", "computer", "medicine_cabinet"
    ],
    "ward": [
        "patient_bed", "bedside_table", "medical_cart", "chair", "monitor"
    ],
    "classroom": [
        "student_desk", "chair", "blackboard", "projector", "podium"
    ],
    "open_office": [
        "desk", "office_chair", "computer", "cabinet", "printer", "document"
    ],
    "warehouse": [
        "shelf", "box", "pallet", "cart", "scanner"
    ]
}
```

layout 摆放需要满足：

- 大件物体先摆放，例如床、桌、货架、机器；
- 小物体放在合理支撑面上，例如文件在桌上，药品在柜中；
- 门、电梯口、楼梯口和主要走廊不能被堵住；
- 可交互物体应处于机器人可到达区域；
- 功能相关物体保持合理相对位置，例如床与床头柜相邻，打印机靠近办公区。

## 5. 输出场景图

场景生成模块输出层次场景图 `G_scene`：

```python
G_scene = {
    "building": {
        "type": "hospital",
        "num_floors": 5
    },
    "floors": [
        {
            "id": "floor_1",
            "function": "outpatient_floor",
            "rooms": ["registration_1", "waiting_area_1", "consultation_room_1"],
            "vertical_connectors": ["elevator_1", "staircase_1"]
        }
    ],
    "rooms": [
        {
            "id": "consultation_room_1",
            "type": "consultation_room",
            "floor": "floor_1",
            "objects": ["desk_1", "chair_1", "computer_1"]
        }
    ],
    "objects": [
        {
            "id": "medicine_1",
            "category": "medicine",
            "room": "pharmacy_1",
            "states": {
                "availability": "available"
            },
            "affordances": ["pick", "place", "deliver"]
        }
    ],
    "relations": [
        ["consultation_room_1", "connected_to", "corridor_1"],
        ["medicine_1", "in", "medicine_cabinet_1"]
    ],
    "navigation_graph": {
        "nodes": ["consultation_room_1", "corridor_1", "elevator_1"],
        "edges": [["consultation_room_1", "corridor_1"], ["corridor_1", "elevator_1"]]
    }
}
```

这个场景图至少需要包含：

- 建筑类型和楼层数；
- 每层的功能；
- 房间类型和房间连接关系；
- 电梯、楼梯等跨楼层连接；
- 房间内物体、对象状态和 affordance；
- 用于导航的拓扑图。

## 6. 生成流程伪代码

```python
def generate_scene(S_types, N_floor_range, behavior_assets, rules):
    G_scene = empty_scene_graph()

    building_type = sample_building_type(S_types)
    N_floors = sample_num_floors(building_type, N_floor_range)

    G_scene.set_building(building_type, N_floors)

    for floor_idx in range(1, N_floors + 1):
        floor_func = select_floor_function(
            floor_idx=floor_idx,
            num_floors=N_floors,
            building_type=building_type
        )

        topology = generate_room_topology(
            building_type=building_type,
            floor_function=floor_func,
            rules=rules.topology_rules
        )

        validate_topology(topology)
        G_scene.add_floor_topology(floor_idx, floor_func, topology)

        for room in topology.rooms:
            objects = select_room_objects(
                room_type=room.type,
                behavior_assets=behavior_assets
            )

            layout = generate_room_layout(
                room=room,
                objects=objects,
                placement_rules=rules.layout_rules
            )

            validate_layout(room, layout)
            G_scene.update_room_layout(floor_idx, room.id, layout)

    G_scene = build_navigation_graph(G_scene)
    validate_scene_graph(G_scene)

    return G_scene
```

## 7. 场景质量检查

生成后的场景需要检查：

- 楼层功能是否符合楼宇类型；
- 房间拓扑是否连通；
- 房间类型和对象类型是否匹配；
- 物体是否有合理支撑关系；
- 门口、走廊、电梯、楼梯是否可通行；
- 跨楼层导航图是否连通；
- 场景是否包含足够的可交互对象。

最终得到的 `G_scene` 就是后续任务生成、规划和执行模拟的输入。
