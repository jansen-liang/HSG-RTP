"""任务生成器 - 根据配置生成不同类型和难度的任务"""

import random
from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass

# ===== 全局映射（建议放在文件顶部）=====
STATE_TO_METHOD = {
    ("temperature", "hot"): "heat",
    ("wetness", "dry"): "dry",
    ("wetness", "wet"): "wet",
    ("cleanliness", "clean"): "wash"
}

class TaskType(Enum):
    DELIVERY = "delivery"
    TIDYING = "tidying"
    GUIDANCE = "guidance"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class Task:
    task_type: TaskType
    difficulty: DifficultyLevel
    parameters: dict
    description: str

def generate_tasks(scene: dict, task_types: List[str], difficulties: List[str], 
                  max_tasks: int = 10, config: dict = None) -> List[Task]:
    """
    为单个场景生成任务列表
    
    Args:
        scene: 场景数据
        task_types: 任务类型列表
        difficulties: 难度级别列表
        max_tasks: 最大任务数量
        config: 任务配置参数
        
    Returns:
        任务列表
    """
    tasks = []
    task_config = config.get("task_config", {}) if config else {}
    difficulty_props = config.get("difficulty_proportions", {
        "easy": 0.5, "medium": 0.3, "hard": 0.2
    }) if config else {"easy": 0.5, "medium": 0.3, "hard": 0.2}
    
    # 根据比例分配难度
    difficulty_counts = {
        diff: int(max_tasks * difficulty_props.get(diff, 0))
        for diff in difficulties
    }
    
    # 确保总数不超过 max_tasks
    total_assigned = sum(difficulty_counts.values())
    if total_assigned < max_tasks:
        difficulty_counts[difficulties[0]] += max_tasks - total_assigned
    
    for difficulty in difficulties:
        count = difficulty_counts.get(difficulty, 0)
        diff_level = DifficultyLevel(difficulty)
        
        for _ in range(count):
            task_type = TaskType(random.choice(task_types))
            task = generate_single_task(scene, task_type, diff_level, task_config)
            if task is not None:
                tasks.append(task)
    
    return tasks

def generate_single_task(scene: dict, task_type: TaskType, difficulty: DifficultyLevel, 
                        config: dict) -> Task:
    """生成单个任务"""
    
    if task_type == TaskType.DELIVERY:
        return generate_delivery_task(scene, difficulty, config.get("delivery", {}))
    elif task_type == TaskType.TIDYING:
        return generate_tidying_task(scene, difficulty, config.get("tidying", {}))
    elif task_type == TaskType.GUIDANCE:
        return generate_guidance_task(scene, difficulty, config.get("guidance", {}))
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def generate_delivery_task(scene: dict, difficulty: DifficultyLevel, config: dict) -> Task:
    """生成配送任务（支持状态变化）"""
    rooms = list(scene["rooms"].keys())
    
    # 根据难度设置参数
    if difficulty == DifficultyLevel.EASY:
        num_objects = random.randint(1, 2)
        num_destinations = 1
    elif difficulty == DifficultyLevel.MEDIUM:
        num_objects = random.randint(2, 3)
        num_destinations = random.randint(1, 2)
    else:  # HARD
        num_objects = random.randint(3, config.get("max_objects", 4))
        num_destinations = random.randint(2, 3)

    # 找出有可用物品的房间（排除电梯）
    rooms_with_objects = []
    for room in rooms:
        if room.startswith("elevator_") or room == "elevator_cabin":
            continue
        available = get_available_objects(scene, room)
        if len(available) >= num_objects:
            rooms_with_objects.append(room)

    # 若无足够物品房间，放宽条件
    if not rooms_with_objects:
        for room in rooms:
            if room.startswith("elevator_") or room == "elevator_cabin":
                continue
            available = get_available_objects(scene, room)
            if available:
                rooms_with_objects.append(room)
                num_objects = min(num_objects, len(available))

    if not rooms_with_objects:
        return None

    source_room = random.choice(rooms_with_objects)
    available_objects = get_available_objects(scene, source_room)
    num_objects = min(num_objects, len(available_objects))
    selected_objects = random.sample(available_objects, num_objects)

    # 选择目标房间（排除源房间和电梯）
    valid_targets = [
        r for r in rooms 
        if r != source_room 
        and not r.startswith("elevator_") 
        and r != "elevator_cabin"
    ]
    if not valid_targets:
        return None
    target_rooms = random.sample(valid_targets, min(num_destinations, len(valid_targets)))

    # ===== 状态目标注入（Medium/Hard） =====
    object_goals = {}
    if difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD):
        for obj in selected_objects:
            obj_info = scene["rooms"][source_room]["small_objects"][obj]
            obj_type = obj_info["type"]
            current_state = obj_info.get("state", {})
            
            goal_state = None
            # 食物类 → hot
            if obj_type in ["food", "food_container", "bread", "frozen_food"] or any(kw in obj for kw in ["pizza", "sushi", "baguette", "takeaway"]):
                if current_state.get("temperature") in ["cold", "room", "frozen"]:
                    goal_state = {"temperature": "hot"}
            # 布草类 → dry
            elif obj_type in ["linen", "clothing"] or any(kw in obj for kw in ["towel", "tshirt"]):
                if current_state.get("wetness") in ["wet", "moist"]:
                    goal_state = {"wetness": "dry"}

            if goal_state:
                # 检查是否有设备支持该转换
                method_needed = None
                for (state_key, target_val), method in STATE_TO_METHOD.items():
                    if state_key in goal_state and goal_state[state_key] == target_val:
                        method_needed = method
                        break

                if method_needed and has_device_supporting_method(scene, method_needed):
                    object_goals[obj] = goal_state

    # 构造参数
    parameters = {
        "objects": selected_objects,
        "objects_origin_state": {obj: scene["rooms"][source_room]["small_objects"][obj].get("state", {}) for obj in selected_objects},
        "objects_goal_state": object_goals,
        "source_room": source_room,
        "target_rooms": target_rooms,
        "num_objects": num_objects
    }
    if object_goals:
        parameters["object_goals"] = object_goals
    
    print(parameters)
    
    # 生成描述
    obj_desc = ", ".join(
        f"{'hot ' if obj in object_goals and 'temperature' in object_goals[obj] else ''}{obj}"
        for obj in selected_objects
    )
    description = f"Deliver {obj_desc} from {source_room} to {', '.join(target_rooms)}"
    print(f"[description]: {description}")
    return Task(TaskType.DELIVERY, difficulty, parameters, description)

def generate_tidying_task(scene: dict, difficulty: DifficultyLevel, config: dict) -> Task:
    rooms = list(scene["rooms"].keys())
    
    # 难度参数
    if difficulty == DifficultyLevel.EASY:
        num_rooms = 1
        min_objects = 2
        max_objects = 3
    elif difficulty == DifficultyLevel.MEDIUM:
        num_rooms = random.randint(1, 2)
        min_objects = 3
        max_objects = 4
    else:  # HARD
        num_rooms = random.randint(2, min(3, len(rooms)))
        min_objects = 4
        max_objects = config.get("max_objects", 6)

    # 收集所有候选物品（按是否需要整理分类）
    tidy_candidates = []  # [(room, obj_id, obj_info), ...]
    all_candidates = []   # 所有可用物品（用于 Easy）

    for room in rooms:
        if room.startswith("elevator_") or room == "elevator_cabin":
            continue
        for obj_id, obj_info in scene["rooms"][room].get("small_objects", {}).items():
            if not is_control_object(obj_id, obj_info):
                all_candidates.append((room, obj_id, obj_info))
                # 判断是否需要整理
                state = obj_info.get("state", {})
                needs_tidy = False
                if state.get("wetness") in ["wet", "moist"]:
                    needs_tidy = True
                elif state.get("cleanliness") == "dirty":
                    needs_tidy = True
                # 可扩展：opened, used, etc.
                if needs_tidy:
                    tidy_candidates.append((room, obj_id, obj_info))

    if difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD):
        # 优先使用需要整理的物品
        candidates = tidy_candidates if tidy_candidates else all_candidates
    else:
        # Easy：使用所有物品
        candidates = all_candidates

    if not candidates:
        return None

    # 按房间分组
    room_to_objects = {}
    for room, obj_id, obj_info in candidates:
        room_to_objects.setdefault(room, []).append((obj_id, obj_info))

    # 选择房间
    available_rooms = list(room_to_objects.keys())
    if not available_rooms:
        return None
    selected_rooms = random.sample(available_rooms, min(num_rooms, len(available_rooms)))

    # 收集物品
    objects_to_tidy = []
    for room in selected_rooms:
        objs = room_to_objects[room]
        n = random.randint(1, min(3, len(objs)))
        objects_to_tidy.extend(objs[:n])

    if len(objects_to_tidy) < min_objects:
        min_objects = max(1, len(objects_to_tidy))
    if len(objects_to_tidy) == 0:
        return None

    num_objects = random.randint(min_objects, min(max_objects, len(objects_to_tidy)))
    selected_pairs = random.sample(objects_to_tidy, num_objects)
    selected_objects = [obj_id for obj_id, _ in selected_pairs]

    # ===== 设置 object_goals（Medium/Hard）=====
    object_goals = {}
    if difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD):
        for obj_id, obj_info in selected_pairs:
            state = obj_info.get("state", {})
            goal = {}

            # 湿 → 干
            if state.get("wetness") in ["wet", "moist"]:
                goal["wetness"] = "dry"
            # 脏 → 净
            if state.get("cleanliness") == "dirty":
                goal["cleanliness"] = "clean"

            if goal:
                # 检查设备支持
                valid = True
                for key, val in goal.items():
                    method = STATE_TO_METHOD.get((key, val))
                    if method and not has_device_supporting_method(scene, method):
                        valid = False
                        break
                if valid:
                    object_goals[obj_id] = goal

    # 构造参数
    parameters = {
        "objects": selected_objects,
        "source_rooms": selected_rooms,
        "num_objects": num_objects,
        "action_type": "organize"
    }
    if object_goals:
        parameters["object_goals"] = object_goals
    
    print(parameters)
    
    # 描述
    desc_items = []
    for obj in selected_objects:
        prefix = ""
        if obj in object_goals:
            if object_goals[obj].get("wetness") == "dry":
                prefix = "dry "
            elif object_goals[obj].get("cleanliness") == "clean":
                prefix = "clean "
        desc_items.append(prefix + obj)

    description = f"Tidy up {', '.join(desc_items)} in {', '.join(selected_rooms)}"
    print(f"[description]: {description}")
    return Task(TaskType.TIDYING, difficulty, parameters, description)

def generate_guidance_task(scene: dict, difficulty: DifficultyLevel, config: dict) -> Task:
    """生成导航任务"""
    rooms = list(scene["rooms"].keys())
    
    # 根据难度选择路径点数量
    if difficulty == DifficultyLevel.EASY:
        num_waypoints = 2
    elif difficulty == DifficultyLevel.MEDIUM:
        num_waypoints = random.randint(2, 3)
    else:  # HARD
        num_waypoints = random.randint(3, min(config.get("max_waypoints", 5), len(rooms)))
    
    # 选择起点和终点（排除电梯）
    start_room = scene["agent"]["position"]
    available_rooms = []
    for r in rooms:
        if (r != start_room and 
            not r.startswith("elevator_") and
            "elevator" not in r.lower()):
            available_rooms.append(r)
    
    if len(available_rooms) < num_waypoints - 1:
        num_waypoints = len(available_rooms) + 1
    
    waypoints = [start_room] + random.sample(available_rooms, num_waypoints - 1)
    
    parameters = {
        "waypoints": waypoints,
        "start_room": start_room,
        "end_room": waypoints[-1],
        "intermediate_points": waypoints[1:-1] if len(waypoints) > 2 else []
    }
    
    print(parameters)

    description = f"Navigate from {start_room} to {waypoints[-1]}"
    
    if len(waypoints) > 2:
        description += f" via {', '.join(waypoints[1:-1])}"
    print(f"[description]: {description}")
    return Task(TaskType.GUIDANCE, difficulty, parameters, description)

def get_available_objects(scene: dict, room_id: str) -> List[str]:
    """获取房间中可用的物品列表（仅包含可操作的小物品，排除电梯按钮等控制设备）"""
    if room_id not in scene["rooms"]:
        return []
    
    room = scene["rooms"][room_id]
    objects = []
    
    # 只添加可移动的小物品，排除控制设备
    if "small_objects" in room:
        for obj_id, obj_info in room["small_objects"].items():
            # 排除电梯按钮和其他控制设备
            if (not obj_id.startswith("elevator_button") and 
                obj_info.get("type") != "control" and
                "button" not in obj_id.lower() and
                "switch" not in obj_id.lower()):
                objects.append(obj_id)
    
    # 大物品不能被操作，所以不添加到可用物品列表中
    # large_objects 只能用于 goto() 操作，不能 pick()
    
    return objects

def has_device_supporting_method(scene: dict, method_name: str) -> bool:
    """检查场景中是否存在支持该方法的设备"""
    for room in scene["rooms"].values():
        for dev in room.get("large_objects", {}).values():
            if method_name in dev.get("methods", {}):
                return True
    return False

def is_control_object(obj_id: str, obj_info: dict) -> bool:
    return (
        obj_id.startswith("elevator_") or
        obj_info.get("type") == "control" or
        "button" in obj_id.lower() or
        "switch" in obj_id.lower()
    )