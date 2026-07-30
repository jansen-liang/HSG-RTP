"""动作规划器 - 将任务转换为具体的动作序列"""

from typing import Dict, List, Any, Tuple
from .task_generator import Task, TaskType
import re
from enum import Enum
import random

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

def _make_container_device_template(method_name: str):
    """为 is_container=True 的设备（如 microwave, dryer）生成标准操作序列"""
    def template(obj: str, device: str) -> List[str]:
        return [
            f"go_to({device})",
            f"open({device})",
            f"place({obj}, {device})",
            f"use({device}, {obj})",
            f"open({device})",
            f"pick({obj})"
        ]
    return template

def _sink_wet_template(obj: str, device: str) -> List[str]:
    """为 sink 这类非容器设备生成简化序列"""
    return [
        f"go_to({device})",
        f"use({device}, {obj})"
    ]

# ===== 动作模板与状态映射 =====
STATE_TO_METHOD = {
    ("temperature", "hot"): "heat",
    ("temperature", "cold"): "cool",      # 如 fridge（当前场景未显式定义，可扩展）
    ("wetness", "wet"): "wet",
    ("wetness", "dry"): "dry",
    # 可继续扩展：("cleanliness", "clean") → "wash" 等
}
# 动作模板库
ACTION_TEMPLATES = {
    "heat": _make_container_device_template("heat"),
    "dry": _make_container_device_template("dry"),
    "cool": _make_container_device_template("cool"),  
    "wet": _sink_wet_template, 
}

def plan_actions(scene: dict, task: Task) -> List[str]:
    """
    根据任务生成具体的动作序列
    
    Args:
        scene: 场景数据
        task: 任务对象
        
    Returns:
        动作序列列表
    """
    rooms = scene["rooms"]
    agent_start = scene["agent"]["position"]
    
    if task.task_type == TaskType.DELIVERY:
        return plan_delivery_actions(rooms, agent_start, task.parameters, task.difficulty)
    elif task.task_type == TaskType.TIDYING:
        return plan_tidying_actions(rooms, agent_start, task.parameters, task.difficulty)
    elif task.task_type == TaskType.GUIDANCE:
        return plan_guidance_actions(rooms, agent_start, task.parameters, task.difficulty)
    else:
        raise ValueError(f"Unknown task type: {task.task_type}")

def plan_delivery_actions(rooms: Dict[str, Any], agent_start: str, params: dict, difficulty: DifficultyLevel) -> List[str]:
    """规划配送任务的动作序列"""
    subtasks = []
    objects = params["objects"]
    source_room = params["source_room"]
    target_rooms = params["target_rooms"]
    
    current_pos = agent_start

    # 对于每个物品，执行 取->送 的序列
    for i, obj in enumerate(objects):
        target_room = target_rooms[i % len(target_rooms)]  # 循环分配目标房间
        
        # 前往源房间
        subtasks.extend(plan_path_with_elevator(rooms, current_pos, source_room, difficulty))
        
        # 检查物品是否依附在大物体上，如果是则需要先处理依附关系
        subtasks.extend(plan_pick_with_dependency(rooms, source_room, obj))
        
        # 前往目标房间
        subtasks.extend(plan_path_with_elevator(rooms, source_room, target_room, difficulty))
        # 找到目标房间内合适的放置表面（桌子、台面、架子等），若无则放地板
        target_surface = find_suitable_surface(rooms, target_room, obj)
        subtasks.append(f"place({obj}, {target_surface})")
        
        current_pos = target_room
    
    return subtasks

def plan_tidying_actions(rooms: Dict[str, Any], agent_start: str, params: dict, difficulty: DifficultyLevel) -> List[str]:
    subtasks = []
    objects = params["objects"]
    current_pos = agent_start
    
    for obj in objects:
        # 🔍 全局搜索物品位置
        obj_room = find_object_location(rooms, obj)
        if not obj_room:
            print(f"Warning: Object {obj} not found in any room!")
            continue  # 跳过不存在的物品
        
        # 前往物品所在房间
        subtasks.extend(plan_path_with_elevator(rooms, current_pos, obj_room, difficulty))
        subtasks.extend(plan_pick_with_dependency(rooms, obj_room, obj))
        
        # 放置
        target_surface = find_suitable_surface(rooms, obj_room, obj)
        subtasks.append(f"place({obj}, {target_surface})")
        
        current_pos = obj_room
    
    return subtasks

def plan_guidance_actions(rooms: Dict[str, Any], agent_start: str, params: dict, difficulty: DifficultyLevel) -> List[str]:
    """规划导航任务的动作序列"""
    subtasks = []
    waypoints = params["waypoints"]
    
    current_pos = agent_start
    
    # 按顺序访问每个路径点
    for waypoint in waypoints[1:]:  # 跳过起始点
        subtasks.extend(plan_path_with_elevator(rooms, current_pos, waypoint, difficulty))
        current_pos = waypoint
    
    return subtasks

def _simulate_elevator_button_states(difficulty: DifficultyLevel) -> Dict[str, bool]:
    """模拟电梯按钮初始状态（仅用于规划决策）"""
    if difficulty.value == DifficultyLevel.EASY.value:
        return {"up_lit": False, "down_lit": False}
    elif difficulty.value == DifficultyLevel.MEDIUM.value or difficulty.value == DifficultyLevel.HARD.value:
        return {
            "up_lit": random.choice([True, False]),
            "down_lit": random.choice([True, False])
        }

def plan_path_with_elevator(
        rooms: Dict[str, Any], 
        start_room: str,
        goal_room: str,
        difficulty: DifficultyLevel) -> List[str]:
    """
    规划带电梯的路径。
    对于跨楼层移动，不生成实际的电梯相关动作，直接使用简化路径。
    这避免了复杂的电梯状态管理和不符合场景图neighbor关系的问题。
    """
    if start_room == goal_room:
        return [f"scan({start_room})"]
    
    start_floor = rooms[start_room]["floor"]
    goal_floor = rooms[goal_room]["floor"]
    
    if start_floor == goal_floor:
        # 同层移动：使用 BFS 获取完整路径，只在最终目标scan
        path = find_shortest_path_within_floor(rooms, start_room, goal_room)
        if not path:
            return [f"goto({goal_room})", f"scan({goal_room})"]  # fallback
        actions = []
        for room in path[1:]:
            actions.append(f"goto({room})")
        # 只在最终目标房间scan
        actions.append(f"scan({goal_room})")
        return actions
    else:
        actions = []
        start_floor = rooms[start_room]["floor"]
        goal_floor = rooms[goal_room]["floor"]
        start_floor_num = extract_floor_number(start_floor)
        goal_floor_num = extract_floor_number(goal_floor)
        
        # 1. 去起始楼层电梯厅
        start_elev = find_elevator_hall_on_floor(rooms, start_floor)
        if not start_elev:
            return []

        if start_room != start_elev:
            path = find_shortest_path_within_floor(rooms, start_room, start_elev)
            for room in path[1:]:
                actions.append(f"goto({room})")
        
        # 扫描面板
        panel = next((k for k in rooms[start_elev]['large_objects'] if 'panel' in k), None)
        if panel:
            actions.append(f"scan({panel})")

        # 2. 处理呼叫按钮（根据难度模拟状态）
        direction = "up" if goal_floor_num > start_floor_num else "down"
        button_states = _simulate_elevator_button_states(difficulty)
        
        if direction == "up":
            if not button_states["up_lit"]:
                call_btn = find_elevator_call_button(rooms[start_elev], goal_floor, start_floor)
                if call_btn:
                    actions.append(f"press({call_btn})")
            # 如果 down_lit 为 True，说明有下行请求，需等待
            if button_states["down_lit"] and direction == "up":
                actions.append("wait(elevator_down_clear)")
        else:  # down
            if not button_states["down_lit"]:
                call_btn = find_elevator_call_button(rooms[start_elev], goal_floor, start_floor)
                if call_btn:
                    actions.append(f"press({call_btn})")
            if button_states["up_lit"] and direction == "down":
                actions.append("wait(elevator_up_clear)")

        # 3. 进入电梯厢
        if "elevator_cabin" in rooms:
            actions.append("goto(elevator_cabin)")
            actions.append("scan(elevator_cabin)")

            # 4. 处理楼层按钮
            target_btn = f"elevator_button_{goal_floor_num}"
            # 模拟按钮初始状态
            target_btn_lit = False
            if difficulty in (DifficultyLevel.MEDIUM, DifficultyLevel.HARD):
                target_btn_lit = random.choice([True, False])
            
            if not target_btn_lit:
                actions.append(f"press({target_btn})")
            # 如果已亮，则不按

            actions.append("scan(elevator_cabin)")  # 等待电梯运行

        # 5. 去目标房间
        goal_elev = find_elevator_hall_on_floor(rooms, goal_floor)
        if goal_elev:
            actions.append(f"goto({goal_elev})")
            if goal_elev != goal_room:
                path = find_shortest_path_within_floor(rooms, goal_elev, goal_room)
                for room in path[1:]:
                    actions.append(f"goto({room})")
                actions.append(f"scan({goal_room})")
            else:
                actions.append(f"scan({goal_elev})")

        return actions
    
# 辅助函数
def infer_required_method(current_state: dict, goal_state: dict) -> str:
    """根据状态差异推断需要调用的方法名"""
    for key, target_val in goal_state.items():
        if current_state.get(key) != target_val:
            method = STATE_TO_METHOD.get((key, target_val))
            if method:
                return method
    return None

def requires_state_change(current_state: dict, goal_state: dict) -> bool:
    """检查当前状态是否满足目标状态"""
    for key, target_val in goal_state.items():
        if current_state.get(key) != target_val:
            return True
    return False

def find_object_location(rooms: Dict[str, Any], obj_id: str) -> str:
    """在所有房间中查找物品位置"""
    for room_id, room in rooms.items():
        # 检查小物品
        if obj_id in room.get("small_objects", {}):
            return room_id
        # 检查大物品（虽然通常不可拾取）
        if obj_id in room.get("large_objects", {}):
            return room_id
    return None

def find_elevator_hall_on_floor(rooms: Dict[str, Any], floor: str) -> str:
    """找到指定楼层的电梯厅（elevator_1f, elevator_2f 等）"""
    for room_id, info in rooms.items():
        if (info["floor"] == floor and 
            room_id.startswith("elevator_") and 
            room_id != "elevator_cabin"):
            return room_id
    return None

def find_shortest_path_within_floor(rooms: Dict[str, Any], start: str, goal: str) -> List[str]:
    from collections import deque
    
    if start == goal:
        return [start]
    if start not in rooms or goal not in rooms:
        print(f"⚠️ Invalid start/goal: {start} -> {goal}")
        return []
    
    start_floor = rooms[start]["floor"]
    if rooms[goal]["floor"] != start_floor:
        print(f"⚠️ Different floors: {start}({start_floor}) -> {goal}({rooms[goal]['floor']})")
        return []
    
    queue = deque([(start, [start])])
    visited = {start}
    max_steps = 100  # 防止无限循环
    
    while queue:
        current, path = queue.popleft()
        
        if len(path) > max_steps:
            print(f"⚠️ Path too long (> {max_steps} steps): {start} -> {goal}")
            return []
        
        for neighbor in rooms[current].get("neighbor", []):
            if neighbor not in rooms:
                continue
            if neighbor == goal:
                found_path = path + [neighbor]
                return found_path
            if neighbor not in visited and rooms[neighbor]["floor"] == start_floor:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    print(f"❌ No path found: {start} -> {goal}")
    return []

def find_shortest_path_cross_floor(rooms: Dict[str, Any], start: str, goal: str) -> List[str]:
    """跨楼层寻找最短路径（允许电梯传送）"""
    from collections import deque
    
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in rooms[current].get("neighbor", []):
            if neighbor == goal:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # 找不到路径

def find_elevator_on_floor(rooms: Dict[str, Any], floor: int) -> str:
    """寻找指定楼层的电梯"""
    for room_id, room_info in rooms.items():
        if room_info["floor"] == floor:
            for obj_id, obj_info in room_info.get("large_objects", {}).items():
                if obj_info.get("type") == "transport" and "elevator" in obj_id:
                    return room_id
    return None

def extract_floor_number(floor_name: str) -> int:
    """从楼层名称中提取数字，如 'floor_2_guest' -> 2"""
    import re
    match = re.search(r'floor_(\d+)', floor_name)
    if match:
        return int(match.group(1))
    # 如果没有找到数字，尝试直接提取数字
    numbers = re.findall(r'\d+', floor_name)
    if numbers:
        return int(numbers[0])
    return 1  # 默认返回1楼

def find_elevator_call_button(room: dict, target_floor: str, current_floor: str) -> str:
    """找到电梯呼叫按钮（电梯外部使用）"""
    # 判断应该按上行还是下行按钮
    target_num = extract_floor_number(target_floor)
    current_num = extract_floor_number(current_floor)
    for btn_id, btn_info in room.get("small_objects", {}).items():
        if "call" in btn_id:
            if target_num > current_num and "up" in btn_id:
                return btn_id
            elif target_num < current_num and "down" in btn_id:
                return btn_id
    
    return None

def find_elevator_button_for_floor(room: dict, target_floor: int) -> str:
    """寻找指定楼层的电梯按钮（电梯内部使用）"""
    for btn_id, btn_info in room.get("small_objects", {}).items():
        btn_info = btn_info if isinstance(btn_info, dict) else {"affordance": []}
        relation = btn_info.get("relation", {})
        
        # 确认是内部按钮并且可以按压
        if "inside" in relation and "press" in btn_info.get("affordance", []):
            # 检查按钮是否对应目标楼层
            # 支持两种格式: elevator_button_X 和 elevator_button_XF_to_Y
            if (btn_id == f"elevator_button_{target_floor}" or 
                btn_id.endswith(f"_to_{target_floor}")):
                return btn_id
    
    # 如果没找到，返回None以便使用fallback逻辑
    return None

def get_room_objects(rooms: Dict[str, Any], room_id: str) -> List[str]:
    """获取房间中可操作的物品（仅可移动的小物品，排除控制设备）"""
    if room_id not in rooms:
        return []
    
    room = rooms[room_id]
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
    
    return objects

def find_suitable_surface(rooms: Dict[str, Any], room_id: str, obj: str) -> str:
    """为物品找到合适的表面放置"""
    if room_id not in rooms:
        return "floor"
    
    room = rooms[room_id]
    
    # 寻找桌子、柜子等表面
    surfaces = ["table", "desk", "counter", "shelf", "cabinet"]
    
    for obj_id, obj_info in room.get("large_objects", {}).items():
        obj_type = obj_info.get("type", "").lower()
        for surface in surfaces:
            if surface in obj_type or surface in obj_id.lower():
                return obj_id
    
    # 默认返回地板
    return "floor"

def plan_pick_with_dependency(rooms: Dict[str, Any], room_id: str, obj_id: str) -> List[str]:
    if room_id not in rooms or obj_id not in rooms[room_id].get("small_objects", {}):
        return [f"pick({obj_id})"]  # fallback
    
    obj_info = rooms[room_id]["small_objects"][obj_id]
    relation = obj_info.get("relation", {})
    
    # 如果物品在某个表面上（如 on: desk）
    if "on" in relation:
        surface = relation["on"]
        return [f"scan({surface})", f"pick({obj_id})"]
    # 如果物品在容器内（如 in: wardrobe）
    elif "in" in relation:
        container = relation["in"]
        return [f"scan({container})", f"pick({obj_id})"]
    
    return [f"pick({obj_id})"]

def generate_global_plan(
    subtasks: List[str],
    rooms: Dict[str, Any],
    task_type: str,
    initial_room: str = None,
) -> Tuple[List[str], List[int]]:
    """
    生成 global_plan 和对应的 subtask 边界。
    
    Returns:
        global_plan: 高层计划列表
        boundaries: 每个 global_plan 步骤结束后的 subtask 索引（即第 i 步覆盖 subtasks[boundaries[i-1] : boundaries[i]]）
    """
    global_plan = []
    boundaries = []
    i = 0
    n = len(subtasks)

    if initial_room in rooms and subtasks and not (
        subtasks[0].startswith("goto(") and subtasks[0][5:-1] in rooms
    ):
        first_room_change = 0
        while first_room_change < n:
            action = subtasks[first_room_change]
            if action.startswith("goto(") and action[5:-1] in rooms:
                break
            first_room_change += 1
        description = _summarize_room_actions(
            subtasks[:first_room_change], initial_room, task_type
        )
        global_plan.append(f"goto({initial_room}): {description}")
        boundaries.append(first_room_change)
        i = first_room_change
    
    while i < n:
        task = subtasks[i]
        step_start = i  # 当前 global 步骤从 i 开始
        
        # === 处理电梯厅（elevator_Xf）===
        if task.startswith("goto(elevator_") and task.endswith("f)"):
            hall_start = task[5:-1]
            floor_match = re.search(r'(\d+)f$', hall_start)
            if not floor_match:
                # 非标准电梯厅，按普通房间处理
                i += 1
                continue
            
            start_floor = floor_match.group(1) + "f"
            target_floor = None
            j = i + 1
            found_press = False
            
            # 向前查找 press(elevator_button_X) 和目标电梯厅
            while j < n:
                t = subtasks[j]
                if t.startswith("press(elevator_button_"):
                    try:
                        floor_num = t.split("elevator_button_")[1].rstrip(")")
                        target_floor = f"{floor_num}f"
                        found_press = True
                    except:
                        pass
                elif t.startswith("goto(elevator_") and t.endswith("f)"):
                    if found_press:
                        # 找到完整电梯移动段：从 hall_start 到当前电梯厅
                        global_plan.append(f"goto({hall_start}): trans from({start_floor}) to({target_floor})")
                        boundaries.append(j + 1)  # 包含 goto(elevator_Yf)
                        i = j + 1  # 下一个步骤从 j+1 开始
                        break
                j += 1
            else:
                # 未找到完整电梯段 → 按普通房间处理
                k = i + 1
                while k < n:
                    next_task = subtasks[k]
                    if next_task.startswith("goto(") and next_task[5:-1] in rooms:
                        break
                    k += 1
                room_actions = subtasks[i+1:k]
                desc = _summarize_room_actions(room_actions, hall_start, task_type)
                global_plan.append(f"goto({hall_start}): {desc}")
                boundaries.append(k)
                i = k
            continue
        
        # === 处理普通房间 ===
        if task.startswith("goto(") and task[5:-1] in rooms:
            room = task[5:-1]
            k = i + 1
            # 收集直到下一个 goto(有效房间)
            while k < n:
                next_task = subtasks[k]
                if next_task.startswith("goto("):
                    next_target = next_task[5:-1]
                    if next_target in rooms:  # 只有有效房间才分割
                        break
                k += 1
            room_actions = subtasks[i+1:k]
            desc = _summarize_room_actions(room_actions, room, task_type)
            global_plan.append(f"goto({room}): {desc}")
            boundaries.append(k)
            i = k
            continue
        
        # === 其他动作（不应出现，但安全处理）===
        i += 1
    
    return global_plan, boundaries

def _summarize_room_actions(actions: List[str], room: str, task_type: str) -> str:
    """总结普通房间内的动作（不处理电梯移动）"""
    pick_actions = [a for a in actions if a.startswith("pick(")]
    place_actions = [a for a in actions if a.startswith("place(")]
    
    if pick_actions and place_actions:
        # 同一房间内 pick+place 视为整理
        items = [a[5:-1] for a in pick_actions]
        return f"organize({', '.join(items)})"
    elif pick_actions:
        items = [a[5:-1] for a in pick_actions]
        return f"pick({', '.join(items)})"
    elif place_actions:
        items = [a[6:a.find(',') if ',' in a else -1] for a in place_actions]
        return f"place({', '.join(items)})"
    else:
        # 检查是否只有 scan/goto，且没有交互动作
        has_scan_or_goto = any(act.startswith(("scan(", "goto(")) for act in actions)
        has_interaction = any(act.startswith(("pick(", "place(", "press(")) for act in actions)
        
        if has_scan_or_goto and not has_interaction:
            return "pass()"
        elif actions:
            return "perform actions"
        else:
            return "pass()"

def is_room(target: str, rooms: Dict[str, Any]) -> bool:
    """
    检查目标是否是房间
    
    Args:
        target: 目标名称
        rooms: 房间字典
        
    Returns:
        是否是房间
    """
    return target in rooms


def is_large_object(target: str, room_id: str, rooms: Dict[str, Any]) -> bool:
    """
    检查目标是否是指定房间中的大物体
    
    Args:
        target: 目标名称  
        room_id: 房间ID
        rooms: 房间字典
        
    Returns:
        是否是大物体
    """
    if room_id not in rooms:
        return False
    
    room = rooms[room_id]
    return target in room.get("large_objects", {})


def get_action_for_target(target: str, current_room: str, rooms: Dict[str, Any]) -> str:
    """
    根据目标类型生成合适的动作
    
    Args:
        target: 目标名称
        current_room: 当前房间
        rooms: 房间字典
        
    Returns:
        合适的动作
    """
    if is_room(target, rooms):
        # 目标是房间 -> 使用 goto
        return f"goto({target})"
    elif is_large_object(target, current_room, rooms):
        # 目标是大物体 -> 使用 scan（不能移动到大物体）
        return f"scan({target})"
    else:
        # 其他情况 -> 可能是小物体或未知目标，使用 scan
        return f"scan({target})"
