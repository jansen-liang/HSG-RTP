"""仿真器 - 执行动作并更新场景状态"""

from copy import deepcopy
from typing import Dict, List, Any
import re

def simulate_execution(initial_scene: dict, subtasks: list) -> dict:
    """
    仿真执行整个动作序列，返回最终状态
    
    Args:
        initial_scene: 初始场景状态
        subtasks: 动作序列
        
    Returns:
        执行后的最终场景状态
    """
    state = deepcopy(initial_scene)
    
    for action in subtasks:
        state = execute_subtask(state, action)
        if state is None:
            # 执行失败，返回上一个有效状态
            print(f"Action execution failed: {action}")
            break
    
    return state

def execute_subtask(scene: dict, action: str) -> dict:
    """
    执行单个动作，更新场景状态
    
    Args:
        scene: 当前场景状态
        action: 要执行的动作
        
    Returns:
        更新后的场景状态，如果执行失败返回None
    """
    if not action or not isinstance(action, str):
        return scene
    
    # 解析动作
    action = action.strip()
    
    if action.startswith("goto("):
        return execute_goto(scene, action)
    elif action.startswith("scan("):
        return execute_scan(scene, action)
    elif action.startswith("pick("):
        return execute_pick(scene, action)
    elif action.startswith("place("):
        return execute_place(scene, action)
    elif action.startswith("press("):
        return execute_press(scene, action)
    elif action.startswith("wait("):
        # 等待动作，不改变场景状态
        return scene
    else:
        print(f"Unknown action type: {action}")
        return scene

def execute_goto(scene: dict, action: str) -> dict:
    """执行移动动作"""
    # 解析目标位置: goto(room_id) 或 goto(object_id)
    target = action[5:-1]  # 去掉 "goto(" 和 ")"
    
    new_scene = deepcopy(scene)
    current_pos = scene["agent"]["position"]
    
    # 检查目标是房间还是物品
    if target in scene["rooms"]:
        # 移动到房间
        new_scene["agent"]["position"] = target
        return new_scene
    else:
        # 移动到物品（在当前房间内）
        current_room = scene["rooms"][current_pos]
        
        # 检查物品是否在当前房间
        if (target in current_room.get("small_objects", {}) or 
            target in current_room.get("large_objects", {})):
            # 更新智能体状态，表示靠近了该物品
            new_scene["agent"]["near_object"] = target
            return new_scene
        else:
            print(f"Object {target} not found in current room {current_pos}")
            return scene

def execute_scan(scene: dict, action: str) -> dict:
    """执行扫描动作"""
    # 解析扫描目标: scan(room_id) 或 scan(object_id)
    target = action[5:-1]  # 去掉 "scan(" 和 ")"
    
    new_scene = deepcopy(scene)
    
    # 记录扫描历史
    if "scan_history" not in new_scene["agent"]:
        new_scene["agent"]["scan_history"] = []
    
    new_scene["agent"]["scan_history"].append(target)
    new_scene["agent"]["last_scanned"] = target
    
    return new_scene

def execute_pick(scene: dict, action: str) -> dict:
    """执行拾取动作"""
    # 解析物品: pick(object_id)
    obj_id = action[5:-1]  # 去掉 "pick(" 和 ")"
    
    new_scene = deepcopy(scene)
    current_pos = scene["agent"]["position"]
    current_room = new_scene["rooms"][current_pos]
    
    # 检查物品是否存在且可操作
    found_in = None
    if obj_id in current_room.get("small_objects", {}):
        found_in = "small_objects"
    elif obj_id in current_room.get("large_objects", {}):
        # 大物品不能拾取，直接返回原场景
        print(f"Error: Cannot pick large object {obj_id}. Large objects are not movable.")
        return scene
    
    if found_in:
        # 移除物品从房间
        obj_info = current_room[found_in].pop(obj_id)
        # 把物品的关系属性改变
        obj_info["relation"] = {"in": "inventory"}  
        # 添加到智能体库存
        if "inventory" not in new_scene["agent"]:
            new_scene["agent"]["inventory"] = {}
        
        new_scene["agent"]["inventory"][obj_id] = obj_info
        new_scene["agent"]["state"] = "holding"
        
        return new_scene
    else:
        print(f"Object {obj_id} not found in current room {current_pos}")
        return scene

def execute_place(scene: dict, action: str) -> dict:
    """执行放置动作"""
    # 解析动作: place(object_id, surface_id)
    parts = action[6:-1].split(", ")  # 去掉 "place(" 和 ")"
    if len(parts) != 2:
        print(f"Invalid place action format: {action}")
        return scene
    
    obj_id, surface_id = parts
    
    new_scene = deepcopy(scene)
    current_pos = scene["agent"]["position"]
    
    # 检查智能体是否持有该物品
    if ("inventory" not in scene["agent"] or 
        obj_id not in scene["agent"]["inventory"]):
        print(f"Agent is not holding {obj_id}")
        return scene
    
    # 移除物品从库存
    obj_info = new_scene["agent"]["inventory"].pop(obj_id)
    if new_scene["agent"].get("holding") == obj_id:
        new_scene["agent"]["holding"] = None
    
    # 放置物品到指定表面
    if surface_id == "floor":
        obj_info["relation"] = {"on": "floor"}
        # 放在地板上（作为小物品）
        if "small_objects" not in new_scene["rooms"][current_pos]:
            new_scene["rooms"][current_pos]["small_objects"] = {}
        new_scene["rooms"][current_pos]["small_objects"][obj_id] = obj_info
    else:
        # 放在指定表面上
        current_room = new_scene["rooms"][current_pos]
        obj_info["relation"] = {"on": surface_id}
        # 检查表面是否存在
        if surface_id in current_room.get("large_objects", {}):
            if "small_objects" not in new_scene["rooms"][current_pos]:
                new_scene["rooms"][current_pos]["small_objects"] = {}
            new_scene["rooms"][current_pos]["small_objects"][obj_id] = obj_info
        else:
            print(f"Surface {surface_id} not found in current room")
            # 默认放在地板上
            if "small_objects" not in new_scene["rooms"][current_pos]:
                new_scene["rooms"][current_pos]["small_objects"] = {}
            new_scene["rooms"][current_pos]["small_objects"][obj_id] = obj_info
    
    return new_scene

def execute_press(scene: dict, action: str) -> dict:
    """执行按压动作"""
    # 解析按钮: press(button_id)
    button_id = action[6:-1]  # 去掉 "press(" 和 ")"
    
    new_scene = deepcopy(scene)
    current_pos = scene["agent"]["position"]
    current_room = scene["rooms"][current_pos]
    
    # 检查按钮是否存在且可按
    if button_id in current_room.get("small_objects", {}):
        button_info = current_room["small_objects"][button_id]
        if isinstance(button_info, dict) and "press" in button_info.get("affordance", []):
            # 记录按钮被按下
            if "pressed_buttons" not in new_scene["agent"]:
                new_scene["agent"]["pressed_buttons"] = []
            
            new_scene["agent"]["pressed_buttons"].append(button_id)
            new_scene["agent"]["last_pressed"] = button_id
            
            # 如果是电梯内按钮，需要更新该按钮的灯状态
            if "elevator_button" in button_id :
                new_scene["rooms"]["elevator_cabin"]["small_objects"][button_id]["state"]["lit"] = True
            if "elevator_call" in button_id:
                current_floor = re.search(r'(\d+)', current_pos).group(1)
                new_scene["rooms"][f"elevator_{current_floor}f"]["small_objects"][button_id]["state"]["lit"] = True
            return new_scene
        else:
            print(f"Button {button_id} cannot be pressed")
            return scene
    else:
        print(f"Button {button_id} not found in current room {current_pos}")
        return scene

def get_execution_summary(initial_scene: dict, final_scene: dict, subtasks: list) -> dict:
    """
    生成执行摘要
    
    Args:
        initial_scene: 初始场景
        final_scene: 最终场景
        subtasks: 执行的动作序列
        
    Returns:
        执行摘要
    """
    summary = {
        "total_actions": len(subtasks),
        "agent_movement": {
            "start_position": initial_scene["agent"]["position"],
            "end_position": final_scene["agent"]["position"]
        },
        "inventory_changes": {},
        "room_changes": {}
    }
    
    # 分析库存变化
    initial_inventory = initial_scene["agent"].get("inventory", {})
    final_inventory = final_scene["agent"].get("inventory", {})
    
    summary["inventory_changes"] = {
        "picked_up": list(set(final_inventory.keys()) - set(initial_inventory.keys())),
        "put_down": list(set(initial_inventory.keys()) - set(final_inventory.keys()))
    }
    
    # 分析房间物品变化
    for room_id in initial_scene["rooms"]:
        initial_objects = set()
        final_objects = set()
        
        # 收集初始和最终的物品
        for obj_type in ["small_objects", "large_objects"]:
            initial_objects.update(initial_scene["rooms"][room_id].get(obj_type, {}).keys())
            final_objects.update(final_scene["rooms"][room_id].get(obj_type, {}).keys())
        
        added = final_objects - initial_objects
        removed = initial_objects - final_objects
        
        if added or removed:
            summary["room_changes"][room_id] = {
                "added": list(added),
                "removed": list(removed)
            }
    
    return summary