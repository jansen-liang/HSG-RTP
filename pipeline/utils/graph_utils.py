"""图工具模块 - 用于场景图的全局和局部视图裁剪"""

from typing import Dict, Any
from copy import deepcopy

def get_global_view(scene: dict) -> dict:
    """
    获取场景的全局视图，只保留宏观信息
    
    Args:
        scene: 完整的场景数据
        
    Returns:
        简化的全局视图
    """
    global_view = {
        "name": scene["name"],
        "agent": {
            "position": scene["agent"]["position"]
        }
    }
    
    # 保留完整的机器人状态信息（对全局推理很重要）
    if "state" in scene["agent"]:
        global_view["agent"]["state"] = scene["agent"]["state"]
    if "battery" in scene["agent"]:
        global_view["agent"]["battery"] = scene["agent"]["battery"]
    if "type" in scene["agent"]:
        global_view["agent"]["type"] = scene["agent"]["type"]
    
    # 保留宏观区域信息（如果存在）
    if "macro_zones" in scene:
        global_view["macro_zones"] = scene["macro_zones"]
    
    # 简化房间信息，只保留楼层和邻居关系
    global_view["rooms"] = {}
    for room_id, room_info in scene["rooms"].items():
        global_view["rooms"][room_id] = {
            "floor": room_info.get("floor", 1),
            "neighbor": room_info.get("neighbor", [])
        }
        
        # 保留房间类型信息（如果存在）
        if "type" in room_info:
            global_view["rooms"][room_id]["type"] = room_info["type"]
    
    return global_view

def get_local_view(scene: dict, room_id: str) -> dict:
    """
    获取指定房间的局部视图，包含详细的物品信息
    
    Args:
        scene: 完整的场景数据
        room_id: 目标房间ID
        
    Returns:
        指定房间的详细视图
    """
    if room_id not in scene["rooms"]:
        raise ValueError(f"Room {room_id} not found in scene")
    
    local_view = {
        "name": scene["name"],
        "current_room": room_id,
        "agent": {
            "position": room_id,
            "state": scene["agent"].get("state", {})
        }
    }
    
    # 获取当前房间的完整信息
    room_info = deepcopy(scene["rooms"][room_id])
    local_view["room"] = room_info
    
    # 限制显示的物品数量（避免信息过载）
    max_objects = 20
    
    # 处理小物品
    if "small_objects" in room_info and len(room_info["small_objects"]) > max_objects:
        # 保留前 max_objects 个物品
        items = list(room_info["small_objects"].items())[:max_objects]
        local_view["room"]["small_objects"] = dict(items)
        local_view["room"]["_truncated_small_objects"] = True
    
    # 处理大物品
    if "large_objects" in room_info and len(room_info["large_objects"]) > max_objects:
        # 保留前 max_objects 个物品
        items = list(room_info["large_objects"].items())[:max_objects]
        local_view["room"]["large_objects"] = dict(items)
        local_view["room"]["_truncated_large_objects"] = True
    
    return local_view

def get_room_summary(scene: dict, room_id: str) -> dict:
    """
    获取房间的摘要信息
    
    Args:
        scene: 场景数据
        room_id: 房间ID
        
    Returns:
        房间摘要信息
    """
    if room_id not in scene["rooms"]:
        return {"error": f"Room {room_id} not found"}
    
    room = scene["rooms"][room_id]
    
    summary = {
        "room_id": room_id,
        "floor": room.get("floor", 1),
        "type": room.get("type", "unknown"),
        "neighbors": room.get("neighbor", []),
        "small_objects_count": len(room.get("small_objects", {})),
        "large_objects_count": len(room.get("large_objects", {}))
    }
    
    # 列出主要物品类型
    small_objects = room.get("small_objects", {})
    large_objects = room.get("large_objects", {})
    
    # 统计物品类型
    object_types = {}
    for obj_name, obj_info in {**small_objects, **large_objects}.items():
        obj_type = obj_info.get("type", "unknown") if isinstance(obj_info, dict) else "unknown"
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
    
    summary["object_types"] = object_types
    
    return summary

def filter_objects_by_type(room_data: dict, object_types: list) -> dict:
    """
    按类型过滤房间中的物品
    
    Args:
        room_data: 房间数据
        object_types: 要保留的物品类型列表
        
    Returns:
        过滤后的房间数据
    """
    filtered_room = deepcopy(room_data)
    
    # 过滤小物品
    if "small_objects" in filtered_room:
        filtered_small = {}
        for obj_name, obj_info in filtered_room["small_objects"].items():
            obj_type = obj_info.get("type", "unknown") if isinstance(obj_info, dict) else "unknown"
            if obj_type in object_types:
                filtered_small[obj_name] = obj_info
        filtered_room["small_objects"] = filtered_small
    
    # 过滤大物品
    if "large_objects" in filtered_room:
        filtered_large = {}
        for obj_name, obj_info in filtered_room["large_objects"].items():
            obj_type = obj_info.get("type", "unknown") if isinstance(obj_info, dict) else "unknown"
            if obj_type in object_types:
                filtered_large[obj_name] = obj_info
        filtered_room["large_objects"] = filtered_large
    
    return filtered_room

def get_accessible_rooms(scene: dict, start_room: str, max_distance: int = 2) -> list:
    """
    获取从起始房间可达的房间列表
    
    Args:
        scene: 场景数据
        start_room: 起始房间
        max_distance: 最大距离
        
    Returns:
        可达房间列表
    """
    from collections import deque
    
    if start_room not in scene["rooms"]:
        return []
    
    visited = set()
    queue = deque([(start_room, 0)])
    accessible_rooms = []
    
    while queue:
        current_room, distance = queue.popleft()
        
        if current_room in visited or distance > max_distance:
            continue
            
        visited.add(current_room)
        accessible_rooms.append(current_room)
        
        # 添加邻居房间
        neighbors = scene["rooms"][current_room].get("neighbor", [])
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    
    return accessible_rooms

def create_subgraph(scene: dict, room_ids: list) -> dict:
    """
    创建包含指定房间的子图
    
    Args:
        scene: 完整场景数据
        room_ids: 要包含的房间ID列表
        
    Returns:
        子场景图
    """
    subgraph = {
        "name": f"{scene['name']}_subgraph",
        "agent": deepcopy(scene["agent"]),
        "rooms": {}
    }
    
    # 复制指定房间的数据
    for room_id in room_ids:
        if room_id in scene["rooms"]:
            subgraph["rooms"][room_id] = deepcopy(scene["rooms"][room_id])
            
            # 过滤邻居，只保留子图中存在的房间
            if "neighbor" in subgraph["rooms"][room_id]:
                filtered_neighbors = [
                    neighbor for neighbor in subgraph["rooms"][room_id]["neighbor"]
                    if neighbor in room_ids
                ]
                subgraph["rooms"][room_id]["neighbor"] = filtered_neighbors
    
    return subgraph