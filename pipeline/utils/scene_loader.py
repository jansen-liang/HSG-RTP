"""场景加载器 - 从 sg/scene_graph.py 加载场景图数据"""

import sys
from pathlib import Path
from typing import Dict, List

def load_scenes(scene_names: List[str], sg_module_path: str = "data/sg/scene_graph.py") -> Dict[str, dict]:
    """
    从 scene_graph.py 模块加载指定场景的场景图数据
    
    Args:
        scene_names: 场景名称列表
        sg_module_path: scene_graph.py 模块路径
        
    Returns:
        场景名称到场景数据的映射
    """
    scenes = {}
    
    try:
        sg_dir = str(Path(__file__).parent.parent / "sg")
        if sg_dir not in sys.path:
            sys.path.insert(0, sg_dir)
        import sg.scene_graph as scene_graph
    except ImportError as e:
        raise ImportError(f"Cannot import scene_graph module: {e}")
    
    # 加载指定的场景
    for scene_name in scene_names:
        scene_name_upper = scene_name.upper()
        if hasattr(scene_graph, scene_name_upper):
            scene_data = getattr(scene_graph, scene_name_upper)
            
            # 验证场景数据完整性
            if validate_scene_data(scene_data, scene_name):
                scenes[scene_name] = scene_data
                print(f"Loaded scene: {scene_name}")
            else:
                print(f"Warning: Invalid scene data: {scene_name}")
        else:
            print(f"Warning: Scene '{scene_name_upper}' not found in scene_graph module")
    
    return scenes

def validate_scene_data(scene: dict, scene_name: str) -> bool:
    """验证场景数据的完整性"""
    required_fields = ["name", "rooms", "agent"]
    
    for field in required_fields:
        if field not in scene:
            print(f"Missing required field '{field}' in scene {scene_name}")
            return False
    
    # 检查房间数据
    if not isinstance(scene["rooms"], dict) or len(scene["rooms"]) == 0:
        print(f"Invalid rooms data in scene {scene_name}")
        return False
    
    # 检查智能体位置
    agent_pos = scene["agent"].get("position")
    if agent_pos not in scene["rooms"]:
        print(f"Agent position '{agent_pos}' not found in rooms for scene {scene_name}")
        return False
    
    return True

def get_scene_stats(scenes: Dict[str, dict]) -> dict:
    """获取场景统计信息"""
    stats = {
        "total_scenes": len(scenes),
        "scene_details": {}
    }
    
    for name, scene in scenes.items():
        room_count = len(scene.get("rooms", {}))
        object_count = sum(
            len(room.get("small_objects", {})) + len(room.get("large_objects", {}))
            for room in scene.get("rooms", {}).values()
        )
        
        stats["scene_details"][name] = {
            "rooms": room_count,
            "objects": object_count,
            "agent_position": scene.get("agent", {}).get("position", "unknown")
        }
    
    return stats

def get_scene_stats(scenes: Dict[str, dict]) -> dict:
    """获取场景统计信息"""
    stats = {
        "total_scenes": len(scenes),
        "scene_details": {}
    }
    
    for name, scene in scenes.items():
        room_count = len(scene.get("rooms", {}))
        object_count = sum(
            len(room.get("small_objects", {})) + len(room.get("large_objects", {}))
            for room in scene.get("rooms", {}).values()
        )
        
        stats["scene_details"][name] = {
            "rooms": room_count,
            "objects": object_count,
            "agent_position": scene.get("agent", {}).get("position", "unknown")
        }
    
    return stats