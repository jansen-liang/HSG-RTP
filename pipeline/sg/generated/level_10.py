level_10_scene = {
    "name": "level_10",
    
    # 宏观层级：正名！这里就是第 10 层
    "macro_zones": {
        "floor_10_main": {
            "rooms": ["corridor", "room_A", "room_B", "room_C"]
        }
    },
    
    "rooms": {
        "corridor": {
            "floor": "floor_10_main",  # 对应上面的宏观区域
            "neighbor": ["room_A", "room_B", "room_C"],
            "large_objects": {},
            "small_objects": {}
        },
        "room_A": {
            "floor": "floor_10_main",
            "neighbor": ["corridor"],
            "large_objects": {},
            "small_objects": {}
        },
        "room_B": {
            "floor": "floor_10_main",
            "neighbor": ["corridor"],
            "large_objects": {},
            "small_objects": {}
        },
        "room_C": {
            "floor": "floor_10_main",
            "neighbor": ["corridor"],
            "large_objects": {},
            "small_objects": {}
        }
    },
    
    "agent": {
        "position": "corridor",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot",
        "inventory": {}
    }
}