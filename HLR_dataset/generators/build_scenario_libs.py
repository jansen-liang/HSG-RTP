import yaml
import logging
import colorlog  
from pathlib import Path


# 配置彩色日志
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',    
        'WARNING':  'yellow',   
        'ERROR':    'red',      
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger("ScenarioBuilder")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 避免重复打印 (如果你在 Jupyter 或多次 reload 时可能会遇到)
logger.propagate = False

# --- 路径配置 (保持不变) ---
DATA_DIR = Path("../data/objects")
FULL_LIB_PATH = DATA_DIR / "behavior_full.yaml"

# --- 核心：场景映射配置表 ---
# 这里定义每个场景需要哪些物体，以及“逻辑借用”关系
# 格式: "目标物体名": "源物体名" (如果名字一样可以省略，或者用列表列出关键词)
SCENARIO_CONFIGS = {
    "hospital": {
        "include_types": [
            "bed", "chair", "table", "computer", "sink", "trash", "bin", 
            "monitor", "screen", "glove", "mask", "soap", "dispenser", 
            "laptop", "curtain", "towel", "blanket", "pillow", "cabinet", 
            "shelf", "door", "floor", "wall", "light", "lamp",
            "elevator_door" 
        ],
        "mappings": {
            "hospital_bed": "bed",
            "surgical_table": "table",
            "autoclave": "microwave",
            # [修正] 找到了，叫 washer
            "mri_scanner": "washer", 
            "x_ray_machine": "printer",
            "surgical_light": "floor_lamp",
            "medical_cart": "handcart",      
            "test_tube": "bottle",
            "scalpel": "knife",
            "biohazard_bin": "ashcan",
            "elevator_button": "switch",
            "elevator_panel": "switch"
        }
    },
    
    "supermarket": {
        "include_types": [
            "fruit", "vegetable", "meat", "fish", "bread", "food", "drink", 
            "bottle", "can", "jar", "box", "bag", "sack", "basket", 
            "shelf", "cabinet", "counter", "register", "money", "card",
            "door", "gate", "sign", "light",
            "elevator_door"
        ],
        "mappings": {
            "checkout_counter": "counter",
            # [修正] 用 vending_machine 代替自助机 (物理形态完美)
            "self_checkout_kiosk": "vending_machine",    
            "shopping_cart": "handcart",     
            "shopping_basket": "wicker_basket", 
            "pallet": "pallet",              
            "freezer": "electric_refrigerator", 
            "freight_elevator_door": "elevator_door",
            "elevator_button": "switch"
        }
    },
    
    "hotel": {
        "include_types": [
            "bed", "sofa", "chair", "table", "tv", "remote", "phone", 
            "towel", "soap", "shampoo", "toothbrush", "toothpaste", 
            "sheet", "pillow", "blanket", "curtain", "rug", "lamp",
            "kettle", "cup", "glass", "plate", "tray", "luggage", "suitcase",
            "elevator_door"
        ],
        "mappings": {
            "luggage_cart": "handcart",
            "room_card": "card",
            "minibar": "electric_refrigerator",
            # [修正] 既然没有 safe，就用 cabinet (柜子) 代替
            "safe_box": "cabinet",
            "elevator_button": "switch"
        }
    },
    
    "office": {
         "include_types": [
             "desk", "table", "chair", "computer", "screen", "monitor", 
             "keyboard", "mouse", "laptop", "printer", "scanner", "phone",
             "paper", "pen", "pencil", "notebook", "book", "folder", "binder",
             "cabinet", "shelf", "trash", "plant", "water_cooler", "coffee_machine",
             "elevator_door"
         ],
         "mappings": {
             "conference_table": "table",
             "office_chair": "chair",
             "photocopier": "printer",
             "elevator_button": "switch"
         }
    },
    
    "residential": {
        "include_types": [
            "table", "chair", "sofa", "bed", "lamp", "tv", "appliance", 
            "kitchen", "bathroom", "bedroom", "decoration", "plant", "toy"
        ],
        "mappings": {
             "dining_table": "table",
             "living_room_sofa": "sofa"
        }
    },
    
    "teaching_building": {
        "include_types": [
            "desk", "chair", "blackboard", "whiteboard", "marker", "eraser",
            "computer", "projector", "screen", "book", "backpack", "notebook",
            "trash", "recycling_bin", "cabinet", "clock",
            "elevator_door"
        ],
        "mappings": {
             "lecture_desk": "desk",
             "student_chair": "chair",
             "podium": "cabinet",
             "elevator_button": "switch"
        }
    },

    "library": {
        "include_types": [
            "bookshelf", "shelf", "book", "table", "chair", "lamp", 
            "computer", "printer", "scanner", "newspaper",
            "couch", "sofa", "rug", "plant",
            "elevator_door"
        ],
        "mappings": {
             "reading_table": "table",
             "librarian_counter": "counter",
             "book_cart": "handcart",  
             # [修正] 图书馆自助机也用 vending_machine
             "self_checkout_machine": "vending_machine",
             "elevator_button": "switch"
        }
    }
}

def load_full_library():
    if not FULL_LIB_PATH.exists():
        logger.error(f"Full library not found at {FULL_LIB_PATH}")
        return {}
    with open(FULL_LIB_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_scenarios():
    full_lib = load_full_library()
    if not full_lib: return

    for scene_name, config in SCENARIO_CONFIGS.items():
        logger.info(f"Building library for: {scene_name}...")
        scene_lib = {}

        # 1. 自动筛选通用物体 (基于 include_types 关键词搜索)
        # 这是一个简单的模糊搜索，把所有包含关键词的物体都加进来
        keywords = config.get("include_types", [])
        for obj_name, entries in full_lib.items():
            # entries 是个列表，通常取第一个作为参考
            ref_entry = entries[0]
            obj_type = ref_entry.get("type", "")
            
            # 如果物体名字或类型包含关键词
            if any(k in obj_name for k in keywords) or any(k in obj_type for k in keywords):
                scene_lib[obj_name] = entries

        # 2. 处理逻辑映射 (The Logic Remapping)
        mappings = config.get("mappings", {})
        for new_name, source_name in mappings.items():
            if source_name in full_lib:
                # 复制源物体的逻辑
                source_entries = full_lib[source_name]
                new_entries = []
                for entry in source_entries:
                    new_entry = entry.copy()
                    new_entry["original_source"] = source_name # 标记来源
                    new_entry["logic_source"] = "mapped_from_behavior"
                    new_entries.append(new_entry)
                
                scene_lib[new_name] = new_entries
                logger.info(f"  Mapped {new_name} <- {source_name}")
            else:
                logger.warning(f"  Source object '{source_name}' not found for mapping '{new_name}'")

        # 3. 保存
        output_path = DATA_DIR / f"{scene_name}_generated.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(scene_lib, f, sort_keys=False, allow_unicode=True)
        logger.info(f"Saved {len(scene_lib)} objects to {output_path}")

if __name__ == "__main__":
    build_scenarios()