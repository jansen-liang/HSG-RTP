import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# 1. 路径配置
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from core.scenegraph import SceneGraph
from core.object_library import get_object_template
from models.nodes import NodeType
from models.edges import SpatialRelation, EdgeCategory

class SceneFactory:
    def __init__(self):
        # 路径定位
        self.rules_dir = PROJECT_ROOT / "data" / "rules"
        print(f"📍 规则目录: {self.rules_dir}")
        self.rules_db = self._load_rules()

    def _load_rules(self) -> Dict[str, dict]:
        """加载规则文件：文件名(不含后缀)作为 Key"""
        db = {}
        if not self.rules_dir.exists():
            print(f"❌ Error: 找不到目录 {self.rules_dir}")
            return db

        for rule_file in self.rules_dir.glob("*.json"):
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    # 例如加载 hospital.json -> key 为 "hospital"
                    # 内容直接就是 {"operating_theatre": ..., "patient_ward": ...}
                    db[rule_file.stem] = json.load(f)
                    print(f"   📖 已加载文件: {rule_file.name} (包含房间: {list(db[rule_file.stem].keys())})")
            except Exception as e:
                print(f"   ❌ 加载失败 {rule_file.name}: {e}")
        return db

    def _invert_placement_rules(self, rules: dict) -> Dict[str, List[str]]:
        """适配您的 JSON 结构：解析 target 列表"""
        surface_map = {}
        placement_rules = rules.get("placement_rules", {})
        
        for item_name, config in placement_rules.items():
            # 您的 JSON 结构是 "target": ["table", "cart"]
            targets = config.get("target", [])
            
            for target in targets:
                if target not in surface_map: 
                    surface_map[target] = []
                surface_map[target].append(item_name)
                
        return surface_map

    def create_scene(self, name: str, scene_type: str = "hospital", n_floors: int = 3, seed: int = None) -> SceneGraph:
        """创建场景"""
        if seed is not None: random.seed(seed)

        scene = SceneGraph(name)
        floors = []
        hallways = {} 

        # --- 1. 生成楼层 ---
        for i in range(n_floors):
            fid = f"F{i+1}"
            scene.add_floor(fid, f"Floor {i+1}", floor_number=i+1)
            floors.append(fid)
            
            # 电梯厅
            hall_id = f"hall_{fid}"
            scene.add_room(hall_id, fid, "Elevator Hall") 
            hallways[fid] = hall_id
            
            # 🔥 动态选择房间类型 🔥
            # 根据 scene_type ("hospital") 去 rules_db 里找所有可用的房间
            available_rooms = []
            if scene_type in self.rules_db:
                available_rooms = list(self.rules_db[scene_type].keys())
            
            # 如果没找到，兜底用 operating_theatre
            if not available_rooms:
                available_rooms = ["operating_theatre"] if scene_type == "hospital" else ["office"]

            for j in range(2): 
                # 随机挑一个，比如有时候是手术室，有时候是病房
                target_room_key = random.choice(available_rooms)
                
                # 房间ID
                rid = f"{target_room_key}_{fid}_{j}"
                scene.add_room(rid, fid, f"{target_room_key} {j+1}")
                scene.connect_rooms(hall_id, rid)
                
                # 填充房间
                self._populate_room(scene, rid, scene_type, target_room_key)

        # --- 2. 电梯 ---
        if n_floors > 1:
            self._setup_elevator(scene, floors, hallways)

        # --- 3. 机器人 ---
        scene.set_agent("robot_01", hallways[floors[0]])
        
        return scene

    def _populate_room(self, scene, room_id, file_key, room_key):
        """直球式填充：去 file_key 文件里找 room_key 数据"""
        
        # 1. 找文件 (hospital.json)
        file_data = self.rules_db.get(file_key)
        if not file_data:
            print(f"⚠️ 没找到文件规则: {file_key}.json")
            return

        # 2. 找房间配置 (operating_theatre)
        # 因为您的 JSON 根 Key 就是房间名，所以直接 get 即可
        room_rules = file_data.get(room_key)
        if not room_rules:
            print(f"⚠️ 在 {file_key}.json 里没找到 Key: '{room_key}'")
            return

        # 3. 获取生成组
        groups = room_rules.get("functional_groups", [])
        # print(f"✅ 正在填充 {room_id} (规则: {room_key}, Anchor数: {len(groups)})")

        # --- 生成固定设施 (Anchors) ---
        created_anchors = set()
        
        for group in groups:
            anchor_type = group.get("anchor")
            
            # 生成 Anchor
            anchor_id = f"{anchor_type}_{room_id}_{random.randint(100,999)}"
            self._create_object(scene, anchor_id, anchor_type)
            scene.place_object_in_room(anchor_id, room_id)
            created_anchors.add(anchor_id)
            
            # 生成 Members
            raw_relation = group.get("relation", "next_to")
            
            for member in group.get("members", []):
                mem_id = f"{member}_{room_id}_{random.randint(1000,9999)}"
                self._create_object(scene, mem_id, member)
                
                # 🔥 关系映射：适配您 JSON 里的 surrounds, on_and_next 等
                if raw_relation in ["on_and_next", "on", "ontop"]:
                    scene.place_object_on_surface(mem_id, anchor_id)
                elif raw_relation in ["surrounds", "next_to", "against"]:
                    scene.connect_objects(mem_id, anchor_id, SpatialRelation.NEXT_TO)
                    scene.place_object_in_room(mem_id, room_id)
                elif raw_relation == "in":
                    scene.connect_objects(anchor_id, mem_id, SpatialRelation.CONTAINS)
                else:
                    scene.connect_objects(mem_id, anchor_id, SpatialRelation.NEXT_TO)
                    scene.place_object_in_room(mem_id, room_id)

        # --- 散落小物品 ---
        placement_map = self._invert_placement_rules(room_rules)
        
        for anchor_id in created_anchors:
            node = scene.get_node(anchor_id)
            if not node: continue
            
            obj_type = getattr(node, "object_type", "")
            allowed_items = placement_map.get(obj_type, [])
            
            if allowed_items:
                # 强制生成 1-3 个小物品
                for _ in range(random.randint(1, 3)):
                    item_type = random.choice(allowed_items)
                    item_id = f"{item_type}_{room_id}_{random.randint(5000,9999)}"
                    self._create_object(scene, item_id, item_type)
                    scene.place_object_on_surface(item_id, anchor_id)

    def _setup_elevator(self, scene: SceneGraph, floors: List[str], hallways: Dict[str, str]):
        lift_id = "elevator_1"
        scene.add_mobile_tool(lift_id, "elevator", initial_location=hallways[floors[0]])

        for fid in floors: 
            btn_id = f"btn_inside_{fid}"
            self._create_object(scene, btn_id, "button")
            scene.connect_objects(lift_id, btn_id, SpatialRelation.CONTAINS)

        for i, fid in enumerate(floors): 
            hall_id = hallways[fid]
            scene.connect_transport_stop(hall_id, lift_id, "sliding_door")
            for direction in (["up", "down"] if 0 < i < len(floors)-1 else ["up"] if i==0 else ["down"]):
                btn_id = f"btn_call_{fid}_{direction}"
                self._create_object(scene, btn_id, "button")
                scene.place_object_in_room(btn_id, hall_id) 
                scene.add_edge(source=btn_id, target=lift_id, relation=SpatialRelation.CONTROLS, category=EdgeCategory.LOGICAL)

    def _create_object(self, scene, obj_id, obj_type):
        """创建物体，处理未知的 object_type"""
        raw = get_object_template(obj_type) or {"object_type": obj_type, "physical_properties": {}}

        # 兼容 object_library 返回 list 的情况（yaml 有时把每个 key 映射为 list）
        if isinstance(raw, list):
            tmpl = raw[0] if raw else {}
            if not isinstance(tmpl, dict):
                tmpl = {}
        elif isinstance(raw, dict):
            tmpl = raw
        else:
            tmpl = {}

        # 兼容字段名称：yaml 中常用 default_states
        if "default_states" in tmpl and "states" not in tmpl:
            tmpl["states"] = tmpl.pop("default_states")

        # 特例：button 仍然被当作不可移动的 fixture
        if "button" in obj_type:
            tmpl = {"object_type": "fixture", "physical_properties": {"movable": False}, "states": tmpl.get("states", {})}

        # 确保字段存在
        states = tmpl.get("states", {}) if isinstance(tmpl.get("states", {}), dict) else {}
        phys = tmpl.get("physical_properties", {}) if isinstance(tmpl.get("physical_properties", {}), dict) else {}

        scene.add_object(
            obj_id, obj_type,
            states=states,
            physical_properties=phys
        )

if __name__ == "__main__":
    OUTPUT_PATH = PROJECT_ROOT / "dataset_output"
    factory = SceneFactory()
    
    configs = [ {"type": "hospital", "count": 1, "floors_range": (3, 6)} ]

    for config in configs:
        s_type = config["type"]
        print(f"🏗️  Building: {s_type}")
        for i in range(config["count"]):
            scene_name = f"{s_type}_scene_{i:03d}"
            try:
                scene = factory.create_scene(scene_name, s_type, random.randint(*config["floors_range"]), seed=i)
                out_file = OUTPUT_PATH / f"{scene_name}.json"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(scene.to_dict(), f, indent=2, ensure_ascii=False)
                print(f"   ✅ Saved: {out_file}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
