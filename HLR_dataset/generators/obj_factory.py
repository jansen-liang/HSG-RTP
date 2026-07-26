import json
import yaml
import sys
import logging
from pathlib import Path

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ObjFactory")

# --- 路径配置 ---
CURRENT_FILE = Path(__file__).resolve()

# HLR_dataset 根目录
DATASET_ROOT = CURRENT_FILE.parent.parent

# HLR 项目根目录
HLR_ROOT = DATASET_ROOT.parent

# BDDL 数据源路径
BDDL_DATA_DIR = HLR_ROOT / "3rdparty" / "BEHAVIOR-1K" / "bddl3" / "bddl" / "generated_data"

# 输出 YAML 的位置
OUTPUT_YAML_PATH = DATASET_ROOT / "data" / "objects" / "behavior_full.yaml"

class BehaviorObjectFactory:
    def __init__(self):
        """初始化工厂，检查数据路径"""
        self.hierarchy_file = BDDL_DATA_DIR / "output_hierarchy.json"
        
        # 使用正确的拼写 propagated
        self.prop_file = BDDL_DATA_DIR / "propagated_annots_canonical.json"
        
        # 路径检查
        if not BDDL_DATA_DIR.exists():
            logger.error(f"BDDL directory not found at: {BDDL_DATA_DIR}")
            sys.exit(1)
        if not self.hierarchy_file.exists():
            logger.error(f"Hierarchy file not found at: {self.hierarchy_file}")
            sys.exit(1)
        if not self.prop_file.exists():
            logger.error(f"Properties file not found at: {self.prop_file}")
            sys.exit(1)
            
        logger.info(f"BDDL Data Source verified: {BDDL_DATA_DIR}")

    def _load_hierarchy_map(self):
        """建立 Synset -> Parent Type 的映射表"""
        logger.info("Loading hierarchy tree...")
        try:
            with open(self.hierarchy_file, "r", encoding="utf-8") as f:
                tree = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load hierarchy file: {e}")
            sys.exit(1)

        synset_to_type = {}

        def traverse(node, path):
            current_name = node["name"]
            # path 记录父级路径，取倒数第2个作为 meaningful type
            meaningful_type = "unknown"
            if len(path) >= 1:
                meaningful_type = path[-1] 
            
            synset_to_type[current_name] = meaningful_type
            
            if "children" in node:
                new_path = path + [current_name]
                for child in node["children"]:
                    traverse(child, new_path)

        traverse(tree, [])
        return synset_to_type

    def build_library(self):
        """核心逻辑：读取属性表并清洗数据"""
        type_map = self._load_hierarchy_map()
        
        logger.info(f"Loading properties map from {self.prop_file.name}...")
        try:
            with open(self.prop_file, "r", encoding="utf-8") as f:
                full_definitions = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load properties file: {e}")
            sys.exit(1)

        logger.info(f"Found {len(full_definitions)} raw objects definitions.")
        
        hsim_library = {}

        for synset, attributes in full_definitions.items():
            # 1. 清洗名字: "microwave.n.02" -> "microwave"
            simple_name = synset.split(".")[0]
            
            # 2. 获取分类 (从 hierarchy 中查)
            obj_type = type_map.get(synset, "object")
            
            # 3. 提取 Affordances (交互属性)
            affordances = list(attributes.keys())

            # 4. 构建条目
            entry = {
                "synset": synset,
                "type": obj_type,
                "logic_source": "behavior-1k",
                "affordances": affordances,
                "default_states": {}
            }

            # 5. 智能推断默认状态 (Mapping to H-Sim Action Space)
            if "openable" in affordances: 
                entry["default_states"]["isOpen"] = False
            if "toggleable" in affordances: 
                entry["default_states"]["isOn"] = False
            if "dirtyable" in affordances: 
                entry["default_states"]["isClean"] = True
            if "sliceable" in affordances:
                entry["default_states"]["isSliced"] = False
            if "cookable" in affordances:
                entry["default_states"]["isCooked"] = False
            if "heatSource" in affordances: 
                entry["default_states"]["isRunnning"] = False

            # 6. 存入字典
            if simple_name not in hsim_library:
                hsim_library[simple_name] = []
            hsim_library[simple_name].append(entry)
            
        return hsim_library

    def export_yaml(self):
        """导出为 YAML 文件"""
        library = self.build_library()

        OUTPUT_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(library)} object categories to {OUTPUT_YAML_PATH}...")
        with open(OUTPUT_YAML_PATH, "w", encoding="utf-8") as f:
            yaml.dump(library, f, sort_keys=False, allow_unicode=True)
        
        logger.info("Object library generation completed successfully.")


if __name__ == "__main__":
    factory = BehaviorObjectFactory()
    factory.export_yaml()
    
