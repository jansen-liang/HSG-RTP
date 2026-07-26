import sys
import json
import yaml
import re
import random
import time
import contextlib
import os
from pathlib import Path
from tqdm import tqdm
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent
sys.path.append(str(ROOT_DIR.parent))
from utils.llmagent import llm_query


# --- 1. 简易颜色打印工具 (替代 Logger) ---
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def log_info(msg):
    timestamp = time.strftime("%H:%M:%S")
    # 使用 tqdm.write 打印，确保不破坏进度条
    tqdm.write(f"{timestamp} - {GREEN}INFO{RESET} - {msg}")

def log_warn(msg):
    timestamp = time.strftime("%H:%M:%S")
    tqdm.write(f"{timestamp} - {YELLOW}WARNING{RESET} - {msg}")

def log_error(msg):
    timestamp = time.strftime("%H:%M:%S")
    tqdm.write(f"{timestamp} - {RED}ERROR{RESET} - {msg}")


# --- 2. 静音工具 ---
@contextlib.contextmanager
def suppress_output():
    """屏蔽 stdout/stderr，防止 llm_query 打印 [ANSWER] 刷屏"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# --- 3. 核心逻辑 ---

def load_yaml_keys(path: Path) -> list:
    if not path.exists():
        log_warn(f"File not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return list(data.keys())

def build_architect_prompt(scene_name: str, object_list: list) -> str:
    """
    完整 Prompt
    """
    display_objects = object_list
    if len(object_list) > 350:
        display_objects = random.sample(object_list, 350)
    objects_str = ", ".join(display_objects)

    return f"""
    [Role]
    You are a Senior Architect and Logic Engine for a Procedural Generation System.

    [Task]
    You are designing a building of type: "{scene_name}".
    Based on the available assets below, perform two steps:
    1. INFER 3-6 distinct, functional Room Types that belong in this building (e.g., for a 'hospital', infer 'operating_room', 'ward', 'office').
    2. GENERATE spatial layout rules SPECIFIC to each room type.

    [Available Assets Sample]
    [{objects_str}]

    [Requirements]
    Output a single JSON object where the **KEYS represent the Room Types**.
    
    For EACH room type, you must define the following 3 sections with strict formats:

    1. "placement_rules" (Spatial & Physics):
       - Define WHERE small items can be placed.
       - Relations: "on", "in", "under", "next_to", "against".
       - Format: "item_name": {{ "relation": ["target1", "target2"] }}
       - Constraint: STRICTLY consider SIZE. Do not put a bed in a drawer.

    2. "functional_groups" (Layout):
       - Define sets of furniture that belong together.
       - "anchor": The main furniture (e.g., table).
       - "members": Associated items (e.g., chair).
       - "relation": "next_to", "in_front_of", "on_and_next", "surrounds".

    3. "state_context" (Semantic States):
       - Define how a container/surface affects an item's state.
       - Format: "ContainerName": {{ "implies_item_state": {{ "state_key": value }} }}
       - Example: "sink": {{ "implies_item_state": {{ "is_wet": true }} }}

    [Output JSON Schema - CRITICAL]
    You MUST output a nested structure. DO NOT output a flat list.
    
    {{
      "inferred_room_name_1": {{
          "functional_groups": [ 
              {{ "anchor": "main_furniture", "members": ["item1", "item2"], "relation": "next_to" }} 
          ],
          "placement_rules": {{ 
              "small_item": {{ "on": ["surface1", "surface2"], "in": ["container1"] }} 
          }},
          "state_context": {{ 
              "container": {{ "implies_item_state": {{ "is_dirty": true }} }} 
          }}
      }},
      "inferred_room_name_2": {{
          ... (rules specific to this room)
      }}
    }}

    IMPORTANT: 
    - Use common sense specific to "{scene_name}".
    - Ensure furniture is assigned to the correct room type.
    - Output STRICT JSON only.
    """

def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        pass
    return {}

def process_scene_rules(scene_yaml_path: Path, common_keys: list, output_dir: Path):
    scene_name = scene_yaml_path.stem.replace("_generated", "")
    
    if "behavior" in scene_name or "common" in scene_name:
        return

    log_info(f"Generating rules for Scene: {scene_name}...")
    
    scene_objects = load_yaml_keys(scene_yaml_path)
    all_objects = list(set(scene_objects + common_keys))
    
    if not all_objects:
        log_warn(f"No objects found for {scene_name}, skipping.")
        return

    prompt = build_architect_prompt(scene_name, all_objects)
    
    try:
        response = ""
        with suppress_output():
            response = llm_query(
                system_prompt="Output valid JSON only.",
                user_query=prompt,
                agent="gemini-3-pro-preview-thinking-*", 
                timeout=600 
            )
        
        rules_data = extract_json(response)
        
        if rules_data and isinstance(rules_data, dict):
            room_list = list(rules_data.keys())
            log_info(f"Identified {len(room_list)} rooms: {room_list}")
            
            output_path = output_dir / f"{scene_name}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            log_info(f"Rules saved to: {output_path}")
        else:
            log_error(f"Failed to parse JSON for {scene_name}.")

    except Exception as e:
        log_error(f"Exception during generation: {e}")

def main():
    data_dir = ROOT_DIR / "data"
    objects_dir = data_dir / "objects"
    rules_dir = data_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    
    common_path = objects_dir / "common.yaml"
    common_keys = load_yaml_keys(common_path) if common_path.exists() else []
    
    target_files = list(objects_dir.glob("*_generated.yaml"))
    
    if not target_files:
        print(f"{YELLOW}No generated library files found.{RESET}")
        return

    print(f"{GREEN}Found {len(target_files)} scene libraries. Starting batch processing...{RESET}")

    # --- 进度条循环 ---
    pbar = tqdm(target_files, desc="Progress", unit="scene", dynamic_ncols=True, colour='green')
    
    for yaml_file in pbar:
        current_scene = yaml_file.stem.replace("_generated", "")
        pbar.set_postfix(current=current_scene)
        process_scene_rules(yaml_file, common_keys, rules_dir)

    print(f"{GREEN}Batch processing completed.{RESET}")

if __name__ == "__main__":
    main()