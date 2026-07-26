"""主入口 - HLR 数据管道"""

import json
import yaml
import time
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from typing import Dict, List
from tqdm import tqdm
from utils.scene_loader import load_scenes, get_scene_stats
from utils.task_generator import generate_tasks
from utils.action_planner import plan_actions, generate_global_plan
from utils.sample_builder import build_optimized_streaming_samples, validate_sample_quality
from utils.simulator import simulate_execution
from utils.llmagent import llm_query
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HLR Data Pipeline')
    
    parser.add_argument('--scenes', nargs='+', 
                       default=["hotel", "supermarket", "allensville", "office"],
                       help='Scene names to process')
    parser.add_argument('--task-types', nargs='+',
                       default=["delivery", "tidying", "guidance"], 
                       help='Task types to generate')
    parser.add_argument('--difficulties', nargs='+',
                       default=["easy", "medium", "hard"],
                       help='Difficulty levels')
    parser.add_argument('--max-tasks', type=int, default=10,
                       help='Maximum tasks per scene')
    parser.add_argument('--easy-prop', type=float, default=0.5,
                       help='Proportion of easy tasks')
    parser.add_argument('--medium-prop', type=float, default=0.3,
                       help='Proportion of medium tasks') 
    parser.add_argument('--hard-prop', type=float, default=0.2,
                       help='Proportion of hard tasks')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--llm-agent', type=str, default='gpt-4o-mini',
                       help='LLM agent to use for instruction generation')
    
    return parser.parse_args()

def build_config_from_args(args):
    """从命令行参数构建配置"""
    return {
        "scenes": args.scenes,
        "task_types": args.task_types,
        "difficulties": args.difficulties,
        "difficulty_proportions": {
            "easy": args.easy_prop,
            "medium": args.medium_prop, 
            "hard": args.hard_prop
        },
        "max_tasks_per_scene": args.max_tasks,
        "output_file": args.output,
        "llm_agent": args.llm_agent
    }

def generate_instruction(subtasks: List[str], task, scene_name: str, scene: dict, config: dict = None) -> str:
    """使用大模型生成自然语言指令"""

    # 导入LLM接口
    sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

    task_type = task.task_type.value
    params = task.parameters
    
    # 构建任务描述
    if task_type == "delivery":
        objects = params.get("objects", [])
        source_room = params.get("source_room", "unknown")
        target_rooms = params.get("target_rooms", [])
        
        task_desc = f"Task: Deliver {', '.join(objects)} from {source_room} to {', '.join(target_rooms)}"
    
    elif task_type == "tidying":
        objects = params.get("objects", [])
        source_rooms = params.get("source_rooms", [])
        
        task_desc = f"Task: Tidy up {', '.join(objects)} in {', '.join(source_rooms)}"
    
    elif task_type == "guidance":
        waypoints = params.get("waypoints", [])
        if len(waypoints) > 2:
            # Hard/Medium 任务：显式路径
            via_rooms = ", ".join(waypoints[1:-1])
            task_desc = f"Navigate from {waypoints[0]} to {waypoints[-1]} via {via_rooms}"
        else:
            # Easy 任务：直达
            end_room = waypoints[-1] if waypoints else params.get("end_room", "destination")
            task_desc = f"Navigate to {end_room}"
    
    else:
        task_desc = f"Task: Complete {task_type} task in {scene_name}"
    
    # 系统提示
    system_prompt = """You are simulating different types of people giving instructions to a robot. Create natural, varied instructions that reflect different speaking styles and personalities.

    Speaking styles to vary:
    - Professional/clear: "Could you please move the documents to the filing cabinet?"
    - Casual/friendly: "Mind grabbing that book and putting it on the shelf?"
    - Child-like/playful: "Can you make the toy go to the toy box? It wants to go home!"
    - Elderly/polite: "Would you be so kind as to bring me the glasses from the kitchen?"
    - Busy person/direct: "Take this to room 3, thanks."
    - Hesitant/uncertain: "Um, could you maybe put this... somewhere safe?"

    CRITICAL RULES:
    - Pick ONLY ONE speaking style randomly
    - Output ONLY ONE sentence 
    - NO explanations, NO examples, NO multiple options
    - Be authentic and natural
    - Vary the style for different tasks"""
        
    # 用户查询
    user_query = f"""Convert this robotic task into a natural human instruction:

    Scene: {scene_name} ({task.difficulty.value} difficulty)
    {task_desc}

    IMPORTANT: Choose exactly ONE random speaking style from the list above. Output ONLY the instruction sentence, nothing else. No explanations, no quotes, no additional text."""

    agent = config.get("llm_agent", "glm-4")
    instruction = llm_query(system_prompt, user_query, agent=agent, timeout=10)
    return instruction.strip().strip('"').strip("'")

def process_scene(scene_name: str, scene: dict, config: dict) -> List[Dict]:
    """处理场景，返回记录列表（不写入文件）"""
    records = []
    tasks = generate_tasks(
        scene=scene,
        task_types=config["task_types"],
        difficulties=config["difficulties"],
        max_tasks=config.get("max_tasks_per_scene", 10),
        config=config
    )
    for task in tqdm(tasks, desc=f"Scene: {scene_name}", leave=False):
        subtasks = plan_actions(scene, task)
        if len(subtasks) < 2:
            continue

        instruction = generate_instruction(subtasks, task, scene_name, scene, config)
        streaming_samples, execution_summary = build_optimized_streaming_samples(instruction, subtasks, scene)
        valid_samples = [s for s in streaming_samples if validate_sample_quality(s)]
        if not valid_samples:
            continue

        try:
            simulate_execution(scene, subtasks)
            tqdm.write(f"Simulated successfully: {instruction}")
        except Exception:
            tqdm.write(f"Simulation failed, skipping: {instruction}")
            continue  # 模拟失败 → 跳过该任务

        record = {
            "instruction": instruction,
            "task_info": {
                "type": task.task_type.value,
                "difficulty": task.difficulty.value,
                "description": task.description,
                "parameters": task.parameters
            },
            "execution_summary": execution_summary,
            "streaming_samples": valid_samples,
            "scene_name": scene_name,
            "timestamp": time.time(),
            "sample_count": len(valid_samples)
        }
        records.append(record)
    return records


def main():
    """主函数"""
    print("Starting HLR Data Pipeline")
    
    # 解析命令行参数
    args = parse_args()
    config = build_config_from_args(args)

    # 加载场景
    scenes = load_scenes(config["scenes"])
    stats = get_scene_stats(scenes)
    print(f"Loaded {stats['total_scenes']} scenes")

    output_file = Path(config["output_file"])
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"Output: {output_file}")
    # 收集所有记录
    all_records = []
    # 处理每个场景
    total_samples = 0
    start_time = time.time()

    for scene_name, scene in tqdm(scenes.items(), desc="Total Progress"):
        records = process_scene(scene_name, scene, config)
        all_records.extend(records)
        total_samples += sum(len(r["streaming_samples"]) for r in records)

    # 总结
    elapsed_time = time.time() - start_time
    print(f"Completed: {total_samples} samples in {elapsed_time:.2f}s")
    
    # 写入文件
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"Records: {len(all_records)} written to {output_file}")


if __name__ == "__main__":
    main()