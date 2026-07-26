import os
import re
from collections import Counter
from pathlib import Path

# --- 🎯 核心修正：指向正确的子目录 ---
BDDL_ROOT = Path(os.getenv(
    "BDDL_ACTIVITY_DIR",
    Path(__file__).resolve().parents[2] / "3rdparty" / "BEHAVIOR-1K" / "bddl3" / "bddl" / "activity_definitions",
))

def analyze_stats():
    if not BDDL_ROOT.exists():
        print(f"❌ Critical Error: Path still not found: {BDDL_ROOT}")
        print("Please double check the path structure.")
        return

    # 1. 统计任务总数 (Activities)
    # BEHAVIOR 的结构通常是 activity_definitions/cleaning_the_floor/problem0.bddl
    all_activities = [d for d in BDDL_ROOT.iterdir() if d.is_dir()]
    total_activities = len(all_activities)

    goal_predicates = Counter()
    
    print(f"🔄 Scanning {total_activities} activities in {BDDL_ROOT.name}...")

    for activity_dir in all_activities:
        # 通常 BDDL 会有 problem0.bddl
        problem_file = activity_dir / "problem0.bddl"
        if not problem_file.exists():
            continue

        try:
            with open(problem_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # --- 核心提取逻辑 ---
                # 提取 (:goal ...) 块里的谓词
                # 正则解释：寻找左括号后紧跟单词，忽略 ? 变量
                # 比如 (inside ?a ?b) -> 提取 'inside'
                matches = re.findall(r'\(\s*([a-zA-Z0-9_]+)\s+[?]', content)
                
                # 过滤掉逻辑连接词，只保留真正的物理/状态谓词
                ignore_list = {'and', 'not', 'exists', 'forall', 'or', 'imply', 'preference'}
                valid_preds = [m for m in matches if m not in ignore_list]
                
                goal_predicates.update(valid_preds)
        except Exception as e:
            print(f"⚠️ Error reading {activity_dir.name}: {e}")

    # --- 输出报告 ---
    print("\n" + "="*50)
    print(f"📊 BEHAVIOR-1K Data-Driven Analysis")
    print("="*50)
    print(f"🔹 Total Scenarios: {total_activities}")
    print("-" * 50)
    
    print(f"🔹 Top 25 Predicates (The 'DNA' of Subtasks):")
    print(f"   (Use these to define your H-Sim Task Templates)")
    print("-" * 50)
    
    # 打印前 25 个最高频的状态
    for i, (pred, count) in enumerate(goal_predicates.most_common(25), 1):
        print(f"   {i:02d}. {pred:<20} : {count} occurrences")

    print("-" * 50)
    print("🚀 Recommendation for H-Sim:")
    
    # 根据数据动态生成建议
    top_keys = [k for k, v in goal_predicates.most_common(25)]
    
    if 'inside' in top_keys or 'on_top' in top_keys:
        print("   ✅ Logistics Task is a MUST (inside/on_top are dominant).")
    if 'cooked' in top_keys or 'hot' in top_keys:
        print("   ✅ Cooking Task is highly relevant.")
    if 'dusty' in top_keys or 'stained' in top_keys or 'soaked' in top_keys:
        print("   ✅ Cleaning Task is highly relevant.")
    if 'toggled_on' in top_keys or 'open' in top_keys:
        print("   ✅ Device Operation Task is highly relevant.")

if __name__ == "__main__":
    analyze_stats()
