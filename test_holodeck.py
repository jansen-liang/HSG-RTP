import os
import torch
import clip  # 需安装 git+https://github.com/openai/CLIP.git
from langchain_openai import OpenAI
# 假设你提供的那个文件保存为 floor_plan_generator.py
from ai2holodeck.generation.rooms import FloorPlanGenerator

# 1. 初始化 (这一步比较重，只做一次)
def init_generator():
    print("正在初始化 CLIP 和 LLM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载 CLIP (用于材质选择，原代码必须)
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 加载 LLM (你需要设置环境变量 OPENAI_API_KEY)
    llm = OpenAI(temperature=0.7, max_tokens=2000)
    # 实例化你提供的那个类
    generator = FloorPlanGenerator(
        clip_model=model, 
        clip_process=preprocess, 
        clip_tokenizer=clip.tokenize,
        llm=llm
    )
    return generator

# 2. 定义每层楼的“硬约束” (关键步骤！)
# 这是解决“每层楼不一样但电梯必须对齐”的核心
COMMON_CONSTRAINT = """
CRITICAL REQUIREMENT:
1. You MUST generate a room named 'ElevatorHall' centered at coordinates (0,0).
2. The ElevatorHall size must be approx 4x4 meters.
3. This is the central hub; corridors should radiate from or connect to this hall.
"""

def generate_building_layout():
    generator = init_generator()
    
    # 定义你的大楼结构
    floors_config = [
        {"id": "F1", "type": "Hotel Lobby with reception and lounge"},
        {"id": "F2", "type": "Hotel Guest Floor with many bedrooms and a long corridor"},
        {"id": "F3", "type": "Hotel Gym and Spa floor"},
    ]

    building_data = {}

    for floor in floors_config:
        print(f"\n🏗️ 正在生成 {floor['id']} ({floor['type']})...")
        
        # 构造输入 scene 字典 (适配原代码接口)
        scene_input = {
            "query": floor['type'],
            # raw_floor_plan 留空，让它去调 LLM 生成
        }
        
        # === 核心调用 ===
        # 把“电梯在(0,0)”作为 additional_requirements 传入
        # 原代码第 46 行会把这个拼接到 Prompt 里
        try:
            rooms_data = generator.generate_rooms(
                scene=scene_input,
                additional_requirements=COMMON_CONSTRAINT,
                visualize=True # 开启可视化，生成图片看看效果
            )
            
            # 保存数据
            building_data[floor['id']] = rooms_data
            print(f"✅ {floor['id']} 生成成功，包含 {len(rooms_data)} 个房间")
            
        except Exception as e:
            print(f"❌ {floor['id']} 生成失败: {e}")

    return building_data

# 3. 运行
if __name__ == "__main__":
    # Configure OPENAI_API_KEY in the environment before using the API.
    layout_data = generate_building_layout()
    
    # 在这里，layout_data 就包含了所有楼层的 vertices (多边形坐标)
    # 你可以把它们转存到你的 SceneVisualizer 里去画图了
