import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.scenegraph import SceneGraph
from visualization.visualizer_3d import SceneVisualizer
import os
import argparse

parser = argparse.ArgumentParser(description="Visualize a generated HLR scene graph")
parser.add_argument(
    "json_path",
    nargs="?",
    default=str(Path(__file__).resolve().parents[1] / "data" / "scene_graphs" / "hospital_scene_0.json"),
)
JSON_PATH = parser.parse_args().json_path

# 2. 读取文件
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 如果 JSON 里包了一层 "scene_graph" 就取出来，否则直接用
graph_data = data.get("scene_graph", data)

# 4. 还原对象并画图
scene = SceneGraph.from_dict(graph_data)
viz = SceneVisualizer(scene)

output_filename = os.path.splitext(JSON_PATH)[0] + ".html"
viz.generate_html(output_filename)

print(f"✅ 可视化已保存: {output_filename}")

