"""
场景图三维可视化模块

功能：
- 将层次化场景图数据结构转换为三维节点图
- 支持宏观区（红色）、房间（绿色）、物品（蓝色/橙色）三层可视化
- 特别支持电梯cabin特殊楼层的可视化（紫色）
- 显示层内邻居连接和层间父子关系
- 支持从JSON数据集文件直接可视化场景结构
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import json
from collections import defaultdict
from pathlib import Path

# 设置字体支持
def setup_chinese_font():
    """设置字体支持，使用已安装的中文字体"""
    try:
        # 尝试使用WenQuanYi字体（已安装，使用精确名称）
        available_fonts = [
            'WenQuanYi Micro Hei,文泉驛微米黑,文泉驿微米黑',
            'WenQuanYi Zen Hei,文泉驛正黑,文泉驿正黑', 
            'AR PL UMing CN',
            'sans-serif'
        ]
        
        font_set = False
        for font in available_fonts:
            try:
                # 对于多名称字体，取第一个名称
                font_name = font.split(',')[0]
                test_font = fm.FontProperties(family=font_name)
                plt.rcParams['font.family'] = font_name
                font_set = True
                print(f"使用字体: {font_name}")
                break
            except:
                continue
        
        if not font_set:
            plt.rcParams['font.family'] = 'sans-serif'
            print("使用默认字体: sans-serif")
        
        plt.rcParams['axes.unicode_minus'] = False
        
        # 禁用字体相关警告
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
        warnings.filterwarnings('ignore', message='.*Glyph.*missing from font.*')
        
        # 设置matplotlib日志级别，减少字体警告
        import matplotlib
        matplotlib.set_loglevel("ERROR")
        
        return 'Default'
    except:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return 'Default'


def visualize_scene_graph_3d(scene_graph, figsize=(16, 14), save_path=None, scene_name=None):
    """
    创建场景图的三维可视化，特别处理电梯cabin
    
    Args:
        scene_graph: 场景图数据结构
        figsize: 图形大小
        save_path: 保存路径
        scene_name: 场景名称
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    setup_chinese_font()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义层级高度 - 为电梯cabin增加特殊层级
    MACRO_LEVEL = 2.5
    ROOM_LEVEL = 1.5
    ELEVATOR_LEVEL = 0.8  # 电梯cabin的特殊层级
    LARGE_OBJECT_LEVEL = 0.3
    SMALL_OBJECT_LEVEL = -0.1
    
    # 定义颜色方案
    COLORS = {
        'macro': '#FF4444',      # 红色 - 宏观区
        'room': '#44AA44',       # 绿色 - 普通房间
        'elevator': '#8844DD',   # 紫色 - 电梯cabin
        'large_object': '#4488FF',  # 蓝色 - 大型物品
        'small_object': '#FF8844',  # 橙色 - 小型物品
        'elevator_button': '#DD44AA'  # 洋红色 - 电梯按钮
    }
    
    nodes = {}
    edges = []
    
    # 1. 宏观区 (macro_zones)
    macro_zones = scene_graph.get('macro_zones', {})
    macro_count = len(macro_zones)
    for i, (zone_id, zone_data) in enumerate(macro_zones.items()):
        angle = 2 * np.pi * i / macro_count if macro_count > 0 else 0
        x, y = 4.5 * np.cos(angle), 4.5 * np.sin(angle)
        nodes[zone_id] = {'pos': (x, y, MACRO_LEVEL), 'type': 'macro', 'label': zone_id}
    
    # 2. 房间 (rooms) - 区分普通房间和电梯cabin
    rooms = scene_graph.get('rooms', {})
    regular_rooms = {}
    elevator_cabins = {}
    
    # 分离普通房间和电梯cabin
    for room_id, room_data in rooms.items():
        if room_data.get('floor') == 'elevator':
            elevator_cabins[room_id] = room_data
        else:
            regular_rooms[room_id] = room_data
    
    # 布局普通房间
    room_count = len(regular_rooms)
    for i, (room_id, room_data) in enumerate(regular_rooms.items()):
        angle = 2 * np.pi * i / room_count if room_count > 0 else 0
        x, y = 3.5 * np.cos(angle), 3.5 * np.sin(angle)
        nodes[room_id] = {
            'pos': (x, y, ROOM_LEVEL),
            'type': 'room',
            'label': room_id,
            'floor': room_data.get('floor')
        }
        # 房间 → 宏观区连接
        floor = room_data.get('floor')
        if floor and floor in nodes:
            edges.append((floor, room_id, 'parent-child'))
        
        # 邻居连接（排除电梯cabin）
        for neighbor in room_data.get('neighbor', []):
            if neighbor in regular_rooms:
                edges.append((room_id, neighbor, 'neighbor'))
    
    # 布局电梯cabin - 放在中心位置
    cabin_count = len(elevator_cabins)
    for i, (cabin_id, cabin_data) in enumerate(elevator_cabins.items()):
        # 电梯cabin放在中心，如果有多个则围成小圆
        if cabin_count == 1:
            x, y = 0, 0
        else:
            angle = 2 * np.pi * i / cabin_count
            x, y = 0.8 * np.cos(angle), 0.8 * np.sin(angle)
        
        nodes[cabin_id] = {
            'pos': (x, y, ELEVATOR_LEVEL),
            'type': 'elevator',
            'label': cabin_id,
            'floor': 'elevator'
        }
        
        # 电梯cabin与其他楼层的连接
        for neighbor in cabin_data.get('neighbor', []):
            if neighbor in regular_rooms:
                edges.append((cabin_id, neighbor, 'elevator-connection'))
    
    # 3. 物品节点：分别处理 large_objects 和 small_objects
    for room_id, room_data in rooms.items():
        room_pos = nodes[room_id]['pos']
        room_type = nodes[room_id]['type']
        
        # 处理 large_objects
        large_objects = room_data.get('large_objects', {})
        large_count = len(large_objects)
        for j, (item_id, item_data) in enumerate(large_objects.items()):
            angle = 2 * np.pi * j / large_count if large_count > 0 else 0
            radius = 0.8 if room_type == 'elevator' else 1.2
            x = room_pos[0] + radius * np.cos(angle)
            y = room_pos[1] + radius * np.sin(angle)
            z = LARGE_OBJECT_LEVEL
            
            nodes[item_id] = {
                'pos': (x, y, z),
                'type': 'large_object',
                'label': item_id,
                'room': room_id
            }
            edges.append((room_id, item_id, 'contains'))
        
        # 处理 small_objects - 特别标记电梯按钮
        small_objects = room_data.get('small_objects', {})
        small_count = len(small_objects)
        for j, (item_id, item_data) in enumerate(small_objects.items()):
            angle = 2 * np.pi * j / small_count if small_count > 0 else 0
            radius = 0.6 if room_type == 'elevator' else 0.9
            x = room_pos[0] + radius * np.cos(angle)
            y = room_pos[1] + radius * np.sin(angle)
            z = SMALL_OBJECT_LEVEL
            
            # 判断是否为电梯按钮
            is_elevator_button = (
                'elevator_button' in item_id or 
                'elevator_call' in item_id or
                item_data.get('type') == 'control'
            )
            
            object_type = 'elevator_button' if is_elevator_button else 'small_object'
            
            nodes[item_id] = {
                'pos': (x, y, z),
                'type': object_type,
                'label': item_id,
                'room': room_id
            }
            edges.append((room_id, item_id, 'contains'))
    
    # 4. 绘制节点
    for node_id, info in nodes.items():
        x, y, z = info['pos']
        node_type = info['type']
        color = COLORS[node_type]
        
        # 根据类型设置标记和大小
        if node_type == 'macro':
            marker, size = 'o', 300
        elif node_type == 'room':
            marker, size = 's', 200
        elif node_type == 'elevator':
            marker, size = 'D', 250  # 菱形表示电梯
        elif node_type == 'large_object':
            marker, size = '^', 150
        elif node_type == 'elevator_button':
            marker, size = 'o', 120  # 圆形表示按钮
        else:  # small_object
            marker, size = 'v', 100
        
        ax.scatter(x, y, z, c=color, s=size, marker=marker, alpha=0.8, 
                  edgecolors='black', linewidth=1.5)
        
        # 添加标签
        ax.text(x, y, z, f'  {info["label"]}', fontsize=9, ha='left', va='center')
    
    # 5. 绘制边
    edge_colors = {
        'parent-child': '#FF2222',      # 红色 - 父子关系
        'neighbor': '#22AA22',          # 绿色 - 邻居关系
        'elevator-connection': '#8844DD', # 紫色 - 电梯连接
        'contains': '#4488FF'           # 蓝色 - 包含关系
    }
    
    edge_styles = {
        'parent-child': '-',
        'neighbor': '--',
        'elevator-connection': ':',
        'contains': '-'
    }
    
    edge_widths = {
        'parent-child': 2.5,
        'neighbor': 2.0,
        'elevator-connection': 3.0,
        'contains': 1.5
    }
    
    for src, dst, etype in edges:
        if src in nodes and dst in nodes:
            x = [nodes[src]['pos'][0], nodes[dst]['pos'][0]]
            y = [nodes[src]['pos'][1], nodes[dst]['pos'][1]]
            z = [nodes[src]['pos'][2], nodes[dst]['pos'][2]]
            
            ax.plot(x, y, z, 
                   color=edge_colors.get(etype, '#888888'),
                   linestyle=edge_styles.get(etype, '-'),
                   linewidth=edge_widths.get(etype, 1.0),
                   alpha=0.7)
    
    # 6. 图形设置
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Level', fontsize=12)
    
    # 设置Z轴刻度
    z_ticks = [SMALL_OBJECT_LEVEL, LARGE_OBJECT_LEVEL, ELEVATOR_LEVEL, ROOM_LEVEL, MACRO_LEVEL]
    z_labels = ['Small Objects', 'Large Objects', 'Elevator', 'Rooms', 'Macro Zones']
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels)
    
    title = f'3D Scene Graph Visualization: {scene_name or scene_graph.get("name", "Unknown Scene")}'
    ax.set_title(title, fontsize=16, pad=20)
    
    # 7. 创建图例
    legend_elements = [
        mpatches.Patch(color=COLORS['macro'], label='Macro Zones'),
        mpatches.Patch(color=COLORS['room'], label='Rooms'),
        mpatches.Patch(color=COLORS['elevator'], label='Elevator Cabin'),
        mpatches.Patch(color=COLORS['large_object'], label='Large Objects'),
        mpatches.Patch(color=COLORS['small_object'], label='Small Objects'),
        mpatches.Patch(color=COLORS['elevator_button'], label='Elevator Buttons'),
        plt.Line2D([0], [0], color=edge_colors['parent-child'], linewidth=2.5, 
                  label='Parent-Child'),
        plt.Line2D([0], [0], color=edge_colors['neighbor'], linewidth=2.0, 
                  linestyle='--', label='Neighbor'),
        plt.Line2D([0], [0], color=edge_colors['elevator-connection'], linewidth=3.0, 
                  linestyle=':', label='Elevator Connection'),
        plt.Line2D([0], [0], color=edge_colors['contains'], linewidth=1.5, 
                  label='Contains')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # 8. 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 9. 统计信息
    stats = {
        'macro_zones': len(macro_zones),
        'regular_rooms': len(regular_rooms),
        'elevator_cabins': len(elevator_cabins),
        'large_objects': sum(len(room_data.get('large_objects', {})) for room_data in rooms.values()),
        'small_objects': sum(len(room_data.get('small_objects', {})) for room_data in rooms.values())
    }
    
    # 在图上添加统计信息
    stats_text = (f"Statistics:\n"
                 f"Macro Zones: {stats['macro_zones']}\n"
                 f"Rooms: {stats['regular_rooms']}\n"
                 f"Elevators: {stats['elevator_cabins']}\n"
                 f"Large Objects: {stats['large_objects']}\n"
                 f"Small Objects: {stats['small_objects']}")
    
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
              verticalalignment='top', 
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 场景图已保存到: {save_path}")
    else:
        plt.show()
    
    return fig, stats

def print_scene_graph_summary(scene_graph):
    """
    打印场景图的摘要信息，特别标注电梯系统
    """
    scene_name = scene_graph.get('name', 'Unknown')
    print(f"\n=== 场景图摘要: {scene_name} ===")
    
    # 统计宏观区
    macro_zones = scene_graph.get('macro_zones', {})
    print(f"\n🏢 宏观区 ({len(macro_zones)}个):")
    for zone_id, zone_data in macro_zones.items():
        room_count = len(zone_data.get('rooms', []))
        print(f"  • {zone_id}: 包含{room_count}个房间")
    
    # 统计房间（区分普通房间和电梯）
    rooms = scene_graph.get('rooms', {})
    regular_rooms = []
    elevator_cabins = []
    
    for room_id, room_data in rooms.items():
        if room_data.get('floor') == 'elevator':
            elevator_cabins.append(room_id)
        else:
            regular_rooms.append(room_id)
    
    print(f"\n🚪 普通房间 ({len(regular_rooms)}个):")
    total_items = 0
    for room_id in regular_rooms:
        room_data = rooms[room_id]
        large_objects = room_data.get('large_objects', {})
        small_objects = room_data.get('small_objects', {})
        item_count = len(large_objects) + len(small_objects)
        total_items += item_count
        neighbors = len(room_data.get('neighbor', []))
        floor = room_data.get('floor', '未知')
        print(f"  • {room_id}: 楼层{floor} ({item_count}个物品, {neighbors}个邻居)")
    
    print(f"\n🛗 电梯系统 ({len(elevator_cabins)}个):")
    for cabin_id in elevator_cabins:
        cabin_data = rooms[cabin_id]
        large_objects = cabin_data.get('large_objects', {})
        small_objects = cabin_data.get('small_objects', {})
        
        # 统计电梯按钮
        elevator_buttons = []
        for obj_id, obj_data in small_objects.items():
            if ('elevator_button' in obj_id or 'elevator_call' in obj_id or 
                obj_data.get('type') == 'control'):
                elevator_buttons.append(obj_id)
        
        neighbors = cabin_data.get('neighbor', [])
        print(f"  • {cabin_id}: 连接{len(neighbors)}个楼层 ({len(elevator_buttons)}个按钮)")
        print(f"    - 连接楼层: {', '.join(neighbors)}")
        print(f"    - 电梯按钮: {', '.join(elevator_buttons)}")
        
        total_items += len(large_objects) + len(small_objects)
    
    print(f"\n📦 物品总数: {total_items}")
    print("=" * 60)


def visualize_scene_by_name(scene_name, save_dir=None, show_summary=True):
    """
    根据场景名称创建三维可视化
    
    参数:
        scene_name (str): 场景名称 (支持: 'hotel', 'office', 'allensville', 'supermarket')
        save_dir (str): 保存目录
        show_summary (bool): 是否显示场景摘要
    
    返回:
        str: 保存的图片路径
    """
    save_dir = Path(save_dir) if save_dir else Path(__file__).resolve().parent / "fig"
    from scene_graph import HOTEL, OFFICE, ALLENSVILLE, SUPERMARKET
    
    # 场景图字典
    scenes = {
        'hotel': HOTEL,
        'office': OFFICE,
        'allensville': ALLENSVILLE,
        'supermarket': SUPERMARKET
    }
    
    if scene_name.lower() not in scenes:
        available_scenes = list(scenes.keys())
        raise ValueError(f"场景 '{scene_name}' 不存在。可用场景: {available_scenes}")
    
    scene_graph = scenes[scene_name.lower()]
    
    if show_summary:
        print_scene_graph_summary(scene_graph)
    
    # 生成保存路径
    import os
    save_path = os.path.join(save_dir, f"{scene_name}_scene_graph_3d.png")
    
    # 创建可视化
    fig, stats = visualize_scene_graph_3d(scene_graph, scene_name=scene_name, save_path=save_path)
    
    print(f"\n✅ 场景图可视化完成！保存至: {save_path}")
    return save_path


def parse_multiline_json(file_path):
    """
    解析多行JSON格式的数据集文件
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        json_str = ""
        brace_count = 0
        
        for line in file:
            line = line.strip()
            if line:
                json_str += line + "\n"
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and json_str.strip():
                    try:
                        data = json.loads(json_str)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
                    json_str = ""
    
    return data_list


def visualize_dataset_scene_with_objects(sample_data, save_path=None, figsize=(15, 12)):
    """
    可视化数据集中的单个样本场景，特别处理电梯系统
    
    Args:
        sample_data: 包含scene_graph的数据样本
        save_path: 保存路径
        figsize: 图形大小
        
    Returns:
        tuple: (figure对象, 统计信息字典)
    """
    setup_chinese_font()
    
    # 提取场景图数据
    scene_graph = sample_data.get('scene_graph', {})
    if not scene_graph:
        raise ValueError("样本数据中缺少scene_graph字段")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义层级和颜色
    MACRO_LEVEL = 2.0
    ROOM_LEVEL = 1.2
    ELEVATOR_LEVEL = 0.6
    LARGE_OBJECT_LEVEL = 0.1
    SMALL_OBJECT_LEVEL = -0.3
    
    COLORS = {
        'macro': '#CC4444',
        'room': '#44AA44',
        'elevator': '#8844CC',
        'large_object': '#4488CC',
        'small_object': '#FF8844',
        'elevator_button': '#CC44AA'
    }
    
    nodes = {}
    edges = []
    
    # 处理宏观区
    macro_zones = scene_graph.get('macro_zones', {})
    macro_count = len(macro_zones)
    for i, (zone_id, zone_data) in enumerate(macro_zones.items()):
        angle = 2 * np.pi * i / macro_count if macro_count > 0 else 0
        x, y = 3.5 * np.cos(angle), 3.5 * np.sin(angle)
        nodes[zone_id] = {'pos': (x, y, MACRO_LEVEL), 'type': 'macro', 'label': zone_id}
    
    # 处理房间
    rooms = scene_graph.get('rooms', {})
    regular_rooms = {}
    elevator_cabins = {}
    
    for room_id, room_data in rooms.items():
        if room_data.get('floor') == 'elevator':
            elevator_cabins[room_id] = room_data
        else:
            regular_rooms[room_id] = room_data
    
    # 布局普通房间
    room_count = len(regular_rooms)
    for i, (room_id, room_data) in enumerate(regular_rooms.items()):
        angle = 2 * np.pi * i / room_count if room_count > 0 else 0
        x, y = 2.8 * np.cos(angle), 2.8 * np.sin(angle)
        nodes[room_id] = {
            'pos': (x, y, ROOM_LEVEL),
            'type': 'room',
            'label': room_id,
            'floor': room_data.get('floor')
        }
        
        # 连接到宏观区
        floor = room_data.get('floor')
        if floor and floor in nodes:
            edges.append((floor, room_id, 'parent-child'))
        
        # 邻居关系
        for neighbor in room_data.get('neighbor', []):
            if neighbor in regular_rooms:
                edges.append((room_id, neighbor, 'neighbor'))
    
    # 布局电梯cabin
    cabin_count = len(elevator_cabins)
    for i, (cabin_id, cabin_data) in enumerate(elevator_cabins.items()):
        if cabin_count == 1:
            x, y = 0, 0
        else:
            angle = 2 * np.pi * i / cabin_count
            x, y = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
        
        nodes[cabin_id] = {
            'pos': (x, y, ELEVATOR_LEVEL),
            'type': 'elevator',
            'label': cabin_id,
            'floor': 'elevator'
        }
        
        # 电梯连接
        for neighbor in cabin_data.get('neighbor', []):
            if neighbor in regular_rooms:
                edges.append((cabin_id, neighbor, 'elevator-connection'))
    
    # 处理物品
    for room_id, room_data in rooms.items():
        room_pos = nodes[room_id]['pos']
        room_type = nodes[room_id]['type']
        
        # 大型物品
        large_objects = room_data.get('large_objects', {})
        large_count = len(large_objects)
        for j, (item_id, item_data) in enumerate(large_objects.items()):
            angle = 2 * np.pi * j / large_count if large_count > 0 else 0
            radius = 0.7 if room_type == 'elevator' else 1.0
            x = room_pos[0] + radius * np.cos(angle)
            y = room_pos[1] + radius * np.sin(angle)
            z = LARGE_OBJECT_LEVEL
            
            nodes[item_id] = {
                'pos': (x, y, z),
                'type': 'large_object',
                'label': item_id,
                'room': room_id
            }
            edges.append((room_id, item_id, 'contains'))
        
        # 小型物品
        small_objects = room_data.get('small_objects', {})
        small_count = len(small_objects)
        for j, (item_id, item_data) in enumerate(small_objects.items()):
            angle = 2 * np.pi * j / small_count if small_count > 0 else 0
            radius = 0.5 if room_type == 'elevator' else 0.8
            x = room_pos[0] + radius * np.cos(angle)
            y = room_pos[1] + radius * np.sin(angle)
            z = SMALL_OBJECT_LEVEL
            
            # 判断是否为电梯按钮
            is_elevator_button = (
                'elevator_button' in item_id or 
                'elevator_call' in item_id or
                item_data.get('type') == 'control'
            )
            
            object_type = 'elevator_button' if is_elevator_button else 'small_object'
            
            nodes[item_id] = {
                'pos': (x, y, z),
                'type': object_type,
                'label': item_id,
                'room': room_id
            }
            edges.append((room_id, item_id, 'contains'))
    
    # 绘制节点
    for node_id, info in nodes.items():
        x, y, z = info['pos']
        node_type = info['type']
        color = COLORS[node_type]
        
        if node_type == 'macro':
            marker, size = 'o', 250
        elif node_type == 'room':
            marker, size = 's', 180
        elif node_type == 'elevator':
            marker, size = 'D', 220
        elif node_type == 'large_object':
            marker, size = '^', 130
        elif node_type == 'elevator_button':
            marker, size = 'o', 100
        else:
            marker, size = 'v', 90
        
        ax.scatter(x, y, z, c=color, s=size, marker=marker, alpha=0.8, 
                  edgecolors='black', linewidth=1)
        ax.text(x, y, z, f'  {info["label"]}', fontsize=8, ha='left')
    
    # 绘制边
    edge_styles = {
        'parent-child': ('#CC2222', '-', 2.0),
        'neighbor': ('#22AA22', '--', 1.5),
        'elevator-connection': ('#8844CC', ':', 2.5),
        'contains': ('#4488CC', '-', 1.0)
    }
    
    for src, dst, etype in edges:
        if src in nodes and dst in nodes:
            x = [nodes[src]['pos'][0], nodes[dst]['pos'][0]]
            y = [nodes[src]['pos'][1], nodes[dst]['pos'][1]]
            z = [nodes[src]['pos'][2], nodes[dst]['pos'][2]]
            
            color, linestyle, linewidth = edge_styles.get(etype, ('#888888', '-', 1.0))
            ax.plot(x, y, z, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=0.6)
    
    # 设置坐标轴
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Level')
    
    z_ticks = [SMALL_OBJECT_LEVEL, LARGE_OBJECT_LEVEL, ELEVATOR_LEVEL, ROOM_LEVEL, MACRO_LEVEL]
    z_labels = ['Small Objects', 'Large Objects', 'Elevator', 'Rooms', 'Zones']
    ax.set_zticks(z_ticks)
    ax.set_zticklabels(z_labels)
    
    # 标题
    task_info = sample_data.get('task_info', {})
    title = f'Dataset Scene Visualization - Type: {task_info.get("type", "unknown")}'
    ax.set_title(title, fontsize=14, pad=15)
    
    # 图例
    legend_elements = [
        mpatches.Patch(color=COLORS['macro'], label='Macro Zones'),
        mpatches.Patch(color=COLORS['room'], label='Rooms'),
        mpatches.Patch(color=COLORS['elevator'], label='Elevator'),
        mpatches.Patch(color=COLORS['large_object'], label='Large Objects'),
        mpatches.Patch(color=COLORS['small_object'], label='Small Objects'),
        mpatches.Patch(color=COLORS['elevator_button'], label='Elevator Buttons'),
        plt.Line2D([0], [0], color='#CC2222', linewidth=2, label='Hierarchical'),
        plt.Line2D([0], [0], color='#22AA22', linestyle='--', label='Neighbor'),
        plt.Line2D([0], [0], color='#8844CC', linestyle=':', linewidth=2.5, label='Elevator Connection'),
        plt.Line2D([0], [0], color='#4488CC', linewidth=1, label='Contains')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.view_init(elev=25, azim=45)
    
    # 统计信息
    stats = {
        'macro_zones': len(macro_zones),
        'regular_rooms': len(regular_rooms),
        'elevator_cabins': len(elevator_cabins),
        'large_objects': sum(len(room_data.get('large_objects', {})) for room_data in rooms.values()),
        'small_objects': sum(len(room_data.get('small_objects', {})) for room_data in rooms.values())
    }
    
    stats_text = f"Stats: {stats['macro_zones']} zones, {stats['regular_rooms']} rooms, {stats['elevator_cabins']} elevators\n{stats['large_objects']} large objects, {stats['small_objects']} small objects"
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 场景可视化已保存到: {save_path}")
    else:
        plt.show()
    
    return fig, stats


def analyze_and_visualize_dataset(json_file_path, output_dir=None,
                                sample_indices=None, max_samples=5):
    """
    分析并可视化数据集中的场景结构，特别关注电梯系统
    
    Args:
        json_file_path: JSON数据集文件路径
        output_dir: 输出目录
        sample_indices: 指定要可视化的样本索引列表，如果为None则随机选择
        max_samples: 最多可视化的样本数量
    """
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "fig"
    import os
    import random
    
    print(f"开始分析数据集: {json_file_path}")
    
    # 解析数据
    data_list = parse_multiline_json(json_file_path)
    print(f"成功解析 {len(data_list)} 个样本")
    
    # 选择要可视化的样本
    if sample_indices is None:
        sample_indices = random.sample(range(len(data_list)), min(max_samples, len(data_list)))
    else:
        sample_indices = sample_indices[:max_samples]
    
    print(f"将可视化 {len(sample_indices)} 个样本: {sample_indices}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = []
    
    for i, idx in enumerate(sample_indices):
        sample = data_list[idx]
        print(f"\n--- 可视化样本 {idx+1} ---")
        
        # 生成保存路径
        save_path = os.path.join(output_dir, f"scene_visualization_sample_{idx+1}.png")
        
        # 创建可视化
        fig, stats = visualize_dataset_scene_with_objects(sample, save_path=save_path)
        all_stats.append(stats)
        
        # 打印样本信息
        task_info = sample.get('task_info', {})
        scene_graph = sample.get('scene_graph', {})
        
        print(f"任务类型: {task_info.get('type', 'unknown')}")
        print(f"难度: {task_info.get('difficulty', 'unknown')}")
        print(f"指令: {sample.get('instruction', 'No instruction')}")
        print(f"统计: {stats}")
        
        # 分析电梯系统
        rooms = scene_graph.get('rooms', {})
        elevator_info = []
        for room_id, room_data in rooms.items():
            if room_data.get('floor') == 'elevator':
                small_objects = room_data.get('small_objects', {})
                buttons = [obj_id for obj_id in small_objects.keys() 
                          if 'elevator_button' in obj_id or 'elevator_call' in obj_id]
                connections = room_data.get('neighbor', [])
                elevator_info.append({
                    'cabin': room_id,
                    'buttons': buttons,
                    'connections': connections
                })
        
        if elevator_info:
            print(f"电梯系统分析:")
            for info in elevator_info:
                print(f"  - {info['cabin']}: {len(info['buttons'])}个按钮, 连接{len(info['connections'])}个楼层")
    
    # 汇总统计
    print(f"\n=== 数据集统计汇总 ===")
    avg_stats = {}
    for key in all_stats[0].keys():
        avg_stats[key] = sum(stats[key] for stats in all_stats) / len(all_stats)
    
    print(f"平均统计: {avg_stats}")
    
    # 电梯系统汇总
    total_elevators = sum(stats.get('elevator_cabins', 0) for stats in all_stats)
    print(f"总电梯数量: {total_elevators}")
    
    return all_stats


# 使用示例和测试函数
if __name__ == "__main__":
    # 方式1: 可视化预定义场景
    print("=== 预定义场景可视化 ===")

    # 可视化酒店场景（包含电梯系统）
    try:
        hotel_path = visualize_scene_by_name('office', show_summary=True)
        print(f"酒店场景可视化完成: {hotel_path}")
    except Exception as e:
        print(f"酒店场景可视化失败: {e}")
