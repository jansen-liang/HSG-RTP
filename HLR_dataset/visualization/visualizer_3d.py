import plotly.graph_objects as go
import networkx as nx
import math
import random
import re
from typing import Dict, Any

# 引入你的枚举
try:
    from models.nodes import NodeType
except ImportError:
    class NodeType:
        MOBILE_TOOL = 'mobile_tool'
        ROOM = 'room'
        FLOOR = 'floor'
        OBJECT = 'object'
        AGENT = 'agent'

class SceneVisualizer:
    def __init__(self, scene):
        self.scene = scene
        self.G_full = self._build_graph()

    def _build_graph(self):
        """
        构建图：核心逻辑 —— 电梯只连当前所在的楼层
        """
        G = nx.Graph()
        for node_id in self.scene.nodes:
            G.add_node(node_id)
            
        for edge in self.scene.edges:
            u_node = self.scene.nodes.get(edge.source_id)
            v_node = self.scene.nodes.get(edge.target_id)
            if not u_node or not v_node: continue

            # =========================================================
            # [核心逻辑] 电梯位置过滤
            # =========================================================
            elevator = None
            location_id = None

            # 判断哪一端是电梯
            if getattr(u_node, 'node_type', None) == 'mobile_tool' or getattr(u_node, 'type', None) == 'mobile_tool':
                elevator, location_id = u_node, edge.target_id
            elif getattr(v_node, 'node_type', None) == 'mobile_tool' or getattr(v_node, 'type', None) == 'mobile_tool':
                elevator, location_id = v_node, edge.source_id

            if elevator:
                location_node = self.scene.nodes.get(location_id)
                if location_node:
                    # 获取目标类型 (Room, Floor, Hall等)
                    target_type = str(getattr(location_node, 'type', getattr(location_node, 'node_type', ''))).lower()
                    if target_type in ['room', 'floor', 'hall', 'corridor']:
                        # 获取电梯当前位置
                        curr_loc = getattr(elevator, 'current_location', None)
                        # 如果电梯有位置信息，且不等于这条边的目标位置 ID -> 断开连接
                        if curr_loc and curr_loc != location_id:
                            continue 
            # =========================================================

            cat = getattr(edge, 'category', 'physical')
            if hasattr(cat, 'value'): cat = cat.value
            G.add_edge(edge.source_id, edge.target_id, relation=edge.relation, category=cat)
        return G

    def _is_large_object(self, name):
        """判断是否为家具/大物体"""
        name = name.lower()
        large_keys = ['table', 'desk', 'shelf', 'bed', 'sofa', 'cabinet', 'door', 'wall', 'refrigerator', 'chair', 'counter', 'bench']
        return any(k in name for k in large_keys)

    def _compute_concentric_layout(self):
        """同心圆布局"""
        pos = {}
        FLOOR_HEIGHT = 160.0 
        R_ROOM = 18.0
        R_LARGE = 32.0
        R_SMALL = 42.0

        def get_floor_num(node_id):
            match = re.search(r'F(\d+)', str(node_id))
            if match: return int(match.group(1))
            return 1

        # 1. 归类
        floors = {} 
        for nid, node in self.scene.nodes.items():
            f = get_floor_num(nid)
            ntype = str(getattr(node, 'type', getattr(node, 'node_type', 'object'))).lower()
            name = str(getattr(node, 'name', nid)).lower()

            if 'mobile' in ntype:
                curr = getattr(node, 'current_location', '')
                if curr: f = get_floor_num(curr)

            if f not in floors: 
                floors[f] = {'center': [], 'rooms': [], 'large': [], 'small': []}
            
            if 'mobile' in ntype or 'floor' in ntype:
                floors[f]['center'].append(nid)
            elif 'room' in ntype or 'hall' in ntype or 'corridor' in ntype:
                floors[f]['rooms'].append(nid)
            elif self._is_large_object(name):
                floors[f]['large'].append(nid)
            else:
                # 按钮(Button) 和其他不属于大物体的，都会落到这里 (Small Layer)
                floors[f]['small'].append(nid)

        # 2. 计算坐标
        for f, data in floors.items():
            z = (f - 1) * FLOOR_HEIGHT
            
            # Layer 0: Center
            for nid in data['center']:
                pos[nid] = (0, 0, z)

            # Layer 1: Rooms
            rooms = data['rooms']
            room_angles = {}
            n_rooms = len(rooms)
            if n_rooms > 0:
                for i, rid in enumerate(rooms):
                    angle = 2 * math.pi * i / n_rooms
                    room_angles[rid] = angle
                    pos[rid] = (R_ROOM * math.cos(angle), R_ROOM * math.sin(angle), z)

            # Layer 2: Furniture
            for oid in data['large']:
                neighbors = list(self.G_full.neighbors(oid))
                parent = next((n for n in neighbors if n in room_angles), None)
                if parent:
                    angle = room_angles[parent] + random.uniform(-0.3, 0.3)
                else:
                    angle = random.uniform(0, 2*math.pi)
                pos[oid] = (R_LARGE * math.cos(angle), R_LARGE * math.sin(angle), z)

            # Layer 3: Small Items (包含按钮)
            for oid in data['small']:
                neighbors = list(self.G_full.neighbors(oid))
                parent_room = next((n for n in neighbors if n in room_angles), None)
                
                if parent_room:
                    angle = room_angles[parent_room] + random.uniform(-0.5, 0.5)
                else:
                    parent_obj = next((n for n in neighbors if n in pos), None)
                    if parent_obj:
                        px, py, _ = pos[parent_obj]
                        angle = math.atan2(py, px) + random.uniform(-0.2, 0.2)
                    else:
                        angle = random.uniform(0, 2*math.pi)
                pos[oid] = (R_SMALL * math.cos(angle), R_SMALL * math.sin(angle), z)

        return pos

    def generate_html(self, output_path="scene_vis_final_v3.html"):
        print(f"🎨 生成可视化 (电梯单层连接 + 按钮归为小物体)...")
        final_pos = self._compute_concentric_layout()
        fig = go.Figure()

        # ==========================================
        # 1. 绘制边
        # ==========================================
        edge_traces = {
            # 蓝色特粗 (Transport): 6 -> 12
            'transport': {'x':[], 'y':[], 'z':[], 'color': '#1f77b4', 'width': 12}, 
            # 橙色加粗 (Logic): 2 -> 6
            'logic':     {'x':[], 'y':[], 'z':[], 'color': '#ff7f0e', 'width': 6}, 
            # 灰色加粗且加深 (Normal): 1 -> 3, #ccc -> #888
            'normal':    {'x':[], 'y':[], 'z':[], 'color': '#888',    'width': 3}, 
        }

        for u, v, data in self.G_full.edges(data=True):
            if u not in final_pos or v not in final_pos: continue
            
            cat = str(data.get('category', '')).lower()
            etype = 'normal'
            if 'transport' in cat: etype = 'transport'
            elif 'logic' in cat: etype = 'logic'
            
            x0, y0, z0 = final_pos[u]
            x1, y1, z1 = final_pos[v]
            edge_traces[etype]['x'].extend([x0, x1, None])
            edge_traces[etype]['y'].extend([y0, y1, None])
            edge_traces[etype]['z'].extend([z0, z1, None])

        for etype, cfg in edge_traces.items():
            fig.add_trace(go.Scatter3d(
                x=cfg['x'], y=cfg['y'], z=cfg['z'],
                mode='lines',
                line=dict(color=cfg['color'], width=cfg['width']),
                name=etype, hoverinfo='none'
            ))

        # ==========================================
        # 2. 绘制节点
        # ==========================================
        # 定义样式：移除了 Button 的金点，统一用 SmallItem 的粉点
        styles = {
            'Floor':     {'color': '#7f7f7f', 'size': 10, 'symbol': 'circle',  'mode': 'markers'},      # 灰小圆
            'Elevator':  {'color': '#d62728', 'size': 20, 'symbol': 'diamond', 'mode': 'markers'},
            'Room':      {'color': '#2ca02c', 'size': 18, 'symbol': 'circle',  'mode': 'markers+text'}, # 只有房间显字
            'Furniture': {'color': '#8c564b', 'size': 12, 'symbol': 'square',  'mode': 'markers'},      # 棕方块
            'SmallItem': {'color': '#e377c2', 'size': 8,  'symbol': 'circle',  'mode': 'markers'},      # 粉小圆 (含按钮)
            'Agent':     {'color': '#9467bd', 'size': 15, 'symbol': 'cross',   'mode': 'markers+text'}
        }
        
        node_groups = {k: {'x':[], 'y':[], 'z':[], 'text':[]} for k in styles}

        for nid, (x, y, z) in final_pos.items():
            node = self.scene.nodes.get(nid)
            ntype = str(getattr(node, 'type', getattr(node, 'node_type', 'object'))).lower()
            name = str(getattr(node, 'name', nid))
            
            # --- 分类逻辑 ---
            group = 'SmallItem' # 默认是小物体 (粉色)
            
            if 'mobile' in ntype: 
                group = 'Elevator'
            elif 'floor' in ntype:
                group = 'Floor'
            elif 'agent' in ntype:
                group = 'Agent'
            elif 'room' in ntype or 'hall' in ntype or 'corridor' in ntype:
                group = 'Room'
            # 注意：不再单独判断 'button'，让它掉落到下面的 else 里（即 SmallItem）
            # 或者是这里的大物体判断
            elif self._is_large_object(name):
                group = 'Furniture'
            # ----------------
            
            node_groups[group]['x'].append(x)
            node_groups[group]['y'].append(y)
            node_groups[group]['z'].append(z)
            node_groups[group]['text'].append(name)

        for group_name, data in node_groups.items():
            if not data['x']: continue
            style = styles[group_name]
            
            fig.add_trace(go.Scatter3d(
                x=data['x'], y=data['y'], z=data['z'],
                mode=style['mode'],
                marker=dict(
                    size=style['size'], 
                    color=style['color'], 
                    symbol=style['symbol'], 
                    line=dict(width=1, color='white')
                ),
                text=data['text'],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                name=group_name, 
                hoverinfo='text'
            ))

        # ==========================================
        # 3. 布局设置
        # ==========================================
        axis = dict(showbackground=False, showgrid=True, gridcolor='rgba(200,200,200,0.3)', showticklabels=False, title='')
        fig.update_layout(
            title='',
            scene=dict(
                xaxis=axis, yaxis=axis, zaxis=axis, 
                aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.2),
                camera=dict(eye=dict(x=0.1, y=0.1, z=2.5))
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02)
        )
        
        fig.write_html(output_path)
        print(f"✅ 可视化已生成: {output_path}")