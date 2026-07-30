import sys
import os
from pathlib import Path
import copy
import json
import re
import importlib.util

# 根据你的目录结构添加路径
sys.path.append(str(Path(__file__).parent / "data" / "sg"))
sys.path.append(str(Path(__file__).parent / "utils"))

from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QListWidget, QAbstractItemView,
    QSplitter, QFrame, QSlider, QGroupBox, QStackedWidget,
    QLineEdit, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# 导入你的业务模块 (如果文件位置不同，请自行调整 import 路径)
from .utils.task_generator import Task, TaskType, DifficultyLevel
from .utils.action_planner import plan_actions
from .utils.simulator import execute_subtask
from editor.services.scene_registry import load_all_scenes
from editor.widgets.editor_page import SceneEditorPage

# ==========================================
# 【动态加载器】自动扫描 generated 文件夹下的所有新地图
# ==========================================
def load_generated_scenes_dynamically(existing_scenes):
    generated_folder = Path(__file__).resolve().parent / "sg" / "generated"
    
    if not generated_folder.exists():
        print(f"⚠️ 警告: 找不到动态地图文件夹 {generated_folder}")
        return

    # 遍历文件夹下所有的 .py 文件
    for filename in os.listdir(generated_folder):
        if filename.endswith(".py") and not filename.startswith("__"):
            filepath = generated_folder / filename
            module_name = filename[:-3]

            try:
                # 使用 Python 的反射机制，将这个文件当成模块动态加载
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 扫描这个文件里的所有变量，寻找合格的“场景图字典”
                for var_name in dir(module):
                    var_value = getattr(module, var_name)
                    # 判断标准：它必须是个字典，且里面包含 'name' 和 'rooms' 字段
                    if isinstance(var_value, dict) and "name" in var_value and "rooms" in var_value:
                        scene_name = var_value["name"]
                        existing_scenes[scene_name] = var_value
                        print(f"✅ 动态挂载成功: 发现新场景 '{scene_name}' (来自 {filename})")
            except Exception as e:
                print(f"❌ 加载动态地图文件 {filename} 时出错: {e}")

# ==========================================
# 核心 3D 画布组件
# ==========================================
class SceneGraphCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self.plotter)
        self.setLayout(layout)

        self.scene = None
        self.nodes = {}
        self.edges = []
        self.robot_actor = None
        self.robot_position = None
        self.on_object_clicked = None 
        self.carried_actor = None
        self.carried_edge_actor = None

        self.COLORS = {
            'macro': '#FF4444', 'room': '#44AA44', 'elevator': '#8844DD',
            'large_object': "#0B51CA", 'small_object': "#EB630F", 'elevator_button': '#DD44AA'
        }
        self.EDGE_CONFIG = {
            'parent-child': {'color': '#FF2222', 'width': 2.5},
            'neighbor': {'color': '#22AA22', 'width': 2.0},
            'elevator-connection': {'color': '#8844DD', 'width': 3.0},
            'contains': {'color': '#4488FF', 'width': 1.5},
            'attached': {'color': '#AA6600', 'width': 1.0}  
        }

        self.plotter.set_background("white")
        self._add_z_level_annotations()

    def _add_z_level_annotations(self):
        z_levels = [(-0.1, "Small Objects"), (0.3, "Large Objects"), (0.8, "Elevator"), (1.5, "Rooms"), (2.5, "Macro Zones")]
        for z, label in z_levels:
            self.plotter.add_point_labels(
                points=np.array([[5.5, 0, z]]), labels=[label], font_size=12,
                text_color='gray', point_size=0, shape_opacity=0, name=f"ztext_{z}"
            )

    def _get_color(self, node_type, extra_info=None):
        hex_color = self.COLORS.get(node_type, '#888888')
        if node_type == 'elevator_button' and extra_info and extra_info.get('pressed', False):
            return (1.0, 0.8, 0.0)
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    def _build_graph(self):
        nodes, edges = {}, []
        MACRO_LEVEL, ROOM_LEVEL, ELEVATOR_LEVEL, LARGE_OBJECT_LEVEL, SMALL_OBJECT_LEVEL = 4, 2, 1, 0.3, -0.5

        for i, (zone_id, _) in enumerate(self.scene.get('macro_zones', {}).items()):
            angle = 2 * np.pi * i / len(self.scene['macro_zones']) if self.scene['macro_zones'] else 0
            nodes[zone_id] = {'pos': (4.5 * np.cos(angle), 4.5 * np.sin(angle), MACRO_LEVEL), 'type': 'macro', 'label': zone_id}

        reg_rooms, elev_cabins = {}, {}
        for r_id, r_data in self.scene.get('rooms', {}).items():
            (elev_cabins if r_data.get('floor') == 'elevator' else reg_rooms)[r_id] = r_data

        for i, (r_id, r_data) in enumerate(reg_rooms.items()):
            angle = 2 * np.pi * i / len(reg_rooms) if reg_rooms else 0
            nodes[r_id] = {'pos': (3.5 * np.cos(angle), 3.5 * np.sin(angle), ROOM_LEVEL), 'type': 'room', 'label': r_id}
            if r_data.get('floor') in nodes: edges.append((r_data.get('floor'), r_id, 'parent-child'))
            for nbr in r_data.get('neighbor', []):
                if nbr in reg_rooms: edges.append((r_id, nbr, 'neighbor'))

        for i, (c_id, c_data) in enumerate(elev_cabins.items()):
            x, y = (0, 0) if len(elev_cabins) == 1 else (0.8 * np.cos(2 * np.pi * i / len(elev_cabins)), 0.8 * np.sin(2 * np.pi * i / len(elev_cabins)))
            nodes[c_id] = {'pos': (x, y, ELEVATOR_LEVEL), 'type': 'elevator', 'label': c_id}
            for nbr in c_data.get('neighbor', []):
                if nbr in reg_rooms: edges.append((c_id, nbr, 'elevator-connection'))

        pressed_btns = set(self.scene.get("agent", {}).get("pressed_buttons", []))

        for r_id, r_data in self.scene.get('rooms', {}).items():
            if r_id not in nodes: continue
            r_pos, r_type = nodes[r_id]['pos'], nodes[r_id]['type']

            large_objs = r_data.get('large_objects', {})
            for j, (obj_id, obj_data) in enumerate(large_objs.items()):
                angle = 2 * np.pi * j / len(large_objs) if large_objs else 0
                radius = 0.8 if r_type == 'elevator' else 1.2
                nodes[obj_id] = {'pos': (r_pos[0] + radius * np.cos(angle), r_pos[1] + radius * np.sin(angle), LARGE_OBJECT_LEVEL), 'type': 'large_object', 'label': obj_id}
                edges.append((r_id, obj_id, 'contains'))
                for tgt in obj_data.get("relation", {}).values():
                    if tgt in large_objs: edges.append((tgt, obj_id, 'attached'))

            small_objs = r_data.get('small_objects', {})
            for j, (obj_id, obj_data) in enumerate(small_objs.items()):
                angle = 2 * np.pi * j / len(small_objs) if small_objs else 0
                radius = 0.6 if r_type == 'elevator' else 0.9
                is_btn = ('elevator_button' in obj_id or 'elevator_call' in obj_id or obj_data.get('type') == 'control')
                nodes[obj_id] = {'pos': (r_pos[0] + radius * np.cos(angle), r_pos[1] + radius * np.sin(angle), SMALL_OBJECT_LEVEL), 'type': 'elevator_button' if is_btn else 'small_object', 'label': obj_id, 'pressed': obj_id in pressed_btns}
                edges.append((r_id, obj_id, 'contains'))
                for tgt in obj_data.get("relation", {}).values():
                    if tgt in large_objs: edges.append((tgt, obj_id, 'attached'))

        self.nodes, self.edges = nodes, edges
        self.node_id_order = list(self.nodes.keys())

    def _render_static(self):
        self.plotter.clear()
        if not self.nodes: return

        points = np.array([info['pos'] for info in self.nodes.values()])
        labels = [info['label'] for info in self.nodes.values()]
        colors = np.array([self._get_color(info['type'], {'pressed': info.get('pressed', False)}) for info in self.nodes.values()])

        self.points_actor = self.plotter.add_points(
            points, scalars=(colors * 255).astype(np.uint8), rgb=True, point_size=20,
            render_points_as_spheres=True, ambient=0.3, opacity=0.9, name="nodes"
        )
        self.plotter.add_point_labels(points, labels, font_size=12, text_color='black', point_size=0, shape_opacity=0, always_visible=True, name="node_labels")

        edge_actors = {etype: [] for etype in self.EDGE_CONFIG}
        for src, dst, etype in self.edges:
            if src in self.nodes and dst in self.nodes:
                edge_actors[etype].append(pv.Line(self.nodes[src]['pos'], self.nodes[dst]['pos']))

        for etype, lines in edge_actors.items():
            if lines:
                combined = lines[0]
                for line in lines[1:]: combined += line
                self.plotter.add_mesh(combined, color=self.EDGE_CONFIG[etype]['color'], line_width=self.EDGE_CONFIG[etype]['width'], name=f"edges_{etype}")

        if self.scene:
            self.plotter.add_text(f"Scene: {self.scene.get('name', 'Unknown')}", font_size=16, position='upper_edge', color='black', name='title')

        self.plotter.view_isometric()
        self.plotter.camera.elevation, self.plotter.camera.azimuth = 25, 45
        self.plotter.reset_camera()
        self.plotter.track_click_position(self._on_mouse_click, side="left")
    
    def _on_mouse_click(self, event):
        if not self.nodes or self.plotter.click_position is None: return
        x, y = int(self.plotter.click_position[0]), int(self.plotter.click_position[1])
        if x < 0 or y < 0: return

        self.plotter.picker.Pick(x, y, 0, self.plotter.renderer)
        point_id = self.plotter.picker.GetPointId()
        if point_id == -1 or point_id >= len(self.node_id_order): return

        node_id = self.node_id_order[point_id]
        if self.on_object_clicked:
            self.on_object_clicked(node_id, self._get_node_full_info(node_id))

    def _get_node_full_info(self, node_id):
        if not self.scene: return {}
        for r_data in self.scene.get("rooms", {}).values():
            if node_id in r_data.get("small_objects", {}): return r_data["small_objects"][node_id].copy()
            if node_id in r_data.get("large_objects", {}): return r_data["large_objects"][node_id].copy()
        if node_id in self.scene.get("macro_zones", {}): return {"_type": "macro_zone", "data": self.scene["macro_zones"][node_id]}
        if node_id in self.scene.get("rooms", {}): return {"_type": "room", "data": self.scene["rooms"][node_id]}
        return {"_type": "unknown", "id": node_id}

    def load_scene(self, scene_graph):
        self.scene = scene_graph
        self._build_graph()
        self._render_static()
        self.robot_actor = self.robot_position = None

    def update_from_scene(self, scene_graph):
        self.scene = scene_graph
        pressed_btns = set(self.scene.get("agent", {}).get("pressed_buttons", []))
        cur_room = self.scene.get("agent", {}).get("position")
        objs_in_room = set()
        if cur_room in self.scene.get("rooms", {}):
            objs_in_room.update(self.scene["rooms"][cur_room].get("small_objects", {}).keys())
            objs_in_room.update(self.scene["rooms"][cur_room].get("large_objects", {}).keys())
        inv_items = set(self.scene.get("agent", {}).get("inventory", {}).keys())

        new_colors = []
        for node_id in self.node_id_order:
            info = self.nodes[node_id]
            is_visible = not (info['type'] == 'small_object' and info['label'] not in objs_in_room and info['label'] not in inv_items)
            color = self._get_color(info['type'], {'pressed': info['label'] in pressed_btns}) if is_visible else (1.0, 1.0, 1.0)
            new_colors.append(color)

        if self.carried_actor: self.plotter.remove_actor(self.carried_actor, render=False); self.carried_actor = None
        if self.carried_edge_actor: self.plotter.remove_actor(self.carried_edge_actor, render=False); self.carried_edge_actor = None

        if inv_items and self.robot_position in self.nodes:
            obj_id = list(inv_items)[0]
            obj_type = 'large_object' if any(obj_id in r.get("large_objects", {}) for r in self.scene.get("rooms", {}).values()) else 'small_object'
            rx, ry, rz = self.nodes[self.robot_position]['pos']
            self.carried_actor = self.plotter.add_points(np.array([[rx, ry, rz + 0.5]]), color=self._get_color(obj_type), point_size=15, render_points_as_spheres=True, name="carried_item")
            self.carried_edge_actor = self.plotter.add_mesh(pv.Line(np.array([rx, ry, rz + 0.2]), np.array([rx, ry, rz + 0.5])), color='gray', line_width=1.5, name="carried_edge")

        if hasattr(self, 'points_actor') and self.points_actor:
            from vtk import vtkUnsignedCharArray
            color_array = vtkUnsignedCharArray()
            color_array.SetName("scalars")
            color_array.SetNumberOfComponents(3)
            scalars_uint8 = (np.clip(np.array(new_colors), 0.0, 1.0) * 255).astype(np.uint8)
            color_array.SetArray(scalars_uint8, scalars_uint8.size, True, True)
            self.points_actor.mapper.GetInput().GetPointData().SetScalars(color_array)
            self.points_actor.mapper.GetInput().Modified()
            self.points_actor.mapper.Update()
            self.plotter.render()

    def set_robot_position(self, position):
        self.robot_position = position
        if self.robot_actor is None:
            self.robot_actor = self.plotter.add_mesh(pv.Sphere(radius=0.25), color='red', ambient=0.5, name='robot')
        if position in self.nodes:
            x, y, z = self.nodes[position]['pos']
            self.robot_actor.SetPosition(x, y, z + 0.2)
        else:
            self.robot_actor.SetPosition(0, 0, -100)
        self.plotter.update()


# ==========================================
# 主窗口框架 (统一画布架构)
# ==========================================
class HSGRTPVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSG-RTP Scene Graph Full Visualizer")

        # ====== 地图加载逻辑 ======
        # 1. 加载系统自带的地图 (HOTEL, OFFICE 等)
        self.scenes = load_all_scenes()
        
        # 2. 动态扫描并挂载 generated 文件夹下的自定义地图
        load_generated_scenes_dynamically(self.scenes)

        self.dataset_tasks = []
        
        # 共享状态
        self.current_actions = []
        self.simulation_snapshots = []
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate_step)
        self.is_playing = False

        self.init_ui()
        self._apply_adaptive_window_geometry()

    def _apply_adaptive_window_geometry(self):
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(max(1100, min(int(avail.width() * 0.88), 1600)), max(750, min(int(avail.height() * 0.90), 1000)))
            frame = self.frameGeometry()
            frame.moveCenter(avail.center())
            self.move(frame.topLeft())
        self.setMinimumSize(900, 600)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # 顶部导航
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Planning Mode (实时生成)", "Dataset Replay Mode (回放数据)", "Scene Editor Mode (编辑地图)"])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        toolbar.addWidget(self.mode_combo)
        toolbar.addStretch()
        root_layout.addLayout(toolbar)

        self.main_stack = QStackedWidget()
        root_layout.addWidget(self.main_stack)

        # --- 共享视图页面 (Page 0) ---
        viewer_page = QWidget()
        viewer_layout = QHBoxLayout(viewer_page)
        viewer_layout.setContentsMargins(0,0,0,0)

        # 左侧控制区
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.Shape.StyledPanel)
        left_panel.setMinimumWidth(280)
        left_panel.setMaximumWidth(380)
        left_layout = QVBoxLayout(left_panel)

        self.controls_stack = QStackedWidget()
        self.controls_stack.addWidget(self._build_planning_controls())
        self.controls_stack.addWidget(self._build_dataset_controls())
        left_layout.addWidget(self.controls_stack)

        # 共享播放控制
        anim_group = QGroupBox("Path Animation Controls")
        anim_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_animation)
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setEnabled(False)
        self.step_slider.valueChanged.connect(self.on_slider_change)
        anim_layout.addWidget(self.play_btn)
        anim_layout.addWidget(self.step_slider)
        anim_group.setLayout(anim_layout)
        left_layout.addWidget(anim_group)

        # 右侧画布区
        right_panel = QSplitter(Qt.Orientation.Horizontal)
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0,0,0,0)
        
        self.canvas = SceneGraphCanvas()  # 唯一的 3D 画布
        self.action_list = QListWidget()
        self.action_list.setMaximumHeight(200)
        self.action_list.setStyleSheet("QListWidget::item:selected { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.action_list)
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setFont(QFont("Courier", 8))
        self.info_display.setMinimumWidth(260)
        self.info_display.setMaximumWidth(340)

        right_panel.addWidget(canvas_container)
        right_panel.addWidget(self.info_display)
        self.canvas.on_object_clicked = self.on_object_selected

        viewer_splitter = QSplitter(Qt.Orientation.Horizontal)
        viewer_splitter.addWidget(left_panel)
        viewer_splitter.addWidget(right_panel)
        viewer_splitter.setSizes([300, 920])
        viewer_layout.addWidget(viewer_splitter)

        # --- 编辑器页面 (Page 1) ---
        self.editor_page = SceneEditorPage()
        self.editor_page.scene_saved.connect(self.on_editor_scene_saved)

        self.main_stack.addWidget(viewer_page)
        self.main_stack.addWidget(self.editor_page)

        # 初始触发
        if self.scenes:
            self.on_scene_changed(list(self.scenes.keys())[0])
            self.update_task_guidance()

    def on_mode_changed(self, index):
        if index == 0:
            self.main_stack.setCurrentIndex(0)
            self.controls_stack.setCurrentIndex(0)
        elif index == 1:
            self.main_stack.setCurrentIndex(0)
            self.controls_stack.setCurrentIndex(1)
        elif index == 2:
            self.main_stack.setCurrentIndex(1)

    # ==========================================
    # 控制面板构建
    # ==========================================
    def _build_planning_controls(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0)
        
        scene_group = QGroupBox("Scene Selection")
        scene_layout = QHBoxLayout()
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(self.scenes.keys())
        self.scene_combo.currentTextChanged.connect(self.on_scene_changed)
        scene_layout.addWidget(QLabel("Scene:"))
        scene_layout.addWidget(self.scene_combo)
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)

        task_group = QGroupBox("Task Configuration")
        task_layout = QVBoxLayout()
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["delivery", "tidying", "guidance"])
        self.task_type_combo.currentTextChanged.connect(self.update_task_guidance)
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["easy", "medium", "hard"])
        self.difficulty_combo.currentTextChanged.connect(self.update_task_guidance)
        task_layout.addWidget(QLabel("Task Type:")); task_layout.addWidget(self.task_type_combo)
        task_layout.addWidget(QLabel("Difficulty:")); task_layout.addWidget(self.difficulty_combo)
        
        self.dynamic_widget = QWidget()
        self.dynamic_layout = QVBoxLayout(self.dynamic_widget)
        self.dynamic_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.addWidget(self.dynamic_widget)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        self.generate_btn = QPushButton("Generate Task & Path")
        self.generate_btn.clicked.connect(self.on_generate_path)
        layout.addWidget(self.generate_btn)
        layout.addStretch()
        return widget

    def _build_dataset_controls(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0)

        load_group = QGroupBox("Load JSONL Dataset")
        load_layout = QVBoxLayout()
        
        self.dataset_path_input = QLineEdit()
        self.dataset_path_input.setText(str(Path(__file__).resolve().parent / "output" / "example.jsonl"))
        
        path_layout = QHBoxLayout()
        self.browse_btn = QPushButton("📂 Browse")
        self.browse_btn.clicked.connect(self.browse_dataset_file)
        path_layout.addWidget(self.dataset_path_input)
        path_layout.addWidget(self.browse_btn)
        
        self.load_dataset_btn = QPushButton("🚀 Load Dataset")
        self.load_dataset_btn.clicked.connect(self.on_load_dataset)
        
        load_layout.addLayout(path_layout)
        load_layout.addWidget(self.load_dataset_btn)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        task_group = QGroupBox("Select Task")
        task_layout = QVBoxLayout()
        self.dataset_task_combo = QComboBox()
        self.dataset_task_combo.currentIndexChanged.connect(self.on_dataset_task_selected)
        task_layout.addWidget(self.dataset_task_combo)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)
        layout.addStretch()
        return widget

    def browse_dataset_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset File", "", "JSON/JSONL Files (*.json *.jsonl);;All Files (*)"
        )
        if file_path:
            self.dataset_path_input.setText(file_path)

    # ==========================================
    # 数据集回放逻辑 (带防崩溃保护)
    # ==========================================
    def on_load_dataset(self):
        path = self.dataset_path_input.text().strip()
        if not os.path.exists(path):
            self.action_list.clear()
            self.action_list.addItem(f"❌ 找不到文件: {path}")
            return
            
        self.dataset_tasks = []
        self.dataset_task_combo.blockSignals(True)
        self.dataset_task_combo.clear()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            decoder = json.JSONDecoder()
            idx, length = 0, len(content)
            
            while idx < length:
                while idx < length and content[idx].isspace(): idx += 1
                if idx >= length: break
                try:
                    obj, new_idx = decoder.raw_decode(content[idx:])
                    if isinstance(obj, list):
                        self.dataset_tasks.extend(obj)
                    elif isinstance(obj, dict):
                        self.dataset_tasks.append(obj)
                    idx += new_idx
                except json.JSONDecodeError:
                    idx += 1

            valid_count = 0
            for i, data in enumerate(self.dataset_tasks):
                if isinstance(data, dict):
                    self.dataset_task_combo.addItem(f"[{i}] {data.get('instruction', f'Task {i}')[:50]}...")
                    valid_count += 1
            
            self.action_list.clear()
            self.action_list.addItem(f"✅ 成功无损解析 {valid_count} 条任务数据！")
        except Exception as e:
            self.action_list.addItem(f"❌ 解析严重失败: {e}")
        finally:
            self.dataset_task_combo.blockSignals(False)
            if self.dataset_tasks and isinstance(self.dataset_tasks[0], dict): 
                self.on_dataset_task_selected(0)

    def on_dataset_task_selected(self, index):
        if not self.dataset_tasks or index < 0 or index >= len(self.dataset_tasks): return
        task_data = self.dataset_tasks[index]
        if not isinstance(task_data, dict): return

        # 核心提取逻辑
        scene = {}
        if "streaming_samples" in task_data and len(task_data["streaming_samples"]) > 0:
            scene = task_data["streaming_samples"][0].get("scene_graph", {})
        else:
            scene = task_data.get("scene_graph", task_data.get("initial_state", {}))

        raw_subtasks = []
        if "execution_summary" in task_data:
            raw_subtasks = task_data["execution_summary"].get("subtasks", [])
        else:
            raw_subtasks = task_data.get("subtasks", [])
        
        clean_subtasks = []
        for action in raw_subtasks:
            if isinstance(action, str) and "{" in action and "}" in action:
                match = re.search(r'\{.*\}', action, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                        t_list = parsed.get("task", [])
                        if isinstance(t_list, list) and t_list:
                            clean_subtasks.append(t_list[0])
                            continue
                    except: pass
            clean_subtasks.append(action)

        self.action_list.clear()
        self.action_list.addItem(f"📖 {task_data.get('instruction', '')}")
        self.action_list.addItem("-" * 30)
        
        self._load_simulation(scene, clean_subtasks)

    # ==========================================
    # Planning 逻辑
    # ==========================================
    def on_scene_changed(self, scene_name):
        scene = self.scenes[scene_name]
        self.canvas.load_scene(scene)
        self.current_actions = []
        self.action_list.clear()
        self.step_slider.setEnabled(False)
        self.step_slider.setValue(0)
        self.play_btn.setEnabled(False)
        self.timer.stop()
        self.is_playing = False
        self.play_btn.setText("Play")
        self.update_dynamic_controls()

    def update_task_guidance(self):
        self.update_dynamic_controls()

    def update_dynamic_controls(self):
        while self.dynamic_layout.count():
            child = self.dynamic_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        task_type = self.task_type_combo.currentText()
        scene = self.scenes[self.scene_combo.currentText()]

        if task_type == "delivery":
            self.start_combo = QComboBox()
            self.start_combo.addItems([r for r, data in scene["rooms"].items() if r != "elevator_cabin" and data.get("small_objects")])
            self.dynamic_layout.addWidget(QLabel("Source Room:")); self.dynamic_layout.addWidget(self.start_combo)
            self.target_list = QListWidget()
            self.target_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            self.target_list.addItems([r for r in scene["rooms"] if r != "elevator_cabin"])
            self.dynamic_layout.addWidget(QLabel("Target Room(s):")); self.dynamic_layout.addWidget(self.target_list)
            self.object_list = QListWidget()
            self.object_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            self.start_combo.currentTextChanged.connect(lambda rm: self._fill_objects(rm, scene))
            self._fill_objects(self.start_combo.currentText(), scene)
            self.dynamic_layout.addWidget(QLabel("Objects:")); self.dynamic_layout.addWidget(self.object_list)
        elif task_type == "tidying":
            self.object_list = QListWidget()
            self.object_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            for r_id, r_data in scene["rooms"].items():
                if r_id != "elevator_cabin":
                    for o_id, o_info in r_data.get("small_objects", {}).items():
                        if o_info.get("type") != "control": self.object_list.addItem(f"{o_id}@{r_id}")
            self.dynamic_layout.addWidget(QLabel("Objects to Tidy:")); self.dynamic_layout.addWidget(self.object_list)
        elif task_type == "guidance":
            all_rooms = [r for r in scene["rooms"] if r != "elevator_cabin"]
            self.start_combo, self.target_combo = QComboBox(), QComboBox()
            self.start_combo.addItems(all_rooms); self.target_combo.addItems(all_rooms)
            self.dynamic_layout.addWidget(QLabel("Start:")); self.dynamic_layout.addWidget(self.start_combo)
            self.dynamic_layout.addWidget(QLabel("End:")); self.dynamic_layout.addWidget(self.target_combo)

    def _fill_objects(self, room, scene):
        self.object_list.clear()
        if room in scene["rooms"]: self.object_list.addItems(scene["rooms"][room].get("small_objects", {}).keys())

    def on_generate_path(self):
        task_type, diff, scene = self.task_type_combo.currentText(), self.difficulty_combo.currentText(), self.scenes[self.scene_combo.currentText()]
        try:
            if task_type == "delivery":
                params = {"source_room": self.start_combo.currentText(), "target_rooms": [item.text() for item in self.target_list.selectedItems()], "objects": [item.text() for item in self.object_list.selectedItems()]}
                desc = "Delivery Task"
            elif task_type == "tidying":
                params = {"objects": [item.text().split('@')[0] for item in self.object_list.selectedItems()]}
                desc = "Tidying Task"
            else:
                params = {"waypoints": [self.start_combo.currentText(), self.target_combo.currentText()]}
                desc = "Guidance Task"

            task = Task(TaskType(task_type), DifficultyLevel(diff), params, desc)
            subtasks = plan_actions(scene, task)
            
            if not subtasks: return self.action_list.addItem("❌ 无法生成路径")

            self.action_list.clear()
            self._load_simulation(scene, subtasks)
        except Exception as e:
            self.action_list.addItem(f"❌ {e}")

    # ==========================================
    # 共享回放执行核心
    # ==========================================
    def _load_simulation(self, scene, actions):
        try:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("Play")

            self.current_actions = actions
            for i, a in enumerate(actions): self.action_list.addItem(f"{i+1}. {a}")

            self.simulation_snapshots = self.generate_snapshots(scene, actions)
            self.canvas.load_scene(self.simulation_snapshots[0])
            self.canvas.set_robot_position(self.simulation_snapshots[0]["agent"]["position"])

            self.step_slider.setMaximum(len(self.simulation_snapshots) - 1)
            self.step_slider.setValue(0)
            self.step_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.info_display.clear()
        except Exception as e:
            self.action_list.addItem(f"❌ 场景渲染初始化失败: {e}")

    def on_slider_change(self, value):
        if not self.simulation_snapshots or value >= len(self.simulation_snapshots): return
        snap = self.simulation_snapshots[value]
        
        if value == 0: self.canvas.load_scene(snap)
        else: self.canvas.update_from_scene(snap)
        
        self.canvas.set_robot_position(snap["agent"]["position"])

        list_offset = 2 if (self.action_list.count() > 0 and self.action_list.item(0).text().startswith("📖")) else 0

        if 0 < value <= len(self.current_actions):
            list_idx = value - 1 + list_offset
            self.action_list.setCurrentRow(list_idx)
            self.action_list.scrollToItem(self.action_list.item(list_idx))
            
            target = self._extract_target_from_action(self.current_actions[value-1])
            self.info_display.setText(f"--- Action Target ---\n{self._format_target_info(snap, target)}\n\n{'='*60}\n--- Agent ---\n{json.dumps(snap.get('agent',{}), indent=2, ensure_ascii=False)}")

    def toggle_animation(self):
        if not self.current_actions: return
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            self.timer.start(1000)
            self.play_btn.setText("Pause")
        self.is_playing = not self.is_playing

    def _animate_step(self):
        if self.step_slider.value() < self.step_slider.maximum():
            self.step_slider.setValue(self.step_slider.value() + 1)
        else:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("Play")

    def on_object_selected(self, node_id, info_dict):
        self.info_display.setText(f"Object: {node_id}\n\n{json.dumps(info_dict, indent=2, ensure_ascii=False)}")

    def generate_snapshots(self, scene, actions):
        snaps, cur = [copy.deepcopy(scene)], copy.deepcopy(scene)
        for a in actions:
            try:
                next_s = execute_subtask(cur, a)
                if next_s: cur = next_s
            except Exception as e:
                print(f"⚠️ 动作 '{a}' 在模拟器中执行失败: {e}")
            snaps.append(copy.deepcopy(cur))
        return snaps

    def _extract_target_from_action(self, action):
        match = re.search(r'\(([^)]+)', action)
        return match.group(1).split(',')[0].strip() if match else None

    def _format_target_info(self, snap, target):
        if not target: return "No target"
        if target in snap.get("rooms", {}): return f"Room: {target}\n{json.dumps(snap['rooms'][target], indent=2, ensure_ascii=False)}"
        for r_data in snap.get("rooms", {}).values():
            if target in r_data.get("large_objects", {}):
                rel = {k: v for k, v in r_data.get("small_objects", {}).items() if target in v.get("relation", {}).values()}
                return f"Large Obj: {target}\n{json.dumps({'info': r_data['large_objects'][target], 'attached': rel}, indent=2, ensure_ascii=False)}"
            if target in r_data.get("small_objects", {}): return f"Small Obj: {target}\n{json.dumps(r_data['small_objects'][target], indent=2, ensure_ascii=False)}"
        if target in snap.get("agent", {}).get("inventory", {}): return f"In Inventory: {target}\n{json.dumps(snap['agent']['inventory'][target], indent=2, ensure_ascii=False)}"
        return f"Target: {target} (Not found)"

    def on_editor_scene_saved(self, scene_name, scene_dict):
        self.scenes[scene_name] = copy.deepcopy(scene_dict)
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItems(self.scenes.keys())
        self.scene_combo.setCurrentText(scene_name)
        self.scene_combo.blockSignals(False)
        self.on_scene_changed(scene_name)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HSGRTPVisualizer()
    window.show()
    sys.exit(app.exec())
