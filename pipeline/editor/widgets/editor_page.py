from __future__ import annotations

import copy
import json
import math

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from editor.models.editable_scene import EditableComponent, EditableMacroZone, EditableObject, EditableRoom, EditableScene
from editor.schemas.enums import MACRO_ZONE_CATEGORIES, OBJECT_CATEGORIES, OBJECT_FORMS
from editor.schemas.object_type_templates import (
    CATEGORY_SCHEMAS,
    get_category_schema,
    get_component_template_config,
    get_component_template_names,
)
from editor.services.py_exporter import export_scene_python
from editor.services.scene_builder import build_scene_dict, create_object_from_template, editable_scene_from_scene_dict
from editor.services.scene_registry import GENERATED_DIR, load_all_scenes
from editor.services.validator import check_scene_dict, validate_scene_dict


COMPONENT_ROLE_FIELD_RULES = {
    "door": {
        "state_variables": {"is_open", "cleanliness", "is_locked"},
        "capabilities": {"openable"},
    },
    "drawer": {
        "state_variables": {"is_open", "fill_level", "cleanliness", "is_locked"},
        "capabilities": {"openable", "receptacle", "fillable"},
    },
    "drum": {
        "state_variables": {"load_level", "cleanliness"},
        "capabilities": {"receptacle", "fillable"},
    },
    "shelf": {
        "state_variables": {"cleanliness"},
        "capabilities": {"receptacle"},
    },
    "panel": {
        "state_variables": {"cleanliness"},
        "capabilities": {"pressable", "toggleable"},
    },
    "handle": {
        "state_variables": {"cleanliness"},
        "capabilities": set(),
    },
}

COMPONENT_CATEGORY_FIELD_RULES = {
    "button": {
        "static_attributes": {"material", "movable", "fragile", "is_container", "is_support"},
        "state_variables": {"is_pressed", "enabled"},
        "capabilities": {"pressable"},
        "relations": {"part_of", "controls", "inside", "outside", "in_room"},
    },
    "container": {
        "static_attributes": {"material", "movable", "fragile", "is_container", "is_support", "capacity"},
        "state_variables": {"is_open", "fill_level", "occupancy", "cleanliness"},
        "capabilities": {"receptacle", "fillable", "openable", "pickupable"},
        "relations": {"part_of", "contains", "inside", "in_room"},
    },
}


class EditorGraphCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plotter = QtInteractor(self, auto_update=False)
        layout.addWidget(self.plotter)
        self.setMinimumHeight(320)

        self.scene = {}
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.node_id_order: list[str] = []
        self.edge_actor_map: dict[object, dict] = {}
        self.points_actor = None
        self.on_node_clicked = None
        self.on_edge_clicked = None

        self.COLORS = {
            "macro": "#FF4444",
            "room": "#44AA44",
            "elevator": "#8844DD",
            "large_object": "#0B51CA",
            "small_object": "#EB630F",
            "elevator_button": "#DD44AA",
            "component": "#0B51CA",
        }
        self.EDGE_CONFIG = {
            "parent-child": {"color": "#FF2222", "width": 2.5},
            "neighbor": {"color": "#22AA22", "width": 2.0},
            "elevator-connection": {"color": "#8844DD", "width": 3.0},
            "contains": {"color": "#4488FF", "width": 1.5},
            "attached": {"color": "#AA6600", "width": 1.0},
        }
        self.plotter.set_background("white")

    def render_scene_graph(self, scene_dict: dict, focus_object_id: str = ""):
        self.scene = scene_dict
        self._build_graph(focus_object_id)
        self._render_static()

    def _build_graph(self, focus_object_id: str = ""):
        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        MACRO_LEVEL = 4
        ROOM_LEVEL = 2
        ELEVATOR_LEVEL = 1
        LARGE_OBJECT_LEVEL = 0.3
        SMALL_OBJECT_LEVEL = -0.5

        macro_zones = self.scene.get("macro_zones", {})
        rooms = self.scene.get("rooms", {})

        macro_count = len(macro_zones)
        for index, (zone_id, _zone_data) in enumerate(macro_zones.items()):
            angle = 2 * np.pi * index / macro_count if macro_count else 0
            x, y = 4.5 * np.cos(angle), 4.5 * np.sin(angle)
            nodes[zone_id] = {"pos": (x, y, MACRO_LEVEL), "type": "macro", "label": zone_id, "kind": "macro_zone"}

        regular_rooms = {}
        elevator_rooms = {}
        for room_id, room_data in rooms.items():
            if room_data.get("floor") == "elevator":
                elevator_rooms[room_id] = room_data
            else:
                regular_rooms[room_id] = room_data

        room_count = len(regular_rooms)
        for index, (room_id, room_data) in enumerate(regular_rooms.items()):
            angle = 2 * np.pi * index / room_count if room_count else 0
            x, y = 3.5 * np.cos(angle), 3.5 * np.sin(angle)
            nodes[room_id] = {
                "pos": (x, y, ROOM_LEVEL),
                "type": "room",
                "label": room_id,
                "kind": "room",
                "room_id": room_id,
                "floor": room_data.get("floor"),
            }
            floor = room_data.get("floor")
            if floor and floor in nodes:
                edges.append({"src": floor, "dst": room_id, "etype": "parent-child"})
            for neighbor in room_data.get("neighbor", []):
                if neighbor in regular_rooms:
                    edges.append({"src": room_id, "dst": neighbor, "etype": "neighbor"})

        for index, (room_id, room_data) in enumerate(elevator_rooms.items()):
            angle = 2 * np.pi * index / max(1, len(elevator_rooms))
            x, y = 0.8 * np.cos(angle), 0.8 * np.sin(angle)
            nodes[room_id] = {
                "pos": (x, y, ELEVATOR_LEVEL),
                "type": "elevator",
                "label": room_id,
                "kind": "room",
                "room_id": room_id,
            }
            for neighbor in room_data.get("neighbor", []):
                if neighbor in regular_rooms:
                    edges.append({"src": room_id, "dst": neighbor, "etype": "elevator-connection"})

        for room_id, room_data in rooms.items():
            if room_id not in nodes:
                continue
            room_pos = nodes[room_id]["pos"]
            room_type = nodes[room_id]["type"]

            for bucket_name, level, radius, node_type in [
                ("large_objects", LARGE_OBJECT_LEVEL, 1.2 if room_type != "elevator" else 0.8, "large_object"),
                ("small_objects", SMALL_OBJECT_LEVEL, 0.9 if room_type != "elevator" else 0.6, "small_object"),
            ]:
                bucket = room_data.get(bucket_name, {})
                count = len(bucket)
                for index, (item_id, item_data) in enumerate(bucket.items()):
                    angle = 2 * np.pi * index / count if count else 0
                    x = room_pos[0] + radius * np.cos(angle)
                    y = room_pos[1] + radius * np.sin(angle)
                    inferred_type = self._infer_canvas_node_type(item_id, item_data, node_type)
                    nodes[item_id] = {
                        "pos": (x, y, level),
                        "type": inferred_type,
                        "label": item_id,
                        "kind": "object",
                        "object_id": item_id,
                        "room_id": room_id,
                        "bucket": bucket_name,
                    }
                    edges.append(
                        {
                            "src": room_id,
                            "dst": item_id,
                            "etype": "contains",
                            "relation_name": "in_room",
                            "source_kind": "object",
                            "room_id": room_id,
                            "source_id": item_id,
                        }
                    )
                    self._append_relation_edges(edges, item_id, item_data, room_id, "object")

                    if focus_object_id and item_id != focus_object_id:
                        continue
                    components = item_data.get("components", {})
                    component_count = len(components)
                    for component_index, (component_id, component_data) in enumerate(components.items()):
                        component_angle = 2 * np.pi * component_index / max(1, component_count)
                        component_x = x + 0.45 * np.cos(component_angle)
                        component_y = y + 0.45 * np.sin(component_angle)
                        component_type = self._infer_canvas_node_type(component_id, component_data, "component")
                        nodes[component_id] = {
                            "pos": (component_x, component_y, SMALL_OBJECT_LEVEL),
                            "type": component_type,
                            "label": component_id,
                            "kind": "component",
                            "component_id": component_id,
                            "parent_object_id": item_id,
                            "room_id": room_id,
                        }
                        edges.append(
                            {
                                "src": item_id,
                                "dst": component_id,
                                "etype": "contains",
                                "relation_name": "part_of",
                                "source_kind": "component",
                                "room_id": room_id,
                                "source_id": component_id,
                                "parent_object_id": item_id,
                            }
                        )
                        self._append_relation_edges(edges, component_id, component_data, room_id, "component", item_id)

        self.nodes = nodes
        self.edges = self._dedupe_edges(edges)
        self.node_id_order = list(self.nodes.keys())

    def _render_static(self):
        self.plotter.clear()
        self.edge_actor_map = {}
        if not self.nodes:
            return
        points = np.array([info["pos"] for info in self.nodes.values()])
        labels = [info["label"] for info in self.nodes.values()]
        colors = np.array([self._get_color(info["type"]) for info in self.nodes.values()])

        self.points_actor = self.plotter.add_points(
            points,
            scalars=(colors * 255).astype(np.uint8),
            rgb=True,
            point_size=20,
            render_points_as_spheres=True,
            ambient=0.3,
            opacity=0.9,
            name="nodes",
        )
        self.plotter.add_point_labels(
            points=points,
            labels=labels,
            font_size=12,
            text_color="black",
            point_size=0,
            shape_opacity=0,
            always_visible=True,
            name="node_labels",
        )

        for edge in self.edges:
            src = edge["src"]
            dst = edge["dst"]
            if src not in self.nodes or dst not in self.nodes:
                continue
            config = self.EDGE_CONFIG.get(edge["etype"], {"color": "#777777", "width": 1.0})
            actor = self.plotter.add_mesh(
                pv.Line(self.nodes[src]["pos"], self.nodes[dst]["pos"]),
                color=config["color"],
                line_width=config["width"],
                opacity=0.9,
            )
            self.edge_actor_map[actor] = edge

        title = f"Scene: {self.scene.get('name', 'Unknown')}"
        self.plotter.add_text(title, font_size=16, position="upper_edge", color="black", name="title")
        self.plotter.view_isometric()
        self.plotter.camera.elevation = 25
        self.plotter.camera.azimuth = 45
        self.plotter.reset_camera()
        self.plotter.track_click_position(self._on_mouse_click, side="left")
        self.plotter.render()

    def _get_color(self, node_type: str):
        hex_color = self.COLORS.get(node_type, "#888888")
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    def _infer_canvas_node_type(self, item_id: str, item_data: dict, default_type: str) -> str:
        subtype = item_data.get("subtype")
        capabilities = item_data.get("capabilities", {})
        if subtype == "button" or capabilities.get("pressable") is True or "button" in item_id:
            return "elevator_button" if "elevator" in item_id else "small_object"
        if default_type == "component":
            return "large_object" if item_data.get("static_attributes", {}).get("is_support") else "small_object"
        return default_type

    def _append_relation_edges(
        self,
        edges: list[dict],
        item_id: str,
        item_data: dict,
        room_id: str,
        source_kind: str,
        parent_object_id: str = "",
    ):
        for relation_name, target in item_data.get("relations", {}).items():
            if not target:
                continue
            etype = "attached"
            if relation_name == "controls":
                etype = "neighbor"
            edge = {
                "src": item_id,
                "dst": target,
                "etype": etype,
                "relation_name": relation_name,
                "source_kind": source_kind,
                "room_id": room_id,
                "source_id": item_id,
                "parent_object_id": parent_object_id,
            }
            edges.append(edge)

    def _dedupe_edges(self, edges: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for edge in edges:
            key = (
                edge["src"],
                edge["dst"],
                edge["etype"],
                edge.get("relation_name", ""),
                edge.get("source_kind", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        return deduped

    def _on_mouse_click(self, _event):
        if not self.nodes:
            return
        click_position = self.plotter.click_position
        if click_position is None:
            return
        x, y = int(click_position[0]), int(click_position[1])
        self.plotter.picker.Pick(x, y, 0, self.plotter.renderer)
        actor = self.plotter.picker.GetActor()
        if actor in self.edge_actor_map and self.on_edge_clicked:
            self.on_edge_clicked(self.edge_actor_map[actor])
            return
        point_id = self.plotter.picker.GetPointId()
        if 0 <= point_id < len(self.node_id_order):
            node_id = self.node_id_order[point_id]
            if self.on_node_clicked:
                self.on_node_clicked(node_id, self.nodes[node_id])
            return


class SceneEditorPage(QWidget):
    scene_saved = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.available_scenes = load_all_scenes()
        self.editable_scene = EditableScene()
        self.editable_scene.ensure_macro_zones(1)
        self._object_field_inputs: dict[str, dict[str, QWidget]] = {}
        self._component_field_inputs: dict[str, dict[str, QWidget]] = {}
        self.init_ui()
        self.load_scene("custom_scene")

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        tree_panel = self._build_tree_panel()
        preview_panel = self._build_preview_panel()
        editor_panel = self._build_editor_panel()
        splitter.addWidget(tree_panel)
        splitter.addWidget(preview_panel)
        splitter.addWidget(editor_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([210, 1080, 320])
        layout.addWidget(splitter)
        self.graph_canvas.on_node_clicked = self.on_graph_node_clicked
        self.graph_canvas.on_edge_clicked = self.on_graph_edge_clicked

    def _build_tree_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)

        self.scene_select_combo = QComboBox()
        self.scene_select_combo.addItems(["custom_scene", *sorted(self.available_scenes.keys())])
        self.scene_select_combo.currentTextChanged.connect(self.load_scene)
        layout.addWidget(self._section_label("Scene"))
        layout.addWidget(self.scene_select_combo)

        self.zone_list = QListWidget()
        self.zone_list.currentTextChanged.connect(self.on_zone_selected)
        layout.addWidget(self._section_label("Macro Zones"))
        layout.addWidget(self.zone_list)

        self.room_list = QListWidget()
        self.room_list.currentTextChanged.connect(self.on_room_selected)
        layout.addWidget(self._section_label("Rooms"))
        layout.addWidget(self.room_list)

        self.object_list = QListWidget()
        self.object_list.currentTextChanged.connect(self.on_object_selected)
        layout.addWidget(self._section_label("Objects"))
        layout.addWidget(self.object_list)

        self.component_list = QListWidget()
        self.component_list.currentTextChanged.connect(self.on_component_selected)
        self.component_list_label = self._section_label("Components")
        layout.addWidget(self.component_list_label)
        layout.addWidget(self.component_list)
        return panel

    def _build_preview_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setMinimumWidth(900)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        preview_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_splitter.setChildrenCollapsible(False)

        graph_panel = QFrame()
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.setSpacing(0)
        self.graph_canvas = EditorGraphCanvas()
        self.graph_canvas.setMinimumHeight(460)
        self.graph_canvas.setMaximumHeight(540)
        graph_layout.addWidget(self.graph_canvas)

        info_panel = QFrame()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(8)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMinimumHeight(180)
        info_layout.addWidget(self._section_label("JSON Preview"))
        info_layout.addWidget(self.preview_text)

        preview_splitter.addWidget(graph_panel)
        preview_splitter.addWidget(info_panel)
        preview_splitter.setStretchFactor(0, 4)
        preview_splitter.setStretchFactor(1, 2)
        preview_splitter.setSizes([620, 260])
        layout.addWidget(preview_splitter)

        button_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Export")
        self.preview_btn.clicked.connect(self.preview_export)
        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self.validate_scene)
        self.save_btn = QPushButton("Save Scene")
        self.save_btn.clicked.connect(self.save_scene)
        self._make_button_compact(self.preview_btn)
        self._make_button_compact(self.validate_btn)
        self._make_button_compact(self.save_btn)
        button_row.addWidget(self.preview_btn)
        button_row.addWidget(self.validate_btn)
        button_row.addWidget(self.save_btn)
        layout.addLayout(button_row)
        layout.setStretch(0, 1)
        return panel

    def _build_editor_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(320)
        self.editor_scroll = QScrollArea()
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.editor_scroll.setMaximumWidth(320)
        container = QWidget()
        container.setMaximumWidth(304)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scene_group = QGroupBox("Scene")
        self._style_group_title(scene_group)
        scene_form = QFormLayout(scene_group)
        self._configure_compact_form(scene_form)
        self.scene_name_edit = QLineEdit()
        self.zone_count_spin = QSpinBox()
        self.zone_count_spin.setRange(1, 20)
        self.zone_count_spin.valueChanged.connect(self.on_zone_count_changed)
        self.elevator_checkbox = QCheckBox("Enable auto elevator generation")
        scene_form.addRow("Scene name", self.scene_name_edit)
        scene_form.addRow("Macro zone count", self.zone_count_spin)
        scene_form.addRow("", self.elevator_checkbox)
        layout.addWidget(scene_group)

        zone_group = QGroupBox("Selected Macro Zone")
        self._style_group_title(zone_group)
        zone_form = QFormLayout(zone_group)
        self._configure_compact_form(zone_form)
        self.zone_id_edit = QLineEdit()
        self.zone_category_combo = QComboBox()
        self.zone_category_combo.addItems(MACRO_ZONE_CATEGORIES)
        self.zone_elevator_checkbox = QCheckBox("Elevator access")
        self.save_zone_btn = QPushButton("Save Macro Zone")
        self.save_zone_btn.clicked.connect(self.save_zone)
        self._make_button_compact(self.save_zone_btn)
        zone_form.addRow("zone_id", self.zone_id_edit)
        zone_form.addRow("category", self.zone_category_combo)
        zone_form.addRow("", self.zone_elevator_checkbox)
        zone_form.addRow("", self.save_zone_btn)
        layout.addWidget(zone_group)

        room_group = QGroupBox("Selected Room")
        self._style_group_title(room_group)
        room_layout = QVBoxLayout(room_group)
        room_form = QFormLayout()
        self._configure_compact_form(room_form)
        self.room_id_edit = QLineEdit()
        room_form.addRow("room_id", self.room_id_edit)
        self.room_zone_label = QLabel("")
        room_form.addRow("macro_zone", self.room_zone_label)
        room_layout.addLayout(room_form)
        self.room_neighbors_list = QListWidget()
        self.room_neighbors_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        room_layout.addWidget(QLabel("neighbor"))
        room_layout.addWidget(self.room_neighbors_list)
        room_btn_row = QHBoxLayout()
        self.add_room_btn = QPushButton("Add Room")
        self.add_room_btn.clicked.connect(self.add_room)
        self.save_room_btn = QPushButton("Save Room")
        self.save_room_btn.clicked.connect(self.save_room)
        self.delete_room_btn = QPushButton("Delete Room")
        self.delete_room_btn.clicked.connect(self.delete_room)
        self._make_button_compact(self.add_room_btn)
        self._make_button_compact(self.save_room_btn)
        self._make_button_compact(self.delete_room_btn)
        room_btn_row.addWidget(self.add_room_btn)
        room_btn_row.addWidget(self.save_room_btn)
        room_btn_row.addWidget(self.delete_room_btn)
        room_layout.addLayout(room_btn_row)
        layout.addWidget(room_group)

        object_group = QGroupBox("Object Editor")
        self._style_group_title(object_group)
        object_layout = QVBoxLayout(object_group)
        object_form = QFormLayout()
        self._configure_compact_form(object_form)
        self.object_form_combo = QComboBox()
        self.object_form_combo.addItems(OBJECT_FORMS)
        self.object_form_combo.currentTextChanged.connect(self.on_object_form_changed)
        self.object_category_combo = QComboBox()
        self.object_category_combo.addItems(OBJECT_CATEGORIES)
        self.object_category_combo.currentTextChanged.connect(self.on_object_category_changed)
        self.object_group_combo = QComboBox()
        self.object_group_combo.addItems(["large_objects", "small_objects"])
        self.object_id_edit = QLineEdit()
        object_form.addRow("object_id", self.object_id_edit)
        object_form.addRow("object form", self.object_form_combo)
        object_form.addRow("category", self.object_category_combo)
        object_form.addRow("object bucket", self.object_group_combo)
        object_layout.addLayout(object_form)

        self.object_static_form = QFormLayout()
        self.object_state_form = QFormLayout()
        self.object_capability_form = QFormLayout()
        self.object_relation_form = QFormLayout()
        self._configure_compact_form(self.object_static_form)
        self._configure_compact_form(self.object_state_form)
        self._configure_compact_form(self.object_capability_form)
        self._configure_compact_form(self.object_relation_form)
        object_layout.addWidget(self._section_label("Static Attributes"))
        object_layout.addLayout(self.object_static_form)
        object_layout.addWidget(self._section_label("State Variables"))
        object_layout.addLayout(self.object_state_form)
        object_layout.addWidget(self._section_label("Capabilities"))
        object_layout.addLayout(self.object_capability_form)
        object_layout.addWidget(self._section_label("Relations"))
        object_layout.addLayout(self.object_relation_form)
        object_btn_row = QHBoxLayout()
        self.add_object_btn = QPushButton("Add Object")
        self.add_object_btn.clicked.connect(self.add_object)
        self.save_object_btn = QPushButton("Save Object")
        self.save_object_btn.clicked.connect(self.save_object)
        self.delete_object_btn = QPushButton("Delete Object")
        self.delete_object_btn.clicked.connect(self.delete_object)
        self._make_button_compact(self.add_object_btn)
        self._make_button_compact(self.save_object_btn)
        self._make_button_compact(self.delete_object_btn)
        object_btn_row.addWidget(self.add_object_btn)
        object_btn_row.addWidget(self.save_object_btn)
        object_btn_row.addWidget(self.delete_object_btn)
        object_layout.addLayout(object_btn_row)
        layout.addWidget(object_group)

        self.component_group = QGroupBox("Component Editor")
        self._style_group_title(self.component_group)
        component_layout = QVBoxLayout(self.component_group)
        component_form = QFormLayout()
        self._configure_compact_form(component_form)
        self.component_category_combo = QComboBox()
        self.component_category_combo.addItems(["component", "button", "container"])
        self.component_category_combo.currentTextChanged.connect(self.on_component_category_changed)
        self.component_id_edit = QLineEdit()
        self.component_template_combo = QComboBox()
        self.component_template_combo.addItems(get_component_template_names())
        self.component_template_combo.currentTextChanged.connect(self.on_component_template_changed)
        component_form.addRow("component_id", self.component_id_edit)
        component_form.addRow("component template", self.component_template_combo)
        component_form.addRow("category", self.component_category_combo)
        component_layout.addLayout(component_form)
        self.component_static_form = QFormLayout()
        self.component_state_form = QFormLayout()
        self.component_capability_form = QFormLayout()
        self.component_relation_form = QFormLayout()
        self._configure_compact_form(self.component_static_form)
        self._configure_compact_form(self.component_state_form)
        self._configure_compact_form(self.component_capability_form)
        self._configure_compact_form(self.component_relation_form)
        component_layout.addWidget(self._section_label("Static Attributes"))
        component_layout.addLayout(self.component_static_form)
        component_layout.addWidget(self._section_label("State Variables"))
        component_layout.addLayout(self.component_state_form)
        component_layout.addWidget(self._section_label("Capabilities"))
        component_layout.addLayout(self.component_capability_form)
        component_layout.addWidget(self._section_label("Relations"))
        component_layout.addLayout(self.component_relation_form)
        component_btn_row = QHBoxLayout()
        self.add_component_btn = QPushButton("Add\nComponent")
        self.add_component_btn.clicked.connect(self.add_component)
        self.save_component_btn = QPushButton("Save\nComponent")
        self.save_component_btn.clicked.connect(self.save_component)
        self.delete_component_btn = QPushButton("Delete\nComponent")
        self.delete_component_btn.clicked.connect(self.delete_component)
        self._make_button_compact(self.add_component_btn, multiline=True)
        self._make_button_compact(self.save_component_btn, multiline=True)
        self._make_button_compact(self.delete_component_btn, multiline=True)
        component_btn_row.addWidget(self.add_component_btn)
        component_btn_row.addWidget(self.save_component_btn)
        component_btn_row.addWidget(self.delete_component_btn)
        component_layout.addLayout(component_btn_row)
        layout.addWidget(self.component_group)

        layout.addStretch()
        self.editor_scroll.setWidget(container)
        return self.editor_scroll

    def load_scene(self, scene_name: str):
        if scene_name and scene_name in self.available_scenes:
            self.editable_scene = editable_scene_from_scene_dict(scene_name, copy.deepcopy(self.available_scenes[scene_name]))
        else:
            self.editable_scene = EditableScene(source_scene="custom_scene", name="custom_scene")
            self.editable_scene.ensure_macro_zones(1)
        self.scene_name_edit.setText(self.editable_scene.name)
        self.zone_count_spin.blockSignals(True)
        self.zone_count_spin.setValue(self.editable_scene.zone_count)
        self.zone_count_spin.blockSignals(False)
        self.elevator_checkbox.setChecked(self.editable_scene.elevator_enabled)
        self.refresh_all(render_preview=True)

    def refresh_all(self, render_preview: bool = False):
        self.populate_zone_list()
        self.populate_room_list()
        self.populate_object_list()
        self.populate_component_list()
        self.populate_neighbor_options()
        self.rebuild_object_forms()
        self.rebuild_component_forms()
        self.update_component_editor_visibility()
        if render_preview:
            self.preview_export()

    def populate_zone_list(self):
        current = self.current_zone_id()
        self.zone_list.clear()
        for zone in self.editable_scene.macro_zones:
            item = QListWidgetItem(zone.zone_id)
            item.setData(256, zone.zone_id)
            self.zone_list.addItem(item)
            if current == zone.zone_id:
                self.zone_list.setCurrentItem(item)
        if self.zone_list.currentItem() is None and self.zone_list.count():
            self.zone_list.setCurrentRow(0)

    def populate_room_list(self):
        current_zone = self.current_zone_id()
        current_room = self.current_room_id()
        self.room_list.clear()
        for room in self.editable_scene.rooms:
            if current_zone and room.floor != current_zone:
                continue
            item = QListWidgetItem(room.room_id)
            item.setData(256, room.room_id)
            self.room_list.addItem(item)
            if room.room_id == current_room:
                self.room_list.setCurrentItem(item)
        if self.room_list.currentItem() is None and self.room_list.count():
            self.room_list.setCurrentRow(0)

    def populate_object_list(self):
        self.object_list.clear()
        room = self.editable_scene.get_room(self.current_room_id())
        if room is None:
            return
        current_object = self.current_object_id()
        for obj in room.objects:
            item = QListWidgetItem(f"{obj.object_id} ({obj.category})")
            item.setData(256, obj.object_id)
            self.object_list.addItem(item)
            if obj.object_id == current_object:
                self.object_list.setCurrentItem(item)
        if self.object_list.currentItem() is None and self.object_list.count():
            self.object_list.setCurrentRow(0)

    def populate_component_list(self):
        self.component_list.clear()
        obj = self.current_object()
        if obj is None or obj.object_form != "composite object":
            return
        current_component = self.current_component_id()
        for component in obj.components:
            item = QListWidgetItem(f"{component.object_id} ({component.category})")
            item.setData(256, component.object_id)
            self.component_list.addItem(item)
            if component.object_id == current_component:
                self.component_list.setCurrentItem(item)
        if self.component_list.currentItem() is None and self.component_list.count():
            self.component_list.setCurrentRow(0)

    def current_zone_id(self) -> str:
        item = self.zone_list.currentItem()
        if item:
            return item.data(256)
        return self.editable_scene.macro_zones[0].zone_id if self.editable_scene.macro_zones else ""

    def current_room_id(self) -> str:
        item = self.room_list.currentItem()
        return item.data(256) if item else ""

    def current_object_id(self) -> str:
        item = self.object_list.currentItem()
        return item.data(256) if item else ""

    def current_component_id(self) -> str:
        item = self.component_list.currentItem()
        return item.data(256) if item else ""

    def current_object(self) -> EditableObject | None:
        return self.editable_scene.find_object(self.current_room_id(), self.current_object_id())

    def current_component(self) -> EditableComponent | None:
        obj = self.current_object()
        if obj is None:
            return None
        return next((component for component in obj.components if component.object_id == self.current_component_id()), None)

    def on_zone_count_changed(self, value: int):
        self.editable_scene.ensure_macro_zones(value)
        self.refresh_all()

    def on_zone_selected(self, _text: str):
        zone = self.editable_scene.get_zone(self.current_zone_id())
        if zone is None:
            return
        self.zone_id_edit.setText(zone.zone_id)
        self.zone_category_combo.setCurrentText(zone.category)
        self.zone_elevator_checkbox.setChecked(zone.has_elevator_access)
        self.populate_room_list()
        self.populate_neighbor_options()

    def on_room_selected(self, _text: str):
        room = self.editable_scene.get_room(self.current_room_id())
        if room is None:
            self.room_id_edit.clear()
            self.update_component_editor_visibility()
            return
        self.room_id_edit.setText(room.room_id)
        self.room_zone_label.setText(room.floor)
        self.populate_neighbor_options(room.neighbor)
        self.populate_object_list()
        self.populate_component_list()
        self.update_component_editor_visibility()

    def on_object_selected(self, _text: str):
        obj = self.current_object()
        if obj is None:
            self.update_component_editor_visibility()
            return
        self.object_form_combo.setCurrentText(obj.object_form)
        self.object_category_combo.setCurrentText(obj.category)
        self.object_group_combo.setCurrentText(obj.object_group)
        self.object_id_edit.setText(obj.object_id)
        self.rebuild_object_forms()
        self.fill_dynamic_section(self._object_field_inputs, obj)
        self.populate_component_list()
        self.update_component_editor_visibility()

    def on_component_selected(self, _text: str):
        component = self.current_component()
        if component is None:
            return
        self.component_template_combo.blockSignals(True)
        self.component_template_combo.setCurrentText(
            component.template_name if component.template_name in get_component_template_names() else "custom"
        )
        self.component_template_combo.blockSignals(False)
        self.component_category_combo.setCurrentText(component.category)
        self.rebuild_component_forms()
        self.component_id_edit.setText(component.object_id)
        self.fill_dynamic_section(self._component_field_inputs, component)
        self._apply_component_field_visibility()

    def save_zone(self):
        zone = self.editable_scene.get_zone(self.current_zone_id())
        if zone is None:
            return
        new_zone_id = self.zone_id_edit.text().strip()
        if not new_zone_id:
            self._warn("zone_id is required.")
            return
        if new_zone_id != zone.zone_id and self.editable_scene.get_zone(new_zone_id):
            self._warn("Another macro zone already uses that zone_id.")
            return
        old_zone_id = zone.zone_id
        zone.zone_id = new_zone_id
        zone.category = self.zone_category_combo.currentText()
        zone.has_elevator_access = self.zone_elevator_checkbox.isChecked()
        for room in self.editable_scene.rooms:
            if room.floor == old_zone_id:
                room.floor = new_zone_id
        self.editable_scene._sync_zone_room_names()
        self.refresh_all(render_preview=True)
        self.select_zone(new_zone_id)

    def add_room(self):
        room_id = self.room_id_edit.text().strip()
        if not room_id:
            self._warn("room_id is required.")
            return
        if self.editable_scene.get_room(room_id):
            self._warn("room_id already exists.")
            return
        room = EditableRoom(room_id=room_id, floor=self.current_zone_id(), neighbor=self.selected_neighbors(), objects=[])
        self.editable_scene.add_room(room)
        self._ensure_bidirectional_neighbors(room)
        self.refresh_all(render_preview=True)
        self.select_room(room_id)

    def save_room(self):
        room = self.editable_scene.get_room(self.current_room_id())
        if room is None:
            return
        new_room_id = self.room_id_edit.text().strip()
        if not new_room_id:
            self._warn("room_id is required.")
            return
        if new_room_id != room.room_id and self.editable_scene.get_room(new_room_id):
            self._warn("Another room already uses that room_id.")
            return
        old_room_id = room.room_id
        room.room_id = new_room_id
        room.floor = self.current_zone_id()
        room.neighbor = self.selected_neighbors()
        for other_room in self.editable_scene.rooms:
            other_room.neighbor = [new_room_id if neighbor == old_room_id else neighbor for neighbor in other_room.neighbor]
        self._ensure_bidirectional_neighbors(room)
        self.editable_scene._sync_zone_room_names()
        self.refresh_all(render_preview=True)
        self.select_room(new_room_id)

    def delete_room(self):
        room_id = self.current_room_id()
        if room_id:
            self.editable_scene.remove_room(room_id)
            self.room_id_edit.clear()
            self.room_zone_label.clear()
            self.refresh_all(render_preview=True)

    def add_object(self):
        room = self.editable_scene.get_room(self.current_room_id())
        if room is None:
            self._warn("Select a room first.")
            return
        obj = self._build_object_from_form()
        if any(existing.object_id == obj.object_id for existing in room.objects):
            self._warn("object_id already exists in this room.")
            return
        room.objects.append(obj)
        self.refresh_all(render_preview=True)
        self.select_object(obj.object_id)

    def save_object(self):
        room = self.editable_scene.get_room(self.current_room_id())
        obj = self.current_object()
        if room is None or obj is None:
            return
        new_obj = self._build_object_from_form()
        if new_obj.object_id != obj.object_id and any(existing.object_id == new_obj.object_id for existing in room.objects):
            self._warn("Another object already uses that object_id in this room.")
            return
        obj.object_id = new_obj.object_id
        obj.name = new_obj.name
        obj.category = new_obj.category
        obj.object_form = new_obj.object_form
        obj.template_name = new_obj.template_name
        obj.description = new_obj.description
        obj.object_group = new_obj.object_group
        obj.static_attributes = new_obj.static_attributes
        obj.state_variables = new_obj.state_variables
        obj.capabilities = new_obj.capabilities
        obj.relations = new_obj.relations
        if new_obj.object_form == "simple object":
            obj.components = []
        else:
            existing_by_id = {component.object_id: component for component in obj.components}
            obj.components = [existing_by_id.get(component.object_id, component) for component in new_obj.components] or obj.components
            for component in obj.components:
                component.relations["part_of"] = obj.object_id
                if component.object_id == "start_button" and obj.template_name == "washing_machine_template":
                    component.relations["controls"] = obj.object_id
        self.refresh_all(render_preview=True)
        self.select_object(obj.object_id)

    def delete_object(self):
        room = self.editable_scene.get_room(self.current_room_id())
        object_id = self.current_object_id()
        if room and object_id:
            room.objects = [obj for obj in room.objects if obj.object_id != object_id]
            self.refresh_all(render_preview=True)

    def add_component(self):
        obj = self.current_object()
        if obj is None:
            self._warn("Select a composite object first.")
            return
        if obj.object_form != "composite object":
            self._warn("Only composite objects can own components.")
            return
        component = self._build_component_from_form(obj.object_id)
        if any(existing.object_id == component.object_id for existing in obj.components):
            self._warn("component_id already exists on this object.")
            return
        obj.components.append(component)
        self.refresh_all(render_preview=True)
        self.select_component(component.object_id)

    def save_component(self):
        obj = self.current_object()
        component = self.current_component()
        if obj is None or component is None:
            return
        new_component = self._build_component_from_form(obj.object_id)
        if new_component.object_id != component.object_id and any(existing.object_id == new_component.object_id for existing in obj.components):
            self._warn("Another component already uses that component_id.")
            return
        component.object_id = new_component.object_id
        component.name = new_component.name
        component.category = new_component.category
        component.description = new_component.description
        component.static_attributes = new_component.static_attributes
        component.state_variables = new_component.state_variables
        component.capabilities = new_component.capabilities
        component.relations = new_component.relations
        self.refresh_all(render_preview=True)
        self.select_component(component.object_id)

    def delete_component(self):
        obj = self.current_object()
        component_id = self.current_component_id()
        if obj and component_id:
            obj.components = [component for component in obj.components if component.object_id != component_id]
            self.refresh_all(render_preview=True)

    def on_object_form_changed(self, object_form: str):
        category = self.object_category_combo.currentText()
        allowed_forms = get_category_schema(category)["allowed_forms"]
        if object_form not in allowed_forms:
            self.object_form_combo.setCurrentText(allowed_forms[0])
            return
        self.rebuild_object_forms()
        self.update_component_editor_visibility()

    def on_object_category_changed(self, _category: str):
        schema = get_category_schema(self.object_category_combo.currentText())
        allowed_forms = schema["allowed_forms"]
        if self.object_form_combo.currentText() not in allowed_forms:
            self.object_form_combo.setCurrentText(allowed_forms[0])
        self.object_group_combo.setCurrentText(schema["default_object_group"])
        self.rebuild_object_forms()
        self.update_component_editor_visibility()

    def on_component_category_changed(self, _category: str):
        self.rebuild_component_forms()
        self._apply_component_field_visibility()

    def on_component_template_changed(self, template_name: str):
        if not template_name:
            return
        self._apply_component_template_to_form(template_name)

    def rebuild_object_forms(self):
        schema = CATEGORY_SCHEMAS[self.object_category_combo.currentText()]
        self._clear_form(self.object_static_form)
        self._clear_form(self.object_state_form)
        self._clear_form(self.object_capability_form)
        self._clear_form(self.object_relation_form)
        self._object_field_inputs = {
            "static_attributes": self._populate_section_form(self.object_static_form, schema["allowed_static_attributes"]),
            "state_variables": self._populate_section_form(self.object_state_form, schema["allowed_state_variables"]),
            "capabilities": self._populate_section_form(self.object_capability_form, schema["allowed_capabilities"]),
            "relations": self._populate_relation_form(self.object_relation_form, schema["allowed_relations"], parent_object_id=""),
        }

    def rebuild_component_forms(self):
        schema = CATEGORY_SCHEMAS[self.component_category_combo.currentText()]
        self._clear_form(self.component_static_form)
        self._clear_form(self.component_state_form)
        self._clear_form(self.component_capability_form)
        self._clear_form(self.component_relation_form)
        parent_id = self.current_object_id()
        self._component_field_inputs = {
            "static_attributes": self._populate_section_form(self.component_static_form, schema["allowed_static_attributes"]),
            "state_variables": self._populate_section_form(self.component_state_form, schema["allowed_state_variables"]),
            "capabilities": self._populate_section_form(self.component_capability_form, schema["allowed_capabilities"]),
            "relations": self._populate_relation_form(self.component_relation_form, schema["allowed_relations"], parent_object_id=parent_id),
        }
        self._apply_component_field_visibility()

    def _apply_component_template_to_form(self, template_name: str):
        template = get_component_template_config(template_name)
        template_category = template.get("category", "component")
        self.component_category_combo.blockSignals(True)
        self.component_category_combo.setCurrentText(template_category)
        self.component_category_combo.blockSignals(False)
        self.rebuild_component_forms()

        temp_component = EditableComponent(
            object_id=self.component_id_edit.text().strip(),
            name=self.component_id_edit.text().strip(),
            category=template_category,
            template_name=template_name,
            description="",
        )
        temp_component.static_attributes = copy.deepcopy(template.get("static_attributes", {}))
        temp_component.state_variables = copy.deepcopy(template.get("state_variables", {}))
        temp_component.capabilities = copy.deepcopy(template.get("capabilities", {}))
        temp_component.relations = copy.deepcopy(template.get("relations", {}))
        parent_object_id = self.current_object_id()
        if parent_object_id:
            temp_component.relations["part_of"] = parent_object_id
            if template_name == "button":
                temp_component.relations.setdefault("controls", parent_object_id)
        self.fill_dynamic_section(self._component_field_inputs, temp_component)

    def _populate_section_form(self, form_layout: QFormLayout, fields: dict) -> dict[str, QWidget]:
        widgets: dict[str, QWidget] = {}
        for field_name, config in fields.items():
            widget = self._create_field_widget(config)
            widgets[field_name] = widget
            form_layout.addRow(field_name, widget)
        return widgets

    def _populate_relation_form(self, form_layout: QFormLayout, relations: list[str], parent_object_id: str) -> dict[str, QWidget]:
        widgets: dict[str, QWidget] = {}
        for relation_name in relations:
            combo = QComboBox()
            combo.addItem("")
            for target in self.available_relation_targets(parent_object_id, relation_name):
                combo.addItem(target)
            widgets[relation_name] = combo
            form_layout.addRow(relation_name, combo)
        return widgets

    def _create_field_widget(self, config: dict) -> QWidget:
        field_type = config.get("type")
        if field_type == "bool":
            widget = QCheckBox()
            if "default" in config:
                widget.setChecked(bool(config["default"]))
            return widget
        if field_type == "enum":
            widget = QComboBox()
            if config.get("optional"):
                widget.addItem("")
            widget.addItems([str(option) for option in config.get("options", [])])
            if "default" in config:
                widget.setCurrentText(str(config["default"]))
            return widget
        widget = QLineEdit()
        if "default" in config:
            widget.setText(str(config["default"]))
        return widget

    def fill_dynamic_section(self, inputs: dict[str, dict[str, QWidget]], item: EditableObject | EditableComponent):
        for section_name, values in [
            ("static_attributes", item.static_attributes),
            ("state_variables", item.state_variables),
            ("capabilities", item.capabilities),
            ("relations", item.relations),
        ]:
            for field_name, widget in inputs.get(section_name, {}).items():
                value = values.get(field_name, "")
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText("" if value is None else str(value))
                else:
                    widget.setText("" if value is None else str(value))
        if isinstance(item, EditableComponent):
            self._apply_component_field_visibility()

    def available_relation_targets(self, parent_object_id: str = "", relation_name: str = "") -> list[str]:
        room = self.editable_scene.get_room(self.current_room_id())
        if room is None:
            return []
        if relation_name == "in_room":
            return sorted(candidate.room_id for candidate in self.editable_scene.rooms)
        targets = set()
        for obj in room.objects:
            if self._relation_target_allowed(obj, relation_name):
                targets.add(obj.object_id)
            for component in obj.components:
                if self._relation_target_allowed(component, relation_name):
                    targets.add(component.object_id)
        if parent_object_id and relation_name == "part_of":
            targets.add(parent_object_id)
        return sorted(targets)

    def _relation_target_allowed(self, item: EditableObject | EditableComponent, relation_name: str) -> bool:
        if relation_name == "part_of":
            return True
        if relation_name == "controls":
            return self._is_controllable_target(item)
        if relation_name in {"inside", "contains"}:
            return self._is_container_target(item)
        if relation_name == "on":
            return self._is_support_target(item)
        return True

    def _is_container_target(self, item: EditableObject | EditableComponent) -> bool:
        static_attributes = getattr(item, "static_attributes", {})
        capabilities = getattr(item, "capabilities", {})
        return static_attributes.get("is_container") is True or capabilities.get("receptacle") is True

    def _is_support_target(self, item: EditableObject | EditableComponent) -> bool:
        static_attributes = getattr(item, "static_attributes", {})
        capabilities = getattr(item, "capabilities", {})
        return static_attributes.get("is_support") is True or capabilities.get("support_surface") is True

    def _is_controllable_target(self, item: EditableObject | EditableComponent) -> bool:
        capabilities = getattr(item, "capabilities", {})
        category = getattr(item, "category", "")
        if category in {"appliance", "component"}:
            return True
        return any(
            capabilities.get(capability_name) is True
            for capability_name in ("toggleable", "programmable", "openable")
        )

    def _build_object_from_form(self, apply_template: bool = False) -> EditableObject:
        object_id = self.object_id_edit.text().strip()
        name = object_id
        category = self.object_category_combo.currentText()
        object_form = self.object_form_combo.currentText()
        obj = create_object_from_template(
            object_id=object_id,
            name=name,
            category=category,
            object_form=object_form,
            template_name="custom",
            description="",
        )
        obj.template_name = "custom"
        obj.object_group = self.object_group_combo.currentText()
        obj.description = ""
        self._apply_dynamic_values(obj, self._object_field_inputs)
        if obj.object_form == "simple object":
            obj.components = []
        for component in obj.components:
            component.relations["part_of"] = obj.object_id
            if component.object_id == "start_button" and obj.template_name == "washing_machine_template":
                component.relations["controls"] = obj.object_id
        return obj

    def _build_component_from_form(self, parent_object_id: str) -> EditableComponent:
        component = EditableComponent(
            object_id=self.component_id_edit.text().strip(),
            name=self.component_id_edit.text().strip(),
            category=self.component_category_combo.currentText(),
            template_name=self.component_template_combo.currentText() or "custom",
            description="",
        )
        self._apply_dynamic_values(component, self._component_field_inputs)
        component.relations["part_of"] = parent_object_id
        if component.object_id == "start_button":
            component.relations.setdefault("controls", parent_object_id)
        return component

    def _apply_dynamic_values(self, item: EditableObject | EditableComponent, inputs: dict[str, dict[str, QWidget]]):
        for section_name in ("static_attributes", "state_variables", "capabilities", "relations"):
            values = {}
            for field_name, widget in inputs.get(section_name, {}).items():
                value = self._read_widget_value(widget)
                if value in ("", None):
                    continue
                values[field_name] = value
            setattr(item, section_name, values)

    def _read_widget_value(self, widget: QWidget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return self._coerce_value(widget.currentText())
        if isinstance(widget, QLineEdit):
            return self._coerce_value(widget.text().strip())
        return None

    def preview_export(self):
        self._apply_scene_form()
        try:
            scene_dict = build_scene_dict(copy.deepcopy(self.editable_scene))
        except Exception as exc:
            self.preview_text.setPlainText(f"Preview failed:\n{exc}")
            return
        self.graph_canvas.render_scene_graph(scene_dict, self.current_object_id())
        self.preview_text.setPlainText(json.dumps(scene_dict, indent=2, ensure_ascii=False))
        validate_scene_dict(scene_dict)

    def validate_scene(self):
        self._apply_scene_form()
        scene_dict = build_scene_dict(copy.deepcopy(self.editable_scene))
        errors = validate_scene_dict(scene_dict)
        warnings = check_scene_dict(scene_dict)
        if errors or warnings:
            details = []
            if errors:
                details.append("Errors:")
                details.extend(f"- {error}" for error in errors[:12])
            if warnings:
                if details:
                    details.append("")
                details.append("Warnings:")
                details.extend(f"- {warning}" for warning in warnings[:12])
            if len(errors) > 12 or len(warnings) > 12:
                details.append("")
                details.append("More issues exist; save flow will show the full list.")
            self._warn("\n".join(details))
        else:
            QMessageBox.information(self, "Validation", "Scene validation passed.")

    def save_scene(self):
        self.preview_export()
        scene_dict = build_scene_dict(copy.deepcopy(self.editable_scene))
        errors = validate_scene_dict(scene_dict)
        warnings = check_scene_dict(scene_dict)
        if errors or warnings:
            if not self._confirm_save_issues(errors, warnings):
                return
        output_path = export_scene_python(scene_dict, GENERATED_DIR)
        self.scene_saved.emit(scene_dict["name"], scene_dict)
        QMessageBox.information(self, "Scene Saved", f"Saved scene to:\n{output_path}")

    def on_graph_node_clicked(self, node_id: str, node_info: dict):
        kind = node_info.get("kind")
        if kind == "macro_zone":
            self.select_zone(node_id)
            self._focus_editor_widget(self.zone_id_edit)
            return
        if kind == "room":
            self.select_zone(node_info.get("floor", self.current_zone_id()))
            self.select_room(node_info.get("room_id", node_id))
            self._focus_editor_widget(self.room_id_edit)
            return
        if kind == "object":
            room_id = node_info.get("room_id")
            if room_id:
                room = self.editable_scene.get_room(room_id)
                if room:
                    self.select_zone(room.floor)
            self.select_room(room_id)
            self.select_object(node_info.get("object_id", node_id))
            self._focus_editor_widget(self.object_id_edit)
            return
        if kind == "component":
            room_id = node_info.get("room_id")
            parent_object_id = node_info.get("parent_object_id")
            if room_id:
                room = self.editable_scene.get_room(room_id)
                if room:
                    self.select_zone(room.floor)
            self.select_room(room_id)
            if parent_object_id:
                self.select_object(parent_object_id)
            self.select_component(node_info.get("component_id", node_id))
            self._focus_editor_widget(self.component_id_edit)

    def on_graph_edge_clicked(self, edge_info: dict):
        relation_name = edge_info.get("relation_name")
        if not relation_name:
            return
        room_id = edge_info.get("room_id")
        if room_id:
            room = self.editable_scene.get_room(room_id)
            if room:
                self.select_zone(room.floor)
        self.select_room(room_id)
        source_kind = edge_info.get("source_kind")
        source_id = edge_info.get("source_id")
        if source_kind == "object":
            self.select_object(source_id)
            relation_widget = self._object_field_inputs.get("relations", {}).get(relation_name)
            if relation_widget is not None:
                self._focus_editor_widget(relation_widget)
        elif source_kind == "component":
            parent_object_id = edge_info.get("parent_object_id")
            if parent_object_id:
                self.select_object(parent_object_id)
            self.select_component(source_id)
            relation_widget = self._component_field_inputs.get("relations", {}).get(relation_name)
            if relation_widget is not None:
                self._focus_editor_widget(relation_widget)

    def _apply_scene_form(self):
        self.editable_scene.name = self.scene_name_edit.text().strip() or "custom_scene"
        self.editable_scene.elevator_enabled = self.elevator_checkbox.isChecked()
        self.editable_scene.ensure_macro_zones(self.zone_count_spin.value())
        zone = self.editable_scene.get_zone(self.current_zone_id())
        if zone is not None:
            zone.category = self.zone_category_combo.currentText()
            zone.has_elevator_access = self.zone_elevator_checkbox.isChecked()

    def populate_neighbor_options(self, selected_neighbors: list[str] | None = None):
        selected_neighbors = selected_neighbors or []
        current_room = self.current_room_id()
        self.room_neighbors_list.clear()
        for room in self.editable_scene.rooms:
            if room.room_id == current_room:
                continue
            item = QListWidgetItem(room.room_id)
            self.room_neighbors_list.addItem(item)
            if room.room_id in selected_neighbors:
                item.setSelected(True)

    def selected_neighbors(self) -> list[str]:
        return [item.text() for item in self.room_neighbors_list.selectedItems()]

    def _ensure_bidirectional_neighbors(self, room: EditableRoom):
        selected = set(room.neighbor)
        for other_room in self.editable_scene.rooms:
            if other_room.room_id == room.room_id:
                continue
            if other_room.room_id in selected and room.room_id not in other_room.neighbor:
                other_room.neighbor.append(room.room_id)
            if other_room.room_id not in selected and room.room_id in other_room.neighbor:
                other_room.neighbor.remove(room.room_id)

    def _clear_form(self, form_layout: QFormLayout):
        while form_layout.rowCount():
            form_layout.removeRow(0)

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-weight: 700;")
        return label

    def _style_group_title(self, group_box: QGroupBox):
        group_box.setStyleSheet("QGroupBox { font-weight: 700; }")

    def _configure_compact_form(self, form_layout: QFormLayout):
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def update_component_editor_visibility(self):
        obj = self.current_object()
        visible = (obj is not None and obj.object_form == "composite object") or self.object_form_combo.currentText() == "composite object"
        self.component_group.setVisible(visible)
        self.component_list.setVisible(visible)
        self.component_list_label.setVisible(visible)

    def _apply_component_field_visibility(self):
        category = self.component_category_combo.currentText()
        if not category:
            return
        allowed_by_category = COMPONENT_CATEGORY_FIELD_RULES.get(category)
        role = ""
        role_widget = self._component_field_inputs.get("static_attributes", {}).get("component_role")
        if isinstance(role_widget, QComboBox):
            role = role_widget.currentText()
            if not role_widget.property("role_visibility_bound"):
                role_widget.currentTextChanged.connect(self._on_component_role_changed)
                role_widget.setProperty("role_visibility_bound", True)
        allowed_by_role = COMPONENT_ROLE_FIELD_RULES.get(role, {})
        for section_name, field_map in self._component_field_inputs.items():
            for field_name, widget in field_map.items():
                visible = True
                if allowed_by_category is not None:
                    visible = field_name in allowed_by_category.get(section_name, set())
                if category == "component" and section_name in {"state_variables", "capabilities"}:
                    role_allowed = allowed_by_role.get(section_name)
                    if role_allowed is not None:
                        visible = field_name in role_allowed or field_name == "cleanliness"
                label = self._find_form_label_for_widget(widget)
                if label is not None:
                    label.setVisible(visible)
                widget.setVisible(visible)

    def _on_component_role_changed(self, _value: str):
        self._apply_component_field_visibility()

    def _find_form_label_for_widget(self, widget: QWidget) -> QWidget | None:
        for form_layout in (
            self.component_static_form,
            self.component_state_form,
            self.component_capability_form,
            self.component_relation_form,
        ):
            for row in range(form_layout.rowCount()):
                if form_layout.itemAt(row, QFormLayout.ItemRole.FieldRole) and form_layout.itemAt(row, QFormLayout.ItemRole.FieldRole).widget() is widget:
                    label_item = form_layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    return label_item.widget() if label_item else None
        return None

    def _warn(self, message: str):
        QMessageBox.warning(self, "Scene Editor", message)

    def _confirm_save_issues(self, errors: list[str], warnings: list[str]) -> bool:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Check Before Save")
        dialog.resize(880, 560)
        dialog.setMinimumSize(880, 560)
        summary = []
        if errors:
            summary.append(f"{len(errors)} validation error(s)")
        if warnings:
            summary.append(f"{len(warnings)} warning(s)")
        dialog.setText("Potential scene issues were found before saving.")
        dialog.setInformativeText(
            f"Detected {' and '.join(summary)}. You can continue saving, or go back and edit first."
        )
        detail_lines = []
        if errors:
            detail_lines.append("Errors:")
            detail_lines.extend(f"- {error}" for error in errors)
        if warnings:
            if detail_lines:
                detail_lines.append("")
            detail_lines.append("Warnings:")
            detail_lines.extend(f"- {warning}" for warning in warnings)
        dialog.setDetailedText("\n".join(detail_lines))
        continue_button = dialog.addButton("Continue Save", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Back to Edit", QMessageBox.ButtonRole.RejectRole)
        def _expand_detail_box():
            for button in dialog.buttons():
                if "details" in button.text().lower():
                    button.click()
                    break
            detail_box = dialog.findChild(QTextEdit)
            if detail_box is not None:
                detail_box.setMinimumSize(820, 340)
                detail_box.resize(820, 340)
                dialog.resize(900, 620)

        QTimer.singleShot(0, _expand_detail_box)
        dialog.exec()
        return dialog.clickedButton() is continue_button

    def _focus_editor_widget(self, widget: QWidget):
        if hasattr(self, "editor_scroll") and self.editor_scroll is not None:
            self.editor_scroll.ensureWidgetVisible(widget, 24, 24)
        if hasattr(widget, "setFocus"):
            widget.setFocus()

    def _make_button_compact(self, button: QPushButton, multiline: bool = False):
        button.setFixedHeight(42 if multiline else 28)
        button.setMinimumWidth(96)
        if multiline:
            button.setStyleSheet("QPushButton { padding: 2px 6px; font-size: 11px; text-align: center; }")
        else:
            button.setStyleSheet("QPushButton { padding: 2px 8px; font-size: 12px; }")

    def _coerce_value(self, value: str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value

    def select_zone(self, zone_id: str):
        for index in range(self.zone_list.count()):
            item = self.zone_list.item(index)
            if item.data(256) == zone_id:
                self.zone_list.setCurrentRow(index)
                break

    def select_room(self, room_id: str):
        for index in range(self.room_list.count()):
            item = self.room_list.item(index)
            if item.data(256) == room_id:
                self.room_list.setCurrentRow(index)
                break

    def select_object(self, object_id: str):
        for index in range(self.object_list.count()):
            item = self.object_list.item(index)
            if item.data(256) == object_id:
                self.object_list.setCurrentRow(index)
                break

    def select_component(self, component_id: str):
        for index in range(self.component_list.count()):
            item = self.component_list.item(index)
            if item.data(256) == component_id:
                self.component_list.setCurrentRow(index)
                break
