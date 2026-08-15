#!/usr/bin/env python3
"""Convert HSG-RTP streaming traces to GRID's raw dataset layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.action_parser import LocalAction, parse_local_action
from pipeline.utils.graph_utils import get_local_view


HSG_ACTIONS = ("goto", "pick", "place", "press", "scan", "wait", "finish")
WAIT_CONDITIONS = ("elevator_down_clear", "elevator_up_clear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-nodes", type=int, default=87)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _state_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {value[key]}" for key in sorted(value))
    if value in (None, ""):
        return ""
    return str(value)


def _node(
    node_id: int,
    node_type: str,
    label: str,
    *,
    operation: list[str] | None = None,
    state: Any = "",
) -> dict[str, Any]:
    descriptive_label = label.replace("_", " ")
    state_description = _state_text(state)
    if state_description:
        descriptive_label = f"{descriptive_label} state {state_description}"
    return {
        "id": node_id,
        "type": node_type,
        "attributes": {
            "color": "",
            "label": descriptive_label,
            "entity_id": label,
            "position": [],
            "operation": operation or [],
            "state": state_description,
        },
    }


def _edge(edge_id: int, edge_type: str, source: int, target: int) -> dict[str, Any]:
    return {
        "id": edge_id,
        "type": edge_type,
        "source": source,
        "target": target,
        "attributes": {"label": ""},
    }


def _target_name(action: LocalAction) -> str:
    if action.name == "place":
        return action.arguments[1]
    return action.arguments[0]


def build_scene_graph(
    local_view: dict[str, Any], action: LocalAction | None = None
) -> tuple[dict[str, Any], int | None]:
    room = local_view["room"]
    current_room = local_view["current_room"]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_ids: dict[str, int] = {}

    def add_node(
        key: str,
        node_type: str,
        *,
        operation: list[str] | None = None,
        state: Any = "",
    ) -> int:
        if key in node_ids:
            return node_ids[key]
        node_id = len(nodes)
        node_ids[key] = node_id
        nodes.append(_node(node_id, node_type, key, operation=operation, state=state))
        return node_id

    def add_edge(edge_type: str, source: int, target: int) -> None:
        edges.append(_edge(len(edges), edge_type, source, target))

    floor_name = room.get("floor", "unknown_floor")
    floor_id = add_node(floor_name, "floor")
    room_id = add_node(current_room, "room")
    add_edge("in", room_id, floor_id)

    for neighbor in room.get("neighbor", []):
        neighbor_id = add_node(neighbor, "room", operation=["goto", "scan"])
        add_edge("neighbor", neighbor_id, room_id)

    for object_name, attributes in room.get("large_objects", {}).items():
        operations = list(attributes.get("affordance", []))
        if attributes.get("is_container") and "place" not in operations:
            operations.append("place")
        object_id = add_node(
            object_name,
            "large_object",
            operation=operations,
            state=attributes.get("state", ""),
        )
        add_edge("in", object_id, room_id)

    for object_name, attributes in room.get("small_objects", {}).items():
        object_id = add_node(
            object_name,
            "small_object",
            operation=list(attributes.get("affordance", [])),
            state=attributes.get("state", ""),
        )
        relation = attributes.get("relation", {})
        parent_name = next(iter(relation.values()), current_room)
        parent_id = node_ids.get(parent_name, room_id)
        edge_type = next(iter(relation.keys()), "in")
        add_edge(edge_type, object_id, parent_id)

    for condition in WAIT_CONDITIONS:
        condition_id = add_node(condition, "small_object", operation=["wait"])
        add_edge("condition", condition_id, room_id)

    if action is None:
        return {"version": "1.0", "nodes": nodes, "edges": edges}, None

    target_name = _target_name(action)
    if target_name == "floor":
        target_id = floor_id
    else:
        target_id = node_ids.get(target_name, -1)

    if target_id < 0:
        raise ValueError(
            f"Target {target_name!r} for {action.canonical!r} is absent from "
            f"local view in room {current_room!r}"
        )
    return {"version": "1.0", "nodes": nodes, "edges": edges}, target_id


def build_robot_graph(local_view: dict[str, Any]) -> dict[str, Any]:
    agent = local_view["agent"]
    agent_state = str(agent.get("state", "hand-free"))
    held_object = agent_state[len("holding-") :] if agent_state.startswith("holding-") else "empty_hand"
    current_room = local_view["current_room"]
    floor_name = local_view["room"].get("floor", "unknown_floor")
    nodes = [
        _node(0, "robot", "robot", state=agent_state),
        _node(1, "small_object", held_object, operation=["place"] if held_object != "empty_hand" else []),
        _node(2, "room", current_room),
        _node(3, "floor", floor_name),
    ]
    edges = [
        _edge(0, "grasp" if held_object != "empty_hand" else "near", 0, 1),
        _edge(1, "in", 0, 2),
        _edge(2, "in", 2, 3),
    ]
    return {"version": "1.0", "nodes": nodes, "edges": edges}


def _low_command(action: LocalAction, target_id: int) -> str:
    return f"{action.name} {_target_name(action)} {target_id}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_record(record: dict[str, Any], scene_id: int, output_dir: Path) -> dict[str, int]:
    local_samples = [sample for sample in record.get("streaming_samples", []) if sample.get("mode") == "local"]
    if not local_samples:
        raise ValueError(f"Task {scene_id} has no local streaming samples")

    graph_dir = output_dir / f"scene.{scene_id}.graphs"
    graph_dir.mkdir(parents=True)
    low_commands: list[str] = []
    action_counts: Counter[str] = Counter()
    max_nodes = 0

    for graph_id, sample in enumerate(local_samples):
        action = parse_local_action(sample["target"])
        local_view = sample["scene_graph"]
        scene_graph, target_id = build_scene_graph(local_view, action)
        assert target_id is not None
        robot_graph = build_robot_graph(local_view)
        max_nodes = max(max_nodes, len(scene_graph["nodes"]))
        action_counts[action.name] += 1
        low_commands.append(_low_command(action, target_id))
        stem = f"scene.{scene_id}.instr.0"
        _write_json(graph_dir / f"{stem}.sg.{graph_id}.json", scene_graph)
        _write_json(graph_dir / f"{stem}.rg.{graph_id}.json", robot_graph)

    final_state = record.get("execution_summary", {}).get("final_state")
    if not final_state:
        raise ValueError(f"Task {scene_id} has no final state for the finish sample")
    final_room = final_state["agent"]["position"]
    final_view = get_local_view(final_state, final_room)
    finish_id = len(local_samples)
    finish_graph, _ = build_scene_graph(final_view)
    max_nodes = max(max_nodes, len(finish_graph["nodes"]))
    low_commands.append("finish floor 0")
    action_counts["finish"] += 1
    stem = f"scene.{scene_id}.instr.0"
    _write_json(graph_dir / f"{stem}.sg.{finish_id}.json", finish_graph)
    _write_json(graph_dir / f"{stem}.rg.{finish_id}.json", build_robot_graph(final_view))

    command = {
        "commands": [
            {
                "id": 0,
                "type": record.get("task_info", {}).get("type", "hsg_rtp"),
                "low": low_commands,
                "high": record["instruction"],
            }
        ]
    }
    _write_json(output_dir / f"scene.{scene_id}.instr.json", command)
    return {"samples": len(low_commands), "max_nodes": max_nodes, **action_counts}


def convert_dataset(input_path: Path, output_dir: Path, max_nodes: int, overwrite: bool) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    totals: Counter[str] = Counter()
    task_count = 0
    observed_max_nodes = 0
    with input_path.open("r", encoding="utf-8") as source:
        for scene_id, line in enumerate(source):
            if not line.strip():
                continue
            stats = convert_record(json.loads(line), scene_id, output_dir)
            task_count += 1
            observed_max_nodes = max(observed_max_nodes, stats.pop("max_nodes"))
            totals.update(stats)

    if observed_max_nodes > max_nodes:
        raise ValueError(
            f"Converted graph has {observed_max_nodes} nodes, above GRID limit {max_nodes}"
        )
    manifest = {
        "source": str(input_path.resolve()),
        "tasks": task_count,
        "samples": totals.pop("samples", 0),
        "actions": {action: totals.get(action, 0) for action in HSG_ACTIONS},
        "max_scene_nodes": observed_max_nodes,
        "grid_max_nodes": max_nodes,
        "action_order": list(HSG_ACTIONS),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = convert_dataset(args.input, args.output, args.max_nodes, args.overwrite)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
