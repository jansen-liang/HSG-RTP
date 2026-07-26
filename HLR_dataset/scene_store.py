from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_DIRS = [
    ROOT / "HLR_dataset" / "dataset_output",
    ROOT / "HLR_dataset" / "data" / "scene_graphs",
]


def _ring(n: int, r: float, cx: float = 0.0, cy: float = 0.0) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [(cx, cy)]
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _room_label(node: dict) -> str:
    name = str(node.get("name") or "")
    node_id = str(node.get("id") or "")
    return name if name and name != node_id else node_id.replace("_", " ")


def _obj_label(node: dict) -> str:
    text = str(node.get("object_type") or node.get("name") or node.get("id") or "object")
    return text.replace("__", "_").replace("_", " ")


class SceneStore:
    def __init__(self) -> None:
        self.scenes: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        for directory in SCENE_DIRS:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                scene_id = str(raw.get("scene_name") or path.stem)
                if scene_id not in self.scenes:
                    self.scenes[scene_id] = {"path": path, "raw": raw}

    def list_scenes(self) -> list[dict]:
        items = []
        for scene_id in sorted(self.scenes):
            raw = self.scenes[scene_id]["raw"]
            nodes = raw.get("nodes", [])
            items.append(
                {
                    "id": scene_id,
                    "name": scene_id,
                    "floor_count": sum(1 for n in nodes if n.get("type") == "floor"),
                    "room_count": sum(1 for n in nodes if n.get("type") == "room"),
                    "object_count": sum(1 for n in nodes if n.get("type") == "object"),
                    "path": str(self.scenes[scene_id]["path"].relative_to(ROOT)),
                }
            )
        return items

    def get_scene(self, scene_id: str) -> dict | None:
        item = self.scenes.get(scene_id)
        if not item:
            return None
        raw = item["raw"]
        nodes = {str(n["id"]): n for n in raw.get("nodes", []) if isinstance(n, dict) and n.get("id")}
        edges = [e for e in raw.get("edges", []) if isinstance(e, dict)]
        agent = raw.get("agent") or {}
        agent_room = str(agent.get("current_room") or "")

        floors = sorted(
            [
                {
                    "id": str(n["id"]),
                    "name": str(n.get("name") or n["id"]),
                    "floor_number": n.get("floor_number"),
                    "room_count": len(n.get("rooms") or []),
                }
                for n in nodes.values()
                if n.get("type") == "floor"
            ],
            key=lambda x: (x["floor_number"] if isinstance(x["floor_number"], int) else 9999, x["id"]),
        )

        def build_floor(floor_id: str) -> dict:
            rooms = [n for n in nodes.values() if n.get("type") == "room" and str(n.get("floor_id") or "") == floor_id]
            room_pos = _ring(len(rooms), 230.0)
            visible = set()
            floor_nodes, floor_edges = [], []
            room_ids = {str(r["id"]) for r in rooms}
            for room, (x, y) in zip(rooms, room_pos, strict=False):
                room_id = str(room["id"])
                visible.add(room_id)
                floor_nodes.append({"id": room_id, "label": _room_label(room), "kind": "room", "x": x, "y": y, "is_agent_room": room_id == agent_room, "meta": {"contained_objects": len(room.get("contained_objects") or []), "neighbors": len(room.get("neighbours") or [])}})
                obj_ids = [oid for oid in room.get("contained_objects") or [] if oid in nodes and nodes[oid].get("type") == "object"]
                for obj, (ox, oy) in zip([nodes[oid] for oid in obj_ids], _ring(len(obj_ids), 92.0, x, y), strict=False):
                    obj_id = str(obj["id"])
                    visible.add(obj_id)
                    floor_nodes.append({"id": obj_id, "label": _obj_label(obj), "kind": "object", "x": ox, "y": oy, "room_id": room_id, "meta": {"object_type": obj.get("object_type"), "states": obj.get("states") or {}, "affordance_count": len(obj.get("affordances") or [])}})
                    floor_edges.append({"source": room_id, "target": obj_id, "kind": "contains"})

            seen = set()
            for e in edges:
                s, t, r = str(e.get("source_id") or ""), str(e.get("target_id") or ""), str(e.get("relation") or "").lower()
                if r in {"neighbour", "neighbor"} and s in room_ids and t in room_ids:
                    key = tuple(sorted((s, t)))
                    if key not in seen:
                        seen.add(key)
                        floor_edges.append({"source": s, "target": t, "kind": "neighbor"})
                elif r in {"ontop", "next_to"} and s in visible and t in visible:
                    key = (s, t, r)
                    if key not in seen:
                        seen.add(key)
                        floor_edges.append({"source": s, "target": t, "kind": r})
            return {"floor_id": floor_id, "floor_name": next((f["name"] for f in floors if f["id"] == floor_id), floor_id), "node_count": len(floor_nodes), "edge_count": len(floor_edges), "nodes": floor_nodes, "edges": floor_edges}

        floor_views = {f["id"]: build_floor(f["id"]) for f in floors}
        agent_floor = str(nodes.get(agent_room, {}).get("floor_id") or floors[0]["id"] if floors else "")
        return {"scene": self.list_scenes()[[x["id"] for x in self.list_scenes()].index(scene_id)], "agent": {"id": agent.get("id"), "current_room": agent_room, "current_floor": agent_floor, "inventory": agent.get("inventory", [])}, "floors": floors, "current_floor": agent_floor, "floor_views": floor_views}
