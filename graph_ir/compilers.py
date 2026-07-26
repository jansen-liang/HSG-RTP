from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from .graph import CanonicalEdge, CanonicalGraph, CanonicalNode
from .ids import StableIdRegistry
from .ontology import normalize_relation, relation_category


def detect_schema(payload: dict[str, Any]) -> str:
    if {"nodes", "edges"} <= set(payload.keys()):
        return "oop_scene"

    if {"name", "rooms", "macro_zones"} <= set(payload.keys()) or {"scene_name", "rooms"} <= set(payload.keys()):
        for room in payload.get("rooms", {}).values():
            for bucket in ("large_objects", "small_objects"):
                for object_payload in room.get(bucket, {}).values():
                    if {"subtype", "static_attributes", "state_variables", "relations"} & set(object_payload.keys()):
                        return "editor_scene"
        return "legacy_scene"

    raise ValueError(f"Unable to detect scene schema from keys: {sorted(payload.keys())}")


def compile_to_canonical(
    payload: dict[str, Any],
    *,
    preserve_raw_ids: bool = False,
) -> CanonicalGraph:
    schema = detect_schema(payload)
    if schema == "oop_scene":
        return compile_oop_scene(payload, preserve_raw_ids=preserve_raw_ids)
    if schema in {"legacy_scene", "editor_scene"}:
        return compile_room_scene(payload, schema=schema, preserve_raw_ids=preserve_raw_ids)
    raise ValueError(f"Unsupported schema: {schema}")


def compile_room_scene(
    payload: dict[str, Any],
    *,
    schema: str,
    preserve_raw_ids: bool = False,
) -> CanonicalGraph:
    scene_name = payload.get("name") or payload.get("scene_name") or "scene"
    graph = CanonicalGraph(name=scene_name, metadata={"source_schema": schema})
    registry = StableIdRegistry(scene_name)

    by_room_raw: dict[tuple[str, str], str] = {}
    by_raw_global: dict[str, list[str]] = defaultdict(list)

    def make_id(node_type: str, raw_id: str, scope: str | None = None) -> str:
        if preserve_raw_ids:
            return str(raw_id)
        return registry.get(node_type=node_type, raw_id=raw_id, scope=scope)

    def register(raw_id: str, canonical_id: str, room_raw_id: str | None = None) -> None:
        by_raw_global[raw_id].append(canonical_id)
        if room_raw_id is not None:
            by_room_raw[(room_raw_id, raw_id)] = canonical_id

    macro_zones = payload.get("macro_zones", {})
    rooms = payload.get("rooms", {})

    for zone_raw_id, zone_payload in macro_zones.items():
        zone_id = make_id("zone", zone_raw_id)
        graph.add_node(
            CanonicalNode(
                id=zone_id,
                type="zone",
                subtype=zone_payload.get("category", "macro_zone"),
                attrs={
                    "raw_id": zone_raw_id,
                    "description": zone_payload.get("description", ""),
                    "room_refs": list(zone_payload.get("rooms", [])),
                },
            )
        )
        register(zone_raw_id, zone_id)

    for room_raw_id, room_payload in rooms.items():
        room_id = make_id("room", room_raw_id)
        room_subtype = room_payload.get("type") or _infer_room_subtype(room_raw_id)
        graph.add_node(
            CanonicalNode(
                id=room_id,
                type="room",
                subtype=room_subtype,
                attrs={
                    "raw_id": room_raw_id,
                    "floor_ref": room_payload.get("floor"),
                    "neighbors": list(room_payload.get("neighbor", [])),
                    "source_schema": schema,
                },
                states=deepcopy(room_payload.get("state_variables", room_payload.get("state", {}))),
            )
        )
        register(room_raw_id, room_id)

    for zone_raw_id, zone_payload in macro_zones.items():
        zone_id = _resolve_raw_ref(zone_raw_id, by_raw_global)
        for room_raw_id in zone_payload.get("rooms", []):
            room_id = _resolve_raw_ref(room_raw_id, by_raw_global)
            if zone_id and room_id:
                graph.add_edge(CanonicalEdge(source=zone_id, target=room_id, relation="contains", category="hierarchical"))

    pending_relations: list[tuple[str, str | None, str, Any]] = []
    room_object_map: dict[str, dict[str, str]] = defaultdict(dict)

    for room_raw_id, room_payload in rooms.items():
        room_id = _resolve_raw_ref(room_raw_id, by_raw_global)
        for bucket_name in ("large_objects", "small_objects"):
            for raw_obj_id, object_payload in room_payload.get(bucket_name, {}).items():
                _compile_object_payload(
                    graph=graph,
                    object_payload=object_payload,
                    raw_obj_id=raw_obj_id,
                    room_raw_id=room_raw_id,
                    room_id=room_id,
                    bucket_name=bucket_name,
                    schema=schema,
                    registry=registry,
                    preserve_raw_ids=preserve_raw_ids,
                    register=register,
                    pending_relations=pending_relations,
                    room_object_map=room_object_map,
                )

    for room_raw_id, room_payload in rooms.items():
        room_id = _resolve_raw_ref(room_raw_id, by_raw_global)
        for neighbor_raw_id in room_payload.get("neighbor", []):
            neighbor_id = _resolve_target_ref(
                target_ref=neighbor_raw_id,
                room_raw_id=room_raw_id,
                by_room_raw=by_room_raw,
                by_raw_global=by_raw_global,
            )
            if room_id and neighbor_id:
                graph.add_edge(CanonicalEdge(source=room_id, target=neighbor_id, relation="neighbor", category="spatial"))

    for source_id, room_raw_id, raw_relation, raw_target in pending_relations:
        relation = normalize_relation(raw_relation)
        targets = raw_target if isinstance(raw_target, list) else [raw_target]
        for target_ref in targets:
            if target_ref in (None, "", False):
                continue
            target_id = _resolve_target_ref(
                target_ref=target_ref,
                room_raw_id=room_raw_id,
                by_room_raw=by_room_raw,
                by_raw_global=by_raw_global,
            )
            if target_id:
                graph.add_edge(
                    CanonicalEdge(
                        source=source_id,
                        target=target_id,
                        relation=relation,
                        category=relation_category(relation),
                    )
                )

    agent_payload = payload.get("agent", {})
    if agent_payload:
        agent_raw_id = agent_payload.get("id", "agent")
        agent_id = make_id("agent", agent_raw_id)
        agent_state = agent_payload.get("state", {})
        if isinstance(agent_state, str):
            agent_state = {"mode": agent_state}
        graph.add_node(
            CanonicalNode(
                id=agent_id,
                type="agent",
                subtype=agent_payload.get("type", "robot"),
                attrs={
                    "raw_id": agent_raw_id,
                    "battery": agent_payload.get("battery"),
                    "pressed_buttons": deepcopy(agent_payload.get("pressed_buttons", [])),
                    "scan_history": deepcopy(agent_payload.get("scan_history", [])),
                    "inventory": list(agent_payload.get("inventory", {}).keys()),
                },
                states=deepcopy(agent_state),
            )
        )
        register(agent_raw_id, agent_id)
        current_room_raw = agent_payload.get("position")
        current_room_id = _resolve_target_ref(
            target_ref=current_room_raw,
            room_raw_id=current_room_raw,
            by_room_raw=by_room_raw,
            by_raw_global=by_raw_global,
        )
        if current_room_id:
            graph.add_edge(CanonicalEdge(source=agent_id, target=current_room_id, relation="located_in", category="agentic"))

        for raw_obj_id in agent_payload.get("inventory", {}).keys():
            object_id = _resolve_target_ref(
                target_ref=raw_obj_id,
                room_raw_id=current_room_raw,
                by_room_raw=by_room_raw,
                by_raw_global=by_raw_global,
            )
            if object_id:
                graph.add_edge(CanonicalEdge(source=agent_id, target=object_id, relation="carries", category="agentic"))

    return graph


def compile_oop_scene(payload: dict[str, Any], *, preserve_raw_ids: bool = False) -> CanonicalGraph:
    scene_name = payload.get("scene_name") or payload.get("name") or "scene"
    graph = CanonicalGraph(name=scene_name, metadata={"source_schema": "oop_scene"})
    registry = StableIdRegistry(scene_name)
    node_id_map: dict[str, str] = {}

    def make_id(node_type: str, raw_id: str) -> str:
        if preserve_raw_ids:
            return str(raw_id)
        return registry.get(node_type=node_type, raw_id=raw_id)

    for node_payload in payload.get("nodes", []):
        raw_type = str(node_payload.get("type", "object")).lower()
        node_type = _map_oop_node_type(raw_type)
        raw_id = node_payload["id"]
        node_id = make_id(node_type, raw_id)
        node_id_map[raw_id] = node_id
        subtype = node_payload.get("object_type") or raw_type
        attrs = {k: deepcopy(v) for k, v in node_payload.items() if k not in {"id", "type", "name", "object_type", "states"}}
        attrs["raw_id"] = raw_id
        graph.add_node(
            CanonicalNode(
                id=node_id,
                type=node_type,
                subtype=subtype,
                attrs=attrs,
                states=deepcopy(node_payload.get("states", {})),
            )
        )

    agent_payload = payload.get("agent")
    if isinstance(agent_payload, dict) and agent_payload.get("id") and agent_payload["id"] not in node_id_map:
        raw_id = agent_payload["id"]
        agent_id = make_id("agent", raw_id)
        node_id_map[raw_id] = agent_id
        graph.add_node(
            CanonicalNode(
                id=agent_id,
                type="agent",
                subtype=agent_payload.get("type", "robot"),
                attrs={"raw_id": raw_id},
                states=deepcopy(agent_payload.get("states", {})),
            )
        )

    if isinstance(agent_payload, dict) and agent_payload.get("id") in node_id_map:
        agent_id = node_id_map[agent_payload["id"]]
        current_room_raw = agent_payload.get("current_room")
        current_room_id = node_id_map.get(current_room_raw)
        if current_room_id:
            graph.add_edge(CanonicalEdge(source=agent_id, target=current_room_id, relation="located_in", category="agentic"))

    for edge_payload in payload.get("edges", []):
        raw_source = edge_payload.get("source_id")
        raw_target = edge_payload.get("target_id")
        source = node_id_map.get(raw_source)
        target = node_id_map.get(raw_target)
        if not source or not target:
            continue
        relation = normalize_relation(edge_payload.get("relation", "connected_to"))
        edge_type = str(edge_payload.get("edge_type", "")).lower()
        attrs = {k: deepcopy(v) for k, v in edge_payload.items() if k not in {"source_id", "target_id", "relation", "category", "edge_type"}}

        # Canonicalize historical edge directions from the OOP scene graph.
        if edge_type == "room_floor_edge" and relation == "contains":
            graph.add_edge(CanonicalEdge(source=target, target=source, relation="contains", category="hierarchical", attrs=attrs))
            continue

        if edge_type == "object_room_edge" and relation == "contains":
            graph.add_edge(CanonicalEdge(source=target, target=source, relation="contains", category="hierarchical", attrs=attrs))
            continue

        category = edge_payload.get("category", relation_category(relation))
        graph.add_edge(CanonicalEdge(source=source, target=target, relation=relation, category=category, attrs=attrs))
        if relation in {"neighbor", "connected_to"}:
            graph.add_edge(CanonicalEdge(source=target, target=source, relation=relation, category=category, attrs=deepcopy(attrs)))

    return graph


def _compile_object_payload(
    *,
    graph: CanonicalGraph,
    object_payload: dict[str, Any],
    raw_obj_id: str,
    room_raw_id: str,
    room_id: str | None,
    bucket_name: str,
    schema: str,
    registry: StableIdRegistry,
    preserve_raw_ids: bool,
    register,
    pending_relations: list[tuple[str, str | None, str, Any]],
    room_object_map: dict[str, dict[str, str]],
    parent_canonical_id: str | None = None,
) -> str:
    node_type = "component" if parent_canonical_id else "object"
    canonical_id = raw_obj_id if preserve_raw_ids else registry.get(node_type=node_type, raw_id=raw_obj_id, scope=room_raw_id)
    subtype = (
        object_payload.get("subtype")
        or object_payload.get("template_subtype")
        or object_payload.get("object_type")
        or object_payload.get("type")
        or "object"
    )

    attrs = {
        "raw_id": raw_obj_id,
        "room_raw_id": room_raw_id,
        "bucket": bucket_name,
        "name": object_payload.get("name", raw_obj_id),
        "description": object_payload.get("description", ""),
        "source_schema": schema,
    }
    attrs.update(deepcopy(object_payload.get("static_attributes", object_payload.get("physical_property", object_payload.get("physical_properties", {})))))
    attrs["affordances"] = list(dict.fromkeys(_extract_affordances(object_payload)))
    attrs["capabilities"] = deepcopy(object_payload.get("capabilities", {}))
    attrs["methods"] = deepcopy(object_payload.get("methods", {}))
    attrs["is_container"] = bool(
        object_payload.get("is_container")
        or attrs.get("is_container")
        or attrs.get("receptacle")
        or attrs.get("support_surface")
    )
    if parent_canonical_id:
        attrs["parent_id"] = parent_canonical_id

    states = deepcopy(object_payload.get("state_variables", object_payload.get("state", object_payload.get("states", {}))))
    graph.add_node(
        CanonicalNode(
            id=canonical_id,
            type=node_type,
            subtype=str(subtype),
            attrs=attrs,
            states=states,
        )
    )
    register(raw_obj_id, canonical_id, room_raw_id)
    room_object_map[room_raw_id][raw_obj_id] = canonical_id

    if room_id and not parent_canonical_id:
        graph.add_edge(CanonicalEdge(source=room_id, target=canonical_id, relation="contains", category="hierarchical"))
    if parent_canonical_id:
        graph.add_edge(CanonicalEdge(source=canonical_id, target=parent_canonical_id, relation="part_of", category="hierarchical"))

    relations = deepcopy(object_payload.get("relations", object_payload.get("relation", {})))
    for relation_name, target in relations.items():
        pending_relations.append((canonical_id, room_raw_id, relation_name, target))

    for component_raw_id, component_payload in object_payload.get("components", {}).items():
        _compile_object_payload(
            graph=graph,
            object_payload=component_payload,
            raw_obj_id=component_raw_id,
            room_raw_id=room_raw_id,
            room_id=room_id,
            bucket_name="components",
            schema=schema,
            registry=registry,
            preserve_raw_ids=preserve_raw_ids,
            register=register,
            pending_relations=pending_relations,
            room_object_map=room_object_map,
            parent_canonical_id=canonical_id,
        )

    return canonical_id


def _extract_affordances(object_payload: dict[str, Any]) -> list[str]:
    affordances = []
    raw_affordance = object_payload.get("affordance", object_payload.get("affordances", []))
    if isinstance(raw_affordance, list):
        affordances.extend(str(item) for item in raw_affordance if item)
    capabilities = object_payload.get("capabilities", {})
    if isinstance(capabilities, dict):
        affordances.extend(str(name) for name, enabled in capabilities.items() if enabled)
    return affordances


def _map_oop_node_type(raw_type: str) -> str:
    mapping = {
        "agent": "agent",
        "floor": "floor",
        "macro": "zone",
        "mobile_tool": "transport",
        "object": "object",
        "room": "room",
        "transport": "transport",
    }
    return mapping.get(raw_type, "object")


def _resolve_raw_ref(raw_id: str, by_raw_global: dict[str, list[str]]) -> str | None:
    candidates = by_raw_global.get(raw_id, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_target_ref(
    *,
    target_ref: Any,
    room_raw_id: str | None,
    by_room_raw: dict[tuple[str, str], str],
    by_raw_global: dict[str, list[str]],
) -> str | None:
    key = str(target_ref)
    if room_raw_id is not None and (room_raw_id, key) in by_room_raw:
        return by_room_raw[(room_raw_id, key)]
    candidates = by_raw_global.get(key, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _infer_room_subtype(room_id: str) -> str:
    raw = str(room_id).lower()
    for token in ("lobby", "bathroom", "bedroom", "kitchen", "office", "restaurant", "bar", "hallway", "elevator", "lab"):
        if token in raw:
            return token
    return "room"
