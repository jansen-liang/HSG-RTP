from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from .graph import CanonicalGraph
from .ontology import CANONICAL_EDGE_CATEGORIES, CANONICAL_NODE_TYPES, RELATION_SPECS, normalize_relation


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_graph(
    graph: CanonicalGraph,
    *,
    action_sequence: list[str] | None = None,
) -> ValidationResult:
    result = ValidationResult()
    result.stats.update(
        {
            "graph_id": graph.graph_id,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "node_type_histogram": _histogram(node.type for node in graph.nodes.values()),
            "relation_histogram": _histogram(edge.relation for edge in graph.edges),
        }
    )

    _validate_node_types(graph, result)
    _validate_edges(graph, result)
    _validate_room_connectivity(graph, result)
    _validate_inverse_consistency(graph, result)
    _validate_state_payloads(graph, result)
    _validate_hierarchy(graph, result)
    _validate_agent_placement(graph, result)

    if action_sequence is not None:
        from .rules import validate_action_sequence

        checks = validate_action_sequence(graph, action_sequence)
        failing = [check for check in checks if not check.ok]
        result.stats["action_checks"] = len(checks)
        result.stats["action_failures"] = len(failing)
        for check in failing:
            result.errors.append(f"Action validation failed for '{check.action}': {check.message}")

    return result


def _validate_node_types(graph: CanonicalGraph, result: ValidationResult) -> None:
    for node in graph.nodes.values():
        if node.type not in CANONICAL_NODE_TYPES:
            result.errors.append(f"Node '{node.id}' uses unsupported type '{node.type}'.")
        if ":" not in node.id:
            result.warnings.append(f"Node '{node.id}' does not look like a stable canonical id.")


def _validate_edges(graph: CanonicalGraph, result: ValidationResult) -> None:
    for edge in graph.edges:
        if edge.source not in graph.nodes:
            result.errors.append(f"Edge source missing: {edge.source} --{edge.relation}--> {edge.target}")
            continue
        if edge.target not in graph.nodes:
            result.errors.append(f"Edge target missing: {edge.source} --{edge.relation}--> {edge.target}")
            continue
        if edge.category not in CANONICAL_EDGE_CATEGORIES:
            result.errors.append(f"Edge '{edge.source}->{edge.target}' uses unsupported category '{edge.category}'.")

        relation = normalize_relation(edge.relation)
        spec = RELATION_SPECS.get(relation)
        if spec is None:
            result.warnings.append(f"Relation '{edge.relation}' is not in the ontology.")
            continue
        source_type = graph.nodes[edge.source].type
        target_type = graph.nodes[edge.target].type
        if source_type not in spec.source_types:
            result.errors.append(
                f"Relation '{relation}' does not allow source type '{source_type}' on edge '{edge.source}->{edge.target}'."
            )
        if target_type not in spec.target_types:
            result.errors.append(
                f"Relation '{relation}' does not allow target type '{target_type}' on edge '{edge.source}->{edge.target}'."
            )


def _validate_room_connectivity(graph: CanonicalGraph, result: ValidationResult) -> None:
    room_ids = [node.id for node in graph.nodes.values() if node.type == "room"]
    if not room_ids:
        return

    agent_room = None
    for edge in graph.edges:
        if edge.relation == "located_in" and graph.nodes.get(edge.source, None) and graph.nodes[edge.source].type == "agent":
            if edge.target in graph.nodes and graph.nodes[edge.target].type == "room":
                agent_room = edge.target
                break
    start = agent_room or room_ids[0]
    reachable = _bfs_rooms(graph, start)
    if len(reachable) != len(room_ids):
        missing = sorted(set(room_ids) - reachable)
        result.errors.append(f"Room graph is disconnected. Unreachable rooms: {missing}")


def _validate_inverse_consistency(graph: CanonicalGraph, result: ValidationResult) -> None:
    neighbors = {(edge.source, edge.target) for edge in graph.edges if edge.relation in {"neighbor", "connected_to"}}
    for source, target in sorted(neighbors):
        if (target, source) not in neighbors:
            result.warnings.append(f"Missing inverse neighbor/connected edge: '{target}' back to '{source}'.")


def _validate_state_payloads(graph: CanonicalGraph, result: ValidationResult) -> None:
    for node in graph.nodes.values():
        if not isinstance(node.states, dict):
            result.errors.append(f"Node '{node.id}' states must be a dict.")
            continue
        try:
            json.dumps(node.states)
        except TypeError as exc:
            result.errors.append(f"Node '{node.id}' states are not JSON-serializable: {exc}")
        if "on" in node.attrs and "in" in node.attrs:
            result.warnings.append(f"Node '{node.id}' sets both 'on' and 'in' in attrs; spatial state may be ambiguous.")


def _validate_hierarchy(graph: CanonicalGraph, result: ValidationResult) -> None:
    incoming_contains = defaultdict(int)
    for edge in graph.edges:
        if edge.relation in {"contains", "part_of", "located_in", "carries"}:
            incoming_contains[edge.target] += 1

    for node in graph.nodes.values():
        if node.type in {"object", "component"} and incoming_contains[node.id] == 0:
            result.warnings.append(f"Object-like node '{node.id}' has no parent/container/location edge.")


def _validate_agent_placement(graph: CanonicalGraph, result: ValidationResult) -> None:
    agent_nodes = [node for node in graph.nodes.values() if node.type == "agent"]
    if not agent_nodes:
        result.warnings.append("Graph has no agent node.")
        return
    if len(agent_nodes) > 1:
        result.warnings.append(f"Graph has {len(agent_nodes)} agent nodes; expected one active agent.")
    for agent in agent_nodes:
        located = [edge for edge in graph.edges if edge.source == agent.id and edge.relation == "located_in"]
        if not located:
            result.errors.append(f"Agent '{agent.id}' is missing located_in relation.")


def _bfs_rooms(graph: CanonicalGraph, start: str) -> set[str]:
    visited: set[str] = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for edge in graph.edges:
            if edge.source != current or edge.relation not in {"neighbor", "connected_to"}:
                continue
            target = graph.nodes.get(edge.target)
            if target and target.type == "room":
                queue.append(edge.target)
            elif target and target.type == "transport":
                for transport_edge in graph.edges:
                    if transport_edge.source == target.id and transport_edge.relation in {"neighbor", "connected_to"}:
                        transport_target = graph.nodes.get(transport_edge.target)
                        if transport_target and transport_target.type == "room":
                            queue.append(transport_target.id)
    return visited


def _histogram(items) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
