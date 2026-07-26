from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .ids import stable_id
from .ontology import relation_category


@dataclass
class CanonicalNode:
    id: str
    type: str
    subtype: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    states: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "subtype": self.subtype,
            "attrs": deepcopy(self.attrs),
            "states": deepcopy(self.states),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalNode":
        return cls(
            id=payload["id"],
            type=payload["type"],
            subtype=payload.get("subtype", ""),
            attrs=deepcopy(payload.get("attrs", {})),
            states=deepcopy(payload.get("states", {})),
        )


@dataclass
class CanonicalEdge:
    source: str
    target: str
    relation: str
    category: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category is None:
            self.category = relation_category(self.relation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "category": self.category,
            "attrs": deepcopy(self.attrs),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalEdge":
        return cls(
            source=payload["source"],
            target=payload["target"],
            relation=payload["relation"],
            category=payload.get("category"),
            attrs=deepcopy(payload.get("attrs", {})),
        )


@dataclass
class CanonicalGraph:
    name: str
    graph_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    nodes: dict[str, CanonicalNode] = field(default_factory=dict)
    edges: list[CanonicalEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.graph_id is None:
            self.graph_id = stable_id("graph", self.name, scene_name=self.name)

    def clone(self) -> "CanonicalGraph":
        return CanonicalGraph.from_dict(self.to_dict())

    def add_node(self, node: CanonicalNode) -> CanonicalNode:
        if node.id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: CanonicalEdge) -> CanonicalEdge:
        if any(
            existing.source == edge.source
            and existing.target == edge.target
            and existing.relation == edge.relation
            for existing in self.edges
        ):
            return edge
        self.edges.append(edge)
        return edge

    def remove_edges(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> None:
        self.edges = [
            edge
            for edge in self.edges
            if not (
                (source is None or edge.source == source)
                and (target is None or edge.target == target)
                and (relation is None or edge.relation == relation)
            )
        ]

    def get_node(self, node_id: str) -> CanonicalNode | None:
        return self.nodes.get(node_id)

    def iter_edges(
        self,
        *,
        source: str | None = None,
        target: str | None = None,
        relation: str | None = None,
    ) -> list[CanonicalEdge]:
        return [
            edge
            for edge in self.edges
            if (source is None or edge.source == source)
            and (target is None or edge.target == target)
            and (relation is None or edge.relation == relation)
        ]

    def neighbors(self, node_id: str, relations: set[str] | None = None) -> list[str]:
        result = []
        for edge in self.edges:
            if edge.source == node_id and (relations is None or edge.relation in relations):
                result.append(edge.target)
        return result

    def incoming(self, node_id: str, relations: set[str] | None = None) -> list[str]:
        result = []
        for edge in self.edges:
            if edge.target == node_id and (relations is None or edge.relation in relations):
                result.append(edge.source)
        return result

    def find_nodes_by_raw_id(self, raw_id: str, types: set[str] | None = None) -> list[CanonicalNode]:
        matches = []
        for node in self.nodes.values():
            if node.id == raw_id or node.attrs.get("raw_id") == raw_id:
                if types is None or node.type in types:
                    matches.append(node)
        return matches

    def resolve_ref(
        self,
        ref: str,
        *,
        types: set[str] | None = None,
        room_scope: str | None = None,
    ) -> CanonicalNode | None:
        if ref in self.nodes:
            node = self.nodes[ref]
            if types is None or node.type in types:
                return node
            return None

        candidates = self.find_nodes_by_raw_id(ref, types=types)
        if room_scope:
            scoped = [
                node
                for node in candidates
                if node.attrs.get("room_raw_id") == room_scope or node.attrs.get("room_id") == room_scope
            ]
            if len(scoped) == 1:
                return scoped[0]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def agent_node(self) -> CanonicalNode | None:
        for node in self.nodes.values():
            if node.type == "agent":
                return node
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "metadata": deepcopy(self.metadata),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalGraph":
        graph = cls(
            name=payload["name"],
            graph_id=payload.get("graph_id"),
            metadata=deepcopy(payload.get("metadata", {})),
        )
        for node_payload in payload.get("nodes", []):
            graph.add_node(CanonicalNode.from_dict(node_payload))
        for edge_payload in payload.get("edges", []):
            graph.add_edge(CanonicalEdge.from_dict(edge_payload))
        return graph
