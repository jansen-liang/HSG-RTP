from __future__ import annotations

from dataclasses import dataclass


CANONICAL_NODE_TYPES = {
    "scene",
    "zone",
    "floor",
    "room",
    "object",
    "component",
    "agent",
    "transport",
}

CANONICAL_EDGE_CATEGORIES = {
    "hierarchical",
    "spatial",
    "logical",
    "agentic",
}


@dataclass(frozen=True)
class RelationSpec:
    name: str
    category: str
    inverse: str | None
    source_types: frozenset[str]
    target_types: frozenset[str]


RELATION_ALIASES = {
    "adjacent": "neighbor",
    "attached": "attached_to",
    "attached_to": "attached_to",
    "beside": "beside",
    "belong_to": "part_of",
    "belongs_to": "part_of",
    "connected": "connected_to",
    "contain": "contains",
    "contains": "contains",
    "control": "controls",
    "controls": "controls",
    "in": "in",
    "in_room": "located_in",
    "inside": "in",
    "next_to": "next_to",
    "near": "near",
    "neighbor": "neighbor",
    "neighbour": "neighbor",
    "on": "on",
    "ontop": "on",
    "outside": "outside_of",
    "part_of": "part_of",
    "far": "far",
    "under": "under",
}


RELATION_SPECS = {
    "contains": RelationSpec(
        name="contains",
        category="hierarchical",
        inverse="contained_by",
        source_types=frozenset({"scene", "zone", "floor", "room", "object", "transport", "agent"}),
        target_types=frozenset({"zone", "floor", "room", "object", "component", "transport"}),
    ),
    "part_of": RelationSpec(
        name="part_of",
        category="hierarchical",
        inverse="has_part",
        source_types=frozenset({"component", "object"}),
        target_types=frozenset({"object", "transport"}),
    ),
    "neighbor": RelationSpec(
        name="neighbor",
        category="spatial",
        inverse="neighbor",
        source_types=frozenset({"room", "transport"}),
        target_types=frozenset({"room", "transport"}),
    ),
    "connected_to": RelationSpec(
        name="connected_to",
        category="spatial",
        inverse="connected_to",
        source_types=frozenset({"room", "transport", "zone", "floor"}),
        target_types=frozenset({"room", "transport", "zone", "floor"}),
    ),
    "on": RelationSpec(
        name="on",
        category="spatial",
        inverse="supports",
        source_types=frozenset({"object", "component"}),
        target_types=frozenset({"object", "room", "transport"}),
    ),
    "in": RelationSpec(
        name="in",
        category="spatial",
        inverse="contains",
        source_types=frozenset({"object", "component"}),
        target_types=frozenset({"object", "room", "transport"}),
    ),
    "next_to": RelationSpec(
        name="next_to",
        category="spatial",
        inverse="next_to",
        source_types=frozenset({"object", "component", "room", "transport"}),
        target_types=frozenset({"object", "component", "room", "transport"}),
    ),
    "near": RelationSpec(
        name="near",
        category="spatial",
        inverse="near",
        source_types=frozenset({"object", "component", "room", "transport"}),
        target_types=frozenset({"object", "component", "room", "transport"}),
    ),
    "far": RelationSpec(
        name="far",
        category="spatial",
        inverse="far",
        source_types=frozenset({"object", "component", "room", "transport"}),
        target_types=frozenset({"object", "component", "room", "transport"}),
    ),
    "beside": RelationSpec(
        name="beside",
        category="spatial",
        inverse="beside",
        source_types=frozenset({"object", "component", "room", "transport"}),
        target_types=frozenset({"object", "component", "room", "transport"}),
    ),
    "under": RelationSpec(
        name="under",
        category="spatial",
        inverse="over",
        source_types=frozenset({"object", "component"}),
        target_types=frozenset({"object", "component", "room", "transport"}),
    ),
    "attached_to": RelationSpec(
        name="attached_to",
        category="spatial",
        inverse="hosts_attachment",
        source_types=frozenset({"object", "component"}),
        target_types=frozenset({"object", "component", "room", "transport"}),
    ),
    "outside_of": RelationSpec(
        name="outside_of",
        category="spatial",
        inverse="contains",
        source_types=frozenset({"object", "component"}),
        target_types=frozenset({"object", "room", "transport"}),
    ),
    "controls": RelationSpec(
        name="controls",
        category="logical",
        inverse="controlled_by",
        source_types=frozenset({"object", "component"}),
        target_types=frozenset({"object", "component", "transport", "room"}),
    ),
    "located_in": RelationSpec(
        name="located_in",
        category="agentic",
        inverse="contains",
        source_types=frozenset({"agent", "object", "transport"}),
        target_types=frozenset({"room", "zone", "floor", "transport", "object"}),
    ),
    "carries": RelationSpec(
        name="carries",
        category="agentic",
        inverse="carried_by",
        source_types=frozenset({"agent", "transport"}),
        target_types=frozenset({"object", "component"}),
    ),
}


def normalize_relation(name: str) -> str:
    key = str(name).strip().lower()
    if not key:
        return "related_to"
    return RELATION_ALIASES.get(key, key)


def relation_category(name: str) -> str:
    normalized = normalize_relation(name)
    spec = RELATION_SPECS.get(normalized)
    return spec.category if spec else "spatial"
