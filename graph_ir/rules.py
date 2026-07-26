from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from .graph import CanonicalEdge, CanonicalGraph, CanonicalNode


ACTION_ALIASES = {
    "go_to": "goto",
}

ACTION_PATTERN = re.compile(r"^(?P<name>[a-zA-Z_]+)\((?P<args>.*)\)$")


@dataclass(frozen=True)
class ParsedAction:
    name: str
    args: tuple[str, ...]
    raw: str


@dataclass
class ActionCheck:
    action: str
    ok: bool
    message: str = ""


@dataclass
class ActionRule:
    name: str
    validate: Callable[[CanonicalGraph, ParsedAction], ActionCheck]
    apply: Callable[[CanonicalGraph, ParsedAction], None]


def parse_action(action: str) -> ParsedAction:
    raw = action.strip()
    if raw == "finish":
        return ParsedAction(name="finish", args=(), raw=raw)
    match = ACTION_PATTERN.match(raw)
    if not match:
        return ParsedAction(name=raw, args=(), raw=raw)
    name = ACTION_ALIASES.get(match.group("name"), match.group("name"))
    arg_text = match.group("args").strip()
    args = tuple(part.strip() for part in arg_text.split(",") if part.strip()) if arg_text else ()
    return ParsedAction(name=name, args=args, raw=raw)


def validate_action_sequence(
    graph: CanonicalGraph,
    actions: list[str],
    rules: dict[str, ActionRule] | None = None,
) -> list[ActionCheck]:
    active_rules = rules or default_rules()
    working = graph.clone()
    results: list[ActionCheck] = []
    for action in actions:
        parsed = parse_action(action)
        rule = active_rules.get(parsed.name)
        if rule is None:
            results.append(ActionCheck(action=action, ok=False, message=f"Unknown action rule: {parsed.name}"))
            continue
        check = rule.validate(working, parsed)
        results.append(check)
        if check.ok:
            rule.apply(working, parsed)
    return results


def apply_action(graph: CanonicalGraph, action: str, rules: dict[str, ActionRule] | None = None) -> CanonicalGraph:
    active_rules = rules or default_rules()
    working = graph.clone()
    parsed = parse_action(action)
    rule = active_rules.get(parsed.name)
    if rule is None:
        raise ValueError(f"Unknown action rule: {parsed.name}")
    check = rule.validate(working, parsed)
    if not check.ok:
        raise ValueError(check.message or f"Action failed validation: {action}")
    rule.apply(working, parsed)
    return working


def default_rules() -> dict[str, ActionRule]:
    return {
        "close": ActionRule("close", _validate_openable, _apply_close),
        "finish": ActionRule("finish", _validate_finish, _apply_finish),
        "goto": ActionRule("goto", _validate_goto, _apply_goto),
        "open": ActionRule("open", _validate_openable, _apply_open),
        "pick": ActionRule("pick", _validate_pick, _apply_pick),
        "place": ActionRule("place", _validate_place, _apply_place),
        "press": ActionRule("press", _validate_press, _apply_press),
        "scan": ActionRule("scan", _validate_scan, _apply_scan),
        "use": ActionRule("use", _validate_use, _apply_use),
        "wait": ActionRule("wait", _validate_wait, _apply_wait),
    }


def _validate_finish(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    return ActionCheck(action=action.raw, ok=True)


def _apply_finish(graph: CanonicalGraph, action: ParsedAction) -> None:
    return None


def _validate_wait(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    return ActionCheck(action=action.raw, ok=True)


def _apply_wait(graph: CanonicalGraph, action: ParsedAction) -> None:
    return None


def _validate_scan(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) != 1:
        return ActionCheck(action=action.raw, ok=False, message="scan expects one argument.")
    target = _resolve_ref(graph, action.args[0])
    if target is None:
        return ActionCheck(action=action.raw, ok=False, message=f"scan target not found: {action.args[0]}")
    return ActionCheck(action=action.raw, ok=True)


def _apply_scan(graph: CanonicalGraph, action: ParsedAction) -> None:
    agent = _agent(graph)
    if agent is None:
        return
    history = list(agent.attrs.get("scan_history", []))
    history.append(action.args[0])
    agent.attrs["scan_history"] = history
    agent.attrs["last_scanned"] = action.args[0]


def _validate_goto(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) != 1:
        return ActionCheck(action=action.raw, ok=False, message="goto expects one argument.")
    agent = _agent(graph)
    target_room = _resolve_ref(graph, action.args[0], expected_types={"room"})
    if agent is None or target_room is None:
        return ActionCheck(action=action.raw, ok=False, message="agent or target room missing.")
    current_room = _current_room(graph, agent)
    if current_room is None:
        return ActionCheck(action=action.raw, ok=False, message="agent has no current room.")
    if current_room.id == target_room.id:
        return ActionCheck(action=action.raw, ok=True)

    direct = _has_edge(graph, current_room.id, target_room.id, {"neighbor", "connected_to"})
    via_transport = False
    for intermediate in graph.neighbors(current_room.id, {"neighbor", "connected_to"}):
        node = graph.get_node(intermediate)
        if node and node.type == "transport" and _has_edge(graph, intermediate, target_room.id, {"neighbor", "connected_to"}):
            via_transport = True
            break
    if not direct and not via_transport:
        return ActionCheck(action=action.raw, ok=False, message=f"room {action.args[0]} is not reachable in one graph step.")
    return ActionCheck(action=action.raw, ok=True)


def _apply_goto(graph: CanonicalGraph, action: ParsedAction) -> None:
    agent = _agent(graph)
    target_room = _resolve_ref(graph, action.args[0], expected_types={"room"})
    if agent is None or target_room is None:
        return
    graph.remove_edges(source=agent.id, relation="located_in")
    graph.add_edge(CanonicalEdge(source=agent.id, target=target_room.id, relation="located_in", category="agentic"))


def _validate_pick(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) != 1:
        return ActionCheck(action=action.raw, ok=False, message="pick expects one argument.")
    agent = _agent(graph)
    if agent is None:
        return ActionCheck(action=action.raw, ok=False, message="agent missing.")
    if graph.neighbors(agent.id, {"carries"}):
        return ActionCheck(action=action.raw, ok=False, message="agent is already carrying an object.")
    current_room = _current_room(graph, agent)
    obj = _resolve_ref(graph, action.args[0], expected_types={"object", "component"}, room_scope=current_room.attrs.get("raw_id") if current_room else None)
    if current_room is None or obj is None:
        return ActionCheck(action=action.raw, ok=False, message=f"pick target not found: {action.args[0]}")
    object_room = _locate_room(graph, obj.id)
    if object_room is None or object_room.id != current_room.id:
        return ActionCheck(action=action.raw, ok=False, message=f"object {action.args[0]} is not in the current room.")
    return ActionCheck(action=action.raw, ok=True)


def _apply_pick(graph: CanonicalGraph, action: ParsedAction) -> None:
    agent = _agent(graph)
    current_room = _current_room(graph, agent) if agent else None
    obj = _resolve_ref(graph, action.args[0], expected_types={"object", "component"}, room_scope=current_room.attrs.get("raw_id") if current_room else None)
    if agent is None or obj is None:
        return
    _detach_from_support(graph, obj.id)
    graph.add_edge(CanonicalEdge(source=agent.id, target=obj.id, relation="carries", category="agentic"))
    agent.states["holding"] = obj.attrs.get("raw_id", obj.id)


def _validate_place(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) != 2:
        return ActionCheck(action=action.raw, ok=False, message="place expects two arguments.")
    agent = _agent(graph)
    if agent is None:
        return ActionCheck(action=action.raw, ok=False, message="agent missing.")
    current_room = _current_room(graph, agent)
    carried = _resolve_ref(graph, action.args[0], expected_types={"object", "component"}, room_scope=current_room.attrs.get("raw_id") if current_room else None)
    if carried is None or not _has_edge(graph, agent.id, carried.id, {"carries"}):
        return ActionCheck(action=action.raw, ok=False, message=f"agent is not carrying {action.args[0]}.")
    target = _resolve_ref(graph, action.args[1], room_scope=current_room.attrs.get("raw_id") if current_room else None)
    if target is None:
        return ActionCheck(action=action.raw, ok=False, message=f"place target not found: {action.args[1]}")
    return ActionCheck(action=action.raw, ok=True)


def _apply_place(graph: CanonicalGraph, action: ParsedAction) -> None:
    agent = _agent(graph)
    current_room = _current_room(graph, agent) if agent else None
    obj = _resolve_ref(graph, action.args[0], expected_types={"object", "component"}, room_scope=current_room.attrs.get("raw_id") if current_room else None)
    target = _resolve_ref(graph, action.args[1], room_scope=current_room.attrs.get("raw_id") if current_room else None)
    if agent is None or current_room is None or obj is None or target is None:
        return
    graph.remove_edges(source=agent.id, target=obj.id, relation="carries")
    graph.add_edge(CanonicalEdge(source=current_room.id, target=obj.id, relation="contains", category="hierarchical"))
    if target.type == "room":
        graph.add_edge(CanonicalEdge(source=obj.id, target=target.id, relation="located_in", category="agentic"))
    else:
        graph.add_edge(CanonicalEdge(source=obj.id, target=target.id, relation="on", category="spatial"))
    agent.states["holding"] = None


def _validate_openable(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) != 1:
        return ActionCheck(action=action.raw, ok=False, message=f"{action.name} expects one argument.")
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target is None:
        return ActionCheck(action=action.raw, ok=False, message=f"{action.name} target not found: {action.args[0]}")
    affordances = set(target.attrs.get("affordances", []))
    capabilities = target.attrs.get("capabilities", {})
    if action.name not in affordances and not capabilities.get("openable", False) and "door" not in target.subtype:
        return ActionCheck(action=action.raw, ok=False, message=f"{action.args[0]} does not look openable.")
    return ActionCheck(action=action.raw, ok=True)


def _apply_open(graph: CanonicalGraph, action: ParsedAction) -> None:
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target:
        target.states["is_open"] = True


def _apply_close(graph: CanonicalGraph, action: ParsedAction) -> None:
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target:
        target.states["is_open"] = False


def _validate_press(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) != 1:
        return ActionCheck(action=action.raw, ok=False, message="press expects one argument.")
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target is None:
        return ActionCheck(action=action.raw, ok=False, message=f"press target not found: {action.args[0]}")
    affordances = set(target.attrs.get("affordances", []))
    if "press" not in affordances and target.subtype not in {"button", "control"}:
        return ActionCheck(action=action.raw, ok=False, message=f"{action.args[0]} does not look pressable.")
    return ActionCheck(action=action.raw, ok=True)


def _apply_press(graph: CanonicalGraph, action: ParsedAction) -> None:
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target:
        target.states["pressed"] = True


def _validate_use(graph: CanonicalGraph, action: ParsedAction) -> ActionCheck:
    if len(action.args) not in {1, 2}:
        return ActionCheck(action=action.raw, ok=False, message="use expects one or two arguments.")
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target is None:
        return ActionCheck(action=action.raw, ok=False, message=f"use target not found: {action.args[0]}")
    affordances = set(target.attrs.get("affordances", []))
    if "use" not in affordances and not target.attrs.get("methods"):
        return ActionCheck(action=action.raw, ok=False, message=f"{action.args[0]} has no use affordance or method metadata.")
    return ActionCheck(action=action.raw, ok=True)


def _apply_use(graph: CanonicalGraph, action: ParsedAction) -> None:
    target = _resolve_ref(graph, action.args[0], expected_types={"object", "component"})
    if target is None:
        return
    target.states["last_used"] = True
    if len(action.args) == 2:
        target.states["last_used_with"] = action.args[1]


def _agent(graph: CanonicalGraph) -> CanonicalNode | None:
    return graph.agent_node()


def _current_room(graph: CanonicalGraph, agent: CanonicalNode | None) -> CanonicalNode | None:
    if agent is None:
        return None
    for edge in graph.iter_edges(source=agent.id, relation="located_in"):
        node = graph.get_node(edge.target)
        if node and node.type == "room":
            return node
    return None


def _locate_room(graph: CanonicalGraph, node_id: str) -> CanonicalNode | None:
    queue = [node_id]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        node = graph.get_node(current)
        if node and node.type == "room":
            return node
        for parent in graph.incoming(current, {"contains", "part_of", "located_in"}):
            queue.append(parent)
    return None


def _resolve_ref(
    graph: CanonicalGraph,
    ref: str,
    *,
    expected_types: set[str] | None = None,
    room_scope: str | None = None,
) -> CanonicalNode | None:
    return graph.resolve_ref(ref, types=expected_types, room_scope=room_scope)


def _has_edge(graph: CanonicalGraph, source: str, target: str, relations: set[str]) -> bool:
    return any(edge.source == source and edge.target == target and edge.relation in relations for edge in graph.edges)


def _detach_from_support(graph: CanonicalGraph, node_id: str) -> None:
    graph.remove_edges(target=node_id, relation="contains")
    graph.remove_edges(source=node_id, relation="on")
    graph.remove_edges(source=node_id, relation="in")
    graph.remove_edges(source=node_id, relation="located_in")
