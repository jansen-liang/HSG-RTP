from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Protocol

from pipeline.utils.graph_utils import get_global_view, get_local_view
from pipeline.utils.state_manager import SceneGraphStateManager

from .action_parser import ParseError, parse_global_step, parse_local_action, parse_prediction
from .goal_evaluator import GoalSpec, build_goal_spec, evaluate_goal, find_object
from .perturbations import PerturbationSchedule, RolloutPerturbation
from .plan_evaluator import (
    PlanEvaluation,
    evaluate_global_plan,
    room_reachable,
    transition_destination,
)
from .recovery import (
    FailureFeedback,
    RecoveryConfig,
    build_global_plan_instruction,
    build_global_replan_instruction,
    build_local_recovery_instruction,
    summarize_local_state,
)


class PlanningPolicy(Protocol):
    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str | dict[str, Any]: ...

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str | dict[str, Any]: ...


@dataclass(frozen=True)
class ExecutionEvaluation:
    success: bool
    actions: tuple[str, ...]
    failure_type: str | None
    failure_message: str | None
    goal_failures: tuple[str, ...]
    plan: tuple[str, ...] = ()
    plan_evaluation: PlanEvaluation | None = None
    recovery_trace: tuple[dict[str, Any], ...] = ()


def evaluate_action_sequence(
    record: dict[str, Any], initial_scene: dict[str, Any], actions: list[str]
) -> ExecutionEvaluation:
    manager = SceneGraphStateManager(verbose=False)
    manager.load_initial_state(deepcopy(initial_scene))
    executed = []
    for step_index, raw_action in enumerate(actions):
        try:
            action = parse_local_action(raw_action).canonical
        except ParseError as error:
            return ExecutionEvaluation(
                False, tuple(executed), "parse_error", f"step {step_index}: {error}", ()
            )
        success, _, execution_error = manager.execute_action(action)
        if not success:
            return ExecutionEvaluation(
                False,
                tuple(executed),
                "execution_error",
                f"step {step_index}: {execution_error}",
                (),
            )
        executed.append(action)

    goal = build_goal_spec(record)
    goal_success, goal_failures = evaluate_goal(manager.current_state, goal)
    return ExecutionEvaluation(
        goal_success,
        tuple(executed),
        None if goal_success else "goal_not_reached",
        None if goal_success else "; ".join(goal_failures),
        tuple(goal_failures),
    )


def subgoal_satisfied(
    raw_step: str,
    state: dict[str, Any],
    actions_since_step: list[str],
) -> bool:
    step = parse_global_step(raw_step)
    agent_room = state.get("agent", {}).get("position")
    if step.action == "pass":
        return agent_room == step.room
    if step.action == "trans":
        destination = transition_destination(
            state.get("rooms", {}), step.arguments[1]
        )
        return agent_room == destination
    if step.action == "pick":
        inventory = state.get("agent", {}).get("inventory", {})
        return all(object_id in inventory for object_id in step.arguments)
    if step.action == "place":
        return all(
            (located := find_object(state, object_id)) is not None and located[0] == step.room
            for object_id in step.arguments
        )
    if step.action == "organize":
        picked = {
            parse_local_action(action).arguments[0]
            for action in actions_since_step
            if parse_local_action(action).name == "pick"
        }
        placed = {
            parse_local_action(action).arguments[0]
            for action in actions_since_step
            if parse_local_action(action).name == "place"
        }
        return all(object_id in picked and object_id in placed for object_id in step.arguments)
    return False


def validate_runtime_plan(
    plan: list[str], state: dict[str, Any]
) -> PlanEvaluation:
    errors = []
    parsed_steps = []
    for step_index, raw_step in enumerate(plan):
        try:
            parsed_steps.append(parse_global_step(raw_step))
        except ParseError as error:
            errors.append(f"step {step_index}: {error}")
    if errors:
        return PlanEvaluation(False, tuple(errors), tuple(parsed_steps))
    if not parsed_steps:
        return PlanEvaluation(False, ("global plan is empty",), ())

    rooms = state.get("rooms", {})
    current_room = state.get("agent", {}).get("position")
    for step_index, step in enumerate(parsed_steps):
        if step.room not in rooms:
            errors.append(f"step {step_index}: room {step.room!r} does not exist")
            continue
        if not room_reachable(rooms, current_room, step.room):
            errors.append(
                f"step {step_index}: room {step.room!r} is unreachable from {current_room!r}"
            )
        current_room = step.room
    return PlanEvaluation(not errors, tuple(errors), tuple(parsed_steps))


def ground_local_action(
    action: str,
    state: dict[str, Any],
    pending: list[str] | None = None,
    repair_stalled_action: bool = True,
) -> tuple[str, dict[str, str] | None]:
    """Apply conservative, uniquely determined navigation grounding."""
    parsed = parse_local_action(action)
    original_action = action
    current_room = state.get("agent", {}).get("position")
    rooms = state.get("rooms", {})
    if current_room not in rooms or not parsed.arguments:
        return action, None
    room = rooms[current_room]
    inventory = state.get("agent", {}).get("inventory", {})
    local_items = room.get("items", {})
    if not isinstance(local_items, dict):
        local_items = {item: {} for item in local_items} if isinstance(local_items, list) else {}

    candidate_sets = {
        "goto": sorted(rooms),
        "scan": sorted(
            {current_room, "floor"}
            | set(room.get("small_objects", {}))
            | set(room.get("large_objects", {}))
            | set(local_items)
            | set(rooms)
        ),
        "pick": sorted(room.get("small_objects", {})),
        "press": sorted(room.get("small_objects", {})),
        "wait": ["elevator_down_clear", "elevator_up_clear"],
    }
    arguments = list(parsed.arguments)
    alias_changes = []
    if parsed.name == "place":
        candidate_arguments = [sorted(inventory), sorted(set(room.get("large_objects", {})) | {"floor"})]
    else:
        candidate_arguments = [candidate_sets.get(parsed.name, [])]
    for argument_index, candidates in enumerate(candidate_arguments):
        if argument_index >= len(arguments) or arguments[argument_index] in candidates:
            continue
        matched = _unique_entity_match(arguments[argument_index], candidates)
        if matched is None:
            continue
        alias_changes.append((arguments[argument_index], matched))
        arguments[argument_index] = matched
    alias_grounding = None
    if alias_changes:
        action = f"{parsed.name}({', '.join(arguments)})"
        parsed = parse_local_action(action)
        alias_grounding = {
            "operation": "normalize_entity_alias",
            "from": original_action,
            "to": action,
        }
    target = parsed.arguments[0]
    neighbors = list(rooms[current_room].get("neighbor", []))
    pending_step = None
    if pending:
        try:
            pending_step = parse_global_step(pending[0])
        except ParseError:
            pass

    if (
        pending_step is not None
        and parsed.name == "scan"
        and target == "elevator_cabin"
        and current_room == "elevator_cabin"
    ):
        if pending_step.action == "trans":
            destination_floor = pending_step.arguments[1]
            destination_room = transition_destination(rooms, destination_floor)
            floor_number = destination_floor.removesuffix("f")
            pressed_buttons = state.get("agent", {}).get("pressed_buttons", [])
            destination_button = f"elevator_button_{floor_number}"
            if (
                destination_button not in pressed_buttons
                and destination_button in room.get("small_objects", {})
            ):
                grounded = f"press({destination_button})"
                return grounded, {
                    "operation": "select_pending_transition_floor",
                    "from": action,
                    "to": grounded,
                }
            if (
                destination_room in neighbors
                and destination_button in pressed_buttons
            ):
                grounded = f"goto({destination_room})"
                return grounded, {
                    "operation": "advance_pending_transition",
                    "from": action,
                    "to": grounded,
                }

    scan_history = state.get("agent", {}).get("scan_history", [])
    if (
        repair_stalled_action
        and pending_step is not None
        and parsed.name == "scan"
        and target == current_room
        and current_room in scan_history
    ):
        if pending_step.action != "trans" and current_room != pending_step.room:
            action = f"goto({pending_step.room})"
            parsed = parse_local_action(action)
            target = pending_step.room
            alias_grounding = {
                "operation": "advance_pending_route",
                "from": original_action,
                "to": action,
            }
        elif pending_step.action in {"pick", "organize"} and current_room == pending_step.room:
            object_id = next(
                (
                    candidate
                    for candidate in pending_step.arguments
                    if candidate not in inventory
                    and candidate in room.get("small_objects", {})
                ),
                None,
            )
            if object_id is not None:
                object_data = room.get("small_objects", {}).get(object_id, {})
                relation = object_data.get("relation", {}) if isinstance(object_data, dict) else {}
                support = relation.get("on") or relation.get("in")
                if support and support not in scan_history:
                    grounded = f"scan({support})"
                elif object_id not in scan_history:
                    grounded = f"scan({object_id})"
                else:
                    grounded = f"pick({object_id})"
                return grounded, {
                    "operation": "advance_pending_object_interaction",
                    "from": original_action,
                    "to": grounded,
                }
            held_object = next(
                (candidate for candidate in pending_step.arguments if candidate in inventory),
                None,
            )
            if pending_step.action == "organize" and held_object is not None:
                grounded = f"place({held_object}, floor)"
                return grounded, {
                    "operation": "complete_pending_organize",
                    "from": original_action,
                    "to": grounded,
                }
        elif pending_step.action == "place" and current_room == pending_step.room:
            held_object = next(
                (candidate for candidate in pending_step.arguments if candidate in inventory),
                None,
            )
            if held_object is not None:
                grounded = f"place({held_object}, floor)"
                return grounded, {
                    "operation": "complete_pending_place",
                    "from": original_action,
                    "to": grounded,
                }

    navigation_request = parsed.name == "goto" or (
        parsed.name == "scan" and target in rooms and target != current_room
    )
    if not navigation_request or target not in rooms:
        return action, alias_grounding

    if target in neighbors:
        grounded = f"goto({target})"
        if grounded == action:
            return action, alias_grounding
        return grounded, {
            "operation": "room_scan_to_goto",
            "from": action,
            "to": grounded,
        }

    queue = deque([current_room])
    previous = {current_room: None}
    while queue:
        room_id = queue.popleft()
        if room_id == target:
            break
        for neighbor in rooms[room_id].get("neighbor", []):
            if neighbor not in rooms or neighbor in previous:
                continue
            if room_id.startswith("elevator_") and neighbor.startswith("elevator_"):
                continue
            previous[neighbor] = room_id
            queue.append(neighbor)
    if target not in previous:
        return action, None

    next_room = target
    while previous[next_room] != current_room:
        parent = previous[next_room]
        if parent is None:
            return action, None
        next_room = parent
    grounded = f"goto({next_room})"
    return grounded, {
        "operation": (
            "room_scan_to_route" if parsed.name == "scan" else "route_next_hop"
        ),
        "from": action,
        "to": grounded,
    }


def _normalize_room_id(room_id: str) -> str:
    return "_".join(part for part in room_id.lower().replace("-", "_").split("_") if part)


def _normalize_entity_id(entity_id: str) -> str:
    substitutions = {
        "china": "chinese",
        "takeaway": "takeout",
    }
    ignored = {"box", "can", "item", "object", "package"}
    parts = [
        substitutions.get(part, part)
        for part in _normalize_room_id(entity_id).split("_")
        if part not in ignored
    ]
    return "_".join(parts)


def _unique_entity_match(entity_id: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    normalized = _normalize_entity_id(entity_id)
    exact = [candidate for candidate in candidates if _normalize_entity_id(candidate) == normalized]
    if len(exact) == 1:
        return exact[0]
    scored = sorted(
        (
            SequenceMatcher(None, normalized, _normalize_entity_id(candidate)).ratio(),
            candidate,
        )
        for candidate in candidates
    )
    if not scored or scored[-1][0] < 0.72:
        return None
    if len(scored) > 1 and scored[-1][0] - scored[-2][0] < 0.08:
        return None
    return scored[-1][1]


def _format_global_step(room: str, action: str, arguments: tuple[str, ...]) -> str:
    if action == "trans":
        return f"goto({room}): trans from({arguments[0]}) to({arguments[1]})"
    return f"goto({room}): {action}({', '.join(arguments)})"


def _normalize_global_syntax(raw_step: str) -> str:
    match = re.fullmatch(
        r"goto\(([^()]+)\)\s*:\s*(pick|place|organize)\s+(.+)",
        raw_step.strip(),
    )
    if not match:
        return raw_step
    arguments = ", ".join(
        argument.strip() for argument in match.group(3).split(",") if argument.strip()
    )
    return f"goto({match.group(1).strip()}): {match.group(2)}({arguments})"


def _object_room(state: dict[str, Any], object_id: str) -> str | None:
    located = find_object(state, object_id)
    return located[0] if located is not None and located[0] != "agent_inventory" else None


def _semantic_anchors(
    parsed_steps: list[Any], state: dict[str, Any], record: dict[str, Any]
) -> list[tuple[str, str, tuple[str, ...]]]:
    task_info = record.get("task_info", {})
    task_type = task_info.get("type")
    parameters = task_info.get("parameters", {})
    objects = list(parameters.get("objects", []))

    if task_type == "guidance":
        if not any(step is not None for step in parsed_steps):
            return []
        required = list(parameters.get("intermediate_points", []))
        end_room = parameters.get("end_room")
        if end_room:
            required.append(end_room)
        return [(room_id, "pass", ()) for room_id in required]

    recognized_objects = set()
    for step in parsed_steps:
        if step is None:
            continue
        for argument in step.arguments:
            matched = argument if argument in objects else _unique_entity_match(argument, objects)
            if matched is not None:
                recognized_objects.add(matched)
    if not recognized_objects:
        return []

    if task_type == "delivery":
        source_room = parameters.get("source_room")
        target_rooms = list(parameters.get("target_rooms", []))
        if not source_room or not target_rooms:
            return []
        repaired = []
        for object_index, object_id in enumerate(objects):
            repaired.append((source_room, "pick", (object_id,)))
            repaired.append(
                (target_rooms[object_index % len(target_rooms)], "place", (object_id,))
            )
        return repaired
    if task_type == "tidying":
        repaired = []
        for object_id in objects:
            room_id = _object_room(state, object_id)
            if room_id is not None:
                repaired.append((room_id, "organize", (object_id,)))
        return repaired
    return []


def _shortest_room_path(rooms: dict[str, Any], start: str, goal: str) -> list[str]:
    if start == goal:
        return [start]
    queue = deque([start])
    previous = {start: None}
    while queue:
        room_id = queue.popleft()
        for neighbor in rooms.get(room_id, {}).get("neighbor", []):
            if neighbor not in rooms or neighbor in previous:
                continue
            if neighbor == "elevator_cabin":
                continue
            previous[neighbor] = room_id
            if neighbor == goal:
                queue.clear()
                break
            queue.append(neighbor)
    if goal not in previous:
        return []
    path = []
    cursor = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    return list(reversed(path))


def _route_global_anchors(
    anchors: list[tuple[str, str, tuple[str, ...]]], state: dict[str, Any]
) -> list[str]:
    rooms = state.get("rooms", {})
    current_room = state.get("agent", {}).get("position", "")
    routed = []
    for target_room, action, arguments in anchors:
        if target_room not in rooms or current_room not in rooms:
            routed.append(_format_global_step(target_room, action, arguments))
            current_room = target_room
            continue
        path = _shortest_room_path(rooms, current_room, target_room)
        if not path:
            routed.append(_format_global_step(target_room, action, arguments))
            current_room = target_room
            continue
        for source, destination in zip(path, path[1:]):
            source_floor = re.fullmatch(r"elevator_(\d+f)", source)
            destination_floor = re.fullmatch(r"elevator_(\d+f)", destination)
            if source_floor and destination_floor:
                routed.append(
                    _format_global_step(
                        source,
                        "trans",
                        (source_floor.group(1), destination_floor.group(1)),
                    )
                )
            elif destination != target_room:
                routed.append(_format_global_step(destination, "pass", ()))
        routed.append(_format_global_step(target_room, action, arguments))
        current_room = target_room
    return routed


def _room_distances(rooms: dict[str, Any], start_room: str) -> dict[str, int]:
    distances = {start_room: 0}
    queue = [start_room]
    for room_id in queue:
        for neighbor in rooms.get(room_id, {}).get("neighbor", []):
            if neighbor in rooms and neighbor not in distances:
                distances[neighbor] = distances[room_id] + 1
                queue.append(neighbor)
    return distances


def _task_room_ids(record: dict[str, Any], valid_rooms: set[str]) -> list[str]:
    parameters = record.get("task_info", {}).get("parameters", {})
    candidates = []
    for key in (
        "waypoints",
        "start_room",
        "end_room",
        "source_room",
        "target_rooms",
    ):
        value = parameters.get(key)
        values = value if isinstance(value, list) else [value]
        for room_id in values:
            if isinstance(room_id, str) and room_id in valid_rooms and room_id not in candidates:
                candidates.append(room_id)
    return candidates


def _predicted_task_anchors(
    record: dict[str, Any], steps: list[Any]
) -> list[tuple[str, str, tuple[str, ...]]]:
    task_info = record.get("task_info", {})
    task_type = task_info.get("type")
    if task_type == "guidance":
        return [
            (step.room, "pass", ())
            for step in steps
            if step.action != "trans"
        ]
    return [
        (step.room, step.action, tuple(step.arguments))
        for step in steps
        if step.action in {"pick", "place", "organize"}
    ]


def _sequence_edit_distance(left: list[Any], right: list[Any]) -> int:
    previous = list(range(len(right) + 1))
    for left_item in left:
        current = [previous[0] + 1]
        for index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def ground_global_plan(
    plan: list[str],
    state: dict[str, Any],
    record: dict[str, Any] | None = None,
    normalize_rooms: bool = True,
    repair_task_plan: bool = True,
    complete_task_coverage: bool = True,
    task_semantic_repair_budget: int | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Conservatively ground room names before strict plan validation."""
    valid_rooms = sorted(state.get("rooms", {}))
    normalized_rooms = {
        room_id: _normalize_room_id(room_id) for room_id in valid_rooms
    }
    grounded_plan = []
    changes = []
    task_type = record.get("task_info", {}).get("type") if record else None

    for original_step in plan:
        raw_step = _normalize_global_syntax(original_step)
        if raw_step != original_step:
            changes.append(
                {
                    "operation": "normalize_global_syntax",
                    "from": original_step,
                    "to": raw_step,
                }
            )
        bare_goto_match = re.fullmatch(r"goto\(([^()]+)\)", raw_step.strip())
        if bare_goto_match:
            room_id = bare_goto_match.group(1).strip()
            grounded_plan.append(f"goto({room_id}): pass()")
            changes.append(
                {
                    "operation": "normalize_bare_goto",
                    "from": raw_step,
                    "to": f"goto({room_id}): pass()",
                }
            )
            continue
        guidance_scan_match = re.fullmatch(
            r"goto\(([^()]+)\)\s*:\s*scan\(([^()]+)\)", raw_step.strip()
        )
        if (
            task_type == "guidance"
            and guidance_scan_match
            and guidance_scan_match.group(1).strip()
            == guidance_scan_match.group(2).strip()
        ):
            room_id = guidance_scan_match.group(1).strip()
            grounded_plan.append(f"goto({room_id}): pass()")
            changes.append(
                {
                    "operation": "normalize_guidance_scan",
                    "from": raw_step,
                    "to": f"goto({room_id}): pass()",
                }
            )
            continue
        try:
            parsed_step = parse_global_step(raw_step)
        except ParseError:
            grounded_plan.append(raw_step)
            continue
        if parsed_step.room in normalized_rooms:
            grounded_plan.append(raw_step)
            continue

        if not normalize_rooms:
            grounded_plan.append(raw_step)
            continue

        normalized_target = _normalize_room_id(parsed_step.room)
        candidates = [
            room_id
            for room_id, normalized_room in normalized_rooms.items()
            if normalized_room.startswith(normalized_target)
            or normalized_target.startswith(normalized_room)
        ]
        if len(candidates) == 1:
            grounded_room = candidates[0]
            grounded_plan.append(
                raw_step.replace(
                    f"goto({parsed_step.room})", f"goto({grounded_room})", 1
                )
            )
            changes.append(
                {
                    "operation": "normalize_room",
                    "from": parsed_step.room,
                    "to": grounded_room,
                }
            )
        elif parsed_step.action == "pass":
            changes.append(
                {
                    "operation": "drop_ungrounded_pass",
                    "from": parsed_step.room,
                    "to": "",
                }
            )
        else:
            grounded_plan.append(raw_step)

    if record is None or not repair_task_plan:
        return grounded_plan, changes

    if not complete_task_coverage or task_semantic_repair_budget is not None:
        predicted_anchors = []
        predicted_steps = []
        for raw_step in grounded_plan:
            try:
                parsed_step = parse_global_step(raw_step)
            except ParseError:
                return grounded_plan, changes
            predicted_steps.append(parsed_step)
            predicted_anchors.append(
                (parsed_step.room, parsed_step.action, tuple(parsed_step.arguments))
            )
        routed_plan = _route_global_anchors(predicted_anchors, state)
        if routed_plan != grounded_plan:
            changes.append(
                {
                    "operation": "complete_predicted_route",
                    "from": " | ".join(grounded_plan),
                    "to": " | ".join(routed_plan),
                }
            )
        if not complete_task_coverage:
            return routed_plan, changes

        expected_anchors = _semantic_anchors(predicted_steps, state, record)
        semantic_distance = _sequence_edit_distance(
            _predicted_task_anchors(record, predicted_steps), expected_anchors
        )
        if not expected_anchors or semantic_distance > task_semantic_repair_budget:
            return routed_plan, changes
        changes.append(
            {
                "operation": "bounded_task_semantic_repair",
                "from": str(semantic_distance),
                "to": str(task_semantic_repair_budget),
            }
        )

    rooms = state.get("rooms", {})
    current_room = state.get("agent", {}).get("position", "")
    required_rooms = _task_room_ids(record, set(valid_rooms))
    required_room_set = set(required_rooms)
    parsed_grounded = []
    for raw_step in grounded_plan:
        try:
            parsed_grounded.append(parse_global_step(raw_step))
        except ParseError:
            parsed_grounded.append(None)

    pruned_plan = []
    route_anchor = current_room
    for step_index, (raw_step, parsed_step) in enumerate(
        zip(grounded_plan, parsed_grounded)
    ):
        if (
            parsed_step is None
            or parsed_step.action != "pass"
            or parsed_step.room in required_room_set
        ):
            pruned_plan.append(raw_step)
            if parsed_step is not None and parsed_step.room in required_room_set:
                route_anchor = parsed_step.room
            continue

        next_required_room = next(
            (
                later_step.room
                for later_step in parsed_grounded[step_index + 1:]
                if later_step is not None and later_step.room in required_room_set
            ),
            None,
        )
        if next_required_room is None:
            pruned_plan.append(raw_step)
            continue

        from_anchor = _room_distances(rooms, route_anchor)
        from_candidate = _room_distances(rooms, parsed_step.room)
        shortest_distance = from_anchor.get(next_required_room)
        candidate_distance = from_anchor.get(parsed_step.room)
        remaining_distance = from_candidate.get(next_required_room)
        lies_on_shortest_path = (
            shortest_distance is not None
            and candidate_distance is not None
            and remaining_distance is not None
            and candidate_distance + remaining_distance == shortest_distance
        )
        if lies_on_shortest_path:
            pruned_plan.append(raw_step)
        else:
            changes.append(
                {
                    "operation": "drop_off_path_pass",
                    "from": parsed_step.room,
                    "to": "",
                }
            )

    reparsed = []
    for raw_step in pruned_plan:
        try:
            reparsed.append(parse_global_step(raw_step))
        except ParseError:
            reparsed.append(None)
    anchors = _semantic_anchors(reparsed, state, record)
    if not anchors:
        return pruned_plan, changes
    routed_plan = _route_global_anchors(anchors, state)
    if routed_plan != pruned_plan:
        changes.append(
            {
                "operation": "repair_task_route_and_coverage",
                "from": " | ".join(pruned_plan),
                "to": " | ".join(routed_plan),
            }
        )
    return routed_plan, changes


def rollout_policy(
    record: dict[str, Any],
    initial_scene: dict[str, Any],
    policy: PlanningPolicy,
    max_steps: int | None = None,
    recovery_config: RecoveryConfig | None = None,
    perturbations: list[RolloutPerturbation] | None = None,
) -> ExecutionEvaluation:
    config = recovery_config or RecoveryConfig()
    perturbation_schedule = PerturbationSchedule(list(perturbations or []))
    manager = SceneGraphStateManager(verbose=False)
    manager.load_initial_state(deepcopy(initial_scene))
    instruction = record.get("instruction", "")
    completed = []
    recovery_trace = []
    global_view = get_global_view(manager.current_state)
    valid_rooms = sorted(global_view.get("rooms", {}))
    plan = []
    plan_evaluation = None
    previous_global_error = None
    global_failure_type = "global_parse_error"

    for attempt_index in range(config.max_initial_plan_retries + 1):
        planning_instruction = build_global_plan_instruction(
            instruction,
            valid_rooms,
            previous_error=previous_global_error,
            retry_count=attempt_index,
            task_info=record.get("task_info", {}),
        )
        recovery_trace.append(
            {
                "event": "initial_global_plan_attempt",
                "attempt": attempt_index + 1,
                "max_retries": config.max_initial_plan_retries,
            }
        )
        try:
            global_prediction = policy.generate_global(
                planning_instruction, global_view, completed
            )
            plan = parse_prediction(global_prediction, "global")
        except (ParseError, ValueError, RuntimeError) as error:
            previous_global_error = str(error)
            global_failure_type = "global_parse_error"
            recovery_trace.append(
                {
                    "event": "initial_global_plan_failure",
                    "attempt": attempt_index + 1,
                    "failure_type": global_failure_type,
                    "reason": previous_global_error,
                }
            )
            continue

        grounded_plan, grounding_changes = ground_global_plan(
            plan,
            initial_scene,
            record,
            normalize_rooms=config.normalize_global_rooms,
            repair_task_plan=config.repair_global_task_plan,
            complete_task_coverage=config.complete_global_task_coverage,
            task_semantic_repair_budget=config.task_semantic_repair_budget,
        )
        grounded_evaluation = evaluate_global_plan(
            record, grounded_plan, initial_scene
        )
        if grounding_changes:
            plan = grounded_plan
            plan_evaluation = grounded_evaluation
            recovery_trace.append(
                {
                    "event": "initial_global_plan_grounded",
                    "attempt": attempt_index + 1,
                    "changes": grounding_changes,
                    "plan": list(plan),
                }
            )
        else:
            plan_evaluation = evaluate_global_plan(record, plan, initial_scene)
        if plan_evaluation.success:
            recovery_trace.append(
                {
                    "event": "initial_global_plan_success",
                    "attempt": attempt_index + 1,
                    "plan": list(plan),
                }
            )
            break

        previous_global_error = "; ".join(plan_evaluation.errors)
        global_failure_type = "invalid_global_plan"
        recovery_trace.append(
            {
                "event": "initial_global_plan_failure",
                "attempt": attempt_index + 1,
                "failure_type": global_failure_type,
                "reason": previous_global_error,
                "plan": list(plan),
            }
        )
    else:
        return ExecutionEvaluation(
            False,
            (),
            global_failure_type,
            previous_global_error,
            (),
            tuple(plan),
            plan_evaluation,
            tuple(recovery_trace),
        )

    pending = list(plan)
    actions_since_step = []
    goal: GoalSpec = build_goal_spec(record)
    step_limit = max_steps or max(10, 2 * len(record.get("execution_summary", {}).get("subtasks", [])))
    global_replan_count = 0

    for step_index in range(step_limit):
        try:
            perturbation_records = perturbation_schedule.apply_before_planning(
                step_index, manager
            )
        except ValueError as error:
            return ExecutionEvaluation(
                False,
                tuple(completed),
                "perturbation_error",
                str(error),
                (),
                tuple(plan),
                plan_evaluation,
                tuple(recovery_trace),
            )
        recovery_trace.extend(
            {"event": "perturbation", **record.as_dict()}
            for record in perturbation_records
        )

        goal_success, _ = evaluate_goal(manager.current_state, goal)
        if goal_success:
            return ExecutionEvaluation(
                True,
                tuple(completed),
                None,
                None,
                (),
                tuple(plan),
                plan_evaluation,
                tuple(recovery_trace),
            )

        while pending and subgoal_satisfied(pending[0], manager.current_state, actions_since_step):
            pending.pop(0)
            actions_since_step = []
        if not pending:
            _, goal_failures = evaluate_goal(manager.current_state, goal)
            return ExecutionEvaluation(
                False,
                tuple(completed),
                "goal_not_reached",
                "; ".join(goal_failures),
                tuple(goal_failures),
                tuple(plan),
                plan_evaluation,
                tuple(recovery_trace),
            )

        local_retry_count = 0
        forbidden_actions = []
        feedback = None
        local_success = False

        while True:
            current_room = manager.current_state["agent"]["position"]
            local_view = get_local_view(manager.current_state, current_room)
            effective_instruction = instruction
            if feedback is not None:
                effective_instruction = build_local_recovery_instruction(
                    instruction, feedback, forbidden_actions
                )
            action = None
            try:
                local_prediction = policy.generate_local(
                    effective_instruction, local_view, completed, pending
                )
                action = parse_local_action(
                    parse_prediction(local_prediction, "local")[0]
                ).canonical
                action, grounding = ground_local_action(
                    action,
                    manager.current_state,
                    pending,
                    repair_stalled_action=config.repair_stalled_local_action,
                )
                if grounding is not None:
                    recovery_trace.append(
                        {
                            "event": "local_action_grounded",
                            "step_index": step_index,
                            **grounding,
                        }
                    )
                if action in forbidden_actions:
                    raise ValueError(
                        f"Action {action!r} repeated without a state change after failure"
                    )
            except (ParseError, ValueError, RuntimeError) as error:
                feedback = FailureFeedback(
                    failure_type="local_parse_error",
                    action=action,
                    reason=str(error),
                    current_room=current_room,
                    retry_count=local_retry_count + 1,
                    observation=summarize_local_state(manager.current_state),
                )
                recovery_trace.append(
                    {"event": "local_failure", **feedback.as_dict()}
                )
            else:
                forced_failure = perturbation_schedule.intercept_execution(
                    step_index, action
                )
                if forced_failure is not None:
                    recovery_trace.append(
                        {"event": "perturbation", **forced_failure.as_dict()}
                    )
                    success = False
                    execution_error = forced_failure.message
                    retryable_same_action = bool(
                        forced_failure.details.get("retryable_same_action")
                    )
                    feedback = FailureFeedback(
                        failure_type="execution_error",
                        action=action,
                        reason=execution_error,
                        current_room=current_room,
                        retry_count=local_retry_count + 1,
                        observation=summarize_local_state(manager.current_state),
                        retryable_same_action=retryable_same_action,
                    )
                    recovery_trace.append(
                        {"event": "local_failure", **feedback.as_dict()}
                    )
                    if retryable_same_action:
                        recovery_trace.append(
                            {
                                "event": "automatic_retry_attempt",
                                "step_index": step_index,
                                "action": action,
                            }
                        )
                        success, _, execution_error = manager.execute_action(action)
                        if success:
                            completed.append(action)
                            actions_since_step.append(action)
                            recovery_trace.append(
                                {
                                    "event": "action_success",
                                    "step_index": step_index,
                                    "action": action,
                                    "local_retry_count": local_retry_count,
                                    "automatic_retry": True,
                                }
                            )
                            local_success = True
                            break

                        feedback = FailureFeedback(
                            failure_type="execution_error",
                            action=action,
                            reason=(
                                f"Automatic retry failed: {execution_error}"
                                if execution_error
                                else "Automatic retry failed"
                            ),
                            current_room=current_room,
                            retry_count=local_retry_count + 1,
                            observation=summarize_local_state(manager.current_state),
                        )
                        if action not in forbidden_actions:
                            forbidden_actions.append(action)
                        recovery_trace.append(
                            {"event": "local_failure", **feedback.as_dict()}
                        )
                        success = False
                else:
                    success, _, execution_error = manager.execute_action(action)
                    retryable_same_action = False

                if success:
                    completed.append(action)
                    actions_since_step.append(action)
                    recovery_trace.append(
                        {
                            "event": "action_success",
                            "step_index": step_index,
                            "action": action,
                            "local_retry_count": local_retry_count,
                        }
                    )
                    local_success = True
                    break

                if forced_failure is None:
                    feedback = FailureFeedback(
                        failure_type="execution_error",
                        action=action,
                        reason=execution_error or "Unknown execution error",
                        current_room=current_room,
                        retry_count=local_retry_count + 1,
                        observation=summarize_local_state(manager.current_state),
                    )
                    if action not in forbidden_actions:
                        forbidden_actions.append(action)
                    recovery_trace.append(
                        {"event": "local_failure", **feedback.as_dict()}
                    )

            if local_retry_count >= config.max_local_retries:
                break
            local_retry_count += 1

        if local_success:
            continue

        if feedback is None:
            feedback = FailureFeedback(
                failure_type="recovery_error",
                reason="Local recovery ended without failure feedback",
                current_room=manager.current_state["agent"]["position"],
                retry_count=local_retry_count,
            )

        recovery_trace.append(
            {
                "event": "local_recovery_exhausted",
                "attempts": local_retry_count + 1,
                "max_local_retries": config.max_local_retries,
                "last_feedback": feedback.as_dict(),
            }
        )

        replanned = False
        previous_pending = list(pending)
        while global_replan_count < config.max_global_replans:
            global_replan_count += 1
            recovery_trace.append(
                {
                    "event": "global_replan_attempt",
                    "replan_count": global_replan_count,
                    "previous_pending": previous_pending,
                }
            )
            replan_instruction = build_global_replan_instruction(
                instruction,
                feedback,
                completed,
                previous_pending,
                sorted(manager.current_state.get("rooms", {})),
            )
            try:
                global_prediction = policy.generate_global(
                    replan_instruction,
                    get_global_view(manager.current_state),
                    completed,
                )
                revised_plan = parse_prediction(global_prediction, "global")
                runtime_evaluation = validate_runtime_plan(
                    revised_plan, manager.current_state
                )
                if not runtime_evaluation.success:
                    raise ValueError("; ".join(runtime_evaluation.errors))
            except (ParseError, ValueError, RuntimeError) as error:
                feedback = FailureFeedback(
                    failure_type="global_replan_error",
                    reason=str(error),
                    current_room=manager.current_state["agent"]["position"],
                    retry_count=global_replan_count,
                    observation=summarize_local_state(manager.current_state),
                )
                recovery_trace.append(
                    {"event": "global_replan_failure", **feedback.as_dict()}
                )
                continue

            plan = list(revised_plan)
            pending = list(revised_plan)
            actions_since_step = []
            recovery_trace.append(
                {
                    "event": "global_replan_success",
                    "replan_count": global_replan_count,
                    "previous_pending": previous_pending,
                    "revised_plan": list(revised_plan),
                }
            )
            replanned = True
            break

        if not replanned:
            failure_type = (
                feedback.failure_type
                if config.max_local_retries == 0
                and config.max_global_replans == 0
                else "recovery_exhausted"
            )
            return ExecutionEvaluation(
                False,
                tuple(completed),
                failure_type,
                feedback.reason,
                (),
                tuple(plan),
                plan_evaluation,
                tuple(recovery_trace),
            )

    _, goal_failures = evaluate_goal(manager.current_state, goal)
    return ExecutionEvaluation(
        False,
        tuple(completed),
        "step_limit",
        f"Exceeded maximum of {step_limit} local actions",
        tuple(goal_failures),
        tuple(plan),
        plan_evaluation,
        tuple(recovery_trace),
    )
