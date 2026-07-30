from collections import deque
from dataclasses import dataclass
from typing import Any

from .action_parser import GlobalStep, ParseError, parse_global_step


@dataclass(frozen=True)
class PlanEvaluation:
    success: bool
    errors: tuple[str, ...]
    parsed_steps: tuple[GlobalStep, ...]


def room_reachable(rooms: dict[str, Any], start: str, goal: str) -> bool:
    if start == goal:
        return True
    if start not in rooms or goal not in rooms:
        return False
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        neighbors = list(rooms[current].get("neighbor", []))
        if current.startswith("elevator_"):
            neighbors.extend(room_id for room_id in rooms if room_id.startswith("elevator_"))
        for neighbor in neighbors:
            if neighbor == goal:
                return True
            if neighbor in rooms and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False


def transition_destination(rooms: dict[str, Any], floor: str) -> str | None:
    direct_match = f"elevator_{floor}"
    if direct_match in rooms:
        return direct_match
    return next(
        (room_id for room_id, room in rooms.items() if room_id.startswith("elevator_") and str(room.get("floor")) == floor),
        None,
    )


def validate_task_coverage(record: dict[str, Any], steps: list[GlobalStep]) -> list[str]:
    task_info = record.get("task_info", {})
    task_type = task_info.get("type")
    parameters = task_info.get("parameters", {})
    errors = []
    requested_states = parameters.get("objects_goal_state", {}) or parameters.get("object_goals", {})
    if requested_states:
        errors.append(
            "task requires object-state transitions that are not represented in the current global action vocabulary"
        )

    if task_type == "guidance":
        required_rooms = list(parameters.get("intermediate_points", [])) + [parameters.get("end_room")]
        visited_rooms = [step.room for step in steps]
        cursor = 0
        for room_id in visited_rooms:
            if cursor < len(required_rooms) and room_id == required_rooms[cursor]:
                cursor += 1
        if cursor != len(required_rooms):
            errors.append(f"guidance plan does not cover ordered waypoints {required_rooms}")
        return errors

    objects = set(parameters.get("objects", []))
    picked_objects = {
        object_id
        for step in steps
        if step.action == "pick"
        for object_id in step.arguments
    }
    if task_type == "delivery":
        target_rooms = set(parameters.get("target_rooms", []))
        source_room = parameters.get("source_room")
        placed_objects = {
            object_id
            for step in steps
            if step.action == "place" and step.room in target_rooms
            for object_id in step.arguments
        }
        missing_pick = sorted(objects - picked_objects)
        missing_place = sorted(objects - placed_objects)
        if missing_pick:
            errors.append(f"delivery plan does not pick {missing_pick}")
        if missing_place:
            errors.append(f"delivery plan does not place {missing_place} in target rooms")
        for object_id in sorted(objects):
            pick_indices = [
                index
                for index, step in enumerate(steps)
                if step.action == "pick" and object_id in step.arguments
            ]
            place_indices = [
                index
                for index, step in enumerate(steps)
                if step.action == "place" and object_id in step.arguments and step.room in target_rooms
            ]
            if pick_indices and steps[pick_indices[0]].room != source_room:
                errors.append(f"delivery plan picks {object_id} outside source room {source_room}")
            if pick_indices and place_indices and pick_indices[0] >= place_indices[0]:
                errors.append(f"delivery plan places {object_id} before picking it")
    elif task_type == "tidying":
        source_rooms = set(parameters.get("source_rooms", []))
        organized_objects = {
            object_id
            for step in steps
            if step.action == "organize"
            for object_id in step.arguments
        }
        missing = sorted(objects - organized_objects)
        if missing:
            errors.append(f"tidying plan does not organize {missing}")
        for step in steps:
            if step.action == "organize" and source_rooms and step.room not in source_rooms:
                errors.append(f"tidying plan organizes objects outside source rooms {sorted(source_rooms)}")
    else:
        errors.append(f"unsupported task type {task_type!r}")
    return errors


def evaluate_global_plan(record: dict[str, Any], plan: list[str], initial_scene: dict[str, Any]) -> PlanEvaluation:
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

    rooms = initial_scene.get("rooms", {})
    current_room = initial_scene.get("agent", {}).get("position")
    for step_index, step in enumerate(parsed_steps):
        if step.room not in rooms:
            errors.append(f"step {step_index}: room {step.room!r} does not exist")
            continue
        if not room_reachable(rooms, current_room, step.room):
            errors.append(f"step {step_index}: room {step.room!r} is unreachable from {current_room!r}")
        current_room = step.room
        if step.action == "trans":
            destination = transition_destination(rooms, step.arguments[1])
            if destination is None:
                errors.append(f"step {step_index}: no elevator hall for floor {step.arguments[1]!r}")
            else:
                current_room = destination

    errors.extend(validate_task_coverage(record, parsed_steps))
    return PlanEvaluation(not errors, tuple(errors), tuple(parsed_steps))
