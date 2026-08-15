from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectGoal:
    object_id: str
    room_id: str
    relation: tuple[tuple[str, str], ...]
    state: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class GoalSpec:
    task_type: str
    agent_room: str | None
    object_goals: tuple[ObjectGoal, ...]


def find_object(state: dict[str, Any], object_id: str) -> tuple[str, dict[str, Any]] | None:
    for room_id, room in state.get("rooms", {}).items():
        for collection in ("small_objects", "large_objects", "items"):
            objects = room.get(collection, {})
            if object_id in objects:
                return room_id, objects[object_id]
    inventory = state.get("agent", {}).get("inventory", {})
    if object_id in inventory:
        return "agent_inventory", inventory[object_id]
    return None


def build_goal_spec(record: dict[str, Any]) -> GoalSpec:
    task_info = record.get("task_info", {})
    task_type = task_info.get("type", "unknown")
    parameters = task_info.get("parameters", {})
    final_state = record.get("execution_summary", {}).get("final_state")
    if not isinstance(final_state, dict):
        raise ValueError("Task record does not contain execution_summary.final_state")

    agent_room = parameters.get("end_room") if task_type == "guidance" else None
    object_goals = []
    requested_states = parameters.get("objects_goal_state", {}) or parameters.get("object_goals", {})
    for object_id in parameters.get("objects", []):
        located = find_object(final_state, object_id)
        if located is None:
            raise ValueError(f"Reference final state does not contain task object {object_id!r}")
        room_id, object_data = located
        required_state = requested_states.get(object_id, {})
        object_goals.append(
            ObjectGoal(
                object_id=object_id,
                room_id=room_id,
                relation=(),
                state=tuple(sorted(required_state.items())),
            )
        )

    return GoalSpec(
        task_type=task_type,
        agent_room=agent_room,
        object_goals=tuple(object_goals),
    )


def evaluate_goal(state: dict[str, Any], goal: GoalSpec) -> tuple[bool, list[str]]:
    failures = []
    if goal.agent_room is not None and state.get("agent", {}).get("position") != goal.agent_room:
        failures.append(
            f"agent expected in {goal.agent_room}, found {state.get('agent', {}).get('position')}"
        )

    for object_goal in goal.object_goals:
        located = find_object(state, object_goal.object_id)
        if located is None:
            failures.append(f"object {object_goal.object_id} is missing")
            continue
        room_id, object_data = located
        if room_id != object_goal.room_id:
            failures.append(
                f"object {object_goal.object_id} expected in {object_goal.room_id}, found in {room_id}"
            )
        relation = object_data.get("relation", {}) if isinstance(object_data, dict) else {}
        for relation_name, relation_target in object_goal.relation:
            if relation.get(relation_name) != relation_target:
                failures.append(
                    f"object {object_goal.object_id} expected {relation_name}={relation_target}"
                )
        object_state = object_data.get("state", {}) if isinstance(object_data, dict) else {}
        for state_name, state_value in object_goal.state:
            if object_state.get(state_name) != state_value:
                failures.append(
                    f"object {object_goal.object_id} expected state {state_name}={state_value}"
                )
    return not failures, failures
