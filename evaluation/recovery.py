from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecoveryConfig:
    max_local_retries: int = 2
    max_global_replans: int = 2
    max_initial_plan_retries: int = 2
    normalize_global_rooms: bool = True
    repair_global_task_plan: bool = True
    complete_global_task_coverage: bool = True
    task_semantic_repair_budget: int | None = None
    repair_stalled_local_action: bool = True


@dataclass(frozen=True)
class FailureFeedback:
    failure_type: str
    reason: str
    current_room: str
    retry_count: int
    action: str | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    retryable_same_action: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True)


def summarize_local_state(state: dict[str, Any]) -> dict[str, Any]:
    agent = state.get("agent", {})
    current_room = agent.get("position", "")
    room = state.get("rooms", {}).get(current_room, {})
    return {
        "current_room": current_room,
        "agent_state": agent.get("state"),
        "inventory": sorted(agent.get("inventory", {})),
        "visible_small_objects": sorted(room.get("small_objects", {})),
        "visible_large_objects": sorted(room.get("large_objects", {})),
        "neighbors": sorted(room.get("neighbor", [])),
    }


def build_local_recovery_instruction(
    instruction: str,
    feedback: FailureFeedback,
    forbidden_actions: list[str],
) -> str:
    retry_directive = (
        f"The failure is explicitly temporary. Retry exactly {feedback.action} now; "
        "do not substitute scan or another action."
        if feedback.retryable_same_action
        else "Do not repeat an action that remains invalid in the unchanged state."
    )
    return (
        f"{instruction}\n\n"
        "RECOVERY CONTEXT\n"
        f"Execution feedback: {feedback.to_prompt()}\n"
        f"Forbidden repeated actions: {json.dumps(forbidden_actions, ensure_ascii=False)}\n"
        f"{retry_directive}\n"
        "Generate exactly one feasible local action for the current scene graph and pending plan."
    )


def build_global_plan_instruction(
    instruction: str,
    valid_rooms: list[str],
    previous_error: str | None = None,
    retry_count: int = 0,
    task_info: dict[str, Any] | None = None,
) -> str:
    task_info = task_info or {}
    task_type = task_info.get("type", "unknown")
    parameters = task_info.get("parameters", {})
    constraints = {"type": task_type}
    for key in (
        "objects",
        "source_room",
        "source_rooms",
        "target_rooms",
        "intermediate_points",
        "end_room",
    ):
        value = parameters.get(key)
        if value not in (None, [], {}):
            constraints[key] = value

    sections = [instruction]
    if previous_error:
        sections.extend(
            [
                "GLOBAL PLAN REPAIR",
                f"Previous attempt {retry_count} was rejected: {previous_error}",
                f"Valid room IDs: {json.dumps(valid_rooms, ensure_ascii=False)}",
                "Correct the reported errors and regenerate the complete plan.",
            ]
        )
    sections.extend(
        [
            "GLOBAL TASK CONSTRAINTS",
            json.dumps(constraints, ensure_ascii=False, separators=(",", ":")),
            'Output mode must be exactly "global".',
            "Use exact scene IDs. Do not abbreviate, translate, or rename them.",
            "Delivery must explicitly pick every object in source_room before placing it in target_rooms. "
            "Tidying must organize every object in source_rooms. Guidance must visit intermediate_points and end_room in order.",
        ]
    )
    return "\n".join(sections)


def build_global_replan_instruction(
    instruction: str,
    feedback: FailureFeedback,
    completed: list[str],
    previous_pending: list[str],
    valid_rooms: list[str],
) -> str:
    return (
        f"{instruction}\n\n"
        "GLOBAL REPLANNING CONTEXT\n"
        f"Execution feedback: {feedback.to_prompt()}\n"
        f"Successfully completed actions: {json.dumps(completed, ensure_ascii=False)}\n"
        f"Previous pending plan: {json.dumps(previous_pending, ensure_ascii=False)}\n"
        'Output mode must be exactly "global".\n'
        f"Valid room IDs: {json.dumps(valid_rooms, ensure_ascii=False)}\n"
        "Every goto target must exactly match one valid room ID.\n"
        "Generate a revised global plan for only the unfinished portion of the task. "
        "Do not repeat completed work."
    )
