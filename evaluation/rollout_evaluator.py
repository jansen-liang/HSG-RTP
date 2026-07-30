from copy import deepcopy
from dataclasses import dataclass
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


def _normalize_room_id(room_id: str) -> str:
    return "_".join(part for part in room_id.lower().replace("-", "_").split("_") if part)


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


def ground_global_plan(
    plan: list[str], state: dict[str, Any], record: dict[str, Any] | None = None
) -> tuple[list[str], list[dict[str, str]]]:
    """Conservatively ground room names before strict plan validation."""
    valid_rooms = sorted(state.get("rooms", {}))
    normalized_rooms = {
        room_id: _normalize_room_id(room_id) for room_id in valid_rooms
    }
    grounded_plan = []
    changes = []
    task_type = record.get("task_info", {}).get("type") if record else None

    for raw_step in plan:
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

    if not changes or record is None:
        return grounded_plan, changes

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

    return pruned_plan, changes


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
            plan, initial_scene, record
        )
        grounded_evaluation = evaluate_global_plan(
            record, grounded_plan, initial_scene
        )
        if grounding_changes and grounded_evaluation.success:
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
            return ExecutionEvaluation(
                False,
                tuple(completed),
                "recovery_exhausted",
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
