from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Protocol

from evaluation.action_parser import ParseError, parse_local_action
from evaluation.goal_evaluator import build_goal_spec, evaluate_goal
from evaluation.grid_baseline import _canonical, _jaccard, _lcs_ratio
from evaluation.rollout_evaluator import evaluate_action_sequence
from pipeline.utils.graph_utils import get_local_view
from pipeline.utils.state_manager import SceneGraphStateManager


TERMINATION_SKILL = "finish"


class OptionScoringBackend(Protocol):
    calls: list[dict[str, Any]]

    def reset_usage(self) -> None: ...

    def score_options(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, float]: ...


def _held_object(agent_state: Any) -> str | None:
    state = str(agent_state)
    prefix = "holding-"
    return state[len(prefix) :] if state.startswith(prefix) else None


def enumerate_saycan_skills(
    local_view: dict[str, Any],
    room_ids: list[str],
    action_history: list[str],
) -> dict[str, float]:
    room = local_view["room"]
    current_room = str(local_view["current_room"])
    agent = local_view.get("agent", {})
    small_objects = room.get("small_objects", {})
    large_objects = room.get("large_objects", {})
    history = set(action_history)
    skills: dict[str, float] = {}

    neighbors = set(room.get("neighbor", []))
    for room_id in sorted(room_ids):
        skills[f"goto({room_id})"] = float(
            room_id != current_room and room_id in neighbors
        )

    scan_targets = [current_room, *small_objects, *large_objects]
    for target in scan_targets:
        action = f"scan({target})"
        skills[action] = float(action not in history)

    held_object = _held_object(agent.get("state"))
    hand_free = held_object is None
    for object_id, object_info in sorted(small_objects.items()):
        affordances = object_info.get("affordance", []) if isinstance(object_info, dict) else []
        skills[f"pick({object_id})"] = float(hand_free and "pick" in affordances)
        pressed = (
            object_info.get("state") == "pressed"
            if isinstance(object_info, dict)
            else False
        )
        skills[f"press({object_id})"] = float(
            "press" in affordances and not pressed
        )

    if held_object is not None:
        skills[f"place({held_object}, floor)"] = 1.0
        for surface_id, surface_info in sorted(large_objects.items()):
            relation = (
                surface_info.get("placement_relation", "on")
                if isinstance(surface_info, dict)
                else "on"
            )
            skills[f"place({held_object}, {surface_id})"] = float(
                relation in {"on", "in"}
            )

    for condition in ("elevator_down_clear", "elevator_up_clear"):
        action = f"wait({condition})"
        skills[action] = float(
            current_room.startswith("elevator_") and action not in history
        )

    skills[TERMINATION_SKILL] = 0.2
    return skills


class SayCanAdaptationPredictor:
    def __init__(self, backend: OptionScoringBackend) -> None:
        self.backend = backend
        self.room_ids: list[str] = []
        self.history: list[str] = []

    def reset(self, record: dict[str, Any], initial_scene: dict[str, Any]) -> None:
        self.backend.reset_usage()
        self.room_ids = sorted(initial_scene.get("rooms", {}))
        self.history = []

    def predict(self, instruction: str, local_view: dict[str, Any]) -> tuple[str, float]:
        skills = enumerate_saycan_skills(local_view, self.room_ids, self.history)
        feasible = {
            action: affordance
            for action, affordance in skills.items()
            if affordance > 0
        }
        system_prompt = (
            "Select the next robot skill for the instruction. The available skills use "
            "exact entity IDs and are filtered by a robot affordance model. Score each "
            "candidate as the next useful step; finish means the instruction is complete. "
            "Respond with exactly one available skill and nothing else."
        )
        user_prompt = json.dumps(
            {
                "instruction": instruction,
                "current_observation": local_view,
                "executed_skills": self.history[-24:],
                "available_skills": list(feasible),
                "next_skill": None,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        before = len(self.backend.calls)
        language_scores = self.backend.score_options(
            "saycan_skill_scoring",
            system_prompt,
            user_prompt,
            list(feasible),
        )
        combined_scores = {
            action: language_scores[action] + math.log(affordance)
            for action, affordance in feasible.items()
        }
        selected = max(combined_scores, key=combined_scores.get)
        self.history.append(selected)
        call = self.backend.calls[before]
        return selected, float(call["inference_time"])


@dataclass(frozen=True)
class SayCanRollout:
    success: bool
    actions: tuple[str, ...]
    finished: bool
    failure_type: str | None
    failure_message: str | None
    inference_time: float
    model_calls: int
    input_tokens: int
    output_tokens: int


def rollout_saycan_policy(
    record: dict[str, Any],
    initial_scene: dict[str, Any],
    predictor: SayCanAdaptationPredictor,
    max_steps: int | None = None,
) -> SayCanRollout:
    reference_actions = record.get("execution_summary", {}).get("subtasks", [])
    step_limit = max_steps or max(10, 2 * len(reference_actions))
    manager = SceneGraphStateManager(verbose=False)
    manager.load_initial_state(deepcopy(initial_scene))
    predictor.reset(record, initial_scene)
    actions: list[str] = []
    inference_time = 0.0

    for _ in range(step_limit):
        current_room = manager.current_state["agent"]["position"]
        local_view = get_local_view(manager.current_state, current_room)
        raw_action, elapsed = predictor.predict(record["instruction"], local_view)
        inference_time += elapsed
        if raw_action == TERMINATION_SKILL:
            goal_success, failures = evaluate_goal(
                manager.current_state, build_goal_spec(record)
            )
            return SayCanRollout(
                goal_success,
                tuple(actions),
                True,
                None if goal_success else "premature_finish",
                None if goal_success else "; ".join(failures),
                inference_time,
                len(predictor.backend.calls),
                sum(call["input_tokens"] for call in predictor.backend.calls),
                sum(call["output_tokens"] for call in predictor.backend.calls),
            )
        try:
            action = parse_local_action(raw_action).canonical
        except ParseError as error:
            return SayCanRollout(
                False,
                tuple(actions),
                False,
                "parse_error",
                str(error),
                inference_time,
                len(predictor.backend.calls),
                sum(call["input_tokens"] for call in predictor.backend.calls),
                sum(call["output_tokens"] for call in predictor.backend.calls),
            )
        success, _, error = manager.execute_action(action)
        if not success:
            return SayCanRollout(
                False,
                tuple(actions),
                False,
                "execution_error",
                error,
                inference_time,
                len(predictor.backend.calls),
                sum(call["input_tokens"] for call in predictor.backend.calls),
                sum(call["output_tokens"] for call in predictor.backend.calls),
            )
        actions.append(action)

    return SayCanRollout(
        False,
        tuple(actions),
        False,
        "step_limit",
        f"SayCan did not finish within {step_limit} actions",
        inference_time,
        len(predictor.backend.calls),
        sum(call["input_tokens"] for call in predictor.backend.calls),
        sum(call["output_tokens"] for call in predictor.backend.calls),
    )


def evaluate_saycan_dataset(
    records: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    predictor: SayCanAdaptationPredictor,
    max_steps: int | None = None,
    progress_callback: Any | None = None,
    start_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for index, record in enumerate(records, start=start_index):
        scene = scenes[record["scene_name"]]
        reference = evaluate_action_sequence(
            record, scene, record.get("execution_summary", {}).get("subtasks", [])
        )
        if reference.success:
            rollout = rollout_saycan_policy(
                record, scene, predictor, max_steps=max_steps
            )
        else:
            rollout = SayCanRollout(
                False,
                (),
                False,
                "invalid_benchmark_task",
                reference.failure_message,
                0.0,
                0,
                0,
                0,
            )
        predicted_actions = _canonical(rollout.actions)
        reference_actions = _canonical(
            record.get("execution_summary", {}).get("subtasks", [])
        )
        result = {
            "index": index,
            "scene": record["scene_name"],
            "task_type": record.get("task_info", {}).get("type", "unknown"),
            "difficulty": record.get("task_info", {}).get("difficulty", "unknown"),
            "benchmark_valid": reference.success,
            "benchmark_error": reference.failure_message,
            "exec_success": rollout.success,
            "finished": rollout.finished,
            "failure_type": rollout.failure_type,
            "failure_message": rollout.failure_message,
            "actions": predicted_actions,
            "reference_actions": reference_actions,
            "local_jaccard": _jaccard(predicted_actions, reference_actions),
            "local_lcs_ratio": _lcs_ratio(predicted_actions, reference_actions),
            "model_calls": rollout.model_calls,
            "input_tokens": rollout.input_tokens,
            "output_tokens": rollout.output_tokens,
            "total_tokens": rollout.input_tokens + rollout.output_tokens,
            "inference_time": rollout.inference_time,
        }
        results.append(result)
        if progress_callback is not None:
            progress_callback(result)

    return results, summarize_saycan_results(results)


def summarize_saycan_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [result for result in results if result["benchmark_valid"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in valid:
        grouped[result["task_type"]].append(result)

    def metrics(items: list[dict[str, Any]]) -> dict[str, float | int]:
        count = len(items)
        def average(name: str) -> float:
            return (
                sum(float(item[name]) for item in items) / count if count else 0.0
            )

        return {
            "tasks": count,
            "exec_sr": (
                sum(item["exec_success"] for item in items) / count if count else 0.0
            ),
            "local_jaccard": average("local_jaccard"),
            "local_lcs_ratio": average("local_lcs_ratio"),
            "avg_model_calls": average("model_calls"),
            "avg_input_tokens": average("input_tokens"),
            "avg_output_tokens": average("output_tokens"),
            "avg_total_tokens": average("total_tokens"),
            "avg_inference_time": average("inference_time"),
        }

    summary = {
        "tasks": len(results),
        "valid_tasks": len(valid),
        "invalid_benchmark_tasks": len(results) - len(valid),
        "plan_sr": None,
        "plan_metric_reason": (
            "SayCan selects executable skills directly and does not emit room-level plans."
        ),
        **{key: value for key, value in metrics(valid).items() if key != "tasks"},
        "failure_types": dict(
            sorted(
                Counter(
                    result["failure_type"]
                    for result in valid
                    if result["failure_type"]
                ).items()
            )
        ),
        "by_task_type": {
            task_type: metrics(items) for task_type, items in sorted(grouped.items())
        },
    }
    return summary
