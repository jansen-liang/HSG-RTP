from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Protocol

from evaluation.action_parser import ParseError, parse_local_action
from evaluation.goal_evaluator import build_goal_spec, evaluate_goal
from evaluation.rollout_evaluator import evaluate_action_sequence
from pipeline.utils.graph_utils import get_local_view
from pipeline.utils.state_manager import SceneGraphStateManager


class GridActionPredictor(Protocol):
    def predict(self, instruction: str, local_view: dict[str, Any]) -> tuple[str, float]: ...


@dataclass(frozen=True)
class GridRollout:
    success: bool
    actions: tuple[str, ...]
    finished: bool
    failure_type: str | None
    failure_message: str | None
    inference_time: float
    model_calls: int


def _canonical(actions: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for action in actions:
        try:
            normalized.append(parse_local_action(action).canonical)
        except ParseError:
            normalized.append(str(action).strip())
    return normalized


def _jaccard(predicted: list[str], reference: list[str]) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not reference_set:
        return 1.0 if not predicted_set else 0.0
    return len(predicted_set & reference_set) / len(predicted_set | reference_set)


def _lcs_ratio(predicted: list[str], reference: list[str]) -> float:
    if not reference:
        return 1.0 if not predicted else 0.0
    previous = [0] * (len(reference) + 1)
    for predicted_item in predicted:
        current = [0]
        for index, reference_item in enumerate(reference, start=1):
            if predicted_item == reference_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1] / len(reference)


def rollout_grid_policy(
    record: dict[str, Any],
    initial_scene: dict[str, Any],
    predictor: GridActionPredictor,
    max_steps: int | None = None,
) -> GridRollout:
    reference_actions = record.get("execution_summary", {}).get("subtasks", [])
    step_limit = max_steps or min(160, max(32, len(reference_actions) * 2 + 20))
    manager = SceneGraphStateManager(verbose=False)
    manager.load_initial_state(deepcopy(initial_scene))
    actions: list[str] = []
    inference_time = 0.0

    for _ in range(step_limit):
        current_room = manager.current_state["agent"]["position"]
        local_view = get_local_view(manager.current_state, current_room)
        raw_action, elapsed = predictor.predict(record["instruction"], local_view)
        inference_time += elapsed
        if raw_action == "finish":
            goal_success, failures = evaluate_goal(
                manager.current_state, build_goal_spec(record)
            )
            return GridRollout(
                success=goal_success,
                actions=tuple(actions),
                finished=True,
                failure_type=None if goal_success else "premature_finish",
                failure_message=None if goal_success else "; ".join(failures),
                inference_time=inference_time,
                model_calls=len(actions) + 1,
            )
        try:
            action = parse_local_action(raw_action).canonical
        except ParseError as error:
            return GridRollout(
                False,
                tuple(actions),
                False,
                "parse_error",
                str(error),
                inference_time,
                len(actions) + 1,
            )
        success, _, error = manager.execute_action(action)
        if not success:
            return GridRollout(
                False,
                tuple(actions),
                False,
                "execution_error",
                error,
                inference_time,
                len(actions) + 1,
            )
        actions.append(action)

    return GridRollout(
        False,
        tuple(actions),
        False,
        "step_limit",
        f"GRID did not finish within {step_limit} actions",
        inference_time,
        step_limit,
    )


def evaluate_grid_dataset(
    records: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    predictor: GridActionPredictor,
    max_steps: int | None = None,
    progress_callback: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for index, record in enumerate(records):
        scene = scenes[record["scene_name"]]
        reference = evaluate_action_sequence(
            record, scene, record.get("execution_summary", {}).get("subtasks", [])
        )
        if reference.success:
            rollout = rollout_grid_policy(record, scene, predictor, max_steps=max_steps)
        else:
            rollout = GridRollout(
                False,
                (),
                False,
                "invalid_benchmark_task",
                reference.failure_message,
                0.0,
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
            "inference_time": rollout.inference_time,
        }
        results.append(result)
        if progress_callback is not None:
            progress_callback(result)

    valid = [result for result in results if result["benchmark_valid"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in valid:
        grouped[result["task_type"]].append(result)

    def metrics(items: list[dict[str, Any]]) -> dict[str, float | int]:
        count = len(items)
        return {
            "tasks": count,
            "exec_sr": sum(item["exec_success"] for item in items) / count if count else 0.0,
            "local_jaccard": sum(item["local_jaccard"] for item in items) / count if count else 0.0,
            "local_lcs_ratio": sum(item["local_lcs_ratio"] for item in items) / count if count else 0.0,
            "avg_model_calls": sum(item["model_calls"] for item in items) / count if count else 0.0,
            "avg_inference_time": sum(item["inference_time"] for item in items) / count if count else 0.0,
        }

    summary = {
        "tasks": len(results),
        "valid_tasks": len(valid),
        "invalid_benchmark_tasks": len(results) - len(valid),
        "plan_sr": None,
        "plan_metric_reason": "GRID is a direct action-object policy and does not emit room-level plans.",
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
    return results, summary
