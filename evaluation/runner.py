from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

from .action_parser import GlobalStep, ParseError, parse_global_step, parse_local_action
from .perturbations import RolloutPerturbation
from .policies import RoutedStreamingModelPolicy, StreamingModelPolicy
from .recovery import RecoveryConfig
from .rollout_evaluator import PlanningPolicy, evaluate_action_sequence, rollout_policy
from pipeline.utils.action_planner import generate_global_plan


def _canonical_global_step(step: GlobalStep) -> str:
    if step.action == "trans":
        action = f"trans from({step.arguments[0]}) to({step.arguments[1]})"
    else:
        action = f"{step.action}({', '.join(step.arguments)})"
    return f"goto({step.room}): {action}"


def _canonical_local_sequence(actions: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for action in actions:
        try:
            normalized.append(parse_local_action(action).canonical)
        except ParseError:
            normalized.append(str(action).strip())
    return normalized


def _canonical_global_sequence(steps: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for step in steps:
        try:
            normalized.append(_canonical_global_step(parse_global_step(step)))
        except ParseError:
            normalized.append(str(step).strip())
    return normalized


def _reference_global_sequence(
    record: dict[str, Any], scene: dict[str, Any]
) -> list[str]:
    actions = record.get("execution_summary", {}).get("subtasks", [])
    generated, _ = generate_global_plan(
        actions,
        scene.get("rooms", {}),
        "general",
        initial_room=scene.get("agent", {}).get("position"),
    )
    return _canonical_global_sequence(generated)


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


def _mean(results: list[dict[str, Any]], key: str) -> float:
    return sum(float(result[key]) for result in results) / len(results) if results else 0.0


def _metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tasks": len(results),
        "plan_sr": _mean(results, "plan_success"),
        "exec_sr": _mean(results, "exec_success"),
        "global_jaccard": _mean(results, "global_jaccard"),
        "global_lcs_ratio": _mean(results, "global_lcs_ratio"),
        "local_jaccard": _mean(results, "local_jaccard"),
        "local_lcs_ratio": _mean(results, "local_lcs_ratio"),
        "avg_model_calls": _mean(results, "model_calls"),
        "avg_input_tokens": _mean(results, "input_tokens"),
        "avg_output_tokens": _mean(results, "output_tokens"),
        "avg_total_tokens": _mean(results, "total_tokens"),
        "avg_inference_time": _mean(results, "inference_time"),
    }


def evaluate_policy_dataset(
    records: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    policy_factory: Callable[[dict[str, Any]], PlanningPolicy],
    max_steps: int | None = None,
    recovery_config: RecoveryConfig | None = None,
    perturbation_factory: Callable[
        [dict[str, Any]], list[RolloutPerturbation]
    ]
    | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for index, record in enumerate(records):
        scene = scenes[record["scene_name"]]
        policy = policy_factory(record)
        if hasattr(policy, "reset_usage"):
            policy.reset_usage()
        reference_result = evaluate_action_sequence(
            record, scene, record.get("execution_summary", {}).get("subtasks", [])
        )
        if reference_result.success:
            execution = rollout_policy(
                record,
                scene,
                policy,
                max_steps=max_steps,
                recovery_config=recovery_config,
                perturbations=(
                    perturbation_factory(record) if perturbation_factory else None
                ),
            )
        else:
            execution = reference_result
        usage = (
            policy.usage_summary()
            if hasattr(policy, "usage_summary")
            else {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "inference_time": 0.0,
            }
        )
        predicted_global = (
            [_canonical_global_step(step) for step in execution.plan_evaluation.parsed_steps]
            if execution.plan_evaluation is not None
            else []
        )
        reference_global = _reference_global_sequence(record, scene)
        predicted_local = _canonical_local_sequence(execution.actions)
        reference_local = _canonical_local_sequence(
            record.get("execution_summary", {}).get("subtasks", [])
        )
        result = {
                "index": index,
                "scene": record["scene_name"],
                "task_type": record.get("task_info", {}).get("type", "unknown"),
                "difficulty": record.get("task_info", {}).get("difficulty", "unknown"),
                "benchmark_valid": reference_result.success,
                "benchmark_error": reference_result.failure_message,
                "plan_success": bool(
                    execution.plan_evaluation and execution.plan_evaluation.success
                ),
                "exec_success": execution.success,
                "failure_type": execution.failure_type,
                "failure_message": execution.failure_message,
                "actions": list(execution.actions),
                "plan": list(execution.plan),
                "recovery_trace": list(execution.recovery_trace),
                "reference_plan": reference_global,
                "reference_actions": reference_local,
                "global_jaccard": _jaccard(predicted_global, reference_global),
                "global_lcs_ratio": _lcs_ratio(predicted_global, reference_global),
                "local_jaccard": _jaccard(predicted_local, reference_local),
                "local_lcs_ratio": _lcs_ratio(predicted_local, reference_local),
                **usage,
            }
        results.append(result)
        if progress_callback is not None:
            progress_callback(result)

    valid_results = [result for result in results if result["benchmark_valid"]]
    grouped = defaultdict(list)
    for result in valid_results:
        grouped[result["task_type"]].append(result)
    valid_count = len(valid_results)
    summary = {
        "tasks": len(results),
        "valid_tasks": valid_count,
        "invalid_benchmark_tasks": len(results) - valid_count,
        "plan_sr": (
            sum(result["plan_success"] for result in valid_results) / valid_count
            if valid_count
            else 0.0
        ),
        "exec_sr": (
            sum(result["exec_success"] for result in valid_results) / valid_count
            if valid_count
            else 0.0
        ),
        "failure_types": dict(
            sorted(
                Counter(
                    result["failure_type"]
                    for result in valid_results
                    if result["failure_type"]
                ).items()
            )
        ),
        **{
            key: value
            for key, value in _metric_summary(valid_results).items()
            if key != "tasks"
        },
        "by_task_type": {
            task_type: _metric_summary(group_results)
            for task_type, group_results in sorted(grouped.items())
        },
    }
    return results, summary


def evaluate_streaming_model(
    model: Any,
    records: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    max_steps: int | None = None,
    recovery_config: RecoveryConfig | None = None,
    perturbation_factory: Callable[
        [dict[str, Any]], list[RolloutPerturbation]
    ]
    | None = None,
    global_generation_config: dict[str, Any] | None = None,
    local_generation_config: dict[str, Any] | None = None,
    static_scene: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = StreamingModelPolicy(
        model,
        global_generation_config=global_generation_config,
        local_generation_config=local_generation_config,
        static_scene=static_scene,
    )
    return evaluate_policy_dataset(
        records,
        scenes,
        lambda _: policy,
        max_steps=max_steps,
        recovery_config=recovery_config,
        perturbation_factory=perturbation_factory,
        progress_callback=progress_callback,
    )


def evaluate_routed_streaming_models(
    global_model: Any,
    local_model: Any,
    records: list[dict[str, Any]],
    scenes: dict[str, dict[str, Any]],
    max_steps: int | None = None,
    recovery_config: RecoveryConfig | None = None,
    global_generation_config: dict[str, Any] | None = None,
    local_generation_config: dict[str, Any] | None = None,
    static_scene: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = RoutedStreamingModelPolicy(
        global_model,
        local_model,
        global_generation_config=global_generation_config,
        local_generation_config=local_generation_config,
        static_scene=static_scene,
    )
    return evaluate_policy_dataset(
        records,
        scenes,
        lambda _: policy,
        max_steps=max_steps,
        recovery_config=recovery_config,
        progress_callback=progress_callback,
    )
