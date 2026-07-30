from collections import Counter, defaultdict
from typing import Any, Callable

from .perturbations import RolloutPerturbation
from .policies import StreamingModelPolicy
from .recovery import RecoveryConfig
from .rollout_evaluator import PlanningPolicy, evaluate_action_sequence, rollout_policy


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    for index, record in enumerate(records):
        scene = scenes[record["scene_name"]]
        reference_result = evaluate_action_sequence(
            record, scene, record.get("execution_summary", {}).get("subtasks", [])
        )
        if reference_result.success:
            execution = rollout_policy(
                record,
                scene,
                policy_factory(record),
                max_steps=max_steps,
                recovery_config=recovery_config,
                perturbations=(
                    perturbation_factory(record) if perturbation_factory else None
                ),
            )
        else:
            execution = reference_result
        results.append(
            {
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
            }
        )

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
        "by_task_type": {
            task_type: {
                "tasks": len(group_results),
                "plan_sr": sum(result["plan_success"] for result in group_results)
                / len(group_results),
                "exec_sr": sum(result["exec_success"] for result in group_results)
                / len(group_results),
            }
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = StreamingModelPolicy(model)
    return evaluate_policy_dataset(
        records,
        scenes,
        lambda _: policy,
        max_steps=max_steps,
        recovery_config=recovery_config,
        perturbation_factory=perturbation_factory,
    )
