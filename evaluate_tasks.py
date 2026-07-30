#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.plan_evaluator import evaluate_global_plan
from evaluation.rollout_evaluator import evaluate_action_sequence
from pipeline.utils.action_planner import generate_global_plan
from pipeline.utils.scene_loader import load_scenes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_id(record: dict[str, Any]) -> str:
    identity = {
        "instruction": record.get("instruction"),
        "scene_name": record.get("scene_name"),
        "task_info": record.get("task_info"),
        "global_plan": record.get("execution_summary", {}).get("global_plan"),
        "subtasks": record.get("execution_summary", {}).get("subtasks"),
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_predictions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    predictions = {}
    for item in load_jsonl(path):
        identifier = item.get("task_id")
        if not identifier:
            raise ValueError("Every prediction record must contain task_id")
        predictions[identifier] = item
    return predictions


def reference_global_plan(record: dict[str, Any], scene: dict[str, Any]) -> list[str]:
    plan, _ = generate_global_plan(
        record.get("execution_summary", {}).get("subtasks", []),
        scene["rooms"],
        record.get("task_info", {}).get("type", "general"),
        initial_room=scene["agent"]["position"],
    )
    return plan


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    task_count = len(results)
    valid_results = [result for result in results if result["benchmark_valid"]]
    valid_task_count = len(valid_results)
    plan_successes = sum(result["plan_success"] for result in valid_results)
    execution_successes = sum(result["exec_success"] for result in valid_results)
    grouped = defaultdict(list)
    for result in valid_results:
        grouped[result["task_type"]].append(result)
    return {
        "tasks": task_count,
        "valid_tasks": valid_task_count,
        "invalid_benchmark_tasks": task_count - valid_task_count,
        "plan_sr": plan_successes / valid_task_count if valid_task_count else 0.0,
        "exec_sr": execution_successes / valid_task_count if valid_task_count else 0.0,
        "raw_plan_sr": sum(result["plan_success"] for result in results) / task_count if task_count else 0.0,
        "raw_exec_sr": sum(result["exec_success"] for result in results) / task_count if task_count else 0.0,
        "plan_successes": plan_successes,
        "exec_successes": execution_successes,
        "failure_types": dict(
            sorted(Counter(result["failure_type"] for result in results if result["failure_type"]).items())
        ),
        "by_task_type": {
            task_type: {
                "tasks": len(group_results),
                "plan_sr": sum(result["plan_success"] for result in group_results) / len(group_results),
                "exec_sr": sum(result["exec_success"] for result in group_results) / len(group_results),
            }
            for task_type, group_results in sorted(grouped.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task-level Plan SR and Strict Exec SR")
    parser.add_argument("dataset", type=Path, help="Task-level JSONL evaluation split")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional JSONL predictions keyed by task_id with global_plan and local_actions",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--stored-reference-plan",
        action="store_true",
        help="Use the stored global plan instead of rebuilding it from reference actions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.dataset)
    predictions = load_predictions(args.predictions)
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))
    results = []

    for index, record in enumerate(records):
        identifier = task_id(record)
        scene = scenes[record["scene_name"]]
        reference_result = evaluate_action_sequence(
            record, scene, record.get("execution_summary", {}).get("subtasks", [])
        )
        prediction = predictions.get(identifier)
        if args.predictions and prediction is None:
            plan = []
            actions = []
            missing_prediction = True
        else:
            missing_prediction = False
            if prediction is not None:
                plan = prediction.get("global_plan", [])
                actions = prediction.get("local_actions", [])
            else:
                plan = (
                    record.get("execution_summary", {}).get("global_plan", [])
                    if args.stored_reference_plan
                    else reference_global_plan(record, scene)
                )
                actions = record.get("execution_summary", {}).get("subtasks", [])

        plan_result = evaluate_global_plan(record, plan, scene)
        execution_result = evaluate_action_sequence(record, scene, actions)
        failure_type = "missing_prediction" if missing_prediction else execution_result.failure_type
        results.append(
            {
                "index": index,
                "task_id": identifier,
                "scene": record["scene_name"],
                "task_type": record.get("task_info", {}).get("type", "unknown"),
                "difficulty": record.get("task_info", {}).get("difficulty", "unknown"),
                "benchmark_valid": reference_result.success,
                "benchmark_error": reference_result.failure_message,
                "plan_success": plan_result.success,
                "exec_success": execution_result.success,
                "plan_errors": list(plan_result.errors),
                "failure_type": failure_type,
                "failure_message": execution_result.failure_message,
                "executed_actions": len(execution_result.actions),
            }
        )

    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "task_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
