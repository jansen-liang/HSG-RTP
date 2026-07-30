import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.recovery import (
    FailureFeedback,
    build_global_plan_instruction,
    build_local_recovery_instruction,
)


GLOBAL_REPAIR_ERRORS = (
    "Expected mode 'global', got 'local'",
    "The previous plan referenced a room ID that does not exist",
    "The previous global plan used an invalid step format",
)


def _mark_sample(sample: dict[str, Any], augmentation_type: str) -> dict[str, Any]:
    augmented = deepcopy(sample)
    metadata = dict(augmented.get("metadata", {}))
    metadata["augmentation_type"] = augmentation_type
    augmented["metadata"] = metadata
    return augmented


def _select_evenly(samples: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not samples:
        return []
    if len(samples) <= count:
        return list(samples)
    if count == 1:
        return [samples[0]]
    indices = {
        round(index * (len(samples) - 1) / (count - 1))
        for index in range(count)
    }
    return [samples[index] for index in sorted(indices)]


def _local_observation(scene_graph: dict[str, Any]) -> dict[str, Any]:
    room = scene_graph.get("room", {})
    agent = scene_graph.get("agent", {})
    return {
        "current_room": scene_graph.get("current_room", agent.get("position", "")),
        "agent_state": agent.get("state"),
        "inventory": sorted(agent.get("inventory", {})),
        "visible_small_objects": sorted(room.get("small_objects", {})),
        "visible_large_objects": sorted(room.get("large_objects", {})),
        "neighbors": sorted(room.get("neighbor", [])),
    }


def augment_task(
    task: dict[str, Any],
    global_replay_copies: int,
    global_repair_variants: int,
    local_recovery_samples: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    augmented_task = deepcopy(task)
    original_samples = list(task.get("streaming_samples", []))
    global_samples = [sample for sample in original_samples if sample.get("mode") == "global"]
    local_samples = [sample for sample in original_samples if sample.get("mode") == "local"]
    additions = []
    stats = {
        "normal": len(original_samples),
        "global_replay": 0,
        "global_repair": 0,
        "local_temporary_recovery": 0,
        "local_invalid_action_recovery": 0,
    }

    if global_samples:
        global_sample = global_samples[0]
        for _ in range(global_replay_copies):
            additions.append(_mark_sample(global_sample, "global_semantic_replay"))
            stats["global_replay"] += 1

        valid_rooms = sorted(global_sample.get("scene_graph", {}).get("rooms", {}))
        for variant_index in range(global_repair_variants):
            error = GLOBAL_REPAIR_ERRORS[variant_index % len(GLOBAL_REPAIR_ERRORS)]
            repair_sample = _mark_sample(global_sample, "global_plan_repair")
            repair_sample["instruction_override"] = build_global_plan_instruction(
                task.get("instruction", ""),
                valid_rooms,
                previous_error=error,
                retry_count=1,
            )
            additions.append(repair_sample)
            stats["global_repair"] += 1

    selected_local_samples = _select_evenly(local_samples, local_recovery_samples)
    for sample_index, local_sample in enumerate(selected_local_samples):
        target_action = str(local_sample.get("target", ""))
        scene_graph = local_sample.get("scene_graph", {})
        current_room = scene_graph.get(
            "current_room", scene_graph.get("agent", {}).get("position", "")
        )
        if sample_index % 2 == 0:
            feedback = FailureFeedback(
                failure_type="execution_error",
                reason="Temporary low-level skill failure",
                current_room=current_room,
                retry_count=1,
                action=target_action,
                observation=_local_observation(scene_graph),
                retryable_same_action=True,
            )
            recovery_sample = _mark_sample(
                local_sample, "local_temporary_recovery"
            )
            forbidden_actions = []
            stats["local_temporary_recovery"] += 1
        else:
            invalid_action = "scan(__non_local_room__)"
            feedback = FailureFeedback(
                failure_type="execution_error",
                reason=(
                    f"Scan target __non_local_room__ is not local to {current_room}"
                ),
                current_room=current_room,
                retry_count=1,
                action=invalid_action,
                observation=_local_observation(scene_graph),
            )
            recovery_sample = _mark_sample(
                local_sample, "local_invalid_action_recovery"
            )
            forbidden_actions = [invalid_action]
            stats["local_invalid_action_recovery"] += 1

        recovery_sample["instruction_override"] = build_local_recovery_instruction(
            task.get("instruction", ""), feedback, forbidden_actions
        )
        additions.append(recovery_sample)

    augmented_task["streaming_samples"] = original_samples + additions
    augmented_task["sample_count"] = len(augmented_task["streaming_samples"])
    augmented_task["mixed_augmentation"] = stats
    return augmented_task, stats


def build_dataset(
    input_path: Path,
    output_path: Path,
    global_replay_copies: int = 2,
    global_repair_variants: int = 3,
    local_recovery_samples: int = 2,
) -> dict[str, Any]:
    totals = {
        "tasks": 0,
        "normal": 0,
        "global_replay": 0,
        "global_repair": 0,
        "local_temporary_recovery": 0,
        "local_invalid_action_recovery": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            task = json.loads(line)
            augmented_task, stats = augment_task(
                task,
                global_replay_copies,
                global_repair_variants,
                local_recovery_samples,
            )
            destination.write(
                json.dumps(augmented_task, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            totals["tasks"] += 1
            for key, value in stats.items():
                totals[key] += value

    totals["total_samples"] = sum(
        value for key, value in totals.items() if key not in {"tasks", "total_samples"}
    )
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-replay-copies", type=int, default=2)
    parser.add_argument("--global-repair-variants", type=int, default=3)
    parser.add_argument("--local-recovery-samples", type=int, default=2)
    args = parser.parse_args()
    totals = build_dataset(
        args.input,
        args.output,
        args.global_replay_copies,
        args.global_repair_variants,
        args.local_recovery_samples,
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
