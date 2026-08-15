import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.utils.action_planner import generate_global_plan
from pipeline.utils.graph_utils import get_global_view
from pipeline.utils.scene_loader import load_scenes
from evaluation.recovery import build_global_plan_instruction
from scripts.build_mixed_recovery_dataset import augment_task, _mark_sample


def rebuild_task(
    task: dict[str, Any],
    scene: dict[str, Any],
    global_replay_copies: int,
    global_repair_variants: int,
    local_recovery_samples: int,
    delivery_extra_replays: int,
    global_only: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    rebuilt = deepcopy(task)
    subtasks = list(task.get("execution_summary", {}).get("subtasks", []))
    global_plan, _ = generate_global_plan(
        subtasks,
        scene["rooms"],
        task.get("task_info", {}).get("type", "general"),
        initial_room=scene["agent"]["position"],
    )

    local_samples = [
        deepcopy(sample)
        for sample in task.get("streaming_samples", [])
        if sample.get("mode") == "local"
    ]
    initial_global = {
        "mode": "global",
        "context": f'instruction: {task.get("instruction", "")}',
        "target": global_plan,
        "completed": [],
        "pending": [],
        "scene_graph": get_global_view(scene),
        "metadata": {"augmentation_type": "corrected_global_reference"},
        "instruction_override": build_global_plan_instruction(
            task.get("instruction", ""),
            sorted(scene["rooms"]),
            task_info=task.get("task_info", {}),
        ),
    }
    final_state = task.get("execution_summary", {}).get("final_state", scene)
    final_global = {
        "mode": "global",
        "context": f'instruction: {task.get("instruction", "")}',
        "target": "finish",
        "completed": subtasks,
        "pending": [],
        "scene_graph": get_global_view(final_state),
        "metadata": {"augmentation_type": "normal_finish"},
    }
    rebuilt["execution_summary"]["global_plan"] = global_plan
    rebuilt["streaming_samples"] = [initial_global, *local_samples, final_global]
    rebuilt["sample_count"] = len(rebuilt["streaming_samples"])

    augmented, stats = augment_task(
        rebuilt,
        global_replay_copies=global_replay_copies,
        global_repair_variants=global_repair_variants,
        local_recovery_samples=local_recovery_samples,
    )
    if task.get("task_info", {}).get("type") == "delivery":
        for _ in range(delivery_extra_replays):
            augmented["streaming_samples"].append(
                _mark_sample(initial_global, "delivery_global_replay")
            )
        stats["delivery_global_replay"] = delivery_extra_replays
    else:
        stats["delivery_global_replay"] = 0
    augmented["sample_count"] = len(augmented["streaming_samples"])
    if global_only:
        augmented["streaming_samples"] = [
            sample
            for sample in augmented["streaming_samples"]
            if sample.get("mode") == "global"
        ]
        augmented["sample_count"] = len(augmented["streaming_samples"])
    augmented["mixed_augmentation"] = stats
    return augmented, stats


def build_dataset(args: argparse.Namespace) -> dict[str, int]:
    with args.input.open(encoding="utf-8") as source:
        tasks = [json.loads(line) for line in source if line.strip()]
    scenes = load_scenes(sorted({task["scene_name"] for task in tasks}))
    totals = {
        "tasks": 0,
        "normal": 0,
        "global_replay": 0,
        "global_repair": 0,
        "delivery_global_replay": 0,
        "local_temporary_recovery": 0,
        "local_invalid_action_recovery": 0,
        "total_samples": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for task in tasks:
            rebuilt, stats = rebuild_task(
                task,
                scenes[task["scene_name"]],
                args.global_replay_copies,
                args.global_repair_variants,
                args.local_recovery_samples,
                args.delivery_extra_replays,
                args.global_only,
            )
            destination.write(
                json.dumps(rebuilt, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            totals["tasks"] += 1
            totals["total_samples"] += len(rebuilt["streaming_samples"])
            for key in totals:
                if key not in {"tasks", "total_samples"}:
                    totals[key] += stats.get(key, 0)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild corrected global references while preserving local samples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-replay-copies", type=int, default=3)
    parser.add_argument("--global-repair-variants", type=int, default=2)
    parser.add_argument("--local-recovery-samples", type=int, default=2)
    parser.add_argument("--delivery-extra-replays", type=int, default=5)
    parser.add_argument("--global-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
