#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.recovery import RecoveryConfig
from evaluation.runner import evaluate_routed_streaming_models
from pipeline.utils.scene_loader import load_scenes
from scripts.evaluate_task_checkpoint import load_model, load_records, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate separate global and local streaming checkpoints."
    )
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "no_hsge",
            "no_local_graph",
            "no_object_tokens",
            "no_global_topology",
            "no_context",
            "no_graph_updates_history",
            "no_dynamic_update",
        ),
        default="full",
        help="Model/input ablation applied consistently to both routed checkpoints.",
    )
    parser.add_argument("--enable-recovery", action="store_true")
    controller_group = parser.add_mutually_exclusive_group()
    controller_group.add_argument(
        "--minimal-controller",
        action="store_true",
        help=(
            "Keep validation, alias/navigation grounding, and bounded model retries, "
            "but disable task-plan completion, room correction, and scripted stall actions."
        ),
    )
    controller_group.add_argument(
        "--lightweight-controller",
        action="store_true",
        help=(
            "Keep execution grounding and bounded retries, but cap global task-plan "
            "repair at four semantic edits instead of rebuilding arbitrary predictions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Routed evaluation requires two visible CUDA devices")
    set_seed(args.seed)
    records = load_records(args.dataset)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    records = records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))

    global_model = load_model(
        args.model_path,
        args.global_checkpoint,
        torch.device("cuda:0"),
        args.ablation,
    )
    local_model = load_model(
        args.model_path,
        args.local_checkpoint,
        torch.device("cuda:1"),
        args.ablation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "task_results.jsonl"
    results_path.write_text("", encoding="utf-8")
    completed = 0

    def record_progress(result: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[{completed}/{len(records)}] index={result['index']} "
            f"type={result['task_type']} plan={int(result['plan_success'])} "
            f"exec={int(result['exec_success'])} calls={result['model_calls']} "
            f"time={result['inference_time']:.2f}s",
            flush=True,
        )

    if args.lightweight_controller:
        recovery_config = RecoveryConfig(
            task_semantic_repair_budget=4,
        )
    elif args.minimal_controller:
        recovery_config = RecoveryConfig(
            normalize_global_rooms=False,
            repair_global_task_plan=False,
            complete_global_task_coverage=False,
            repair_stalled_local_action=False,
        )
    elif args.enable_recovery:
        recovery_config = RecoveryConfig()
    else:
        recovery_config = RecoveryConfig(
            max_local_retries=0,
            max_global_replans=0,
            max_initial_plan_retries=0,
        )
    _, summary = evaluate_routed_streaming_models(
        global_model,
        local_model,
        records,
        scenes,
        max_steps=args.max_steps,
        recovery_config=recovery_config,
        global_generation_config={"do_sample": False},
        local_generation_config={"do_sample": False},
        static_scene=args.ablation in (
            "no_dynamic_update",
            "no_graph_updates_history",
        ),
        progress_callback=record_progress,
    )
    summary.update(
        {
            "global_checkpoint": str(args.global_checkpoint),
            "local_checkpoint": str(args.local_checkpoint),
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "start_index": args.start_index,
            "seed": args.seed,
            "local_decoding": "deterministic",
            "recovery_enabled": bool(
                args.enable_recovery
                or args.minimal_controller
                or args.lightweight_controller
            ),
            "controller_profile": (
                "lightweight"
                if args.lightweight_controller
                else "minimal"
                if args.minimal_controller
                else "full"
                if args.enable_recovery
                else "retry_disabled"
            ),
            "routing": "global/local",
            "ablation": args.ablation,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
