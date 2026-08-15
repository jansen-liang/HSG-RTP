#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.delta_baseline import DeltaAdaptationPolicy, DeltaPDDLPlanner
from evaluation.delta_upstream import (
    UpstreamDeltaAdaptationPolicy,
    UpstreamDeltaPDDLPlanner,
)
from evaluation.external_baselines import HuggingFaceJSONBackend, SayPlanAdaptationPolicy
from evaluation.recovery import RecoveryConfig
from evaluation.runner import evaluate_policy_dataset
from pipeline.utils.scene_loader import load_scenes


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def raw_recovery_config() -> RecoveryConfig:
    return RecoveryConfig(
        max_local_retries=0,
        max_global_replans=0,
        max_initial_plan_retries=0,
        normalize_global_rooms=False,
        repair_global_task_plan=False,
        complete_global_task_coverage=False,
        repair_stalled_local_action=False,
    )


def mean(results: list[dict[str, Any]], key: str) -> float:
    values = [float(result.get(key, 0)) for result in results if result["benchmark_valid"]]
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adapted external planning methods")
    parser.add_argument(
        "--method", choices=("sayplan", "delta", "delta_upstream"), required=True
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-search-attempts", type=int, default=2)
    parser.add_argument("--max-plan-revisions", type=int, default=4)
    parser.add_argument("--max-decomposition-attempts", type=int, default=2)
    parser.add_argument("--max-problem-attempts", type=int, default=2)
    parser.add_argument(
        "--fast-downward",
        type=Path,
        default=REPO_ROOT / "third_party/DELTA/downward/fast-downward.py",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.dataset)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    records = records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))
    backend = HuggingFaceJSONBackend(
        args.model_path,
        args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "task_results.jsonl"
    results_path.write_text("", encoding="utf-8")
    completed = 0

    delta_planner = DeltaPDDLPlanner(args.fast_downward)
    delta_root = REPO_ROOT / "third_party/DELTA"
    upstream_delta_planner = (
        UpstreamDeltaPDDLPlanner(delta_root)
        if args.method == "delta_upstream"
        else None
    )

    def policy_factory(record: dict[str, Any]):
        if args.method == "sayplan":
            return SayPlanAdaptationPolicy(
                backend,
                record,
                scenes[record["scene_name"]],
                max_search_attempts=args.max_search_attempts,
                max_plan_revisions=args.max_plan_revisions,
            )
        if args.method == "delta_upstream":
            if upstream_delta_planner is None:
                raise RuntimeError("Upstream DELTA planner was not initialized")
            return UpstreamDeltaAdaptationPolicy(
                backend,
                record,
                scenes[record["scene_name"]],
                upstream_delta_planner,
                delta_root,
                max_problem_attempts=args.max_problem_attempts,
                max_decomposition_attempts=args.max_decomposition_attempts,
            )
        return DeltaAdaptationPolicy(
            backend,
            record,
            scenes[record["scene_name"]],
            delta_planner,
            max_decomposition_attempts=args.max_decomposition_attempts,
        )

    def record_progress(result: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[{completed}/{len(records)}] index={result['index']} "
            f"plan={int(result['plan_success'])} exec={int(result['exec_success'])} "
            f"search={result.get('search_calls', 0)} "
            f"revisions={result.get('plan_revisions', 0)} "
            f"decompose={result.get('decomposition_calls', 0)}",
            flush=True,
        )

    results, summary = evaluate_policy_dataset(
        records,
        scenes,
        policy_factory,
        recovery_config=raw_recovery_config(),
        progress_callback=record_progress,
    )
    summary.update(
        {
            "method": args.method,
            "method_label": (
                "paper-based reimplementation"
                if args.method == "sayplan"
                else "upstream-code partial adaptation (fixed benchmark problem builder)"
                if args.method == "delta_upstream"
                else "paper-based adaptation (Fast Downward planner)"
            ),
            "backbone": args.model_name,
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "start_index": args.start_index,
            "seed": args.seed,
            "controller_profile": "raw_external",
            "max_search_attempts": args.max_search_attempts,
            "max_plan_revisions": args.max_plan_revisions,
            "max_decomposition_attempts": args.max_decomposition_attempts,
            "max_problem_attempts": args.max_problem_attempts,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "avg_search_calls": mean(results, "search_calls"),
            "avg_plan_revisions": mean(results, "plan_revisions"),
            "avg_local_calls": mean(results, "local_calls"),
            "avg_decomposition_calls": mean(results, "decomposition_calls"),
            "avg_problem_generation_calls": mean(
                results, "problem_generation_calls"
            ),
            "avg_planner_time": mean(results, "planner_time"),
            "avg_planner_failures": mean(results, "planner_failures"),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
