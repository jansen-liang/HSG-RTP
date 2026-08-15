#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASK_FIELDS = (
    "instruction",
    "task_info",
    "execution_summary",
    "scene_name",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def benchmark_hash(path: Path) -> str:
    records = [
        {field: record.get(field) for field in TASK_FIELDS}
        for record in load_jsonl(path)
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_dataset(summary_path: Path, dataset: str) -> Path:
    path = Path(dataset)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, summary_path.parents[2] / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ValueError(f"Dataset does not exist: {dataset}")


def require_equal(summary: dict[str, Any], key: str, expected: Any) -> None:
    actual = summary.get(key)
    if actual != expected:
        raise ValueError(f"{key}: expected {expected!r}, found {actual!r}")


def audit_summary(
    summary_path: Path,
    reference_hash: str,
    expected_ablation: str | None = None,
    grid: bool = False,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_equal(summary, "tasks", 70)
    require_equal(summary, "valid_tasks", 66)
    require_equal(summary, "invalid_benchmark_tasks", 4)
    dataset = resolve_dataset(summary_path, str(summary.get("dataset", "")))
    actual_hash = benchmark_hash(dataset)
    if actual_hash != reference_hash:
        raise ValueError(
            f"dataset task-level hash mismatch: {dataset} has {actual_hash}, "
            f"expected {reference_hash}"
        )

    if grid:
        require_equal(summary, "method", "GRID")
        require_equal(summary, "controller_profile", "strict_direct_action_rollout")
        require_equal(summary, "plan_sr", None)
        if not summary.get("plan_metric_reason"):
            raise ValueError("GRID summary must explain why Plan SR is not applicable")
    else:
        require_equal(summary, "seed", 42)
        require_equal(summary, "controller_profile", "lightweight")
        require_equal(summary, "routing", "global/local")
        if expected_ablation is not None:
            require_equal(summary, "ablation", expected_ablation)

    return {
        "summary": str(summary_path.resolve()),
        "dataset": str(dataset),
        "task_hash": actual_hash,
        "plan_sr": summary.get("plan_sr"),
        "exec_sr": summary.get("exec_sr"),
        "global_jaccard": summary.get("global_jaccard"),
        "global_lcs_ratio": summary.get("global_lcs_ratio"),
        "local_jaccard": summary.get("local_jaccard"),
        "local_lcs_ratio": summary.get("local_lcs_ratio"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "avg_inference_time": summary.get("avg_inference_time"),
        "by_task_type": summary.get("by_task_type", {}),
    }


def audit_delta_summary(
    summary_path: Path,
    reference_hash: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_equal(summary, "tasks", 70)
    require_equal(summary, "valid_tasks", 66)
    require_equal(summary, "invalid_benchmark_tasks", 4)
    require_equal(summary, "method", "delta_upstream")
    require_equal(summary, "controller_profile", "raw_external")
    require_equal(summary, "seed", 42)
    require_equal(summary, "max_input_tokens", 8192)
    require_equal(summary, "max_new_tokens", 2048)
    dataset = resolve_dataset(summary_path, str(summary.get("dataset", "")))
    actual_hash = benchmark_hash(dataset)
    if actual_hash != reference_hash:
        raise ValueError(
            f"dataset task-level hash mismatch: {dataset} has {actual_hash}, "
            f"expected {reference_hash}"
        )
    for key in (
        "avg_problem_generation_calls",
        "avg_decomposition_calls",
        "avg_planner_time",
    ):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key}: expected a positive value, found {value!r}")

    return {
        "summary": str(summary_path.resolve()),
        "dataset": str(dataset),
        "task_hash": actual_hash,
        "method_label": summary.get("method_label"),
        "plan_sr": summary.get("plan_sr"),
        "exec_sr": summary.get("exec_sr"),
        "global_jaccard": summary.get("global_jaccard"),
        "global_lcs_ratio": summary.get("global_lcs_ratio"),
        "local_jaccard": summary.get("local_jaccard"),
        "local_lcs_ratio": summary.get("local_lcs_ratio"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "avg_inference_time": summary.get("avg_inference_time"),
        "avg_problem_generation_calls": summary.get(
            "avg_problem_generation_calls"
        ),
        "avg_decomposition_calls": summary.get("avg_decomposition_calls"),
        "avg_planner_time": summary.get("avg_planner_time"),
        "by_task_type": summary.get("by_task_type", {}),
    }


def audit_sayplan_summary(
    summary_path: Path,
    reference_hash: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_equal(summary, "tasks", 70)
    require_equal(summary, "valid_tasks", 66)
    require_equal(summary, "invalid_benchmark_tasks", 4)
    require_equal(summary, "method", "sayplan")
    require_equal(summary, "method_label", "paper-based reimplementation")
    require_equal(summary, "controller_profile", "raw_external")
    require_equal(summary, "seed", 42)
    require_equal(summary, "max_input_tokens", 4096)
    require_equal(summary, "max_new_tokens", 448)
    require_equal(summary, "max_search_attempts", 2)
    require_equal(summary, "max_plan_revisions", 4)
    dataset = resolve_dataset(summary_path, str(summary.get("dataset", "")))
    actual_hash = benchmark_hash(dataset)
    if actual_hash != reference_hash:
        raise ValueError(
            f"dataset task-level hash mismatch: {dataset} has {actual_hash}, "
            f"expected {reference_hash}"
        )
    for key in ("avg_search_calls", "avg_plan_revisions"):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{key}: expected a positive value, found {value!r}")
    return {
        "summary": str(summary_path.resolve()),
        "dataset": str(dataset),
        "task_hash": actual_hash,
        "method_label": summary.get("method_label"),
        "plan_sr": summary.get("plan_sr"),
        "exec_sr": summary.get("exec_sr"),
        "global_jaccard": summary.get("global_jaccard"),
        "global_lcs_ratio": summary.get("global_lcs_ratio"),
        "local_jaccard": summary.get("local_jaccard"),
        "local_lcs_ratio": summary.get("local_lcs_ratio"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "avg_inference_time": summary.get("avg_inference_time"),
        "avg_search_calls": summary.get("avg_search_calls"),
        "avg_plan_revisions": summary.get("avg_plan_revisions"),
        "by_task_type": summary.get("by_task_type", {}),
    }


def audit_saycan_summary(
    summary_path: Path,
    reference_hash: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_equal(summary, "tasks", 70)
    require_equal(summary, "valid_tasks", 66)
    require_equal(summary, "invalid_benchmark_tasks", 4)
    require_equal(summary, "method", "SayCan")
    require_equal(
        summary,
        "method_label",
        "official-code adaptation (notebook; symbolic affordance adapter)",
    )
    require_equal(summary, "controller_profile", "strict_direct_skill_rollout")
    require_equal(summary, "plan_sr", None)
    require_equal(summary, "seed", 42)
    require_equal(summary, "max_input_tokens", 4096)
    require_equal(summary, "affordance_model", "symbolic action preconditions")
    require_equal(
        summary,
        "official_commit",
        "a0080d35561b0a02504bf303edc4ba7f8011b5f8",
    )
    if not summary.get("plan_metric_reason"):
        raise ValueError("SayCan summary must explain why Plan SR is not applicable")
    dataset = resolve_dataset(summary_path, str(summary.get("dataset", "")))
    actual_hash = benchmark_hash(dataset)
    if actual_hash != reference_hash:
        raise ValueError(
            f"dataset task-level hash mismatch: {dataset} has {actual_hash}, "
            f"expected {reference_hash}"
        )
    return {
        "summary": str(summary_path.resolve()),
        "dataset": str(dataset),
        "task_hash": actual_hash,
        "method_label": summary.get("method_label"),
        "plan_sr": None,
        "exec_sr": summary.get("exec_sr"),
        "global_jaccard": None,
        "global_lcs_ratio": None,
        "local_jaccard": summary.get("local_jaccard"),
        "local_lcs_ratio": summary.get("local_lcs_ratio"),
        "avg_total_tokens": summary.get("avg_total_tokens"),
        "avg_inference_time": summary.get("avg_inference_time"),
        "by_task_type": summary.get("by_task_type", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit experiment summaries before inserting results into the paper."
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=Path("pipeline/output/task_split/test_corrected_streaming.jsonl"),
    )
    parser.add_argument(
        "--matched",
        action="append",
        default=[],
        metavar="ABLATION=SUMMARY",
    )
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--delta", type=Path)
    parser.add_argument("--sayplan", type=Path)
    parser.add_argument("--saycan", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_hash = benchmark_hash(args.reference_dataset)
    audited: dict[str, Any] = {
        "reference_dataset": str(args.reference_dataset.resolve()),
        "task_hash": reference_hash,
        "matched": {},
    }
    for item in args.matched:
        if "=" not in item:
            raise ValueError(f"Invalid --matched value: {item!r}")
        ablation, raw_path = item.split("=", 1)
        audited["matched"][ablation] = audit_summary(
            Path(raw_path), reference_hash, expected_ablation=ablation
        )
    if args.grid is not None:
        audited["grid"] = audit_summary(args.grid, reference_hash, grid=True)
    if args.delta is not None:
        audited["delta"] = audit_delta_summary(args.delta, reference_hash)
    if args.sayplan is not None:
        audited["sayplan"] = audit_sayplan_summary(args.sayplan, reference_hash)
    if args.saycan is not None:
        audited["saycan"] = audit_saycan_summary(args.saycan, reference_hash)

    output = json.dumps(audited, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
