import json
from pathlib import Path

import pytest

from scripts.audit_paper_results import (
    audit_delta_summary,
    audit_saycan_summary,
    audit_sayplan_summary,
    audit_summary,
    benchmark_hash,
)


def write_dataset(path: Path, extra: bool = False) -> None:
    records = []
    for index in range(70):
        record = {
            "instruction": f"task {index}",
            "task_info": {"type": "guidance"},
            "execution_summary": {"subtasks": [f"goto(room_{index})"]},
            "scene_name": "scene",
        }
        if extra:
            record["streaming_samples"] = [{"mode": "global"}]
        records.append(record)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_benchmark_hash_ignores_streaming_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_dataset(first)
    write_dataset(second, extra=True)
    assert benchmark_hash(first) == benchmark_hash(second)


def test_audit_rejects_wrong_controller(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    write_dataset(dataset)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tasks": 70,
                "valid_tasks": 66,
                "invalid_benchmark_tasks": 4,
                "dataset": str(dataset),
                "seed": 42,
                "controller_profile": "full",
                "routing": "global/local",
                "ablation": "no_hsge",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controller_profile"):
        audit_summary(
            summary_path,
            benchmark_hash(dataset),
            expected_ablation="no_hsge",
        )


def test_delta_audit_requires_exercised_upstream_stages(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    write_dataset(dataset)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tasks": 70,
                "valid_tasks": 66,
                "invalid_benchmark_tasks": 4,
                "dataset": str(dataset),
                "method": "delta_upstream",
                "controller_profile": "raw_external",
                "seed": 42,
                "max_input_tokens": 8192,
                "max_new_tokens": 2048,
                "avg_problem_generation_calls": 1.0,
                "avg_decomposition_calls": 0.0,
                "avg_planner_time": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="avg_decomposition_calls"):
        audit_delta_summary(summary_path, benchmark_hash(dataset))


def test_sayplan_audit_rejects_nonmatching_task_file(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"
    different = tmp_path / "different.jsonl"
    write_dataset(reference)
    write_dataset(different)
    records = [json.loads(line) for line in different.read_text().splitlines()]
    records[0]["instruction"] = "different task"
    different.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tasks": 70,
                "valid_tasks": 66,
                "invalid_benchmark_tasks": 4,
                "dataset": str(different),
                "method": "sayplan",
                "method_label": "paper-based reimplementation",
                "controller_profile": "raw_external",
                "seed": 42,
                "max_input_tokens": 4096,
                "max_new_tokens": 448,
                "max_search_attempts": 2,
                "max_plan_revisions": 4,
                "avg_search_calls": 1.0,
                "avg_plan_revisions": 1.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task-level hash mismatch"):
        audit_sayplan_summary(summary_path, benchmark_hash(reference))


def test_saycan_audit_requires_direct_skill_protocol(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    write_dataset(dataset)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "tasks": 70,
                "valid_tasks": 66,
                "invalid_benchmark_tasks": 4,
                "dataset": str(dataset),
                "method": "SayCan",
                "method_label": (
                    "official-code adaptation (notebook; symbolic affordance adapter)"
                ),
                "controller_profile": "lightweight",
                "plan_sr": None,
                "plan_metric_reason": "direct skill method",
                "seed": 42,
                "max_input_tokens": 4096,
                "affordance_model": "symbolic action preconditions",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controller_profile"):
        audit_saycan_summary(summary_path, benchmark_hash(dataset))
