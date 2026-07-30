#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        first_character = handle.read(1)
        handle.seek(0)
        if first_character == "[":
            records = json.load(handle)
        else:
            records = [json.loads(line) for line in handle if line.strip()]

    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Dataset must contain a list or JSONL stream of task objects")
    return records


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


def stratum(record: dict[str, Any]) -> tuple[str, str, str]:
    task_info = record.get("task_info", {})
    return (
        str(record.get("scene_name", "unknown")),
        str(task_info.get("type", "unknown")),
        str(task_info.get("difficulty", "unknown")),
    )


def allocate_test_counts(group_sizes: dict[tuple[str, str, str], int], target: int) -> dict[tuple[str, str, str], int]:
    total = sum(group_sizes.values())
    ideals = {key: size * target / total for key, size in group_sizes.items()}
    allocation = {key: math.floor(value) for key, value in ideals.items()}
    remaining = target - sum(allocation.values())
    ranked_keys = sorted(group_sizes, key=lambda key: (-(ideals[key] - allocation[key]), key))

    for key in ranked_keys[:remaining]:
        allocation[key] += 1
    return allocation


def split_records(
    records: list[dict[str, Any]], test_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[stratum(record)].append(record)

    target_test_count = round(len(records) * test_ratio)
    allocation = allocate_test_counts(
        {key: len(group) for key, group in groups.items()}, target_test_count
    )
    train_records = []
    test_records = []

    for key in sorted(groups):
        identity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in groups[key]:
            identity_groups[task_id(record)].append(record)

        grouped_records = sorted(identity_groups.items())
        group_seed = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16)
        random.Random(group_seed).shuffle(grouped_records)
        test_ids = select_identity_groups(grouped_records, allocation[key])

        for identifier, identity_records in grouped_records:
            if identifier in test_ids:
                test_records.extend(identity_records)
            else:
                train_records.extend(identity_records)

    train_records.sort(key=task_id)
    test_records.sort(key=task_id)
    return train_records, test_records


def select_identity_groups(
    grouped_records: list[tuple[str, list[dict[str, Any]]]], target_records: int
) -> set[str]:
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for identifier, records in grouped_records:
        group_size = len(records)
        for current_total, selected in sorted(reachable.items(), reverse=True):
            new_total = current_total + group_size
            if new_total not in reachable:
                reachable[new_total] = selected + (identifier,)

    best_total = min(reachable, key=lambda total: (abs(total - target_records), total > target_records, total))
    return set(reachable[best_total])


def count_streaming_samples(records: list[dict[str, Any]]) -> int:
    return sum(len(record.get("streaming_samples", [])) for record in records)


def distribution(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "scene": dict(sorted(Counter(record.get("scene_name", "unknown") for record in records).items())),
        "task_type": dict(
            sorted(Counter(record.get("task_info", {}).get("type", "unknown") for record in records).items())
        ),
        "difficulty": dict(
            sorted(
                Counter(record.get("task_info", {}).get("difficulty", "unknown") for record in records).items()
            )
        ),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    source: Path,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    all_records = train_records + test_records
    all_ids = [task_id(record) for record in all_records]
    return {
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "strategy": "task-level stratification by scene, task type, and difficulty",
        "seed": seed,
        "test_ratio": test_ratio,
        "dataset": {
            "task_records": len(all_records),
            "unique_task_identities": len(set(all_ids)),
            "repeated_records": len(all_ids) - len(set(all_ids)),
            "streaming_samples": count_streaming_samples(all_records),
        },
        "train": {
            "tasks": len(train_records),
            "unique_task_identities": len({task_id(record) for record in train_records}),
            "streaming_samples": count_streaming_samples(train_records),
            "distribution": distribution(train_records),
            "task_ids": [task_id(record) for record in train_records],
        },
        "test": {
            "tasks": len(test_records),
            "unique_task_identities": len({task_id(record) for record in test_records}),
            "streaming_samples": count_streaming_samples(test_records),
            "distribution": distribution(test_records),
            "task_ids": [task_id(record) for record in test_records],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic task-level dataset split")
    parser.add_argument("source", type=Path, help="Source JSON array or JSONL dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("pipeline/output/split"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.source)
    train_records, test_records = split_records(records, args.test_ratio, args.seed)

    train_ids = {task_id(record) for record in train_records}
    test_ids = {task_id(record) for record in test_records}
    if train_ids & test_ids:
        raise RuntimeError("Train/test task leakage detected")
    if len(train_records) + len(test_records) != len(records):
        raise RuntimeError("Split record count does not match the source dataset")

    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "test.jsonl", test_records)
    manifest = build_manifest(
        args.source, train_records, test_records, args.test_ratio, args.seed
    )
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"train: {len(train_records)} tasks, {count_streaming_samples(train_records)} streaming samples"
    )
    print(
        f"test:  {len(test_records)} tasks, {count_streaming_samples(test_records)} streaming samples"
    )
    print(f"manifest: {args.output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
