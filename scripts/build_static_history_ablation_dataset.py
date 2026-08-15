#!/usr/bin/env python3

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_views(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    global_view = None
    local_views: dict[str, dict[str, Any]] = {}
    for sample in samples:
        scene_graph = sample.get("scene_graph", {})
        if sample.get("mode") == "global" and global_view is None:
            global_view = deepcopy(scene_graph)
        if sample.get("mode") != "local":
            continue
        room_id = str(
            scene_graph.get("current_room")
            or scene_graph.get("room", {}).get("id")
            or scene_graph.get("room", {}).get("name")
            or "unknown"
        )
        local_views.setdefault(room_id, deepcopy(scene_graph))
    return global_view, local_views


def transform_task(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    transformed = deepcopy(task)
    samples = transformed.get("streaming_samples", [])
    global_view, local_views = static_views(samples)
    stats = {"global_samples": 0, "local_samples": 0, "local_rooms": len(local_views)}
    for sample in samples:
        mode = sample.get("mode")
        if mode == "global":
            if global_view is None:
                raise ValueError("Global sample has no reusable global view")
            sample["scene_graph"] = deepcopy(global_view)
            stats["global_samples"] += 1
        elif mode == "local":
            scene_graph = sample.get("scene_graph", {})
            room_id = str(
                scene_graph.get("current_room")
                or scene_graph.get("room", {}).get("id")
                or scene_graph.get("room", {}).get("name")
                or "unknown"
            )
            if room_id not in local_views:
                raise ValueError(f"Missing initial local view for room {room_id}")
            sample["scene_graph"] = deepcopy(local_views[room_id])
            stats["local_samples"] += 1
    transformed["ablation_transform"] = {
        "name": "static_graph_no_history",
        **stats,
    }
    return transformed, stats


def build_dataset(input_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"tasks": 0, "global_samples": 0, "local_samples": 0, "local_rooms": 0}
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                task, stats = transform_task(json.loads(line))
            except Exception as error:
                raise ValueError(f"Failed to transform line {line_number}: {error}") from error
            output.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
            totals["tasks"] += 1
            for key in ("global_samples", "local_samples", "local_rooms"):
                totals[key] += stats[key]
    return {
        "transform": "static_graph_no_history",
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        **totals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze each task's global view and per-room local views for ablation training."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args.input, args.output)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
