from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .compilers import compile_to_canonical, detect_schema
from .validation import validate_graph


def _load_scene(path: Path, scene_name: str | None) -> dict:
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import scene from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if scene_name:
            if not hasattr(module, scene_name):
                raise ValueError(f"Scene constant '{scene_name}' not found in {path}")
            return getattr(module, scene_name)
        if hasattr(module, "SCENE"):
            return module.SCENE
        for key, value in module.__dict__.items():
            if key.isupper() and isinstance(value, dict):
                return value
        raise ValueError(f"No scene dict found in {path}")

    raise ValueError(f"Unsupported input file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and validate scene files against the canonical graph IR.")
    parser.add_argument("source", type=Path, help="Path to a .json or .py scene file.")
    parser.add_argument("--scene", type=str, default=None, help="Optional scene constant name when loading a Python module.")
    parser.add_argument("--dump", type=Path, default=None, help="Optional output path for the compiled canonical graph JSON.")
    args = parser.parse_args()

    scene = _load_scene(args.source, args.scene)
    schema = detect_schema(scene)
    graph = compile_to_canonical(scene)
    validation = validate_graph(graph)

    print(f"source: {args.source}")
    print(f"schema: {schema}")
    print(f"graph: {graph.name}")
    print(f"nodes: {len(graph.nodes)}")
    print(f"edges: {len(graph.edges)}")
    print(f"valid: {validation.ok}")
    print(f"errors: {len(validation.errors)}")
    print(f"warnings: {len(validation.warnings)}")
    if validation.errors:
        print("error_details:")
        for item in validation.errors:
            print(f"  - {item}")
    if validation.warnings:
        print("warning_details:")
        for item in validation.warnings:
            print(f"  - {item}")

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote: {args.dump}")


if __name__ == "__main__":
    main()
