from __future__ import annotations

import importlib.util
from pathlib import Path


def import_scene_from_py(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scene module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "SCENE"):
        return module.SCENE
    for key, value in module.__dict__.items():
        if key.isupper() and isinstance(value, dict) and {"name", "rooms", "macro_zones"} <= set(value.keys()):
            return value
    raise ValueError(f"No scene dict found in {path}")

