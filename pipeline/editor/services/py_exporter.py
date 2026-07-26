from __future__ import annotations

import pprint
import re
from pathlib import Path


def export_scene_python(scene_dict: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = scene_dict["name"]
    module_name = _sanitize_python_name(scene_name)
    const_name = module_name.upper()
    target_file = output_dir / f"{module_name}.py"

    scene_literal = pprint.pformat(scene_dict, width=100, sort_dicts=False)
    content = (
        f"{const_name} = {scene_literal}\n\n"
        f"SCENE = {const_name}\n"
    )
    target_file.write_text(content, encoding="utf-8")
    return target_file


def _sanitize_python_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        normalized = "scene"
    if normalized[0].isdigit():
        normalized = f"scene_{normalized}"
    return normalized

