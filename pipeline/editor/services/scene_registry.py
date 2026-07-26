from __future__ import annotations

import copy
from pathlib import Path

from sg.scene_graph import HOTEL, OFFICE, ALLENSVILLE, SUPERMARKET, PUDU

from .py_importer import import_scene_from_py


GENERATED_DIR = Path(__file__).resolve().parents[2] / "sg" / "generated"


def load_all_scenes() -> dict[str, dict]:
    scenes = {
        "hotel": copy.deepcopy(HOTEL),
        "office": copy.deepcopy(OFFICE),
        "allensville": copy.deepcopy(ALLENSVILLE),
        "supermarket": copy.deepcopy(SUPERMARKET),
        "pudu": copy.deepcopy(PUDU),
    }
    scenes.update(load_generated_scenes())
    return scenes


def load_generated_scenes() -> dict[str, dict]:
    scenes: dict[str, dict] = {}
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(GENERATED_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            scene = import_scene_from_py(path)
        except Exception:
            continue
        scene_name = scene.get("name", path.stem)
        scenes[scene_name] = scene
    return scenes

