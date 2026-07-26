from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_ir import compile_to_canonical, detect_schema, validate_action_sequence, validate_graph


def _load_python_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compile_legacy_scene_smoke():
    module = _load_python_module(ROOT / "pipeline" / "sg" / "scene_graph.py", "scene_graph")
    graph = compile_to_canonical(module.HOTEL)

    assert detect_schema(module.HOTEL) == "legacy_scene"
    assert graph.name == "hotel"
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    assert validate_graph(graph).ok


def test_compile_oop_scene_smoke():
    with (ROOT / "HLR_dataset" / "data" / "scene_graphs" / "hospital_scene_0.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)

    graph = compile_to_canonical(payload)

    assert detect_schema(payload) == "oop_scene"
    assert graph.name == "hospital_scene_0"
    assert validate_graph(graph).ok


def test_compile_editor_scene_smoke():
    importer = _load_python_module(ROOT / "pipeline" / "editor" / "services" / "py_importer.py", "py_importer")
    payload = importer.import_scene_from_py(ROOT / "pipeline" / "sg" / "generated" / "hotel.py")
    graph = compile_to_canonical(payload)

    assert detect_schema(payload) == "editor_scene"
    assert graph.name == "hotel"
    assert validate_graph(graph).ok


def test_validate_action_sequence_smoke():
    module = _load_python_module(ROOT / "pipeline" / "sg" / "scene_graph.py", "scene_graph_actions")
    graph = compile_to_canonical(module.HOTEL)
    checks = validate_action_sequence(graph, ["goto(restaurant)", "scan(restaurant)", "goto(lobby)"])

    assert all(check.ok for check in checks)
