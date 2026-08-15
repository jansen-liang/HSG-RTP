from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from evaluation.delta_baseline import (
    DOMAIN,
    DeltaAdaptationPolicy,
    DeltaPDDLPlanner,
    expected_subgoal_schema,
    parse_delta_subgoals,
    pddl_actions_to_local,
)
from pipeline.utils.action_planner import generate_global_plan


def _state_text(state: Any) -> str:
    if isinstance(state, dict):
        return ", ".join(f"{key}={state[key]}" for key in sorted(state)) or "free"
    if state in (None, ""):
        return "free"
    return str(state)


def _delta_affordances(data: dict[str, Any], is_large: bool) -> list[str]:
    source = set(data.get("affordance", [])) | set(data.get("capabilities", []))
    affordances = set(source)
    if "place" in source:
        affordances.update(("drop", "place_on"))
    if is_large and data.get("is_container"):
        affordances.update(("load", "unload", "place_on"))
    return sorted(affordances)


def hsg_scene_to_delta(scene: dict[str, Any]) -> dict[str, Any]:
    rooms = {}
    for room_id, room in scene.get("rooms", {}).items():
        items = {}
        for collection, is_large in (("large_objects", True), ("small_objects", False)):
            for item_id, item in room.get(collection, {}).items():
                state = item.get("state", "free")
                unavailable = (
                    isinstance(state, dict)
                    and state.get("availability") in {"unavailable", "missing"}
                )
                items[item_id] = {
                    "accessible": not unavailable,
                    "affordance": _delta_affordances(item, is_large),
                    "state": _state_text(state),
                }
                relation = item.get("relation")
                if relation:
                    items[item_id]["relation"] = dict(relation)
        rooms[room_id] = {
            "items": items,
            "neighbor": list(room.get("neighbor", [])),
            "floor": room.get("floor", "unknown"),
        }

    agent = scene.get("agent", {})
    converted = {
        "name": scene.get("name", "hsg_rtp_scene"),
        "rooms": rooms,
        "agent": {
            "position": agent.get("position"),
            "state": agent.get("state", "hand-free"),
        },
    }
    inventory = agent.get("inventory", {})
    if inventory:
        converted["agent"]["inventory"] = sorted(inventory)
    return converted


def required_delta_items(record: dict[str, Any], scene: dict[str, Any]) -> list[str]:
    parameters = record.get("task_info", {}).get("parameters", {})
    required = list(parameters.get("objects", []))
    for room in scene.get("rooms", {}).values():
        for item_id, item in room.get("large_objects", {}).items():
            if item.get("is_container"):
                required.append(item_id)
    return list(dict.fromkeys(required))


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class UpstreamDeltaPDDLPlanner(DeltaPDDLPlanner):
    def __init__(self, delta_root: Path, timeout: float = 60.0) -> None:
        self.delta_root = delta_root.resolve()
        super().__init__(self.delta_root / "downward/fast-downward.py", timeout)
        root = str(self.delta_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        self.upstream_planner = importlib.import_module("planner")

    def _solve(
        self,
        domain_path: Path,
        problem_path: Path,
        plan_path: Path,
    ) -> tuple[str | None, float, str]:
        started = time.perf_counter()
        with working_directory(self.delta_root):
            plan, _, _, exit_code, error = self.upstream_planner.query(
                str(domain_path.resolve()),
                str(problem_path.resolve()),
                str(plan_path.resolve()),
                False,
                max_time=self.timeout,
            )
        elapsed = time.perf_counter() - started
        if exit_code != 1 or plan is None:
            return None, elapsed, error
        return plan, elapsed, ""

    def validate_problem(
        self, domain_text: str, problem_text: str
    ) -> tuple[bool, float, str]:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="hsg_delta_upstream_validate_") as directory:
            root = Path(directory)
            domain_path = root / "domain.pddl"
            problem_path = root / "problem.pddl"
            plan_path = root / "plan.txt"
            domain_path.write_text(domain_text, encoding="utf-8")
            problem_path.write_text(problem_text, encoding="utf-8")
            plan, elapsed, error = self._solve(domain_path, problem_path, plan_path)
        return plan is not None, elapsed, error


def build_adapted_problem(
    scene: dict[str, Any], record: dict[str, Any]
) -> tuple[str, dict[str, str], dict[str, str]]:
    schema = expected_subgoal_schema(record)
    object_ids = list(schema.get("objects", []))
    room_symbols, object_symbols = DeltaPDDLPlanner._symbols(scene, object_ids)
    object_rooms = DeltaPDDLPlanner._object_locations(scene, object_ids)
    state = {
        "robot_room": scene["agent"]["position"],
        "object_rooms": object_rooms,
        "carrying": None,
        "organized": set(),
        "visited": {scene["agent"]["position"]},
    }
    if schema["kind"] == "deliver":
        target_rooms = list(schema["rooms"])
        assignments = {
            object_id: target_rooms[min(index, len(target_rooms) - 1)]
            for index, object_id in enumerate(object_ids)
        }
        literals = [
            f"(object-at {object_symbols[object_id]} {room_symbols[room_id]})"
            for object_id, room_id in assignments.items()
        ]
    elif schema["kind"] == "tidy":
        literals = [f"(organized {object_symbols[object_id]})" for object_id in object_ids]
    else:
        literals = [f"(visited {room_symbols[room_id]})" for room_id in schema["rooms"]]
    goal = literals[0] if len(literals) == 1 else f"(and {' '.join(literals)})"
    problem = DeltaPDDLPlanner._problem(
        scene,
        state,
        goal,
        room_symbols,
        object_symbols,
    )
    return problem, room_symbols, object_symbols


def build_upstream_decomposition_prompt(
    delta_root: Path,
    record: dict[str, Any],
    scene: dict[str, Any],
    problem_text: str | None = None,
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    root = str(delta_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    prompt_module = importlib.import_module("prompt")
    example_module = importlib.import_module("data.example")
    domain_example = "office"
    scene_example = "allensville"
    example = example_module.get_example(domain_example)
    domain_text = DOMAIN
    adapted_problem, room_symbols, object_symbols = build_adapted_problem(scene, record)
    problem_text = problem_text or adapted_problem
    example_domain = (delta_root / "data/pddl/domain/office_domain.pddl").read_text(
        encoding="utf-8"
    )
    example_problem = (
        delta_root / "data/pddl/problem/allensville_office_problem.pddl"
    ).read_text(encoding="utf-8")
    schema = expected_subgoal_schema(record)
    relevant = [object_symbols[item] for item in schema.get("objects", [])]
    content, prompt = prompt_module.decompose_problem(
        example["goal"],
        example["subgoal"],
        example["subgoal_pddl"],
        example["item_keep"],
        record["instruction"],
        example_problem,
        relevant,
        problem_text,
        domain_text,
        schema["kind"] == "visit",
    )
    if "(:predicates" not in example_domain:
        raise ValueError("Upstream DELTA example domain is malformed")
    return content, prompt, room_symbols, object_symbols


def build_upstream_problem_prompt(
    delta_root: Path,
    record: dict[str, Any],
    scene: dict[str, Any],
) -> tuple[str, str]:
    root = str(delta_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    prompt_module = importlib.import_module("prompt")
    scene_module = importlib.import_module("data.scene_graph")
    example_hsg_scene = {
        "name": "hsg_delta_example",
        "agent": {"position": "example_start", "state": "hand-free"},
        "rooms": {
            "example_start": {
                "floor": "floor_1",
                "neighbor": ["example_goal"],
                "small_objects": {},
                "large_objects": {},
            },
            "example_goal": {
                "floor": "floor_1",
                "neighbor": ["example_start"],
                "small_objects": {},
                "large_objects": {},
            },
        },
    }
    example_record = {
        "instruction": "Visit the example goal room.",
        "task_info": {
            "type": "guidance",
            "parameters": {
                "intermediate_points": [],
                "end_room": "example_goal",
            },
        },
    }
    example_problem, _, _ = build_adapted_problem(
        example_hsg_scene, example_record
    )
    example_scene = hsg_scene_to_delta(example_hsg_scene)
    query_scene = hsg_scene_to_delta(scene)
    relevant_items = required_delta_items(record, scene)
    query_scene = scene_module.prune_sg_with_item(query_scene, relevant_items)
    return prompt_module.sg_2_pddl_problem(
        "hsg-delta",
        DOMAIN,
        example_problem,
        example_scene,
        query_scene,
        example_record["instruction"],
        record["instruction"],
        DOMAIN,
        "hsg-delta",
    )


def extract_pddl_problem(response: str) -> str:
    start = response.find("(define")
    if start < 0:
        raise ValueError("Problem-generation response contains no PDDL define block")
    depth = 0
    end = None
    for index, character in enumerate(response[start:], start=start):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError("Problem-generation response has unbalanced parentheses")
    problem = response[start:end]
    required = ("(:domain hsg-delta)", "(:objects", "(:init", "(:goal")
    missing = [section for section in required if section not in problem]
    if missing:
        raise ValueError(f"Generated PDDL problem is missing {missing}")
    return problem + "\n"


def parse_upstream_subgoals(
    response: str,
    record: dict[str, Any],
    room_symbols: dict[str, str],
    object_symbols: dict[str, str],
) -> list[dict[str, str]]:
    symbol_to_room = {symbol: room_id for room_id, symbol in room_symbols.items()}
    symbol_to_object = {
        symbol: object_id for object_id, symbol in object_symbols.items()
    }
    predicates = re.findall(
        r"\((object-at|organized|visited)\s+([^()\s]+)(?:\s+([^()\s]+))?\)",
        response,
        flags=re.IGNORECASE,
    )
    subgoals: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for predicate, first, second in predicates:
        predicate = predicate.lower()
        if predicate == "object-at":
            key = (predicate, first, second)
            item = {
                "kind": "deliver",
                "object": symbol_to_object.get(first, first),
                "room": symbol_to_room.get(second, second),
            }
        elif predicate == "organized":
            key = (predicate, first)
            item = {
                "kind": "tidy",
                "object": symbol_to_object.get(first, first),
            }
        else:
            key = (predicate, first)
            item = {
                "kind": "visit",
                "room": symbol_to_room.get(first, first),
            }
        if key not in seen:
            seen.add(key)
            subgoals.append(item)
    return parse_delta_subgoals({"subgoals": subgoals}, record)


class UpstreamDeltaAdaptationPolicy(DeltaAdaptationPolicy):
    def __init__(
        self,
        backend: Any,
        record: dict[str, Any],
        initial_scene: dict[str, Any],
        planner: UpstreamDeltaPDDLPlanner,
        delta_root: Path,
        max_problem_attempts: int = 2,
        max_decomposition_attempts: int = 2,
    ) -> None:
        super().__init__(
            backend,
            record,
            initial_scene,
            planner,
            max_decomposition_attempts=max_decomposition_attempts,
        )
        self.delta_root = delta_root.resolve()
        self.max_problem_attempts = max_problem_attempts
        self.problem_generation_calls = 0
        self.problem_generation_error = ""
        self.problem_generation_prediction = ""

    def reset_usage(self) -> None:
        super().reset_usage()
        self.problem_generation_calls = 0
        self.problem_generation_error = ""
        self.problem_generation_prediction = ""

    def usage_summary(self) -> dict[str, float | int | str]:
        summary = super().usage_summary()
        summary.update(
            {
                "problem_generation_calls": self.problem_generation_calls,
                "problem_generation_error": self.problem_generation_error,
                "problem_generation_prediction": self.problem_generation_prediction,
            }
        )
        return summary

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        problem_content, problem_prompt = build_upstream_problem_prompt(
            self.delta_root, self.record, self.initial_scene
        )
        problem_text = None
        feedback = ""
        for _ in range(self.max_problem_attempts):
            self.problem_generation_calls += 1
            user_prompt = problem_prompt
            if feedback:
                user_prompt += (
                    "\nThe previous PDDL problem was invalid: "
                    f"{feedback}\nReturn a corrected complete PDDL problem only."
                )
            prediction = self.backend.generate_text(
                "upstream_problem_generation", problem_content, user_prompt
            )
            self.problem_generation_prediction = prediction
            try:
                candidate = extract_pddl_problem(prediction)
            except ValueError as error:
                feedback = str(error)
                self.problem_generation_error = feedback
                continue
            valid, elapsed, planner_error = self.planner.validate_problem(
                DOMAIN, candidate
            )
            self.planner_time += elapsed
            if valid:
                problem_text = candidate
                self.problem_generation_error = ""
                break
            self.planner_failures += 1
            feedback = planner_error or "Fast Downward rejected the generated problem"
            self.problem_generation_error = feedback
        if problem_text is None:
            return json.dumps({"mode": "global", "task": []})

        content, prompt, room_symbols, object_symbols = (
            build_upstream_decomposition_prompt(
                self.delta_root,
                self.record,
                self.initial_scene,
                problem_text=problem_text,
            )
        )
        subgoals = None
        feedback = ""
        for _ in range(self.max_decomposition_attempts):
            self.decomposition_calls += 1
            user_prompt = prompt
            if feedback:
                user_prompt += (
                    "\nThe previous response was invalid for this benchmark: "
                    f"{feedback}\nReturn corrected PDDL sub-goals only."
                )
            prediction = self.backend.generate_text(
                "upstream_decomposition", content, user_prompt
            )
            self.decomposition_prediction = prediction
            try:
                subgoals = parse_upstream_subgoals(
                    prediction,
                    self.record,
                    room_symbols,
                    object_symbols,
                )
                self.decomposition_error = ""
                break
            except Exception as error:
                feedback = str(error)
                self.decomposition_error = feedback
        if subgoals is None:
            return json.dumps({"mode": "global", "task": []})

        pddl_actions, planner_time, planner_failures = self.planner.plan(
            self.initial_scene, subgoals
        )
        self.planner_time += planner_time
        self.planner_failures += planner_failures
        self.planner_error = self.planner.last_error
        if not pddl_actions:
            return json.dumps({"mode": "global", "task": []})
        self.local_actions = pddl_actions_to_local(
            pddl_actions, self.initial_scene
        )
        global_plan, _ = generate_global_plan(
            self.local_actions,
            self.initial_scene["rooms"],
            self.record.get("task_info", {}).get("type", "general"),
            initial_room=self.initial_scene.get("agent", {}).get("position"),
        )
        return json.dumps({"mode": "global", "task": global_plan})
