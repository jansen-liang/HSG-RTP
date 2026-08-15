from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from .action_parser import ParseError, extract_json_object
from pipeline.utils.action_planner import (
    DifficultyLevel,
    find_suitable_surface,
    generate_global_plan,
    plan_path_with_elevator,
    plan_pick_with_dependency,
)


DOMAIN = """(define (domain hsg-delta)
  (:requirements :strips :typing)
  (:types room object)
  (:predicates
    (robot-at ?room - room)
    (connected ?from - room ?to - room)
    (object-at ?object - object ?room - room)
    (carrying ?object - object)
    (handempty)
    (organized ?object - object)
    (visited ?room - room))
  (:action move
    :parameters (?from - room ?to - room)
    :precondition (and (robot-at ?from) (connected ?from ?to))
    :effect (and (not (robot-at ?from)) (robot-at ?to) (visited ?to)))
  (:action pick
    :parameters (?object - object ?room - room)
    :precondition (and (robot-at ?room) (object-at ?object ?room) (handempty))
    :effect (and (not (object-at ?object ?room)) (not (handempty)) (carrying ?object)))
  (:action place
    :parameters (?object - object ?room - room)
    :precondition (and (robot-at ?room) (carrying ?object))
    :effect (and (not (carrying ?object)) (handempty) (object-at ?object ?room)))
  (:action organize
    :parameters (?object - object ?room - room)
    :precondition (and (robot-at ?room) (object-at ?object ?room) (handempty))
    :effect (organized ?object)))
"""


def expected_subgoal_schema(record: dict[str, Any]) -> dict[str, Any]:
    task_info = record.get("task_info", {})
    task_type = task_info.get("type")
    parameters = task_info.get("parameters", {})
    if task_type == "delivery":
        return {
            "kind": "deliver",
            "objects": list(parameters.get("objects", [])),
            "rooms": list(parameters.get("target_rooms", [])),
        }
    if task_type == "tidying":
        return {
            "kind": "tidy",
            "objects": list(parameters.get("objects", [])),
        }
    if task_type == "guidance":
        return {
            "kind": "visit",
            "rooms": list(parameters.get("intermediate_points", []))
            + [parameters.get("end_room")],
        }
    raise ValueError(f"Unsupported task type {task_type!r}")


def parse_delta_subgoals(
    prediction: str | dict[str, Any], record: dict[str, Any]
) -> list[dict[str, str]]:
    parsed = extract_json_object(prediction)
    subgoals = parsed.get("subgoals")
    if not isinstance(subgoals, list) or not all(isinstance(item, dict) for item in subgoals):
        raise ParseError("DELTA output must contain a list named 'subgoals'")
    normalized = [
        {key: str(value) for key, value in item.items() if key in {"kind", "object", "room"}}
        for item in subgoals
    ]
    expected = expected_subgoal_schema(record)
    if expected["kind"] == "visit":
        normalized = [
            {"kind": item.get("kind", ""), "room": item.get("room", "")}
            for item in normalized
        ]
    else:
        normalized = [
            {
                "kind": item.get("kind", ""),
                "object": item.get("object", ""),
                **({"room": item["room"]} if "room" in item else {}),
            }
            for item in normalized
        ]
    if expected["kind"] == "deliver":
        if any(item.get("kind") != "deliver" for item in normalized):
            raise ParseError("Every delivery subgoal must use kind 'deliver'")
        canonical = []
        for item in normalized:
            object_id = item.get("object")
            if object_id not in expected["objects"] or item.get("room") not in expected["rooms"]:
                continue
            canonical = [entry for entry in canonical if entry.get("object") != object_id]
            canonical.append(item)
        covered = {item.get("object") for item in canonical}
        missing = set(expected["objects"]) - covered
        if missing:
            raise ParseError(f"Delivery decomposition has no valid target for {sorted(missing)}")
        normalized = canonical
    elif expected["kind"] == "tidy":
        if any(item.get("kind") != "tidy" for item in normalized):
            raise ParseError("Every tidying subgoal must use kind 'tidy'")
        if {item.get("object") for item in normalized} != set(expected["objects"]):
            raise ParseError("Tidying decomposition must cover every requested object once")
        if len(normalized) != len(expected["objects"]):
            raise ParseError("Tidying decomposition contains duplicate objects")
    else:
        rooms = [item.get("room") for item in normalized]
        if any(item.get("kind") != "visit" for item in normalized):
            raise ParseError("Every guidance subgoal must use kind 'visit'")
        if rooms != expected["rooms"]:
            raise ParseError("Guidance decomposition must preserve waypoint order")
    return normalized


class DeltaPDDLPlanner:
    def __init__(self, fast_downward: Path, timeout: float = 60.0) -> None:
        self.fast_downward = fast_downward
        self.timeout = timeout
        self.last_error = ""

    @staticmethod
    def _symbols(scene: dict[str, Any], object_ids: list[str]):
        room_to_symbol = {
            room_id: f"r{index}" for index, room_id in enumerate(scene["rooms"])
        }
        object_to_symbol = {
            object_id: f"o{index}" for index, object_id in enumerate(object_ids)
        }
        return room_to_symbol, object_to_symbol

    @staticmethod
    def _object_locations(scene: dict[str, Any], object_ids: list[str]) -> dict[str, str]:
        locations = {}
        for room_id, room in scene["rooms"].items():
            room_objects = set(room.get("small_objects", {})) | set(
                room.get("large_objects", {})
            )
            for object_id in object_ids:
                if object_id in room_objects:
                    locations[object_id] = room_id
        return locations

    @staticmethod
    def _goal_literal(
        subgoal: dict[str, str], room_symbols: dict[str, str], object_symbols: dict[str, str]
    ) -> str:
        if subgoal["kind"] == "deliver":
            return f"(object-at {object_symbols[subgoal['object']]} {room_symbols[subgoal['room']]})"
        if subgoal["kind"] == "tidy":
            return f"(organized {object_symbols[subgoal['object']]})"
        return f"(visited {room_symbols[subgoal['room']]})"

    @staticmethod
    def _problem(
        scene: dict[str, Any],
        state: dict[str, Any],
        goal: str,
        room_symbols: dict[str, str],
        object_symbols: dict[str, str],
    ) -> str:
        room_objects = " ".join(room_symbols.values())
        object_objects = " ".join(object_symbols.values())
        object_declaration = (
            f" {object_objects} - object" if object_objects else ""
        )
        init = [f"(robot-at {room_symbols[state['robot_room']]})"]
        init.extend(
            f"(connected {room_symbols[source]} {room_symbols[target]})"
            for source, room in scene["rooms"].items()
            for target in room.get("neighbor", [])
            if target in room_symbols
            and (
                room.get("floor") == scene["rooms"][target].get("floor")
                or source == "elevator_cabin"
                or target == "elevator_cabin"
            )
        )
        init.extend(
            f"(object-at {object_symbols[object_id]} {room_symbols[room_id]})"
            for object_id, room_id in state["object_rooms"].items()
            if object_id in object_symbols
        )
        init.extend(
            f"(organized {object_symbols[object_id]})"
            for object_id in state["organized"]
        )
        init.extend(f"(visited {room_symbols[room_id]})" for room_id in state["visited"])
        if state["carrying"] is None:
            init.append("(handempty)")
        else:
            init.append(f"(carrying {object_symbols[state['carrying']]})")
        return (
            "(define (problem hsg-task) (:domain hsg-delta)\n"
            f"  (:objects {room_objects} - room{object_declaration})\n"
            f"  (:init {' '.join(init)})\n"
            f"  (:goal {goal}))\n"
        )

    @staticmethod
    def _parse_plan(
        plan_text: str,
        room_symbols: dict[str, str],
        object_symbols: dict[str, str],
    ) -> list[tuple[str, ...]]:
        symbol_to_room = {symbol: room_id for room_id, symbol in room_symbols.items()}
        symbol_to_object = {
            symbol: object_id for object_id, symbol in object_symbols.items()
        }
        actions = []
        for line in plan_text.splitlines():
            line = line.strip()
            if not line.startswith("("):
                continue
            parts = line[1 : line.index(")")].split()
            if not parts:
                continue
            name = parts[0]
            arguments = [
                symbol_to_room.get(argument, symbol_to_object.get(argument, argument))
                for argument in parts[1:]
            ]
            actions.append(tuple([name, *arguments]))
        return actions

    @staticmethod
    def _apply(state: dict[str, Any], action: tuple[str, ...]) -> None:
        if action[0] == "move":
            state["robot_room"] = action[2]
            state["visited"].add(action[2])
        elif action[0] == "pick":
            state["object_rooms"].pop(action[1], None)
            state["carrying"] = action[1]
        elif action[0] == "place":
            state["object_rooms"][action[1]] = action[2]
            state["carrying"] = None
        elif action[0] == "organize":
            state["organized"].add(action[1])

    def _solve(
        self,
        domain_path: Path,
        problem_path: Path,
        plan_path: Path,
    ) -> tuple[str | None, float, str]:
        started = time.perf_counter()
        completed = subprocess.run(
            [
                str(self.fast_downward),
                "--plan-file",
                str(plan_path),
                str(domain_path),
                str(problem_path),
                "--search",
                "astar(lmcut())",
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0 or not plan_path.exists():
            return None, elapsed, (completed.stderr or completed.stdout).strip()
        return plan_path.read_text(encoding="utf-8"), elapsed, ""

    def plan(
        self,
        scene: dict[str, Any],
        subgoals: list[dict[str, str]],
    ) -> tuple[list[tuple[str, ...]], float, int]:
        self.last_error = ""
        object_ids = list(
            dict.fromkeys(
                subgoal["object"] for subgoal in subgoals if "object" in subgoal
            )
        )
        room_symbols, object_symbols = self._symbols(scene, object_ids)
        state = {
            "robot_room": scene["agent"]["position"],
            "object_rooms": self._object_locations(scene, object_ids),
            "carrying": None,
            "organized": set(),
            "visited": {scene["agent"]["position"]},
        }
        if set(object_ids) - set(state["object_rooms"]):
            return [], 0.0, 1
        all_actions = []
        planner_time = 0.0
        failures = 0
        with tempfile.TemporaryDirectory(prefix="hsg_delta_") as temporary_dir:
            temporary = Path(temporary_dir)
            domain_path = temporary / "domain.pddl"
            problem_path = temporary / "problem.pddl"
            plan_path = temporary / "plan.txt"
            domain_path.write_text(DOMAIN, encoding="utf-8")
            for subgoal in subgoals:
                if (
                    subgoal["kind"] == "visit"
                    and subgoal["room"] in state["visited"]
                ):
                    all_actions.append(("visit", subgoal["room"]))
                    continue
                goal = self._goal_literal(subgoal, room_symbols, object_symbols)
                problem_path.write_text(
                    self._problem(
                        scene, state, goal, room_symbols, object_symbols
                    ),
                    encoding="utf-8",
                )
                if plan_path.exists():
                    plan_path.unlink()
                plan_text, elapsed, error = self._solve(
                    domain_path, problem_path, plan_path
                )
                planner_time += elapsed
                if plan_text is None:
                    failures += 1
                    self.last_error = error
                    break
                actions = self._parse_plan(
                    plan_text,
                    room_symbols,
                    object_symbols,
                )
                for action in actions:
                    self._apply(state, action)
                all_actions.extend(actions)
        return all_actions, planner_time, failures


def pddl_actions_to_local(
    actions: list[tuple[str, ...]], scene: dict[str, Any]
) -> list[str]:
    rooms = scene["rooms"]
    local_actions = []
    index = 0
    while index < len(actions):
        action = actions[index]
        if (
            action[0] == "move"
            and action[2] == "elevator_cabin"
            and index + 1 < len(actions)
            and actions[index + 1][0] == "move"
            and actions[index + 1][1] == "elevator_cabin"
        ):
            local_actions.extend(
                plan_path_with_elevator(
                    rooms, action[1], actions[index + 1][2], DifficultyLevel.EASY
                )
            )
            index += 2
            continue
        if action[0] == "move":
            local_actions.append(f"goto({action[2]})")
        elif action[0] == "pick":
            local_actions.extend(plan_pick_with_dependency(rooms, action[2], action[1]))
        elif action[0] == "place":
            surface = find_suitable_surface(rooms, action[2], action[1])
            local_actions.append(f"place({action[1]}, {surface})")
        elif action[0] == "organize":
            local_actions.extend(plan_pick_with_dependency(rooms, action[2], action[1]))
            surface = find_suitable_surface(rooms, action[2], action[1])
            local_actions.append(f"place({action[1]}, {surface})")
        elif action[0] == "visit":
            local_actions.append(f"scan({action[1]})")
        index += 1
    return local_actions


class DeltaAdaptationPolicy:
    def __init__(
        self,
        backend: Any,
        record: dict[str, Any],
        initial_scene: dict[str, Any],
        planner: DeltaPDDLPlanner,
        max_decomposition_attempts: int = 2,
    ) -> None:
        self.backend = backend
        self.record = record
        self.initial_scene = initial_scene
        self.planner = planner
        self.max_decomposition_attempts = max_decomposition_attempts
        self.decomposition_calls = 0
        self.planner_time = 0.0
        self.planner_failures = 0
        self.planner_error = ""
        self.decomposition_error = ""
        self.decomposition_prediction = ""
        self.local_actions: list[str] = []

    def reset_usage(self) -> None:
        self.backend.reset_usage()
        self.decomposition_calls = 0
        self.planner_time = 0.0
        self.planner_failures = 0
        self.planner_error = ""
        self.decomposition_error = ""
        self.decomposition_prediction = ""
        self.local_actions = []

    def usage_summary(self) -> dict[str, float | int]:
        calls = self.backend.calls
        return {
            "model_calls": len(calls),
            "input_tokens": sum(call["input_tokens"] for call in calls),
            "output_tokens": sum(call["output_tokens"] for call in calls),
            "total_tokens": sum(
                call["input_tokens"] + call["output_tokens"] for call in calls
            ),
            "inference_time": sum(call["inference_time"] for call in calls),
            "decomposition_calls": self.decomposition_calls,
            "planner_time": self.planner_time,
            "planner_failures": self.planner_failures,
            "planner_error": self.planner_error,
            "decomposition_error": self.decomposition_error,
            "decomposition_prediction": self.decomposition_prediction,
        }

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        schema = expected_subgoal_schema(self.record)
        system_prompt = (
            "You perform DELTA-style long-horizon task decomposition. Return exactly "
            "{\"subgoals\":[...]}. Each subgoal uses only keys kind, object, and room. "
            "Use kind deliver for object delivery, tidy for object tidying, and visit "
            "for guidance. Cover every requested object exactly once and preserve the "
            "given waypoint order. For example, one delivery object must produce "
            "exactly {\"subgoals\":[{\"kind\":\"deliver\",\"object\":\"cup\","
            "\"room\":\"kitchen\"}]}; one tidying object must produce exactly one "
            "tidy item. Do not output explanations."
        )
        feedback = ""
        subgoals = None
        for _ in range(self.max_decomposition_attempts):
            self.decomposition_calls += 1
            prediction = self.backend.generate(
                "decomposition",
                system_prompt,
                {
                    "instruction": instruction,
                    "task_kind": schema["kind"],
                    "required_objects": schema.get("objects", []),
                    "valid_rooms_in_required_order": schema.get("rooms", []),
                    "output_item_schema": {
                        "kind": schema["kind"],
                        "object": "ONE_EXACT_OBJECT_ID when applicable",
                        "room": "ONE_EXACT_ROOM_ID when applicable",
                    },
                    "previous_error": feedback,
                },
            )
            self.decomposition_prediction = prediction
            try:
                subgoals = parse_delta_subgoals(prediction, self.record)
                self.decomposition_error = ""
                break
            except ParseError as error:
                feedback = str(error)
                self.decomposition_error = feedback
        if subgoals is None:
            return json.dumps({"mode": "global", "task": []})
        pddl_actions, self.planner_time, self.planner_failures = self.planner.plan(
            self.initial_scene, subgoals
        )
        self.planner_error = self.planner.last_error
        if not pddl_actions:
            return json.dumps({"mode": "global", "task": []})
        self.local_actions = pddl_actions_to_local(pddl_actions, self.initial_scene)
        global_plan, _ = generate_global_plan(
            self.local_actions,
            self.initial_scene["rooms"],
            self.record.get("task_info", {}).get("type", "general"),
            initial_room=self.initial_scene.get("agent", {}).get("position"),
        )
        return json.dumps({"mode": "global", "task": global_plan})

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        if not self.local_actions:
            return json.dumps({"mode": "local", "task": ["wait(elevator_up_clear)"]})
        action = self.local_actions.pop(0)
        return json.dumps({"mode": "local", "task": [action]})
