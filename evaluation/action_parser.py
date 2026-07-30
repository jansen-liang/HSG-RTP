import json
import re
from dataclasses import dataclass
from typing import Any


LOCAL_ACTION_PATTERN = re.compile(r"^(goto|scan|pick|place|press|wait)\(([^()]*)\)$")
GLOBAL_STEP_PATTERN = re.compile(r"^goto\(([^()]+)\)\s*:\s*(.+)$")
ABSTRACT_ACTION_PATTERN = re.compile(r"^(pass|pick|place|organize)\(([^()]*)\)$")
TRANSITION_PATTERN = re.compile(r"^trans\s+from\(([^()]+)\)\s+to\(([^()]+)\)$")


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class LocalAction:
    name: str
    arguments: tuple[str, ...]

    @property
    def canonical(self) -> str:
        return f"{self.name}({', '.join(self.arguments)})"


@dataclass(frozen=True)
class GlobalStep:
    room: str
    action: str
    arguments: tuple[str, ...]


def parse_local_action(action: str) -> LocalAction:
    if not isinstance(action, str):
        raise ParseError("Local action must be a string")
    match = LOCAL_ACTION_PATTERN.fullmatch(action.strip())
    if not match:
        raise ParseError(f"Invalid local action syntax: {action!r}")

    name = match.group(1)
    raw_arguments = match.group(2).strip()
    arguments = tuple(argument.strip() for argument in raw_arguments.split(",") if argument.strip())
    expected_arguments = {"goto": 1, "scan": 1, "pick": 1, "place": 2, "press": 1, "wait": 1}
    if len(arguments) != expected_arguments[name]:
        raise ParseError(f"{name} expects {expected_arguments[name]} argument(s), got {len(arguments)}")
    return LocalAction(name=name, arguments=arguments)


def parse_global_step(step: str) -> GlobalStep:
    if not isinstance(step, str):
        raise ParseError("Global step must be a string")
    match = GLOBAL_STEP_PATTERN.fullmatch(step.strip())
    if not match:
        raise ParseError(f"Invalid global step syntax: {step!r}")

    room = match.group(1).strip()
    action_text = match.group(2).strip()
    transition_match = TRANSITION_PATTERN.fullmatch(action_text)
    if transition_match:
        return GlobalStep(
            room=room,
            action="trans",
            arguments=(transition_match.group(1).strip(), transition_match.group(2).strip()),
        )

    action_match = ABSTRACT_ACTION_PATTERN.fullmatch(action_text)
    if not action_match:
        raise ParseError(f"Invalid global action syntax: {action_text!r}")
    raw_arguments = action_match.group(2).strip()
    arguments = tuple(argument.strip() for argument in raw_arguments.split(",") if argument.strip())
    action = action_match.group(1)
    if action == "pass" and arguments:
        raise ParseError("pass() must not contain arguments")
    if action != "pass" and not arguments:
        raise ParseError(f"{action} requires at least one object")
    return GlobalStep(room=room, action=action, arguments=arguments)


def extract_json_object(text: str) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ParseError("Prediction must be a JSON object or string")

    decoder = json.JSONDecoder()
    for start_index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ParseError("No valid JSON object found in prediction")


def parse_prediction(text: str | dict[str, Any], expected_mode: str) -> list[str]:
    parsed = extract_json_object(text)
    mode = parsed.get("mode")
    if mode != expected_mode:
        raise ParseError(f"Expected mode {expected_mode!r}, got {mode!r}")
    tasks = parsed.get("task")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ParseError("Prediction field 'task' must be a list of strings")
    if expected_mode == "local" and len(tasks) != 1:
        raise ParseError("Local prediction must contain exactly one action")
    return tasks
