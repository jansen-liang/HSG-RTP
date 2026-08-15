from __future__ import annotations

import json
from copy import deepcopy
import time
from typing import Any

import torch


class StreamingModelPolicy:
    def __init__(
        self,
        model: Any,
        global_generation_config: dict[str, Any] | None = None,
        local_generation_config: dict[str, Any] | None = None,
        static_scene: bool = False,
    ):
        self.model = model
        self.global_generation_config = global_generation_config or {
            "do_sample": False,
        }
        self.local_generation_config = local_generation_config or {
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.95,
        }
        self.static_scene = static_scene
        self.reset_usage()

    def reset_usage(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._initial_global_scene: dict[str, Any] | None = None
        self._initial_local_scenes: dict[str, dict[str, Any]] = {}

    def _planning_scene(
        self, mode: str, scene_graph: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.static_scene:
            return scene_graph
        if mode == "global":
            if self._initial_global_scene is None:
                self._initial_global_scene = deepcopy(scene_graph)
            return self._initial_global_scene

        room = scene_graph.get("room", {})
        room_id = str(
            room.get("id")
            or room.get("name")
            or scene_graph.get("current_room")
            or "unknown"
        )
        if room_id not in self._initial_local_scenes:
            self._initial_local_scenes[room_id] = deepcopy(scene_graph)
        return self._initial_local_scenes[room_id]

    def usage_summary(self) -> dict[str, float | int]:
        return {
            "model_calls": len(self.calls),
            "input_tokens": sum(call["input_tokens"] for call in self.calls),
            "output_tokens": sum(call["output_tokens"] for call in self.calls),
            "total_tokens": sum(call["total_tokens"] for call in self.calls),
            "inference_time": sum(call["inference_time"] for call in self.calls),
        }

    def _synchronize(self) -> None:
        try:
            device = next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _generate(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
        generation_config: dict[str, Any],
        mode: str,
    ) -> str:
        self._synchronize()
        start_time = time.perf_counter()
        try:
            planning_scene = self._planning_scene(mode, scene_graph)
            outputs = self.model(
                instructions=[instruction],
                completed=[completed],
                pending=[pending],
                scene_graphs=[planning_scene],
                target_subtasks=None,
                generation_config=generation_config,
            )
        finally:
            self._synchronize()
            elapsed = time.perf_counter() - start_time
        usage_records = outputs.get("usage", [{}])
        usage = usage_records[0] if usage_records else {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        self.calls.append(
            {
                "mode": mode,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(
                    usage.get("total_tokens", input_tokens + output_tokens)
                ),
                "inference_time": elapsed,
            }
        )
        predictions = outputs.get("predictions", [])
        if not predictions:
            raise RuntimeError("Model returned no predictions")
        return predictions[0]

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        return self._generate(
            instruction,
            scene_graph,
            completed,
            [],
            self.global_generation_config,
            "global",
        )

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        return self._generate(
            instruction,
            scene_graph,
            completed,
            pending,
            self.local_generation_config,
            "local",
        )


class RoutedStreamingModelPolicy:
    """Route global and local generation to independently loaded models."""

    def __init__(
        self,
        global_model: Any,
        local_model: Any,
        global_generation_config: dict[str, Any] | None = None,
        local_generation_config: dict[str, Any] | None = None,
        static_scene: bool = False,
    ):
        self.global_policy = StreamingModelPolicy(
            global_model,
            global_generation_config=global_generation_config,
            static_scene=static_scene,
        )
        self.local_policy = StreamingModelPolicy(
            local_model,
            local_generation_config=local_generation_config,
            static_scene=static_scene,
        )

    def reset_usage(self) -> None:
        self.global_policy.reset_usage()
        self.local_policy.reset_usage()

    def usage_summary(self) -> dict[str, float | int]:
        global_usage = self.global_policy.usage_summary()
        local_usage = self.local_policy.usage_summary()
        return {
            key: global_usage[key] + local_usage[key]
            for key in global_usage
        }

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        return self.global_policy.generate_global(instruction, scene_graph, completed)

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        return self.local_policy.generate_local(
            instruction, scene_graph, completed, pending
        )


class OraclePolicy:
    def __init__(self, global_plan: list[str], local_actions: list[str]):
        self.global_plan = list(global_plan)
        self.local_actions = iter(local_actions)

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        return json.dumps({"mode": "global", "task": self.global_plan})

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        try:
            action = next(self.local_actions)
        except StopIteration as error:
            raise RuntimeError("Oracle action sequence exhausted before reaching the goal") from error
        return json.dumps({"mode": "local", "task": [action]})
