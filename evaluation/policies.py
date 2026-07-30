import json
from typing import Any


class StreamingModelPolicy:
    def __init__(
        self,
        model: Any,
        global_generation_config: dict[str, Any] | None = None,
        local_generation_config: dict[str, Any] | None = None,
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

    def _generate(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
        generation_config: dict[str, Any],
    ) -> str:
        outputs = self.model(
            instructions=[instruction],
            completed=[completed],
            pending=[pending],
            scene_graphs=[scene_graph],
            target_subtasks=None,
            generation_config=generation_config,
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
