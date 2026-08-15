from __future__ import annotations

import json
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .action_parser import ParseError, extract_json_object, parse_prediction
from .plan_evaluator import evaluate_global_plan


def collapsed_scene_graph(scene: dict[str, Any]) -> dict[str, Any]:
    rooms = {}
    for room_id, room in scene.get("rooms", {}).items():
        object_types: dict[str, int] = {}
        for objects_key in ("large_objects", "small_objects"):
            for object_info in room.get(objects_key, {}).values():
                object_type = str(object_info.get("type", "unknown"))
                object_types[object_type] = object_types.get(object_type, 0) + 1
        rooms[room_id] = {
            "floor": room.get("floor"),
            "neighbors": list(room.get("neighbor", [])),
            "object_types": object_types,
        }
    return {
        "name": scene.get("name"),
        "agent_room": scene.get("agent", {}).get("position"),
        "macro_zones": scene.get("macro_zones", {}),
        "rooms": rooms,
    }


def expanded_scene_subgraph(
    scene: dict[str, Any], room_ids: list[str]
) -> dict[str, Any]:
    valid_rooms = scene.get("rooms", {})
    selected = [room_id for room_id in room_ids if room_id in valid_rooms]
    return {
        "name": scene.get("name"),
        "agent": scene.get("agent", {}),
        "rooms": {room_id: valid_rooms[room_id] for room_id in selected},
    }


def parse_room_selection(prediction: str | dict[str, Any], valid_rooms: set[str]) -> list[str]:
    parsed = extract_json_object(prediction)
    room_ids = parsed.get("rooms")
    if not isinstance(room_ids, list) or not all(isinstance(room_id, str) for room_id in room_ids):
        raise ParseError("Room search output must contain a string list named 'rooms'")
    return list(dict.fromkeys(room_id for room_id in room_ids if room_id in valid_rooms))


class HuggingFaceJSONBackend:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 448,
        max_input_tokens: int = 4096,
        option_batch_size: int = 4,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.option_batch_size = option_batch_size
        self.calls: list[dict[str, Any]] = []

    def reset_usage(self) -> None:
        self.calls = []

    def generate(self, stage: str, system_prompt: str, payload: dict[str, Any]) -> str:
        return self.generate_text(
            stage,
            system_prompt,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def generate_text(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        input_tokens = int(encoded["input_ids"].shape[1])
        output_ids = generated[0, input_tokens:]
        output_tokens = int(output_ids.shape[0])
        self.calls.append(
            {
                "stage": stage,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "inference_time": elapsed,
            }
        )
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def score_options(
        self,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        options: list[str],
    ) -> dict[str, float]:
        if not options:
            return {}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][:, -self.max_input_tokens :]
        device = next(self.model.parameters()).device
        prompt_ids = prompt_ids.to(device)
        prompt_mask = torch.ones_like(prompt_ids)
        option_ids = [
            self.tokenizer(option, add_special_tokens=False)["input_ids"]
            for option in options
        ]
        if any(not token_ids for token_ids in option_ids):
            raise ValueError("SayCan options must tokenize to at least one token")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        scores: dict[str, float] = {}
        with torch.inference_mode():
            prompt_output = self.model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                use_cache=True,
            )
            first_token_log_probs = torch.log_softmax(
                prompt_output.logits[0, -1].float(), dim=-1
            )
            base_cache = prompt_output.past_key_values
            legacy_cache = (
                base_cache.to_legacy_cache()
                if hasattr(base_cache, "to_legacy_cache")
                else base_cache
            )

            for batch_start in range(0, len(options), self.option_batch_size):
                batch_options = options[
                    batch_start : batch_start + self.option_batch_size
                ]
                batch_ids = option_ids[
                    batch_start : batch_start + self.option_batch_size
                ]
                batch_scores = [
                    float(first_token_log_probs[token_ids[0]].item())
                    for token_ids in batch_ids
                ]
                continuation_length = max(len(token_ids) - 1 for token_ids in batch_ids)
                if continuation_length:
                    batch_size = len(batch_ids)
                    pad_id = self.tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = self.tokenizer.eos_token_id
                    continuation = torch.full(
                        (batch_size, continuation_length),
                        int(pad_id),
                        dtype=prompt_ids.dtype,
                        device=device,
                    )
                    continuation_mask = torch.zeros_like(continuation)
                    for row, token_ids in enumerate(batch_ids):
                        previous_tokens = token_ids[:-1]
                        if previous_tokens:
                            continuation[row, : len(previous_tokens)] = torch.tensor(
                                previous_tokens,
                                dtype=prompt_ids.dtype,
                                device=device,
                            )
                            continuation_mask[row, : len(previous_tokens)] = 1
                    expanded_legacy_cache = tuple(
                        tuple(
                            tensor.repeat_interleave(batch_size, dim=0)
                            for tensor in layer
                        )
                        for layer in legacy_cache
                    )
                    expanded_cache = DynamicCache.from_legacy_cache(
                        expanded_legacy_cache
                    )
                    attention_mask = torch.cat(
                        (
                            prompt_mask.repeat_interleave(batch_size, dim=0),
                            continuation_mask,
                        ),
                        dim=1,
                    )
                    continuation_output = self.model(
                        input_ids=continuation,
                        attention_mask=attention_mask,
                        past_key_values=expanded_cache,
                        use_cache=False,
                    )
                    continuation_log_probs = torch.log_softmax(
                        continuation_output.logits.float(), dim=-1
                    )
                    for row, token_ids in enumerate(batch_ids):
                        for position, target_id in enumerate(token_ids[1:]):
                            batch_scores[row] += float(
                                continuation_log_probs[
                                    row, position, target_id
                                ].item()
                            )
                scores.update(zip(batch_options, batch_scores))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        prompt_length = int(prompt_ids.shape[1])
        self.calls.append(
            {
                "stage": stage,
                "input_tokens": sum(prompt_length + len(ids) for ids in option_ids),
                "output_tokens": 0,
                "inference_time": elapsed,
                "options": len(options),
            }
        )
        return scores


class SayPlanAdaptationPolicy:
    def __init__(
        self,
        backend: HuggingFaceJSONBackend,
        record: dict[str, Any],
        initial_scene: dict[str, Any],
        max_search_attempts: int = 2,
        max_plan_revisions: int = 4,
    ) -> None:
        self.backend = backend
        self.record = record
        self.initial_scene = initial_scene
        self.max_search_attempts = max_search_attempts
        self.max_plan_revisions = max_plan_revisions
        self.search_calls = 0
        self.plan_revisions = 0
        self.local_calls = 0

    def reset_usage(self) -> None:
        self.backend.reset_usage()
        self.search_calls = 0
        self.plan_revisions = 0
        self.local_calls = 0

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
            "search_calls": self.search_calls,
            "plan_revisions": self.plan_revisions,
            "local_calls": self.local_calls,
        }

    def _search_rooms(self, instruction: str) -> list[str]:
        collapsed = collapsed_scene_graph(self.initial_scene)
        valid_rooms = set(collapsed["rooms"])
        parameters = self.record.get("task_info", {}).get("parameters", {})
        required_rooms = []
        for key in ("source_room", "end_room"):
            room_id = parameters.get(key)
            if room_id in valid_rooms:
                required_rooms.append(room_id)
        for key in ("source_rooms", "target_rooms", "intermediate_points"):
            required_rooms.extend(
                room_id
                for room_id in parameters.get(key, [])
                if room_id in valid_rooms
            )
        system_prompt = (
            "You perform SayPlan-style semantic search over a collapsed scene graph. "
            "Select every room that may be needed to complete the instruction, including "
            "transit and elevator rooms. Return exactly {\"rooms\":[\"EXACT_ROOM_ID\"]}."
        )
        for _ in range(self.max_search_attempts):
            self.search_calls += 1
            prediction = self.backend.generate(
                "search",
                system_prompt,
                {"instruction": instruction, "collapsed_scene_graph": collapsed},
            )
            try:
                selected = parse_room_selection(prediction, valid_rooms)
            except ParseError:
                continue
            if selected:
                return list(dict.fromkeys(required_rooms + selected))
        return list(dict.fromkeys(required_rooms + list(collapsed["rooms"])))

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        selected_rooms = self._search_rooms(instruction)
        expanded = expanded_scene_subgraph(self.initial_scene, selected_rooms)
        system_prompt = (
            "You implement the SayPlan planning stage. Return exactly one JSON object "
            "with mode global and a task list. Every step must be "
            "goto(EXACT_ROOM_ID): ACTION. ACTION is pass(), pick(EXACT_OBJECT_ID), "
            "place(EXACT_OBJECT_ID), organize(EXACT_OBJECT_ID), or "
            "trans from(FLOOR_ID) to(FLOOR_ID). Use only exact IDs in the expanded graph."
        )
        previous_errors: list[str] = []
        last_prediction = ""
        last_parseable_prediction = ""
        for attempt in range(self.max_plan_revisions + 1):
            last_prediction = self.backend.generate(
                "planning" if attempt == 0 else "revision",
                system_prompt,
                {
                    "instruction": instruction,
                    "expanded_scene_graph": expanded,
                    "completed": completed,
                    "simulator_feedback": previous_errors,
                },
            )
            try:
                plan = parse_prediction(last_prediction, "global")
            except ParseError as error:
                previous_errors = [str(error)]
            else:
                last_parseable_prediction = last_prediction
                evaluation = evaluate_global_plan(
                    self.record, plan, self.initial_scene
                )
                if evaluation.success:
                    return last_prediction
                previous_errors = list(evaluation.errors)
            if attempt < self.max_plan_revisions:
                self.plan_revisions += 1
        return last_parseable_prediction or last_prediction

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        self.local_calls += 1
        system_prompt = (
            "Return exactly one JSON object in the form "
            "{\"mode\":\"local\",\"task\":[\"ONE_ACTION\"]}. ONE_ACTION must be one "
            "immediately executable goto, scan, pick, place, press, or wait action using "
            "exact IDs from the current room graph. Do not return explanations."
        )
        return self.backend.generate(
            "local",
            system_prompt,
            {
                "instruction": instruction,
                "scene": scene_graph,
                "completed": completed,
                "pending": pending,
            },
        )
