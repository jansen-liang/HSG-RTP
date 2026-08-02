#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.recovery import RecoveryConfig
from evaluation.runner import evaluate_policy_dataset
from pipeline.utils.scene_loader import load_scenes


class TextBaselinePolicy:
    def __init__(self, model_path: str, max_new_tokens: int = 448) -> None:
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
        self.reset_usage()

    def reset_usage(self) -> None:
        self.calls: list[dict[str, float | int | str]] = []

    def usage_summary(self) -> dict[str, float | int]:
        return {
            "model_calls": len(self.calls),
            "input_tokens": sum(int(call["input_tokens"]) for call in self.calls),
            "output_tokens": sum(int(call["output_tokens"]) for call in self.calls),
            "total_tokens": sum(int(call["total_tokens"]) for call in self.calls),
            "inference_time": sum(float(call["inference_time"]) for call in self.calls),
        }

    @staticmethod
    def _system_prompt(mode: str) -> str:
        common = (
            "You plan actions for a robot in a symbolic scene. Return exactly one "
            "JSON object with no markdown or explanation. Use exact IDs and never emit "
            "placeholder words such as room, object, or action. "
        )
        if mode == "global":
            return common + (
                "This request is GLOBAL. Every list item must have the exact form "
                "goto(EXACT_ROOM_ID): GLOBAL_ACTION. The only GLOBAL_ACTION forms are "
                "pass(), pick(EXACT_OBJECT_ID), place(EXACT_OBJECT_ID), "
                "organize(EXACT_OBJECT_ID), and trans from(FLOOR_ID) to(FLOOR_ID). "
                "Never emit standalone goto, scan, pick, or place commands in a global "
                "plan. Example JSON shape: "
                '{"mode":"global","task":["goto(kitchen): pick(cup)",'
                '"goto(dining_room): place(cup)"]}.'
            )
        return common + (
            "This request is LOCAL. Return "
            '{"mode":"local","task":["ONE_ACTION"]}, where ONE_ACTION is one '
            "immediately executable goto, scan, pick, place, press, wait, follow, or "
            "stop action using exact IDs from the current scene. Never return a global "
            "step containing a colon. Use completed actions and the pending global plan "
            "to avoid repeating finished work."
        )

    def _generate(
        self,
        mode: str,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        context = {
            "requested_mode": mode,
            "instruction": instruction,
            "scene": scene_graph,
            "completed": completed,
            "pending": pending,
        }
        messages = [
            {"role": "system", "content": self._system_prompt(mode)},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
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
            max_length=4096,
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
        prediction = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        self.calls.append(
            {
                "mode": mode,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "inference_time": elapsed,
            }
        )
        return prediction

    def generate_global(
        self, instruction: str, scene_graph: dict[str, Any], completed: list[str]
    ) -> str:
        return self._generate("global", instruction, scene_graph, completed, [])

    def generate_local(
        self,
        instruction: str,
        scene_graph: dict[str, Any],
        completed: list[str],
        pending: list[str],
    ) -> str:
        return self._generate("local", instruction, scene_graph, completed, pending)


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a text-only causal LLM under the task-level protocol"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))
    policy = TextBaselinePolicy(args.model_path, args.max_new_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "task_results.jsonl"
    results_path.write_text("", encoding="utf-8")
    completed = 0

    def record_progress(result: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[{completed}/{len(records)}] index={result['index']} "
            f"plan={int(result['plan_success'])} exec={int(result['exec_success'])} "
            f"calls={result['model_calls']} time={result['inference_time']:.2f}s",
            flush=True,
        )

    _, summary = evaluate_policy_dataset(
        records,
        scenes,
        lambda _: policy,
        max_steps=args.max_steps,
        recovery_config=RecoveryConfig(
            max_local_retries=0,
            max_global_replans=0,
            max_initial_plan_retries=0,
        ),
        progress_callback=record_progress,
    )
    summary.update(
        {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "seed": args.seed,
            "baseline": "text_only_zero_shot",
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
