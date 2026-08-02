#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from peft import PeftModel
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.runner import evaluate_streaming_model
from evaluation.recovery import RecoveryConfig
from pipeline.utils.scene_loader import load_scenes
from utils.streaming_hlr import StreamingSceneInstructionQwenModel


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_model(
    model_path: str,
    checkpoint: Path,
    device: torch.device,
    ablation: str,
) -> Any:
    model = StreamingSceneInstructionQwenModel(
        llm_model_name=model_path,
        use_hsge=ablation != "no_hsge",
        use_local_graph=ablation != "no_local_graph",
        use_context=ablation != "no_context",
    )
    training_state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu"
    )
    additional_state = training_state.get("additional_components", {})
    for component_name in ("graph_encoder", "graph_proj", "instruction_encoder"):
        component_state = additional_state.get(component_name)
        component = getattr(model, component_name, None)
        if component_state is not None and component is not None:
            component.load_state_dict(component_state, strict=False)

    new_embeddings = additional_state.get("new_token_embeddings")
    if new_embeddings is not None:
        embedding_layer = model.llm.get_input_embeddings()
        start_index = embedding_layer.weight.shape[0] - new_embeddings.shape[0]
        embedding_layer.weight.data[start_index:] = new_embeddings

    model.llm = PeftModel.from_pretrained(model.llm, checkpoint)
    model.to(device)
    model.eval()
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a streaming checkpoint with strict task-level rollout"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--limit", type=int, help="Evaluate only the first N records (for smoke tests)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-local",
        action="store_true",
        help="Use temperature sampling for local actions instead of deterministic decoding",
    )
    parser.add_argument(
        "--enable-recovery",
        action="store_true",
        help="Enable plan repair, local retries, and global replanning",
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "no_hsge",
            "no_local_graph",
            "no_context",
            "no_dynamic_update",
        ),
        default="full",
        help="Inference-time component removal using the same trained checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    records = load_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.checkpoint, device, args.ablation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_results_path = args.output_dir / "task_results.jsonl"
    task_results_path.write_text("", encoding="utf-8")

    completed = 0

    def record_progress(result: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        with task_results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[{completed}/{len(records)}] index={result['index']} "
            f"type={result['task_type']} plan={int(result['plan_success'])} "
            f"exec={int(result['exec_success'])} "
            f"calls={result['model_calls']} time={result['inference_time']:.2f}s",
            flush=True,
        )

    local_generation_config = (
        {"do_sample": True, "temperature": 0.1, "top_p": 0.95}
        if args.sample_local
        else {"do_sample": False}
    )
    recovery_config = (
        RecoveryConfig()
        if args.enable_recovery
        else RecoveryConfig(
            max_local_retries=0,
            max_global_replans=0,
            max_initial_plan_retries=0,
        )
    )
    _, summary = evaluate_streaming_model(
        model,
        records,
        scenes,
        max_steps=args.max_steps,
        recovery_config=recovery_config,
        global_generation_config={"do_sample": False},
        local_generation_config=local_generation_config,
        static_scene=args.ablation == "no_dynamic_update",
        progress_callback=record_progress,
    )
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "seed": args.seed,
            "local_decoding": "sampled" if args.sample_local else "deterministic",
            "recovery_enabled": args.enable_recovery,
            "ablation": args.ablation,
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output_dir}")


if __name__ == "__main__":
    main()
