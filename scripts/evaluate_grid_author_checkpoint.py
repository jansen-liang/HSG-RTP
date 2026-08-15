#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.grid_baseline import evaluate_grid_dataset
from pipeline.utils.scene_loader import load_scenes
from scripts.convert_hsg_rtp_to_grid import build_robot_graph, build_scene_graph


class GridAuthorPredictor:
    def __init__(
        self,
        author_root: Path,
        config_path: Path,
        checkpoint: Path,
        device: torch.device,
    ) -> None:
        sys.path.insert(0, str(author_root))
        from arguments import Config
        from dataloader.data_preprocessor import data_preprocessor
        from framework import LitRIDSG

        self.device = device
        self.config = Config(str(config_path))
        self.config.dataset_size = 1
        self.config.preprocessor_show_progress_bar = False
        self.config.rg_encoder_in_channels = (
            self.config.sg_encoder_in_channels
        ) = self.config.lm_sentence_embedding_dim * 2
        runtime_args = SimpleNamespace(gpu_devices=[str(device)])
        self.processor = data_preprocessor(runtime_args, self.config)
        self.model = LitRIDSG.load_from_checkpoint(
            str(checkpoint),
            config=self.config,
            save_dir=str(checkpoint.parent),
            map_location=device,
        ).to(device)
        self.model.eval()
        self.action_names = list(self.processor.action_list)

    def predict(self, instruction: str, local_view: dict[str, Any]) -> tuple[str, float]:
        scene_graph, _ = build_scene_graph(local_view)
        robot_graph = build_robot_graph(local_view)
        batch = self.processor.preprocess_input_once(
            [robot_graph, scene_graph, instruction]
        )
        started = time.perf_counter()
        with torch.inference_mode():
            action_logits, object_logits = self.model(batch)
        elapsed = time.perf_counter() - started
        action_name = self.action_names[int(action_logits[0].argmax().item())]
        if action_name == "finish":
            return "finish", elapsed

        node_mask = batch["input"]["scene_graph"]["node_index_mask"][0].bool()
        object_scores = object_logits[0].masked_fill(node_mask, float("-inf"))
        target_index = int(object_scores.argmax().item())
        target = scene_graph["nodes"][target_index]["attributes"]["entity_id"]
        if action_name == "place":
            agent_state = str(local_view["agent"].get("state", "hand-free"))
            held_object = (
                agent_state[len("holding-") :]
                if agent_state.startswith("holding-")
                else "empty_hand"
            )
            return f"place({held_object}, {target})", elapsed
        return f"{action_name}({target})", elapsed


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a retrained GRID author-code checkpoint by direct action rollout"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, default=Path("/home/swzz/disk2T/grid/ridsg"))
    parser.add_argument("--config", type=Path, default=Path("/home/swzz/disk2T/grid/ridsg/hparams_hsg_rtp.cfg"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    records = load_records(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))
    predictor = GridAuthorPredictor(
        args.author_root.resolve(),
        args.config.resolve(),
        args.checkpoint.resolve(),
        torch.device(args.device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "task_results.jsonl"
    results_path.write_text("", encoding="utf-8")
    completed = 0

    def progress(result: dict[str, Any]) -> None:
        nonlocal completed
        completed += 1
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(
            f"[{completed}/{len(records)}] index={result['index']} "
            f"type={result['task_type']} exec={int(result['exec_success'])} "
            f"calls={result['model_calls']} time={result['inference_time']:.2f}s",
            flush=True,
        )

    _, summary = evaluate_grid_dataset(
        records,
        scenes,
        predictor,
        max_steps=args.max_steps,
        progress_callback=progress,
    )
    summary.update(
        {
            "method": "GRID",
            "method_label": "local author-code adaptation (RN50, retrained)",
            "checkpoint": str(args.checkpoint.resolve()),
            "dataset": str(args.dataset.resolve()),
            "controller_profile": "strict_direct_action_rollout",
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
