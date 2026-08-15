#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.external_baselines import HuggingFaceJSONBackend
from evaluation.saycan_baseline import (
    SayCanAdaptationPredictor,
    evaluate_saycan_dataset,
)
from pipeline.utils.scene_loader import load_scenes


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a SayCan official-notebook adaptation"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--option-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_records(args.dataset)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    records = records[args.start_index :]
    if args.limit is not None:
        records = records[: args.limit]
    scenes = load_scenes(sorted({record["scene_name"] for record in records}))
    backend = HuggingFaceJSONBackend(
        args.model_path,
        max_input_tokens=args.max_input_tokens,
        option_batch_size=args.option_batch_size,
    )
    predictor = SayCanAdaptationPredictor(backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "task_results.jsonl"
    results_path.write_text("", encoding="utf-8")
    completed = 0

    def progress(result: dict) -> None:
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

    _, summary = evaluate_saycan_dataset(
        records,
        scenes,
        predictor,
        max_steps=args.max_steps,
        progress_callback=progress,
        start_index=args.start_index,
    )
    summary.update(
        {
            "method": "SayCan",
            "method_label": (
                "official-code adaptation (notebook; symbolic affordance adapter)"
            ),
            "official_repository": (
                "https://github.com/google-research/google-research/tree/master/saycan"
            ),
            "official_commit": "a0080d35561b0a02504bf303edc4ba7f8011b5f8",
            "backbone": args.model_name,
            "model_path": str(Path(args.model_path).resolve()),
            "dataset": str(args.dataset.resolve()),
            "start_index": args.start_index,
            "seed": args.seed,
            "controller_profile": "strict_direct_skill_rollout",
            "max_input_tokens": args.max_input_tokens,
            "max_steps": args.max_steps,
            "option_batch_size": args.option_batch_size,
            "affordance_model": "symbolic action preconditions",
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
