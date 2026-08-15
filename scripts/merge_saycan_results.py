#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.saycan_baseline import summarize_saycan_results


METHOD_LABEL = "official-code adaptation (notebook; symbolic affordance adapter)"
OFFICIAL_COMMIT = "a0080d35561b0a02504bf303edc4ba7f8011b5f8"


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge indexed SayCan result shards")
    parser.add_argument("--part", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-tasks", type=int, default=70)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--option-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_index: dict[int, dict] = {}
    for path in args.part:
        for result in load_results(path):
            index = int(result["index"])
            if index in by_index:
                raise ValueError(f"Duplicate SayCan result index: {index}")
            by_index[index] = result
    expected_indices = list(range(args.expected_tasks))
    if sorted(by_index) != expected_indices:
        missing = sorted(set(expected_indices) - set(by_index))
        extra = sorted(set(by_index) - set(expected_indices))
        raise ValueError(f"SayCan shard indices mismatch: missing={missing}, extra={extra}")
    results = [by_index[index] for index in expected_indices]
    summary = summarize_saycan_results(results)
    summary.update(
        {
            "method": "SayCan",
            "method_label": METHOD_LABEL,
            "official_repository": (
                "https://github.com/google-research/google-research/tree/master/saycan"
            ),
            "official_commit": OFFICIAL_COMMIT,
            "backbone": args.model_name,
            "model_path": str(Path(args.model_path).resolve()),
            "dataset": str(args.dataset.resolve()),
            "seed": args.seed,
            "controller_profile": "strict_direct_skill_rollout",
            "max_input_tokens": args.max_input_tokens,
            "max_steps": None,
            "option_batch_size": args.option_batch_size,
            "affordance_model": "symbolic action preconditions",
            "merged_parts": [str(path.resolve()) for path in args.part],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
