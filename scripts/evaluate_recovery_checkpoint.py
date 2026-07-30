import argparse
from collections import Counter, defaultdict
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

from evaluation.perturbations import FailActionOncePerturbation
from evaluation.policies import StreamingModelPolicy
from evaluation.recovery import RecoveryConfig
from evaluation.rollout_evaluator import rollout_policy
from pipeline.utils.scene_loader import load_scenes
from utils.streaming_hlr import StreamingSceneInstructionQwenModel


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_model(model_path: str, checkpoint: Path, device: torch.device) -> Any:
    model = StreamingSceneInstructionQwenModel(llm_model_name=model_path)
    training_state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu"
    )
    additional_state = training_state.get("additional_components", {})
    component_names = (
        "graph_encoder",
        "graph_proj",
        "instruction_encoder",
    )
    for component_name in component_names:
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


def trace_has(trace: list[dict[str, Any]], event: str) -> bool:
    return any(record.get("event") == event for record in trace)


def summarize_result(
    spec: dict[str, Any], execution: Any
) -> dict[str, Any]:
    trace = list(execution.recovery_trace)
    initial_attempts = sum(
        record.get("event") == "initial_global_plan_attempt" for record in trace
    )
    automatic_retry_success = any(
        record.get("event") == "action_success"
        and record.get("automatic_retry")
        for record in trace
    )
    return {
        "trial": spec.get("trial"),
        "seed": spec["seed"],
        "index": spec["index"],
        "scene": spec["scene"],
        "task_type": spec["task_type"],
        "injected_action": spec["injected_action"],
        "success": execution.success,
        "failure_type": execution.failure_type,
        "failure_message": execution.failure_message,
        "actions": list(execution.actions),
        "plan": list(execution.plan),
        "initial_plan_attempts": initial_attempts,
        "initial_plan_grounded": trace_has(trace, "initial_global_plan_grounded"),
        "initial_plan_repaired": initial_attempts > 1,
        "feedback_triggered": trace_has(trace, "local_failure"),
        "perturbation_triggered": trace_has(trace, "perturbation"),
        "automatic_retry_attempted": trace_has(trace, "automatic_retry_attempt"),
        "automatic_retry_success": automatic_retry_success,
        "local_recovery_exhausted": trace_has(trace, "local_recovery_exhausted"),
        "global_replan_attempted": trace_has(trace, "global_replan_attempt"),
        "global_replan_success": trace_has(trace, "global_replan_success"),
        "recovery_trace": trace,
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    failure_types = Counter(
        result["failure_type"] or "success" for result in results
    )
    summary = {
        "trials": len(results),
        "successes": sum(result["success"] for result in results),
        "success_rate": (
            sum(result["success"] for result in results) / len(results)
            if results
            else 0.0
        ),
        "initial_plan_grounded": sum(
            result["initial_plan_grounded"] for result in results
        ),
        "initial_plan_repaired": sum(
            result["initial_plan_repaired"] for result in results
        ),
        "feedback_triggered": sum(
            result["feedback_triggered"] for result in results
        ),
        "perturbation_triggered": sum(
            result["perturbation_triggered"] for result in results
        ),
        "automatic_retry_attempted": sum(
            result["automatic_retry_attempted"] for result in results
        ),
        "automatic_retry_successes": sum(
            result["automatic_retry_success"] for result in results
        ),
        "local_recovery_exhausted": sum(
            result["local_recovery_exhausted"] for result in results
        ),
        "global_replan_attempted": sum(
            result["global_replan_attempted"] for result in results
        ),
        "global_replan_successes": sum(
            result["global_replan_success"] for result in results
        ),
        "failure_types": dict(sorted(failure_types.items())),
    }

    grouped = defaultdict(list)
    for result in results:
        grouped[result["task_type"]].append(result)
    summary["by_type"] = {
        task_type: {
            "runs": len(group_results),
            "successes": sum(result["success"] for result in group_results),
            "success_rate": sum(result["success"] for result in group_results)
            / len(group_results),
            "failure_types": dict(
                sorted(
                    Counter(
                        result["failure_type"] or "success"
                        for result in group_results
                    ).items()
                )
            ),
        }
        for task_type, group_results in sorted(grouped.items())
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    records = load_records(args.dataset)
    baseline = json.loads(args.baseline_results.read_text(encoding="utf-8"))
    specs = baseline["results"]
    scenes = load_scenes(sorted({spec["scene"] for spec in specs}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.checkpoint, device)
    policy = StreamingModelPolicy(model)
    recovery_config = RecoveryConfig()

    results = []
    for run_index, spec in enumerate(specs, start=1):
        set_seed(spec["seed"])
        record = records[spec["index"]]
        execution = rollout_policy(
            record,
            scenes[spec["scene"]],
            policy,
            max_steps=args.max_steps,
            recovery_config=recovery_config,
            perturbations=[FailActionOncePerturbation(spec["injected_action"])],
        )
        result = summarize_result(spec, execution)
        results.append(result)
        print(
            f"[{run_index}/{len(specs)}] index={spec['index']} "
            f"type={spec['task_type']} success={result['success']} "
            f"failure={result['failure_type']}",
            flush=True,
        )

    output = {"summary": build_summary(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
