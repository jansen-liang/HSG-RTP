# eval.py

import torch
from tqdm import tqdm
from utils.dataloader import SceneGraphDataLoader
from typing import List, Dict, Any
import time


def tokenize_action_sequence(seq: str) -> List[str]:
    if not isinstance(seq, str):
        return []
    return [action.strip() for action in seq.split(" -> ") if action.strip()]


def compute_action_jaccard(pred: str, target: str) -> float:
    pred_actions = set(tokenize_action_sequence(pred))
    target_actions = set(tokenize_action_sequence(target))
    
    if len(target_actions) == 0:
        return 1.0 if len(pred_actions) == 0 else 0.0

    intersection = pred_actions & target_actions
    union = pred_actions | target_actions
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_lcs_ratio(pred: str, target: str) -> float:
    pred_actions = tokenize_action_sequence(pred)
    target_actions = tokenize_action_sequence(target)

    if len(target_actions) == 0:
        return 1.0 if len(pred_actions) == 0 else 0.0

    lcs_len = compute_lcs_length(pred_actions, target_actions)
    return lcs_len / len(target_actions)


def format_targets(target_subtasks: list) -> List[str]:
    formatted_targets = []
    for action_sequence in target_subtasks:
        if isinstance(action_sequence, list):
            formatted_text = " -> ".join(action_sequence)
        else:
            formatted_text = str(action_sequence)
        formatted_targets.append(formatted_text)
    return formatted_targets


def validate(model, dataloader, device, rank=0, world_size=1):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_jaccard = 0.0
    total_lcs_ratio = 0.0
    total_inference_time = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", disable=(rank != 0))):
            try:
                instructions = batch['instructions']
                scene_graphs = batch['scene_graphs']
                target_subtasks = batch['subtasks']
                formatted_targets = format_targets(target_subtasks)

                # Compute loss
                outputs = model(
                    instructions=instructions,
                    scene_graphs=scene_graphs,
                    target_subtasks=formatted_targets
                )
                loss = outputs["loss"]
                batch_size = len(instructions)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Generate predictions and measure inference time
                torch.cuda.synchronize()
                start_time = time.time()
                outputs_gen = model(instructions, scene_graphs, target_subtasks=None)
                torch.cuda.synchronize()
                end_time = time.time()
                total_inference_time += (end_time - start_time)

                generated_texts = outputs_gen.get("predictions", [""] * len(instructions))

                for i, (pred, target) in enumerate(zip(generated_texts, formatted_targets)):
                    jaccard = compute_action_jaccard(pred, target)
                    lcs_ratio = compute_lcs_ratio(pred, target)
                    total_jaccard += jaccard
                    total_lcs_ratio += lcs_ratio

                    if rank == 0 and batch_idx == 0 and i < len(generated_texts):
                        print(f"\n--- Validation Sample {i+1} ---")
                        print(f"Target:  {target}")
                        print(f"Predict: {pred}")
                        print(f"Jaccard: {jaccard:.4f}, LCS%: {lcs_ratio:.4f}")

            except Exception as e:
                print(f"[RANK {rank}] Validation error in batch {batch_idx}: {e}")
                continue

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_jaccard = total_jaccard / total_samples if total_samples > 0 else 0.0
    avg_lcs_ratio = total_lcs_ratio / total_samples if total_samples > 0 else 0.0
    avg_inference_time = total_inference_time / len(dataloader) if len(dataloader) > 0 else 0.0

    if rank == 0:
        print(f"\n[Validation Summary]")
        print(f"  Samples:           {total_samples}")
        print(f"  Avg Loss:          {avg_loss:.4f}")
        print(f"  Avg Jaccard:       {avg_jaccard:.4f}")
        print(f"  Avg LCS%:          {avg_lcs_ratio:.4f}")
        print(f"  Avg Inference Time per Batch: {avg_inference_time:.4f}s")

    torch.cuda.empty_cache()

    return avg_loss, {
        'jaccard': avg_jaccard,
        'lcs_ratio': avg_lcs_ratio,
        'inference_time': avg_inference_time
    }