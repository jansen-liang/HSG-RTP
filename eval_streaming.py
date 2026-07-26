# eval_streaming.py - 流式任务级评估脚本

import torch
import argparse
import os
import sys
from typing import List, Tuple, Dict, Any
import time
import json
import re
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataloader_streaming import StreamingSceneGraphDataLoader
from utils.streaming_hlr import StreamingSceneInstructionQwenModel
from peft import LoraConfig, get_peft_model, TaskType


# ======================
# 解析与指标计算
# ======================

def parse_global_action(action_str: str) -> Tuple[str, str]:
    """解析 global 动作: 'goto(room): task()' → ('room', 'task()')"""
    if not isinstance(action_str, str):
        return ("", "")
    action_str = action_str.strip()
    parts = action_str.split(": ", 1)
    if len(parts) != 2:
        parts = action_str.split(":", 1)
    if len(parts) != 2:
        return ("", action_str)
    goto_part, task_part = parts
    if goto_part.startswith("goto(") and goto_part.endswith(")"):
        location = goto_part[5:-1].strip()
    else:
        location = goto_part.strip()
    return (location, task_part.strip())


def parse_local_action(action_str: str) -> Tuple[str]:
    """解析 local 动作: 'scan(lobby)' → ('scan(lobby)',)"""
    if not isinstance(action_str, str):
        return ("",)
    return (action_str.strip(),)

def parse_task_sequence(text: str) -> List[str]:
    """
    将任务文本序列（如 'goto(kitchen) - pick(apple)'）解析为动作列表。
    """
    if not isinstance(text, str) or not text:
        return []
    # 根据 stream_validate 中的逻辑，任务是用 " - " 连接的
    return [s.strip() for s in text.split(" - ") if s.strip()]

def compute_lcs_length(seq1: List[Tuple], seq2: List[Tuple]) -> int:
    """计算最长公共子序列长度"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_action_jaccard(pred_text: str, target_text: str) -> float:
    """
    计算两个任务序列文本的 Jaccard 相似度
    """
    # 使用 parse_task_sequence 将文本转为列表
    pred_seq = set(parse_task_sequence(pred_text))
    target_seq = set(parse_task_sequence(target_text))
    
    if len(target_seq) == 0:
        return 1.0 if len(pred_seq) == 0 else 0.0
    
    intersection = len(pred_seq & target_seq)
    union = len(pred_seq | target_seq)
    return intersection / union if union > 0 else 0.0


def compute_lcs_ratio(pred_text: str, target_text: str) -> float:
    """
    计算两个任务序列文本的最长公共子序列 (LCS) 比例
    """
    pred_seq = parse_task_sequence(pred_text)
    target_seq = parse_task_sequence(target_text)
    
    if len(target_seq) == 0:
        return 1.0 if len(pred_seq) == 0 else 0.0
        
    # 复用已有的 compute_lcs_length
    lcs_len = compute_lcs_length(pred_seq, target_seq)
    return lcs_len / len(target_seq)

# ======================
# 主函数
# ======================
def stream_validate(model, dataloader, rank=0, max_samples=500):
    """流式验证：基于 task_id 重建完整任务"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_jaccard = 0.0
    total_lcs_ratio = 0.0
    total_mode_accuracy = 0.0
    total_inference_time = 0.0
    completed_tasks = 0

    # 缓存: task_id -> list of (pred, target)
    task_buffer = {}
    current_task_id = None

    def flush_task(task_id):
        nonlocal total_jaccard, total_lcs_ratio, total_mode_accuracy, completed_tasks
  
        if task_id not in task_buffer or not task_buffer[task_id]['samples']:
            return
        samples = task_buffer[task_id]['samples']
        
        # 从任意一个样本获取完整任务信息（所有样本相同）
        first_sample = samples[0]
        global_target = first_sample['metadata'].get('full_global_plan', [])
        local_target = first_sample['metadata'].get('full_subtasks', [])

        # 提取所有预测
        global_preds = []
        local_preds = []
        modes = []
        for s in samples:
            # print(s)
            pred_text = s['pred']
            true_mode = s['metadata']['mode']  # 真实 mode
            modes.append(true_mode)
            # 解析预测
            try:
                parsed = json.loads(re.search(r'\{.*?\}', pred_text, re.DOTALL).group()) if re.search(r'\{.*?\}', pred_text, re.DOTALL) else {}
            except (json.JSONDecodeError, AttributeError):
                parsed = {}
            pred_mode = parsed.get('mode', 'unknown')
            task_content = parsed.get('task', '')
            
            # 调试信息：打印原始 task_content 类型
            print(f"[DEBUG] task_content type: {type(task_content)}, value: {task_content}")
            
            # 处理 task_content 可能是列表、字典等各种情况
            if isinstance(task_content, list):
                # 列表：连接成字符串
                task_content = ' - '.join(str(t) for t in task_content if t)
            elif isinstance(task_content, dict):
                # 字典：尝试提取有用信息
                if 'action' in task_content:
                    task_content = task_content.get('action', '')
                else:
                    task_content = str(task_content)
            elif not isinstance(task_content, str):
                # 其他类型：转换为字符串
                task_content = str(task_content) if task_content else ''
            
            if pred_mode == 'global':
                global_preds.append(task_content)
            elif pred_mode == 'local':
                local_preds.append(task_content)

        # === Global 验证 ===
        global_pred_str = ""
        if global_preds:
            # 确保 global_preds[0] 是字符串
            first_pred = global_preds[0]
            if isinstance(first_pred, list):
                global_pred_str = ' - '.join(str(p) for p in first_pred if p)
            elif isinstance(first_pred, str):
                global_pred_str = first_pred.strip()
            else:
                global_pred_str = str(first_pred) if first_pred else ""
        else:
            # 回退策略 1：使用第一个样本的原始预测（即使解析失败）
            if samples:
                global_pred_str = samples[0]['pred'].strip()
            else:
                global_pred_str = ""  # 极端情况：无样本

        # 安全地分割步骤（即使为空字符串）
        global_pred_steps = [s.strip() for s in global_pred_str.split(" - ") if s.strip()]

        # 同样处理 global_target（你已处理，但再确认）
        global_target_steps = []
        if global_target:
            global_target_steps = [s.strip() for s in global_target if s.strip()]
        # Jaccard
        # gl_pred_actions = set(auto_tokenize(global_pred_steps))
        # gl_target_actions = set(auto_tokenize(global_target_steps))
        
        # === 解析目标动作序列 ===
        gl_target_actions = set()
        for step in global_target_steps:
            if isinstance(step, str) and step.strip():
                # 每个 step 是一个完整动作字符串，如 "goto(room): task()"
                gl_target_actions.add(parse_global_action(step.strip()))

        # === 解析预测动作序列 ===
        gl_pred_actions = set()
        for step in global_pred_steps:
            if isinstance(step, str) and step.strip():
                # 同样，每个 step 应该是一个动作字符串
                # 但注意：你的 global_pred_steps 现在是自由文本！⚠️
                gl_pred_actions.add(parse_global_action(step.strip()))
        if len(gl_target_actions) == 0:
            jaccard = 1.0 if len(gl_pred_actions) == 0 else 0.0
        else:
            jaccard = len(gl_pred_actions & gl_target_actions) / len(gl_pred_actions | gl_target_actions)

        # LCS
        # gl_pred_seq = auto_tokenize(global_pred_steps)
        # gl_target_seq = auto_tokenize(global_target_steps)
        gl_pred_seq = [parse_global_action(s) for s in global_pred_steps if isinstance(s, str) and s.strip()]
        gl_target_seq = [parse_global_action(s) for s in global_target_steps if isinstance(s, str) and s.strip()]

        if len(gl_target_seq) == 0:
            lcs_ratio = 1.0 if len(gl_pred_seq) == 0 else 0.0
        else:
            lcs_len = compute_lcs_length(gl_pred_seq, gl_target_seq)
            lcs_ratio = lcs_len / len(gl_target_seq)
        
        # === Local 验证 ===
        local_pred_seq = [a for a in local_preds if isinstance(a, str) and a.strip()]
        local_target_seq = [a for a in local_target if isinstance(a, str) and a.strip()]

        # 解析每个动作字符串为 (location, task) 元组
        # 注意：local 动作没有 location，所以用 parse_local_action
        lc_pred_actions = set(
            parse_local_action(s) for s in local_pred_seq
        )
        lc_target_actions = set(
            parse_local_action(s) for s in local_target_seq
        )

        # Jaccard
        if len(lc_target_actions) == 0:
            jaccard = 1.0 if len(lc_pred_actions) == 0 else 0.0
        else:
            jaccard = len(lc_pred_actions & lc_target_actions) / len(lc_pred_actions | lc_target_actions)

        # LCS（顺序敏感）
        lc_pred_seq_parsed = [parse_local_action(s) for s in local_pred_seq]
        lc_target_seq_parsed = [parse_local_action(s) for s in local_target_seq]

        if len(lc_target_seq_parsed) == 0:
            lcs_ratio = 1.0 if len(lc_pred_seq_parsed) == 0 else 0.0
        else:
            lcs_len = compute_lcs_length(lc_pred_seq_parsed, lc_target_seq_parsed)
            lcs_ratio = lcs_len / len(lc_target_seq_parsed)
        # Mode accuracy
        mode_correct_steps = 0
        for sample in samples:

            pred_mode = extract_mode_from_text(sample['pred'])
            target_mode = sample['metadata']['mode']
            if pred_mode == target_mode:
                mode_correct_steps += 1
        mode_accuracy = mode_correct_steps / len(samples) if samples else 1.0
        print(f"Jaccard: {jaccard}, LCS: {lcs_ratio}, Mode Accuracy: {mode_accuracy}")
        total_jaccard += jaccard
        total_lcs_ratio += lcs_ratio
        total_mode_accuracy += mode_accuracy
        completed_tasks += 1
        del task_buffer[task_id]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Stream Validating", disable=(rank != 0))):
            if completed_tasks >= max_samples:
                break

            instructions = batch['instructions']
            scene_graphs = batch['scene_graphs']
            completed = batch['completed']
            pending = batch['pending']
            target_subtasks = batch['subtasks']
            metadata_batch = batch.get('metadata', [{}] * len(instructions))

            # Compute loss
            outputs = model(
                instructions=instructions,
                completed=completed,
                pending=pending,
                scene_graphs=scene_graphs,
                target_subtasks=target_subtasks
            )
            loss = outputs["loss"]

            batch_size = len(instructions)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            # Generate predictions
            torch.cuda.synchronize()
            start_time = time.time()
            outputs_gen = model(
                instructions=instructions,
                completed=completed,
                pending=pending,
                scene_graphs=scene_graphs,
                target_subtasks=None
            )
            torch.cuda.synchronize()
            end_time = time.time()
            total_inference_time += (end_time - start_time)

            generated_texts = outputs_gen.get("predictions", [""] * len(instructions))

            # 按 task_id 分组
            for i in range(len(instructions)):
                if completed_tasks >= max_samples:
                    break
                pred = generated_texts[i]
                target = target_subtasks[i]
                task_id = metadata_batch[i].get('task_id', f"unknown_{batch_idx}_{i}")
                metadata = metadata_batch[i]
                print(f"[RAW PRED] {pred}")
                print(f"[TARGET] {target}")
                print(f"[DEBUG] Processing task_id={task_id}, step={metadata_batch[i].get('step_in_task', 'N/A')}")
                if current_task_id is not None and current_task_id != task_id:
                    flush_task(current_task_id)
                if task_id not in task_buffer:
                    task_buffer[task_id] = {'samples': []}
                task_buffer[task_id]['samples'].append({
                'pred': pred,
                'metadata': metadata
                })
                current_task_id = task_id

            if completed_tasks >= max_samples:
                break

        # Flush last task
        if current_task_id is not None and completed_tasks < max_samples:
            flush_task(current_task_id)

    # Final metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_jaccard = total_jaccard / completed_tasks if completed_tasks > 0 else 0.0
    avg_lcs_ratio = total_lcs_ratio / completed_tasks if completed_tasks > 0 else 0.0
    avg_mode_accuracy = total_mode_accuracy / completed_tasks if completed_tasks > 0 else 0.0
    avg_inference_time = total_inference_time / (batch_idx + 1) if batch_idx >= 0 else 0.0

    if rank == 0:
        print(f"\n[Stream Validation Summary]")
        print(f"  Completed Tasks:   {completed_tasks}")
        print(f"  Avg Loss:          {avg_loss:.4f}")
        print(f"  Avg Jaccard:       {avg_jaccard:.4f}")
        print(f"  Avg LCS%:          {avg_lcs_ratio:.4f}")
        print(f"  Avg Mode Accuracy: {avg_mode_accuracy:.4f}")
        print(f"  Avg Inference Time per Batch: {avg_inference_time:.4f}s")

    torch.cuda.empty_cache()
    return avg_loss, {
        'jaccard': avg_jaccard,
        'lcs_ratio': avg_lcs_ratio,
        'mode_accuracy': avg_mode_accuracy,
        'inference_time': avg_inference_time
    }


# ======================
# 辅助函数（JSON解析）
# ======================

def extract_mode_from_text(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    try:
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            json_str = re.sub(r'",\s*"",\s*""', '"', json_str)
            json_str = re.sub(r',\s*""\s*,\s*""', '', json_str)
            json_str = re.sub(r',\s*""', '', json_str)
            json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
            try:
                parsed = json.loads(json_str)
                return parsed.get('mode', 'unknown')
            except json.JSONDecodeError:
                mode_match = re.search(r'"mode":\s*"([^"]*)"', json_str)
                if mode_match:
                    return mode_match.group(1)
    except Exception:
        pass
    return "unknown"


def clean_prediction_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    try:
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            json_str = re.sub(r'",\s*"",\s*""', '"', json_str)
            json_str = re.sub(r',\s*""\s*,\s*""', '', json_str)
            json_str = re.sub(r',\s*""', '', json_str)
            json_str = json_str.replace('\\"', '"').replace('\\\\', '\\')
            try:
                parsed = json.loads(json_str)
                task_content = parsed.get('task', '')
                if task_content:
                    return task_content.strip()
            except json.JSONDecodeError:
                task_match = re.search(r'"task":\s*"([^"]*)"', json_str)
                if task_match:
                    return task_match.group(1)
    except Exception:
        pass
    return text.strip()


# ======================
# 主函数
# ======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model_path', type=str, default=os.getenv('HLR_MODEL_PATH', 'Qwen/Qwen3-8B'), help='Base model path')
    parser.add_argument('--data_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--max_samples', type=int, default=100, help='Max complete tasks to evaluate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = StreamingSceneInstructionQwenModel(llm_model_name=args.model_path)

    # Load checkpoint
    resume_path = args.checkpoint
    print(f"Resuming from checkpoint: {resume_path}")
    train_state_path = os.path.join(resume_path, 'training_state.pt')
    train_state = torch.load(train_state_path, map_location='cpu')
    start_epoch = train_state['epoch']

    if 'additional_components' in train_state:
        additional_state = train_state['additional_components']
        if 'graph_encoder' in additional_state and hasattr(model, 'graph_encoder'):
            model.graph_encoder.load_state_dict(additional_state['graph_encoder'])
        if 'graph_proj' in additional_state and hasattr(model, 'graph_proj'):
            model.graph_proj.load_state_dict(additional_state['graph_proj'])
        if 'new_token_embeddings' in additional_state:
            embed_layer = model.llm.get_input_embeddings()
            original_vocab_size = 151669
            current_vocab_size = embed_layer.weight.shape[0]
            if current_vocab_size > original_vocab_size:
                new_embeddings = additional_state['new_token_embeddings']
                embed_layer.weight.data[original_vocab_size:] = new_embeddings
                print(f"Loaded {new_embeddings.shape[0]} new token embeddings")
        if 'instruction_encoder' in additional_state and hasattr(model, 'instruction_encoder'):
            model.instruction_encoder.load_state_dict(additional_state['instruction_encoder'], strict=False)
            print("Instruction encoder weights loaded")
    print(f"Resuming from epoch {start_epoch}")

    # Load LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias="none"
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.load_adapter(resume_path, adapter_name="default")
    print("LoRA weights loaded")

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Create dataloader (shuffle=False for validation!)
    print(f"Loading data from {args.data_path}")
    val_dataloader = StreamingSceneGraphDataLoader(
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        chunk_size=50,
        shuffle=False,  
        num_workers=args.num_workers,
        rank=0,
        world_size=1,
        distributed=False,
        seed=None
    )

    print(f"Starting evaluation with max {args.max_samples} complete tasks...")
    avg_loss, metrics = stream_validate(model, val_dataloader, rank=0, max_samples=args.max_samples)

    print(f"\n=== Final Results ===")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Jaccard: {metrics['jaccard']:.4f}")
    print(f"LCS Ratio: {metrics['lcs_ratio']:.4f}")
    print(f"Mode Accuracy: {metrics['mode_accuracy']:.4f}")
    print(f"Inference Time: {metrics['inference_time']:.4f}s")
