# inference.py

import torch
import argparse
import time
import json
import os
import deepspeed
from collections import defaultdict
from tqdm import tqdm
from utils.hlr import SceneInstructionQwenModel
from eval import tokenize_action_sequence, compute_action_jaccard, compute_lcs_ratio
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


def load_samples(jsonl_path: str, max_samples: int = None):
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            # Try loading as a single JSON array first
            samples = json.load(f)
            if not isinstance(samples, list):
                # If it's not a list, maybe it's a single object or something else, 
                # but let's assume if json.load works it returns the data.
                # If it was meant to be JSONL, json.load might fail or return one object.
                pass
    except json.JSONDecodeError:
        # Fallback to JSONL
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
            
    return samples[:max_samples] if max_samples else samples


def get_difficulty(sample) -> str:
    diff = sample.get('difficulty', 'unknown')
    if isinstance(diff, int):
        return {1: 'easy', 2: 'medium', 3: 'hard'}.get(diff, f'level_{diff}')
    return str(diff).lower()


def format_target(subtasks):
    if not subtasks:
        return None
    if isinstance(subtasks, list):
        return " -> ".join(subtasks)
    return str(subtasks)


def load_lora_model(model, lora_path: str, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """Inject LoRA and load weights without replacing the entire LLM."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias="none"
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.load_adapter(lora_path, adapter_name="default")
    
    # Load additional components if they exist
    train_state_path = os.path.join(lora_path, 'training_state.pt')
    if os.path.exists(train_state_path):
        train_state = torch.load(train_state_path, map_location='cpu')
        
        if 'additional_components' in train_state:
            additional_state = train_state['additional_components']
            
            # Load scene graph encoder
            if 'graph_encoder' in additional_state and hasattr(model, 'graph_encoder'):
                model.graph_encoder.load_state_dict(additional_state['graph_encoder'])
                print("Scene graph encoder weights loaded for inference")
            
            # Load graph projection layer
            if 'graph_proj' in additional_state and hasattr(model, 'graph_proj'):
                model.graph_proj.load_state_dict(additional_state['graph_proj'])
                print("Graph projection weights loaded for inference")
            
            # Load new token embeddings
            if 'new_token_embeddings' in additional_state:
                embed_layer = model.llm.get_input_embeddings()
                original_vocab_size = 151669
                current_vocab_size = embed_layer.weight.shape[0]
                if current_vocab_size > original_vocab_size:
                    new_embeddings = additional_state['new_token_embeddings']
                    embed_layer.weight.data[original_vocab_size:] = new_embeddings
                    print(f"Loaded {new_embeddings.shape[0]} new token embeddings for inference")
            
            # Load instruction encoder parameters if they exist
            if 'instruction_encoder' in additional_state and hasattr(model, 'instruction_encoder'):
                model.instruction_encoder.load_state_dict(additional_state['instruction_encoder'], strict=False)
                print("Instruction encoder weights loaded for inference")
    
    model.llm.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run inference with HLR model")
    parser.add_argument('--model_path', type=str, default=os.getenv('HLR_MODEL_PATH', 'Qwen/Qwen3-8B'))
    # text_model_name 已移除，现在直接使用 Qwen 的文本编码器
    parser.add_argument('--data_path', type=str, default=os.getenv('HLR_EVAL_DATA', 'pipeline/output/test.jsonl'))
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_checkpoint', type=str, default=os.getenv('HLR_LORA_CHECKPOINT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--deepspeed_config', type=str, default=None, help='DeepSpeed config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--output_json', type=str, default=None, 
                       help='Path to save inference results as JSON file')
    args = parser.parse_args()

    # Initialize distributed training if DeepSpeed is used
    if args.deepspeed_config:
        deepspeed.init_distributed()
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda', args.local_rank)
    
    # Load base model
    if args.local_rank <= 0:  # Only print on main process
        print("Loading model...")
    model = SceneInstructionQwenModel(
        llm_model_name=args.model_path
    )

    # Load LoRA if enabled
    if args.use_lora and args.lora_checkpoint:
        if args.local_rank <= 0:
            print(f"Injecting LoRA from {args.lora_checkpoint}")
        model = load_lora_model(
            model,
            lora_path=args.lora_checkpoint,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    else:
        if args.local_rank <= 0:
            print("Using base model (no LoRA)")

    # Critical: Ensure instruction encoder uses the same embedding layer as LLM
    model.instruction_encoder.set_embed_tokens(model.llm.get_input_embeddings())
    
    # Initialize DeepSpeed if config is provided
    if args.deepspeed_config:
        model, _, _, _ = deepspeed.initialize(
            model=model,
            config_params=args.deepspeed_config
        )
        model.eval()
        if args.local_rank <= 0:
            print("✅ Model loaded with DeepSpeed and ready.")
    else:
        model.to(args.device).eval()
        if args.local_rank <= 0:
            print("✅ Model loaded and ready.")

    # Load data (only on main process or when not using DeepSpeed)
    if args.local_rank <= 0 or not args.deepspeed_config:
        samples = load_samples(args.data_path, args.max_samples)
        if args.local_rank <= 0:
            print(f"Total samples: {len(samples)}\n")
    else:
        samples = []  # Other processes don't need the data

    # Only run inference on main process to avoid duplicate outputs
    if args.local_rank <= 0 or not args.deepspeed_config:
        # Inference loop
        stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})
        total_time, total_count = 0.0, 0
        results = []  # Collect results for JSON export

        with torch.no_grad():
            # 添加进度条
            pbar = tqdm(enumerate(samples), total=len(samples), desc="Inference Progress", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for idx, sample in pbar:
                instruction = sample['instruction']
                scene_graph = sample.get('initial_state', {})  # 旧数据格式使用 initial_state
                target = format_target(sample.get('subtasks'))

                print("\n" + "=" * 60)
                print(f"Instruction: {instruction}")
                if target:
                    print(f"Target: {target}")
                print("=" * 60)

                difficulty = get_difficulty(sample)
                torch.cuda.synchronize() if args.device == 'cuda' else None
                start = time.time()

                output = model(
                    instructions=[instruction],
                    scene_graphs=[scene_graph],
                    target_subtasks=None
                )

                torch.cuda.synchronize() if args.device == 'cuda' else None
                elapsed = time.time() - start

                pred = output.get("predictions", [""])[0]
                print(f"Prediction: {pred}")
                print(f"Inference Time: {elapsed:.4f} seconds")

                # Collect result for JSON export
                result_item = {
                    "id": idx,
                    "instruction": instruction,
                    "prediction": pred,
                    "target": target,
                    "difficulty": difficulty,
                    "inference_time": elapsed,
                    "scene_graph": scene_graph
                }
                results.append(result_item)

                # Update stats
                stats[difficulty]['count'] += 1
                stats[difficulty]['total_time'] += elapsed
                total_time += elapsed
                total_count += 1

                # 更新进度条显示信息
                avg_time = total_time / total_count if total_count > 0 else 0
                pbar.set_postfix({
                    'avg_time': f'{avg_time:.3f}s',
                    'current_time': f'{elapsed:.3f}s',
                    'difficulty': difficulty
                })

        # Final report
        print("\n" + "=" * 70)
        print("📊 INFERENCE TIME ANALYSIS BY DIFFICULTY")
        print("=" * 70)
        print(f"{'Difficulty':<12} | {'# Samples':<10} | {'Avg Time (s)':<12} | {'Total Time (s)':<14}")
        print("-" * 70)

        for diff in sorted(stats):
            s = stats[diff]
            avg = s['total_time'] / s['count'] if s['count'] else 0
            print(f"{diff:<12} | {s['count']:<10} | {avg:<12.4f} | {s['total_time']:<14.2f}")

        overall_avg = total_time / total_count if total_count else 0
        print("-" * 70)
        print(f"{'Overall':<12} | {total_count:<10} | {overall_avg:<12.4f} | {total_time:<14.2f}")
        print("=" * 70)
        
        # Save results to JSON if output path is specified
        if args.output_json:
            # Add summary statistics to the JSON
            summary = {
                "total_samples": total_count,
                "total_time": total_time,
                "average_time": overall_avg,
                "stats_by_difficulty": {}
            }
            
            for diff in sorted(stats):
                s = stats[diff]
                avg = s['total_time'] / s['count'] if s['count'] else 0
                summary["stats_by_difficulty"][diff] = {
                    "count": s['count'],
                    "total_time": s['total_time'],
                    "average_time": avg
                }
            
            output_data = {
                "summary": summary,
                "results": results
            }
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Results saved to: {args.output_json}")
        
        print("\n✅ Inference completed successfully.")
    
    # Synchronize all processes if using DeepSpeed
    if args.deepspeed_config:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
