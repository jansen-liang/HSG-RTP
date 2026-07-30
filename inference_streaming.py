# 流式模型推理和评估脚本

import torch
import argparse
import json
import os
import deepspeed
from tqdm import tqdm
from utils.streaming_hlr import StreamingSceneInstructionQwenModel
from utils.dataloader_streaming import StreamingSceneGraphDataLoader
from eval_streaming import stream_validate, extract_mode_from_text, clean_prediction_text, parse_task_sequence, compute_action_jaccard, compute_lcs_ratio
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import time


def load_streaming_lora_model(model, lora_path: str):
    """从 checkpoint 配置加载流式模型的 LoRA 权重。"""
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"LoRA adapter config not found: {adapter_config_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as config_file:
        adapter_config = json.load(config_file)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(adapter_config["r"]),
        lora_alpha=int(adapter_config["lora_alpha"]),
        lora_dropout=float(adapter_config.get("lora_dropout", 0.0)),
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        bias="none"
    )
    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.load_adapter(lora_path, adapter_name="default")
    
    # 加载额外组件
    train_state_path = os.path.join(lora_path, 'training_state.pt')
    if os.path.exists(train_state_path):
        try:
            train_state = torch.load(train_state_path, map_location='cpu')
            if 'additional_components' in train_state:
                additional_state = train_state['additional_components']
                
                # 加载图编码器
                if 'graph_encoder' in additional_state and hasattr(model, 'graph_encoder'):
                    model.graph_encoder.load_state_dict(additional_state['graph_encoder'])
                    print("Graph encoder weights loaded")
                
                # 加载图投影层
                if 'graph_proj' in additional_state and hasattr(model, 'graph_proj'):
                    model.graph_proj.load_state_dict(additional_state['graph_proj'])
                    print("Graph projection weights loaded")
                
                # 加载instruction encoder参数
                if 'instruction_encoder' in additional_state and hasattr(model, 'instruction_encoder'):
                    model.instruction_encoder.load_state_dict(additional_state['instruction_encoder'], strict=False)
                    print("Instruction encoder weights loaded")
                
                # 加载新token embeddings（思考模式tokens）
                if 'new_token_embeddings' in additional_state:
                    embed_layer = model.llm.get_input_embeddings()
                    current_vocab_size = embed_layer.weight.shape[0]
                    original_vocab_size = 151669  # Qwen3-8B原始词汇量
                    
                    if current_vocab_size > original_vocab_size:
                        new_embeddings = additional_state['new_token_embeddings']
                        with torch.no_grad():
                            embed_layer.weight[original_vocab_size:original_vocab_size + new_embeddings.shape[0]] = new_embeddings
                        print(f"Loaded {new_embeddings.shape[0]} new token embeddings (thinking mode tokens)")
        except Exception as e:
            print(f"Warning: Could not load additional components: {e}")


def task_level_inference(model, instruction, scene_graph, target_sequence, rank=0):
    """
    针对完整任务进行推理：
    - Global模式: 一次推理生成完整序列
    - Local模式: 逐步推理，累积生成完整序列
    """
    # 提取目标模式
    target_mode = extract_mode_from_text(target_sequence)
    
    if target_mode == "global":
        # Global模式：一次生成完整序列
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model(
            instructions=[instruction], 
            scene_graphs=[scene_graph], 
            target_subtasks=None
        )
        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        
        prediction = outputs.get("predictions", [""])[0]
        return prediction, inference_time, 1  # 1次推理
        
    elif target_mode == "local":
        # Local模式：逐步推理，组合完整序列
        accumulated_actions = []
        total_inference_time = 0.0
        inference_count = 0
        
        # 解析目标序列获取期望的步骤数
        target_tasks = parse_task_sequence(target_sequence)
        expected_steps = len(target_tasks)
        
        # 逐步推理直到完成或达到最大步数
        max_steps = min(expected_steps * 2, 10)  # 防止无限循环
        
        for step in range(max_steps):
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(
                instructions=[instruction], 
                scene_graphs=[scene_graph], 
                target_subtasks=None
            )
            torch.cuda.synchronize()
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            inference_count += 1
            
            prediction = outputs.get("predictions", [""])[0]
            cleaned_pred = clean_prediction_text(prediction)
            
            if cleaned_pred and cleaned_pred not in accumulated_actions:
                accumulated_actions.append(cleaned_pred)
            
            # 如果达到期望步数或生成了结束标记，停止
            if len(accumulated_actions) >= expected_steps or "finish" in cleaned_pred.lower():
                break
        
        # 组合完整序列
        final_prediction = " -> ".join(accumulated_actions) if accumulated_actions else ""
        return final_prediction, total_inference_time, inference_count
        
    else:
        # None模式或未知模式，按Global处理
        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model(
            instructions=[instruction], 
            scene_graphs=[scene_graph], 
            target_subtasks=None
        )
        torch.cuda.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        
        prediction = outputs.get("predictions", [""])[0]
        return prediction, inference_time, 1


def stream_validate_with_error_tracking(model, dataloader, device, rank=0, world_size=1, max_samples=500):
    """流式数据专用验证函数，按任务级别处理，记录错误案例"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_jaccard = 0.0
    total_lcs_ratio = 0.0
    total_mode_accuracy = 0.0
    total_inference_time = 0.0
    total_inference_count = 0
    sample_count = 0
    error_cases = []
    
    # 重新组织数据：按原始 task_id 分组，每个 task 内保持序列顺序
    tasks_map = {}
    if rank == 0:
        print("🔄 重新组织数据按 task_id 聚合（同一 task 的 streaming samples 保持顺序）...")

    # 收集所有数据并按 task_id 组织
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= max_samples:
            break

        instructions = batch['instructions']
        scene_graphs = batch['scene_graphs']
        target_subtasks = batch['subtasks']
        metadatas = batch.get('metadata', [{} for _ in instructions])

        for i, (instruction, scene_graph, target, metadata) in enumerate(zip(instructions, scene_graphs, target_subtasks, metadatas)):
            if sample_count >= max_samples:
                break

            # 格式化目标（保留原有格式）
            if isinstance(target, list):
                formatted_target = " -> ".join(target)
            else:
                formatted_target = str(target)

            task_id = metadata.get('task_id', metadata.get('task_index', batch_idx))
            step_in_task = metadata.get('step_in_task', i)

            sample_entry = {
                'instruction': instruction,
                'scene_graph': scene_graph,
                'target': formatted_target,
                'metadata': metadata,
                'task_id': task_id,
                'step_in_task': step_in_task,
                'global_sample_idx': sample_count + 1
            }

            tasks_map.setdefault(task_id, []).append(sample_entry)
            sample_count += 1

    if rank == 0:
        print(f"📊 收集到 {len(tasks_map)} 个 top-level tasks（样本总数: {sample_count}）")

    # 按 task 逐个处理：在 task 内遍历 samples，global 单独评估，连续的 local 合并为一个 block 评估
    processed_count = 0
    with torch.no_grad():
        for task_id in tqdm(sorted(tasks_map.keys()), desc="Task-level Inference", disable=(rank != 0)):
            if processed_count >= max_samples:
                break

            samples = sorted(tasks_map[task_id], key=lambda x: x.get('step_in_task', 0))

            # 遍历 samples，识别 local blocks
            idx = 0
            while idx < len(samples) and processed_count < max_samples:
                sample = samples[idx]
                instruction = sample['instruction']
                scene_graph = sample['scene_graph']
                target = sample['target']
                sample_global_idx = sample['global_sample_idx']

                target_mode = extract_mode_from_text(target)

                # 如果是 global 或 unknown/none，则作为单独块处理
                if target_mode != 'local':
                    # 计算 loss（可选）
                    try:
                        loss_outputs = model(
                            instructions=[instruction],
                            scene_graphs=[scene_graph],
                            target_subtasks=[target]
                        )
                        loss = loss_outputs.get("loss")
                        if loss is not None:
                            total_loss += loss.item()
                            total_samples += 1
                    except Exception:
                        pass

                    # 直接推理评估
                    prediction, inference_time, inference_count = task_level_inference(
                        model, instruction, scene_graph, target, rank
                    )

                    total_inference_time += inference_time
                    total_inference_count += inference_count

                    cleaned_pred = clean_prediction_text(prediction)
                    target_tasks = parse_task_sequence(target)
                    main_target = target_tasks[0] if target_tasks else target

                    pred_mode = extract_mode_from_text(prediction)
                    mode_correct = pred_mode == target_mode

                    # 指标按单个目标计算（global/none），需要清理目标文本的模式标记
                    cleaned_target = clean_prediction_text(main_target)
                    jaccard = compute_action_jaccard(cleaned_pred, cleaned_target)
                    lcs_ratio = compute_lcs_ratio(cleaned_pred, cleaned_target)

                    total_jaccard += jaccard
                    total_lcs_ratio += lcs_ratio
                    total_mode_accuracy += 1.0 if mode_correct else 0.0

                    # 错误判定
                    is_error_case = False
                    error_reasons = []
                    if not mode_correct:
                        is_error_case = True
                        error_reasons.append(f"模式错误: 预测{pred_mode} vs 目标{target_mode}")
                    if jaccard < 0.3:
                        is_error_case = True
                        error_reasons.append(f"Jaccard过低: {jaccard:.4f}")
                    if lcs_ratio < 0.3:
                        is_error_case = True
                        error_reasons.append(f"LCS%过低: {lcs_ratio:.4f}")
                    if "ERROR" in prediction or "error" in prediction.lower():
                        is_error_case = True
                        error_reasons.append("生成错误信息")

                    if is_error_case:
                        error_case = {
                            'task_id': task_id,
                            'start_step': sample.get('step_in_task', 0),
                            'end_step': sample.get('step_in_task', 0),
                            'instruction': instruction,
                            'scene_graph': scene_graph,
                            'target_raw': target,
                            'target_parsed': main_target,
                            'target_mode': target_mode,
                            'prediction_raw': prediction,
                            'prediction_cleaned': cleaned_pred,
                            'prediction_mode': pred_mode,
                            'metrics': {
                                'jaccard': jaccard,
                                'lcs_ratio': lcs_ratio,
                                'mode_correct': mode_correct
                            },
                            'error_reasons': error_reasons,
                            'inference_count': inference_count,
                            'global_sample_idx': sample_global_idx
                        }
                        error_cases.append(error_case)

                    if rank == 0 and (processed_count < 5 or is_error_case or processed_count % 20 == 0):
                        status = "❌ ERROR" if is_error_case else "✅ OK"
                        print(f"\n--- {status} Task {task_id} Step {sample.get('step_in_task',0)} [{target_mode.upper()}] ---")
                        print(f"Instruction: {instruction[:60]}...")
                        print(f"Target:      {target}")
                        print(f"Target Parsed: {main_target}")
                        print(f"Target Mode: {target_mode}")
                        print(f"Prediction:  {prediction}")
                        print(f"Prediction Cleaned:  {cleaned_pred}")
                        print(f"Prediction Mode: {pred_mode}")
                        print(f"Mode Correct: {'✅' if mode_correct else '❌'}")
                        print(f"Jaccard: {jaccard:.4f}, LCS%: {lcs_ratio:.4f}")
                        print(f"Inference Steps: {inference_count}")
                        if is_error_case:
                            print(f"Error Reasons: {'; '.join(error_reasons)}")

                    processed_count += 1
                    idx += 1

                else:
                    # local 模式：收集连续的 local block
                    local_block = []
                    while idx < len(samples) and extract_mode_from_text(samples[idx]['target']) == 'local':
                        local_block.append(samples[idx])
                        idx += 1

                    if not local_block:
                        continue

                    # 合并 local block 的目标为一个序列（保持顺序）
                    combined_tasks = []
                    for s in local_block:
                        t_tasks = parse_task_sequence(s['target'])
                        if t_tasks:
                            combined_tasks.extend(t_tasks)
                        else:
                            # 作为兜底，使用原始字符串
                            combined_tasks.append(s['target'])

                    combined_target = " -> ".join(combined_tasks)

                    # 选择第一个 sample 的 instruction（通常相同），使用 block 中最后一个 scene_graph 作为当前环境状态
                    block_instruction = local_block[0]['instruction']
                    block_scene_graph = local_block[-1]['scene_graph']
                    block_start = local_block[0].get('step_in_task', 0)
                    block_end = local_block[-1].get('step_in_task', 0)
                    block_global_idx = local_block[0].get('global_sample_idx')

                    # 计算 loss（可选）
                    try:
                        loss_outputs = model(
                            instructions=[block_instruction],
                            scene_graphs=[block_scene_graph],
                            target_subtasks=[combined_target]
                        )
                        loss = loss_outputs.get("loss")
                        if loss is not None:
                            total_loss += loss.item()
                            total_samples += 1
                    except Exception:
                        pass

                    # 对合并后的序列进行一次（或多次）推理，task_level_inference 会按照 expected_steps 循环获取多个局部步骤
                    prediction, inference_time, inference_count = task_level_inference(
                        model, block_instruction, block_scene_graph, combined_target, rank
                    )

                    total_inference_time += inference_time
                    total_inference_count += inference_count

                    cleaned_pred = clean_prediction_text(prediction)
                    pred_mode = extract_mode_from_text(prediction)
                    mode_correct = pred_mode == 'local'

                    # 与合并目标比较，需要清理目标文本的模式标记
                    cleaned_combined_target = clean_prediction_text(combined_target)
                    jaccard = compute_action_jaccard(cleaned_pred, cleaned_combined_target)
                    lcs_ratio = compute_lcs_ratio(cleaned_pred, cleaned_combined_target)

                    total_jaccard += jaccard
                    total_lcs_ratio += lcs_ratio
                    total_mode_accuracy += 1.0 if mode_correct else 0.0

                    # 错误判定
                    is_error_case = False
                    error_reasons = []
                    if not mode_correct:
                        is_error_case = True
                        error_reasons.append(f"模式错误: 预测{pred_mode} vs 目标local")
                    if jaccard < 0.3:
                        is_error_case = True
                        error_reasons.append(f"Jaccard过低: {jaccard:.4f}")
                    if lcs_ratio < 0.3:
                        is_error_case = True
                        error_reasons.append(f"LCS%过低: {lcs_ratio:.4f}")
                    if "ERROR" in prediction or "error" in prediction.lower():
                        is_error_case = True
                        error_reasons.append("生成错误信息")

                    if is_error_case:
                        error_case = {
                            'task_id': task_id,
                            'start_step': block_start,
                            'end_step': block_end,
                            'instruction': block_instruction,
                            'scene_graph': block_scene_graph,
                            'target_raw': combined_target,
                            'target_parsed': combined_tasks,
                            'target_mode': 'local_block',
                            'prediction_raw': prediction,
                            'prediction_cleaned': cleaned_pred,
                            'prediction_mode': pred_mode,
                            'metrics': {
                                'jaccard': jaccard,
                                'lcs_ratio': lcs_ratio,
                                'mode_correct': mode_correct
                            },
                            'error_reasons': error_reasons,
                            'inference_count': inference_count,
                            'global_sample_idx': block_global_idx
                        }
                        error_cases.append(error_case)

                    if rank == 0 and (processed_count < 5 or is_error_case or processed_count % 20 == 0):
                        status = "❌ ERROR" if is_error_case else "✅ OK"
                        print(f"\n--- {status} Task {task_id} Steps {block_start}-{block_end} [LOCAL-BLOCK] ---")
                        print(f"Instruction: {block_instruction[:60]}...")
                        print(f"Combined Target:      {combined_target}")
                        print(f"Prediction:  {cleaned_pred}")
                        print(f"Mode Correct: {'✅' if mode_correct else '❌'}")
                        print(f"Jaccard: {jaccard:.4f}, LCS%: {lcs_ratio:.4f}")
                        print(f"Inference Steps: {inference_count}")
                        if is_error_case:
                            print(f"Error Reasons: {'; '.join(error_reasons)}")

                    processed_count += 1

    

    # 计算平均指标
    processed_count = max(processed_count, 1)  # 防止除零
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_jaccard = total_jaccard / processed_count
    avg_lcs_ratio = total_lcs_ratio / processed_count
    avg_mode_accuracy = total_mode_accuracy / processed_count
    avg_inference_time = total_inference_time / processed_count
    avg_inference_count = total_inference_count / processed_count

    if rank == 0:
        print(f"\n[Task-level Validation Summary]")
        print(f"  Processed Tasks:   {processed_count}")
        print(f"  Error Cases:       {len(error_cases)} ({len(error_cases)/processed_count*100:.1f}%)")
        print(f"  Avg Loss:          {avg_loss:.4f}")
        print(f"  Avg Jaccard:       {avg_jaccard:.4f}")
        print(f"  Avg LCS%:          {avg_lcs_ratio:.4f}")
        print(f"  Avg Mode Accuracy: {avg_mode_accuracy:.4f}")
        print(f"  Avg Inference Time per Task: {avg_inference_time:.4f}s")
        print(f"  Avg Inference Steps per Task: {avg_inference_count:.2f}")

    torch.cuda.empty_cache()

    return avg_loss, {
        'jaccard': avg_jaccard,
        'lcs_ratio': avg_lcs_ratio,
        'mode_accuracy': avg_mode_accuracy,
        'inference_time': avg_inference_time,
        'inference_count': avg_inference_count
    }, error_cases


def main():
    parser = argparse.ArgumentParser(description='流式模型推理和评估脚本')
    parser.add_argument('--model_path', type=str, 
                       default=os.getenv('HSG_RTP_MODEL_PATH', os.getenv('HLR_MODEL_PATH', 'Qwen/Qwen3-8B')),
                       help='基础模型路径')
    parser.add_argument('--lora_path', type=str, required=True,
                       help='LoRA checkpoint路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='测试数据路径')
    parser.add_argument('--output_path', type=str, 
                       default='streaming_inference_results.json',
                       help='结果输出路径')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大测试样本数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='数据加载chunk大小')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='详细输出')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='DeepSpeed local rank')
    
    # Ablation arguments
    parser.add_argument('--no_hsge', action='store_true', help='Disable Hierarchical Scene Graph Encoder (Ablation)')
    parser.add_argument('--no_local_graph', action='store_true', help='Disable Local Graph details (Ablation)')
    parser.add_argument('--no_context', action='store_true', help='Disable Context (Completed/Pending) (Ablation)')
    
    args = parser.parse_args()
    
    # 初始化 DeepSpeed
    deepspeed.init_distributed()
    
    # 获取当前进程的 rank
    local_rank = args.local_rank
    if local_rank == -1:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print("=== 流式模型推理和评估 ===")
        print(f"模型路径: {args.model_path}")
        print(f"LoRA路径: {args.lora_path}")
        print(f"数据路径: {args.data_path}")
        print(f"最大样本数: {args.max_samples}")
        print(f"使用设备: {device}")
    
    try:
        # 加载流式模型
        if local_rank == 0:
            print(f"加载流式模型: {args.model_path}")
            print(f"Ablation Settings: HSGE={not args.no_hsge}, LocalGraph={not args.no_local_graph}, Context={not args.no_context}")
            
        model = StreamingSceneInstructionQwenModel(
            llm_model_name=args.model_path,
            use_hsge=not args.no_hsge,
            use_local_graph=not args.no_local_graph,
            use_context=not args.no_context
        )
        
        # 加载LoRA权重
        if local_rank == 0:
            print(f"加载LoRA权重: {args.lora_path}")
        load_streaming_lora_model(model, args.lora_path)
        
        # 使用 DeepSpeed 初始化模型
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params={
                "train_batch_size": args.batch_size,
                "train_micro_batch_size_per_gpu": args.batch_size,
                "steps_per_print": 1000,
                "zero_optimization": {"stage": 0},  # 推理时不需要 ZeRO
                "fp16": {"enabled": True},
                "bf16": {"enabled": False}
            }
        )
        
        model_engine.eval()
        
        # 创建数据加载器
        print(f"创建数据加载器: {args.data_path}")
        dataloader = StreamingSceneGraphDataLoader(
            dataset_path=args.data_path,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"数据集大小: {len(dataloader.dataset)}")
        
        # 执行流式验证并记录错误案例
        if local_rank == 0:
            print("\n🚀 开始流式推理和评估...")
        avg_loss, metrics, error_cases = stream_validate_with_error_tracking(
            model=model_engine,
            dataloader=dataloader,
            device=device,
            rank=local_rank,
            world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            max_samples=args.max_samples
        )
        
        # 准备输出结果 (仅主进程)
        if local_rank == 0:
            results = {
                'config': vars(args),
                'model_path': args.model_path,
                'lora_path': args.lora_path,
                'data_path': args.data_path,
                'max_samples': args.max_samples,
                'metrics': {
                    'loss': avg_loss,
                    'jaccard': metrics['jaccard'],
                    'lcs_ratio': metrics['lcs_ratio'],
                    'mode_accuracy': metrics['mode_accuracy'],
                    'inference_time': metrics['inference_time']
                },
                'error_cases': error_cases,
                'error_stats': {
                    'total_errors': len(error_cases),
                    'error_rate': len(error_cases) / args.max_samples if args.max_samples > 0 else 0
                }
            }
            
            # 保存结果
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n📊 推理和评估完成!")
            print(f"结果已保存到: {args.output_path}")
            print(f"📈 最终指标:")
            print(f"  - Loss: {avg_loss:.4f}")
            print(f"  - Jaccard: {metrics['jaccard']:.4f}")
            print(f"  - LCS%: {metrics['lcs_ratio']:.4f}")
            print(f"  - Mode Accuracy: {metrics['mode_accuracy']:.4f}")
            print(f"  - Inference Time: {metrics['inference_time']:.4f}s/batch")
            print(f"🚨 错误分析:")
            print(f"  - 错误案例数: {len(error_cases)}")
            print(f"  - 错误率: {len(error_cases)/args.max_samples*100:.1f}%")
            
            if error_cases:
                print(f"📋 错误原因统计:")
                error_reason_counts = {}
                for case in error_cases:
                    for reason in case['error_reasons']:
                        error_reason_counts[reason] = error_reason_counts.get(reason, 0) + 1
                
                for reason, count in sorted(error_reason_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {reason}: {count} 次")
            
            # 保存错误案例到单独文件
            if error_cases:
                error_file = args.output_path.replace('.json', '_errors.json')
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_cases, f, ensure_ascii=False, indent=2)
                print(f"🔍 错误案例详情已保存到: {error_file}")
        
    except Exception as e:
        print(f"❌ 推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()
