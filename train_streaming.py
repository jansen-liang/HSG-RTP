# train_streaming.py
# 支持流式输出和思考模式的训练脚本（使用Qwen文本编码器）

import torch
import torch.nn as nn
import deepspeed
import os
import argparse
import json
from tqdm import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from utils.dataloader_streaming import StreamingSceneGraphDataLoader
from utils.streaming_hlr import StreamingSceneInstructionQwenModel
from eval_streaming import stream_validate
from datetime import datetime
import wandb


def print_trainable_parameters(model, rank=0):
    if rank != 0:
        return

    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")


def freeze_module(module):
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad = False


def freeze_disabled_components(
    model,
    use_hsge,
    use_local_graph,
    use_global_topology,
):
    if not use_hsge:
        for component_name in ("graph_encoder", "graph_proj"):
            freeze_module(getattr(model, component_name, None))
        return

    graph_encoder = getattr(model, "graph_encoder", None)
    if graph_encoder is None:
        return
    if not use_global_topology:
        freeze_module(getattr(graph_encoder, "room_gnn", None))
        freeze_module(getattr(graph_encoder, "room_post_gnn", None))
    if not use_local_graph:
        for component_name in ("item_proj", "item_attn_gate"):
            freeze_module(getattr(graph_encoder, component_name, None))


def configure_ablation(ablation, rank):
    use_hsge = True
    use_local_graph = True
    use_context = True
    use_global_topology = True

    if ablation == "no_hsge":
        message = "HSG-RTP (w/o HSGE) - Pure Text"
        use_hsge = False
        use_local_graph = False
    elif ablation in ("no_hsge_local", "no_object_tokens"):
        message = "HSG-RTP (w/o independent-object tokens)"
        use_local_graph = False
    elif ablation == "no_global_topology":
        message = "HSG-RTP (w/o global topology GNN)"
        use_global_topology = False
    elif ablation in ("no_context", "no_graph_updates_history"):
        message = "HSG-RTP (w/o graph updates/history context)"
        use_context = False
    else:
        message = "Full HSG-RTP"

    if rank == 0:
        print(f">>> Ablation Mode: {message}")
    return use_hsge, use_local_graph, use_context, use_global_topology


def train_epoch(
    model_engine,
    dataloader,
    rank=0,
    epoch=0,
    world_size=1,
    max_batches=None,
    log_batch_metadata=False,
):
    model_engine.train()
    total_loss = 0.0
    processed_batches = 0
    num_batches = len(dataloader)

    if rank == 0:
        print(f"🔄 开始训练epoch {epoch+1}，总batch数: {num_batches}")
    
    # 👇 关键：所有 rank 都创建 tqdm，但 disable=True 对非 rank 0
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch+1}", 
        disable=(rank != 0),  # 只有 rank 0 显示进度条
        total=num_batches
    )
    
    for batch_idx, batch in enumerate(pbar):  # 直接用 dataloader，不要 pbar
        if max_batches is not None and batch_idx >= max_batches:
            break
        try:
            instructions = batch['instructions']
            completed = batch['completed']
            pending = batch['pending']
            scene_graphs = batch['scene_graphs']
            target_subtasks = batch['subtasks']

            if log_batch_metadata:
                metadata = []
                for scene_graph in scene_graphs:
                    mode = "local" if "current_room" in scene_graph else "global"
                    length = model_engine.module.graph_encoder.sequence_length(scene_graph)
                    metadata.append(f"{mode}:{length}")
                print(f"[Rank {rank}] batch={batch_idx} graph_sequences={metadata}", flush=True)
            
            outputs = model_engine(
                instructions=instructions,
                completed=completed,
                pending=pending,
                scene_graphs=scene_graphs,
                target_subtasks=target_subtasks
            )
            loss = outputs["loss"]

            # 👇 关键：打印 loss 的 grad_fn 和 shape
            # if hasattr(loss, 'grad_fn'):
            #     print(f"[Rank {rank}] Loss grad_fn: {loss.grad_fn}")
            # print(f"[Rank {rank}] Loss:{loss},  Loss shape: {loss.shape}, dtype: {loss.dtype}")
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            processed_batches += 1
            
            # 👇 关键：所有 rank 都执行相同的代码路径！
            try:
                lr_list = model_engine.get_lr()
                current_lr = lr_list[0] if lr_list else 0.0
            except:
                current_lr = 0.0
            
            # 所有 rank 都计算 avg_loss，但只在 rank 0 打印/log
            avg_loss = total_loss / processed_batches
            
            # 只在 rank 0 更新进度条和 log
            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}'
                })
                if batch_idx % 2 == 0 and model_engine.global_steps > 0:
                    wandb.log({
                        "train_batch_loss": loss.item(),
                        "epoch": epoch + 1,
                        "learning_rate": current_lr
                    }, step=model_engine.global_steps)
            
            # Avoid forcing a CUDA allocator flush after every micro-batch.
            # Periodic cleanup keeps memory bounded without stalling training.
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Rank {rank}] Error in batch {batch_idx}: {e}")
            raise

    return total_loss / processed_batches if processed_batches > 0 else 0.0

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.getenv('HSG_RTP_TRAIN_DATA', os.getenv('HLR_TRAIN_DATA', 'pipeline/output/train.jsonl')))
    parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--model_path', type=str, default=os.getenv('HSG_RTP_MODEL_PATH', os.getenv('HLR_MODEL_PATH', 'Qwen/Qwen3-8B')))
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--chunk_size', type=int, default=200)
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Path to a specific checkpoint directory')
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gpu_nums', type=int, default=1)
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=30)
    parser.add_argument('--log_batch_metadata', action='store_true')
    parser.add_argument(
        '--freeze_non_lora',
        action='store_true',
        help='Freeze graph encoders, projections, embeddings, and all non-LoRA parameters.',
    )
    parser.add_argument('--ablation', type=str, default="none", 
                        choices=[
                            "none",
                            "no_hsge",
                            "no_hsge_local",
                            "no_object_tokens",
                            "no_global_topology",
                            "no_context",
                            "no_graph_updates_history",
                        ],
                        help="Ablation setting: 'none' (Full HSG-RTP), 'no_hsge' (Pure Text), 'no_hsge_local' (No Local Graph), 'no_context' (No History/Prompt)")
  
    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # 兼容新旧两种local_rank设置方式
    if hasattr(args, 'local_rank') and args.local_rank != -1:
        local_rank = args.local_rank
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    start_epoch = 0

    use_hsge, use_local_graph, use_context, use_global_topology = (
        configure_ablation(args.ablation, rank)
    )

    # 使用支持流式输出的模型（现在使用Qwen文本编码器）
    model = StreamingSceneInstructionQwenModel(
        llm_model_name=args.model_path,
        use_hsge=use_hsge,
        use_local_graph=use_local_graph,
        use_context=use_context,
        use_global_topology=use_global_topology,
    )
    
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    # LoRA配置和加载
    if args.resume_from_checkpoint:
        # 加载checkpoint逻辑
        resume_path = args.resume_from_checkpoint
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint path not found: {resume_path}")

        adapter_file = os.path.join(resume_path, 'adapter_model.safetensors')
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(f"LoRA adapter not found in {resume_path}")

        if rank == 0:
            print(f"Resuming from checkpoint: {resume_path}")

        train_state_path = os.path.join(resume_path, 'training_state.pt')
        if os.path.exists(train_state_path):
            train_state = torch.load(train_state_path, map_location='cpu')
            checkpoint_ablation = train_state.get('ablation')
            if checkpoint_ablation and checkpoint_ablation != args.ablation:
                raise ValueError(
                    "Checkpoint ablation mismatch: "
                    f"checkpoint={checkpoint_ablation}, requested={args.ablation}"
                )
            start_epoch = train_state['epoch']
            
            # Load additional components if they exist
            if 'additional_components' in train_state:
                additional_state = train_state['additional_components']
                
                # Load scene graph encoder
                if 'graph_encoder' in additional_state and hasattr(model, 'graph_encoder'):
                    model.graph_encoder.load_state_dict(additional_state['graph_encoder'])
                    if rank == 0:
                        print("Scene graph encoder weights loaded")
                
                # Load graph projection layer
                if 'graph_proj' in additional_state and hasattr(model, 'graph_proj'):
                    model.graph_proj.load_state_dict(additional_state['graph_proj'])
                    if rank == 0:
                        print("Graph projection weights loaded")
                
                # Load new token embeddings
                if 'new_token_embeddings' in additional_state:
                    embed_layer = model.llm.get_input_embeddings()
                    original_vocab_size = 151669
                    current_vocab_size = embed_layer.weight.shape[0]
                    if current_vocab_size > original_vocab_size:
                        new_embeddings = additional_state['new_token_embeddings']
                        embed_layer.weight.data[original_vocab_size:] = new_embeddings
                        if rank == 0:
                            print(f"Loaded {new_embeddings.shape[0]} new token embeddings")
                
                # Load instruction encoder parameters if they exist
                if 'instruction_encoder' in additional_state and hasattr(model, 'instruction_encoder'):
                    model.instruction_encoder.load_state_dict(additional_state['instruction_encoder'], strict=False)
                    if rank == 0:
                        print("Instruction encoder weights loaded")
            
            if rank == 0:
                print(f"Resuming from epoch {start_epoch}")
        else:
            start_epoch = 0
            if rank == 0:
                print("training_state.pt not found, starting from epoch 0")
        
        model.llm = PeftModel.from_pretrained(
            model.llm,
            resume_path,
            is_trainable=True,
        )

        if rank == 0:
            print("LoRA weights loaded")
    else:
        if args.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                bias="none"
            )
            model.llm = get_peft_model(model.llm, lora_config)

    model.llm.config.use_cache = False
    freeze_disabled_components(
        model,
        use_hsge,
        use_local_graph,
        use_global_topology,
    )
    if args.freeze_non_lora:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = 'lora_' in name
        if rank == 0:
            print("Non-LoRA parameters frozen")
    model.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model.llm, "enable_input_require_grads"):
        model.llm.enable_input_require_grads()

    if rank == 0:
        print(f"DeepSpeed enabled, World size: {world_size}")
        wandb.init(
            project="StreamingSceneInstructionQwen-QwenText",
            name=f"streaming_qwen_r{args.lora_r}_{args.ablation}_{timestamp}",
            config=vars(args),
            resume="allow"
        )

    print_trainable_parameters(model, rank)

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config
    )
    
    micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    
    # 使用流式数据加载器 - 与常规训练保持一致的配置
    if rank == 0:
        print("🔄 使用流式模式标记数据加载器 (训练数据)")
    train_dataloader = StreamingSceneGraphDataLoader(
        dataset_path=args.data_path,
        batch_size=micro_batch_size,
        chunk_size=args.chunk_size,
        shuffle=True,
        num_workers=2,
        rank=rank,
        world_size=world_size,
        distributed=True,
        seed=42,  # 添加固定seed保持一致性
        rank_nums=args.gpu_nums  
    )

    val_dataloader = None
    if args.val_data_path:
        if rank == 0:
            print("🔄 使用流式数据加载器 (验证数据)")
        val_dataloader = StreamingSceneGraphDataLoader(
            dataset_path=args.val_data_path,
            batch_size=micro_batch_size,
            chunk_size=10,  # 验证数据很少，用小chunk
            shuffle=False,
            num_workers=0,
            rank=rank,
            world_size=world_size,
            distributed=True,
            seed=None  # 验证不需要固定seed
        )

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        train_dataloader.set_epoch(epoch)
        train_loss = train_epoch(
            model_engine, 
            train_dataloader, 
            rank, 
            epoch,
            world_size,
            args.max_train_batches,
            args.log_batch_metadata,
        )

        val_loss = None
        val_metrics = None
        # Validate every 4 epochs and limit to 30 samples
        if val_dataloader and (epoch+1) % 2 == 0:
            val_loss, val_metrics = stream_validate(
                model_engine.module, 
                val_dataloader, 
                model_engine.device, 
                max_samples=args.max_val_samples
            )
            if rank == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if val_metrics:
                    print(f"Validation Metrics: {val_metrics}")
        
        if rank == 0 and (val_loss is None):
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

        if rank == 0:
            try:
                lr_list = model_engine.get_lr()
                current_lr = lr_list[0] if lr_list else 0.0
            except:
                current_lr = 0.0
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": current_lr
            }
            if val_loss is not None:
                log_dict.update({
                    "val_loss": val_loss,
                    "val_jaccard": val_metrics['jaccard'],
                    "val_lcs_ratio": val_metrics['lcs_ratio'],
                    "val_inference_time": val_metrics['inference_time']
                })
            wandb.log(log_dict, step=model_engine.global_steps)
            torch.cuda.empty_cache()

        # 保存checkpoint（与原脚本类似，但保存streaming模型）
        if rank == 0:
            save_path = os.path.join(args.save_dir, f'streaming_qwen_{timestamp}', f'epoch_{epoch+1}')
            os.makedirs(save_path, exist_ok=True)
            
            # 保存模型和额外组件
            model_engine.module.llm.save_pretrained(save_path)
            
            # 保存流式和思考模式的额外状态
            additional_state = {}
            model_module = model_engine.module
            
            # 保存新token embeddings（包含思考模式token）
            if hasattr(model_module.llm, 'get_input_embeddings'):
                embed_weights = model_module.llm.get_input_embeddings().weight.data
                original_vocab_size = 151669  # Qwen3-8B original vocab size
                current_vocab_size = embed_weights.shape[0]
                if current_vocab_size > original_vocab_size:
                    new_token_embeddings = embed_weights[original_vocab_size:]
                    additional_state['new_token_embeddings'] = new_token_embeddings
                    if rank == 0:
                        print(f"Saving {current_vocab_size - original_vocab_size} new token embeddings (including thinking tokens)")
            
            # 保存其他组件
            if hasattr(model_module, 'graph_encoder'):
                additional_state['graph_encoder'] = model_module.graph_encoder.state_dict()
            
            if hasattr(model_module, 'graph_proj') and not isinstance(model_module.graph_proj, nn.Identity):
                additional_state['graph_proj'] = model_module.graph_proj.state_dict()
            
            # 保存instruction encoder if it has trainable parameters
            if hasattr(model_module, 'instruction_encoder'):
                inst_state_dict = model_module.instruction_encoder.state_dict()
                # Only save non-embedding parts (tokenizer config will be saved separately)
                filtered_state = {k: v for k, v in inst_state_dict.items() if 'embed_tokens' not in k}
                if filtered_state:
                    additional_state['instruction_encoder'] = filtered_state
            
            # 保存训练状态
            training_state = {
                'epoch': epoch + 1,
                'global_step': model_engine.global_steps,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'additional_components': additional_state,
                'model_type': 'streaming_qwen',
                'ablation': args.ablation,
                'training_args': vars(args),
                'world_size': world_size,
            }
            torch.save(training_state, os.path.join(save_path, 'training_state.pt'))
            print(f"Streaming checkpoint (Qwen text encoder) saved to {save_path}")

    if rank == 0:
        print("Training completed!")
        wandb.finish()


if __name__ == "__main__":
    train()
