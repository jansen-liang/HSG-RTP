# train.py

import torch
import torch.nn as nn
import deepspeed
import os
import argparse
import json
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from utils.dataloader import SceneGraphDataLoader
from utils.hlr import SceneInstructionQwenModel
from eval import validate
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


def train_epoch(model_engine, dataloader, rank=0, epoch=0):
    model_engine.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}") if rank == 0 else dataloader

    for batch_idx, batch in enumerate(pbar):
        try:
            instructions = batch['instructions']
            scene_graphs = batch['scene_graphs']
            target_subtasks = batch['subtasks']

            formatted_targets = []
            for action_sequence in target_subtasks:
                if isinstance(action_sequence, list):
                    formatted_text = " -> ".join(action_sequence)
                else:
                    formatted_text = str(action_sequence)
                formatted_targets.append(formatted_text)

            outputs = model_engine(
                instructions=instructions,
                scene_graphs=scene_graphs,
                target_subtasks=formatted_targets
            )
            loss = outputs["loss"]

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                if batch_idx % 2 == 0 and model_engine.global_steps > 0:
                    try:
                        lr_list = model_engine.get_lr()
                        current_lr = lr_list[0] if lr_list else 0.0
                    except:
                        current_lr = 0.0  # Fallback if scheduler not started
                    wandb.log({
                        "train_batch_loss": loss.item(),
                        "epoch": epoch + 1,
                        "learning_rate": current_lr
                    }, step=model_engine.global_steps)
            if batch_idx % 1 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"Error in batch {batch_idx}: {e}")
            continue

    return total_loss / num_batches if num_batches > 0 else 0.0


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.getenv('HLR_TRAIN_DATA', 'pipeline/output/train.jsonl'))
    parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--model_path', type=str, default=os.getenv('HLR_MODEL_PATH', 'Qwen/Qwen3-8B'))
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--use_lora', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--chunk_size', type=int, default=200)
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Path to a specific checkpoint directory, e.g.: ./checkpoints/TIMESTAMP/epoch_3')
    parser.add_argument('--deepspeed_config', type=str, default='deepspeed_config.json')
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    start_epoch = 0

    model = SceneInstructionQwenModel(llm_model_name=args.model_path)
    breakpoint()
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
    
    if args.resume_from_checkpoint:
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

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            bias="none"
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.load_adapter(resume_path, adapter_name="default")

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

    if rank == 0:
        print(f"Training on device: cuda")
        print(f"DeepSpeed enabled, World size: {world_size}")
        wandb.init(
            project="SceneInstructionQwen-DeepSpeed",
            name=f"deepspeed_lora_r{args.lora_r}_lr{args.lr}_bs{ds_config['train_batch_size']}_{timestamp}",
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
    pin_memory = ds_config['zero_optimization']['offload_optimizer']['pin_memory'] 
    
    train_dataloader = SceneGraphDataLoader(
        jsonl_file=args.data_path,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
        use_ddp=True,
        ddp_rank=rank,
        ddp_world_size=world_size,
        seed=42,
        chunk_size=args.chunk_size
    )

    val_dataloader = None
    if args.val_data_path:
        val_dataloader = SceneGraphDataLoader(
            jsonl_file=args.val_data_path,
            batch_size=micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=False,
            use_ddp=True,
            ddp_rank=rank,
            ddp_world_size=world_size,
            chunk_size=args.chunk_size
        )

    
    if args.resume_from_checkpoint and start_epoch > 0:
        train_state_path = os.path.join(args.resume_from_checkpoint, 'training_state.pt')
        if os.path.exists(train_state_path):
            train_state = torch.load(train_state_path, map_location='cpu')
            torch.set_rng_state(train_state['rng_state'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(train_state['cuda_rng_state'])
            if rank == 0:
                print("Random states restored")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        train_dataloader.set_epoch(epoch)
        train_loss = train_epoch(model_engine, train_dataloader, rank, epoch)

        val_loss = None
        val_metrics = None
        if val_dataloader and epoch % 4 == 0:
            val_loss, val_metrics = validate(model_engine.module, val_dataloader, model_engine.device, rank, world_size)
            if rank == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        elif rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")

        if rank == 0:
            try:
                lr_list = model_engine.get_lr()
                current_lr = lr_list[0] if lr_list else 0.0
            except:
                current_lr = 0.0  # Fallback if scheduler not started
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

        if rank == 0 and epoch % 2 == 0:
            save_path = os.path.join(args.save_dir, timestamp, f'epoch_{epoch+1}')
            os.makedirs(save_path, exist_ok=True)
            
            # Save LLM with LoRA
            model_engine.module.llm.save_pretrained(save_path)
            
            # Save other trainable components (scene graph encoder, projections, etc.)
            additional_state = {}
            model_module = model_engine.module
            
            # Save embedding weights separately (for new special tokens)
            if hasattr(model_module.llm, 'get_input_embeddings'):
                embed_weights = model_module.llm.get_input_embeddings().weight.data
                original_vocab_size = 151669  # Qwen3-8B original vocab size
                current_vocab_size = embed_weights.shape[0]
                if current_vocab_size > original_vocab_size:
                    # Only save the new token embeddings
                    new_token_embeddings = embed_weights[original_vocab_size:]
                    additional_state['new_token_embeddings'] = new_token_embeddings
                    if rank == 0:
                        print(f"Saving {current_vocab_size - original_vocab_size} new token embeddings")
            
            # Save scene graph encoder
            if hasattr(model_module, 'graph_encoder'):
                additional_state['graph_encoder'] = model_module.graph_encoder.state_dict()
            
            # Save graph projection layer if exists
            if hasattr(model_module, 'graph_proj') and not isinstance(model_module.graph_proj, nn.Identity):
                additional_state['graph_proj'] = model_module.graph_proj.state_dict()
            
            # Save instruction encoder if it has trainable parameters
            if hasattr(model_module, 'instruction_encoder'):
                # Note: instruction_encoder mainly shares embedding with LLM, but save its state for completeness
                inst_state_dict = model_module.instruction_encoder.state_dict()
                # Only save non-embedding parts (tokenizer config will be saved separately)
                filtered_state = {k: v for k, v in inst_state_dict.items() if 'embed_tokens' not in k}
                if filtered_state:
                    additional_state['instruction_encoder'] = filtered_state
            
            # Save tokenizer configuration
            tokenizer_config = {
                'vocab_size': len(model_module.instruction_encoder.tokenizer),
                'special_tokens': model_module.instruction_encoder.tokenizer.added_tokens_encoder
            }
            
            # Save training state
            training_state = {
                'epoch': epoch + 1,
                'global_step': model_engine.global_steps,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'additional_components': additional_state,
                'tokenizer_config': tokenizer_config
            }
            torch.save(training_state, os.path.join(save_path, 'training_state.pt'))
            print(f"Checkpoint saved to {save_path} (LLM + additional components)")

    if rank == 0:
        print("Training completed!")
        wandb.finish()


if __name__ == "__main__":
    train()
