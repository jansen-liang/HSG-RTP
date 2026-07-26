# models/hlr.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from .instruction_encoder import QwenInstructionEncoder
from .graph_encoder import HierarchicalSceneGraphEncoder
from typing import List, Dict, Optional, Any


class SceneInstructionQwenModel(nn.Module):
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen3-8B",
        max_instruction_length: int = 128,
        max_output_length: int = 96,
        max_prefix_length: int = 512,
        use_hsge: bool = True,
        use_local_graph: bool = True,
        use_context: bool = True
    ):
        super().__init__()
        self.max_instruction_length = max_instruction_length
        self.max_output_length = max_output_length
        self.max_prefix_length = max_prefix_length
        self.use_hsge = use_hsge
        self.use_local_graph = use_local_graph
        self.use_context = use_context

        # 1. 自动选择精度
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 2. 加载配置
        config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        
        # 3. 加载 LLM（修复了 dtype 参数错误，并针对 4090 开启了加速）
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=dtype,              # 修复：从 dtype 改为 torch_dtype
            device_map=None,
            trust_remote_code=True,
            config=config,
            attn_implementation="flash_attention_2"  # 优化：4090 必开，大幅提速且省显存
        )

        # Instruction encoder
        self.instruction_encoder = QwenInstructionEncoder(
            model_name=llm_model_name,
            max_length=max_instruction_length
        )
        self.instruction_encoder.set_embed_tokens(self.llm.get_input_embeddings())
        self.hidden_size = self.instruction_encoder.hidden_size
        
        # Register special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<|system|>", "</|system|>",
                "<|scene|>", "</|scene|>",
                "<|instruction|>", "</|instruction|>",
                "<|completed|>", "</|completed|>",
                "<|pending|>", "</|pending|>",
                "<|output|>"
            ]
        }
        num_added = self.instruction_encoder.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.llm.resize_token_embeddings(len(self.instruction_encoder.tokenizer))
        
        # Graph encoder
        self.graph_encoder = HierarchicalSceneGraphEncoder(llm_hidden_dim=self.hidden_size, qwen_model_name=llm_model_name)
        self.graph_encoder.set_embed_tokens(self.llm.get_input_embeddings())

        # Projection
        graph_out_dim = getattr(self.graph_encoder, 'llm_dim', self.hidden_size)
        if graph_out_dim != self.hidden_size:
            self.graph_proj = nn.Linear(graph_out_dim, self.hidden_size)
        else:
            self.graph_proj = nn.Identity()

    def forward(
        self,
        instructions: List[str],
        scene_graphs: List[Dict],
        target_subtasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        
        device = next(self.parameters()).device
        tokenizer = self.instruction_encoder.tokenizer
        batch_size = len(instructions)

        # === Step 1: Encode system prompt ===
        system_msg = '''
        You are a helpful assistant for household robot task planning.
        The input contains:
        (1) scene objects in <|scene|>...</|scene|>,
        (2) user instruction in <|instruction|>...</|instruction|>.
        Output the action sequence immediately after <|output|>.
        Format: action1 -> action2 -> action3
        Do not provide any explanations or additional text.
        '''
        system_texts = [f"<|system|>{system_msg}</|system|>"] * batch_size
        system_outputs = self.instruction_encoder(system_texts)
        system_embeds = system_outputs["embeddings"].to(device)
        system_attn = system_outputs["attention_mask"].to(device)

        # === Step 2: Encode scene graph ===
        scene_start_id = tokenizer.convert_tokens_to_ids("<|scene|>")
        scene_end_id = tokenizer.convert_tokens_to_ids("</|scene|>")
        scene_marker_ids = torch.tensor([[scene_start_id, scene_end_id]], device=device)
        scene_marker_embeds = self.llm.get_input_embeddings()(scene_marker_ids) 
        scene_marker_embeds = scene_marker_embeds.expand(batch_size, -1, -1)
        scene_marker_attn = torch.ones(batch_size, 2, dtype=torch.long, device=device)

        graph_embeds = self.graph_encoder(scene_graphs) 
        graph_embeds = self.graph_proj(graph_embeds).to(device)
        graph_lengths = torch.tensor([
            len(self.graph_encoder.bfs_traversal(sg, sg["agent"]["position"]))
            for sg in scene_graphs
        ], device=device)

        graph_attention_mask = torch.arange(graph_embeds.size(1), device=device)[None, :] < graph_lengths[:, None]
        graph_attention_mask = graph_attention_mask.long()

        scene_embeds = torch.cat([scene_marker_embeds[:, :1, :], graph_embeds, scene_marker_embeds[:, 1:, :]], dim=1)
        scene_attn = torch.cat([scene_marker_attn[:, :1], graph_attention_mask, scene_marker_attn[:, 1:]], dim=1)

        # === Step 3: Encode instruction ===
        instruction_texts = [f"<|instruction|>{instr}</|instruction|>" for instr in instructions]
        instr_outputs = self.instruction_encoder(instruction_texts)
        instr_embeds = instr_outputs["embeddings"].to(device)
        instr_attn = instr_outputs["attention_mask"].to(device)

        # === Step 4: Encode output prefix ===
        output_prefix = "<|output|>\n"
        output_outputs = self.instruction_encoder([output_prefix] * batch_size)
        output_embeds = output_outputs["embeddings"].to(device)
        output_attn = output_outputs["attention_mask"].to(device)

        # === Step 5: Fuse all parts ===
        fused_embeds = torch.cat([
            system_embeds,
            scene_embeds,
            instr_embeds,
            output_embeds
        ], dim=1).to(self.llm.dtype)
        
        fused_attention_mask = torch.cat([
            system_attn,
            scene_attn,
            instr_attn,
            output_attn
        ], dim=1)

        if target_subtasks is not None:
            # Train mode
            end_token = "<|endoftext|>"
            targets_with_end = [t + end_token for t in target_subtasks]
            targets_encoded = self.instruction_encoder.tokenizer(
                targets_with_end,
                padding='max_length',
                truncation=True,
                max_length=self.max_output_length,
                return_tensors="pt",
                add_special_tokens=False
            )
            labels = targets_encoded["input_ids"].to(device)
            target_attention_mask = targets_encoded["attention_mask"].to(device)

            target_embeds = self.llm.get_input_embeddings()(labels)
            full_embeds = torch.cat([fused_embeds, target_embeds], dim=1)
            full_attention_mask = torch.cat([fused_attention_mask, target_attention_mask], dim=1)

            prefix_len = fused_embeds.size(1)
            extended_labels = torch.full(
                (labels.size(0), full_embeds.size(1)),
                -100,
                dtype=labels.dtype,
                device=device
            )
            extended_labels[:, prefix_len:] = labels

            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_attention_mask,
                labels=extended_labels,
                use_cache=False
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        else:
            # Inference mode
            end_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            generated_ids = self.llm.generate(
                inputs_embeds=fused_embeds,
                attention_mask=fused_attention_mask,
                max_new_tokens=self.max_output_length,
                eos_token_id=end_id,  
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.5,
                num_beams=2,
                early_stopping=True,
                use_cache=False
            )

            raw_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            clean_predictions = []
            for pred in raw_predictions:
                if "<|endoftext|>" in pred:
                    pred = pred.split("<|endoftext|>")[0].strip()
                
                import json
                import re
                pred = re.sub(r'[^\x00-\x7F]+', '', pred)
                
                try:
                    json_match = re.search(r'\{.*\}', pred, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        parsed = json.loads(json_str)
                        clean_predictions.append(parsed.get('task', pred.strip()))
                    else:
                        clean_predictions.append(pred.strip())
                except (json.JSONDecodeError, Exception):
                    clean_predictions.append(pred.strip())

            return {"predictions": clean_predictions}
