# utils/streaming_hlr.py

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Generator, Tuple
from .hlr import SceneInstructionQwenModel
import json

class StreamingSceneInstructionQwenModel(SceneInstructionQwenModel):
    """
    扩展原始模型以支持流式输出
    核心逻辑与 hlr.py 一致，针对 4090 + Flash Attention 2 进行了 Padding 与 Loss 对齐的终极优化
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 全局默认设置为左对齐，以保证推理/生成阶段 Flash Attention 不报错
        if hasattr(self, 'instruction_encoder'):
            self.instruction_encoder.tokenizer.padding_side = 'left'
            if self.instruction_encoder.tokenizer.pad_token is None:
                self.instruction_encoder.tokenizer.pad_token = self.instruction_encoder.tokenizer.eos_token

    def get_system_message(self) -> str:
        return '''
You are a task planner for a household robot.

Input:
- Scene: Current environment state (rooms, objects, agent location)
- Instruction: Original user task request
- Completed: List of low-level actions already executed successfully
- Pending: Remaining high-level plan steps to complete

Output:
- If you need to output a multi-step (high-level) plan, use GLOBAL mode:
{"mode": "global", "task": ["goto(room1): action1","goto(room2): action2", ...]}
- If you only need to output the next immediate action (low-level), use LOCAL mode. Remember that there is only one action in local mode.:
{"mode": "local", "task": ["action(object)"]}

Rules:
1. Use ONLY the exact JSON format above. No extra fields.
2. For global plans: Each segment must start with "goto(room):".
3. For local actions: Output ONLY the next single action (e.g., "pick(menu)").
4. Use the "Completed" and "Pending" inputs to track progress.
5. DO NOT output explanations, markdown, or extra text.

Scene: <|scene|>{scene}|</scene>
Instruction: <|instruction|>{instruction}|</instruction>
Completed: <|completed|>{completed}|</completed>
Pending: <|pending|>{pending}|</pending>

Output ONLY the JSON.
'''

    def forward(
        self,
        instructions: List[str],
        completed: List[str],
        pending: List[str],
        scene_graphs: List[Dict],
        target_subtasks: Optional[List[str]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        tokenizer = self.instruction_encoder.tokenizer
        batch_size = len(instructions)

        # 消融实验逻辑
        if not self.use_context:
            completed = [""] * batch_size
            pending = [""] * batch_size

        # 1. 编码 System message
        system_msg = self.get_system_message()
        system_texts = [f"<|system|>{system_msg}</|system|>"] * batch_size
        system_outputs = self.instruction_encoder(system_texts)
        system_embeds = system_outputs["embeddings"].to(device)
        system_attn = system_outputs["attention_mask"].to(device)

        # 2. 编码 Scene 
        scene_start_id = tokenizer.convert_tokens_to_ids("<|scene|>")
        scene_end_id = tokenizer.convert_tokens_to_ids("</|scene|>")
        scene_marker_ids = torch.tensor([[scene_start_id, scene_end_id]], device=device)
        scene_marker_embeds = self.llm.get_input_embeddings()(scene_marker_ids).expand(batch_size, -1, -1)
        scene_marker_attn = torch.ones(batch_size, 2, dtype=torch.long, device=device)

        if self.use_hsge:
            if not self.use_local_graph:
                import copy
                graphs_to_encode = []
                for sg in scene_graphs:
                    sg_copy = copy.deepcopy(sg)
                    if "rooms" in sg_copy:
                        for room in sg_copy["rooms"].values():
                            room["items"] = {}; room["small_objects"] = {}; room["large_objects"] = {}
                    if "room" in sg_copy:
                        sg_copy["room"]["items"] = {}; sg_copy["room"]["small_objects"] = {}; sg_copy["room"]["large_objects"] = {}
                    graphs_to_encode.append(sg_copy)
            else:
                graphs_to_encode = scene_graphs

            graph_embeds = self.graph_encoder(graphs_to_encode)
            graph_embeds = self.graph_proj(graph_embeds).to(device)
            graph_lengths = torch.tensor(
                [self.graph_encoder.sequence_length(sg) for sg in graphs_to_encode],
                device=device,
            )
            graph_attention_mask = torch.arange(graph_embeds.size(1), device=device)[None, :] < graph_lengths[:, None]
            graph_attention_mask = graph_attention_mask.long()

            scene_embeds = torch.cat([scene_marker_embeds[:, :1, :], graph_embeds, scene_marker_embeds[:, 1:, :]], dim=1)
            scene_attn = torch.cat([scene_marker_attn[:, :1], graph_attention_mask, scene_marker_attn[:, 1:]], dim=1)
        else:
            scene_embeds = scene_marker_embeds
            scene_attn = scene_marker_attn

        # 3. 编码指令与状态
        instr_outputs = self.instruction_encoder([f"<|instruction|>{i}</|instruction|>" for i in instructions])
        comp_outputs = self.instruction_encoder([f"<|completed|>{c}</|completed|>" for c in completed])
        pend_outputs = self.instruction_encoder([f"<|pending|>{p}</|pending|>" for p in pending])
        out_outputs = self.instruction_encoder(["<|output|>\n"] * batch_size)

        # 4. 融合所有 Prompt Embeddings
        fused_embeds = torch.cat([
            system_embeds, scene_embeds, instr_outputs["embeddings"].to(device),
            comp_outputs["embeddings"].to(device), pend_outputs["embeddings"].to(device),
            out_outputs["embeddings"].to(device)
        ], dim=1).to(self.llm.dtype)

        fused_attention_mask = torch.cat([
            system_attn, scene_attn, instr_outputs["attention_mask"].to(device),
            comp_outputs["attention_mask"].to(device), pend_outputs["attention_mask"].to(device),
            out_outputs["attention_mask"].to(device)
        ], dim=1)

        # 5. Prompt 左侧填充（适配 Flash Attention）
        if fused_embeds.size(1) > self.max_prefix_length:
            fused_embeds = fused_embeds[:, -self.max_prefix_length:]
            fused_attention_mask = fused_attention_mask[:, -self.max_prefix_length:]
        else:
            pad_length = self.max_prefix_length - fused_embeds.size(1)
            pad_embeds = self.llm.get_input_embeddings()(
                torch.full((batch_size, pad_length), tokenizer.pad_token_id, device=device)
            )
            pad_attn = torch.zeros(batch_size, pad_length, dtype=torch.long, device=device)
            fused_embeds = torch.cat([pad_embeds, fused_embeds], dim=1)
            fused_attention_mask = torch.cat([pad_attn, fused_attention_mask], dim=1)

        if target_subtasks is not None:
            # === Training 模式：动态切换右对齐并使用 -100 掩盖 PAD ===
            end_token = "<|endoftext|>"
            
            # 临时将 Tokenizer 切回右对齐，防止答案跑到左边去
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'right'
            
            targets_encoded = tokenizer(
                [t + end_token for t in target_subtasks],
                padding='max_length',
                truncation=True,
                max_length=self.max_output_length,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            # 恢复左对齐
            tokenizer.padding_side = original_padding_side

            labels = targets_encoded["input_ids"].to(device)
            target_embeds = self.llm.get_input_embeddings()(labels)

            full_embeds = torch.cat([fused_embeds, target_embeds], dim=1)
            full_attention_mask = torch.cat([fused_attention_mask, targets_encoded["attention_mask"].to(device)], dim=1)

            # --- 救命的 Loss 掩码逻辑 ---
            prefix_len = fused_embeds.size(1)
            extended_labels = torch.full((batch_size, full_embeds.size(1)), -100, dtype=labels.dtype, device=device)
            
            # 把 labels 中所有被 padding 的地方 (attention_mask == 0) 替换为 -100
            # 这样模型计算 Loss 时就会忽略这些填充字符
            masked_labels = labels.clone()
            masked_labels[targets_encoded["attention_mask"] == 0] = -100
            
            extended_labels[:, prefix_len:] = masked_labels

            logits_to_keep = self.max_output_length + 1
            shift_labels = torch.nn.functional.pad(extended_labels, (0, 1), value=-100)
            shift_labels = shift_labels[:, 1:][:, -logits_to_keep:].contiguous()

            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_attention_mask,
                labels=extended_labels,
                use_cache=False,
                logits_to_keep=logits_to_keep,
                shift_labels=shift_labels,
            )
            return {"loss": outputs.loss}

        else:
            # === Inference 模式：享受 Flash Attention 左对齐加持 ===
            generation_config = dict(generation_config or {})
            generation_kwargs = {
                "max_new_tokens": self.max_output_length,
                "eos_token_id": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                "pad_token_id": tokenizer.pad_token_id,
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True,
                "use_cache": True,
            }
            generation_kwargs.update(generation_config)
            if not generation_kwargs.get("do_sample", False):
                generation_kwargs["temperature"] = None
                generation_kwargs["top_p"] = None
                generation_kwargs["top_k"] = None
            generated_ids = self.llm.generate(
                inputs_embeds=fused_embeds,
                attention_mask=fused_attention_mask,
                **generation_kwargs,
            )
            raw_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
            clean_predictions = [p.split("<|endoftext|>")[0].strip() if "<|endoftext|>" in p else p.strip() for p in raw_predictions]
            return {"predictions": clean_predictions}

    def stream_generate(
        self,
        instructions: List[str],
        completed: List[str],
        pending: List[str],
        scene_graphs: List[Dict],
    ) -> Generator[Tuple[str, str], None, None]:
        if len(instructions) != 1:
            raise ValueError("stream_generate only supports batch_size=1")

        outputs = self.forward(instructions, completed, pending, scene_graphs)
        prediction = outputs.get("predictions", [""])[0]

        try:
            parsed = json.loads(prediction)
            mode = parsed.get("mode", "none")
            task_list = parsed.get("task", [])
            if not isinstance(task_list, list): task_list = [str(task_list)]

            yield ("mode", mode)
            if mode == "local":
                yield ("action", task_list[0] if task_list else "none")
            else:
                yield ("action", json.dumps(task_list))
        except:
            yield ("mode", "error")
            yield ("action", "none")
        yield ("complete", "")
