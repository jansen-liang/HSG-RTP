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
        kwargs.setdefault("max_output_length", 448)
        kwargs.setdefault("max_prefix_length", 768)
        super().__init__(*args, **kwargs)
        self.segment_token_limits = {
            "system": 128,
            "instruction": 96,
            "completed": 128,
            "pending": 192,
        }
        
        # 全局默认设置为左对齐，以保证推理/生成阶段 Flash Attention 不报错
        if hasattr(self, 'instruction_encoder'):
            self.instruction_encoder.tokenizer.padding_side = 'left'
            if self.instruction_encoder.tokenizer.pad_token is None:
                self.instruction_encoder.tokenizer.pad_token = self.instruction_encoder.tokenizer.eos_token

    def get_system_message(self) -> str:
        return (
            "Plan robot tasks from the structured scene and execution context. "
            "Return exactly one JSON object and no other text. "
            'GLOBAL format: {"mode":"global","task":["goto(room): action",...]}. '
            'LOCAL format: {"mode":"local","task":["action(object)"]}. '
            "A GLOBAL step must use an exact scene room ID and start with goto(room):. "
            "A LOCAL response contains exactly one immediately executable action. "
            "Use Completed and Pending to continue the task without repeating finished work."
        )

    @staticmethod
    def _active_tokens(
        embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_index: int,
        limit: Optional[int] = None,
        keep: str = "last",
    ) -> torch.Tensor:
        tokens = embeds[batch_index][attention_mask[batch_index].bool()]
        if limit is None or tokens.size(0) <= limit:
            return tokens
        if keep == "first":
            return tokens[:limit]
        if keep == "last":
            return tokens[-limit:]
        raise ValueError(f"Unsupported token retention policy: {keep}")

    def _assemble_segmented_prefix(
        self,
        system_embeds: torch.Tensor,
        system_attn: torch.Tensor,
        scene_embeds: torch.Tensor,
        scene_attn: torch.Tensor,
        instruction_embeds: torch.Tensor,
        instruction_attn: torch.Tensor,
        completed_embeds: torch.Tensor,
        completed_attn: torch.Tensor,
        pending_embeds: torch.Tensor,
        pending_attn: torch.Tensor,
        output_embeds: torch.Tensor,
        output_attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        prefixes = []
        prefix_lengths = []
        batch_size = system_embeds.size(0)

        for batch_index in range(batch_size):
            system_tokens = self._active_tokens(
                system_embeds,
                system_attn,
                batch_index,
                self.segment_token_limits["system"],
                "first",
            )
            scene_tokens = self._active_tokens(
                scene_embeds, scene_attn, batch_index
            )
            instruction_tokens = self._active_tokens(
                instruction_embeds,
                instruction_attn,
                batch_index,
                self.segment_token_limits["instruction"],
                "last",
            )
            completed_tokens = self._active_tokens(
                completed_embeds,
                completed_attn,
                batch_index,
                self.segment_token_limits["completed"],
                "last",
            )
            pending_tokens = self._active_tokens(
                pending_embeds,
                pending_attn,
                batch_index,
                self.segment_token_limits["pending"],
                "first",
            )
            output_tokens = self._active_tokens(
                output_embeds, output_attn, batch_index
            )

            mandatory_length = sum(
                tokens.size(0)
                for tokens in (
                    system_tokens,
                    scene_tokens,
                    instruction_tokens,
                    output_tokens,
                )
            )
            if mandatory_length > self.max_prefix_length:
                raise ValueError(
                    "System, scene graph, instruction, and output marker exceed "
                    f"the prefix budget ({mandatory_length} > {self.max_prefix_length})"
                )

            optional_budget = self.max_prefix_length - mandatory_length
            if completed_tokens.size(0) + pending_tokens.size(0) > optional_budget:
                completed_budget = max(
                    0, optional_budget - pending_tokens.size(0)
                )
                completed_tokens = (
                    completed_tokens[-completed_budget:]
                    if completed_budget
                    else completed_tokens[:0]
                )
                remaining_budget = optional_budget - completed_tokens.size(0)
                pending_tokens = pending_tokens[:remaining_budget]

            prefix = torch.cat(
                [
                    system_tokens,
                    scene_tokens,
                    instruction_tokens,
                    completed_tokens,
                    pending_tokens,
                    output_tokens,
                ],
                dim=0,
            )
            prefixes.append(prefix)
            prefix_lengths.append(prefix.size(0))

        padded_length = max(prefix_lengths)
        hidden_size = prefixes[0].size(-1)
        fused_embeds = prefixes[0].new_zeros(
            (batch_size, padded_length, hidden_size)
        )
        fused_attention_mask = torch.zeros(
            (batch_size, padded_length),
            dtype=torch.long,
            device=prefixes[0].device,
        )
        for batch_index, prefix in enumerate(prefixes):
            start_index = padded_length - prefix.size(0)
            fused_embeds[batch_index, start_index:] = prefix
            fused_attention_mask[batch_index, start_index:] = 1

        return fused_embeds, fused_attention_mask, prefix_lengths

    def _encode_training_targets(
        self, tokenizer: Any, target_subtasks: List[str]
    ) -> Dict[str, torch.Tensor]:
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            encoded = tokenizer(
                [target + "<|endoftext|>" for target in target_subtasks],
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = original_padding_side

        target_lengths = encoded["attention_mask"].sum(dim=1)
        longest_target = int(target_lengths.max().item())
        if longest_target > self.max_output_length:
            raise ValueError(
                f"Target length {longest_target} exceeds max_output_length "
                f"{self.max_output_length}; refusing to truncate supervision"
            )
        return encoded

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

        # 4. Assemble the prefix without ever dropping system, HSG, or instruction.
        fused_embeds, fused_attention_mask, prefix_lengths = self._assemble_segmented_prefix(
            system_embeds,
            system_attn,
            scene_embeds,
            scene_attn,
            instr_outputs["embeddings"].to(device),
            instr_outputs["attention_mask"].to(device),
            comp_outputs["embeddings"].to(device),
            comp_outputs["attention_mask"].to(device),
            pend_outputs["embeddings"].to(device),
            pend_outputs["attention_mask"].to(device),
            out_outputs["embeddings"].to(device),
            out_outputs["attention_mask"].to(device),
        )
        fused_embeds = fused_embeds.to(self.llm.dtype)

        if target_subtasks is not None:
            targets_encoded = self._encode_training_targets(
                tokenizer, target_subtasks
            )

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

            logits_to_keep = labels.size(1) + 1
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
            output_lengths = [
                len(tokenizer.encode(prediction, add_special_tokens=False))
                for prediction in clean_predictions
            ]
            usage = [
                {
                    "input_tokens": int(prefix_length),
                    "output_tokens": int(output_length),
                    "total_tokens": int(prefix_length + output_length),
                }
                for prefix_length, output_length in zip(prefix_lengths, output_lengths)
            ]
            return {"predictions": clean_predictions, "usage": usage}

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
