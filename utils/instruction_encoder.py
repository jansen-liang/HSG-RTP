# models/instruction_encoder.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict

class QwenInstructionEncoder(nn.Module):
    """
    使用 Qwen3 的 tokenizer + embed_tokens 作为指令编码器
    输出: [batch_size, seq_len, hidden_size]
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_length: int = 256
    ):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = config.hidden_size

        self.embed_tokens = None

    def set_embed_tokens(self, embed_tokens: nn.Module):
        self.embed_tokens = embed_tokens

    def tokenize(self, instructions: List[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

    def forward(self, instructions: List[str]) -> Dict[str, torch.Tensor]:
        outputs = self.tokenize(instructions)

        if self.embed_tokens is not None:
            input_ids = outputs["input_ids"].to(self.embed_tokens.weight.device)
            outputs["embeddings"] = self.embed_tokens(input_ids)

        return outputs
