import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
from transformers import AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import deque

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
        seq_len = x.shape[seq_dim]
        t = torch.arange(seq_len, device=x.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return (x * cos) + (self._rotate_half(x) * sin)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)


class HierarchicalSceneGraphEncoder(nn.Module):
    def __init__(self, llm_hidden_dim: int = 4096, qwen_model_name: str = "Qwen/Qwen3-8B"):
        super().__init__()
        self.llm_dim = llm_hidden_dim

        # 使用Qwen的tokenizer和config
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
        print(f"qwen has eos_token: {self.tokenizer.eos_token}")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"qwen pad_token set to: {self.tokenizer.pad_token}")
        config = AutoConfig.from_pretrained(qwen_model_name, trust_remote_code=True)
        self.qwen_hidden_size = config.hidden_size

        # 将在主模型中设置，用于共享Qwen的embedding层
        self.embed_tokens = None
        
        self.item_proj = nn.Sequential(
            nn.Linear(llm_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, llm_hidden_dim)
        )

        self.room_gnn = GATv2Conv(
            in_channels=llm_hidden_dim,
            out_channels=llm_hidden_dim // 4,
            heads=4,
            concat=False,
            dropout=0.1,
            add_self_loops=True
        )
        self.room_post_gnn = nn.Sequential(
            nn.Linear(llm_hidden_dim // 4, llm_hidden_dim),
            nn.ReLU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )
        self.room_residual_proj = nn.Linear(llm_hidden_dim, llm_hidden_dim) if llm_hidden_dim // 4 != llm_hidden_dim else nn.Identity()

        self.item_attn_gate = nn.Sequential(
            nn.Linear(llm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.type_embedding = nn.Embedding(4, llm_hidden_dim)
        self.type_to_idx = {"macro": 0, "room": 1, "item": 2, "agent": 3}

        self.rotary_emb = RotaryEmbedding(dim=llm_hidden_dim // 8)

        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.to(dtype=target_dtype)
    
    def set_embed_tokens(self, embed_tokens: nn.Module):
        """设置共享的Qwen embedding层"""
        self.embed_tokens = embed_tokens

    def encode_text(self, text: str) -> torch.Tensor:
        """使用Qwen的tokenizer和embedding编码文本"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        if self.embed_tokens is None:
            raise ValueError("embed_tokens not set. Call set_embed_tokens() first.")
        
        # 使用Qwen tokenizer编码文本
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,  # 对于描述文本使用较短长度
            return_tensors="pt",
            add_special_tokens=False  # 不需要特殊tokens，只是编码描述文本
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # 获取embeddings
        with torch.no_grad():
            token_embeddings = self.embed_tokens(input_ids)  # [1, seq_len, hidden_size]
            
            # 使用attention mask进行平均池化
            masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1).float()
            sum_embeddings = masked_embeddings.sum(dim=1)  # [1, hidden_size]
            sum_mask = attention_mask.sum(dim=1, keepdim=True).float()  # [1, 1]
            avg_embeddings = sum_embeddings / (sum_mask + 1e-9)  # 避免除零
        
        emb = avg_embeddings.to(dtype)
        return emb

    def _zero_emb(self, device, dtype, size=1):
        return torch.zeros(size, self.llm_dim, device=device, dtype=dtype)

    def bfs_traversal(self, scene_data: Dict, start_room_id: str) -> List[Tuple[str, str]]:
        visited = set()
        queue = deque()
        traversal_order = []

        # 判断场景类型并获取房间数据
        is_global_scene = "macro_zones" in scene_data and "rooms" in scene_data
        is_local_scene = "current_room" in scene_data and "room" in scene_data
        
        if is_global_scene:
            rooms_data = scene_data["rooms"]
        elif is_local_scene:
            rooms_data = {scene_data["current_room"]: scene_data["room"]}
        else:
            rooms_data = scene_data.get("rooms", {})

        if start_room_id in rooms_data:
            start_key = ("room", start_room_id)
            queue.append(start_key)
            visited.add(start_key)
        elif is_local_scene:
            # 在local场景中，如果agent的position不匹配current_room，使用current_room作为起始点
            start_room_id = scene_data["current_room"]
            if start_room_id in rooms_data:
                start_key = ("room", start_room_id)
                queue.append(start_key)
                visited.add(start_key)

        while queue:
            node_type, node_id = queue.popleft()
            traversal_order.append((node_type, node_id))

            if node_type == "room":
                neighbors = rooms_data[node_id].get("neighbor", [])
                for neighbor_id in neighbors:
                    neighbor_key = ("room", neighbor_id)
                    if neighbor_key not in visited and neighbor_id in rooms_data:
                        visited.add(neighbor_key)
                        queue.append(neighbor_key)

        # 只在全局场景中处理macro_zones
        if is_global_scene and scene_data["macro_zones"]:
            for zone_id in scene_data["macro_zones"]:
                key = ("macro", zone_id)
                if key not in visited:
                    traversal_order.append(key)

        for room_id, room in rooms_data.items():
            # 处理items（可能在items, small_objects, large_objects中）
            items = room.get("items", {})
            if not items:
                items = {}
                items.update(room.get("small_objects", {}))
                items.update(room.get("large_objects", {}))
                
            for item_name in items.keys():
                key = ("item", f"{room_id}_{item_name}")
                if key not in visited:
                    traversal_order.append(key)

        agent_key = ("agent", "agent")
        if agent_key not in visited:
            traversal_order.append(agent_key)

        return traversal_order

    def encode_local_scene(self, scene_data: Dict[str, Any]) -> torch.Tensor:
        """简化的单房间场景编码"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        current_room = scene_data["current_room"]
        room_data = scene_data["room"]
        
        # 编码房间信息
        room_desc = f"Current room: {current_room}"
        room_emb = self.encode_text(room_desc)
        
        # 收集所有物品
        items = {}
        items.update(room_data.get("small_objects", {}))
        items.update(room_data.get("large_objects", {}))
        items.update(room_data.get("items", {}))
        
        # 编码物品
        item_embeddings = []
        for item_name, item_info in items.items():
            type_str = item_info.get("type", "object")
            state_str = item_info.get("state", "unknown")
            affordances = item_info.get("affordance", [])
            affordance_str = " ".join(affordances) if affordances else "no interaction"
            
            item_desc = (f"Item '{item_name}' is a {type_str}, "
                        f"state: {state_str}, "
                        f"affordances: {affordance_str}")
            item_emb = self.encode_text(item_desc)
            item_emb = self.item_proj(item_emb)
            item_embeddings.append(item_emb)
        
        # 如果有物品，用注意力机制聚合
        if item_embeddings:
            item_embs_cat = torch.cat(item_embeddings, dim=0)
            attn_weights = self.item_attn_gate(item_embs_cat)
            attn_weights = torch.softmax(attn_weights, dim=0)
            aggregated_items = torch.sum(attn_weights * item_embs_cat, dim=0, keepdim=True)
        else:
            aggregated_items = self._zero_emb(device, dtype)
        
        # 编码agent信息
        agent = scene_data["agent"]
        agent_desc = f"Agent at {agent.get('position', current_room)}, state: {agent.get('state', 'unknown')}"
        agent_emb = self.encode_text(agent_desc)
        
        # 简单的序列：[房间, 物品聚合, agent]
        embeddings_list = []
        
        # 添加房间embedding + type embedding
        room_type_idx = torch.tensor([self.type_to_idx.get("room", 0)], device=device)
        room_type_emb = self.type_embedding(room_type_idx)
        embeddings_list.append(room_emb + room_type_emb)
        
        # 添加物品聚合embedding (当作特殊的item type)
        if item_embeddings:
            item_type_idx = torch.tensor([self.type_to_idx.get("item", 1)], device=device)
            item_type_emb = self.type_embedding(item_type_idx)
            embeddings_list.append(aggregated_items + item_type_emb)
        
        # 添加agent embedding + type embedding
        agent_type_idx = torch.tensor([self.type_to_idx.get("agent", 2)], device=device)
        agent_type_emb = self.type_embedding(agent_type_idx)
        embeddings_list.append(agent_emb + agent_type_emb)
        
        # 拼接所有embeddings
        final_sequence = torch.cat(embeddings_list, dim=0)
        
        # 应用rotary embedding
        if final_sequence.shape[1] >= 512:
            final_sequence[:, -512:] = self.rotary_emb(final_sequence[:, -512:], seq_dim=0)
        
        return final_sequence.to(dtype=dtype)

    def encode_single_scene(self, scene_data: Dict[str, Any]) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # 检查场景图类型：全局（有macro_zones和rooms）vs 局部（有current_room和room）
        is_global_scene = "macro_zones" in scene_data and "rooms" in scene_data
        is_local_scene = "current_room" in scene_data and "room" in scene_data
        
        # 对local场景使用简化编码
        if is_local_scene:
            return self.encode_local_scene(scene_data)
        
        # 对global场景使用完整编码（保持原有逻辑）
        node_emb_dict = {}
        
        # 处理macro_zones（只在全局场景中存在）
        if is_global_scene and scene_data["macro_zones"]:
            for zone_id, zone in scene_data["macro_zones"].items():
                funcs = zone.get("function", [])
                desc = "This is a macro zone for: " + (" and ".join(funcs) if funcs else "unknown functions")
                zone_emb = self.encode_text(desc)
                node_emb_dict[("macro", zone_id)] = zone_emb

        # 处理房间数据（global场景）
        room_id_list = list(scene_data["rooms"].keys())
        rooms_data = scene_data["rooms"]
            
        room_id_to_idx = {rid: i for i, rid in enumerate(room_id_list)}
        item_embs_for_room = {room_id: [] for room_id in room_id_list}

        for room_id in room_id_list:
            room = rooms_data[room_id]
            items = room.get("items", {})
            for item_name, item in items.items():
                type_str = item.get("type", "object")
                state_str = item.get("state", "unknown")
                affordances = item.get("affordance", [])
                affordance_str = " ".join(affordances) if affordances else "no interaction"
                social_rule = item.get("social_rule", "")
                social_str = f" Social rule: {social_rule}" if social_rule else ""
                physical_props = item.get("physical_property", {})
                physical_str = " ".join([f"{k}: {v}" for k, v in physical_props.items()]) if physical_props else ""
                item_desc = (f"Item named '{item_name}' is a {type_str}. "
                            f"Its current state is '{state_str}'. "
                            f"Available affordances: {affordance_str}."
                            f"{social_str}"
                            f"{physical_str if physical_str else ''}")
                item_emb = self.encode_text(item_desc)
                item_emb = self.item_proj(item_emb)
                item_key = ("item", f"{room_id}_{item_name}")
                node_emb_dict[item_key] = item_emb
                item_embs_for_room[room_id].append(item_emb)

        room_initial_embs = []
        for room_id in room_id_list:
            room = rooms_data[room_id]
            funcs = room.get("function", [])
            room_desc = "This room is used for: " + (" and ".join(funcs) if funcs else "unknown purposes")

            item_embs = item_embs_for_room[room_id]
            if item_embs:
                item_embs = torch.cat(item_embs, dim=0)
                attn_weights = self.item_attn_gate(item_embs)
                attn_weights = torch.softmax(attn_weights, dim=0)
                room_content_emb = torch.sum(attn_weights * item_embs, dim=0, keepdim=True)
            else:
                room_content_emb = self._zero_emb(device, dtype)

            room_func_emb = self.encode_text(room_desc)
            room_initial_emb = room_func_emb + room_content_emb
            room_initial_embs.append(room_initial_emb)
            node_emb_dict[("room", room_id)] = room_initial_emb

        if room_initial_embs:
            room_inputs = torch.cat(room_initial_embs, dim=0)
            edges = []
            for room_id in room_id_list:
                src_idx = room_id_to_idx[room_id]
                neighbors = rooms_data[room_id].get("neighbor", [])
                for neighbor_id in neighbors:
                    if neighbor_id in room_id_to_idx:
                        tgt_idx = room_id_to_idx[neighbor_id]
                        edges.append([src_idx, tgt_idx])
                        edges.append([tgt_idx, src_idx])

            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
            room_emb_gnn = self.room_gnn(room_inputs.to(dtype), edge_index)
            room_emb_post = self.room_post_gnn(room_emb_gnn)
            room_emb_final = room_emb_post + self.room_residual_proj(room_inputs)

            for i, room_id in enumerate(room_id_list):
                node_emb_dict[("room", room_id)] = room_emb_final[i:i+1]

        agent_room_id = scene_data["agent"]["position"]
        agent_emb = node_emb_dict.get(("room", agent_room_id), self._zero_emb(device, dtype))
        node_emb_dict[("agent", "agent")] = agent_emb

        start_room_id = scene_data["agent"]["position"]
        node_sequence = self.bfs_traversal(scene_data, start_room_id)

        embeddings_list = []
        for node_type, node_id in node_sequence:
            emb = node_emb_dict.get((node_type, node_id), self._zero_emb(device, dtype))
            type_idx = torch.tensor([self.type_to_idx[node_type]], device=device)
            type_emb = self.type_embedding(type_idx)
            emb = emb + type_emb
            embeddings_list.append(emb)

        if not embeddings_list:
            # 如果没有任何节点可以编码，至少返回一个agent节点
            agent_emb = node_emb_dict.get(("agent", "agent"), self._zero_emb(device, dtype))
            agent_type_idx = torch.tensor([self.type_to_idx.get("agent", 0)], device=device)
            agent_type_emb = self.type_embedding(agent_type_idx)
            final_emb = agent_emb + agent_type_emb
            return final_emb

        final_sequence = torch.cat(embeddings_list, dim=0)
        final_sequence[:, -512:] = self.rotary_emb(final_sequence[:, -512:], seq_dim=0)

        return final_sequence.to(dtype=dtype)

    def forward(self, scene_graphs) -> torch.Tensor:
        if isinstance(scene_graphs, dict):
            return self.encode_single_scene(scene_graphs)
        elif isinstance(scene_graphs, list):
            batch_sequences = []
            max_seq_len = 0
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

            for scene_data in scene_graphs:
                seq = self.encode_single_scene(scene_data)
                batch_sequences.append(seq)
                max_seq_len = max(max_seq_len, seq.shape[0])

            if max_seq_len == 0:
                return torch.zeros(len(scene_graphs), 0, self.llm_dim, device=device, dtype=dtype)

            padded_sequences = []
            for seq in batch_sequences:
                seq_len = seq.shape[0]
                if seq_len < max_seq_len:
                    padding = self._zero_emb(device, dtype, max_seq_len - seq_len)
                    padded_seq = torch.cat([seq, padding], dim=0)
                else:
                    padded_seq = seq
                padded_sequences.append(padded_seq.unsqueeze(0))

            return torch.cat(padded_sequences, dim=0).to(dtype=dtype)
        else:
            raise TypeError(f"Expected Dict or List[Dict], got {type(scene_graphs)}")
