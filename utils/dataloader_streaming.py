# utils/dataloader_streaming.py

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Any, Iterator, Optional, Union, Tuple
import json
import random
import math
import os
from tqdm import tqdm
import json
import re

class StreamingSceneGraphInstructionDataset(Dataset):
    """
    支持流式模式标记的数据集
    基于原有SceneGraphInstructionDataset，添加了模式标记支持
    """

    def __init__(
        self, 
        jsonl_file: str, 
        chunk_size: int = 600,
        rank: int = 0,
    ):
        self.jsonl_file = jsonl_file
        self.chunk_size = chunk_size
        self.rank = rank
        
        # 初始化文件格式检测
        self._is_multiline_json = False
        self._json_boundaries = []
        
        # 构建任务级别的chunk偏移（以task为单位）
        self.chunk_offsets = self._build_task_chunk_offsets()
        
        # 计算总样本数和chunk数
        self.total_samples = self._count_total_samples()
        self.num_chunks = len(self.chunk_offsets)
        
        # 初始化：不加载任何数据
        self.current_chunk_idx = -1
        self.current_chunk_data = []
        
        # 任务级别shuffle支持
        self._task_groups = []  # 存储每个task的样本范围



    def _build_task_chunk_offsets(self) -> List[int]:
        """构建以任务为单位的chunk偏移，支持任务级shuffle"""
        offsets = []
        
        try:
            # 先尝试多行JSON格式
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 解析JSON对象边界
            json_boundaries = []
            current_pos = 0
            current_obj = ""
            brace_count = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(content):
                current_obj += char
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        if brace_count == 0:
                            current_pos = i  # 记录JSON对象开始位置
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        
                        if brace_count == 0:
                            # 找到完整的JSON对象（每个对象代表一个task）
                            json_boundaries.append((current_pos, i + 1))
                            current_obj = ""
            
            # 每个task作为一个单位，按chunk_size分组
            for i in range(0, len(json_boundaries), self.chunk_size):
                if i < len(json_boundaries):
                    offsets.append(json_boundaries[i][0])  # chunk开始位置
            
            if offsets:
                self._json_boundaries = json_boundaries
                self._is_multiline_json = True
                return offsets
                
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️  多行JSON格式解析失败: {e}")
        
        # 回退到标准JSONL格式
        if self.rank == 0:
            print("📝 使用标准JSONL格式...")
        
        offsets = []
        self._is_multiline_json = False
        
        # JSONL格式：每行一个task
        with open(self.jsonl_file, 'rb') as f:
            offset = 0
            task_count = 0
            for line in f:
                if task_count % self.chunk_size == 0:
                    offsets.append(offset)
                offset += len(line)
                task_count += 1
        
        return offsets

    def _load_chunk(self, chunk_idx: int):
        """加载指定块的数据到内存"""
        if chunk_idx == self.current_chunk_idx:
            return  # 已加载

        self.current_chunk_data = []
        self._current_task_groups = []  # 重置当前chunk的任务组
        
        if self._is_multiline_json:
            self._load_multiline_json_chunk(chunk_idx)
        else:
            self._load_jsonl_chunk(chunk_idx)
        
        # 保存当前chunk的任务组信息    
        if not hasattr(self, '_chunk_task_groups'):
            self._chunk_task_groups = {}
        self._chunk_task_groups[chunk_idx] = getattr(self, '_current_task_groups', [])
        
        self.current_chunk_idx = chunk_idx

    def _load_multiline_json_chunk(self, chunk_idx: int):
        """加载多行JSON格式的chunk"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self._json_boundaries))
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for i in range(start_idx, end_idx):
            start_pos, end_pos = self._json_boundaries[i]
            json_str = content[start_pos:end_pos]
            
            try:
                data = json.loads(json_str)
                self._process_data_item(data, i)
            except json.JSONDecodeError as e:
                if self.rank == 0:
                    print(f"⚠️  JSON对象 {i} 解析失败: {e}")
                continue

    def _load_jsonl_chunk(self, chunk_idx: int):
        """加载JSONL格式的chunk"""
        start_offset = self.chunk_offsets[chunk_idx]
        end_offset = self.chunk_offsets[chunk_idx + 1] if chunk_idx + 1 < len(self.chunk_offsets) else None

        with open(self.jsonl_file, 'rb') as f:
            f.seek(start_offset)
            count = 0
            while count < self.chunk_size and (end_offset is None or f.tell() < end_offset):
                line = f.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode('utf-8').strip())
                    self._process_data_item(data, start_offset + count)
                    count += 1
                except Exception as e:
                    if self.rank == 0:
                        print(f"⚠️  第{count}行处理失败: {e}")
                    continue

    def _process_data_item(self, data: dict, item_idx: int):
        """处理单个数据项，支持任务级分组"""
        try:
            task_start_idx = len(self.current_chunk_data)  # 记录当前task的起始位置
            
            # 检查数据格式
            if "streaming_samples" in data:
                # HSG-RTP streaming format containing streaming_samples.
                streaming_samples = data.get("streaming_samples", [])
                full_global_plan = data.get("execution_summary", {}).get("global_plan", [])
                full_subtasks = data.get("execution_summary", {}).get("subtasks", [])  # 完整底层动作序列
                
                # 为每个streaming sample创建一个训练样本（保持顺序）
                for j, sample_data in enumerate(streaming_samples):
                    # 提取streaming sample的字段
                    mode = sample_data.get("mode", "none")
                    context = sample_data.get("context", "")
                    target = sample_data.get("target", "")
                    completed = sample_data.get("completed", [])
                    pending = sample_data.get("pending", [])
                    scene_graph = sample_data.get("scene_graph", {})
                    sample_metadata = sample_data.get("metadata", {})
                    
                    # 从context中提取instruction（如果有的话）
                    instruction = (
                        sample_data.get("instruction_override")
                        or data["instruction"]
                    )
                    
                    # 构建JSON格式的目标输出
                    if mode == "global" and target:
                        # target 已是字符串列表，如 ["goto(A):act1", "goto(B):act2"]
                        task_value = target
                    elif mode == "local" and target:
                        # target 是字符串，如 "goto(elevator_1f)" → 包装成单元素列表
                        task_value = [target]
                    else:
                        # 默认情况：空列表或占位
                        task_value = [""]

                    formatted_target = json.dumps(
                        {"mode": mode, "task": task_value},
                        separators=(',', ':'),
                        ensure_ascii=False
                    )
                    sample = {
                        "scene_graph": scene_graph,
                        "instruction": instruction,
                        "subtasks": formatted_target,  # 使用格式化后的target作为subtasks
                        "context": context,  # 保留原始context
                        "mode": mode,  # 保留模式信息
                        "completed":completed, #历史已完成的local步
                        "pending":pending, #当前待完成的global步
                        "metadata": {
                            "id": f"task_{item_idx}_sample_{j}",
                            "scene_name": data.get("scene_name", "unknown"),
                            "difficulty": data.get("task_info", {}).get("difficulty", "unknown"),
                            "task_type": data.get("task_info", {}).get("type", "unknown"),
                            "mode": mode,
                            "sample_metadata": sample_metadata,
                            "task_id": item_idx,  # 添加任务ID
                            "step_in_task": j,     # 添加在任务中的步骤序号
                            "full_global_plan": full_global_plan,   # List[str]
                            "full_subtasks": full_subtasks,         # List[str]
                        }
                    }
                    self.current_chunk_data.append(sample)
                
                # 记录这个task的样本范围（用于shuffle时保持组内顺序）
                task_end_idx = len(self.current_chunk_data)
                if task_end_idx > task_start_idx:
                    if not hasattr(self, '_current_task_groups'):
                        self._current_task_groups = []
                    self._current_task_groups.append((task_start_idx, task_end_idx, item_idx))
                    
            elif "initial_state" in data:
                # 原有格式（单个样本作为一个task）
                sample = {
                    "scene_graph": data["initial_state"],
                    "instruction": data["instruction"],
                    "subtasks": data["subtasks"],
                    "global_plan": data.get("global_plan", []),
                    "metadata": {
                        "id": data.get("id", f"sample_{item_idx}"),
                        "scene_name": data.get("scene_name", "unknown"),
                        "difficulty": data.get("difficulty", "unknown"),
                        "task_type": data.get("task_type", "unknown"),
                        "complexity": data.get("complexity", "unknown"),
                        "expected_steps": data.get("expected_steps", 0),
                        "actual_steps": data.get("actual_steps", 0),
                        "variant_id": data.get("variant_id", "N/A"),
                        "task_id": item_idx,
                        "step_in_task": 0
                    }
                }
                self.current_chunk_data.append(sample)
                
                # 单个样本也记录为一个task组
                task_end_idx = len(self.current_chunk_data)
                if not hasattr(self, '_current_task_groups'):
                    self._current_task_groups = []
                self._current_task_groups.append((task_start_idx, task_end_idx, item_idx))
                
            else:
                # 简单格式：直接包含scene_graph等字段（单个样本作为一个task）
                sample = {
                    "scene_graph": data.get("scene_graph", data.get("initial_state", {})),
                    "instruction": data["instruction"],
                    "subtasks": data["subtasks"],
                    "global_plan": data.get("global_plan", []),
                    "metadata": {
                        "id": data.get("id", f"sample_{item_idx}"),
                        "scene_name": data.get("scene_name", "unknown"),
                        "difficulty": data.get("difficulty", "unknown"),
                        "task_type": data.get("task_type", "unknown"),
                        "task_id": item_idx,
                        "step_in_task": 0
                    }
                }
                self.current_chunk_data.append(sample)
                
                # 单个样本也记录为一个task组
                task_end_idx = len(self.current_chunk_data)
                if not hasattr(self, '_current_task_groups'):
                    self._current_task_groups = []
                self._current_task_groups.append((task_start_idx, task_end_idx, item_idx))
                
        except Exception as e:
            if self.rank == 0:
                print(f"⚠️  数据项 {item_idx} 处理失败: {e}")
            return

    def _count_total_samples(self) -> int:
        """计算总样本数"""
        if hasattr(self, '_total_samples_cached'):
            return self._total_samples_cached
            
        total = 0
        try:
            # 尝试多行JSON格式
            with open(self.jsonl_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 解析JSON对象数量
            json_count = 0
            brace_count = 0
            in_string = False
            escape_next = False
            
            for char in content:
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_count += 1
            
            if json_count > 0:
                # 需要检查streaming_samples来计算实际样本数
                # 为了性能，这里先估算，实际数量在加载时确定
                total = json_count * 3  # 估算每个JSON对象有3个streaming samples
                self._is_multiline_json = True
            else:
                raise Exception("No valid JSON objects found")
                
        except Exception:
            # 回退到JSONL格式
            self._is_multiline_json = False
            with open(self.jsonl_file, 'rb') as f:
                total = sum(1 for _ in f)
        
        self._total_samples_cached = total
        return total



    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        """根据全局索引返回样本"""
        # 计算chunk和本地索引（这里需要更复杂的逻辑来处理动态样本数）
        # 为了简化，我们先加载所有chunks来建立索引映射
        if not hasattr(self, '_sample_to_chunk_map'):
            self._build_sample_index()
        
        if global_idx >= len(self._sample_to_chunk_map):
            global_idx = len(self._sample_to_chunk_map) - 1
            
        chunk_idx, local_idx = self._sample_to_chunk_map[global_idx]
        
        # 加载对应的chunk
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)
        
        if local_idx >= len(self.current_chunk_data):
            local_idx = len(self.current_chunk_data) - 1
            
        return self.current_chunk_data[local_idx]
    
    def _build_sample_index(self):
        """构建样本到chunk的映射"""
        self._sample_to_chunk_map = []
        sample_count = 0
        
        for chunk_idx in range(len(self.chunk_offsets)):
            # 临时加载chunk来统计样本数
            temp_current_chunk = self.current_chunk_idx
            temp_current_data = self.current_chunk_data.copy() if self.current_chunk_data else []
            
            self._load_chunk(chunk_idx)
            chunk_sample_count = len(self.current_chunk_data)
            
            # 建立映射
            for local_idx in range(chunk_sample_count):
                self._sample_to_chunk_map.append((chunk_idx, local_idx))
            
            sample_count += chunk_sample_count
            
            # 恢复之前的状态
            self.current_chunk_idx = temp_current_chunk
            self.current_chunk_data = temp_current_data
        
        # 更新实际的总样本数
        self._total_samples_cached = sample_count
        self.total_samples = sample_count
        
        if self.rank == 0:
            print(f"✅ 索引构建完成，实际样本数: {sample_count}")
    
    def _count_lines(self, file_path: str) -> int:
        """计算文件行数（兼容性保持）"""
        return self._count_total_samples()


class TaskLevelShuffleSampler(Sampler):
    """任务级shuffle采样器 - 在task之间shuffle，但保持task内部样本的顺序"""
    
    def __init__(
        self,
        dataset: 'StreamingSceneGraphInstructionDataset',
        shuffle: bool = True,
        seed: int = 0
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 构建完整的任务组映射
        self._build_task_mapping()
    
    def _build_task_mapping(self):
        """构建任务到样本的映射"""
        self.task_groups = []  # [(start_idx, end_idx, task_id), ...]
        
        # 确保数据集已经构建好索引
        if not hasattr(self.dataset, '_sample_to_chunk_map'):
            self.dataset._build_sample_index()
        
        # 遍历所有chunks收集任务组信息
        current_sample_idx = 0
        for chunk_idx in range(len(self.dataset.chunk_offsets)):
            # 临时加载chunk来获取任务组信息
            temp_current_chunk = self.dataset.current_chunk_idx
            temp_current_data = self.dataset.current_chunk_data.copy() if self.dataset.current_chunk_data else []
            
            self.dataset._load_chunk(chunk_idx)
            
            # 获取这个chunk的任务组
            chunk_task_groups = getattr(self.dataset, '_chunk_task_groups', {}).get(chunk_idx, [])
            
            # 转换为全局索引
            for start_local, end_local, task_id in chunk_task_groups:
                global_start = current_sample_idx + start_local
                global_end = current_sample_idx + end_local
                self.task_groups.append((global_start, global_end, task_id))
            
            current_sample_idx += len(self.dataset.current_chunk_data)
            
            # 恢复之前的状态
            self.dataset.current_chunk_idx = temp_current_chunk
            self.dataset.current_chunk_data = temp_current_data
    
    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # 打乱任务顺序，但保持每个任务内部的样本顺序
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # 生成打乱后的任务索引
            task_indices = torch.randperm(len(self.task_groups), generator=g).tolist()
            
            # 按打乱后的任务顺序生成样本索引
            indices = []
            for task_idx in task_indices:
                start_idx, end_idx, _ = self.task_groups[task_idx]
                # 保持任务内部样本的原始顺序
                indices.extend(list(range(start_idx, end_idx)))
        else:
            # 不shuffle时，保持原有顺序
            indices = list(range(len(self.dataset)))
        
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class TaskLevelDistributedSampler(Sampler):
    """任务级分布式采样器 - 支持任务级shuffle + 分布式训练"""

    def __init__(
        self, 
        dataset: 'StreamingSceneGraphInstructionDataset', 
        num_replicas: Optional[int] = None, 
        rank: Optional[int] = None, 
        shuffle: bool = True, 
        seed: int = 0,
        rank_nums: Optional[int] = None  # 新增参数，指定每个rank的编号列表
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.rank_nums = rank_nums 
        
        # 使用任务级采样器
        self.task_sampler = TaskLevelShuffleSampler(dataset, shuffle, seed)
        
        # 计算分布式参数
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # # 使用任务级shuffle获取索引
        # self.task_sampler.set_epoch(self.epoch)
        # indices = list(self.task_sampler)
        
        # # 填充到总大小以确保所有GPU获得相同数量的样本
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # # 为每个worker分配索引
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        # assert len(indices) == self.num_samples

        # return iter(indices)
        # 使用任务级shuffle获取索引
        self.task_sampler.set_epoch(self.epoch)
        indices = list(self.task_sampler)
        
        # 👇 关键：确保每个 rank 的样本数能被 gradient_accumulation_steps 整除

        # 计算每个 rank 应该有的样本数（向下取整到能被 grad_acc_steps 整除）
        total_samples = len(indices)
        samples_per_rank_raw = total_samples // self.num_replicas
        samples_per_rank = (samples_per_rank_raw // self.rank_nums ) * self.rank_nums 
        
        # 确保不超出总样本数
        total_needed = samples_per_rank * self.num_replicas
        if total_needed > total_samples:
            samples_per_rank = ((total_samples // self.num_replicas) // self.rank_nums ) * self.rank_nums 
            total_needed = samples_per_rank * self.num_replicas
        
        # 截断 indices 到 total_needed
        indices = indices[:total_needed]
        
        # 为每个worker分配索引
        indices = indices[self.rank:total_needed:self.num_replicas]
        assert len(indices) == samples_per_rank

        return iter(indices)

    def __len__(self) -> int:
        # return self.num_samples
        # 重新计算长度
        total_samples = len(self.task_sampler)

        samples_per_rank_raw = total_samples // self.num_replicas
        samples_per_rank = (samples_per_rank_raw // self.rank_nums ) * self.rank_nums
        return samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class StreamingDistributedSampler(Sampler):
    """原始流式分布式采样器（向后兼容）"""

    def __init__(
        self, 
        dataset: Dataset, 
        num_replicas: Optional[int] = None, 
        rank: Optional[int] = None, 
        shuffle: bool = True, 
        seed: int = 0
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # 填充到总大小
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # 为每个worker分配索引
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def streaming_scene_graph_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    流式模式的 collate 函数
    处理带模式标记的数据
    """
    scene_graph_batch = [item["scene_graph"] for item in batch]
    instruction_batch = [item["instruction"] for item in batch]
    subtasks_batch = [item["subtasks"] for item in batch]  
    completed_batch = [item.get("completed", []) for item in batch]
    pending_batch = [item.get("pending", []) for item in batch]
    metadata_batch = [item.get("metadata", {}) for item in batch]

    return {
        "scene_graphs": scene_graph_batch,
        "instructions": instruction_batch,
        "subtasks": subtasks_batch,
        "completed": completed_batch,
        "pending": pending_batch,
        "metadata": metadata_batch
    }


class StreamingSceneGraphDataLoader:
    """流式场景图数据加载器"""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 4,
        chunk_size: int = 600,
        mode_augment: bool = False,  # 默认关闭，数据已包含模式
        shuffle: bool = True,
        num_workers: int = 0,
        rank: int = 0,
        world_size: int = 1,
        distributed: bool = False,
        pin_memory: bool = True,  # 添加pin_memory参数
        seed: Optional[int] = None,  # 添加seed参数
        rank_nums: int = 1  # youduoshaoge rank
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.mode_augment = mode_augment
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rank = rank
        self.pin_memory = pin_memory
        self.seed = seed
        self.world_size = world_size
        self.distributed = distributed
        self.rank_nums = rank_nums  # 每

        # 创建数据集
        self.dataset = StreamingSceneGraphInstructionDataset(
            dataset_path,
            chunk_size=chunk_size,
            rank=rank,
        )

        # 创建采样器 - 使用任务级shuffle
        if distributed:
            self.sampler = TaskLevelDistributedSampler(
                self.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed if seed is not None else 0,
                rank_nums=self.rank_nums
            )
            shuffle_for_dataloader = False  # 分布式时不在DataLoader中shuffle
        else:
            # 单卡训练也使用任务级shuffle
            if shuffle:
                self.sampler = TaskLevelShuffleSampler(
                    self.dataset,
                    shuffle=shuffle,
                    seed=seed if seed is not None else 0
                )
                shuffle_for_dataloader = False
            else:
                self.sampler = None
                shuffle_for_dataloader = False

        if not hasattr(self.dataset, '_sample_to_chunk_map'):
            self.dataset._build_sample_index()

        # 设置随机种子（如果提供）
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)

        # 创建DataLoader - 分布式训练时建议使用较少的worker避免冲突
        # 但不强制为0，给用户选择权
        if distributed and num_workers > 2:
            if rank == 0:
                print(f"⚠️  分布式训练建议使用较少的num_workers (当前: {num_workers})")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            shuffle=shuffle_for_dataloader,
            collate_fn=streaming_scene_graph_collate_fn,
            num_workers=num_workers,  # 使用用户指定的值
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False  # 添加persistent_workers优化
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def set_epoch(self, epoch: int):
        """设置epoch，用于分布式训练"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)


# 使用示例
if __name__ == "__main__":
    # 测试流式数据加载器
    data_path = "pipeline/output/example.json"
    
    # 创建流式数据加载器
    loader = StreamingSceneGraphDataLoader(
        dataset_path=data_path,
        batch_size=2,
        chunk_size=100,
        shuffle=True,
        num_workers=0
    )
    
    print(f"数据集大小: {len(loader.dataset)}")
    print(f"批次数量: {len(loader)}")
    
    # 测试几个批次
    for i, batch in enumerate(loader):
        if i >= 3:  # 只测试前3个批次
            break
            
        print(f"\n=== 批次 {i+1} ===")
        print(f"场景图数量: {len(batch['scene_graphs'])}")
        print(f"指令数量: {len(batch['instructions'])}")

        for j, (instruction, subtask, completed, pending) in enumerate(zip(batch['instructions'], batch['subtasks'], batch['completed'], batch['pending'])):
            print(f"样本{j+1}:")
            print(f"  指令: {instruction[:50]}...")
            print(f"  任务: {subtask[:80]}...")
            print(f"  历史已完成: {completed}")
            print(f"  当前待完成: {pending}")
            # 解析JSON格式
            json_match = re.search(r'\{.*\}', subtask, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                mode = parsed.get('mode', 'unknown')
                content = parsed.get('task', '')
                print(f"  模式: {mode} (JSON格式)")
                print(f"  内容: {content[:50]}...")
            else:
                print(f"  模式: 未识别JSON格式")
