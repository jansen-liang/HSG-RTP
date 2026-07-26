# utils/dataloader.py

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Dict, Any, Iterator, Optional, Union, Tuple
import json
import random
import math
import os

class SceneGraphInstructionDataset(Dataset):
    """支持分块加载的数据集"""

    def __init__(self, jsonl_file: str, chunk_size: int = 600,rank:int=0):
        self.jsonl_file = jsonl_file
        self.chunk_size = chunk_size
        self.total_samples = self._count_lines(jsonl_file)
        self.num_chunks = math.ceil(self.total_samples / chunk_size)
        self.rank=rank
        # 初始化：不加载任何数据
        self.current_chunk_idx = -1
        self.current_chunk_data = []
        self.chunk_offsets = self._build_chunk_offsets()

    def _count_lines(self, file_path: str) -> int:
        with open(file_path, 'rb') as f:
            return sum(1 for _ in f)

    def _build_chunk_offsets(self) -> List[int]:
        """预计算每个 chunk 的起始行偏移"""
        offsets = []
        with open(self.jsonl_file, 'rb') as f:
            offset = 0
            for i, line in enumerate(f):
                if i % self.chunk_size == 0:
                    offsets.append(offset)
                offset += len(line)
        return offsets

    def _load_chunk(self, chunk_idx: int):
        """加载指定块的数据到内存"""
        if chunk_idx == self.current_chunk_idx:
            return  # 已加载

        start_offset = self.chunk_offsets[chunk_idx]
        end_offset = self.chunk_offsets[chunk_idx + 1] if chunk_idx + 1 < len(self.chunk_offsets) else None

        self.current_chunk_data = []
        with open(self.jsonl_file, 'rb') as f:
            f.seek(start_offset)
            count = 0
            while count < self.chunk_size and (end_offset is None or f.tell() < end_offset):
                line = f.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode('utf-8').strip())
                    sample = {
                        "scene_graph": data["initial_state"],
                        "instruction": data["instruction"],
                        "subtasks": data["subtasks"],
                        "metadata": {
                            "id": data.get("id", f"line_{start_offset + count}"),
                            "scene_name": data.get("scene_name", "unknown"),
                            "difficulty": data.get("difficulty", "unknown"),
                            "task_type": data.get("task_type", "unknown"),
                            "complexity": data.get("complexity", "unknown"),
                            "expected_steps": data.get("expected_steps", 0),
                            "actual_steps": data.get("actual_steps", 0),
                            "variant_id": data.get("variant_id", "N/A")
                        }
                    }
                    self.current_chunk_data.append(sample)
                    count += 1
                except Exception as e:
                    print(f"加载 chunk {chunk_idx} 时跳过一行: {e}")
                    continue

        self.current_chunk_idx = chunk_idx

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        """根据全局索引返回样本"""
        chunk_idx = global_idx // self.chunk_size
        local_idx = global_idx % self.chunk_size

        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)

        if local_idx >= len(self.current_chunk_data):
            raise IndexError(f"local_idx {local_idx} 超出当前 chunk 范围")

        return self.current_chunk_data[local_idx]

class DistributedSceneGraphSampler(Sampler):
    """用于 DistributedDataParallel 的采样器"""

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True, seed: int = 0):
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

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

def scene_graph_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    自定义 collate 函数。
    将一批样本组合成一个批次。
    """
    scene_graph_batch = [item["scene_graph"] for item in batch]
    instruction_batch = [item["instruction"] for item in batch]
    subtasks_batch = [item["subtasks"] for item in batch]  # <-- 关键修改
    metadata_batch = [item["metadata"] for item in batch]

    return {
        "scene_graphs": scene_graph_batch,      # List[Dict] - 长度为 batch_size
        "instructions": instruction_batch,      # List[str] - 长度为 batch_size
        "subtasks": subtasks_batch,                # List[str] - 长度为 batch_size
        "metadata": metadata_batch              # List[Dict] - 长度为 batch_size
    }

class SceneGraphDataLoader:
    """高层次封装，方便创建 DataLoader"""

    def __init__(
        self,
        jsonl_file: str, 
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        use_ddp: bool = False,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        seed: int = 42,
        chunk_size: int = 200
    ):
        """
        Args:
            jsonl_file: 包含任务数据的 JSONL 文件路径。
            其余参数同标准 DataLoader
        """
        # 修改: 初始化 Dataset 时只传入 jsonl_file
        self.dataset = SceneGraphInstructionDataset(jsonl_file, chunk_size=chunk_size, rank=ddp_rank if use_ddp else 0)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.use_ddp = use_ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.seed = seed

        # 创建 Sampler
        if self.use_ddp:
            self.sampler = DistributedSceneGraphSampler(
                self.dataset,
                num_replicas=self.ddp_world_size,
                rank=self.ddp_rank,
                shuffle=self.shuffle,
                seed=self.seed
            )
            shuffle_in_loader = False
        else:
            self.sampler = None
            shuffle_in_loader = self.shuffle

        # 创建 PyTorch DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_in_loader,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=scene_graph_collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def set_epoch(self, epoch: int):
        """在每个 epoch 开始时调用，用于 DDP 的 shuffle"""
        if self.use_ddp and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

# ============ 使用和测试示例 ============

if __name__ == "__main__":
    import os

    # ✅ 安全默认值：单卡测试
    use_ddp = False
    ddp_rank = 0
    ddp_world_size = 1

    # ✅ 仅当环境变量存在时才启用 DDP（用于 torchrun 启动）
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_ddp = True
        ddp_rank = int(os.environ['RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(ddp_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"✅ DDP initialized: rank={ddp_rank}, world_size={ddp_world_size}")

    # 2. 创建 DataLoader
    dataloader = SceneGraphDataLoader(
        jsonl_file="pipeline/output/train.jsonl",
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        use_ddp=use_ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        seed=42,
        chunk_size=400
    )

    print(f"Rank {ddp_rank}: Dataloader 创建成功，总 batch 数: {len(dataloader)}")

    try:
        # 3. 迭代数据
        for epoch in range(1):
            if use_ddp:
                dataloader.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                scene_graphs = batch["scene_graphs"]
                instructions = batch["instructions"]
                answers = batch["answers"]
                metadata = batch["metadata"]

                if batch_idx >= 2:
                    break

    finally:
        if use_ddp and dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Distributed process group destroyed.")
