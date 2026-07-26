"""
Edge 基类体系
定义场景图中所有边的基类和子类，支持物理与逻辑关系的区分
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class EdgeType(Enum):
    """边类型枚举 (细粒度分类)"""
    OBJECT_EDGE = "object_edge"         # 物体-物体 (e.g., on, next_to)
    OBJECT_ROOM_EDGE = "object_room_edge" # 物体-房间 (e.g., inside)
    ROOM_EDGE = "room_edge"             # 房间-房间 (e.g., neighbour)
    ROOM_FLOOR_EDGE = "room_floor_edge" # 房间-楼层 (e.g., belongs_to)
    TRANSPORT_EDGE = "transport_edge"   # 交通工具-站点 (e.g., parked_at)
    CONTROL_EDGE = "control_edge"       # 新增: 控制/逻辑关系 (e.g., controls)


class EdgeCategory(Enum):
    """
    边的大类：决定了可视化的连线方式和机器人的理解方式
    """
    PHYSICAL = "physical"  # 【实线】物理连接
    LOGICAL = "logical"    # 【虚线】逻辑连接


class SpatialRelation(Enum):
    """空间与逻辑关系枚举"""
    # --- Physical Relations (Where?) ---
    ONTOP = "ontop"
    INSIDE = "inside"
    UNDER = "under"
    BESIDE = "beside"
    NEXT_TO = "next_to"
    NEAR = "near"
    FAR = "far"
    NEIGHBOUR = "neighbour"      
    CONTAINS = "contains"        
    BELONGS_TO = "belongs_to"    
    CONNECTED = "connected"      
    
    # --- Logical Relations (What?) ---
    CONTROLS = "controls"        
    LINKED_TO = "linked_to"      
    POWERED_BY = "powered_by"    


@dataclass
class BaseEdge(ABC):
    """场景图边基类"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    relation: SpatialRelation
    
    # 默认为 PHYSICAL
    category: EdgeCategory = field(default=EdgeCategory.PHYSICAL)
    
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if not self.source_id:
            raise ValueError("Source ID cannot be empty")
        if not self.target_id:
            raise ValueError("Target ID cannot be empty")

        # 自动推断 Category (防止手动漏写)
        if self.relation in {SpatialRelation.CONTROLS, SpatialRelation.LINKED_TO, SpatialRelation.POWERED_BY}:
            self.category = EdgeCategory.LOGICAL
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "relation": self.relation.value,
            "category": self.category.value,
            "properties": self.properties
        }
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEdge':
        pass
    
    def update_property(self, key: str, value: Any) -> None:
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)
    
    def __repr__(self) -> str:
        return f"[{self.category.value.upper()}] {self.source_id} --{self.relation.value}--> {self.target_id}"
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.relation.value))
    
    def __eq__(self, other):
        if not isinstance(other, BaseEdge):
            return False
        return (self.source_id == other.source_id and 
                self.target_id == other.target_id and 
                self.relation == other.relation)


@dataclass
class ObjectEdge(BaseEdge):
    """物体间的边"""
    distance: Optional[float] = None
    
    def __post_init__(self):
        if self.edge_type not in {EdgeType.OBJECT_EDGE, EdgeType.CONTROL_EDGE}:
            self.edge_type = EdgeType.OBJECT_EDGE
        super().__post_init__()
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        if self.distance is not None:
            base["distance"] = self.distance
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectEdge':
        cat = EdgeCategory(data.get("category", "physical"))
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            relation=SpatialRelation(data["relation"]),
            category=cat,
            distance=data.get("distance"),
            properties=data.get("properties", {})
        )


@dataclass
class ObjectRoomEdge(BaseEdge):
    """物体-房间边"""
    def __post_init__(self):
        if self.edge_type != EdgeType.OBJECT_ROOM_EDGE:
            self.edge_type = EdgeType.OBJECT_ROOM_EDGE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectRoomEdge':
        cat = EdgeCategory(data.get("category", "physical"))
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType.OBJECT_ROOM_EDGE,
            relation=SpatialRelation(data["relation"]),
            category=cat,
            properties=data.get("properties", {})
        )


@dataclass
class RoomEdge(BaseEdge):
    """房间-房间边"""
    is_accessible: bool = True
    door_state: str = "open"
    
    def __post_init__(self):
        if self.edge_type != EdgeType.ROOM_EDGE:
            self.edge_type = EdgeType.ROOM_EDGE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"is_accessible": self.is_accessible, "door_state": self.door_state})
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoomEdge':
        cat = EdgeCategory(data.get("category", "physical"))
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType.ROOM_EDGE,
            relation=SpatialRelation(data["relation"]),
            category=cat,
            is_accessible=data.get("is_accessible", True),
            door_state=data.get("door_state", "open"),
            properties=data.get("properties", {})
        )


@dataclass
class RoomFloorEdge(BaseEdge):
    """房间-楼层边"""
    def __post_init__(self):
        if self.edge_type != EdgeType.ROOM_FLOOR_EDGE:
            self.edge_type = EdgeType.ROOM_FLOOR_EDGE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoomFloorEdge':
        cat = EdgeCategory(data.get("category", "physical"))
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType.ROOM_FLOOR_EDGE,
            relation=SpatialRelation(data["relation"]),
            category=cat,
            properties=data.get("properties", {})
        )


@dataclass
class TransportEdge(BaseEdge):
    """交通工具边"""
    def to_dict(self) -> Dict[str, Any]:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransportEdge':
        cat = EdgeCategory(data.get("category", "physical"))
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            relation=SpatialRelation(data["relation"]),
            category=cat,
            properties=data.get("properties", {})
        )


# --- 工厂函数 (Fix applied here) ---
def create_edge(source_id: str, target_id: str, relation: SpatialRelation, **kwargs) -> BaseEdge:
    """工厂函数：自动根据 relation 决定类型"""
    
    # 🔥 核心修复：防止 kwargs 里的 category/edge_type 与硬编码参数冲突
    # 如果 kwargs 里有 category，先弹出来，防止重复传参
    if "category" in kwargs:
        kwargs.pop("category")
    if "edge_type" in kwargs:
        kwargs.pop("edge_type")

    # 1. 逻辑控制关系
    if relation in {SpatialRelation.CONTROLS, SpatialRelation.LINKED_TO}:
        return ObjectEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.CONTROL_EDGE, 
            relation=relation,
            category=EdgeCategory.LOGICAL,   # 👈 这里硬编码了，所以必须 pop 掉 kwargs 里的
            **kwargs
        )

    # 2. 物体空间关系
    elif relation in {SpatialRelation.ONTOP, SpatialRelation.INSIDE, 
                      SpatialRelation.UNDER, SpatialRelation.BESIDE,
                      SpatialRelation.NEAR, SpatialRelation.FAR, 
                      SpatialRelation.CONTAINS,SpatialRelation.NEXT_TO}: 
        return ObjectEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.OBJECT_EDGE,
            relation=relation,
            category=EdgeCategory.PHYSICAL,
            **kwargs
        )
    
    # 3. 房间邻接
    elif relation in {SpatialRelation.NEIGHBOUR, SpatialRelation.CONNECTED}:
        return RoomEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=EdgeType.ROOM_EDGE,
            relation=relation,
            category=EdgeCategory.PHYSICAL,
            **kwargs
        )
    
    # 默认兜底
    raise ValueError(f"Unknown or unmapped spatial relation: {relation}")