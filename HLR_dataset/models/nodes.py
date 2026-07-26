"""
Node 基类体系
定义场景图中所有节点的基类和子类
"""

from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """节点类型枚举"""
    OBJECT = "object"
    ROOM = "room"
    FLOOR = "floor"
    MOBILE_TOOL = "mobile_tool"
    AGENT = "agent"


@dataclass
class BaseNode(ABC):
    """
    场景图节点基类
    所有物体、房间、楼层等都继承此类
    """
    id: str
    name: str
    node_type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后的验证"""
        if not self.id:
            raise ValueError("Node id cannot be empty")
        if not self.name:
            raise ValueError("Node name cannot be empty")
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于序列化）"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "properties": self.properties
        }
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseNode':
        """从字典创建节点（用于反序列化）"""
        pass
    
    def update_property(self, key: str, value: Any) -> None:
        """更新节点属性"""
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """获取节点属性"""
        return self.properties.get(key, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, name={self.name})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, BaseNode):
            return False
        return self.id == other.id


@dataclass
class Object(BaseNode):
    """
    物体节点
    表示场景中的具体物体（small_objects 和 large_objects）
    """
    object_type: str = ""  # 如 "food", "tool", "furniture"
    is_container: bool = False
    affordances: List[str] = field(default_factory=list)  # 可执行的动作
    states: Dict[str, Any] = field(default_factory=dict)  # 物体状态
    physical_properties: Dict[str, Any] = field(default_factory=dict)  # 物理属性
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_type != NodeType.OBJECT:
            self.node_type = NodeType.OBJECT
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "object_type": self.object_type,
            "is_container": self.is_container,
            "affordances": self.affordances,
            "states": self.states,
            "physical_properties": self.physical_properties
        })
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Object':
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.OBJECT,
            object_type=data.get("object_type", ""),
            is_container=data.get("is_container", False),
            affordances=data.get("affordances", []),
            states=data.get("states", {}),
            physical_properties=data.get("physical_properties", {}),
            properties=data.get("properties", {})
        )
    
    def has_affordance(self, action: str) -> bool:
        """检查是否支持某个动作"""
        return action in self.affordances
    
    def get_state(self, state_name: str, default: Any = None) -> Any:
        """获取物体状态"""
        return self.states.get(state_name, default)
    
    def set_state(self, state_name: str, value: Any) -> None:
        """设置物体状态"""
        self.states[state_name] = value


@dataclass
class Room(BaseNode):
    """
    房间节点
    继承自 BaseNode，添加了 neighbours（相邻房间）
    """
    floor_id: str = ""  # 所属楼层
    neighbours: Set[str] = field(default_factory=set)  # 相邻房间 ID
    contained_objects: Set[str] = field(default_factory=set)  # 房间内的物体 ID
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_type != NodeType.ROOM:
            self.node_type = NodeType.ROOM
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "floor_id": self.floor_id,
            "neighbours": list(self.neighbours),
            "contained_objects": list(self.contained_objects)
        })
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Room':
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.ROOM,
            floor_id=data.get("floor_id", ""),
            neighbours=set(data.get("neighbours", [])),
            contained_objects=set(data.get("contained_objects", [])),
            properties=data.get("properties", {})
        )
    
    def add_neighbour(self, room_id: str) -> None:
        """添加相邻房间"""
        self.neighbours.add(room_id)
    
    def remove_neighbour(self, room_id: str) -> None:
        """移除相邻房间"""
        self.neighbours.discard(room_id)
    
    def is_neighbour(self, room_id: str) -> bool:
        """检查是否为相邻房间"""
        return room_id in self.neighbours
    
    def add_object(self, object_id: str) -> None:
        """添加物体到房间"""
        self.contained_objects.add(object_id)
    
    def remove_object(self, object_id: str) -> None:
        """从房间移除物体"""
        self.contained_objects.discard(object_id)
    
    def has_object(self, object_id: str) -> bool:
        """检查房间是否包含某物体"""
        return object_id in self.contained_objects


@dataclass
class Floor(BaseNode):
    """
    楼层节点
    继承自 Room，包含多个房间
    """
    rooms: Set[str] = field(default_factory=set)  # 该楼层的房间 ID 列表
    floor_number: int = 1  # 楼层编号
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_type != NodeType.FLOOR:
            self.node_type = NodeType.FLOOR
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "rooms": list(self.rooms),
            "floor_number": self.floor_number
        })
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Floor':
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.FLOOR,
            rooms=set(data.get("rooms", [])),
            floor_number=data.get("floor_number", 1),
            properties=data.get("properties", {})
        )
    
    def add_room(self, room_id: str) -> None:
        """添加房间到楼层"""
        self.rooms.add(room_id)
    
    def remove_room(self, room_id: str) -> None:
        """从楼层移除房间"""
        self.rooms.discard(room_id)
    
    def has_room(self, room_id: str) -> bool:
        """检查楼层是否包含某房间"""
        return room_id in self.rooms


@dataclass
class MobileTool(BaseNode):
    """
    移动型工具节点（如电梯、推车等）
    可以在不同位置之间移动
    """
    object_type: str = "mobile_tool" # 工具类型 (e.g. elevator, cart)
    current_location: str = ""  # 当前位置（房间/楼层 ID）
    accessible_locations: Set[str] = field(default_factory=set)  # 可到达的位置
    capacity: int = 1  # 容量（如电梯能承载的人数或物品数量）
    is_occupied: bool = False  # 是否被占用
    states: Dict[str, Any] = field(default_factory=dict)  # 状态
    physical_properties: Dict[str, Any] = field(default_factory=dict)  # 物理属性
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_type != NodeType.MOBILE_TOOL:
            self.node_type = NodeType.MOBILE_TOOL
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "object_type": self.object_type,
            "current_location": self.current_location,
            "accessible_locations": list(self.accessible_locations),
            "capacity": self.capacity,
            "is_occupied": self.is_occupied,
            "states": self.states,
            "physical_properties": self.physical_properties
        })
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MobileTool':
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.MOBILE_TOOL,
            object_type=data.get("object_type", "mobile_tool"),
            current_location=data.get("current_location", ""),
            accessible_locations=set(data.get("accessible_locations", [])),
            capacity=data.get("capacity", 1),
            is_occupied=data.get("is_occupied", False),
            states=data.get("states", {}),
            physical_properties=data.get("physical_properties", {}),
            properties=data.get("properties", {})
        )
    
    def move_to(self, location_id: str) -> bool:
        """移动到指定位置"""
        if location_id in self.accessible_locations:
            self.current_location = location_id
            return True
        return False
    
    def add_accessible_location(self, location_id: str) -> None:
        """添加可到达位置"""
        self.accessible_locations.add(location_id)
    
    def can_access(self, location_id: str) -> bool:
        """检查是否可以到达某位置"""
        return location_id in self.accessible_locations


@dataclass
class Agent(BaseNode):
    """
    智能体节点（机器人或人）
    """
    current_room: str = ""  # 当前所在房间
    inventory: Set[str] = field(default_factory=set)  # 持有的物体 ID
    max_inventory: int = 2  # 最大持有数量
    state: Dict[str, Any] = field(default_factory=dict)  # 智能体状态
    
    def __post_init__(self):
        super().__post_init__()
        if self.node_type != NodeType.AGENT:
            self.node_type = NodeType.AGENT
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "current_room": self.current_room,
            "inventory": list(self.inventory),
            "max_inventory": self.max_inventory,
            "state": self.state
        })
        return base
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType.AGENT,
            current_room=data.get("current_room", ""),
            inventory=set(data.get("inventory", [])),
            max_inventory=data.get("max_inventory", 2),
            state=data.get("state", {}),
            properties=data.get("properties", {})
        )
    
    def pick_object(self, object_id: str) -> bool:
        """拾取物体"""
        if len(self.inventory) < self.max_inventory:
            self.inventory.add(object_id)
            return True
        return False
    
    def drop_object(self, object_id: str) -> bool:
        """放下物体"""
        if object_id in self.inventory:
            self.inventory.remove(object_id)
            return True
        return False
    
    def has_object(self, object_id: str) -> bool:
        """检查是否持有某物体"""
        return object_id in self.inventory
    
    def move_to_room(self, room_id: str) -> None:
        """移动到指定房间"""
        self.current_room = room_id
    
    def is_inventory_full(self) -> bool:
        """检查背包是否已满"""
        return len(self.inventory) >= self.max_inventory
