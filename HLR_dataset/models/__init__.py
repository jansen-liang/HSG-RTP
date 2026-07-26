"""
Scene Graph Models
提供面向对象的场景图节点和边的抽象
"""

from .nodes import (
    BaseNode,
    Object,
    Room,
    Floor,
    MobileTool,
    Agent,
    NodeType
)

from .edges import (
    BaseEdge,
    ObjectEdge,
    ObjectRoomEdge,
    RoomEdge,
    RoomFloorEdge,
    EdgeType,
    SpatialRelation,
    create_edge
)

__all__ = [
    # Nodes
    'BaseNode',
    'Object',
    'Room',
    'Floor',
    'MobileTool',
    'Agent',
    'NodeType',
    # Edges
    'BaseEdge',
    'ObjectEdge',
    'ObjectRoomEdge',
    'RoomEdge',
    'RoomFloorEdge',
    'EdgeType',
    'SpatialRelation',
    'create_edge',
    # State Changes
    'StateTransition',
    'ActionTemplate',
    'StateChangeRegistry',
    'ActionType',
    'get_state_registry',
    'register_custom_action',
    'register_custom_transition',
    # Logic Rules
    'LogicRule',
    'PrerequisiteRule',
    'SequenceRule',
    'ConstraintRule',
    'MutualExclusionRule',
    'LogicRuleRegistry',
    'Condition',
    'ActionStep',
    'RuleType',
    'get_rule_registry',
    'register_custom_rule',
]
