"""
增强的场景图状态管理器
支持动态更新、持久化、验证和回滚
"""

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class SceneGraphStateManager:
    """
    场景图状态管理器
    
    功能：
    - 动态更新场景图状态
    - 状态持久化和加载
    - 状态验证和一致性检查
    - 状态历史记录和回滚
    - 可视化状态变化
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        default_workspace = Path(__file__).resolve().parents[1] / "output" / "states"
        self.workspace_dir = Path(workspace_dir) if workspace_dir else default_workspace
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态历史
        self.state_history: List[Dict] = []
        self.current_state: Optional[Dict] = None
        self.execution_log: List[Dict] = []
        
        # 验证规则
        self.validation_rules = self._init_validation_rules()
    
    def _init_validation_rules(self) -> Dict[str, callable]:
        """初始化状态验证规则"""
        return {
            "agent_exists": self._validate_agent_exists,
            "agent_in_valid_room": self._validate_agent_in_valid_room,
            "objects_have_valid_relations": self._validate_object_relations,
            "no_duplicate_objects": self._validate_no_duplicate_objects,
            "container_consistency": self._validate_container_consistency
        }
    
    def load_initial_state(self, scene_data: Dict) -> Dict:
        """
        加载初始场景状态
        
        Args:
            scene_data: 场景图数据
            
        Returns:
            初始化的状态
        """
        self.current_state = deepcopy(scene_data)
        self.state_history = [deepcopy(scene_data)]
        self.execution_log = []
        
        # 添加状态管理字段
        if "state_metadata" not in self.current_state:
            self.current_state["state_metadata"] = {
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "last_action": None,
                "action_count": 0
            }
        
        print(f"✅ 加载初始状态: {scene_data.get('name', 'unknown')} 场景")
        return deepcopy(self.current_state)
    
    def execute_action(self, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """
        执行动作并更新状态
        
        Args:
            action: 要执行的动作
            
        Returns:
            (成功标志, 更新后的状态, 错误信息)
        """
        if not self.current_state:
            return False, {}, "No initial state loaded"
        
        # 备份当前状态
        backup_state = deepcopy(self.current_state)
        
        try:
            # 执行动作
            success, new_state, error = self._execute_single_action(self.current_state, action)
            
            if success:
                # 验证新状态
                validation_result = self.validate_state(new_state)
                if not validation_result["valid"]:
                    return False, backup_state, f"State validation failed: {validation_result['errors']}"
                
                # 更新状态
                self.current_state = new_state
                self.state_history.append(deepcopy(new_state))
                
                # 记录执行日志
                self.execution_log.append({
                    "action": action,
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "state_version": len(self.state_history)
                })
                
                # 更新元数据
                self.current_state["state_metadata"]["last_action"] = action
                self.current_state["state_metadata"]["action_count"] += 1
                self.current_state["state_metadata"]["version"] = len(self.state_history)
                
                print(f"✅ 执行成功: {action}")
                return True, deepcopy(self.current_state), None
            else:
                return False, backup_state, error
                
        except Exception as e:
            return False, backup_state, f"Execution error: {str(e)}"
    
    def _execute_single_action(self, state: Dict, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """执行单个动作的具体实现"""
        new_state = deepcopy(state)
        action = action.strip()
        
        if action.startswith("goto("):
            return self._execute_goto(new_state, action)
        elif action.startswith("pick("):
            return self._execute_pick(new_state, action)
        elif action.startswith("place("):
            return self._execute_place(new_state, action)
        elif action.startswith("scan("):
            return self._execute_scan(new_state, action)
        elif action.startswith("press("):
            return self._execute_press(new_state, action)
        elif action.startswith("wait("):
            if "elevator_down_clear" in action:
                pass  
            elif "elevator_up_clear" in action:
                pass
            return True, state, None
        else:
            return False, state, f"Unknown action type: {action}"
    
    def _execute_goto(self, state: Dict, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """执行移动动作"""
        target = action[5:-1]  # 去掉 "goto(" 和 ")"
        
        current_pos = state["agent"]["position"]
        
        # 检查目标房间是否存在
        if target not in state.get("rooms", {}):
            return False, state, f"Target room {target} does not exist"
        
        # 检查是否可达 (简单的邻居检查)
        current_room = state["rooms"][current_pos]
        if target != current_pos and target not in current_room.get("neighbor", []):
            # 允许电梯间的特殊连接
            if not (current_pos.startswith("elevator_") and target.startswith("elevator_")):
                return False, state, f"Cannot reach {target} from {current_pos}"
        
        # 更新位置
        state["agent"]["position"] = target
        return True, state, None
    
    def _execute_pick(self, state: Dict, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """执行拾取动作"""
        obj_id = action[5:-1]
        current_pos = state["agent"]["position"]
        current_room = state["rooms"][current_pos]
        
        # 检查agent是否已持有物品
        if state["agent"].get("state", "").startswith("holding-"):
            return False, state, f"Agent is already holding an object"
        
        # 检查物品是否存在且可拾取
        if obj_id in current_room.get("small_objects", {}):
            obj_info = current_room["small_objects"].pop(obj_id)
            
            # 添加到agent库存
            if "inventory" not in state["agent"]:
                state["agent"]["inventory"] = {}
            state["agent"]["inventory"][obj_id] = obj_info
            state["agent"]["state"] = f"holding-{obj_id}"
            
            return True, state, None
        elif obj_id in current_room.get("large_objects", {}):
            return False, state, f"Cannot pick large object {obj_id}"
        else:
            return False, state, f"Object {obj_id} not found in {current_pos}"
    
    def _execute_place(self, state: Dict, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """执行放置动作"""
        # 解析 place(obj_id, surface_id) 格式
        content = action[6:-1]
        parts = content.split(", ")
        if len(parts) != 2:
            return False, state, f"Invalid place format: {action}"
        
        obj_id, surface_id = parts
        current_pos = state["agent"]["position"]
        
        # 检查agent是否持有该物品
        if ("inventory" not in state["agent"] or 
            obj_id not in state["agent"]["inventory"]):
            return False, state, f"Agent is not holding {obj_id}"
        
        # 移除物品从库存
        obj_info = state["agent"]["inventory"].pop(obj_id)
        state["agent"]["state"] = "hand-free"
        
        # 添加到当前房间
        if "small_objects" not in state["rooms"][current_pos]:
            state["rooms"][current_pos]["small_objects"] = {}
        
        obj_info["relation"] = {"on": surface_id}
        state["rooms"][current_pos]["small_objects"][obj_id] = obj_info
        
        return True, state, None
    
    def _execute_scan(self, state: Dict, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """执行扫描动作"""
        target = action[5:-1]
        
        if "scan_history" not in state["agent"]:
            state["agent"]["scan_history"] = []
        
        state["agent"]["scan_history"].append(target)
        state["agent"]["last_scanned"] = target
        
        return True, state, None
    
    def _execute_press(self, state: Dict, action: str) -> Tuple[bool, Dict, Optional[str]]:
        """执行按压动作"""
        button_id = action[6:-1]
        current_pos = state["agent"]["position"]
        current_room = state["rooms"][current_pos]
        
        if button_id in current_room.get("small_objects", {}):
            button_info = current_room["small_objects"][button_id]
            if isinstance(button_info, dict) and "press" in button_info.get("affordance", []):
                # 更新按钮状态
                button_info["state"] = "pressed"
                
                # 记录按压历史
                if "pressed_buttons" not in state["agent"]:
                    state["agent"]["pressed_buttons"] = []
                state["agent"]["pressed_buttons"].append(button_id)
                
                return True, state, None
            else:
                return False, state, f"Button {button_id} cannot be pressed"
        else:
            return False, state, f"Button {button_id} not found"
    
    def validate_state(self, state: Dict) -> Dict[str, Any]:
        """
        验证状态一致性
        
        Returns:
            验证结果字典
        """
        result = {"valid": True, "errors": [], "warnings": []}
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(state)
                if not rule_result["valid"]:
                    result["valid"] = False
                    result["errors"].extend(rule_result.get("errors", []))
                result["warnings"].extend(rule_result.get("warnings", []))
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Validation rule {rule_name} failed: {str(e)}")
        
        return result
    
    def _validate_agent_exists(self, state: Dict) -> Dict[str, Any]:
        """验证agent存在"""
        if "agent" not in state:
            return {"valid": False, "errors": ["Agent not found in state"]}
        return {"valid": True, "errors": [], "warnings": []}
    
    def _validate_agent_in_valid_room(self, state: Dict) -> Dict[str, Any]:
        """验证agent在有效房间中"""
        agent_pos = state["agent"].get("position")
        if not agent_pos or agent_pos not in state.get("rooms", {}):
            return {"valid": False, "errors": [f"Agent position {agent_pos} is invalid"]}
        return {"valid": True, "errors": [], "warnings": []}
    
    def _validate_object_relations(self, state: Dict) -> Dict[str, Any]:
        """验证物品关系的一致性"""
        errors = []
        warnings = []
        
        for room_id, room_data in state.get("rooms", {}).items():
            for obj_id, obj_data in room_data.get("small_objects", {}).items():
                if isinstance(obj_data, dict) and "relation" in obj_data:
                    relation = obj_data["relation"]
                    for rel_type, rel_target in relation.items():
                        # 检查关系目标是否存在
                        if rel_target not in room_data.get("large_objects", {}):
                            warnings.append(f"Object {obj_id} has relation to non-existent {rel_target}")
        
        return {"valid": True, "errors": errors, "warnings": warnings}
    
    def _validate_no_duplicate_objects(self, state: Dict) -> Dict[str, Any]:
        """验证没有重复物品（允许电梯按钮等系统性重复）"""
        object_locations = {}  # obj_id -> [room_ids]
        errors = []
        warnings = []
        
        # 允许重复的系统对象（使用模式匹配）
        def is_allowed_duplicate(obj_id):
            return (obj_id.startswith("elevator_call_") or 
                   obj_id.startswith("elevator_button_"))
        
        # 检查房间中的物品
        for room_id, room_data in state.get("rooms", {}).items():
            for obj_id in room_data.get("small_objects", {}):
                if obj_id not in object_locations:
                    object_locations[obj_id] = []
                object_locations[obj_id].append(room_id)
        
        # 检查agent库存
        agent_inventory = state.get("agent", {}).get("inventory", {})
        for obj_id in agent_inventory:
            if obj_id in object_locations:
                errors.append(f"Object {obj_id} exists in both room and agent inventory")
            object_locations[obj_id] = ["agent_inventory"]
        
        # 检查重复（除了允许的系统对象）
        for obj_id, locations in object_locations.items():
            if len(locations) > 1 and not is_allowed_duplicate(obj_id):
                errors.append(f"Object {obj_id} found in multiple locations: {locations}")
            elif len(locations) > 1:
                warnings.append(f"System object {obj_id} found in {len(locations)} locations (normal for elevators)")
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def _validate_container_consistency(self, state: Dict) -> Dict[str, Any]:
        """验证容器一致性"""
        warnings = []
        
        # 检查agent持有状态与库存的一致性
        agent_state = state.get("agent", {}).get("state", "")
        inventory = state.get("agent", {}).get("inventory", {})
        
        if agent_state.startswith("holding-"):
            held_obj = agent_state[8:]  # 去掉 "holding-"
            if held_obj not in inventory:
                warnings.append(f"Agent claims to hold {held_obj} but it's not in inventory")
        elif agent_state == "hand-free" and inventory:
            warnings.append("Agent claims to be hand-free but has items in inventory")
        
        return {"valid": True, "errors": [], "warnings": warnings}
    
    def save_state(self, session_id: str, step: int = None) -> str:
        """
        保存当前状态到文件
        
        Returns:
            保存的文件路径
        """
        if not self.current_state:
            raise ValueError("No state to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_suffix = f"_step_{step}" if step is not None else ""
        filename = f"state_{session_id}_{timestamp}{step_suffix}.json"
        filepath = self.workspace_dir / filename
        
        state_to_save = {
            "state": self.current_state,
            "execution_log": self.execution_log,
            "metadata": {
                "session_id": session_id,
                "step": step,
                "saved_at": datetime.now().isoformat(),
                "state_version": len(self.state_history)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"💾 状态已保存: {filepath}")
        return str(filepath)
    
    def load_state(self, filepath: str) -> bool:
        """
        从文件加载状态
        
        Returns:
            加载是否成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_state = data["state"]
            self.execution_log = data.get("execution_log", [])
            
            print(f"📂 状态已加载: {filepath}")
            return True
        except Exception as e:
            print(f"❌ 加载状态失败: {str(e)}")
            return False
    
    def rollback_to_step(self, step: int) -> bool:
        """
        回滚到指定步骤
        
        Args:
            step: 目标步骤 (0为初始状态)
            
        Returns:
            回滚是否成功
        """
        if step < 0 or step >= len(self.state_history):
            return False
        
        self.current_state = deepcopy(self.state_history[step])
        
        # 截断历史和日志
        self.state_history = self.state_history[:step+1]
        self.execution_log = self.execution_log[:step]
        
        print(f"⏪ 已回滚到步骤 {step}")
        return True
    
    def get_state_changes(self, from_step: int = -2, to_step: int = -1) -> Dict:
        """
        获取状态变化差异
        
        Returns:
            状态变化信息
        """
        if len(self.state_history) < 2:
            return {"changes": [], "summary": "No changes available"}
        
        from_state = self.state_history[from_step]
        to_state = self.state_history[to_step]
        
        changes = []
        
        # 比较agent位置
        if from_state["agent"]["position"] != to_state["agent"]["position"]:
            changes.append({
                "type": "agent_move",
                "from": from_state["agent"]["position"],
                "to": to_state["agent"]["position"]
            })
        
        # 比较agent状态
        if from_state["agent"].get("state") != to_state["agent"].get("state"):
            changes.append({
                "type": "agent_state",
                "from": from_state["agent"].get("state"),
                "to": to_state["agent"].get("state")
            })
        
        # 比较物品位置 (简化实现)
        # TODO: 实现更详细的物品变化跟踪
        
        return {
            "changes": changes,
            "summary": f"Found {len(changes)} changes"
        }
    
    def get_execution_summary(self) -> Dict:
        """获取执行摘要"""
        if not self.execution_log:
            return {"total_actions": 0, "success_rate": 0}
        
        total = len(self.execution_log)
        successful = sum(1 for log in self.execution_log if log.get("success", False))
        
        return {
            "total_actions": total,
            "successful_actions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "current_state_version": len(self.state_history),
            "last_action": self.execution_log[-1]["action"] if self.execution_log else None
        }
