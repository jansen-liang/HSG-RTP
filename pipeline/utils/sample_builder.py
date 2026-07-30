"""
优化的样本构建器 - 统一状态管理和场景图更新

核心改进：
1. 统一状态管理：只用一套状态更新机制
2. 实时场景图：每个样本使用当前最新的状态生成场景图
3. 精确视图：global样本用全局视图，local样本用实时局部视图
4. 状态同步：确保训练样本中的场景图与实际执行状态一致
"""

from copy import deepcopy
from typing import List, Dict, Any, Tuple
from .graph_utils import get_global_view, get_local_view
from .action_planner import generate_global_plan
from .state_manager import SceneGraphStateManager


def build_optimized_streaming_samples(
    instruction: str, 
    subtasks: list, 
    initial_scene: dict
) -> Tuple[List[Dict], Dict]:
    """
    优化的流式样本生成 - 实时状态同步
    
    关键改进：
    - 每个样本的 scene_graph 是执行 target 动作 *之前* 的状态
    - completed 不包含当前 target 动作
    - pending 包含从当前 global_plan 步骤开始的剩余计划
    """
    state_manager = SceneGraphStateManager()
    state_manager.load_initial_state(initial_scene)
    
    samples = []
    executed = []  # 已执行的动作（在生成样本之后才添加）
    current_global_step = 0  # 下一个要完成的 global 步骤索引
    
    # Step 1: 生成全局计划
    global_plan, boundaries = generate_global_plan(
        subtasks,
        initial_scene["rooms"],
        "general",
        initial_room=initial_scene["agent"]["position"],
    )
    
    # Step 2: 初始全局样本（无 completed/pending）
    samples.append({
        "mode": "global",
        "context": f"instruction: {instruction}",
        "target": global_plan,
        "completed": [],
        "pending": [],
        "scene_graph": get_global_view(state_manager.current_state)
    })
    
    # Step 3: 为每个 subtask 生成 local 样本（在执行前）
    for step, action in enumerate(subtasks):
        current_state_before = state_manager.current_state.copy()  # 执行前状态
        current_pos_before = current_state_before["agent"]["position"]
        
        # 计算 pending：基于已执行动作数
        while (current_global_step < len(boundaries) and 
               step >= boundaries[current_global_step]):
            current_global_step += 1
        
        pending_plan = global_plan[current_global_step:]
        
        # 生成 local 样本：target 是即将执行的动作
        samples.append({
            "mode": "local",
            "context": f"instruction: {instruction}",
            "target": action,
            "completed": executed.copy(),
            "pending": pending_plan,
            "scene_graph": get_local_view(current_state_before, current_pos_before)
        })
        
        # 执行动作（更新状态）
        success, updated_state, error = state_manager.execute_action(action)
        if not success:
            print(f"⚠️ 动作执行失败: {action} - {error}")
            # 可选择中断或继续，这里继续
        else:
            executed.append(action)  # 成功后才加入 executed
    
    # Step 4: 最终全局完成样本
    final_state = state_manager.current_state
    samples.append({
        "mode": "global",
        "context": f"instruction: {instruction}",
        "target": "finish",
        "completed": executed.copy(),
        "pending": [],
        "scene_graph": get_global_view(final_state)
    })
    
    execution_summary = {
        "global_plan": global_plan,  
        "subtasks": subtasks,          
        "total_steps": len(subtasks),
        "successful_actions": len(executed),
        "final_state": final_state
    }
    
    return samples, execution_summary


def build_state_aware_samples(
    instruction: str,
    subtasks: list, 
    initial_scene: dict,
    sample_mode: str = "streaming"  # "streaming", "step_by_step", "hierarchical"
) -> List[Dict]:
    """
    状态感知的样本生成 - 统一接口
    
    核心特点：
    - 所有样本都基于实际执行状态生成场景图
    - 支持多种样本模式但保持状态一致性
    - 实时状态验证和错误处理
    
    Args:
        instruction: 指令
        subtasks: 动作序列
        initial_scene: 初始场景
        sample_mode: 样本生成模式
        
    Returns:
        状态同步的训练样本列表
    """
    if sample_mode == "streaming":
        samples, _ = build_optimized_streaming_samples(instruction, subtasks, initial_scene)
        return samples
    elif sample_mode == "step_by_step":
        return build_step_by_step_with_state_sync(instruction, subtasks, initial_scene)
    elif sample_mode == "hierarchical":
        return build_hierarchical_with_state_sync(instruction, subtasks, initial_scene)
    else:
        raise ValueError(f"Unsupported sample mode: {sample_mode}")


def build_step_by_step_with_state_sync(
    instruction: str, 
    subtasks: list, 
    initial_scene: dict
) -> List[Dict]:
    """逐步样本 + 状态同步"""
    
    state_manager = SceneGraphStateManager()
    state_manager.load_initial_state(initial_scene)
    
    samples = []
    executed = []
    
    for step, action in enumerate(subtasks):
        # 执行前状态
        current_state = state_manager.current_state
        
        # 生成样本（基于执行前状态）
        context = f"instruction: {instruction}\\nlocation: {current_state['agent']['position']}\\nexecuted: {executed}"
        
        if action.startswith("goto(") and action[5:-1] in initial_scene["rooms"]:
            scene_graph = get_global_view(current_state)
            mode = "global"
        else:
            scene_graph = get_local_view(current_state, current_state["agent"]["position"])
            mode = "local"
        
        samples.append({
            "mode": mode,
            "context": context,
            "target": action,
            "scene_graph": scene_graph,  # 🎯 基于当前真实状态
            "metadata": {
                "step": step + 1,
                "total_steps": len(subtasks),
                "state_version": current_state.get("state_metadata", {}).get("version", 0)
            }
        })
        
        # 执行动作
        success, _, error = state_manager.execute_action(action)
        if success:
            executed.append(action)
        else:
            print(f"执行失败: {action} - {error}")
    
    return samples


def build_hierarchical_with_state_sync(
    instruction: str, 
    subtasks: list, 
    initial_scene: dict
) -> Dict[str, List[Dict]]:
    """分层样本 + 状态同步"""
    
    state_manager = SceneGraphStateManager()
    state_manager.load_initial_state(initial_scene)
    
    # 分离全局和局部动作
    global_actions = []
    local_action_groups = {}
    current_room = initial_scene["agent"]["position"]
    
    for action in subtasks:
        if action.startswith("goto(") and action[5:-1] in initial_scene["rooms"]:
            target_room = action[5:-1]
            global_actions.append(action)
            current_room = target_room
            local_action_groups[current_room] = []
        else:
            local_action_groups.setdefault(current_room, []).append(action)
    
    # 构建全局样本（基于初始状态）
    global_samples = []
    if global_actions:
        initial_state = state_manager.current_state
        global_samples.append({
            "mode": "global",
            "context": f"instruction: {instruction}\\nlocation: {initial_state['agent']['position']}\\nmode: planning",
            "target": " -> ".join(global_actions),
            "scene_graph": get_global_view(initial_state),  # 🎯 基于初始状态
            "metadata": {"type": "global_planning"}
        })
    
    # 构建局部样本（模拟执行到各房间的状态）
    local_samples = []
    executed_global = []
    
    for room, actions in local_action_groups.items():
        if not actions:
            continue
        
        # 模拟执行到当前房间
        if room != initial_scene["agent"]["position"]:
            state_manager.execute_action(f"goto({room})")
            executed_global.append(f"goto({room})")
        
        current_room_state = state_manager.current_state
        executed_local = []
        
        for action in actions:
            context = f"instruction: {instruction}\\nlocation: {room}\\nexecuted: {executed_global + executed_local}"
            
            local_samples.append({
                "mode": "local",
                "context": context,
                "target": action,
                "scene_graph": get_local_view(current_room_state, room),  # 🎯 基于实际房间状态
                "metadata": {
                    "room": room,
                    "local_step": len(executed_local) + 1
                }
            })
            
            # 执行局部动作
            state_manager.execute_action(action)
            executed_local.append(action)
            current_room_state = state_manager.current_state  # 更新房间状态
    
    return {
        "global_samples": global_samples,
        "local_samples": local_samples,
        "hierarchical": True
    }


# 向后兼容性
def build_streaming_samples(instruction: str, subtasks: list, initial_scene: dict) -> tuple:
    """向后兼容的流式样本生成"""
    samples, summary = build_optimized_streaming_samples(instruction, subtasks, initial_scene)
    return samples, {}  # 保持API兼容性


def validate_sample_quality(sample: dict) -> bool:
    """
    验证样本质量
    
    Returns:
        样本是否有效
    """
    required_fields = ["mode", "context", "target", "scene_graph"]
    
    # 检查必需字段
    for field in required_fields:
        if field not in sample:
            return False
    
    # 检查 context 有效性（始终是字符串）
    if not isinstance(sample["context"], str) or not sample["context"].strip():
        return False
    
    # 检查 target 有效性（global 是列表，local/completion 是字符串）
    target = sample["target"]
    mode = sample["mode"]
    
    if mode == "global":
        # global 的 target 必须是非空列表
        if not isinstance(target, list) or len(target) == 0:
            return False
        # 检查列表中的每个元素是否为非空字符串
        for item in target:
            if not isinstance(item, str) or not item.strip():
                return False
    else:
        # local/completion 的 target 必须是非空字符串
        if not isinstance(target, str) or not target.strip():
            return False
    
    # 检查场景图完整性
    scene_graph = sample["scene_graph"]
    if not isinstance(scene_graph, dict):
        return False
    
    return True
