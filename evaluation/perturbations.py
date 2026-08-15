from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from pipeline.utils.state_manager import SceneGraphStateManager


@dataclass(frozen=True)
class PerturbationRecord:
    perturbation_type: str
    step_index: int
    message: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RolloutPerturbation:
    def apply_before_planning(
        self, step_index: int, manager: SceneGraphStateManager
    ) -> PerturbationRecord | None:
        return None

    def intercept_execution(
        self, step_index: int, action: str
    ) -> PerturbationRecord | None:
        return None


def _commit_state_perturbation(
    manager: SceneGraphStateManager,
    state: dict[str, Any],
    record: PerturbationRecord,
) -> None:
    validation = manager.validate_state(state)
    if not validation["valid"]:
        raise ValueError(
            f"Perturbation {record.perturbation_type!r} produced an invalid state: "
            f"{validation['errors']}"
        )
    metadata = state.setdefault("state_metadata", {})
    metadata["version"] = len(manager.state_history) + 1
    metadata["last_action"] = f"perturbation:{record.perturbation_type}"
    manager.current_state = state
    manager.state_history.append(deepcopy(state))
    manager.execution_log.append(
        {
            "action": metadata["last_action"],
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "state_version": len(manager.state_history),
            "details": record.as_dict(),
        }
    )


@dataclass
class MoveObjectPerturbation(RolloutPerturbation):
    trigger_step: int
    object_id: str
    source_room: str
    target_room: str
    applied: bool = False

    def apply_before_planning(
        self, step_index: int, manager: SceneGraphStateManager
    ) -> PerturbationRecord | None:
        if self.applied or step_index != self.trigger_step:
            return None
        state = deepcopy(manager.current_state)
        rooms = state.get("rooms", {})
        if self.source_room not in rooms or self.target_room not in rooms:
            raise ValueError("Move-object perturbation references an unknown room")
        source_objects = rooms[self.source_room].get("small_objects", {})
        if self.object_id not in source_objects:
            raise ValueError(
                f"Object {self.object_id!r} is not in source room {self.source_room!r}"
            )
        object_info = source_objects.pop(self.object_id)
        rooms[self.target_room].setdefault("small_objects", {})[self.object_id] = object_info
        record = PerturbationRecord(
            "move_object",
            step_index,
            f"Moved {self.object_id} from {self.source_room} to {self.target_room}",
            {
                "object_id": self.object_id,
                "source_room": self.source_room,
                "target_room": self.target_room,
            },
        )
        _commit_state_perturbation(manager, state, record)
        self.applied = True
        return record


@dataclass
class BlockEdgePerturbation(RolloutPerturbation):
    trigger_step: int
    room_a: str
    room_b: str
    applied: bool = False

    def apply_before_planning(
        self, step_index: int, manager: SceneGraphStateManager
    ) -> PerturbationRecord | None:
        if self.applied or step_index != self.trigger_step:
            return None
        state = deepcopy(manager.current_state)
        rooms = state.get("rooms", {})
        if self.room_a not in rooms or self.room_b not in rooms:
            raise ValueError("Block-edge perturbation references an unknown room")
        rooms[self.room_a]["neighbor"] = [
            room for room in rooms[self.room_a].get("neighbor", []) if room != self.room_b
        ]
        rooms[self.room_b]["neighbor"] = [
            room for room in rooms[self.room_b].get("neighbor", []) if room != self.room_a
        ]
        record = PerturbationRecord(
            "block_edge",
            step_index,
            f"Blocked the edge between {self.room_a} and {self.room_b}",
            {"room_a": self.room_a, "room_b": self.room_b},
        )
        _commit_state_perturbation(manager, state, record)
        self.applied = True
        return record


@dataclass
class FailActionOncePerturbation(RolloutPerturbation):
    action: str
    trigger_step: int | None = None
    reason: str = "Simulated temporary skill failure"
    applied: bool = False

    def intercept_execution(
        self, step_index: int, action: str
    ) -> PerturbationRecord | None:
        if self.applied or action != self.action:
            return None
        if self.trigger_step is not None and step_index != self.trigger_step:
            return None
        self.applied = True
        return PerturbationRecord(
            "fail_action_once",
            step_index,
            self.reason,
            {"action": action, "retryable_same_action": True},
        )


@dataclass
class PerturbationSchedule:
    perturbations: list[RolloutPerturbation]

    def apply_before_planning(
        self, step_index: int, manager: SceneGraphStateManager
    ) -> list[PerturbationRecord]:
        records = []
        for perturbation in self.perturbations:
            record = perturbation.apply_before_planning(step_index, manager)
            if record is not None:
                records.append(record)
        return records

    def intercept_execution(
        self, step_index: int, action: str
    ) -> PerturbationRecord | None:
        for perturbation in self.perturbations:
            record = perturbation.intercept_execution(step_index, action)
            if record is not None:
                return record
        return None
