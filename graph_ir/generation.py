from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .ids import slugify


@dataclass(frozen=True)
class GenerationConstraints:
    floors: tuple[int, int] = (1, 3)
    rooms: tuple[int, int] = (4, 24)
    cross_floor_edge_density: float = 0.15
    room_neighbor_density: float = 0.35
    object_density: tuple[int, int] = (2, 10)
    container_depth: tuple[int, int] = (1, 2)
    min_plan_length: int = 4
    max_plan_length: int = 32
    ensure_connected: bool = True
    ensure_task_reachable: bool = True
    seed: int = 0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.floors[0] < 1 or self.floors[0] > self.floors[1]:
            errors.append("floors must be a valid inclusive range with min >= 1.")
        if self.rooms[0] < 1 or self.rooms[0] > self.rooms[1]:
            errors.append("rooms must be a valid inclusive range with min >= 1.")
        if not 0.0 <= self.cross_floor_edge_density <= 1.0:
            errors.append("cross_floor_edge_density must be in [0, 1].")
        if not 0.0 <= self.room_neighbor_density <= 1.0:
            errors.append("room_neighbor_density must be in [0, 1].")
        if self.object_density[0] < 0 or self.object_density[0] > self.object_density[1]:
            errors.append("object_density must be a valid inclusive range with min >= 0.")
        if self.container_depth[0] < 0 or self.container_depth[0] > self.container_depth[1]:
            errors.append("container_depth must be a valid inclusive range.")
        if self.min_plan_length < 1 or self.min_plan_length > self.max_plan_length:
            errors.append("plan length bounds are invalid.")
        return errors


class StableNameAllocator:
    def __init__(self) -> None:
        self._counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    def allocate(self, kind: str, subtype: str, scope: str | None = None) -> str:
        key = (scope or "", f"{kind}:{subtype}")
        index = self._counters[key]
        self._counters[key] += 1
        base = slugify(subtype or kind)
        prefix = slugify(scope) if scope else slugify(kind)
        return f"{prefix}_{base}_{index:03d}"


def sample_generation_plan(constraints: GenerationConstraints) -> dict[str, int | float]:
    rng = random.Random(constraints.seed)
    errors = constraints.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "floors": rng.randint(*constraints.floors),
        "rooms": rng.randint(*constraints.rooms),
        "cross_floor_edge_density": constraints.cross_floor_edge_density,
        "room_neighbor_density": constraints.room_neighbor_density,
        "objects_per_room": rng.randint(*constraints.object_density),
        "container_depth": rng.randint(*constraints.container_depth),
        "target_plan_length": rng.randint(constraints.min_plan_length, constraints.max_plan_length),
    }
