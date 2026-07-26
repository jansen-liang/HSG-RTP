from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EditableComponent:
    object_id: str
    name: str
    category: str = "component"
    template_name: str = "custom"
    description: str = ""
    object_group: str = "small_objects"
    static_attributes: dict = field(default_factory=dict)
    state_variables: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)


@dataclass
class EditableObject:
    object_id: str
    name: str
    category: str = "furniture"
    object_form: str = "simple object"
    template_name: str = "custom"
    description: str = ""
    subtype: str = ""
    object_group: str = "small_objects"
    static_attributes: dict = field(default_factory=dict)
    state_variables: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    relations: dict = field(default_factory=dict)
    components: list[EditableComponent] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        return self.object_form == "composite object"


@dataclass
class EditableRoom:
    room_id: str
    floor: str = ""
    neighbor: list[str] = field(default_factory=list)
    objects: list[EditableObject] = field(default_factory=list)


@dataclass
class EditableMacroZone:
    zone_id: str
    category: str = "custom"
    has_elevator_access: bool = True
    rooms: list[str] = field(default_factory=list)


@dataclass
class EditableScene:
    source_scene: str = "custom_scene"
    name: str = "custom_scene"
    zone_count: int = 1
    elevator_enabled: bool = False
    macro_zones: list[EditableMacroZone] = field(default_factory=list)
    rooms: list[EditableRoom] = field(default_factory=list)

    def ensure_macro_zones(self, zone_count: int) -> None:
        zone_count = max(1, zone_count)
        zones = list(self.macro_zones[:zone_count])
        while len(zones) < zone_count:
            index = len(zones) + 1
            zones.append(
                EditableMacroZone(
                    zone_id=f"zone_{index}",
                    category="public" if index == 1 else "custom",
                    has_elevator_access=True,
                )
            )
        self.macro_zones = zones
        self.zone_count = zone_count
        valid_zone_ids = {zone.zone_id for zone in self.macro_zones}
        for room in self.rooms:
            if room.floor not in valid_zone_ids and self.macro_zones:
                room.floor = self.macro_zones[0].zone_id
        self._sync_zone_room_names()

    def _sync_zone_room_names(self) -> None:
        for zone in self.macro_zones:
            zone.rooms = []
        zone_map = {zone.zone_id: zone for zone in self.macro_zones}
        for room in self.rooms:
            zone = zone_map.get(room.floor)
            if zone and room.room_id not in zone.rooms:
                zone.rooms.append(room.room_id)

    def get_zone(self, zone_id: str) -> EditableMacroZone | None:
        return next((zone for zone in self.macro_zones if zone.zone_id == zone_id), None)

    def get_room(self, room_id: str) -> EditableRoom | None:
        return next((room for room in self.rooms if room.room_id == room_id), None)

    def find_object(self, room_id: str, object_id: str) -> EditableObject | None:
        room = self.get_room(room_id)
        if room is None:
            return None
        return next((obj for obj in room.objects if obj.object_id == object_id), None)

    def add_room(self, room: EditableRoom) -> None:
        self.rooms.append(room)
        self._sync_zone_room_names()

    def remove_room(self, room_id: str) -> None:
        self.rooms = [room for room in self.rooms if room.room_id != room_id]
        for room in self.rooms:
            room.neighbor = [neighbor for neighbor in room.neighbor if neighbor != room_id]
        self._sync_zone_room_names()
