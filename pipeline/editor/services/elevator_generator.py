from __future__ import annotations

import copy


def _floor_suffix(index: int) -> str:
    return f"{index}f"


def add_elevator_structure(scene_dict: dict, enabled_floor_zone_ids: list[str]) -> dict:
    scene = copy.deepcopy(scene_dict)
    zone_to_rooms = scene.setdefault("macro_zones", {})
    rooms = scene.setdefault("rooms", {})
    enabled_floor_zone_ids = [zone_id for zone_id in enabled_floor_zone_ids if zone_id in zone_to_rooms]
    if len(enabled_floor_zone_ids) < 2:
        return scene

    cabin_name = "elevator_cabin"
    cabin_room = rooms.setdefault(
        cabin_name,
        {
            "floor": "elevator",
            "neighbor": [],
            "large_objects": {},
            "small_objects": {},
        },
    )

    for floor_index, zone_id in enumerate(enabled_floor_zone_ids, start=1):
        hall_name = f"elevator_{_floor_suffix(floor_index)}"
        zone_rooms = zone_to_rooms[zone_id].setdefault("rooms", [])
        if hall_name not in zone_rooms:
            zone_rooms.append(hall_name)
        rooms.setdefault(
            hall_name,
            {
                "floor": zone_id,
                "neighbor": [],
                "large_objects": {
                    f"panel_{_floor_suffix(floor_index)}": {
                        "type": "object",
                        "subtype": "appliance",
                        "static_attributes": {"is_container": False},
                        "capabilities": {},
                        "state_variables": {},
                        "relations": {},
                        "description": f"Elevator hall panel on {zone_id}.",
                    }
                },
                "small_objects": {},
            },
        )
        hall_room = rooms[hall_name]
        panel_name = f"panel_{_floor_suffix(floor_index)}"
        hall_room.setdefault("large_objects", {}).setdefault(
            panel_name,
            {
                "type": "object",
                "subtype": "appliance",
                "static_attributes": {"is_container": False},
                "capabilities": {},
                "state_variables": {},
                "relations": {},
                "description": f"Elevator hall panel on {zone_id}.",
            },
        )

        connect_target = _preferred_hall_neighbor(zone_rooms, hall_name)
        if connect_target and connect_target not in hall_room["neighbor"]:
            hall_room["neighbor"].append(connect_target)
        if connect_target:
            neighbor_room = rooms.get(connect_target)
            if neighbor_room is not None and hall_name not in neighbor_room.setdefault("neighbor", []):
                neighbor_room["neighbor"].append(hall_name)

        if cabin_name not in hall_room["neighbor"]:
            hall_room["neighbor"].append(cabin_name)
        if hall_name not in cabin_room["neighbor"]:
            cabin_room["neighbor"].append(hall_name)

        if floor_index > 1:
            up_name = "elevator_call_up"
            hall_room["small_objects"].setdefault(
                up_name,
                _call_button(panel_name, "Call elevator up button - press when outside elevator"),
            )
        if floor_index < len(enabled_floor_zone_ids):
            down_name = "elevator_call_down"
            hall_room["small_objects"].setdefault(
                down_name,
                _call_button(panel_name, "Call elevator down button - press when outside elevator"),
            )

        cabin_room.setdefault("small_objects", {}).setdefault(
            f"elevator_button_{floor_index}",
            {
                "type": "object",
                "subtype": "button",
                "static_attributes": {
                    "movable": False,
                    "fragile": False,
                    "is_container": False,
                    "is_support": False,
                },
                "capabilities": {"pressable": True},
                "state_variables": {"is_pressed": False},
                "relations": {"inside": cabin_name},
                "description": f"Go to floor {floor_index}",
            },
        )

    return scene


def _preferred_hall_neighbor(zone_rooms: list[str], hall_name: str) -> str | None:
    for room_name in zone_rooms:
        if room_name == hall_name:
            continue
        if room_name.startswith("lobby") or room_name.startswith("hallway"):
            return room_name
    for room_name in zone_rooms:
        if room_name != hall_name:
            return room_name
    return None


def _call_button(panel_name: str, description: str) -> dict:
    return {
        "type": "object",
        "subtype": "button",
        "static_attributes": {
            "movable": False,
            "fragile": False,
            "is_container": False,
            "is_support": False,
        },
        "capabilities": {"pressable": True},
        "state_variables": {"is_pressed": False},
        "relations": {"on": panel_name},
        "description": description,
    }
