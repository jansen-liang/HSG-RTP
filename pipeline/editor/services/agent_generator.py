from __future__ import annotations


def generate_agent(scene_dict: dict) -> dict:
    room_names = list(scene_dict.get("rooms", {}).keys())
    preferred_prefixes = ("lobby", "hallway")

    chosen = None
    for prefix in preferred_prefixes:
        chosen = next((name for name in room_names if name.startswith(prefix)), None)
        if chosen:
            break
    if not chosen:
        chosen = next(
            (name for name, room in scene_dict.get("rooms", {}).items() if room.get("floor") != "elevator"),
            None,
        )
    if not chosen and room_names:
        chosen = room_names[0]

    return {
        "position": chosen or "",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot",
    }

