from __future__ import annotations


def validate_scene_dict(scene_dict: dict) -> list[str]:
    errors: list[str] = []
    macro_zones = scene_dict.get("macro_zones", {})
    rooms = scene_dict.get("rooms", {})
    room_ids = set(rooms.keys())
    object_index: dict[str, tuple[str, dict]] = {}

    if not scene_dict.get("name"):
        errors.append("Scene name is required.")
    if not macro_zones:
        errors.append("At least one macro zone is required.")
    if not rooms:
        errors.append("At least one room is required.")

    for zone_id, zone_data in macro_zones.items():
        for room_id in zone_data.get("rooms", []):
            if room_id not in rooms:
                errors.append(f"Macro zone '{zone_id}' references missing room '{room_id}'.")

    for room_id, room_data in rooms.items():
        floor = room_data.get("floor")
        if floor != "elevator" and floor not in macro_zones:
            errors.append(f"Room '{room_id}' has unknown floor '{floor}'.")
        for neighbor in room_data.get("neighbor", []):
            if neighbor not in rooms:
                errors.append(f"Room '{room_id}' has missing neighbor '{neighbor}'.")
            elif room_id not in rooms[neighbor].get("neighbor", []):
                errors.append(f"Neighbor link '{room_id}' -> '{neighbor}' is not bidirectional.")

        for bucket_name in ("large_objects", "small_objects"):
            for object_id, object_data in room_data.get(bucket_name, {}).items():
                _index_object(object_id, room_id, object_data, object_index, errors)

    for object_id, (room_id, object_data) in list(object_index.items()):
        _validate_object(object_id, room_id, object_data, object_index, room_ids, errors)

    agent = scene_dict.get("agent", {})
    agent_position = agent.get("position")
    if not agent_position or agent_position not in rooms:
        errors.append("Agent position must point to an existing room.")

    return errors


def check_scene_dict(scene_dict: dict) -> list[str]:
    warnings: list[str] = []
    rooms = scene_dict.get("rooms", {})
    room_ids = set(rooms.keys())
    object_index: dict[str, tuple[str, dict]] = {}

    for room_id, room_data in rooms.items():
        for bucket_name in ("large_objects", "small_objects"):
            for object_id, object_data in room_data.get(bucket_name, {}).items():
                if object_id not in object_index:
                    object_index[object_id] = (room_id, object_data)
                for component_id, component_data in object_data.get("components", {}).items():
                    object_index.setdefault(component_id, (room_id, component_data))

    for object_id, (room_id, object_data) in object_index.items():
        _check_object(object_id, room_id, object_data, object_index, room_ids, warnings)

    return warnings


def _index_object(
    object_id: str,
    room_id: str,
    object_data: dict,
    object_index: dict[str, tuple[str, dict]],
    errors: list[str],
) -> None:
    if object_id in object_index:
        errors.append(f"Duplicate object_id '{object_id}'.")
        return
    object_index[object_id] = (room_id, object_data)
    for component_id, component_data in object_data.get("components", {}).items():
        _index_object(component_id, room_id, component_data, object_index, errors)


def _validate_object(
    object_id: str,
    room_id: str,
    object_data: dict,
    object_index: dict[str, tuple[str, dict]],
    room_ids: set[str],
    errors: list[str],
) -> None:
    subtype = object_data.get("subtype", "")
    static_attributes = object_data.get("static_attributes", {})
    state_variables = object_data.get("state_variables", {})
    relations = object_data.get("relations", {})

    if subtype != "food" and "spoilage" in state_variables:
        errors.append(f"Object '{object_id}' in room '{room_id}' uses spoilage but is not food.")
    if subtype != "button" and "is_pressed" in state_variables:
        errors.append(f"Object '{object_id}' in room '{room_id}' uses is_pressed but is not a button.")
    if subtype == "container" and static_attributes.get("is_container") is not True:
        errors.append(f"Container '{object_id}' must set static_attributes.is_container=true.")

    for relation_name, target in relations.items():
        if relation_name in {"in_room", "inside", "outside"} and target in room_ids:
            continue
        if relation_name in {"part_of", "controls"} and target and target not in object_index:
            errors.append(f"Object '{object_id}' in room '{room_id}' has missing {relation_name} target '{target}'.")
        elif relation_name not in {"in_room"} and target and target not in object_index:
            errors.append(f"Object '{object_id}' in room '{room_id}' points to missing relation target '{target}'.")

    components = object_data.get("components", {})
    for component_id, component_data in components.items():
        component_relations = component_data.get("relations", {})
        if "part_of" not in component_relations:
            errors.append(f"Component '{component_id}' should define part_of.")
        elif component_relations.get("part_of") != object_id:
            errors.append(f"Component '{component_id}' must part_of '{object_id}'.")
        _validate_object(component_id, room_id, component_data, object_index, room_ids, errors)

    if object_data.get("template") == "washing_machine_template":
        required = {"washer_door", "detergent_drawer", "washer_drum", "start_button"}
        actual = set(components.keys())
        missing = sorted(required - actual)
        if missing:
            errors.append(
                f"Washing machine '{object_id}' is missing required components: {', '.join(missing)}."
            )


def _check_object(
    object_id: str,
    room_id: str,
    object_data: dict,
    object_index: dict[str, tuple[str, dict]],
    room_ids: set[str],
    warnings: list[str],
) -> None:
    subtype = object_data.get("subtype", "")
    static_attributes = object_data.get("static_attributes", {})
    state_variables = object_data.get("state_variables", {})
    capabilities = object_data.get("capabilities", {})
    relations = object_data.get("relations", {})
    components = object_data.get("components", {})

    if relations.get("in_room") and relations.get("in_room") != room_id:
        warnings.append(
            f"Object '{object_id}' is stored in room '{room_id}' but relations.in_room points to '{relations.get('in_room')}'."
        )
    if relations.get("inside") and relations.get("on"):
        warnings.append(f"Object '{object_id}' sets both inside and on. Please confirm only one spatial relation is intended.")
    if subtype == "button" and not relations.get("controls"):
        warnings.append(f"Button '{object_id}' usually should define controls.")
    if object_data.get("object_form") == "composite object" and not components:
        warnings.append(f"Composite object '{object_id}' has no components configured.")
    if capabilities.get("openable") and "is_open" not in state_variables:
        warnings.append(f"Openable object '{object_id}' is missing state_variables.is_open.")
    if capabilities.get("receptacle") and static_attributes.get("is_container") is False:
        warnings.append(f"Object '{object_id}' is receptacle-capable but static_attributes.is_container is false.")
    if capabilities.get("support_surface") and static_attributes.get("is_support") is False:
        warnings.append(f"Object '{object_id}' is support_surface-capable but static_attributes.is_support is false.")

    inside_target = relations.get("inside")
    if inside_target and inside_target not in room_ids and not _is_container_like(inside_target, object_index):
        warnings.append(f"Object '{object_id}' points inside '{inside_target}', but that target does not look like a container.")

    on_target = relations.get("on")
    if on_target and on_target not in room_ids and not _is_support_like(on_target, object_index):
        warnings.append(f"Object '{object_id}' points on '{on_target}', but that target does not look like a support surface.")

    for component_id, component_data in components.items():
        component_relations = component_data.get("relations", {})
        if component_relations.get("part_of") != object_id:
            warnings.append(f"Component '{component_id}' should normally part_of '{object_id}'.")
        _check_object(component_id, room_id, component_data, object_index, room_ids, warnings)


def _is_container_like(target_id: str, object_index: dict[str, tuple[str, dict]]) -> bool:
    target = object_index.get(target_id)
    if not target:
        return False
    payload = target[1]
    static_attributes = payload.get("static_attributes", {})
    capabilities = payload.get("capabilities", {})
    return static_attributes.get("is_container") is True or capabilities.get("receptacle") is True


def _is_support_like(target_id: str, object_index: dict[str, tuple[str, dict]]) -> bool:
    target = object_index.get(target_id)
    if not target:
        return False
    payload = target[1]
    static_attributes = payload.get("static_attributes", {})
    capabilities = payload.get("capabilities", {})
    return static_attributes.get("is_support") is True or capabilities.get("support_surface") is True
