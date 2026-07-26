from __future__ import annotations

from copy import deepcopy

from editor.models.editable_scene import EditableComponent, EditableMacroZone, EditableObject, EditableRoom, EditableScene
from editor.schemas.object_type_templates import (
    build_category_defaults,
    get_category_schema,
    get_template_config,
    normalize_object_payload,
)

from .agent_generator import generate_agent
from .elevator_generator import add_elevator_structure


def build_scene_dict(editable_scene: EditableScene) -> dict:
    editable_scene._sync_zone_room_names()
    scene = {
        "name": editable_scene.name.strip() or "custom_scene",
        "macro_zones": {},
        "rooms": {},
    }

    for zone in editable_scene.macro_zones:
        scene["macro_zones"][zone.zone_id] = {"rooms": list(zone.rooms)}

    for room in editable_scene.rooms:
        room_dict = {
            "floor": room.floor,
            "neighbor": list(dict.fromkeys(room.neighbor)),
            "large_objects": {},
            "small_objects": {},
        }
        for obj in room.objects:
            room_dict[obj.object_group][obj.object_id] = build_object_payload(obj)
        scene["rooms"][room.room_id] = room_dict

    enabled_zones = [zone.zone_id for zone in editable_scene.macro_zones if zone.has_elevator_access]
    if editable_scene.elevator_enabled:
        scene = add_elevator_structure(scene, enabled_zones)

    scene["agent"] = generate_agent(scene)
    return scene


def build_object_payload(obj: EditableObject | EditableComponent) -> dict:
    payload = {
        "type": "object",
        "subtype": obj.category,
        "static_attributes": deepcopy(obj.static_attributes),
        "capabilities": deepcopy(obj.capabilities),
        "state_variables": deepcopy(obj.state_variables),
        "relations": deepcopy(obj.relations),
        "description": obj.description,
    }
    if getattr(obj, "template_name", "custom") != "custom":
        payload["template"] = obj.template_name
    if getattr(obj, "subtype", ""):
        payload["template_subtype"] = obj.subtype
    if isinstance(obj, EditableObject):
        payload["object_form"] = obj.object_form
        payload["name"] = obj.name
        if obj.components:
            payload["components"] = {
                component.object_id: build_object_payload(component)
                for component in obj.components
            }
    else:
        payload["name"] = obj.name
    return payload


def create_object_from_template(
    object_id: str,
    name: str,
    category: str,
    object_form: str,
    template_name: str,
    description: str = "",
) -> EditableObject:
    defaults = build_category_defaults(category, object_form)
    obj = EditableObject(
        object_id=object_id,
        name=name,
        category=category,
        object_form=object_form,
        template_name=template_name,
        description=description,
        object_group=get_category_schema(category)["default_object_group"],
        static_attributes=defaults["static_attributes"],
        state_variables=defaults["state_variables"],
        capabilities=defaults["capabilities"],
        relations=defaults["relations"],
        components=[],
    )
    apply_template_to_object(obj, template_name, keep_identity=True)
    if description:
        obj.description = description
    return obj


def apply_template_to_object(obj: EditableObject, template_name: str, keep_identity: bool = False) -> EditableObject:
    template = get_template_config(template_name)
    if template.get("category"):
        obj.category = template["category"]
    if template.get("object_form"):
        obj.object_form = template["object_form"]
    obj.template_name = template_name
    obj.object_group = get_category_schema(obj.category)["default_object_group"]
    defaults = build_category_defaults(obj.category, obj.object_form)
    obj.static_attributes = defaults["static_attributes"]
    obj.state_variables = defaults["state_variables"]
    obj.capabilities = defaults["capabilities"]
    obj.relations = defaults["relations"]

    normalized = normalize_object_payload(
        obj.category,
        {
            "static_attributes": template.get("static_attributes", {}),
            "state_variables": template.get("state_variables", {}),
            "capabilities": template.get("capabilities", {}),
            "relations": template.get("relations", {}),
        },
    )
    obj.static_attributes.update(normalized["static_attributes"])
    obj.state_variables.update(normalized["state_variables"])
    obj.capabilities.update(normalized["capabilities"])
    obj.relations.update(normalized["relations"])
    if template.get("description") and (not keep_identity or not obj.description):
        obj.description = template["description"]
    obj.components = [
        _component_from_template_payload(component_payload, obj.object_id)
        for component_payload in template.get("components", [])
    ]
    for component in obj.components:
        if component.object_id == "start_button" and obj.template_name == "washing_machine_template":
            component.relations["controls"] = obj.object_id
    return obj


def editable_scene_from_scene_dict(scene_name: str, scene_dict: dict) -> EditableScene:
    editable_scene = EditableScene(source_scene=scene_name, name=scene_dict.get("name", scene_name), elevator_enabled=False)
    macro_zones = scene_dict.get("macro_zones", {})
    rooms = scene_dict.get("rooms", {})

    editable_scene.macro_zones = [
        EditableMacroZone(
            zone_id=zone_id,
            category=_infer_zone_category(zone_id),
            has_elevator_access=True,
            rooms=list(zone_data.get("rooms", [])),
        )
        for zone_id, zone_data in macro_zones.items()
    ] or [EditableMacroZone(zone_id="zone_1", category="custom", has_elevator_access=True)]
    editable_scene.zone_count = len(editable_scene.macro_zones)

    for room_id, room_data in rooms.items():
        if _is_generated_elevator_room(room_id, room_data):
            editable_scene.elevator_enabled = True
            continue
        room = EditableRoom(
            room_id=room_id,
            floor=room_data.get("floor") or editable_scene.macro_zones[0].zone_id,
            neighbor=[
                neighbor
                for neighbor in room_data.get("neighbor", [])
                if neighbor in rooms and not _is_generated_elevator_room(neighbor, rooms.get(neighbor, {}))
            ],
            objects=[],
        )
        room.objects.extend(_editable_objects_from_room_bucket(room_data.get("large_objects", {}), "large_objects"))
        room.objects.extend(_editable_objects_from_room_bucket(room_data.get("small_objects", {}), "small_objects"))
        editable_scene.rooms.append(room)

    editable_scene._sync_zone_room_names()
    return editable_scene


def _editable_objects_from_room_bucket(bucket: dict, object_group: str) -> list[EditableObject]:
    objects: list[EditableObject] = []
    for object_id, payload in bucket.items():
        subtype = payload.get("subtype") or payload.get("type") or "furniture"
        category = subtype if subtype in {"food", "furniture", "appliance", "container", "cloth", "tool", "button", "component"} else _legacy_category(payload)
        object_form = payload.get("object_form", "composite object" if payload.get("components") else "simple object")
        obj = EditableObject(
            object_id=object_id,
            name=payload.get("name", object_id),
            category=category,
            object_form=object_form,
            template_name=payload.get("template", "custom"),
            description=payload.get("description", ""),
            subtype=payload.get("template_subtype", ""),
            object_group=object_group,
            static_attributes=deepcopy(payload.get("static_attributes", payload.get("physical_property", {}))),
            state_variables=deepcopy(payload.get("state_variables", payload.get("state", {}))),
            capabilities=deepcopy(payload.get("capabilities", _legacy_capabilities(payload))),
            relations=deepcopy(payload.get("relations", payload.get("relation", {}))),
            components=[],
        )
        for component_id, component_payload in payload.get("components", {}).items():
            component_category = component_payload.get("subtype") or component_payload.get("type") or "component"
            component_category = component_category if component_category in {"food", "furniture", "appliance", "container", "cloth", "tool", "button", "component"} else "component"
            component = EditableComponent(
                object_id=component_id,
                name=component_payload.get("name", component_id),
                category=component_category,
                template_name=component_payload.get("template", "custom"),
                description=component_payload.get("description", ""),
                static_attributes=deepcopy(component_payload.get("static_attributes", component_payload.get("physical_property", {}))),
                state_variables=deepcopy(component_payload.get("state_variables", component_payload.get("state", {}))),
                capabilities=deepcopy(component_payload.get("capabilities", _legacy_capabilities(component_payload))),
                relations=deepcopy(component_payload.get("relations", component_payload.get("relation", {}))),
            )
            if "part_of" not in component.relations:
                component.relations["part_of"] = object_id
            if component.object_id == "start_button":
                component.relations.setdefault("controls", object_id)
            obj.components.append(component)
        objects.append(obj)
    return objects


def _legacy_capabilities(payload: dict) -> dict:
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, dict):
        return deepcopy(capabilities)
    affordance = payload.get("affordance", [])
    if isinstance(affordance, list):
        return {nameable_affordance: True for nameable_affordance in affordance}
    return {}


def _legacy_category(payload: dict) -> str:
    if "spoilage" in payload.get("state", {}):
        return "food"
    if "pressed" in payload.get("state", {}) or "press" in payload.get("affordance", []):
        return "button"
    if payload.get("is_container"):
        return "container"
    return "furniture"


def _infer_zone_category(zone_id: str) -> str:
    for category in ["public", "mixed", "guest", "office", "service", "custom"]:
        if zone_id.endswith(category):
            return category
    return "custom"


def _is_generated_elevator_room(room_id: str, room_data: dict) -> bool:
    return room_data.get("floor") == "elevator" or room_id.startswith("elevator_")


def _component_from_template_payload(payload: dict, parent_object_id: str) -> EditableComponent:
    category = payload.get("category", "component")
    defaults = build_category_defaults(category, "simple object")
    component = EditableComponent(
        object_id=payload["object_id"],
        name=payload.get("name", payload["object_id"]),
        category=category,
        template_name=payload.get("template_name", "custom"),
        description=payload.get("description", ""),
        object_group=get_category_schema(category)["default_object_group"],
        static_attributes=defaults["static_attributes"],
        state_variables=defaults["state_variables"],
        capabilities=defaults["capabilities"],
        relations={},
    )
    normalized = normalize_object_payload(
        category,
        {
            "static_attributes": payload.get("static_attributes", {}),
            "state_variables": payload.get("state_variables", {}),
            "capabilities": payload.get("capabilities", {}),
            "relations": payload.get("relations", {}),
        },
    )
    component.static_attributes.update(normalized["static_attributes"])
    component.state_variables.update(normalized["state_variables"])
    component.capabilities.update(normalized["capabilities"])
    component.relations.update(normalized["relations"])
    component.relations.setdefault("part_of", parent_object_id)
    return component
