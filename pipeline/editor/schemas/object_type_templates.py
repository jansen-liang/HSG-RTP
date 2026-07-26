from __future__ import annotations

from copy import deepcopy

from .enums import OBJECT_CATEGORIES


CATEGORY_SCHEMAS = {
    "food": {
        "default_object_group": "small_objects",
        "allowed_forms": ["simple object", "composite object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool", "default": True},
            "fragile": {"type": "bool"},
            "edible": {"type": "bool", "default": True},
            "is_container": {"type": "bool", "default": False},
            "is_support": {"type": "bool", "default": False},
            "perishable": {"type": "bool", "default": True},
        },
        "allowed_state_variables": {
            "spoilage": {"type": "enum", "options": ["fresh", "stale", "spoiled"]},
            "temperature": {"type": "enum", "options": ["cold", "room", "hot"]},
            "cooked_state": {"type": "enum", "options": ["raw", "cooked", "burnt"], "optional": True},
            "cleanliness": {"type": "enum", "options": ["clean", "dirty"], "optional": True},
        },
        "allowed_capabilities": {
            "pickupable": {"type": "bool"},
            "placeable": {"type": "bool"},
            "cookable": {"type": "bool", "optional": True},
            "washable": {"type": "bool", "optional": True},
        },
        "allowed_relations": ["in_room", "on", "inside", "held_by"],
        "rules": ["only_food_supports_spoilage"],
    },
    "furniture": {
        "default_object_group": "large_objects",
        "allowed_forms": ["simple object", "composite object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool"},
            "fragile": {"type": "bool"},
            "is_container": {"type": "bool"},
            "is_support": {"type": "bool"},
        },
        "allowed_state_variables": {
            "is_open": {"type": "bool", "optional": True},
            "cleanliness": {"type": "enum", "options": ["clean", "dirty"]},
            "is_broken": {"type": "bool"},
            "occupancy": {"type": "enum", "options": ["empty", "partial", "full"], "optional": True},
        },
        "allowed_capabilities": {
            "openable": {"type": "bool", "optional": True},
            "receptacle": {"type": "bool", "optional": True},
            "support_surface": {"type": "bool", "optional": True},
        },
        "allowed_relations": ["in_room", "contains", "on", "part_of", "inside"],
        "rules": [],
    },
    "appliance": {
        "default_object_group": "large_objects",
        "allowed_forms": ["simple object", "composite object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool"},
            "fragile": {"type": "bool"},
            "requires_power": {"type": "bool", "default": True},
            "is_container": {"type": "bool"},
            "is_support": {"type": "bool", "default": False},
            "is_composite": {"type": "bool"},
        },
        "allowed_state_variables": {
            "power_state": {"type": "enum", "options": ["off", "on"]},
            "machine_state": {"type": "enum", "options": ["idle", "ready", "running", "paused", "finished", "error"]},
            "is_open": {"type": "bool", "optional": True},
            "door_locked": {"type": "bool", "optional": True},
            "current_mode": {"type": "text", "optional": True},
            "fill_level": {"type": "number", "optional": True},
            "temperature_mode": {"type": "enum", "options": ["cold", "room", "hot"], "optional": True},
        },
        "allowed_capabilities": {
            "toggleable": {"type": "bool"},
            "openable": {"type": "bool", "optional": True},
            "programmable": {"type": "bool", "optional": True},
            "receptacle": {"type": "bool", "optional": True},
        },
        "allowed_relations": ["in_room", "part_of", "controls", "contains", "inside", "connected_to"],
        "rules": [],
    },
    "container": {
        "default_object_group": "small_objects",
        "allowed_forms": ["simple object", "composite object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool"},
            "fragile": {"type": "bool"},
            "is_container": {"type": "bool", "default": True},
            "is_support": {"type": "bool"},
            "capacity": {"type": "number", "optional": True},
        },
        "allowed_state_variables": {
            "is_open": {"type": "bool", "optional": True},
            "fill_level": {"type": "number"},
            "occupancy": {"type": "enum", "options": ["empty", "partial", "full"]},
            "cleanliness": {"type": "enum", "options": ["clean", "dirty"]},
        },
        "allowed_capabilities": {
            "receptacle": {"type": "bool"},
            "fillable": {"type": "bool"},
            "openable": {"type": "bool", "optional": True},
            "pickupable": {"type": "bool", "optional": True},
        },
        "allowed_relations": ["in_room", "inside", "on", "contains", "held_by", "part_of"],
        "rules": ["container_requires_is_container"],
    },
    "cloth": {
        "default_object_group": "small_objects",
        "allowed_forms": ["simple object", "composite object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool", "default": True},
            "fragile": {"type": "bool"},
            "is_container": {"type": "bool", "default": False},
            "is_support": {"type": "bool", "default": False},
            "wearable": {"type": "bool", "optional": True},
        },
        "allowed_state_variables": {
            "cleanliness": {"type": "enum", "options": ["clean", "dirty"]},
            "wetness": {"type": "enum", "options": ["dry", "damp", "wet"]},
            "folded": {"type": "bool"},
        },
        "allowed_capabilities": {
            "pickupable": {"type": "bool"},
            "washable": {"type": "bool"},
            "dryable": {"type": "bool"},
            "foldable": {"type": "bool"},
        },
        "allowed_relations": ["in_room", "inside", "on", "held_by"],
        "rules": [],
    },
    "tool": {
        "default_object_group": "small_objects",
        "allowed_forms": ["simple object", "composite object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool", "default": True},
            "fragile": {"type": "bool"},
            "is_container": {"type": "bool", "default": False},
            "is_support": {"type": "bool", "default": False},
            "sharp": {"type": "bool", "optional": True},
        },
        "allowed_state_variables": {
            "cleanliness": {"type": "enum", "options": ["clean", "dirty"]},
            "is_broken": {"type": "bool"},
            "sharpness": {"type": "enum", "options": ["dull", "normal", "sharp"], "optional": True},
        },
        "allowed_capabilities": {
            "pickupable": {"type": "bool"},
            "usable": {"type": "bool"},
            "washable": {"type": "bool"},
        },
        "allowed_relations": ["in_room", "inside", "on", "held_by"],
        "rules": [],
    },
    "button": {
        "default_object_group": "small_objects",
        "allowed_forms": ["simple object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool", "default": False},
            "fragile": {"type": "bool", "default": False},
            "is_container": {"type": "bool", "default": False},
            "is_support": {"type": "bool", "default": False},
        },
        "allowed_state_variables": {
            "is_pressed": {"type": "bool"},
            "enabled": {"type": "bool", "optional": True},
        },
        "allowed_capabilities": {
            "pressable": {"type": "bool"},
        },
        "allowed_relations": ["in_room", "part_of", "controls", "inside", "outside"],
        "rules": ["button_supports_is_pressed"],
    },
    "component": {
        "default_object_group": "small_objects",
        "allowed_forms": ["simple object"],
        "allowed_static_attributes": {
            "movable": {"type": "bool"},
            "fragile": {"type": "bool"},
            "is_container": {"type": "bool"},
            "is_support": {"type": "bool"},
            "component_role": {"type": "enum", "options": ["door", "drawer", "drum", "shelf", "panel", "handle"]},
        },
        "allowed_state_variables": {
            "is_open": {"type": "bool", "optional": True},
            "fill_level": {"type": "number", "optional": True},
            "load_level": {"type": "number", "optional": True},
            "cleanliness": {"type": "enum", "options": ["clean", "dirty"]},
            "is_locked": {"type": "bool", "optional": True},
        },
        "allowed_capabilities": {
            "openable": {"type": "bool"},
            "receptacle": {"type": "bool"},
            "fillable": {"type": "bool"},
            "pressable": {"type": "bool", "optional": True},
            "toggleable": {"type": "bool", "optional": True},
        },
        "allowed_relations": ["part_of", "controls", "contains", "inside", "in_room"],
        "rules": ["component_requires_part_of"],
    },
}


OBJECT_TEMPLATE_LIBRARY = {
    "custom": {
        "label": "Custom",
        "category": None,
        "object_form": None,
        "description": "",
        "static_attributes": {},
        "state_variables": {},
        "capabilities": {},
        "relations": {},
        "components": [],
    },
    "apple_template": {
        "label": "Apple",
        "category": "food",
        "object_form": "simple object",
        "description": "A default apple object.",
        "static_attributes": {
            "movable": True,
            "fragile": True,
            "edible": True,
            "perishable": True,
            "is_container": False,
            "is_support": False,
        },
        "state_variables": {
            "spoilage": "fresh",
            "temperature": "room",
        },
        "capabilities": {
            "pickupable": True,
            "placeable": True,
        },
        "relations": {},
        "components": [],
    },
    "mug_template": {
        "label": "Mug",
        "category": "container",
        "object_form": "simple object",
        "description": "A cup-shaped receptacle.",
        "static_attributes": {
            "movable": True,
            "fragile": True,
            "is_container": True,
        },
        "state_variables": {
            "fill_level": 0,
            "cleanliness": "clean",
            "occupancy": "empty",
        },
        "capabilities": {
            "pickupable": True,
            "fillable": True,
            "receptacle": True,
        },
        "relations": {},
        "components": [],
    },
    "fridge_template": {
        "label": "Fridge",
        "category": "appliance",
        "object_form": "composite object",
        "description": "A composite fridge with shelves and compartments.",
        "static_attributes": {
            "requires_power": True,
            "is_container": True,
            "is_support": False,
            "is_composite": True,
        },
        "state_variables": {
            "power_state": "on",
            "machine_state": "idle",
            "temperature_mode": "cold",
        },
        "capabilities": {
            "toggleable": True,
            "receptacle": True,
        },
        "relations": {},
        "components": [
            {
                "object_id": "fridge_door",
                "name": "Fridge Door",
                "category": "component",
                "description": "Primary fridge door.",
                "static_attributes": {"component_role": "door"},
                "state_variables": {"is_open": False, "cleanliness": "clean"},
                "capabilities": {"openable": True},
                "relations": {},
            },
            {
                "object_id": "fridge_shelf_upper",
                "name": "Upper Shelf",
                "category": "component",
                "description": "Upper storage shelf.",
                "static_attributes": {"component_role": "shelf", "is_support": True},
                "state_variables": {"cleanliness": "clean"},
                "capabilities": {"receptacle": True},
                "relations": {},
            },
            {
                "object_id": "fridge_shelf_lower",
                "name": "Lower Shelf",
                "category": "component",
                "description": "Lower storage shelf.",
                "static_attributes": {"component_role": "shelf", "is_support": True},
                "state_variables": {"cleanliness": "clean"},
                "capabilities": {"receptacle": True},
                "relations": {},
            },
            {
                "object_id": "freezer_compartment",
                "name": "Freezer Compartment",
                "category": "container",
                "description": "Cold inner freezer compartment.",
                "static_attributes": {"is_container": True, "movable": False},
                "state_variables": {"fill_level": 0, "occupancy": "empty", "cleanliness": "clean"},
                "capabilities": {"receptacle": True},
                "relations": {},
            },
        ],
    },
    "washing_machine_template": {
        "label": "Washing Machine",
        "category": "appliance",
        "object_form": "composite object",
        "description": "A composite washing machine with essential parts.",
        "static_attributes": {
            "requires_power": True,
            "is_container": True,
            "is_support": False,
            "is_composite": True,
        },
        "state_variables": {
            "power_state": "off",
            "machine_state": "idle",
            "door_locked": False,
        },
        "capabilities": {
            "toggleable": True,
            "programmable": True,
        },
        "relations": {},
        "components": [
            {
                "object_id": "washer_door",
                "name": "Washer Door",
                "category": "component",
                "description": "Front washer door.",
                "static_attributes": {"component_role": "door"},
                "state_variables": {"is_open": False, "cleanliness": "clean"},
                "capabilities": {"openable": True},
                "relations": {},
            },
            {
                "object_id": "detergent_drawer",
                "name": "Detergent Drawer",
                "category": "component",
                "description": "Drawer for detergent.",
                "static_attributes": {"component_role": "drawer", "is_container": True},
                "state_variables": {"is_open": False, "fill_level": 0, "cleanliness": "clean"},
                "capabilities": {"openable": True, "fillable": True, "receptacle": True},
                "relations": {},
            },
            {
                "object_id": "washer_drum",
                "name": "Washer Drum",
                "category": "component",
                "description": "Main washing drum.",
                "static_attributes": {"component_role": "drum", "is_container": True},
                "state_variables": {"load_level": 0, "cleanliness": "clean"},
                "capabilities": {"receptacle": True},
                "relations": {},
            },
            {
                "object_id": "start_button",
                "name": "Start Button",
                "category": "button",
                "description": "Start program button.",
                "static_attributes": {"movable": False, "fragile": False},
                "state_variables": {"is_pressed": False},
                "capabilities": {"pressable": True},
                "relations": {"controls": ""},
            },
        ],
    },
}


for category in OBJECT_CATEGORIES:
    if category not in CATEGORY_SCHEMAS:
        raise ValueError(f"Missing category schema for '{category}'.")


def get_category_schema(category: str) -> dict:
    return deepcopy(CATEGORY_SCHEMAS[category])


def get_template_config(template_name: str) -> dict:
    return deepcopy(OBJECT_TEMPLATE_LIBRARY[template_name])


def get_template_names_for_category(category: str | None) -> list[str]:
    names = ["custom"]
    for template_name, template in OBJECT_TEMPLATE_LIBRARY.items():
        if template_name == "custom":
            continue
        if category is None or template.get("category") == category:
            names.append(template_name)
    return names


def build_category_defaults(category: str, object_form: str) -> dict:
    schema = CATEGORY_SCHEMAS[category]
    static_attributes = {}
    state_variables = {}
    capabilities = {}
    for field_name, config in schema["allowed_static_attributes"].items():
        if "default" in config:
            static_attributes[field_name] = config["default"]
    for field_name, config in schema["allowed_state_variables"].items():
        if "default" in config:
            state_variables[field_name] = config["default"]
    for field_name, config in schema["allowed_capabilities"].items():
        if "default" in config:
            capabilities[field_name] = config["default"]
    if object_form == "composite object":
        static_attributes["is_composite"] = True
    return {
        "static_attributes": static_attributes,
        "state_variables": state_variables,
        "capabilities": capabilities,
        "relations": {},
    }


def normalize_object_payload(category: str, payload: dict) -> dict:
    schema = CATEGORY_SCHEMAS[category]
    normalized = {
        "static_attributes": {},
        "state_variables": {},
        "capabilities": {},
        "relations": {},
    }
    for section_name, schema_key in [
        ("static_attributes", "allowed_static_attributes"),
        ("state_variables", "allowed_state_variables"),
        ("capabilities", "allowed_capabilities"),
    ]:
        allowed = schema[schema_key]
        for field_name, value in payload.get(section_name, {}).items():
            if field_name in allowed:
                normalized[section_name][field_name] = value
    for relation_name, target in payload.get("relations", {}).items():
        if relation_name in schema["allowed_relations"]:
            normalized["relations"][relation_name] = target
    return normalized


COMPONENT_TEMPLATE_LIBRARY = {
    "custom": {
        "label": "Custom",
        "category": "component",
        "static_attributes": {},
        "state_variables": {},
        "capabilities": {},
        "relations": {},
    },
    "door": {
        "label": "Door",
        "category": "component",
        "static_attributes": {
            "component_role": "door",
            "movable": False,
            "is_container": False,
            "is_support": False,
        },
        "state_variables": {
            "is_open": False,
            "cleanliness": "clean",
            "is_locked": False,
        },
        "capabilities": {
            "openable": True,
        },
        "relations": {},
    },
    "drawer": {
        "label": "Drawer",
        "category": "component",
        "static_attributes": {
            "component_role": "drawer",
            "is_container": True,
            "is_support": False,
        },
        "state_variables": {
            "is_open": False,
            "fill_level": 0,
            "cleanliness": "clean",
        },
        "capabilities": {
            "openable": True,
            "receptacle": True,
            "fillable": True,
        },
        "relations": {},
    },
    "button": {
        "label": "Button",
        "category": "button",
        "static_attributes": {
            "movable": False,
            "fragile": False,
            "is_container": False,
            "is_support": False,
        },
        "state_variables": {
            "is_pressed": False,
        },
        "capabilities": {
            "pressable": True,
        },
        "relations": {},
    },
    "handle": {
        "label": "Handle",
        "category": "component",
        "static_attributes": {
            "component_role": "handle",
            "movable": False,
            "is_container": False,
            "is_support": False,
        },
        "state_variables": {
            "cleanliness": "clean",
        },
        "capabilities": {},
        "relations": {},
    },
    "shelf": {
        "label": "Shelf",
        "category": "component",
        "static_attributes": {
            "component_role": "shelf",
            "is_container": False,
            "is_support": True,
        },
        "state_variables": {
            "cleanliness": "clean",
        },
        "capabilities": {
            "receptacle": True,
        },
        "relations": {},
    },
    "panel": {
        "label": "Panel",
        "category": "component",
        "static_attributes": {
            "component_role": "panel",
            "movable": False,
            "is_container": False,
            "is_support": False,
        },
        "state_variables": {
            "cleanliness": "clean",
        },
        "capabilities": {
            "pressable": True,
            "toggleable": True,
        },
        "relations": {},
    },
}


def get_component_template_names() -> list[str]:
    return list(COMPONENT_TEMPLATE_LIBRARY.keys())


def get_component_template_config(template_name: str) -> dict:
    return deepcopy(COMPONENT_TEMPLATE_LIBRARY[template_name])
