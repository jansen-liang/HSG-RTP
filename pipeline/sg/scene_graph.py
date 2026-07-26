"""
场景图数据结构

数据格式:
{
    "name": "场景名称",
    "macro_zones": {
        "区域ID": {
            "rooms": ["房间列表"]
        }
    },
    "rooms": {
        "房间ID": {
            "floor": "所属区域",
            "neighbor": ["相邻房间"],
            "large_objects": {
                "对象ID": {
                    "type": "类型",
                    "is_container": true/false
                }
            },
            "small_objects": {
                "对象ID": {
                    "type": "类型", 
                    "affordance": ["动作"],
                    "state": "状态",
                    "relation": {"on/in": "位置"}
                }
            }
        }
        "特殊电梯cabin":{
            "floor": "elevator",#不在macrozone里，假定这是一个虚拟楼层
            "neighbor": ["相邻房间"],
            "large_objects": {
                "对象ID": {
                    "type": "类型",
                    "is_container": true/false
                }
            },
            "small_objects": {
                "对象ID": {
                    "type": "类型", 
                    "affordance": ["动作"],
                    "state": "状态",
                    "relation": {"on/in": "位置"}
                }
            }
        }
    },
    "agent": {
        "position": "初始房间",
        "state": "初始状态"
    }
}

电梯按钮命名规范:
- 外部按钮: elevator_call_up/down (relation: {"outside": "elevator_cabin"})
- 内部按钮: elevator_button_[楼层] (relation: {"inside": "elevator_cabin"})
"""

HOTEL = {
    "name": "hotel",
    "macro_zones": {
        "floor_1_public": {
            "rooms": ["lobby", "restaurant", "bar", "elevator_1f"]
        },
        "floor_2_guest": {
            "rooms": ["hallway_2f", "room_201", "room_202", "elevator_2f"]
        },
        "floor_3_guest": {
            "rooms": ["hallway_3f", "room_301", "room_303", "elevator_3f"]
        }
    },
    "rooms": {
        "lobby": {
            "floor": "floor_1_public",
            "neighbor": ["restaurant", "bar", "elevator_1f"],
            "large_objects": {
                "front_desk": {"type": "furniture", "is_container": True},
                "reception_counter": {"type": "furniture", "is_container": True},
                "charging_station_1": {"type": "device"},
                "flower_vase_large": {"type": "decor"},
                "sofa_set": {"type": "furniture"},
                "coffee_table": {"type": "furniture"},
                "luggage_cart": {"type": "transport"},
                "concierge_desk": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "room_keycard": {
                    "type": "access_item",
                    "affordance": ["pick", "place", "use"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "activated": False
                    },
                    "physical_property": {"magnetic": True},
                    "relation": {"on": "front_desk"}
                },
                "hotel_brochure": {
                    "type": "information",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "readability": "readable"
                    },
                    "relation": {"on": "front_desk"}
                },
                "welcome_gift": {
                    "type": "package",
                    "affordance": ["pick", "place", "unwrap"],
                    "state": {
                        "wrapped": True,
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "physical_property": {"fragile": False},
                    "relation": {"on": "reception_counter"}
                },
                "takeaway_menu": {
                    "type": "information",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "readability": "readable"
                    },
                    "relation": {"on": "coffee_table"}
                },
                "fresh_flowers": {
                    "type": "decor",
                    "affordance": ["smell"],
                    "state": {
                        "temperature": "room",
                        "wetness": "moist",
                        "freshness": "fresh",
                        "availability": "available"
                    },
                    "physical_property": {"fragile": True},
                    "relation": {"in": "flower_vase_large"}
                }
            }
        },
        "restaurant": {
            "floor": "floor_1_public",
            "neighbor": ["lobby"],
            "large_objects": {
                "table_1": {"type": "furniture"},
                "table_2": {"type": "furniture"},
                "dining_chair_set": {"type": "furniture"},
                "buffet_counter": {"type": "furniture", "is_container": True},
                "service_cart": {"type": "transport"},
                "cash_register": {"type": "device"},
                "display_cabinet": {"type": "furniture", "is_container": True},
                "microwave": {
                    "type": "appliance", 
                    "is_container": True,
                    "methods": {
                        "heat": {
                            "description": "Heat items from cold/room to hot",
                            "state_changes": {"temperature": {"from": ["cold", "room"], "to": "hot"}},
                            "duration": 60
                        }
                    }
                }
            },
            "small_objects": {
                "pizza_box": {
                    "type": "food_container",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "warm",
                        "wetness": "dry",
                        "availability": "ready",
                        "opened": False
                    },
                    "physical_property": {"contains": "pizza"},
                    "relation": {"on": "table_1"}
                },
                "sushi_takeaway": {
                    "type": "food_container",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "fresh",
                        "opened": False
                    },
                    "physical_property": {"contains": "sushi", "fragile": True},
                    "relation": {"on": "table_2"}
                },
                "delivery_bag": {
                    "type": "container",
                    "affordance": ["pick", "place", "open", "close"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "ready",
                        "opened": False
                    },
                    "physical_property": {"insulated": True},
                    "relation": {"on": "service_cart"}
                },
                "restaurant_menu": {
                    "type": "information",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "readability": "readable"
                    },
                    "relation": {"on": "table_1"}
                },
                "chopsticks": {
                    "type": "utensil",
                    "affordance": ["pick", "use"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "relation": {"on": "buffet_counter"}
                },
                "napkins": {
                    "type": "supply",
                    "affordance": ["pick", "use"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "used": False
                    },
                    "relation": {"on": "table_2"}
                }
            }
        },
        "bar": {
            "floor": "floor_1_public",
            "neighbor": ["lobby"],
            "large_objects": {
                "bar_counter": {"type": "furniture", "is_container": True},
                "bar_stools": {"type": "furniture"},
                "wine_cabinet": {"type": "furniture", "is_container": True},
                "refrigerator": {"type": "appliance", "is_container": True},
                "cocktail_station": {"type": "furniture"}
            },
            "small_objects": {
                "red_wine": {
                    "type": "beverage",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "sealed",
                        "opened": False
                    },
                    "physical_property": {"fragile": True, "contains": "liquid"},
                    "relation": {"on": "bar_counter"}
                },
                "wine_glass": {
                    "type": "container",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "free"
                    },
                    "physical_property": {"fragile": True, "can_hold": "liquid"},
                    "relation": {"on": "bar_counter"}
                },
                "cocktail_shaker": {
                    "type": "tool",
                    "affordance": ["pick", "shake", "pour"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "physical_property": {"metal": True},
                    "relation": {"on": "cocktail_station"}
                },
                "beer_bottle": {
                    "type": "beverage",
                    "affordance": ["pick", "open"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "sealed",
                        "opened": False
                    },
                    "physical_property": {"fragile": True},
                    "relation": {"in": "refrigerator"}
                },
                "bar_towel": {
                    "type": "cleaning_supply",
                    "affordance": ["pick", "wipe"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "relation": {"on": "bar_counter"}
                },
                "coaster_set": {
                    "type": "accessory",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "used": False
                    },
                    "relation": {"on": "bar_counter"}
                }
            }
        },
        "elevator_1f": {
            "floor": "floor_1_public",
            "neighbor": ["lobby", "elevator_2f", "elevator_3f","elevator_cabin"],
            "large_objects": {
                "panel_1f": {"type": "appliance", "is_container": False},
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "hallway_2f": {
            "floor": "floor_2_guest",
            "neighbor": ["room_201", "room_202", "elevator_2f"],
            "large_objects": {
                "charging_pad": {"type": "device"},
                "housekeeping_cart": {"type": "transport"},
                "fire_extinguisher": {"type": "safety_equipment"},
                "decorative_table": {"type": "furniture"}
            },
            "small_objects": {
                "room_service_tray": {
                    "type": "service_item",
                    "affordance": ["pick", "carry"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "ready"
                    },
                    "physical_property": {"metal": True},
                    "relation": {"on": "housekeeping_cart"}
                },
                "towel_set": {
                    "type": "linen",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "physical_property": {"soft": True},
                    "relation": {"on": "housekeeping_cart"}
                },
                "emergency_phone": {
                    "type": "communication",
                    "affordance": ["use"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "active": False
                    },
                    "relation": {"on": "decorative_table"}
                }
            }
        },
        "room_201": {
            "floor": "floor_2_guest",
            "neighbor": ["hallway_2f"],
            "large_objects": {
                "guest": {"type": "human"},
                "door_201": {"type": "entrance", "affordance": ["open", "close"]},
                "king_bed": {"type": "furniture"},
                "wardrobe": {"type": "furniture", "is_container": True},
                "bedside_table": {"type": "furniture", "is_container": True},
                "armchair": {"type": "furniture"},
                "mini_fridge": {"type": "appliance", "is_container": True}
            },
            "small_objects": {
                "room_service_menu": {
                    "type": "information",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "readability": "readable"
                    },
                    "relation": {"on": "bedside_table"}
                },
                "chinese_takeout": {
                    "type": "food_container",
                    "affordance": ["pick", "open", "eat"],
                    "state": {
                        "temperature": "warm",
                        "wetness": "dry",
                        "availability": "delivered",
                        "opened": False
                    },
                    "physical_property": {"contains": "chinese_food"},
                    "relation": {"on": "bedside_table"}
                },
                "hotel_slippers": {
                    "type": "clothing",
                    "affordance": ["pick", "wear"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "relation": {"in": "wardrobe"}
                },
                "bottled_water": {
                    "type": "beverage",
                    "affordance": ["pick", "drink"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "sealed",
                        "opened": False
                    },
                    "relation": {"in": "mini_fridge"}
                },
                "tv_remote": {
                    "type": "device",
                    "affordance": ["pick", "use"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "battery_level": "full"
                    },
                    "relation": {"on": "bedside_table"}
                }
            }
        },
        "room_202": {
            "floor": "floor_2_guest",
            "neighbor": ["hallway_2f"],
            "large_objects": {
                "guest": {"type": "human"},
                "door_202": {"type": "entrance", "affordance": ["open", "close"]},
                "twin_beds": {"type": "furniture"},
                "desk": {"type": "furniture", "is_container": True},
                "dresser": {"type": "furniture", "is_container": True},
                "window": {"type": "structural"},
                "bathroom_door": {"type": "entrance", "affordance": ["open", "close"]}
            },
            "small_objects": {
                "pizza_delivery": {
                    "type": "food_container",
                    "affordance": ["pick", "open", "eat"],
                    "state": {
                        "temperature": "hot",
                        "wetness": "dry",
                        "availability": "delivered",
                        "opened": False
                    },
                    "physical_property": {"contains": "pizza"},
                    "relation": {"on": "desk"}
                },
                "laptop_charger": {
                    "type": "electronic_accessory",
                    "affordance": ["pick", "plug"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "plugged": False
                    },
                    "relation": {"on": "desk"}
                },
                "phone_charger": {
                    "type": "electronic_accessory",
                    "affordance": ["pick", "plug"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "plugged": False
                    },
                    "relation": {"in": "dresser"}
                },
                "travel_guide": {
                    "type": "information",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "readability": "readable"
                    },
                    "relation": {"on": "desk"}
                },
                "snack_bag": {
                    "type": "food_container",
                    "affordance": ["pick", "open", "eat"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "sealed",
                        "opened": False
                    },
                    "relation": {"in": "dresser"}
                }
            }
        },
        "elevator_2f": {
            "floor": "floor_2_guest",
            "neighbor": ["hallway_2f", "elevator_1f", "elevator_3f", "elevator_cabin"],
            "large_objects": {
                "panel_2f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "hallway_3f": {
            "floor": "floor_3_guest",
            "neighbor": ["room_301", "room_303", "elevator_3f"],
            "large_objects": {
                "charging_pad": {"type": "device"},
                "vending_machine": {"type": "appliance", "is_container": True},
                "ice_machine": {"type": "appliance"},
                "security_camera": {"type": "security_equipment"}
            },
            "small_objects": {
                "ice_bucket": {
                    "type": "container",
                    "affordance": ["pick", "fill"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "filled": False,
                        "availability": "empty"
                    },
                    "physical_property": {"plastic": True},
                    "relation": {"near": "ice_machine"}
                },
                "candy_bar": {
                    "type": "snack",
                    "affordance": ["pick", "buy"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "wrapped": True
                    },
                    "relation": {"in": "vending_machine"}
                },
                "soda_can": {
                    "type": "beverage",
                    "affordance": ["pick", "buy"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "available",
                        "opened": False
                    },
                    "physical_property": {"metal": True},
                    "relation": {"in": "vending_machine"}
                }
            }
        },
        "room_301": {
            "floor": "floor_3_guest",
            "neighbor": ["hallway_3f"],
            "large_objects": {
                "guest": {"type": "human"},
                "door_301": {"type": "entrance", "affordance": ["open", "close"]},
                "queen_bed": {"type": "furniture"},
                "work_desk": {"type": "furniture", "is_container": True},
                "closet": {"type": "furniture", "is_container": True},
                "lounge_chair": {"type": "furniture"},
                "balcony_door": {"type": "entrance", "affordance": ["open", "close"]},
                "sink": {
                    "type": "fixture",
                    "is_container": False,
                    "methods": {
                        "wet": {
                            "description": "Wet dry items",
                            "state_changes": {"wetness": {"from": ["dry"], "to": "wet"}},
                            "duration": 10
                        }
                    }
                }
            },
            "small_objects": {
                "sushi_delivery": {
                    "type": "food_container",
                    "affordance": ["pick", "open", "eat"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "fresh",
                        "opened": False
                    },
                    "physical_property": {"contains": "sushi", "fragile": True},
                    "relation": {"on": "work_desk"}
                },
                "business_cards": {
                    "type": "information",
                    "affordance": ["pick", "give"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "quantity": "full"
                    },
                    "relation": {"in": "work_desk"}
                },
                "conference_materials": {
                    "type": "document",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "important",
                        "readability": "readable"
                    },
                    "relation": {"on": "work_desk"}
                },
                "suit_jacket": {
                    "type": "clothing",
                    "affordance": ["pick", "wear"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "relation": {"in": "closet"}
                },
                "room_key": {
                    "type": "access_item",
                    "affordance": ["pick", "use"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "activated": False
                    },
                    "physical_property": {"magnetic": True},
                    "relation": {"on": "work_desk"}
                },
                "cleaning_cloth": {
                    "type": "cleaning_supply",
                    "affordance": ["pick", "wipe"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "relation": {"near": "sink"}
                }
            }
        },
        "room_303": {
            "floor": "floor_3_guest",
            "neighbor": ["hallway_3f"],
            "large_objects": {
                "guest": {"type": "human"},
                "door_303": {"type": "entrance", "affordance": ["open", "close"]},
                "double_bed": {"type": "furniture"},
                "vanity_table": {"type": "furniture", "is_container": True},
                "luggage_stand": {"type": "furniture"},
                "reading_chair": {"type": "furniture"},
                "safe_box": {"type": "security_furniture", "is_container": True},
                "dryer": {
                    "type": "appliance",
                    "is_container": True,
                    "methods": {
                        "dry": {
                            "description": "Dry wet items",
                            "state_changes": {"wetness": {"from": ["wet", "moist"], "to": "dry"}},
                            "duration": 120
                        }
                    }
                }
            },
            "small_objects": {
                "indian_takeaway": {
                    "type": "food_container",
                    "affordance": ["pick", "open", "eat"],
                    "state": {
                        "temperature": "warm",
                        "wetness": "dry",
                        "availability": "aromatic",
                        "opened": False
                    },
                    "physical_property": {"contains": "curry", "spicy": True},
                    "relation": {"on": "vanity_table"}
                },
                "jewelry_box": {
                    "type": "container",
                    "affordance": ["pick", "open"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "closed",
                        "locked": True
                    },
                    "physical_property": {"valuable": True},
                    "relation": {"in": "safe_box"}
                },
                "perfume_bottle": {
                    "type": "cosmetic",
                    "affordance": ["pick", "spray"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "fragrant": True
                    },
                    "physical_property": {"fragile": True},
                    "relation": {"on": "vanity_table"}
                },
                "travel_documents": {
                    "type": "document",
                    "affordance": ["pick", "read"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "important",
                        "readability": "readable"
                    },
                    "relation": {"in": "safe_box"}
                },
                "room_service_bell": {
                    "type": "service_item",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "pressed": False
                    },
                    "relation": {"on": "vanity_table"}
                },
                "chocolate_box": {
                    "type": "snack",
                    "affordance": ["pick", "open", "eat"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "sealed",
                        "wrapped": True
                    },
                    "physical_property": {"sweet": True, "gift": True},
                    "relation": {"on": "reading_chair"}
                }
            }
        },
        "elevator_3f": {
            "floor": "floor_3_guest",
            "neighbor": ["hallway_3f", "elevator_1f", "elevator_2f", "elevator_cabin"],
            "large_objects": {
                "panel_3f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "elevator_cabin": {
            "floor": "elevator",
            "neighbor": ["elevator_1f", "elevator_2f", "elevator_3f"],
            "large_objects": {},
            "small_objects": {
                "elevator_button_1": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_2": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_3": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
            }
        },
    },
    "agent": {
        "position": "lobby",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot"
    }
}

ALLENSVILLE = {
    "name": "allensville",
    "macro_zones": {
        "floor_1": {
            "rooms": ["living_room", "dining_room", "kitchen", "corridor_1", "elevator_1f"]
        },
        "floor_2": {
            "rooms": ["bedroom_1", "bedroom_2", "bathroom_1", "corridor_2", "elevator_2f"]
        },
        "floor_3": {
            "rooms": ["lobby", "bathroom_2", "corridor_3", "elevator_3f"]
        }
    },
    "rooms": {
        "bathroom_1": {
            "floor": "floor_2",
            "neighbor": ["corridor_2"],
            "large_objects": {
                "sink_1": {"type": "fixture"},
            },
            "small_objects": {
                "psu": {
                    "type": "electronics",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "powered": False
                    },
                    "physical_property": {"fragile": False},
                    "relation": {"on": "floor"}
                },
                "mop": {
                    "type": "tool",
                    "affordance": ["pick", "place", "use"],
                    "state": {
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "physical_property": {"absorbent": True},
                    "relation": {"leaning_on": "sink_1"}
                }
            }
        },
        "bedroom_1": {
            "floor": "floor_2",
            "neighbor": ["corridor_2"],
            "large_objects": {
                "bed_1": {"type": "furniture"},
                "shelf": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "mainboard": {
                    "type": "electronics",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                        "assembled": False
                    },
                    "physical_property": {"fragile": True},
                    "relation": {"on": "bed_1"} 
                },
                "book": {
                    "type": "document",
                    "affordance": ["pick", "read", "place"],
                    "state": {
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available",
                        "read": False
                    },
                    "relation": {"in": "shelf"}
                }
            }
        },
        "kitchen": {
            "floor": "floor_1",
            "neighbor": ["corridor_1", "dining_room"],
            "large_objects": {
                "microwave": {"type": "appliance", "is_container": True},
                "rubbish_bin": {"type": "container"},
                "counter": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "leftover_pizza_box": {
                    "type": "food_container",
                    "affordance": ["pick", "heat", "eat"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "available",
                        "edible": True
                    },
                    "physical_property": {"contains": "pizza", "fragile": False},
                    "relation": {"on": "counter"}  
                }
            }
        },
        "living_room": {
            "floor": "floor_1",
            "neighbor": ["corridor_1", "dining_room"],
            "large_objects": {
                "robot_hub": {"type": "device"},
                "couch": {"type": "furniture"}
            },
            "small_objects": {
                "tv_remote": {
                    "type": "control",
                    "affordance": ["pick", "use"],
                    "state": {
                        "battery": "low",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"on": "couch"}
                }
            }
        },
        "lobby": {
            "floor": "floor_3",
            "neighbor": ["corridor_3"],
            "large_objects": {
                "locker": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "paper": {
                    "type": "document",
                    "affordance": ["pick", "read", "file"],
                    "state": {
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available",
                        "signed": False
                    },
                    "relation": {"in": "locker"}
                }
            }
        },
        "corridor_1": {
            "floor": "floor_1",
            "neighbor": ["living_room", "kitchen", "elevator_1f", "corridor_2"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_1f": {
            "floor": "floor_1",
            "neighbor": ["corridor_1", "elevator_2f", "elevator_3f","elevator_cabin"],
            "large_objects": {
                "panel_1f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "corridor_2": {
            "floor": "floor_2",
            "neighbor": ["corridor_1", "corridor_3", "bathroom_1", "bedroom_1", "bedroom_2", "elevator_2f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_2f": {
            "floor": "floor_2",
            "neighbor": ["corridor_2", "elevator_1f", "elevator_3f", "elevator_cabin"],
            "large_objects": {
                "panel_2f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "corridor_3": {
            "floor": "floor_3",
            "neighbor": ["corridor_2", "lobby", "bathroom_2", "elevator_3f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_3f": {
            "floor": "floor_3",
            "neighbor": ["corridor_3", "elevator_1f", "elevator_2f","elevator_cabin"],
            "large_objects": {
                "panel_3f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "dining_room": {
            "floor": "floor_1",
            "neighbor": ["living_room", "kitchen"],
            "large_objects": {
                "dining_table": {"type": "furniture"},
                "chairs": {"type": "furniture"}
            },
            "small_objects": {
                "coffee_mug": {
                    "type": "container",
                    "affordance": ["pick", "drink", "wash"],
                    "state": {
                        "temperature": "warm",
                        "wetness": "wet",
                        "cleanliness": "dirty",
                        "availability": "available"
                    },
                    "relation": {"on": "dining_table"}
                }
            }
        },
        "bedroom_2": {
            "floor": "floor_2",
            "neighbor": ["corridor_2"],
            "large_objects": {
                "bed_2": {"type": "furniture"},
                "wardrobe": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "dirty_tshirt": {
                    "type": "clothing",
                    "affordance": ["pick", "wash"],
                    "state": {
                        "wetness": "dry",
                        "cleanliness": "dirty",
                        "availability": "available"
                    },
                    "relation": {"on": "bed_2"}
                }
            }
        },
        "bathroom_2": {
            "floor": "floor_3",
            "neighbor": ["corridor_3"],
            "large_objects": {
                "sink_2": {"type": "fixture"},
                "towel_rack": {"type": "fixture"}
            },
            "small_objects": {
                "bath_towel": {
                    "type": "linen",
                    "affordance": ["pick", "use", "wash"],
                    "state": {
                        "wetness": "dry",
                        "cleanliness": "clean",
                        "availability": "available"
                    },
                    "relation": {"on": "towel_rack"}
                }
            }
        },
        "elevator_cabin": {
            "floor": "elevator",
            "neighbor": ["elevator_1f", "elevator_2f", "elevator_3f"],
            "large_objects": {},
            "small_objects": {
                "elevator_button_1": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {"lit": False, "availability": "available"},
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_2": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {"lit": False, "availability": "available"},
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_3": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {"lit": False, "availability": "available"},
                    "relation": {"inside": "elevator_cabin"}
                },
            }
        },
    },
    "agent": {
        "position": "living_room",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot"
    }
}

SUPERMARKET = {
    "name": "supermarket",
    "macro_zones": {
        "floor_1": {
            "rooms": ["entrance", "shopping_cart_area", "produce_aisle_1", "dairy_aisle", "customer_service_desk", "elevator_1f"]
        },
        "floor_2": {
            "rooms": ["produce_aisle_2", "frozen_food_aisle", "bakery", "checkout_1", "checkout_2", "elevator_2f"]
        },
        "floor_3": {
            "rooms": ["checkout_3", "express_checkout", "exit", "elevator_3f"]
        }
    },
    "rooms": {
        "entrance": {
            "floor": "floor_1",
            "neighbor": ["shopping_cart_area", "produce_aisle_1","customer_service_desk"],
            "large_objects": {
                "shopping_cart_dispenser": {"type": "device"}
            },
            "small_objects": {}
        },
        "shopping_cart_area": {
            "floor": "floor_1",
            "neighbor": ["entrance", "produce_aisle_1"],
            "large_objects": {
                "shopping_cart_1": {"type": "container", "is_container": True},
                "shopping_cart_2": {"type": "container", "is_container": True}
            },
            "small_objects": {}
        },
        "produce_aisle_1": {
            "floor": "floor_1",
            "neighbor": ["entrance", "shopping_cart_area", "produce_aisle_2", "dairy_aisle"],
            "large_objects": {
                "apple_display": {"type": "display"},
                "banana_display": {"type": "display"},
                "lettuce_bin": {"type": "container"}
            },
            "small_objects": {
                "apple": {
                    "type": "fruit",
                    "affordance": ["pick", "place"],
                    "state": {
                        "freshness": "fresh",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"on": "apple_display"}
                },
                "banana": {
                    "type": "fruit",
                    "affordance": ["pick", "place"],
                    "state": {
                        "freshness": "fresh",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"on": "banana_display"}
                },
                "lettuce": {
                    "type": "vegetable",
                    "affordance": ["pick", "place"],
                    "state": {
                        "freshness": "fresh",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "lettuce_bin"}
                }
            }
        },
        "dairy_aisle": {
            "floor": "floor_1",
            "neighbor": ["produce_aisle_1", "bakery", "frozen_food_aisle"],
            "large_objects": {
                "milk_cooler": {"type": "appliance", "is_container": True},
                "cheese_case": {"type": "display", "is_container": True}
            },
            "small_objects": {
                "milk_gallon": {
                    "type": "dairy",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "milk_cooler"}
                },
                "cheddar_cheese": {
                    "type": "dairy",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "cheese_case"}
                }
            }
        },
        "customer_service_desk": {
            "floor": "floor_1",
            "neighbor": ["entrance", "elevator_1f"],
            "large_objects": {},
            "small_objects": {
                "service_bell": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "lit": False,
                        "availability": "available"
                    },
                    "relation": {"on": "desk"}  
                }
            }
        },
        "elevator_1f": {
            "floor": "floor_1",
            "neighbor": ["customer_service_desk", "elevator_2f", "elevator_3f", "elevator_cabin"],
            "large_objects": {
                "panel_1f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "bakery": {
            "floor": "floor_2",
            "neighbor": ["dairy_aisle", "checkout_1"],
            "large_objects": {
                "bakery_counter": {"type": "furniture"},
                "bread_rack": {"type": "display", "is_container": True}
            },
            "small_objects": {
                "baguette": {
                    "type": "bread",
                    "affordance": ["pick", "place", "heat"],
                    "state": {
                        "temperature": "cold",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"on": "bread_rack"}
                }
            }
        },
        "checkout_1": {
            "floor": "floor_2",
            "neighbor": ["bakery", "checkout_2"],
            "large_objects": {
                "checkout_belt": {"type": "device"},
                "bagging_area": {"type": "surface"}
            },
            "small_objects": {}
        },
        "produce_aisle_2": {
            "floor": "floor_2",
            "neighbor": ["produce_aisle_1", "frozen_food_aisle"],
            "large_objects": {
                "vegetable_display": {"type": "display", "is_container": True},
                "fruit_display": {"type": "display", "is_container": True}
            },
            "small_objects": {
                "carrot_bunch": {
                    "type": "vegetable",
                    "affordance": ["pick", "place"],
                    "state": {
                        "freshness": "fresh",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "vegetable_display"}
                },
                "banana_bunch": {
                    "type": "fruit",
                    "affordance": ["pick", "place"],
                    "state": {
                        "freshness": "fresh",
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "fruit_display"}
                }
            }
        },
        "frozen_food_aisle": {
            "floor": "floor_2",
            "neighbor": ["dairy_aisle", "produce_aisle_2", "checkout_2"],
            "large_objects": {
                "ice_cream_freezer": {"type": "appliance", "is_container": True},
                "frozen_meals_freezer": {"type": "appliance", "is_container": True}
            },
            "small_objects": {
                "ice_cream_container": {
                    "type": "frozen_dessert",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "frozen",
                        "melted": False,
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "ice_cream_freezer"}
                },
                "frozen_pizza": {
                    "type": "frozen_food",
                    "affordance": ["pick", "place", "heat"],
                    "state": {
                        "temperature": "frozen",
                        "cooked": False,
                        "wetness": "dry",
                        "availability": "available"
                    },
                    "relation": {"in": "frozen_meals_freezer"}
                }
            }
        },
        "checkout_2": {
            "floor": "floor_2",
            "neighbor": ["frozen_food_aisle", "checkout_1", "checkout_3", "elevator_2f"],
            "large_objects": {
                "checkout_belt_2": {"type": "device"},
                "bagging_area_2": {"type": "surface"}
            },
            "small_objects": {}
        },
        "elevator_2f": {
            "floor": "floor_2",
            "neighbor": ["checkout_2", "elevator_1f", "elevator_3f", "elevator_cabin"],
            "large_objects": {
                "panel_2f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "checkout_3": {
            "floor": "floor_3",
            "neighbor": ["checkout_2", "express_checkout"],
            "large_objects": {
                "checkout_belt_3": {"type": "device"},
                "bagging_area_3": {"type": "surface"}
            },
            "small_objects": {}
        },
        "express_checkout": {
            "floor": "floor_3",
            "neighbor": ["checkout_3", "exit"],
            "large_objects": {
                "express_belt": {"type": "device"}
            },
            "small_objects": {}
        },
        "exit": {
            "floor": "floor_3",
            "neighbor": ["express_checkout", "elevator_3f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_3f": {
            "floor": "floor_3",
            "neighbor": ["exit", "elevator_1f", "elevator_2f","elevator_cabin"],
            "large_objects": {
                "panel_3f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "elevator_cabin": {
            "floor": "elevator",
            "neighbor": ["elevator_1f", "elevator_2f", "elevator_3f"],
            "large_objects": {},
            "small_objects": {
                "elevator_button_1": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {"lit": False, "availability": "available"},
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_2": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {"lit": False, "availability": "available"},
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_3": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {"lit": False, "availability": "available"},
                    "relation": {"inside": "elevator_cabin"}
                },
            }
        },
    },
    "agent": {
        "position": "entrance",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot"
    }
}

OFFICE = {
    "name": "office",
    "macro_zones": {
        "floor_1": {
            "rooms": [
                "peters_office", "tobis_office", "nikos_office", "michaels_office",
                "meeting_room_1", "meeting_room_2", "mobile_robotics_lab", "agriculture_lab","manipulation_lab",
                "corridor_1", "corridor_2", "corridor_3", "corridor_4", "corridor_5", "corridor_6", "elevator_1f"
            ]
        },
        "floor_2": {
            "rooms": [
                "aarons_office", "jasons_office", "filipes_office", "luis_office", "wills_office",
                "meeting_room_3", "meeting_room_4",
                "corridor_7", "corridor_8", "corridor_9", "corridor_10", "corridor_11", "corridor_12", "elevator_2f"
            ]
        },
        "floor_3": {
            "rooms": [
                "ajays_office", "chris_office", "lauriannes_office",
                "dimitys_office", "presentation_lounge", "printing_zone_2",
                "corridor_13", "corridor_14", "corridor_15", "corridor_16", "corridor_17", "corridor_18", "elevator_3f"
            ]
        },
        "floor_4": {
            "rooms": [
                "kitchen", "cafeteria", "lobby", "robot_lounge_1", "robot_lounge_2",
                "phd_bay_1", "postdoc_bay_1", "printing_zone_1", "supplies_station",
                "corridor_19", "corridor_20", "corridor_21", "corridor_22", "corridor_23", "elevator_4f"
            ]
        },
        "floor_5": {
            "rooms": [
                "postdoc_bay_2", "postdoc_bay_3", 
                "phd_bay_2", "phd_bay_3", "phd_bay_4",
                "corridor_24", "corridor_25", "corridor_26", "elevator_5f"
            ]
        }
    },
    "rooms": {
        "peters_office": {
            "floor": "floor_1",
            "neighbor": ["corridor_1"],
            "large_objects": {
                "desk_2": {"type": "furniture"},
                "cabinet_2": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "phone": {
                    "type": "electronics",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"in": "cabinet_2"}
                },
                "apple_3": {
                    "type": "fruit",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"in": "cabinet_2"}
                },
                "stapler_1": {
                    "type": "stationery",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"in": "cabinet_2"}
                }
            }
        },
        "tobis_office": {
            "floor": "floor_1",
            "neighbor": ["corridor_4"],
            "large_objects": {},
            "small_objects": {}
        },
        "nikos_office": {
            "floor": "floor_1",
            "neighbor": ["corridor_5"],
            "large_objects": {},
            "small_objects": {}
        },
        "michaels_office": {
            "floor": "floor_1",
            "neighbor": ["corridor_6"],
            "large_objects": {},
            "small_objects": {}
        },
        "aarons_office": {
            "floor": "floor_2",
            "neighbor": ["corridor_7"],
            "large_objects": {},
            "small_objects": {}
        },
        "jasons_office": {
            "floor": "floor_2",
            "neighbor": ["corridor_8"],
            "large_objects": {},
            "small_objects": {}
        },
        "filipes_office": {
            "floor": "floor_2",
            "neighbor": ["corridor_9"],
            "large_objects": {},
            "small_objects": {}
        },
        "luis_office": {
            "floor": "floor_2",
            "neighbor": ["corridor_10"],
            "large_objects": {},
            "small_objects": {}
        },
        "wills_office": {
            "floor": "floor_2",
            "neighbor": ["corridor_11"],
            "large_objects": {},
            "small_objects": {}
        },
        "ajays_office": {
            "floor": "floor_3",
            "neighbor": ["corridor_13"],
            "large_objects": {},
            "small_objects": {}
        },
        "chris_office": {
            "floor": "floor_3",
            "neighbor": ["corridor_13"],
            "large_objects": {},
            "small_objects": {}
        },
        "lauriannes_office": {
            "floor": "floor_3",
            "neighbor": ["corridor_14"],
            "large_objects": {},
            "small_objects": {}
        },
        "dimitys_office": {
            "floor": "floor_3",
            "neighbor": ["corridor_15"],
            "large_objects": {},
            "small_objects": {}
        },
        "meeting_room_1": {
            "floor": "floor_1",
            "neighbor": ["corridor_1", "corridor_2", "meeting_room_2"],
            "large_objects": {
                "table_5": {"type": "furniture"},
                "chair_3": {"type": "furniture"},
                "chair_4": {"type": "furniture"},
                "chair_5": {"type": "furniture"}
            },
            "small_objects": {}
        },
        "meeting_room_2": {
            "floor": "floor_1",
            "neighbor": ["meeting_room_1"],
            "large_objects": {
                "conference_table": {"type": "furniture"},
                "projector": {"type": "device"}
            },
            "small_objects": {}
        },
        "meeting_room_3": {
            "floor": "floor_2",
            "neighbor": ["corridor_12"],
            "large_objects": {},
            "small_objects": {}
        },
        "meeting_room_4": {
            "floor": "floor_2",
            "neighbor": ["corridor_9"],
            "large_objects": {},
            "small_objects": {}
        },
        "presentation_lounge": {
            "floor": "floor_3",
            "neighbor": ["corridor_18"],
            "large_objects": {},
            "small_objects": {}
        },
        "mobile_robotics_lab": {
            "floor": "floor_1",
            "neighbor": ["corridor_2", "manipulation_lab"],
            "large_objects": {
                "robot_platform": {"type": "equipment"},
                "workbench": {"type": "furniture"}
            },
            "small_objects": {
                "sensor_array": {
                    "type": "equipment",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "physical_property": {"fragile": True},
                    "relation": {"on": "workbench"}
                },
                "laptop": {
                    "type": "electronics",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "physical_property": {"fragile": True},
                    "relation": {"on": "workbench"}
                }
            }
        },
        "manipulation_lab": {
            "floor": "floor_1",
            "neighbor": ["mobile_robotics_lab", "agriculture_lab"],
            "large_objects": {
                "robotic_arm": {"type": "equipment"},
                "control_panel": {"type": "device"}
            },
            "small_objects": {}
        },
        "agriculture_lab": {
            "floor": "floor_1",
            "neighbor": ["manipulation_lab"],
            "large_objects": {
                "growth_chamber": {"type": "equipment"}
            },
            "small_objects": {
                "plant_samples": {
                    "type": "biological",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "physical_property": {"fragile": True, "living": True},
                    "relation": {"on": "growth_chamber"}
                }
            }
        },
        "printing_zone_1": {
            "floor": "floor_4",
            "neighbor": ["corridor_19"],
            "large_objects": {
                "printer_1": {"type": "device"}
            },
            "small_objects": {}
        },
        "printing_zone_2": {
            "floor": "floor_3",
            "neighbor": ["corridor_16"],
            "large_objects": {
                "printer_2": {"type": "device"}
            },
            "small_objects": {
                "document": {
                    "type": "paper",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"on": "printer_2"}
                }
            }
        },
        "supplies_station": {
            "floor": "floor_4",
            "neighbor": ["corridor_20"],
            "large_objects": {},
            "small_objects": {}
        },
        "kitchen": {
            "floor": "floor_4",
            "neighbor": ["corridor_20"],
            "large_objects": {
                "fridge": {"type": "appliance", "is_container": True},
                "coffee_machine": {"type": "appliance"},
                "recycling_bin": {"type": "container", "is_container": True},
                "kitchen_bench": {"type": "surface"}
            },
            "small_objects": {
                "salmon_bagel": {
                    "type": "food",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "cold",  
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"in": "fridge"}
                },
                "orange_peel": {
                    "type": "waste",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"in": "recycling_bin"}
                }
            }
        },
        "cafeteria": {
            "floor": "floor_4",
            "neighbor": ["corridor_20"],
            "large_objects": {
                "dining_table_1": {"type": "furniture"},
                "vending_machine": {"type": "device"}
            },
            "small_objects": {}
        },
        "lobby": {
            "floor": "floor_4",
            "neighbor": ["corridor_19"],
            "large_objects": {
                "shelf_2": {"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "parcel": {
                    "type": "package",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "free"
                    },
                    "relation": {"on": "shelf_2"}
                }
            }
        },
        "robot_lounge_1": {
            "floor": "floor_4",
            "neighbor": ["corridor_21"],
            "large_objects": {},
            "small_objects": {}
        },
        "robot_lounge_2": {
            "floor": "floor_4",
            "neighbor": ["corridor_22"],
            "large_objects": {},
            "small_objects": {}
        },
        "phd_bay_1": {
            "floor": "floor_4",
            "neighbor": ["corridor_23"],
            "large_objects": {},
            "small_objects": {}
        },
        "phd_bay_2": {
            "floor": "floor_5",
            "neighbor": ["corridor_24"],
            "large_objects": {},
            "small_objects": {}
        },
        "phd_bay_3": {
            "floor": "floor_5",
            "neighbor": ["corridor_25"],
            "large_objects": {},
            "small_objects": {}
        },
        "phd_bay_4": {
            "floor": "floor_5",
            "neighbor": ["corridor_26"],
            "large_objects": {},
            "small_objects": {}
        },
        "postdoc_bay_1": {
            "floor": "floor_4",
            "neighbor": ["corridor_23"],
            "large_objects": {},
            "small_objects": {}
        },
        "postdoc_bay_2": {
            "floor": "floor_5",
            "neighbor": ["corridor_24"],
            "large_objects": {},
            "small_objects": {}
        },
        "postdoc_bay_3": {
            "floor": "floor_5",
            "neighbor": ["corridor_25"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_1": {
            "floor": "floor_1",
            "neighbor": ["peters_office", "meeting_room_1", "corridor_2"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_2": {
            "floor": "floor_1",
            "neighbor": ["corridor_1", "corridor_3", "mobile_robotics_lab", "elevator_1f", "meeting_room_1"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_1f": {
            "floor": "floor_1",
            "neighbor": ["corridor_2", "elevator_2f", "elevator_3f", "elevator_4f", "elevator_5f", "elevator_cabin"],
            "large_objects": {
                "panel_1f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_1f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "corridor_3": {
            "floor": "floor_1",
            "neighbor": ["corridor_2", "corridor_4"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_4": {
            "floor": "floor_1",
            "neighbor": ["corridor_3", "tobis_office", "corridor_5"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_5": {
            "floor": "floor_1",
            "neighbor": ["corridor_4", "nikos_office", "corridor_6"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_6": {
            "floor": "floor_1",
            "neighbor": ["corridor_5", "michaels_office"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_7": {
            "floor": "floor_2",
            "neighbor": ["aarons_office", "corridor_8"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_8": {
            "floor": "floor_2",
            "neighbor": ["corridor_7", "jasons_office", "corridor_9"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_9": {
            "floor": "floor_2",
            "neighbor": ["corridor_8", "meeting_room_4","filipes_office", "corridor_10"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_10": {
            "floor": "floor_2",
            "neighbor": ["corridor_9", "luis_office", "corridor_11", "elevator_2f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_2f": {
            "floor": "floor_2",
            "neighbor": ["corridor_10", "elevator_1f", "elevator_3f", "elevator_4f", "elevator_5f", "elevator_cabin"],
            "large_objects": {
                "panel_2f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_2f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "corridor_11": {
            "floor": "floor_2",
            "neighbor": ["corridor_10", "wills_office", "corridor_12"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_12": {
            "floor": "floor_2",
            "neighbor": ["corridor_11", "meeting_room_3"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_13": {
            "floor": "floor_3",
            "neighbor": ["ajays_office", "chris_office", "corridor_14"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_14": {
            "floor": "floor_3",
            "neighbor": ["corridor_13", "lauriannes_office", "corridor_15"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_15": {
            "floor": "floor_3",
            "neighbor": ["corridor_14", "dimitys_office", "corridor_16"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_16": {
            "floor": "floor_3",
            "neighbor": ["corridor_15","printing_zone_2", "corridor_17", "elevator_3f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_3f": {
            "floor": "floor_3",
            "neighbor": ["corridor_16", "elevator_1f", "elevator_2f", "elevator_4f", "elevator_5f", "elevator_cabin"],
             "large_objects": {
                "panel_3f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_3f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "corridor_17": {
            "floor": "floor_3",
            "neighbor": ["corridor_16", "corridor_18"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_18": {
            "floor": "floor_3",
            "neighbor": ["corridor_17", "presentation_lounge"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_19": {
            "floor": "floor_4",
            "neighbor": ["printing_zone_1", "corridor_20", "lobby"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_20": {
            "floor": "floor_4",
            "neighbor": ["corridor_19", "supplies_station", "corridor_21", "cafeteria", "kitchen"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_21": {
            "floor": "floor_4",
            "neighbor": ["corridor_20", "robot_lounge_1", "corridor_22"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_22": {
            "floor": "floor_4",
            "neighbor": ["corridor_21", "robot_lounge_2", "corridor_23", "elevator_4f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_4f": {
            "floor": "floor_4",
            "neighbor": ["corridor_22", "elevator_1f", "elevator_2f", "elevator_3f", "elevator_5f", "elevator_cabin"],
            "large_objects": {
                "panel_4f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_4f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_4f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "corridor_23": {
            "floor": "floor_4",
            "neighbor": ["corridor_22", "phd_bay_1", "postdoc_bay_1"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_24": {
            "floor": "floor_5",
            "neighbor": ["phd_bay_2", "postdoc_bay_2", "corridor_25"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_25": {
            "floor": "floor_5",
            "neighbor": ["corridor_24", "phd_bay_3", "postdoc_bay_3", "corridor_26"],
            "large_objects": {},
            "small_objects": {}
        },
        "corridor_26": {
            "floor": "floor_5",
            "neighbor": ["corridor_25", "phd_bay_4", "elevator_5f"],
            "large_objects": {},
            "small_objects": {}
        },
        "elevator_5f": {
            "floor": "floor_5",
            "neighbor": ["corridor_26", "elevator_1f", "elevator_2f", "elevator_3f", "elevator_4f", "elevator_cabin"],
            "large_objects": {
                "panel_5f": {"type": "appliance", "is_container": False}
            },
            "small_objects": {
                "elevator_call_up": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_5f"},
                    "description": "Call elevator up button - press when outside elevator"
                },
                "elevator_call_down": {
                    "type": "control", 
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"on": "panel_5f"},
                    "description": "Call elevator down button - press when outside elevator"
                }
            }
        },
        "elevator_cabin": {
            "floor": "elevator",
            "neighbor": ["elevator_1f", "elevator_2f", "elevator_3f", "elevator_4f", "elevator_5f"],
            "large_objects": {},
            "small_objects": {
                "elevator_button_1": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_2": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_3": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_4": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                },
                "elevator_button_5": {
                    "type": "control",
                    "affordance": ["press"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "lit": False,
                        "pressed": False
                    },
                    "relation": {"inside": "elevator_cabin"}
                }
            }
        }
    },
    "agent": {
        "position": "mobile_robotics_lab",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot"
    }
}


PUDU = {
    "name": "pudu_hotel",
    "macro_zones": {
        "floor_1_public": {
            "rooms": ["lobby", "elevator_1f"]
        },
        "floor_14_pudu": {
            "rooms": ["hallway_14f", "meeting_room", "elevator_2f"]
        },
    },
    "rooms": {
        "lobby": {
            "floor": "floor_1_public",
            "neighbor": ["elevator_1f"],
            "large_objects": {
                "takeout_rack": {"type": "furniture", "is_container": True},
                "desk":{"type": "furniture", "is_container": True}
            },
            "small_objects": {
                "apple": {
                    "type": "access_item",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                    },
                    "relation": {"on": "desk"}
                },
                "banana": {
                    "type": "access_item",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                    },
                    "relation": {"on": "desk"}
                },
                "orange": {
                    "type": "access_item",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "room",
                        "wetness": "dry",
                        "availability": "available",
                    },
                    "relation": {"on": "desk"}
                },
                "takeout": {
                    "type": "access_item",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "warm",
                        "wetness": "dry",
                        "availability": "available",
                    },
                    "relation": {"on": "takeout_rack"}
                },
                "coke": {
                    "type": "access_item",
                    "affordance": ["pick", "place"],
                    "state": {
                        "temperature": "cold",
                        "availability": "available",
                    },
                    "relation": {"on": "desk"}
                },
            }
        },
        "elevator_1f": {
            "floor": "floor_1_public",
            "neighbor": ["lobby", "elevator_14f","elevator_cabin"],
                "large_objects": {
                    "panel_1f": {"type": "appliance", "is_container": False}
                },
                "small_objects": {
                    "elevator_call_up": {
                        "type": "control",
                        "affordance": ["press"],
                        "state": {
                            "temperature": "room",
                            "wetness": "dry",
                            "lit": False,
                            "pressed": False
                        },
                        "relation": {"on": "panel_14f"},
                        "description": "Call elevator up button - press when outside elevator"
                    },
                    "elevator_call_down": {
                        "type": "control", 
                        "affordance": ["press"],
                        "state": {
                            "temperature": "room",
                            "wetness": "dry",
                            "lit": False,
                            "pressed": False
                        },
                        "relation": {"on": "panel_14f"},
                        "description": "Call elevator down button - press when outside elevator"
                    }
                }
        },
        "hallway_14f":{
            "floor":"floor_14_pudu",
            "neighbor": ["elevator_14f", "meeting_room"],
                "large_objects": {
            },
            "small_objects": {

            }
        },
        "meeting_room":{
            "floor":"floor_14_pudu",
            "neighbor": ["hallway_14f"],
            "large_objects": {
                "meeting_desk":{"type": "furniture", "is_container": True}
            },
            "small_objects": {
            }
        },
        "elevator_14f": {
            "floor": "floor_14_pudu",
                "neighbor": ["hallway_14f", "elevator_1f", "elevator_cabin"],
                "large_objects": {
                    "panel_14f": {"type": "appliance", "is_container": False}
                },
                "small_objects": {
                    "elevator_call_up": {
                        "type": "control",
                        "affordance": ["press"],
                        "state": {
                            "temperature": "room",
                            "wetness": "dry",
                            "lit": False,
                            "pressed": False
                        },
                        "relation": {"on": "panel_14f"},
                        "description": "Call elevator up button - press when outside elevator"
                    },
                    "elevator_call_down": {
                        "type": "control", 
                        "affordance": ["press"],
                        "state": {
                            "temperature": "room",
                            "wetness": "dry",
                            "lit": False,
                            "pressed": False
                        },
                        "relation": {"on": "panel_14f"},
                        "description": "Call elevator down button - press when outside elevator"
                    }
                }
        },
        "elevator_cabin": {
            "floor": "elevator",
                "neighbor": ["elevator_1f", "elevator_14f"],
                "large_objects": {},
                "small_objects": {
                    "elevator_button_1": {
                        "type": "control",
                        "affordance": ["press"],
                        "state": {
                            "temperature": "room",
                            "wetness": "dry",
                            "lit": False,
                            "pressed": False
                        },
                        "relation": {"inside": "elevator_cabin"}
                    },
                    "elevator_button_14": {
                        "type": "control",
                        "affordance": ["press"],
                        "state": {
                            "temperature": "room",
                            "wetness": "dry",
                            "lit": False,
                            "pressed": False
                        },
                        "relation": {"inside": "elevator_cabin"}
                    }
                }
        }
    },
    "agent": {
        "position": "lobby",
        "state": "hand-free",
        "battery": 100,
        "type": "default_robot"
    }
}
