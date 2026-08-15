import pytest
from pathlib import Path

from evaluation.action_parser import ParseError
from evaluation.delta_baseline import DeltaPDDLPlanner, parse_delta_subgoals


def record(task_type, parameters):
    return {"task_info": {"type": task_type, "parameters": parameters}}


def test_delivery_decomposition_accepts_any_valid_target_assignment():
    task = record(
        "delivery",
        {"objects": ["cup", "plate"], "target_rooms": ["kitchen", "lobby"]},
    )

    parsed = parse_delta_subgoals(
        {
            "subgoals": [
                {"kind": "deliver", "object": "plate", "room": "lobby"},
                {"kind": "deliver", "object": "cup", "room": "lobby"},
            ]
        },
        task,
    )

    assert [item["object"] for item in parsed] == ["plate", "cup"]


def test_guidance_decomposition_rejects_reordered_waypoints():
    task = record(
        "guidance",
        {"intermediate_points": ["room_a", "room_b"], "end_room": "room_c"},
    )

    with pytest.raises(ParseError, match="waypoint order"):
        parse_delta_subgoals(
            {
                "subgoals": [
                    {"kind": "visit", "room": "room_b"},
                    {"kind": "visit", "room": "room_a"},
                    {"kind": "visit", "room": "room_c"},
                ]
            },
            task,
        )


def test_fast_downward_solves_sequential_delivery_subgoal():
    fast_downward = (
        Path(__file__).resolve().parents[1]
        / "third_party/DELTA/downward/fast-downward.py"
    )
    if not fast_downward.exists():
        pytest.skip("Fast Downward is not installed")
    scene = {
        "agent": {"position": "room_a"},
        "rooms": {
            "room_a": {
                "neighbor": ["room_b"],
                "small_objects": {"cup": {}},
                "large_objects": {},
            },
            "room_b": {
                "neighbor": ["room_a"],
                "small_objects": {},
                "large_objects": {},
            },
        },
    }
    planner = DeltaPDDLPlanner(fast_downward)

    actions, _, failures = planner.plan(
        scene,
        [{"kind": "deliver", "object": "cup", "room": "room_b"}],
    )

    assert failures == 0
    assert actions[-1] == ("place", "cup", "room_b")
